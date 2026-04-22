"""Interactively tune GS ``body_mirrors`` parameters with a live preview.

Side-by-side native MuJoCo + GS render through a shared orbit camera, so you
can see where the physics body is vs. where the mirrored/transformed gaussians
actually end up.

Every keypress on a mirror parameter:
  1. mutates the in-memory ``BodyMirrorSpec`` fields on the env config,
  2. calls ``gs_cfg.resolved_body_gaussians()`` — cached under
     ``.cache/gs_body_mirrors/``,
  3. rebuilds the env's foreground Gaussian renderer, and
  4. re-renders both panels.

Mouse:
  Left drag      orbit (azimuth / elevation)
  Right drag     zoom (drag down = zoom out)
  Middle drag    pan
  Wheel          zoom

Keyboard (matches ``gs_frame_tuner.py`` conventions):
  i/k  j/l  o/u    position x / y / z  +/-        (post-reflection translate)
  y/h  t/g  n/m    orientation roll / pitch / yaw +/-   (post-reflection rot)
  Shift + i/k / j/l / o/u    mirror center x / y / z +/-  (pivot in GS coords)
  1/2              xyz step  /2  *2
  3/4              rot step  /2  *2
  Tab              cycle selected body
  a                toggle link-all (apply edits to every body in body_mirrors)
  f                cycle mirror axis: GS-local X / Y / Z (clears body_quat)
  Shift+M          toggle mirror on / off for the selected body
  0                clear post-reflection orientation to identity
  p                print YAML snippet
  r                reset to YAML values
  s                save snapshot PNG to /tmp/tuner_<timestamp>.png
  b                print help
  q / Esc          quit

Usage:
    python3 examples/tune_gs_body_mirror.py --config-name open_door_airbot_play_back_gs
    python3 examples/tune_gs_body_mirror.py \
        --config-name open_door_airbot_play_back_gs \
        --camera env0_cam --bg-index 3 --width 720 --height 540

Extra Hydra overrides can be appended after ``--``:
    ... -- env.gaussian_render.minibatch=256
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from PySide6 import QtCore, QtGui, QtWidgets
from scipy.spatial.transform import Rotation

from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import ComponentRegistry

# ---------------------------------------------------------------------------
# Hydra / config helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="open_door_airbot_play_back_gs")
    parser.add_argument(
        "--camera",
        default=None,
        help="Named MuJoCo camera to initialise the orbit from. "
        "Defaults to the first static camera in the config.",
    )
    parser.add_argument(
        "--bg-index",
        type=int,
        default=0,
        help="When background_ply is a glob/list, force this index for a "
        "stable preview.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--pos-step",
        type=float,
        default=0.01,
        help="Initial step (m) for position/center nudges.",
    )
    parser.add_argument(
        "--rot-step-deg",
        type=float,
        default=1.0,
        help="Initial step (deg) for roll/pitch/yaw nudges.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional Hydra overrides after '--'.",
    )
    return parser.parse_args()


def _compose_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = get_config_dir()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _resolve_camera_name(cfg: DictConfig, requested: str | None) -> str:
    cameras = list(cfg.env.cameras)
    if not cameras:
        raise ValueError("Config has no cameras.")
    available = [str(cam.name) for cam in cameras]
    if requested:
        if requested not in available:
            raise ValueError(f"Unknown camera '{requested}'. Available: {available}")
        return requested
    for cam in cameras:
        if bool(cam.get("is_static", False)):
            return str(cam.name)
    return available[0]


def _prepare_cfg_for_preview(cfg: DictConfig, bg_index: int) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.env.batch_size = 1
    if "viewer" not in cfg.env or cfg.env.viewer is None:
        cfg.env.viewer = {"disable": True}
    else:
        cfg.env.viewer.disable = True

    # Lock background to one PLY for deterministic preview.
    bg_ply = cfg.env.gaussian_render.get("background_ply")
    if isinstance(bg_ply, str) and any(ch in bg_ply for ch in "*?["):
        from glob import glob

        matches = sorted(glob(bg_ply))
        if not matches:
            raise FileNotFoundError(f"No backgrounds match glob: {bg_ply}")
        idx = max(0, min(bg_index, len(matches) - 1))
        cfg.env.gaussian_render.background_ply = matches[idx]
    elif isinstance(bg_ply, (list, tuple)) and len(bg_ply) > 1:
        idx = max(0, min(bg_index, len(bg_ply) - 1))
        cfg.env.gaussian_render.background_ply = bg_ply[idx]

    # Color-only, no depth / mask — we only need one rendered frame per panel.
    for cam in cfg.env.cameras:
        cam.enable_color = True
        cam.enable_depth = False
        cam.enable_heat_map = False
        cam.enable_mask = False
    return cfg


# ---------------------------------------------------------------------------
# Mirror tuner state
# ---------------------------------------------------------------------------


class MirrorState:
    """Euler + position + optional center in GS-local coords for one body."""

    AXIS_PRESETS = ("X", "Y", "Z")

    def __init__(self, body_name: str, spec) -> None:
        self.body_name = body_name
        self.spec = spec

        self.initial_euler = self._euler_from_spec(spec)
        self.euler = self.initial_euler.copy()

        pos = spec.position if spec.position is not None else [0.0, 0.0, 0.0]
        self.initial_position = np.asarray(pos, dtype=np.float64)
        self.position = self.initial_position.copy()

        self.center_was_set = spec.center is not None
        ctr = spec.center if spec.center is not None else None
        self.initial_center = None if ctr is None else np.asarray(ctr, dtype=np.float64)
        self.center = (
            None if self.initial_center is None else self.initial_center.copy()
        )
        self.share_center_with = spec.share_center_with

        # Save initial axis-related fields so `r` can restore them.
        self._initial_axis = None if spec.axis is None else list(spec.axis)
        self._initial_body_quat = (
            None if spec.body_quat is None else list(spec.body_quat)
        )
        self._initial_body_axis = (
            None if spec.body_axis is None else list(spec.body_axis)
        )
        # Tracks whether mirror is toggled off. When off, we store the original
        # spec fields in self._stashed_* and replace them with a no-op
        # (axis=[1,0,0] with center = current center; orientation/position held).
        self._mirror_off = False
        self._stashed_axis = None
        self._stashed_body_quat = None
        self._stashed_body_axis = None

    @staticmethod
    def _euler_from_spec(spec) -> np.ndarray:
        if spec.orientation is None:
            return np.zeros(3, dtype=np.float64)
        arr = np.asarray(spec.orientation, dtype=np.float64).ravel()
        if arr.shape == (3,):
            return arr.copy()
        if arr.shape == (4,):
            norm = float(np.linalg.norm(arr))
            if norm < 1e-12:
                return np.zeros(3, dtype=np.float64)
            return Rotation.from_quat(arr / norm).as_euler("xyz")
        raise ValueError(f"orientation must be length 3 or 4, got shape {arr.shape}")

    def reset(self) -> None:
        self.euler = self.initial_euler.copy()
        self.position = self.initial_position.copy()
        self.center = (
            None if self.initial_center is None else self.initial_center.copy()
        )
        self.spec.axis = (
            list(self._initial_axis) if self._initial_axis is not None else None
        )
        self.spec.body_quat = (
            list(self._initial_body_quat)
            if self._initial_body_quat is not None
            else None
        )
        self.spec.body_axis = (
            list(self._initial_body_axis)
            if self._initial_body_axis is not None
            else [1.0, 0.0, 0.0]
        )
        self._mirror_off = False

    def effective_axis_gs_local(self) -> np.ndarray:
        """Return the current mirror-plane normal in GS-local coords."""
        return self.spec.resolved_axis()

    def set_axis_preset(self, which: str) -> None:
        """Force mirror normal to world-aligned GS-local X / Y / Z axis.
        Clears ``body_quat`` so ``resolved_axis`` takes ``axis`` literally."""
        v = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}[which]
        self.spec.axis = v
        self.spec.body_quat = None
        # body_axis is only read when body_quat is set; leave it alone.
        self._mirror_off = False

    def mirror_off(self) -> bool:
        return self._mirror_off

    def set_mirror_off(self, off: bool) -> None:
        self._mirror_off = bool(off)

    def seed_center_if_missing(self, src_ply: str) -> None:
        if self.center is not None:
            return
        if self.spec.center is not None:
            self.center = np.asarray(self.spec.center, dtype=np.float64)
            return
        from auto_atom.basis.mjc.gs_mujoco_env import load_ply

        self.center = load_ply(str(src_ply)).xyz.mean(axis=0).astype(np.float64)

    def write_back(self) -> None:
        """Push current tuner state onto the pydantic spec so the next call
        to ``resolved_body_gaussians()`` hashes the updated params."""
        quat_xyzw = Rotation.from_euler("xyz", self.euler).as_quat()
        n = float(np.linalg.norm(quat_xyzw))
        if n < 1e-12:
            quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            quat_xyzw = quat_xyzw / n
        self.spec.orientation = [float(v) for v in quat_xyzw]
        self.spec.position = [float(v) for v in self.position]
        if self.center is not None:
            self.spec.center = [float(v) for v in self.center]

    def yaml_snippet(self) -> str:
        lines = [f"  {self.body_name}:"]
        if self.spec.body_quat is not None:
            q = self.spec.body_quat
            lines.append(f"    body_quat: [{q[0]}, {q[1]}, {q[2]}, {q[3]}]")
            if self.spec.body_axis is not None:
                a = self.spec.body_axis
                lines.append(f"    body_axis: [{a[0]}, {a[1]}, {a[2]}]")
        if self.spec.axis is not None:
            a = self.spec.axis
            lines.append(f"    axis: [{a[0]}, {a[1]}, {a[2]}]")
        qx, qy, qz, qw = Rotation.from_euler("xyz", self.euler).as_quat()
        lines.append(f"    orientation: [{qx:.10f}, {qy:.10f}, {qz:.10f}, {qw:.10f}]")
        if not np.allclose(self.position, 0.0):
            lines.append(
                f"    position: [{self.position[0]:.6f}, "
                f"{self.position[1]:.6f}, {self.position[2]:.6f}]"
            )
        if self.center is not None and self.center_was_set:
            lines.append(
                f"    center: [{self.center[0]:.10f}, "
                f"{self.center[1]:.10f}, {self.center[2]:.10f}]"
            )
        if self.share_center_with is not None:
            lines.append(f"    share_center_with: {self.share_center_with}")
        return "\n".join(lines)

    def state_line(self) -> str:
        roll, pitch, yaw = (math.degrees(v) for v in self.euler)
        parts = [
            f"{self.body_name:<16}",
            f"pos=[{self.position[0]:+.4f}, {self.position[1]:+.4f}, {self.position[2]:+.4f}]",
            f"rpy=[{roll:+7.3f}, {pitch:+7.3f}, {yaw:+7.3f}] deg",
        ]
        if self.center is not None:
            parts.append(
                f"ctr=[{self.center[0]:+.4f}, {self.center[1]:+.4f}, "
                f"{self.center[2]:+.4f}]"
            )
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Orbit camera
# ---------------------------------------------------------------------------


class OrbitCamera:
    """Orbit camera in MuJoCo's native convention.

    Camera position relative to ``lookat``:
        ``rel = distance * [cos(az)*cos(el), sin(az)*cos(el), sin(el)]``
    Camera-to-world rotation follows MuJoCo's convention: columns are
    ``[right, up, back]`` where ``back = -forward`` (camera looks along
    its local -Z). ``azimuth`` / ``elevation`` can be pushed straight to
    ``mujoco.MjvCamera`` for the native Renderer with no sign flips.
    """

    def __init__(
        self, lookat: np.ndarray, distance: float, az_deg: float, el_deg: float
    ):
        self.lookat = np.asarray(lookat, dtype=np.float64).copy()
        self.distance = float(distance)
        self.az = float(az_deg)
        self.el = float(el_deg)

    @classmethod
    def from_named_camera(
        cls,
        model,
        data,
        cam_name: str,
        focus_xyz: np.ndarray | None = None,
    ) -> "OrbitCamera":
        """Initialise orbit from a named camera.

        When ``focus_xyz`` is given (e.g. the world position of the body the
        user actually wants to inspect), the orbit's ``lookat`` is pinned to
        it and only the camera's azimuth / elevation / distance are taken
        from the named camera. This keeps the orbit centered on the target
        body even when ``initial_pose`` has moved the body far from where
        the static camera's forward ray happens to hit.
        """
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            lookat = (
                np.array([0.4, 0.4, 0.5])
                if focus_xyz is None
                else np.asarray(focus_xyz, dtype=np.float64)
            )
            return cls(lookat, 2.0, 90.0, -20.0)
        cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        if focus_xyz is None:
            cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
            fwd = -cam_mat[:, 2]
            distance = 1.5
            lookat = cam_pos + fwd * distance
        else:
            lookat = np.asarray(focus_xyz, dtype=np.float64)
            distance = float(np.linalg.norm(cam_pos - lookat))
            if distance < 1e-3:
                distance = 1.5
        # MuJoCo MjvCamera convention: azimuth / elevation describe the
        # *viewing* direction (from camera toward lookat), NOT the camera
        # offset. So the ``rel`` vector for angle extraction is
        # ``lookat - cam_pos``.
        rel = lookat - cam_pos
        horiz = math.hypot(rel[0], rel[1])
        el = math.degrees(math.atan2(rel[2], horiz))
        az = math.degrees(math.atan2(rel[1], rel[0]))
        return cls(lookat, distance, az, el)

    def cam_pos(self) -> np.ndarray:
        # MJ convention: camera sits OPPOSITE to the viewing direction,
        # so cam_pos = lookat - dist * viewing_dir.
        az = math.radians(self.az)
        el = math.radians(self.el)
        view_dir = np.array(
            [
                math.cos(az) * math.cos(el),
                math.sin(az) * math.cos(el),
                math.sin(el),
            ]
        )
        return self.lookat - self.distance * view_dir

    def cam_xmat(self) -> np.ndarray:
        back = self.cam_pos() - self.lookat
        n = np.linalg.norm(back) + 1e-12
        back /= n
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(world_up, back)
        rn = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0]) if rn < 1e-6 else right / rn
        up = np.cross(back, right)
        return np.column_stack([right, up, back])

    def orbit(self, daz: float, del_: float) -> None:
        self.az -= daz
        self.el = float(np.clip(self.el - del_, -89.0, 89.0))

    def zoom(self, factor: float) -> None:
        self.distance = max(0.05, self.distance * factor)

    def pan(self, dx: float, dy: float) -> None:
        scale = self.distance * 0.0015
        xmat = self.cam_xmat()
        right = xmat[:, 0]
        up = xmat[:, 1]
        self.lookat -= right * dx * scale - up * dy * scale


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _rebuild_foreground(env) -> None:
    from auto_atom.basis.mjc.gs_mujoco_env import (
        BatchSplatConfig,
        MjxBatchSplatRenderer,
    )

    gs_cfg = env.config.gaussian_render
    env._gs_body_gaussians = gs_cfg.resolved_body_gaussians()
    model = env.envs[0].model if hasattr(env, "envs") else env.model
    env._fg_gs_renderer = MjxBatchSplatRenderer(
        BatchSplatConfig(
            body_gaussians=dict(env._gs_body_gaussians),
            background_ply=None,
            minibatch=gs_cfg.minibatch,
        ),
        model,
    )


def _bg_renderer(env):
    """Return the single active background MjxBatchSplatRenderer, or None."""
    if hasattr(env, "_bg_gs_renderers") and env._bg_gs_renderers:
        return env._bg_gs_renderers[0]
    return getattr(env, "_bg_gs_renderer", None)


class DualRenderer:
    """Side-by-side native MuJoCo + GS renderer using a shared orbit camera."""

    def __init__(self, env, width: int, height: int, fovy_deg: float = 45.0):
        self.env = env
        self.width = width
        self.height = height
        self.fovy_deg = fovy_deg

        mj_env = env.envs[0] if hasattr(env, "envs") else env
        self.mj_model = mj_env.model
        self.mj_data = mj_env.data

        self.mj_renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
        self._mj_cam = mujoco.MjvCamera()
        self._mj_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self._mj_opt = mujoco.MjvOption()

    def close(self) -> None:
        try:
            self.mj_renderer.close()
        except Exception:
            pass

    # -- camera arrays -----------------------------------------------------

    def _cam_arrays(self, orbit: OrbitCamera):
        cam_pos = orbit.cam_pos().astype(np.float32).reshape(1, 1, 3)
        cam_xmat = orbit.cam_xmat().astype(np.float32).reshape(1, 1, 9)
        fovy = np.asarray([[self.fovy_deg]], dtype=np.float32)
        return cam_pos, cam_xmat, fovy

    # -- native ------------------------------------------------------------

    def render_native(self, orbit: OrbitCamera) -> np.ndarray:
        self._mj_cam.lookat[:] = orbit.lookat
        self._mj_cam.distance = orbit.distance
        self._mj_cam.azimuth = orbit.az
        self._mj_cam.elevation = orbit.el
        self.mj_renderer.update_scene(self.mj_data, camera=self._mj_cam)
        return np.asarray(self.mj_renderer.render(), dtype=np.uint8)

    # -- GS ----------------------------------------------------------------

    def render_gs(self, orbit: OrbitCamera) -> np.ndarray:
        cam_pos, cam_xmat, fovy = self._cam_arrays(orbit)
        body_pos = np.asarray(self.mj_data.xpos, dtype=np.float32).reshape(
            1, self.mj_model.nbody, 3
        )
        body_quat = np.asarray(self.mj_data.xquat, dtype=np.float32).reshape(
            1, self.mj_model.nbody, 4
        )

        fg_gsb = self.env._fg_gs_renderer.batch_update_gaussians(body_pos, body_quat)

        bg_imgs = None
        bg = _bg_renderer(self.env)
        if bg is not None:
            bg_gsb = bg.batch_update_gaussians(body_pos, body_quat)
            bg_rgb, _ = bg.batch_env_render(
                bg_gsb, cam_pos, cam_xmat, self.height, self.width, fovy
            )
            bg_imgs = bg_rgb

        fg_rgb, _ = self.env._fg_gs_renderer.batch_env_render(
            fg_gsb,
            cam_pos,
            cam_xmat,
            self.height,
            self.width,
            fovy,
            bg_imgs=bg_imgs,
        )
        img = torch.clamp(fg_rgb[0, 0], 0.0, 1.0).mul(255).to(torch.uint8)
        return img.cpu().numpy()


# ---------------------------------------------------------------------------
# Image composition
# ---------------------------------------------------------------------------


def _label_panel(img_rgb: np.ndarray, title: str) -> np.ndarray:
    out = img_rgb.copy()
    cv2.putText(
        out, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        out,
        title,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def _hud_strip(
    width: int,
    selected: MirrorState,
    sel_idx: int,
    n_states: int,
    link_all: bool,
    step_pos: float,
    step_rot_deg: float,
    status_lines: list[str],
) -> np.ndarray:
    h = 26 + 22 * (len(status_lines) + 2)
    strip = np.full((h, width, 3), 32, dtype=np.uint8)
    link_tag = "LINK-ALL" if link_all else "SINGLE"
    header = (
        f"body [{sel_idx + 1}/{n_states}] {selected.body_name}   "
        f"[{link_tag}]   step_pos={step_pos:.4f}m   step_rot={step_rot_deg:.3f}deg"
    )
    for i, line in enumerate([header, selected.state_line(), *status_lines]):
        cv2.putText(
            strip,
            line,
            (14, 24 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return strip


# ---------------------------------------------------------------------------
# Qt window
# ---------------------------------------------------------------------------


def _ndarray_rgb_to_qpixmap(img_rgb: np.ndarray) -> QtGui.QPixmap:
    img_rgb = np.ascontiguousarray(img_rgb)
    h, w, _ = img_rgb.shape
    qimg = QtGui.QImage(img_rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def _status_help_lines() -> list[str]:
    return [
        "pos: i/k=x  j/l=y  o/u=z   |   rpy: y/h=roll  t/g=pitch  n/m=yaw",
        "Shift+(i/k j/l o/u)=center xyz   |   1/2=xyz step   3/4=rot step",
        "Tab=next body   a=link-all   f=flip-axis   Shift+M=mirror on/off   0=ori=identity",
        "p=print YAML   r=reset   s=snapshot   b=help   q=quit",
        "Mouse: L-drag orbit   R-drag zoom   Mid-drag pan   wheel=zoom",
    ]


class TunerWindow(QtWidgets.QWidget):
    def __init__(
        self,
        env,
        states: list[MirrorState],
        camera_name: str,
        args: argparse.Namespace,
        backend=None,
    ) -> None:
        super().__init__()
        self.env = env
        self.states = states
        self.args = args
        self.backend = backend

        self.dual = DualRenderer(env, args.width, args.height)
        focus_xyz = self._resolve_focus_xyz(states)
        if focus_xyz is not None:
            print(
                f"[init] orbit focus = {states[0].body_name} world xpos "
                f"[{focus_xyz[0]:.3f}, {focus_xyz[1]:.3f}, {focus_xyz[2]:.3f}]"
            )
        self.orbit = OrbitCamera.from_named_camera(
            self.dual.mj_model, self.dual.mj_data, camera_name, focus_xyz=focus_xyz
        )

        self.sel_idx = 0
        self.link_all = len(states) > 1
        self.step_pos = float(args.pos_step)
        self.step_rot = math.radians(float(args.rot_step_deg))

        self.setWindowTitle(
            f"GS body_mirror tuner | {args.config_name} | orbit from '{camera_name}'"
        )
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(args.width * 2, args.height)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setFocus()

        self._mouse_button = None
        self._mouse_last = QtCore.QPoint()
        self._last_status_lines = _status_help_lines()

        self._render_and_show()

    # -- init helpers ------------------------------------------------------

    def _resolve_focus_xyz(self, states: list[MirrorState]) -> np.ndarray | None:
        """World xpos of the first mirror body, for centering the orbit.

        Using the body the user is actually tuning as the orbit pivot
        keeps the camera pointed at it regardless of how much
        ``initial_pose`` has shifted the assembly from its keyframe.
        """
        for st in states:
            bid = mujoco.mj_name2id(
                self.dual.mj_model, mujoco.mjtObj.mjOBJ_BODY, st.body_name
            )
            if bid >= 0:
                return np.asarray(self.dual.mj_data.xpos[bid], dtype=np.float64)
        return None

    # -- orchestration -----------------------------------------------------

    def _selected(self) -> MirrorState:
        return self.states[self.sel_idx]

    def _targeted(self) -> list[MirrorState]:
        return self.states if self.link_all else [self.states[self.sel_idx]]

    def _apply_and_rebuild(self) -> None:
        for st in self.states:
            st.write_back()
        # Honor mirror-off: pop disabled entries from the live body_mirrors
        # dict; restore them afterwards so subsequent edits keep working.
        gs_cfg = self.env.config.gaussian_render
        stashed: dict = {}
        for st in self.states:
            if st.mirror_off() and st.body_name in gs_cfg.body_mirrors:
                stashed[st.body_name] = gs_cfg.body_mirrors.pop(st.body_name)
        try:
            _rebuild_foreground(self.env)
        finally:
            for name, spec in stashed.items():
                gs_cfg.body_mirrors[name] = spec

    def _render_and_show(self) -> None:
        try:
            native = self.dual.render_native(self.orbit)
            gs = self.dual.render_gs(self.orbit)
        except Exception as exc:  # noqa: BLE001
            print(f"[render error] {exc}")
            return
        native = _label_panel(native, "MuJoCo native")
        gs = _label_panel(gs, "Gaussian Splatting")
        combined = cv2.hconcat([native, gs])
        hud = _hud_strip(
            combined.shape[1],
            self._selected(),
            self.sel_idx,
            len(self.states),
            self.link_all,
            self.step_pos,
            math.degrees(self.step_rot),
            self._last_status_lines,
        )
        canvas = cv2.vconcat([combined, hud])
        self._last_canvas_rgb = canvas
        pix = _ndarray_rgb_to_qpixmap(canvas)
        self.label.setPixmap(pix)
        self.resize(pix.size())

    def _save_snapshot(self) -> None:
        import time

        if getattr(self, "_last_canvas_rgb", None) is None:
            return
        out = Path("/tmp") / f"tuner_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(out), cv2.cvtColor(self._last_canvas_rgb, cv2.COLOR_RGB2BGR))
        print(f"[snapshot] saved {out}")

    def _notify(self, msg: str) -> None:
        print(msg)

    # -- mouse -------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._mouse_button = event.button()
        self._mouse_last = event.position().toPoint()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._mouse_button = None

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._mouse_button is None:
            return
        cur = event.position().toPoint()
        dx = cur.x() - self._mouse_last.x()
        dy = cur.y() - self._mouse_last.y()
        self._mouse_last = cur
        if self._mouse_button == QtCore.Qt.LeftButton:
            self.orbit.orbit(dx * 0.3, dy * 0.3)
        elif self._mouse_button == QtCore.Qt.RightButton:
            self.orbit.zoom(1.0 + dy * 0.005)
        elif self._mouse_button == QtCore.Qt.MiddleButton:
            self.orbit.pan(dx, dy)
        self._render_and_show()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self.orbit.zoom(0.9**steps)
        self._render_and_show()

    # -- keys --------------------------------------------------------------

    def _print_state(self, tag: str = "[state]") -> None:
        print(f"{tag} " + self._selected().state_line())

    def _print_help(self) -> None:
        print("\n===== gs_body_mirror tuner =====")
        for line in _status_help_lines():
            print("  " + line)
        print("================================\n")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()
        shift = bool(mods & QtCore.Qt.ShiftModifier)

        if key in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Q):
            self.close()
            return
        if key == QtCore.Qt.Key_B:
            self._print_help()
            return
        if key == QtCore.Qt.Key_P:
            print("\nbody_mirrors:")
            for st in self.states:
                print(st.yaml_snippet())
            print()
            return
        if key == QtCore.Qt.Key_R:
            for st in self.states:
                st.reset()
            self._apply_and_rebuild()
            self._print_state("[reset]")
            self._render_and_show()
            return
        if key == QtCore.Qt.Key_S and not shift:
            self._save_snapshot()
            return

        if key == QtCore.Qt.Key_F:
            # Cycle selected body(s) through mirror axis presets X / Y / Z.
            if not hasattr(self, "_axis_idx"):
                self._axis_idx = 0
            self._axis_idx = (self._axis_idx + 1) % len(MirrorState.AXIS_PRESETS)
            which = MirrorState.AXIS_PRESETS[self._axis_idx]
            for st in self._targeted():
                st.set_axis_preset(which)
            self._apply_and_rebuild()
            targets = ",".join(s.body_name for s in self._targeted())
            print(f"[flip] {targets}: mirror axis -> GS-local {which}")
            self._render_and_show()
            return

        if key == QtCore.Qt.Key_M and shift:
            # Shift+M: toggle mirror on/off for the targeted body(s).
            # (Lowercase m is yaw- under rot_keys below.)
            for st in self._targeted():
                st.set_mirror_off(not st.mirror_off())
            self._apply_and_rebuild()
            status = ", ".join(
                f"{s.body_name}={'OFF' if s.mirror_off() else 'ON'}"
                for s in self._targeted()
            )
            print(f"[mirror] {status}")
            self._render_and_show()
            return

        if key == QtCore.Qt.Key_0:
            # Clear post-reflection orientation to identity (no post-rotation).
            for st in self._targeted():
                st.euler = np.zeros(3, dtype=np.float64)
            self._apply_and_rebuild()
            self._print_state("[ori=id]")
            self._render_and_show()
            return
        if key == QtCore.Qt.Key_Tab:
            self.sel_idx = (self.sel_idx + 1) % len(self.states)
            self._print_state("[select]")
            self._render_and_show()
            return
        if key == QtCore.Qt.Key_A and not shift:
            self.link_all = not self.link_all
            self._notify(f"[link] link-all = {self.link_all}")
            self._render_and_show()
            return

        if key == QtCore.Qt.Key_1:
            self.step_pos = max(1e-5, self.step_pos * 0.5)
            self._notify(f"[step] xyz step = {self.step_pos:.6f} m")
            self._render_and_show()
            return
        if key == QtCore.Qt.Key_2:
            self.step_pos = min(1.0, self.step_pos * 2.0)
            self._notify(f"[step] xyz step = {self.step_pos:.6f} m")
            self._render_and_show()
            return
        if key == QtCore.Qt.Key_3:
            self.step_rot = max(math.radians(0.001), self.step_rot * 0.5)
            self._notify(f"[step] rot step = {math.degrees(self.step_rot):.4f} deg")
            self._render_and_show()
            return
        if key == QtCore.Qt.Key_4:
            self.step_rot = min(math.radians(45.0), self.step_rot * 2.0)
            self._notify(f"[step] rot step = {math.degrees(self.step_rot):.4f} deg")
            self._render_and_show()
            return

        # pos (no shift) / center (shift) — same key set
        pos_keys = {
            QtCore.Qt.Key_I: (0, +1),
            QtCore.Qt.Key_K: (0, -1),
            QtCore.Qt.Key_J: (1, +1),
            QtCore.Qt.Key_L: (1, -1),
            QtCore.Qt.Key_O: (2, +1),
            QtCore.Qt.Key_U: (2, -1),
        }
        rot_keys = {
            QtCore.Qt.Key_Y: (0, +1),
            QtCore.Qt.Key_H: (0, -1),
            QtCore.Qt.Key_T: (1, +1),
            QtCore.Qt.Key_G: (1, -1),
            QtCore.Qt.Key_N: (2, +1),
            QtCore.Qt.Key_M: (2, -1),
        }

        if key in pos_keys:
            axis, sign = pos_keys[key]
            field = "center" if shift else "position"
            gs_cfg = self.env.config.gaussian_render
            for st in self._targeted():
                if field == "center":
                    src_ply = gs_cfg.body_gaussians[st.body_name]
                    st.seed_center_if_missing(src_ply)
                    st.center[axis] += sign * self.step_pos
                else:
                    st.position[axis] += sign * self.step_pos
            self._apply_and_rebuild()
            self._print_state(f"[{field}]")
            self._render_and_show()
            return

        if key in rot_keys:
            axis, sign = rot_keys[key]
            for st in self._targeted():
                st.euler[axis] += sign * self.step_rot
            self._apply_and_rebuild()
            self._print_state("[rotate]")
            self._render_and_show()
            return

    # -- teardown ----------------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        print("\nFinal body_mirrors:")
        for st in self.states:
            print(st.yaml_snippet())
        try:
            self.dual.close()
        except Exception:
            pass
        if self.backend is not None:
            try:
                self.backend.teardown()
            except Exception as exc:  # noqa: BLE001
                print(f"backend.teardown() raised: {exc}")
        else:
            try:
                self.env.close()
            except Exception as exc:  # noqa: BLE001
                print(f"env.close() raised: {exc}")
        event.accept()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    overrides = [ov for ov in args.overrides if ov != "--"]
    cfg = _compose_cfg(args.config_name, overrides)
    camera_name = _resolve_camera_name(cfg, args.camera)
    cfg = _prepare_cfg_for_preview(cfg, bg_index=args.bg_index)

    task_file = prepare_task_file(cfg)
    env = ComponentRegistry.get_env(task_file.task.env_name)

    gs_cfg = env.config.gaussian_render
    if not gs_cfg.body_mirrors:
        raise ValueError(
            f"Config '{args.config_name}' has no env.gaussian_render.body_mirrors "
            "entries — nothing to tune."
        )

    states = [MirrorState(name, spec) for name, spec in gs_cfg.body_mirrors.items()]

    # IMPORTANT: ``env.reset()`` alone only restores the XML keyframe; the
    # task's ``initial_pose`` overrides (e.g. the 180° door rotation in the
    # back-side mixin) are applied by the *backend*, not the env. Construct
    # the full backend so ``setup()`` runs ``_apply_initial_poses()``.
    backend = task_file.backend(task_file.task, task_file.task_operators)
    backend.setup(task_file.task)

    print(f"Orbit camera initialised from '{camera_name}'")
    print(f"Bodies with mirrors: {[s.body_name for s in states]}")
    print("\n===== gs_body_mirror tuner =====")
    for line in _status_help_lines():
        print("  " + line)
    print("================================\n")
    for st in states:
        ax = st.effective_axis_gs_local()
        print(
            f"[init] {st.state_line()} | mirror_axis_gs={ax[0]:+.4f} {ax[1]:+.4f} {ax[2]:+.4f}"
        )

    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TunerWindow(env, states, camera_name, args, backend=backend)
    window.show()
    qt_app.exec()


if __name__ == "__main__":
    main()
