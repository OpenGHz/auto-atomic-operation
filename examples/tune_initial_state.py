"""Interactively tune operator initial state (base pose, EEF pose, gripper).

Loads a task config via Hydra (same pattern as ``aao_demo``), opens the
MuJoCo viewer, and provides a tkinter panel for editing poses.  No task
execution loop is run — this is purely for tuning and visualising the
initial configuration.

Usage::

    python examples/tune_initial_state.py
    python examples/tune_initial_state.py --config-name cup_on_coaster
"""

from __future__ import annotations

import mujoco
import numpy as np
import re
import sys
import tkinter as tk
from tkinter import ttk
from typing import List, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from auto_atom.backend.mjc.mujoco_backend import (
    MujocoOperatorHandler,
    MujocoTaskBackend,
)
from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv
from auto_atom.runner.common import get_config_dir
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner
from auto_atom.utils.pose import (
    PoseState,
    euler_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_to_rpy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_floats(text: str) -> List[float]:
    """Parse a string of comma or space-separated floats."""
    parts = re.split(r"[,\s]+", text.strip())
    return [float(p) for p in parts if p]


def _fmt(values, precision: int = 6) -> str:
    """Format a sequence of floats as a comma-separated string."""
    return ", ".join(f"{v:.{precision}f}" for v in values)


# ---------------------------------------------------------------------------
# Virtual base marker (coordinate frame axes in the MuJoCo viewer)
# ---------------------------------------------------------------------------

_AXIS_COLORS = [
    (1.0, 0.2, 0.2, 0.8),  # X — red
    (0.2, 1.0, 0.2, 0.8),  # Y — green
    (0.2, 0.2, 1.0, 0.8),  # Z — blue
]
_AXIS_LEN = 0.08


def _draw_base_marker(viewer, base_pose: PoseState) -> None:
    """Draw a small RGB coordinate frame at *base_pose* using the viewer's
    user scene (``viewer.user_scn``).  Previous user geoms are cleared first.
    """
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0  # clear previous markers
        pos = np.asarray(base_pose.position, dtype=np.float64)
        R = quaternion_to_rotation_matrix(base_pose.orientation)
        for axis_idx in range(3):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            tip = pos + R[:, axis_idx] * _AXIS_LEN
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_ARROW,
                0.004,
                pos,
                tip,
            )
            g.rgba = np.array(_AXIS_COLORS[axis_idx], dtype=np.float32)
            scn.ngeom += 1


# ---------------------------------------------------------------------------
# Tkinter panel for one operator
# ---------------------------------------------------------------------------


class OperatorPanel:
    """Editable panel for a single operator's base_pose, EEF pose, and gripper."""

    def __init__(
        self,
        parent: tk.Widget,
        handler: MujocoOperatorHandler,
        env: UnifiedMujocoEnv,
        viewer,
    ):
        self.handler = handler
        self.env = env
        self.viewer = viewer
        self._is_mocap = not handler._joint_mode

        frame = ttk.LabelFrame(parent, text=f"Operator: {handler.name}")
        frame.pack(fill="x", padx=6, pady=4)

        # ── Base Pose ──
        self._build_pose_group(
            frame, "Base Pose", self._read_base_pose, self._apply_base
        )

        # ── EEF Pose ──
        self._build_pose_group(frame, "EEF Pose", self._read_eef_pose, self._apply_eef)

        # ── EEF ctrl ──
        ef = ttk.Frame(frame)
        ef.pack(fill="x", padx=4, pady=2)
        ttk.Label(ef, text="EEF ctrl:").pack(side="left")
        self.eef_ctrl_var = tk.StringVar(
            value=str(float(handler._home_ctrl[handler.eef_ctrl_index]))
        )
        ttk.Entry(ef, textvariable=self.eef_ctrl_var, width=12).pack(
            side="left", padx=4
        )
        ttk.Button(ef, text="Apply", command=self._apply_eef_ctrl).pack(side="left")

        # ── Print YAML ──
        ttk.Button(frame, text="Print YAML", command=self._print_yaml).pack(pady=(6, 4))

        # Initial marker draw.
        if self._is_mocap and self.viewer is not None:
            _draw_base_marker(self.viewer, handler._base_pose)

    # -- internal builders --

    def _build_pose_group(
        self,
        parent: tk.Widget,
        title: str,
        read_fn,
        apply_fn,
    ):
        """Build a position + orientation editing group inside *parent*."""
        grp = ttk.LabelFrame(parent, text=title)
        grp.pack(fill="x", padx=4, pady=2)

        # Position row.
        pf = ttk.Frame(grp)
        pf.pack(fill="x", padx=2, pady=1)
        ttk.Label(pf, text="Position:").pack(side="left")
        pos_var = tk.StringVar()
        ttk.Entry(pf, textvariable=pos_var, width=36).pack(side="left", padx=4)

        # Mode radio.
        mode_var = tk.StringVar(value="quat")
        mf = ttk.Frame(grp)
        mf.pack(fill="x", padx=2, pady=1)
        ttk.Radiobutton(
            mf,
            text="Quaternion",
            variable=mode_var,
            value="quat",
            command=lambda: self._on_mode_change(mode_var, ori_var, ori_label, read_fn),
        ).pack(side="left")
        ttk.Radiobutton(
            mf,
            text="Euler (ypr)",
            variable=mode_var,
            value="euler",
            command=lambda: self._on_mode_change(mode_var, ori_var, ori_label, read_fn),
        ).pack(side="left")

        # Orientation row.
        of = ttk.Frame(grp)
        of.pack(fill="x", padx=2, pady=1)
        ori_label = ttk.Label(of, text="Quat (xyzw):")
        ori_label.pack(side="left")
        ori_var = tk.StringVar()
        ttk.Entry(of, textvariable=ori_var, width=36).pack(side="left", padx=4)

        # Apply button.
        ttk.Button(grp, text="Apply", command=apply_fn).pack(pady=2)

        # Store references keyed by title.
        attr = title.lower().replace(" ", "_")
        setattr(self, f"_{attr}_pos_var", pos_var)
        setattr(self, f"_{attr}_ori_var", ori_var)
        setattr(self, f"_{attr}_mode_var", mode_var)
        setattr(self, f"_{attr}_ori_label", ori_label)

        # Populate with current values.
        pose = read_fn()
        pos_var.set(_fmt(pose.position))
        ori_var.set(_fmt(pose.orientation))

    def _on_mode_change(self, mode_var, ori_var, ori_label, read_fn):
        """Switch between quaternion and euler display."""
        pose = read_fn()
        if mode_var.get() == "euler":
            ori_label.config(text="Euler (ypr):")
            rpy = quaternion_to_rpy(pose.orientation)
            # Display as yaw, pitch, roll.
            ori_var.set(_fmt((rpy[2], rpy[1], rpy[0])))
        else:
            ori_label.config(text="Quat (xyzw):")
            ori_var.set(_fmt(pose.orientation))

    # -- read current state --

    def _read_base_pose(self) -> PoseState:
        return self.handler._base_pose

    def _read_eef_pose(self) -> PoseState:
        return self.handler.get_end_effector_pose()

    # -- parse helpers --

    def _parse_pose(self, attr_prefix: str) -> Optional[PoseState]:
        """Parse position + orientation from the named group's entry variables."""
        try:
            pos = _parse_floats(getattr(self, f"_{attr_prefix}_pos_var").get())
            ori_raw = _parse_floats(getattr(self, f"_{attr_prefix}_ori_var").get())
            mode = getattr(self, f"_{attr_prefix}_mode_var").get()
        except (ValueError, AttributeError) as exc:
            print(f"Parse error: {exc}")
            return None
        if len(pos) != 3:
            print(f"Position needs 3 values, got {len(pos)}")
            return None
        if mode == "euler":
            if len(ori_raw) != 3:
                print(f"Euler needs 3 values (yaw, pitch, roll), got {len(ori_raw)}")
                return None
            yaw, pitch, roll = ori_raw
            quat = euler_to_quaternion((roll, pitch, yaw))
        else:
            if len(ori_raw) != 4:
                print(f"Quaternion needs 4 values (x,y,z,w), got {len(ori_raw)}")
                return None
            quat = tuple(ori_raw)
        return PoseState(position=tuple(pos), orientation=quat)

    # -- apply callbacks --

    def _apply_base(self):
        pose = self._parse_pose("base_pose")
        if pose is None:
            return
        # base_pose is a *virtual* arm base — it does NOT move the EEF body.
        self.handler._base_pose = pose
        if self.viewer is not None:
            _draw_base_marker(self.viewer, pose)
        self.env.refresh_viewer()
        print(
            f"[base_pose] position={list(pose.position)}, orientation={list(pose.orientation)}"
        )

    def _apply_eef(self):
        pose = self._parse_pose("eef_pose")
        if pose is None:
            return
        self.handler.set_home_end_effector_pose(pose)
        self.handler.home()
        self.env.refresh_viewer()
        # Re-read actual EEF pose after physics settle.
        actual = self.handler.get_end_effector_pose()
        print(f"[eef] target={list(pose.position)}, actual={list(actual.position)}")

    def _apply_eef_ctrl(self):
        try:
            val = float(self.eef_ctrl_var.get().strip())
        except ValueError:
            print("EEF ctrl must be a single float")
            return
        self.handler._home_ctrl[self.handler.eef_ctrl_index] = val
        self.handler.home()
        # Step physics directly (no viewer sync per step) so coupled gripper
        # joints (spring, follower, pad) settle without freezing the UI.
        for _ in range(200):
            mujoco.mj_step(self.env.model, self.env.data)
        self.env.refresh_viewer()
        print(f"[eef_ctrl] {val}")

    def _print_yaml(self):
        bp = self.handler._base_pose
        ep = self.handler.get_end_effector_pose()
        eef_val = float(self.handler._home_ctrl[self.handler.eef_ctrl_index])

        def _flist(vals):
            return "[" + ", ".join(f"{v:.6f}" for v in vals) + "]"

        print()
        print("initial_state:")
        print("  base_pose:")
        print(f"    position: {_flist(bp.position)}")
        print(f"    orientation: {_flist(bp.orientation)}")
        print("  arm:")
        print(f"    position: {_flist(ep.position)}")
        print(f"    orientation: {_flist(ep.orientation)}")
        print(f"  eef: {eef_val}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    ComponentRegistry.clear()
    if "env" in cfg and cfg.env is not None:
        instantiate(cfg.env)

    task_file = TaskFileConfig.model_validate(raw)
    runner = TaskRunner().from_config(task_file)
    backend = runner._context.backend
    if not isinstance(backend, MujocoTaskBackend):
        raise TypeError("Only MujocoTaskBackend is supported.")

    env = backend.env

    # Reach initial state.
    backend.reset()
    env.refresh_viewer()

    # Resolve viewer handle.
    viewer = env._viewer  # may be None if viewer is disabled

    # ── tkinter UI ──
    root = tk.Tk()
    root.title("Tune Initial State")
    root.geometry("420x520")

    panels: list[OperatorPanel] = []
    for name, handler in backend.operator_handlers.items():
        if isinstance(handler, MujocoOperatorHandler):
            panels.append(OperatorPanel(root, handler, env, viewer))

    # Periodic viewer sync.
    def tick():
        if env._viewer_running():
            env.refresh_viewer()
        root.after(50, tick)

    root.after(50, tick)

    try:
        root.mainloop()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
