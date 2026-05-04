"""Launch the interactive MuJoCo viewer on a composed scene+robot.

Since scene XMLs no longer embed their robot include or keyframe, opening
``demo.xml`` directly in the MuJoCo simulator shows just the empty scene.
This script reads a Hydra task config, composes the scene with the robot(s)
declared under ``env.robot_paths``, applies ``env.initial_joint_positions``
as the home pose, and hands the model to ``mujoco.viewer.launch``.

When the config carries a ``gaussian_render`` section, the script switches
to a passive viewer and opens a second OpenCV window that re-renders the
scene with Gaussian Splatting from the same free-camera pose as the MuJoCo
viewer (synced live as you orbit / pan / zoom).

In GS mode, press ``R`` in either window or click the Reload button in the
GS window to re-read YAML, XML, and PLY files without restarting the Python
process.

Usage::

    python examples/view_scene.py --config-name pick_and_place
    python examples/view_scene.py --config-name open_door_airbot_play_gs
    python examples/view_scene.py --config-name open_door_p7_ik
    python examples/view_scene.py --debug --config-name open_door_p7_ik
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from typing import Callable

import hydra
import mujoco
import mujoco.viewer
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

from auto_atom.runner.common import get_config_dir
from auto_atom.utils.pose import euler_to_quaternion
from auto_atom.utils.scene_loader import load_scene

# Joint qpos widths by mjtJoint enum value: free=7, ball=4, slide=1, hinge=1.
_QPOS_WIDTH = {0: 7, 1: 4, 2: 1, 3: 1}
_DEBUG = False


def _strip_debug_arg(argv: list[str]) -> bool:
    """Consume this script's --debug flag before Hydra parses argv."""
    debug = False
    stripped: list[str] = []
    hydra_separator_seen = False
    for arg in argv:
        if arg == "--":
            hydra_separator_seen = True
            stripped.append(arg)
        elif not hydra_separator_seen and arg == "--debug":
            debug = True
        else:
            stripped.append(arg)
    argv[:] = stripped
    return debug


def _print_debug_exception(context: str) -> None:
    print(f"[debug] {context} failed; full traceback:", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)


def _to_container(value):
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _orientation_to_wxyz(orientation: list[float]) -> np.ndarray:
    """Accept YAML orientation (4 floats xyzw or 3 floats Euler rpy) and
    return a wxyz quaternion suitable for MuJoCo body_quat / mocap_quat."""
    if len(orientation) == 4:
        x, y, z, w = (float(v) for v in orientation)
    elif len(orientation) == 3:
        x, y, z, w = euler_to_quaternion(tuple(float(v) for v in orientation))
    else:
        raise ValueError(
            f"orientation must be 3 floats (Euler) or 4 floats (xyzw quat), "
            f"got {len(orientation)}"
        )
    return np.array([w, x, y, z], dtype=np.float64)


def _find_freejoint_for_body(model: mujoco.MjModel, body_name: str) -> int:
    """Return the joint id of a freejoint directly attached to ``body_name``,
    or -1 if the body is static (no freejoint child)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return -1
    jnt_start = int(model.body_jntadr[bid])
    jnt_count = int(model.body_jntnum[bid])
    for j in range(jnt_start, jnt_start + jnt_count):
        if int(model.jnt_type[j]) == 0:  # mjJNT_FREE
            return j
    return -1


def _set_body_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    position: list[float] | None,
    orientation: list[float] | None,
) -> bool:
    """Apply a position+orientation override to a body. Uses the body's
    freejoint qpos when available, otherwise mutates ``model.body_pos`` /
    ``model.body_quat`` (treating it as world-relative — the parent must be
    world for this to be an exact override). Returns True on success."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        print(f"[warn] body '{body_name}' not found; skipping initial_pose")
        return False
    free_jid = _find_freejoint_for_body(model, body_name)
    if free_jid >= 0:
        addr = int(model.jnt_qposadr[free_jid])
        if position is not None:
            data.qpos[addr : addr + 3] = [float(v) for v in position[:3]]
        if orientation is not None:
            data.qpos[addr + 3 : addr + 7] = _orientation_to_wxyz(orientation)
        return True
    # Static body: write into the model's body_pos / body_quat slot.
    if position is not None:
        model.body_pos[bid] = [float(v) for v in position[:3]]
    if orientation is not None:
        model.body_quat[bid] = _orientation_to_wxyz(orientation)
    return True


def _apply_home_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    initial_joint_positions: dict,
) -> None:
    """Mirror ``MujocoBasis.reset()``: write scalars + multi-DOF entries and
    let parallel-linkage equality constraints settle under zero gravity."""
    multi_dof: list[tuple[int, np.ndarray]] = []
    pin_addrs: list[int] = []
    for joint_name, value in initial_joint_positions.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            print(f"[warn] joint '{joint_name}' not found; skipping")
            continue
        addr = int(model.jnt_qposadr[jid])
        width = _QPOS_WIDTH[int(model.jnt_type[jid])]
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value, dtype=float)
            if arr.size != width:
                raise ValueError(
                    f"initial_joint_positions['{joint_name}']: got {arr.size} "
                    f"values, joint expects {width}"
                )
            multi_dof.append((addr, arr))
        else:
            if width != 1:
                raise ValueError(
                    f"initial_joint_positions['{joint_name}']: scalar given for "
                    f"a {width}-DOF joint; pass a list"
                )
            data.qpos[addr] = float(value)
            pin_addrs.append(addr)

    if pin_addrs and model.neq > 0:
        # Settle equality-constrained passive joints (e.g. gripper coupler /
        # follower) by stepping in zero gravity while pinning all freejoint
        # bodies and the configured scalar joints.
        free_addrs: list[int] = []
        for j in range(model.njnt):
            if int(model.jnt_type[j]) == 0:  # free
                a = int(model.jnt_qposadr[j])
                free_addrs.extend(range(a, a + 7))
        free_snap = data.qpos[free_addrs].copy() if free_addrs else None

        saved_g = model.opt.gravity.copy()
        model.opt.gravity[:] = 0
        target = data.qpos[pin_addrs].copy()
        for _ in range(500):
            mujoco.mj_step(model, data)
            data.qpos[pin_addrs] = target
            if free_snap is not None:
                data.qpos[free_addrs] = free_snap
        data.qvel[:] = 0.0
        model.opt.gravity[:] = saved_g

    for addr, arr in multi_dof:
        data.qpos[addr : addr + arr.size] = arr

    mujoco.mj_forward(model, data)
    # Sync mocap bodies (if any) onto the freejoint they're welded to so the
    # arm doesn't snap back when the viewer takes its first step.
    for eq in range(model.neq):
        if int(model.eq_type[eq]) != 1:  # mjEQ_WELD
            continue
        b1, b2 = int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq])
        m1, m2 = int(model.body_mocapid[b1]), int(model.body_mocapid[b2])
        if m1 >= 0 and m2 < 0:
            mocap_id, phys_id = m1, b2
        elif m2 >= 0 and m1 < 0:
            mocap_id, phys_id = m2, b1
        else:
            continue
        data.mocap_pos[mocap_id] = data.xpos[phys_id].copy()
        data.mocap_quat[mocap_id] = data.xquat[phys_id].copy()


def _extract_overrides(cfg: DictConfig) -> dict:
    """Pull the four override surfaces out of a freshly-composed Hydra cfg
    so reload can re-read them without restarting the process."""
    env_cfg = cfg.env
    out: dict = {
        "model_path": str(env_cfg.model_path),
        "robot_paths": [str(p) for p in (env_cfg.get("robot_paths") or [])],
        "ijp": _to_container(env_cfg.get("initial_joint_positions")) or {},
        "initial_pose": _to_container(cfg.get("task", {}).get("initial_pose")) or {},
        "op_bases": [],
    }

    operators_cfg = env_cfg.get("operators") or {}
    task_operators_cfg = cfg.get("task_operators") or {}
    items = task_operators_cfg.items() if task_operators_cfg else []
    for op_name, op_node in items:
        bp_cfg = _to_container((op_node.get("initial_state") or {}).get("base_pose"))
        if not bp_cfg:
            continue
        root_body = (operators_cfg.get(op_name) or {}).get("root_body")
        if not root_body:
            print(
                f"[warn] task_operators.{op_name}.initial_state.base_pose set "
                f"but env.operators.{op_name}.root_body is empty; skipping"
            )
            continue
        out["op_bases"].append((root_body, bp_cfg))
    return out


def _build(overrides: dict) -> tuple[mujoco.MjModel, mujoco.MjData]:
    m = load_scene(overrides["model_path"], overrides["robot_paths"])
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    # 1) task_operators.*.initial_state.base_pose — relocates the arm root
    #    body so the arm sits at the right world pose.
    for root_body, bp in overrides["op_bases"]:
        _set_body_pose(m, d, root_body, bp.get("position"), bp.get("orientation"))

    # 2) task.initial_pose — per-object overrides (freejoint qpos or static body_pos).
    for body_name, override in overrides["initial_pose"].items():
        override = override or {}
        _set_body_pose(
            m,
            d,
            body_name,
            override.get("position"),
            override.get("orientation"),
        )

    # 3) env.initial_joint_positions — joint-level home pose.
    _apply_home_pose(m, d, overrides["ijp"])
    return m, d


def _compose_config_from_disk(
    config_dir: str,
    config_name: str,
    cli_overrides: list[str] | None = None,
) -> DictConfig:
    """Re-compose the Hydra config from disk for viewer reloads."""
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=cli_overrides or [])


def _load_reloaded_scene(
    config_dir: str,
    config_name: str,
    cli_overrides: list[str] | None = None,
) -> tuple[DictConfig, dict, mujoco.MjModel, mujoco.MjData]:
    cfg_now = _compose_config_from_disk(config_dir, config_name, cli_overrides)
    overrides = _extract_overrides(cfg_now)
    m, d = _build(overrides)
    return cfg_now, overrides, m, d


def _print_model_summary(model: mujoco.MjModel, overrides: dict) -> None:
    print(
        f"[info] model  : nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nbody={model.nbody} ngeom={model.ngeom}  "
        f"(robots={overrides['robot_paths']}, ijp={len(overrides['ijp'])}, "
        f"body_pose={len(overrides['initial_pose'])}, "
        f"op_base={len(overrides['op_bases'])})"
    )


def _print_gs_summary(gs_cfg) -> None:
    print(
        f"[info] gs     : {len(gs_cfg.body_gaussians)} body gaussian(s), "
        f"background_ply={'list/glob' if gs_cfg.is_multi_background() else gs_cfg.background_ply!r}"
    )


def _maybe_gs_config(cfg: DictConfig):
    """Return a ``GaussianRenderConfig`` if the task config requests GS rendering.

    Detection is content-based (looks for ``env.gaussian_render`` with at least
    one body gaussian or a background ply), so it works whether the task uses
    the GS env target directly or composes GS into a non-GS env.
    """
    env_cfg = cfg.get("env", {})
    gs_node = env_cfg.get("gaussian_render", None)
    if gs_node is None:
        return None
    gs_dict = OmegaConf.to_container(gs_node, resolve=True) or {}
    if not (gs_dict.get("body_gaussians") or gs_dict.get("background_ply")):
        return None
    from auto_atom.basis.mjc.gs_mujoco_env import GaussianRenderConfig

    return GaussianRenderConfig.model_validate(gs_dict)


def _build_gs_renderer(gs_cfg, model: mujoco.MjModel):
    """Build a ``GSRendererMuJoCo`` covering body PLYs + a single background.

    Multi-background configs (list / glob) only get their first entry here —
    the viewer is for previewing geometry alignment, not for sweeping bgs.
    """
    from gaussian_renderer import GSRendererMuJoCo

    combined = dict(gs_cfg.resolved_body_gaussians())
    if gs_cfg.is_multi_background():
        bgs = gs_cfg.resolved_background_plys()
        if bgs:
            combined["background"] = bgs[0]
    else:
        bg = gs_cfg.resolved_background_ply()
        if bg:
            combined["background"] = bg
    return GSRendererMuJoCo(combined, model)


def _run_gs_synced_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gs_cfg,
    reload_callback: Callable[
        [], tuple[DictConfig, dict, mujoco.MjModel, mujoco.MjData]
    ]
    | None = None,
    width: int = 640,
    height: int = 480,
) -> None:
    """Passive MuJoCo viewer + cv2 window showing the GS render of the same
    free-camera pose, refreshed every step."""
    import cv2
    import torch

    win = "GS view (synced with MuJoCo viewer)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, width, height)

    def reload_button_rect() -> tuple[int, int, int, int]:
        button_width = 142
        button_height = 36
        x1 = width - 18
        y0 = 18
        return max(18, x1 - button_width), y0, x1, y0 + button_height

    def draw_reload_button(frame: np.ndarray) -> None:
        if reload_callback is None:
            return
        x0, y0, x1, y1 = reload_button_rect()
        cv2.rectangle(frame, (x0, y0), (x1, y1), (42, 112, 170), -1)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 210, 255), 1)
        cv2.putText(
            frame,
            "Reload (R)",
            (x0 + 16, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    def show_status(title: str, lines: list[str]) -> None:
        frame = np.full((height, width, 3), 24, dtype=np.uint8)
        accent = (80, 180, 255)
        text = (235, 235, 235)
        muted = (170, 170, 170)
        cv2.putText(
            frame,
            title,
            (28, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            accent,
            2,
            cv2.LINE_AA,
        )
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (28, 104 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                text if idx == 0 else muted,
                1,
                cv2.LINE_AA,
            )
        draw_reload_button(frame)
        cv2.imshow(win, frame)
        cv2.waitKey(1)

    reload_event = threading.Event()
    startup_start = time.perf_counter()
    first_visible_frame = False
    render_attempts = 0
    last_black_notice = 0.0

    def reset_warmup_state() -> None:
        nonlocal startup_start, first_visible_frame, render_attempts, last_black_notice
        startup_start = time.perf_counter()
        first_visible_frame = False
        render_attempts = 0
        last_black_notice = 0.0

    def on_mouse(event, x: int, y: int, _flags, _param) -> None:
        if reload_callback is None or event != cv2.EVENT_LBUTTONUP:
            return
        x0, y0, x1, y1 = reload_button_rect()
        if x0 <= x <= x1 and y0 <= y <= y1:
            print("[reload] requested from GS window button", flush=True)
            reload_event.set()

    cv2.setMouseCallback(win, on_mouse)

    def build_renderer_with_status(current_gs_cfg, current_model: mujoco.MjModel):
        show_status(
            "Loading Gaussian renderer...",
            [
                "Reading Gaussian PLY files and preparing GPU resources.",
                "The first render can take a few seconds; this is expected.",
                "Terminal logs will report when GS is ready.",
            ],
        )
        print(
            "[info] GS renderer: loading Gaussian PLYs; first render may take "
            "a few seconds...",
            flush=True,
        )
        load_start = time.perf_counter()
        renderer = _build_gs_renderer(current_gs_cfg, current_model)
        print(
            f"[info] GS renderer: loaded in {time.perf_counter() - load_start:.1f}s; "
            "warming up first frame...",
            flush=True,
        )
        show_status(
            "Warming up GS render...",
            [
                "Waiting for the first visible GS frame.",
                "If the renderer returns a black warmup frame, it is hidden here.",
                "Press ESC to exit; click Reload or press R after edits.",
            ],
        )
        reset_warmup_state()
        return renderer

    def key_callback(key: int) -> None:
        if reload_callback is not None and key in (ord("R"), ord("r")):
            reload_event.set()

    gs_renderer = build_renderer_with_status(gs_cfg, model)
    print(
        "[info] GS sync: orbit/pan/zoom in the MuJoCo viewer to drive the GS"
        f" view (size {width}x{height}). ESC in the GS window to close it;"
        " close the MuJoCo viewer to exit."
        + (
            " Press R in either window or click Reload in the GS window "
            "to re-read YAML/XML/PLY. The MuJoCo viewer window will reopen "
            "after reload."
            if reload_callback is not None
            else ""
        )
    )

    def clone_camera(camera) -> mujoco.MjvCamera:
        cloned = mujoco.MjvCamera()
        cloned.type = camera.type
        cloned.fixedcamid = camera.fixedcamid
        cloned.trackbodyid = camera.trackbodyid
        cloned.lookat[:] = camera.lookat
        cloned.distance = camera.distance
        cloned.azimuth = camera.azimuth
        cloned.elevation = camera.elevation
        return cloned

    def restore_camera(target, source: mujoco.MjvCamera) -> None:
        target.type = source.type
        target.fixedcamid = source.fixedcamid
        target.trackbodyid = source.trackbodyid
        target.lookat[:] = source.lookat
        target.distance = source.distance
        target.azimuth = source.azimuth
        target.elevation = source.elevation

    def wait_for_viewer_exit(v) -> None:
        deadline = time.perf_counter() + 2.0
        sim_ref = getattr(v, "_sim", None)
        while time.perf_counter() < deadline:
            if sim_ref is None or sim_ref() is None:
                return
            time.sleep(0.01)

    def load_reloaded_scene_with_status():
        if reload_callback is None:
            raise RuntimeError("GS reload is not enabled")
        show_status(
            "Reloading GS scene...",
            [
                "Re-reading YAML, XML, and Gaussian PLY files.",
                "The MuJoCo viewer is restarted to avoid passive reload races.",
                "Keep this GS window open; errors will appear here.",
            ],
        )
        cfg_now, overrides_now, new_model, new_data = reload_callback()
        new_gs_cfg = _maybe_gs_config(cfg_now)
        if new_gs_cfg is None:
            raise RuntimeError("reloaded config no longer defines env.gaussian_render")
        new_renderer = build_renderer_with_status(new_gs_cfg, new_model)
        return cfg_now, overrides_now, new_model, new_data, new_gs_cfg, new_renderer

    camera_state: mujoco.MjvCamera | None = None
    while True:
        restart_requested = False
        with mujoco.viewer.launch_passive(
            model,
            data,
            key_callback=key_callback if reload_callback is not None else None,
        ) as v:
            if camera_state is not None:
                with v.lock():
                    restore_camera(v.cam, camera_state)
                v.sync(state_only=True)

            while v.is_running():
                step_start = time.time()
                if reload_event.is_set():
                    reload_event.clear()
                    if reload_callback is not None:
                        with v.lock():
                            camera_state = clone_camera(v.cam)
                        restart_requested = True
                        print(
                            "[reload] closing MuJoCo viewer before rebuilding scene...",
                            flush=True,
                        )
                        show_status(
                            "Reloading GS scene...",
                            [
                                "Closing the MuJoCo viewer before scene rebuild.",
                                "This avoids passive-viewer scene/frustum races.",
                                "The viewer will reopen automatically.",
                            ],
                        )
                        v.close()
                        break
                with v.lock():
                    mujoco.mj_step(model, data)
                    camera_for_render = clone_camera(v.cam)
                v.sync()
                try:
                    render_attempts += 1
                    gs_renderer.update_gaussians(data)
                    results = gs_renderer.render(
                        model, data, [-1], width, height, free_camera=camera_for_render
                    )
                    rgb_t, _depth = results[-1]
                    rgb = rgb_t.clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu().numpy()
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    if not first_visible_frame and int(rgb.max(initial=0)) <= 1:
                        now = time.perf_counter()
                        if now - last_black_notice >= 2.0:
                            last_black_notice = now
                            print(
                                "[info] GS warmup: renderer returned a black frame; "
                                "keeping the loading screen visible...",
                                flush=True,
                            )
                        show_status(
                            "Warming up GS render...",
                            [
                                f"Black warmup frame hidden (attempt {render_attempts}).",
                                "This can happen while CUDA kernels or PLY data settle.",
                                "If it persists, check GS paths/camera alignment or press R.",
                            ],
                        )
                    else:
                        if not first_visible_frame:
                            first_visible_frame = True
                            print(
                                "[info] GS ready: first visible frame after "
                                f"{time.perf_counter() - startup_start:.1f}s "
                                f"({render_attempts} render attempt(s)).",
                                flush=True,
                            )
                        draw_reload_button(bgr)
                        cv2.imshow(win, bgr)
                except Exception as e:
                    if _DEBUG:
                        _print_debug_exception("GS render")
                    else:
                        print(f"[warn] GS render error: {e}")
                    show_status(
                        "GS render error",
                        [
                            str(e)[:78],
                            "The viewer will retry on the next frame.",
                            "Use --debug for a full traceback.",
                        ],
                    )
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    v.close()
                    break  # ESC exits both windows
                if reload_callback is not None and key in (ord("R"), ord("r")):
                    reload_event.set()
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    v.close()
                    break  # GS window closed via X
                elapsed = time.time() - step_start
                sleep_for = float(model.opt.timestep) - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        wait_for_viewer_exit(v)
        if not restart_requested:
            break
        try:
            print("[reload] re-reading YAML/XML/PLY...", flush=True)
            (
                _cfg_now,
                overrides_now,
                new_model,
                new_data,
                new_gs_cfg,
                new_renderer,
            ) = load_reloaded_scene_with_status()
            model, data, gs_cfg, gs_renderer = (
                new_model,
                new_data,
                new_gs_cfg,
                new_renderer,
            )
            _print_model_summary(model, overrides_now)
            _print_gs_summary(gs_cfg)
            print("[reload] done; reopening MuJoCo viewer", flush=True)
        except Exception as e:
            if _DEBUG:
                _print_debug_exception("GS reload")
            else:
                print(f"[warn] GS reload error: {e}")
            show_status(
                "GS reload failed",
                [
                    str(e)[:78],
                    "Old scene will reopen; fix the error, then reload again.",
                    "Use --debug for a full traceback.",
                ],
            )
            print("[reload] keeping previous scene", flush=True)
        reload_event.clear()
    cv2.destroyAllWindows()


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    # Recover the absolute config directory + config name from HydraConfig so
    # the loader can re-compose the same task on every reload click. (Hydra
    # changes cwd inside @hydra.main, so we can't rely on get_config_dir().)
    hc = HydraConfig.get()
    config_name = hc.job.config_name
    cli_overrides = list(hc.overrides.task)
    config_dir = next(
        (s.path for s in hc.runtime.config_sources if s.provider == "main"),
        None,
    )
    if config_dir is None:
        raise RuntimeError("Could not resolve absolute config dir from HydraConfig.")

    overrides = _extract_overrides(cfg)
    print(f"[info] scene  : {overrides['model_path']}")
    print(f"[info] robots : {overrides['robot_paths'] or '(none)'}")
    print(
        f"[info] home   : {len(overrides['ijp'])} joint override(s), "
        f"{len(overrides['initial_pose'])} body pose(s), "
        f"{len(overrides['op_bases'])} operator base(s)"
    )

    if _DEBUG:
        print("[debug] preflight build before launching viewer...", flush=True)
        try:
            m, d = _build(overrides)
            print(
                f"[debug] preflight ok: nq={m.nq} nv={m.nv} nu={m.nu} "
                f"nbody={m.nbody} ngeom={m.ngeom}",
                flush=True,
            )
            del m, d
        except Exception:
            _print_debug_exception("preflight build")
            raise

    gs_cfg = _maybe_gs_config(cfg)
    if gs_cfg is not None:
        # GS path: passive viewer whose free-camera state drives a synced GS
        # render in a cv2 window. Passive viewer has no loader hook, so reload
        # is wired through R / viewer reload requests in the render loop below.
        m, d = _build(overrides)
        _print_model_summary(m, overrides)
        _print_gs_summary(gs_cfg)

        def gs_reload_loader() -> tuple[
            DictConfig, dict, mujoco.MjModel, mujoco.MjData
        ]:
            return _load_reloaded_scene(config_dir, config_name, cli_overrides)

        _run_gs_synced_viewer(m, d, gs_cfg, reload_callback=gs_reload_loader)
        return

    def loader() -> tuple[mujoco.MjModel, mujoco.MjData]:
        try:
            # Re-compose the Hydra config from disk so the reload button picks up
            # YAML edits (initial_pose, base_pose, robot_paths, joint_positions, ...),
            # then rebuild the model. ``load_scene`` already re-reads the scene/robot
            # XML files from disk so XML edits also flow through.
            _cfg_now, ov, m, d = _load_reloaded_scene(
                config_dir, config_name, cli_overrides
            )
            _print_model_summary(m, ov)
            return m, d
        except Exception:
            if _DEBUG:
                _print_debug_exception("viewer loader")
            raise

    print(
        "[info] launching viewer (close the window to exit; reload button"
        " re-reads YAML+XML and re-applies overrides)..."
    )
    mujoco.viewer.launch(loader=loader)


if __name__ == "__main__":
    _DEBUG = _strip_debug_arg(sys.argv)
    main()
