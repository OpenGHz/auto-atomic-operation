"""Interactive MuJoCo viewers over initialized backend environments."""

from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from .mujoco_backend import MujocoTaskBackend


def _viewer_env(backend: MujocoTaskBackend):
    env = backend.get_env()
    return env.envs[env.config.viewer_env_index]


def _print_debug_exception(context: str) -> None:
    print(f"[debug] {context} failed; full traceback:", flush=True)
    traceback.print_exc()


def print_model_summary(backend: MujocoTaskBackend) -> None:
    """Print the physical model dimensions for one initialized backend."""

    model = _viewer_env(backend).model
    print(
        f"[info] model  : nq={model.nq} nv={model.nv} nu={model.nu} "
        f"nbody={model.nbody} ngeom={model.ngeom}",
        flush=True,
    )


def gaussian_config(backend: MujocoTaskBackend):
    """Return the backend-owned Gaussian config, when present."""

    config = getattr(backend.get_env().config, "gaussian_render", None)
    if config is None:
        return None
    if not (config.body_gaussians or config.background_ply):
        return None
    return config


def run_native_viewer(
    backend: MujocoTaskBackend,
    reload_callback: Callable[[], MujocoTaskBackend],
) -> MujocoTaskBackend:
    """Launch the native viewer and rebuild the backend on reload."""

    active = backend
    first_load = True

    def loader() -> tuple[mujoco.MjModel, mujoco.MjData]:
        nonlocal active, first_load
        if first_load:
            first_load = False
            env = _viewer_env(active)
            return env.model, env.data
        replacement = reload_callback()
        previous = active
        active = replacement
        previous.teardown()
        print_model_summary(active)
        env = _viewer_env(active)
        return env.model, env.data

    mujoco.viewer.launch(loader=loader)
    return active


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


def run_gs_synced_viewer(
    backend: MujocoTaskBackend,
    gs_cfg,
    reload_callback: Callable[[], MujocoTaskBackend] | None = None,
    width: int = 640,
    height: int = 480,
    *,
    debug: bool = False,
) -> MujocoTaskBackend:
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
        try:
            cv2.imshow(win, frame)
            cv2.waitKey(1)
        except cv2.error:
            # Window was closed (X) — silently skip; outer loop will detect
            # via getWindowProperty and exit cleanly.
            pass

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

    def build_renderer_with_status(current_gs_cfg, current_backend: MujocoTaskBackend):
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
        renderer = _build_gs_renderer(
            current_gs_cfg,
            _viewer_env(current_backend).model,
        )
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

    gs_renderer = build_renderer_with_status(gs_cfg, backend)
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
        new_backend = reload_callback()
        new_gs_cfg = gaussian_config(new_backend)
        if new_gs_cfg is None:
            new_backend.teardown()
            raise RuntimeError("reloaded config no longer defines env.gaussian_render")
        try:
            new_renderer = build_renderer_with_status(new_gs_cfg, new_backend)
        except BaseException:
            # A freshly constructed backend owns resources even when its GS
            # renderer cannot be created.  Keep the previous scene usable and
            # do not leak the failed replacement across reload attempts.
            new_backend.teardown()
            raise
        return new_backend, new_gs_cfg, new_renderer

    camera_state: mujoco.MjvCamera | None = None
    while True:
        restart_requested = False
        with mujoco.viewer.launch_passive(
            _viewer_env(backend).model,
            _viewer_env(backend).data,
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
                    backend.get_env().update()
                    camera_for_render = clone_camera(v.cam)
                v.sync()
                try:
                    render_attempts += 1
                    env = _viewer_env(backend)
                    gs_renderer.update_gaussians(env.data)
                    results = gs_renderer.render(
                        env.model,
                        env.data,
                        [-1],
                        width,
                        height,
                        free_camera=camera_for_render,
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
                    if debug:
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
                # Detect the user clicking the X on the cv2 window. After the
                # Qt backend destroys the window, ``getWindowProperty`` itself
                # raises ``NULL guiReceiver`` instead of returning <1, so treat
                # any error here as "window gone".
                try:
                    win_visible = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE)
                except cv2.error:
                    win_visible = 0.0
                if win_visible < 1:
                    v.close()
                    break  # GS window closed via X
                elapsed = time.time() - step_start
                sleep_for = float(_viewer_env(backend).model.opt.timestep) - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        wait_for_viewer_exit(v)
        if not restart_requested:
            break
        try:
            print("[reload] re-reading YAML/XML/PLY...", flush=True)
            (
                new_backend,
                new_gs_cfg,
                new_renderer,
            ) = load_reloaded_scene_with_status()
            old_backend = backend
            backend, gs_cfg, gs_renderer = (
                new_backend,
                new_gs_cfg,
                new_renderer,
            )
            old_backend.teardown()
            print_model_summary(backend)
            print("[reload] done; reopening MuJoCo viewer", flush=True)
        except Exception as e:
            if debug:
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
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    return backend
