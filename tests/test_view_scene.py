from __future__ import annotations

import contextlib
import os
import signal
import threading
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.backend.mjc import viewer as viewer_module
from examples import view_scene


class _FakeEnv:
    batch_size = 1

    def __init__(self) -> None:
        model = mujoco.MjModel.from_xml_string(
            """
            <mujoco>
              <worldbody><body name="root"><geom type="sphere" size="0.01"/></body></worldbody>
            </mujoco>
            """
        )
        self.envs = [SimpleNamespace(model=model, data=mujoco.MjData(model))]
        self.config = SimpleNamespace(
            viewer=None,
            viewer_env_index=0,
            initial_joint_positions=None,
            gaussian_render=None,
        )
        self.reset_calls = 0
        self.update_calls = 0
        self.refresh_calls = 0
        self.closed = False

    def reset(self, _env_mask=None) -> None:
        self.reset_calls += 1

    def update(self) -> None:
        self.update_calls += 1

    def refresh_viewer(self) -> None:
        self.refresh_calls += 1

    def close(self) -> None:
        self.closed = True


def _backend() -> tuple[MujocoTaskBackend, _FakeEnv]:
    env = _FakeEnv()
    return MujocoTaskBackend(env=env, operator_handlers={}, object_handlers={}), env


def test_load_backend_disables_embedded_viewer_and_runs_canonical_reset(
    monkeypatch,
) -> None:
    backend, env = _backend()
    task_file = SimpleNamespace(task=SimpleNamespace())
    seen = {}

    def prepare(config):
        seen["viewer"] = config.env.viewer
        return task_file

    monkeypatch.setattr(view_scene, "prepare_task_file", prepare)
    monkeypatch.setattr(
        view_scene,
        "construct_scene_backend",
        lambda *_args, **_kwargs: backend,
    )

    cfg = OmegaConf.create({"env": {"viewer": {"distance": 1.0}}})
    loaded = view_scene._load_backend(cfg)

    assert loaded is backend
    assert seen["viewer"] is None
    assert env.reset_calls == 1
    assert env.refresh_calls == 1


def test_without_embedded_viewer_does_not_mutate_hydra_config() -> None:
    cfg = OmegaConf.create({"env": {"viewer": {"distance": 1.0}}})

    isolated = view_scene._without_embedded_viewer(cfg)

    assert cfg.env.viewer.distance == 1.0
    assert isolated.env.viewer is None


def test_script_cli_config_enables_object_frames_without_consuming_hydra_args() -> None:
    argv = [
        "view_scene.py",
        "--debug",
        "--show-object-frames",
        "--config-name",
        "pick_and_place",
        "--",
        "--show-object-frames",
    ]

    config = view_scene._parse_script_cli_config(argv)

    assert config.debug
    assert config.show_object_frames
    assert argv == [
        "view_scene.py",
        "--config-name",
        "pick_and_place",
        "--",
        "--show-object-frames",
    ]


def test_native_viewer_reload_replaces_and_tears_down_backend(monkeypatch) -> None:
    current, current_env = _backend()
    replacement, replacement_env = _backend()
    loaded = {}

    def launch_interruptibly(
        loader,
        *,
        show_object_frames=False,
        on_simulate_ready=None,
    ):
        loaded["show_object_frames"] = show_object_frames
        model, data = loader()
        loaded["initial_model"] = model
        loaded["initial_data"] = data
        model, data = loader()
        loaded["reloaded_model"] = model
        loaded["reloaded_data"] = data

    monkeypatch.setattr(
        viewer_module,
        "_launch_native_viewer_interruptibly",
        launch_interruptibly,
    )

    active = view_scene.run_native_viewer(
        current,
        lambda: replacement,
        show_object_frames=True,
        show_cameras=False,
    )

    assert active is replacement
    assert loaded["show_object_frames"] is True
    assert loaded["initial_model"] is current_env.envs[0].model
    assert loaded["initial_data"] is current_env.envs[0].data
    assert loaded["reloaded_model"] is replacement_env.envs[0].model
    assert loaded["reloaded_data"] is replacement_env.envs[0].data
    assert current_env.closed
    assert not replacement_env.closed


def test_native_viewer_sigint_wakeup_exits_active_simulate(monkeypatch) -> None:
    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    class _FakeSim:
        def __init__(self, *_args, **_kwargs) -> None:
            self.exited = threading.Event()

        def exit(self) -> None:
            self.exited.set()

    created = []

    def fake_simulate(*args, **kwargs):
        simulate = _FakeSim(*args, **kwargs)
        created.append(simulate)
        return simulate

    options = []

    def launch(*, loader):
        del loader
        option = mujoco.MjvOption()
        options.append(option)
        simulate = viewer_module.mujoco.viewer._Simulate(object(), option)
        os.kill(os.getpid(), signal.SIGINT)
        assert simulate.exited.wait(timeout=1.0)

    monkeypatch.setattr(viewer_module.mujoco.viewer, "_Simulate", fake_simulate)
    monkeypatch.setattr(viewer_module.mujoco.viewer, "launch", launch)
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)

    with pytest.raises(KeyboardInterrupt):
        viewer_module._launch_native_viewer_interruptibly(
            lambda: (model, data),
            show_object_frames=True,
        )

    assert len(created) == 1
    assert created[0].exited.is_set()
    assert options[0].frame == mujoco.mjtFrame.mjFRAME_BODY
    assert signal.getsignal(signal.SIGINT) is previous_sigint_handler
    assert not any(
        thread.name == "view-scene-sigint" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_object_frame_visualization_is_opt_in() -> None:
    option = mujoco.MjvOption()

    viewer_module._configure_object_frame_visualization(option, enabled=False)
    assert option.frame == mujoco.mjtFrame.mjFRAME_NONE

    viewer_module._configure_object_frame_visualization(option, enabled=True)
    assert option.frame == mujoco.mjtFrame.mjFRAME_BODY


def test_gaussian_config_comes_from_backend_environment() -> None:
    backend, env = _backend()
    assert view_scene.gaussian_config(backend) is None

    env.config.gaussian_render = SimpleNamespace(
        body_gaussians={"root": "root.ply"},
        background_ply=None,
    )

    assert view_scene.gaussian_config(backend) is env.config.gaussian_render


class _FakeCameraEnv:
    """Environment stub exposing the camera surface the overlay consumes."""

    def __init__(self, cameras=(("cam_a", 640, 480), ("cam_b", 640, 480))) -> None:
        self.config = SimpleNamespace(
            cameras=[
                SimpleNamespace(name=name, width=width, height=height)
                for name, width, height in cameras
            ]
        )
        self.rendered: list[str] = []

    def render_camera_rgb(self, camera_name: str) -> np.ndarray:
        self.rendered.append(camera_name)
        return np.full((48, 64, 3), 60 + len(camera_name), dtype=np.uint8)


class _FakeSimulate:
    """Stands in for MuJoCo's ``_Simulate``: lock, viewport, set_images."""

    def __init__(self, viewport: mujoco.MjrRect) -> None:
        self.viewport = viewport
        self.uploads: list[list[tuple[mujoco.MjrRect, np.ndarray]]] = []
        self.lock_calls = 0

    def lock(self):
        self.lock_calls += 1
        return contextlib.nullcontext()

    def set_images(self, images) -> None:
        self.uploads.append(list(images))


def _viewport(width: int = 1200, height: int = 700) -> mujoco.MjrRect:
    return mujoco.MjrRect(0, 0, width, height)


def test_camera_overlay_publishes_one_labelled_tile_per_camera() -> None:
    env = _FakeCameraEnv()
    overlay = viewer_module.CameraOverlay(lambda: env)
    simulate = _FakeSimulate(_viewport())

    assert overlay.show(simulate) is True

    assert env.rendered == ["cam_a", "cam_b"]
    assert simulate.lock_calls == 1
    assert len(simulate.uploads) == 1
    tiles = simulate.uploads[0]
    assert [rect.height for rect, _ in tiles] == [180, 180]
    assert [rect.width for rect, _ in tiles] == [240, 240]
    assert [rect.bottom for rect, _ in tiles] == [12, 12]
    # Right-aligned inside the scene viewport, preserving config order.
    assert [rect.left for rect, _ in tiles] == [702, 948]
    assert [rect.left + rect.width for rect, _ in tiles] == [942, 1188]
    for rect, image in tiles:
        assert image.shape == (rect.height, rect.width, 3)
        # Uploaded rows are bottom-up, so the label plate lands last.
        assert tuple(image[-1, 0]) == viewer_module._OVERLAY_LABEL_BG
        assert (image > 200).all(axis=-1).any()


def test_camera_overlay_wraps_into_rows_when_one_row_is_too_wide() -> None:
    env = _FakeCameraEnv(
        cameras=tuple((f"cam_{index}", 640, 480) for index in range(4))
    )
    overlay = viewer_module.CameraOverlay(lambda: env)
    simulate = _FakeSimulate(_viewport(width=560, height=420))

    assert overlay.show(simulate) is True

    tiles = simulate.uploads[0]
    assert len(tiles) == 4
    assert len({rect.bottom for rect, _ in tiles}) == 2


def test_camera_overlay_throttles_and_rebinds_after_an_env_swap() -> None:
    first = _FakeCameraEnv(cameras=(("cam_a", 640, 480),))
    current = {"env": first}
    overlay = viewer_module.CameraOverlay(lambda: current["env"])
    simulate = _FakeSimulate(_viewport())

    assert overlay.show(simulate) is True
    assert overlay.show(simulate) is False
    assert len(simulate.uploads) == 1

    overlay.min_interval_s = 0.0
    second = _FakeCameraEnv(cameras=(("cam_x", 320, 240),))
    current["env"] = second

    assert overlay.show(simulate) is True
    assert second.rendered == ["cam_x"]
    assert first.rendered == ["cam_a"]


def test_camera_overlay_disables_itself_when_the_config_has_no_cameras(
    capsys: pytest.CaptureFixture[str],
) -> None:
    overlay = viewer_module.CameraOverlay(lambda: _FakeCameraEnv(cameras=()))
    simulate = _FakeSimulate(_viewport())

    assert overlay.show(simulate) is False
    assert overlay.disabled
    assert simulate.uploads == []
    assert "declares no cameras" in capsys.readouterr().out


def test_camera_overlay_reports_render_failures_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _BrokenEnv(_FakeCameraEnv):
        def render_camera_rgb(self, camera_name: str) -> np.ndarray:
            raise RuntimeError("no GL context")

    overlay = viewer_module.CameraOverlay(lambda: _BrokenEnv())
    simulate = _FakeSimulate(_viewport())

    assert overlay.show(simulate) is False
    assert overlay.disabled
    assert overlay.show(simulate) is False
    assert capsys.readouterr().out.count("camera overlay disabled") == 1


def test_native_viewer_runs_and_joins_the_camera_overlay_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _env = _backend()
    shown: list[tuple[object, str]] = []

    class _Overlay:
        min_interval_s = 0.01

        def __init__(self, env_provider) -> None:
            self.env_provider = env_provider

        def show(self, simulate) -> bool:
            shown.append((simulate, threading.current_thread().name))
            return True

        @property
        def disabled(self) -> bool:
            return False

    monkeypatch.setattr(viewer_module, "CameraOverlay", _Overlay)

    def launch(loader, *, show_object_frames=False, on_simulate_ready=None):
        loader()
        assert on_simulate_ready is not None
        on_simulate_ready("simulate")
        deadline = time.monotonic() + 2.0
        while not shown and time.monotonic() < deadline:
            time.sleep(0.01)
        assert shown

    monkeypatch.setattr(
        viewer_module,
        "_launch_native_viewer_interruptibly",
        launch,
    )

    assert view_scene.run_native_viewer(backend, lambda: backend) is backend

    assert shown[0][0] == "simulate"
    assert shown[0][1] == "view-scene-camera-overlay"
    assert not any(
        thread.name == "view-scene-camera-overlay" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_native_viewer_skips_the_camera_overlay_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, _env = _backend()

    def fail_overlay(_env_provider):
        raise AssertionError("camera overlay must not be built")

    monkeypatch.setattr(viewer_module, "CameraOverlay", fail_overlay)

    def launch(loader, *, show_object_frames=False, on_simulate_ready=None):
        loader()
        assert on_simulate_ready is not None
        on_simulate_ready("simulate")

    monkeypatch.setattr(
        viewer_module,
        "_launch_native_viewer_interruptibly",
        launch,
    )

    active = view_scene.run_native_viewer(
        backend,
        lambda: backend,
        show_cameras=False,
    )

    assert active is backend


def test_script_cli_config_controls_camera_previews() -> None:
    assert view_scene.ViewSceneCliConfig().show_cameras is True

    argv = ["view_scene.py", "--no-show-cameras", "--config-name", "pick_and_place"]

    config = view_scene._parse_script_cli_config(argv)

    assert config.show_cameras is False
    assert argv == ["view_scene.py", "--config-name", "pick_and_place"]
