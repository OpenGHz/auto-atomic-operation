"""Tests for the native camera-render seam used by preview tools."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from auto_atom.basis.mjc import mujoco_basis
from auto_atom.basis.mjc.mujoco_basis import MujocoBasis


class _FakeRenderer:
    def __init__(self, image: np.ndarray | None = None) -> None:
        self.image = np.zeros((4, 6, 3), dtype=np.uint8) if image is None else image
        self.scene = SimpleNamespace(geoms=[], ngeom=0)
        self.update_calls: list[tuple] = []
        self.depth_enabled = True
        self.segmentation_enabled = True
        self.closed = False

    def update_scene(self, data, camera, scene_option) -> None:
        self.update_calls.append((data, camera, scene_option))

    def disable_depth_rendering(self) -> None:
        self.depth_enabled = False

    def disable_segmentation_rendering(self) -> None:
        self.segmentation_enabled = False

    def render(self) -> np.ndarray:
        return self.image

    def close(self) -> None:
        self.closed = True


def _camera_env(*, with_renderer: bool = True) -> MujocoBasis:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody><camera name="cam_a" pos="0 -1 0.5"/></worldbody>
        </mujoco>
        """
    )
    env = object.__new__(MujocoBasis)
    env.model = model
    env.data = mujoco.MjData(model)
    env._camera_specs = {"cam_a": SimpleNamespace(name="cam_a", width=8, height=6)}
    env._camera_ids = {"cam_a": 0}
    env._renderer_scene_option = mujoco.MjvOption()
    env._renderer_scene_option.sitegroup[:] = 0
    env._camera_hidden_geom_ids = frozenset()
    env._renderers = {"cam_a": _FakeRenderer()} if with_renderer else {}
    env._preview_renderers = {}
    return env


def _recording_renderer_factory(created: list[tuple]) -> type:
    class _Factory:
        def __init__(self, model, *, height, width) -> None:
            created.append((model, height, width))
            self._renderer = _FakeRenderer()

        def __getattr__(self, name):
            return getattr(self._renderer, name)

    return _Factory


def test_render_camera_rgb_uses_the_observation_renderer() -> None:
    env = _camera_env()
    renderer = env._renderers["cam_a"]
    image = np.full((4, 6, 3), 7, dtype=np.uint8)
    renderer.image = image

    frame = env.render_camera_rgb("cam_a")

    assert frame is image
    assert renderer.update_calls == [(env.data, 0, env._renderer_scene_option)]
    assert not renderer.depth_enabled
    assert not renderer.segmentation_enabled
    assert env._preview_renderers == {}


def test_render_camera_rgb_allocates_a_cached_preview_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple] = []
    monkeypatch.setattr(
        mujoco_basis.mujoco,
        "Renderer",
        _recording_renderer_factory(created),
    )
    env = _camera_env(with_renderer=False)

    first = env.render_camera_rgb("cam_a")
    second = env.render_camera_rgb("cam_a")

    assert list(created) == [(env.model, 6, 8)]
    assert set(env._preview_renderers) == {"cam_a"}
    assert first.shape == (4, 6, 3)
    assert second.shape == (4, 6, 3)


def test_render_camera_rgb_rejects_unknown_cameras() -> None:
    env = _camera_env()

    with pytest.raises(KeyError):
        env.render_camera_rgb("missing")


def test_close_releases_preview_renderers() -> None:
    env = _camera_env(with_renderer=False)
    preview = _FakeRenderer()
    env._preview_renderers = {"cam_a": preview}
    env._viewer = None

    env.close()

    assert preview.closed
    assert env._preview_renderers == {}
