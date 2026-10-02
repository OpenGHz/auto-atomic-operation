"""``EnvConfig.gravity`` overrides the composed scene's gravity on every backend."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv
from auto_atom.config.env_config import EnvConfig
from auto_atom.scene_composition import SceneConfig

_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="ball" pos="0 0 1">
      <freejoint name="ball_free"/>
      <geom type="sphere" size="0.05" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def scene_path(tmp_path: Path) -> Path:
    path = tmp_path / "scene.xml"
    path.write_text(_XML, encoding="utf-8")
    return path


def _ball_height(env: UnifiedMujocoEnv) -> float:
    body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    return float(env.data.xpos[body, 2])


def test_gravity_defaults_to_scene_option(scene_path: Path) -> None:
    env = UnifiedMujocoEnv(EnvConfig(scene=SceneConfig(base=scene_path)))
    try:
        env.reset()
        np.testing.assert_allclose(env.model.opt.gravity, [0.0, 0.0, -9.81])
        for _ in range(50):
            env.update()
        assert _ball_height(env) < 0.99
    finally:
        env.close()


def test_zero_gravity_keeps_free_body_in_place_across_reset(scene_path: Path) -> None:
    env = UnifiedMujocoEnv(
        EnvConfig(scene=SceneConfig(base=scene_path), gravity=[0.0, 0.0, 0.0])
    )
    try:
        env.reset()
        np.testing.assert_array_equal(env.model.opt.gravity, 0.0)
        for _ in range(50):
            env.update()
        assert _ball_height(env) == pytest.approx(1.0)
    finally:
        env.close()


def test_gravity_requires_three_components(scene_path: Path) -> None:
    with pytest.raises(ValidationError):
        EnvConfig(scene=SceneConfig(base=scene_path), gravity=[0.0, -9.81])


def test_mjwarp_applies_gravity_to_device_model(scene_path: Path) -> None:
    pytest.importorskip("mujoco_warp")
    from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv

    env = MjWarpObjectOnlyEnv(
        EnvConfig(scene=SceneConfig(base=scene_path), gravity=[0.0, 0.0, 0.0])
    )
    try:
        np.testing.assert_array_equal(env.host_model.opt.gravity, 0.0)
        env.state.step(50)
        positions, _ = env.state.get_body_pose_batch("ball")
        np.testing.assert_allclose(positions[:, 2], 1.0, atol=1e-6)
    finally:
        env.close()
