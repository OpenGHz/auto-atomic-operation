"""Parallel full-batch stepping must be bit-identical to sequential stepping.

``EnvConfig.parallel_batch_step`` routes replicated ``step``/``update`` calls
through a worker pool because ``mj_step`` releases the GIL.  Replicas are
independent ``MjModel``/``MjData`` pairs, so the threaded path must produce
byte-for-byte the same per-replica state as the sequential dispatch path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv
from auto_atom.config.env_config import EnvConfig
from auto_atom.scene_composition import SceneConfig


_SCENE_XML = """
<mujoco model="parallel_step_probe">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="arm" pos="0 0 0.1">
      <inertial mass="1" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <joint name="j0" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="obj" pos="0.4 0 0.15">
      <inertial mass="0.1" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>
      <freejoint name="fj"/>
      <geom type="sphere" size="0.03"/>
    </body>
  </worldbody>
  <actuator>
    <position name="a0" joint="j0" kp="50" ctrlrange="-3 3"/>
  </actuator>
</mujoco>
"""


def _make_env(scene_path: Path, *, parallel: bool, batch_size: int = 3):
    config = EnvConfig(
        scene=SceneConfig(base=scene_path),
        batch_size=batch_size,
        parallel_batch_step=parallel,
        enabled_sensors=set(),
    )
    return BatchedUnifiedMujocoEnv(config)


def _drive(env: BatchedUnifiedMujocoEnv, ticks: int, seed: int = 0) -> np.ndarray:
    """Step every replica with identical per-tick ctrl; return final qpos (B, nq)."""
    rng = np.random.default_rng(seed)
    nu = env.envs[0].model.nu
    assert nu == 1
    for tick in range(ticks):
        # Deterministic per-tick target shared by all replicas.
        ctrl = 1.2 * np.sin(tick / 5.0) + 0.05 * rng.uniform(-1, 1)
        env.step(np.full((env.batch_size, nu), ctrl))
    return np.stack([env.envs[i].data.qpos.copy() for i in range(env.batch_size)])


def _final_qpos(env: BatchedUnifiedMujocoEnv) -> np.ndarray:
    return np.stack([env.envs[i].data.qpos.copy() for i in range(env.batch_size)])


def test_parallel_pool_is_created_only_for_replicated_batches(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(_SCENE_XML, encoding="utf-8")
    on = _make_env(scene, parallel=True)
    off = _make_env(scene, parallel=False)
    try:
        assert on._step_pool is not None
        assert off._step_pool is None
        assert len(on.envs) == on.batch_size == 3
    finally:
        on.close()
        off.close()


def test_parallel_step_is_bit_identical_to_sequential(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(_SCENE_XML, encoding="utf-8")
    parallel_env = _make_env(scene, parallel=True)
    sequential_env = _make_env(scene, parallel=False)
    try:
        parallel_env.reset()
        sequential_env.reset()
        initial = _final_qpos(sequential_env)
        final_par = _drive(parallel_env, ticks=120)
        final_seq = _drive(sequential_env, ticks=120)
        for i in range(parallel_env.batch_size):
            np.testing.assert_array_equal(
                final_par[i],
                final_seq[i],
                err_msg=(
                    f"replica {i} diverged between threaded and sequential stepping"
                ),
            )
        # Probe actually exercised physics: state moved from the reset baseline.
        assert np.linalg.norm(final_seq - initial) > 0.0
    finally:
        parallel_env.close()
        sequential_env.close()


def test_parallel_update_with_mask_is_identical(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(_SCENE_XML, encoding="utf-8")
    parallel_env = _make_env(scene, parallel=True, batch_size=2)
    sequential_env = _make_env(scene, parallel=False, batch_size=2)
    try:
        parallel_env.reset()
        sequential_env.reset()
        mask = np.asarray([True, False])
        for _ in range(60):
            parallel_env.update(env_mask=mask)
            sequential_env.update(env_mask=mask)
        for i in range(2):
            np.testing.assert_array_equal(
                parallel_env.envs[i].data.qpos,
                sequential_env.envs[i].data.qpos,
                err_msg=f"masked-update replica {i} diverged",
            )
        # The enabled replica moved while the masked-out replica stayed frozen
        # at the reset baseline in BOTH modes, so the mask was honoured.
        frozen = sequential_env.envs[1].data.qpos.copy()
        assert not np.array_equal(sequential_env.envs[0].data.qpos, frozen)
    finally:
        parallel_env.close()
        sequential_env.close()


def test_parallel_pool_shutdown_is_idempotent(tmp_path: Path) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(_SCENE_XML, encoding="utf-8")
    env = _make_env(scene, parallel=True, batch_size=2)
    env.close()
    env.close()  # second close must not raise
    assert env._step_pool is None


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_parallel_step_worker_caps_keep_results_identical(
    tmp_path: Path, workers: int
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(_SCENE_XML, encoding="utf-8")
    config = EnvConfig(
        scene=SceneConfig(base=scene),
        batch_size=4,
        parallel_batch_step=True,
        parallel_batch_workers=workers,
        enabled_sensors=set(),
    )
    env = BatchedUnifiedMujocoEnv(config)
    seq = _make_env(scene, parallel=False, batch_size=4)
    try:
        assert env._step_pool is not None
        assert env._step_pool._max_workers == workers
        env.reset()
        seq.reset()
        final_par = _drive(env, ticks=60)
        final_seq = _drive(seq, ticks=60)
        for i in range(4):
            np.testing.assert_array_equal(final_par[i], final_seq[i])
    finally:
        env.close()
        seq.close()
