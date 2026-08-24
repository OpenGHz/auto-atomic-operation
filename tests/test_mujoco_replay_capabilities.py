from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np

from auto_atom.basis.mjc.mujoco_env import (
    BatchedUnifiedMujocoEnv,
    UnifiedMujocoEnv,
)


def test_unified_env_sets_scene_joint_state_and_simulation_time() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint name="door_hinge" type="hinge"/>
              <geom type="box" size="0.1 0.1 0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    data.qvel[0] = 2.0
    env = SimpleNamespace(model=model, data=data)

    UnifiedMujocoEnv.set_scene_joint_positions(env, ["door_hinge"], [0.45])
    UnifiedMujocoEnv.set_simulation_time(env, 1.25)

    assert data.qpos[0] == 0.45
    assert data.qvel[0] == 0.0
    assert data.time == 1.25


def test_batched_env_dispatches_replay_state_with_mask() -> None:
    class SingleEnv:
        def __init__(self) -> None:
            self.scene_positions: np.ndarray | None = None
            self.time_sec: float | None = None

        def set_scene_joint_positions(self, names, positions) -> None:
            assert tuple(names) == ("door_hinge",)
            self.scene_positions = np.asarray(positions, dtype=np.float64).copy()

        def set_simulation_time(self, time_sec: float) -> None:
            self.time_sec = time_sec

    env = object.__new__(BatchedUnifiedMujocoEnv)
    env.batch_size = 2
    env.envs = [SingleEnv(), SingleEnv()]

    env.set_scene_joint_positions(
        ["door_hinge"],
        np.asarray([[0.25], [0.75]], dtype=np.float64),
        env_mask=np.asarray([False, True]),
    )
    env.set_simulation_time(2.5, env_mask=np.asarray([True, False]))

    assert env.envs[0].scene_positions is None
    np.testing.assert_allclose(env.envs[1].scene_positions, [0.75])
    assert env.envs[0].time_sec == 2.5
    assert env.envs[1].time_sec is None


def test_batched_shared_physics_updates_replay_state_once() -> None:
    class SingleEnv:
        def __init__(self) -> None:
            self.calls = 0

        def set_simulation_time(self, time_sec: float) -> None:
            assert time_sec == 3.0
            self.calls += 1

    physical_env = SingleEnv()
    env = object.__new__(BatchedUnifiedMujocoEnv)
    env.batch_size = 3
    env.envs = [physical_env, physical_env, physical_env]
    env._share_physics = True

    env.set_simulation_time(3.0)

    assert physical_env.calls == 1


def test_batched_env_exposes_operator_actuator_order() -> None:
    env = object.__new__(BatchedUnifiedMujocoEnv)
    env.config = SimpleNamespace(
        operators={
            "arm": SimpleNamespace(
                arm_actuators=["joint_a", "joint_b"],
                eef_actuators=["gripper"],
            )
        }
    )

    arm, eef = env.get_operator_actuator_names("arm")

    assert arm == ("joint_a", "joint_b")
    assert eef == ("gripper",)
