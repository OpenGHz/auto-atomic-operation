"""Physical contract tests for the switchable door latch."""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest
from hydra import compose, initialize_config_dir
from pydantic import ValidationError

from auto_atom.callbacks.door_latch import DoorLatchCallback, DoorLatchConfig
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import TaskRunner


_MODEL_XML = """
<mujoco>
  <option timestep="0.001" iterations="20"/>
  <worldbody>
    <body name="door">
      <joint name="door_hinge" type="hinge" damping="1"/>
      <geom type="box" size="0.4 0.02 0.8" pos="-0.4 0 0.8" mass="20"/>
      <body name="handle" pos="-0.7 0 0.8">
        <joint name="handle_hinge" type="hinge" springref="0" stiffness="1"/>
        <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.01" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint name="door_latch_lock" joint1="door_hinge"
           polycoef="0 0 0 0 0" active="true"
           solref="0.002 1" solimp="0.99 0.999 0.001"/>
  </equality>
</mujoco>
"""


def _latch(
    *,
    handle_direction: int = 1,
) -> tuple[mujoco.MjModel, mujoco.MjData, DoorLatchCallback]:
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    callback = DoorLatchCallback(
        DoorLatchConfig(
            lock_constraint="door_latch_lock",
            door_joint="door_hinge",
            handle_joint="handle_hinge",
            handle_direction=handle_direction,
            unlock_travel=0.12,
            relock_travel=0.08,
            relock_zone=0.02,
        )
    )
    callback.bind(model, data)
    return model, data, callback


def _joint_addresses(model: mujoco.MjModel, name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    callback: DoorLatchCallback,
    *,
    torque: float = 0.0,
    count: int = 1,
) -> None:
    _, door_dof = _joint_addresses(model, "door_hinge")
    for _ in range(count):
        data.qfrc_applied[door_dof] = torque
        callback(model, data)
        mujoco.mj_step(model, data)


def test_locked_door_resists_torque_until_handle_unlocks() -> None:
    model, data, callback = _latch()
    door_qpos, _ = _joint_addresses(model, "door_hinge")
    handle_qpos, _ = _joint_addresses(model, "handle_hinge")

    _step(model, data, callback, torque=1000.0, count=200)
    assert bool(data.eq_active[0])
    assert abs(float(data.qpos[door_qpos])) < 1e-3

    data.qpos[handle_qpos] = 0.12
    _step(model, data, callback, torque=100.0, count=100)
    assert not bool(data.eq_active[0])
    assert float(data.qpos[door_qpos]) > 0.05


def test_latch_relocks_only_after_hysteresis_and_inside_capture_zone() -> None:
    model, data, callback = _latch()
    door_qpos, door_dof = _joint_addresses(model, "door_hinge")
    handle_qpos, _ = _joint_addresses(model, "handle_hinge")

    data.qpos[handle_qpos] = 0.12
    callback(model, data)
    assert not bool(data.eq_active[0])

    data.qpos[handle_qpos] = 0.10
    callback(model, data)
    assert not bool(data.eq_active[0])

    data.qpos[door_qpos] = 0.03
    data.qpos[handle_qpos] = 0.08
    callback(model, data)
    assert not bool(data.eq_active[0])

    data.qpos[door_qpos] = 0.01
    data.qvel[door_dof] = 0.0
    callback(model, data)
    assert bool(data.eq_active[0])
    _step(model, data, callback, count=50)
    assert abs(float(data.qpos[door_qpos])) < 1e-3


def test_negative_handle_direction_and_reset_are_supported() -> None:
    model, data, callback = _latch(handle_direction=-1)
    handle_qpos, _ = _joint_addresses(model, "handle_hinge")

    data.qpos[handle_qpos] = -0.12
    callback(model, data)
    assert not bool(data.eq_active[0])

    other = mujoco.MjData(model)
    assert bool(other.eq_active[0])
    mujoco.mj_resetData(model, data)
    assert bool(data.eq_active[0])


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"handle_direction": 0}, id="zero-direction"),
        pytest.param({"unlock_travel": 0.0}, id="zero-unlock"),
        pytest.param({"relock_travel": 0.12}, id="no-hysteresis"),
        pytest.param({"relock_zone": 0.0}, id="zero-zone"),
    ],
)
def test_latch_config_rejects_invalid_values(overrides: dict[str, float | int]) -> None:
    values = {
        "lock_constraint": "door_latch_lock",
        "door_joint": "door_hinge",
        "handle_joint": "handle_hinge",
        "handle_direction": 1,
        "unlock_travel": 0.12,
        "relock_travel": 0.08,
        "relock_zone": 0.02,
        **overrides,
    }
    with pytest.raises(ValidationError):
        DoorLatchConfig.model_validate(values)


def test_bind_rejects_a_missing_lock_constraint() -> None:
    model = mujoco.MjModel.from_xml_string(_MODEL_XML)
    data = mujoco.MjData(model)
    callback = DoorLatchCallback(
        DoorLatchConfig(
            lock_constraint="missing",
            door_joint="door_hinge",
            handle_joint="handle_hinge",
            unlock_travel=0.12,
            relock_travel=0.08,
        )
    )
    with pytest.raises(ValueError, match="equality constraint 'missing' not found"):
        callback.bind(model, data)


def test_default_open_door_task_binds_and_releases_the_hard_latch() -> None:
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "aao_configs"),
    ):
        config = compose(
            config_name="open_door",
            overrides=[
                "env.cameras=[]",
                "env.enabled_sensors=[]",
                "env.viewer=null",
            ],
        )

    runner = TaskRunner().from_config(prepare_task_file(config))
    try:
        env = runner._context.backend.get_env().envs[0]
        latch_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            "door_latch_lock",
        )
        handle_qpos, _ = _joint_addresses(env.model, "handle_hinge")

        assert len(env._pre_step_callbacks) == 1
        assert isinstance(env._pre_step_callbacks[0], DoorLatchCallback)
        assert bool(env.data.eq_active[latch_id])

        env.data.qpos[handle_qpos] = 0.12
        env._pre_step_callbacks[0](env.model, env.data)
        assert not bool(env.data.eq_active[latch_id])
    finally:
        runner.close()
