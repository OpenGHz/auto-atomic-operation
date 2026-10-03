"""Native MuJoCo gripper opening: the pre-release hold and release settling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from auto_atom.backend.mjc.mujoco_backend import (
    MujocoControlConfig,
    MujocoGraspConfig,
    MujocoOperatorHandler,
)
from auto_atom.config.motion import EefControlConfig
from auto_atom.runtime import ControlSignal

_CLOSED = 0.8


class _InstantGripperEnv:
    """Batch-1 env whose gripper joint tracks its command within one step."""

    batch_size = 1

    def __init__(self) -> None:
        data = SimpleNamespace(
            ctrl=np.asarray([_CLOSED], dtype=np.float64),
            qpos=np.asarray([_CLOSED], dtype=np.float64),
        )
        self.envs = [
            SimpleNamespace(
                data=data,
                model=SimpleNamespace(nu=1),
                _op_eef_qidx={"arm": [0]},
            )
        ]
        self.commands: list[float] = []

    def register_operator(self, _operator_name: str, **_kwargs: object) -> None:
        pass

    def step(self, ctrl: np.ndarray, env_mask: np.ndarray) -> None:
        np.testing.assert_array_equal(env_mask, np.asarray([True]))
        data = self.envs[0].data
        data.ctrl[:] = ctrl[0]
        data.qpos[0] = data.ctrl[0]
        self.commands.append(float(data.ctrl[0]))


def _open_until_reached(handler: MujocoOperatorHandler) -> int:
    for update in range(1, 20):
        result = handler.control_eef(EefControlConfig(close=False), target=None)
        if result.signals[0] == ControlSignal.REACHED:
            return update
    raise AssertionError("opening never completed")


def _handler(env: _InstantGripperEnv, **grasp: int) -> MujocoOperatorHandler:
    return MujocoOperatorHandler(
        operator_name="arm",
        env=env,
        eef_open_value=0.0,
        eef_close_value=_CLOSED,
        control=MujocoControlConfig(grasp=MujocoGraspConfig(**grasp)),
    )


def test_opening_holds_the_grip_for_the_pre_release_window() -> None:
    """The jaws stay commanded closed while the arm settles, then open."""
    env = _InstantGripperEnv()
    handler = _handler(env, pre_release_settle_steps=3, release_settle_steps=2)

    reached_at = _open_until_reached(handler)

    assert env.commands[:3] == [_CLOSED] * 3
    assert env.commands[3] == 0.0
    # Release settling is counted after the hold, not concurrently with it.
    assert reached_at == 5


def test_opening_without_a_hold_starts_on_the_first_update() -> None:
    env = _InstantGripperEnv()
    handler = _handler(env)

    assert _open_until_reached(handler) == 1
    assert env.commands == [0.0]


def test_closing_ignores_the_pre_release_window() -> None:
    env = _InstantGripperEnv()
    env.envs[0].data.ctrl[:] = 0.0
    env.envs[0].data.qpos[:] = 0.0
    handler = _handler(env, pre_release_settle_steps=5)

    result = handler.control_eef(EefControlConfig(close=True), target=None)

    assert env.commands == [_CLOSED]
    assert result.signals[0] == ControlSignal.REACHED
