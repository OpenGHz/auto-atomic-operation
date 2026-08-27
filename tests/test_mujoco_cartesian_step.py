"""Regression tests for MuJoCo Cartesian command shaping."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from auto_atom.backend.mjc.mujoco_backend import (
    MujocoControlConfig,
    MujocoOperatorHandler,
)
from auto_atom.framework import PoseControlConfig
from auto_atom.runtime import ControlSignal


class _StepFollowingMujocoEnv:
    """Minimal batch-1 env that exactly reaches each Cartesian sub-step."""

    batch_size = 1

    def __init__(self) -> None:
        self.eef_position = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        self.eef_orientation = np.asarray(
            [[0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.commanded_positions: list[np.ndarray] = []
        self.envs = [
            SimpleNamespace(
                data=SimpleNamespace(ctrl=np.zeros(1, dtype=np.float64)),
                model=SimpleNamespace(nu=1),
                get_operator_ik_failure_streak=lambda _operator_name: 0,
            )
        ]

    def register_operator(self, _operator_name: str, **_kwargs: object) -> None:
        pass

    def get_operator_eef_pose_in_world(
        self,
        _operator_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.eef_position.copy(), self.eef_orientation.copy()

    def world_to_base(
        self,
        _operator_name: str,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(position, dtype=np.float64).reshape(1, 3),
            np.asarray(orientation, dtype=np.float64).reshape(1, 4),
        )

    def step_operator_toward_target(
        self,
        _operator_name: str,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        *,
        env_mask: np.ndarray,
    ) -> None:
        np.testing.assert_array_equal(env_mask, np.asarray([True]))
        self.commanded_positions.append(target_position[0].astype(np.float64).copy())
        self.eef_position[0] = target_position[0]
        self.eef_orientation[0] = target_orientation[0]


def test_small_cartesian_step_does_not_complete_final_waypoint() -> None:
    """Reaching a clamped sub-step is not reaching the configured waypoint."""
    env = _StepFollowingMujocoEnv()
    handler = MujocoOperatorHandler(
        operator_name="arm",
        env=env,
        control=MujocoControlConfig(timeout_steps=10),
    )
    final_position = np.asarray([0.1, 0.0, 0.0], dtype=np.float64)
    max_linear_step = 0.005
    position_tolerance = 0.001
    pose = PoseControlConfig.model_validate(
        {
            "position": final_position.tolist(),
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "reference": "world",
            "max_linear_step": max_linear_step,
            "tolerance": {
                "position": position_tolerance,
                "orientation": 0.01,
            },
        }
    )

    result = handler.move_to_pose(
        pose,
        target=None,
        env_mask=np.asarray([True]),
    )

    assert len(env.commanded_positions) == 1
    intermediate_goal = env.commanded_positions[0]
    np.testing.assert_allclose(
        intermediate_goal,
        [max_linear_step, 0.0, 0.0],
        atol=1.0e-9,
    )
    intermediate_error = np.linalg.norm(env.eef_position[0] - intermediate_goal)
    final_error = np.linalg.norm(env.eef_position[0] - final_position)
    assert intermediate_error < position_tolerance
    assert final_error > position_tolerance
    assert result.signals[0] == ControlSignal.RUNNING
    assert result.details[0]["event"] == "moving"
    assert result.details[0]["command_step_position_error"] == pytest.approx(
        0.0,
        abs=1.0e-9,
    )
    assert result.details[0]["position_error"] == pytest.approx(final_error)
