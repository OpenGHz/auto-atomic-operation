"""Contract tests for the shared Stage execution module.

These tests exercise both adapters that feed the shared Stage execution state:
an arbitrary external policy and the configuration-driven scripted policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from auto_atom.framework import TaskFileConfig
from auto_atom.mock import MockObjectHandler
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.runtime import ComponentRegistry, PoseState, StageExecutionStatus


def _world_pose(x: float, *, randomization: dict[str, Any] | None = None) -> dict:
    pose: dict[str, Any] = {
        "reference": "world",
        "position": [x, 0.0, 0.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }
    if randomization is not None:
        pose["randomization"] = randomization
    return pose


def _move_stage(name: str, *poses: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "object": "",
        "operation": "move",
        "operator": "arm",
        "param": {"pre_move": list(poses)},
    }


def _task_file(env_name: str, stages: list[dict[str, Any]]) -> TaskFileConfig:
    ComponentRegistry.register_env(env_name, {"kind": "mock_env", "batch_size": 1})
    return TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {"env_name": env_name, "stages": stages},
            "task_operators": {"arm": {}},
        }
    )


def _run_config_policy(
    evaluator: PolicyEvaluator,
    policy: ConfigDrivenDemoPolicy,
) -> Any:
    policy.reset()
    update = evaluator.reset()
    for _ in range(20):
        if bool(np.all(update.done)):
            return update
        update = evaluator.update(policy.act({}, update, evaluator))
    raise AssertionError("configured policy did not reach a terminal Stage status")


@pytest.fixture(autouse=True)
def _clear_component_registry() -> None:
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_external_policy_target_ignores_waypoint_randomization() -> None:
    """External actions and their nominal success target stay independent of randomization."""

    nominal_position = np.asarray([0.31, 0.0, 0.3], dtype=np.float64)
    config = _task_file(
        "stage_execution_external_randomization",
        [
            _move_stage(
                "nominal_move",
                _world_pose(
                    float(nominal_position[0]),
                    randomization={"x": [0.25, 0.25]},
                ),
            )
        ],
    )
    seen_actions: list[dict[str, Any]] = []

    def action_applier(
        context: Any, action: dict[str, Any], env_mask: Any = None
    ) -> None:
        seen_actions.append(action)
        assert np.asarray(env_mask, dtype=bool).tolist() == [True]
        operator = context.backend.get_operator_handler("arm")
        operator.end_effector_pose.position[:] = np.asarray(
            action["position"], dtype=np.float64
        ).reshape(1, 3)
        operator.end_effector_pose.orientation[:] = np.asarray(
            action["orientation"], dtype=np.float64
        ).reshape(1, 4)

    evaluator = PolicyEvaluator(action_applier=action_applier).from_config(config)
    action = {
        "position": nominal_position.tolist(),
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }
    try:
        evaluator.reset()
        update = evaluator.update(action)

        assert len(seen_actions) == 1
        np.testing.assert_allclose(seen_actions[0]["position"], nominal_position)
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.status.tolist() == [StageExecutionStatus.SUCCEEDED]
        operator_position = (
            evaluator._context.backend.get_operator_handler("arm")
            .get_end_effector_pose()
            .position[0]
        )
        np.testing.assert_allclose(operator_position, nominal_position)
    finally:
        evaluator.close()


def test_config_driven_policy_uses_shared_stage_primitive_cursor() -> None:
    """A reached primitive advances the shared cursor used by the next ``act`` call."""

    config = _task_file(
        "stage_execution_config_cursor",
        [
            _move_stage(
                "two_waypoint_move",
                _world_pose(0.11),
                _world_pose(0.42),
            )
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )

    def waypoint_x(action: Any) -> tuple[int, float]:
        env_action = action.env_actions[0]
        assert env_action is not None
        assert env_action.action.pose is not None
        assert env_action.action.pose.position is not None
        return env_action.action.waypoint, float(env_action.action.pose.position[0])

    try:
        update = evaluator.reset()

        first_action = policy.act({}, update, evaluator)
        assert waypoint_x(first_action) == (0, pytest.approx(0.11))
        update = evaluator.update(first_action)
        assert update.done.tolist() == [False]

        first_retry = policy.act({}, update, evaluator)
        assert waypoint_x(first_retry) == (0, pytest.approx(0.11))
        update = evaluator.update(first_retry)
        assert update.done.tolist() == [False]

        second_action = policy.act({}, update, evaluator)
        assert waypoint_x(second_action) == (1, pytest.approx(0.42))
        update = evaluator.update(second_action)
        assert update.done.tolist() == [False]

        second_retry = policy.act({}, update, evaluator)
        assert waypoint_x(second_retry) == (1, pytest.approx(0.42))
        update = evaluator.update(second_retry)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert [record.status for record in evaluator.records] == [
            StageExecutionStatus.SUCCEEDED
        ]
    finally:
        evaluator.close()


def test_config_driven_place_checks_held_object_target_pose() -> None:
    config = _task_file(
        "stage_execution_place_condition",
        [
            {
                "name": "place_far_from_target",
                "object": "target",
                "operation": "place",
                "operator": "arm",
                "param": {
                    "pre_move": [_world_pose(0.2)],
                    "placed_tolerance": {"position": 0.01},
                },
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    backend = evaluator._context.backend
    backend.objects["held"] = MockObjectHandler(
        name="held",
        pose=PoseState(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    backend.object_handlers = backend.objects
    grasp_checks = iter((True, False))
    backend.is_operator_grasping = lambda _operator: np.asarray(
        [next(grasp_checks)], dtype=bool
    )
    backend.is_object_grasped = lambda _operator, name: np.asarray(
        [name == "held"], dtype=bool
    )
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "placement_failed"
        assert update.details[0]["held_object"] == "held"
    finally:
        evaluator.close()


def test_config_driven_pull_checks_grasp_after_eef() -> None:
    config = _task_file(
        "stage_execution_pull_condition",
        [
            {
                "name": "pull_without_grasp",
                "object": "block",
                "operation": "pull",
                "operator": "arm",
                "param": {},
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "missing_grasp"
    finally:
        evaluator.close()


def test_config_driven_press_checks_contact_after_eef() -> None:
    config = _task_file(
        "stage_execution_press_condition",
        [
            {
                "name": "press_without_contact",
                "object": "button",
                "operation": "press",
                "operator": "arm",
                "param": {"pre_move": [_world_pose(0.2)]},
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "no_contact"
    finally:
        evaluator.close()
