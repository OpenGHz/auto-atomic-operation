"""Contract tests for the shared Stage execution module.

These tests exercise both adapters that feed the shared Stage execution state:
an arbitrary external policy and the configuration-driven scripted policy.
"""

from __future__ import annotations

from copy import deepcopy
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from auto_atom.framework import ArcControlConfig, StageConfig, TaskFileConfig
from auto_atom.mock import MockObjectHandler
from auto_atom.policy_eval import (
    ConfigDrivenDemoPolicy,
    PolicyActionFeedback,
    PolicyEvaluator,
)
from auto_atom.runtime import (
    ComponentRegistry,
    ControlResult,
    ControlSignal,
    PrimitiveAction,
    PoseState,
    StageExecutionStatus,
    TaskFlowBuilder,
    TaskRunner,
)


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


def test_external_policy_press_waits_for_contact() -> None:
    config = _task_file(
        "stage_execution_external_press",
        [
            {
                "name": "press_button",
                "object": "button",
                "operation": "press",
                "operator": "arm",
                "param": {"pre_move": [_world_pose(0.2)]},
            }
        ],
    )
    evaluator = PolicyEvaluator(
        action_applier=lambda *_args, **_kwargs: None
    ).from_config(config)
    backend = evaluator._context.backend
    backend.is_operator_contacting = lambda _operator, _object: np.asarray(
        [False], dtype=bool
    )
    try:
        evaluator.reset()
        pending = evaluator.update(None)

        assert pending.done.tolist() == [False]
        assert pending.success.tolist() == [False]
        assert pending.status.tolist() == [StageExecutionStatus.RUNNING]
        assert pending.details[0]["failure_category"] == "no_contact"
        assert evaluator.records == []

        backend.is_operator_contacting = lambda _operator, _object: np.asarray(
            [True], dtype=bool
        )
        completed = evaluator.update(None)

        assert completed.done.tolist() == [True]
        assert completed.success.tolist() == [True]
        assert completed.status.tolist() == [StageExecutionStatus.SUCCEEDED]
    finally:
        evaluator.close()


def test_legacy_policy_feedback_still_gates_stage_completion() -> None:
    target_position = np.asarray([0.31, 0.0, 0.3], dtype=np.float64)
    config = _task_file(
        "stage_execution_legacy_feedback",
        [_move_stage("move_with_feedback", _world_pose(float(target_position[0])))],
    )
    apply_count = 0

    def action_applier(context: Any, _action: Any, _env_mask: Any = None):
        nonlocal apply_count
        apply_count += 1
        operator = context.backend.get_operator_handler("arm")
        operator.end_effector_pose.position[:] = target_position.reshape(1, 3)
        return SimpleNamespace(
            signals=[ControlSignal.REACHED],
            details=[{"event": "legacy_feedback"}],
            stage_action_sequence_done=np.asarray([apply_count >= 2], dtype=bool),
        )

    evaluator = PolicyEvaluator(action_applier=action_applier).from_config(config)
    try:
        evaluator.reset()
        pending = evaluator.update(None)

        assert pending.done.tolist() == [False]
        assert pending.status.tolist() == [StageExecutionStatus.RUNNING]
        assert pending.details[0]["event"] == "legacy_feedback"

        completed = evaluator.update(None)

        assert completed.done.tolist() == [True]
        assert completed.success.tolist() == [True]
        assert completed.status.tolist() == [StageExecutionStatus.SUCCEEDED]
    finally:
        evaluator.close()


def test_policy_action_override_can_extend_nominal_timeline() -> None:
    """Custom policy action metadata still works past the nominal action span."""

    target_position = np.asarray([0.31, 0.0, 0.3], dtype=np.float64)
    config = _task_file(
        "stage_execution_custom_action_span",
        [_move_stage("move_with_custom_span", _world_pose(float(target_position[0])))],
    )
    evaluator: PolicyEvaluator

    def action_applier(_context: Any, _action: Any, _env_mask: Any = None):
        operator = evaluator._context.backend.get_operator_handler("arm")
        operator.end_effector_pose.position[:] = target_position.reshape(1, 3)
        nominal = evaluator._require_timeline().clone_stage_actions(0)
        # The custom adapter deliberately emits one extra primitive.  The
        # compiled timeline must fall back to the action's own metadata for it.
        custom_actions = [*nominal, deepcopy(nominal[0])]
        assert all(isinstance(action, PrimitiveAction) for action in custom_actions)
        for action in custom_actions:
            if action.pose is not None:
                action.resolved_pose = action.pose
        return PolicyActionFeedback(
            signals=[ControlSignal.REACHED],
            details=[{"event": "custom_action_span"}],
            stage_action_sequence_done=[False],
            stage_actions=[custom_actions],
        )

    evaluator = PolicyEvaluator(action_applier=action_applier).from_config(config)
    try:
        evaluator.reset()
        pending = evaluator.update(None)
        assert pending.done.tolist() == [False]
        completed = evaluator.update(None)
        assert completed.done.tolist() == [True]
        assert completed.success.tolist() == [True]
    finally:
        evaluator.close()


def test_legacy_policy_feedback_press_completion_requires_contact() -> None:
    config = _task_file(
        "stage_execution_legacy_press_feedback",
        [
            {
                "name": "press_button",
                "object": "button",
                "operation": "press",
                "operator": "arm",
                "param": {"pre_move": [_world_pose(0.2)]},
            }
        ],
    )

    def action_applier(_context: Any, _action: Any, _env_mask: Any = None):
        return SimpleNamespace(
            signals=[ControlSignal.REACHED],
            details=[{"event": "legacy_sequence_done"}],
            stage_action_sequence_done=[True],
        )

    evaluator = PolicyEvaluator(action_applier=action_applier).from_config(config)
    evaluator._context.backend.is_operator_contacting = (
        lambda _operator, _object: np.asarray([False], dtype=bool)
    )
    try:
        evaluator.reset()
        failed = evaluator.update(None)

        assert failed.done.tolist() == [True]
        assert failed.success.tolist() == [False]
        assert failed.details[0]["failure_category"] == "no_contact"
    finally:
        evaluator.close()


def test_legacy_policy_failure_feedback_includes_env_index() -> None:
    config = _task_file(
        "stage_execution_legacy_failure",
        [_move_stage("failed_move", _world_pose(0.2))],
    )

    def action_applier(_context: Any, _action: Any, _env_mask: Any = None):
        return PolicyActionFeedback(
            signals=[ControlSignal.FAILED],
            details=[{"event": "forced_failure"}],
            stage_action_sequence_done=[False],
        )

    evaluator = PolicyEvaluator(action_applier=action_applier).from_config(config)
    try:
        evaluator.reset()
        failed = evaluator.update(None)

        assert failed.done.tolist() == [True]
        assert failed.success.tolist() == [False]
        assert failed.details[0]["env_index"] == 0
        assert failed.details[0]["failure_category"] == "controller_failure"
        assert evaluator.records[0].details["env_index"] == 0
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


def test_config_driven_place_uses_public_backend_contracts() -> None:
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
    assert not hasattr(backend, "object_handlers")
    grasp_checks = iter((True, False))
    backend.is_operator_grasping = lambda _operator: np.asarray(
        [next(grasp_checks)], dtype=bool
    )
    backend.get_grasped_object_name = lambda _operator, _env_index: "held"
    backend.get_operator_handler("arm").get_placed_tolerances = lambda: (
        0.01,
        None,
    )
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "placement_failed"
        assert update.details[0]["held_object"] == "held"
    finally:
        evaluator.close()


def test_config_driven_place_does_not_accept_unknown_released_object() -> None:
    config = _task_file(
        "stage_execution_place_unknown_held_object",
        [
            {
                "name": "place_unknown_object",
                "object": "target",
                "operation": "place",
                "operator": "arm",
                "param": {"pre_move": [_world_pose(0.2)]},
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    backend = evaluator._context.backend
    grasp_checks = iter((True, False))
    backend.is_operator_grasping = lambda _operator: np.asarray(
        [next(grasp_checks)], dtype=bool
    )
    backend.get_grasped_object_name = lambda _operator, _env_index: None
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "placement_failed"
        assert update.details[0]["held_object"] == ""
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


@pytest.mark.parametrize("operation", ["pick", "pull"])
@pytest.mark.parametrize(
    ("param", "expected_joint_positions"),
    [
        pytest.param({}, [], id="generated-eef"),
        pytest.param(
            {
                "eef": {
                    "close": True,
                    "joint_positions": [0.25],
                    "require_grasp": False,
                }
            },
            [0.25],
            id="configured-eef",
        ),
    ],
)
def test_pick_and_pull_compile_target_required_grasp(
    operation: str,
    param: dict[str, Any],
    expected_joint_positions: list[float],
) -> None:
    stage = StageConfig.model_validate(
        {
            "name": f"{operation}_target",
            "object": "target",
            "operation": operation,
            "operator": "arm",
            "param": param,
        }
    )

    actions, _ = TaskFlowBuilder.build_actions(stage)

    eef_actions = [action for action in actions if action.kind == "eef"]
    assert len(eef_actions) == 1
    assert eef_actions[0].eef is not None
    assert eef_actions[0].eef.close is True
    assert eef_actions[0].eef.joint_positions == expected_joint_positions
    assert eef_actions[0].eef.require_grasp is True


@pytest.mark.parametrize("operation", ["pick", "pull"])
def test_pick_and_pull_reject_open_eef_override(operation: str) -> None:
    stage = StageConfig.model_validate(
        {
            "name": f"{operation}_target",
            "object": "target",
            "operation": operation,
            "operator": "arm",
            "param": {"eef": {"close": False}},
        }
    )

    with pytest.raises(
        ValueError,
        match="pick and pull operations require a closing EEF command",
    ):
        TaskFlowBuilder.build_actions(stage)


@pytest.mark.parametrize("operation", ["pick", "pull"])
def test_pick_and_pull_reject_grasp_of_a_different_object(operation: str) -> None:
    config = _task_file(
        f"stage_execution_{operation}_target_grasp",
        [
            {
                "name": f"{operation}_target",
                "object": "target",
                "operation": operation,
                "operator": "arm",
                "param": {},
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    backend = evaluator._context.backend
    if operation == "pick":
        grasp_checks = iter((False, True))
        backend.is_operator_grasping = lambda _operator: np.asarray(
            [next(grasp_checks)], dtype=bool
        )
    else:
        backend.is_operator_grasping = lambda _operator: np.asarray([True], dtype=bool)
    backend.is_object_grasped = lambda _operator, _object: np.asarray(
        [False], dtype=bool
    )
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "missing_grasp"
        assert update.details[0]["target_object"] == "target"
        assert update.details[0]["is_operator_grasping"] is True
        assert update.details[0]["is_target_grasped"] is False
    finally:
        evaluator.close()


@pytest.mark.parametrize("operation", ["pick", "pull"])
def test_pick_and_pull_require_the_same_target_after_effect(
    operation: str,
) -> None:
    config = _task_file(
        f"stage_execution_{operation}_retained_target_grasp",
        [
            {
                "name": f"{operation}_target",
                "object": "target",
                "operation": operation,
                "operator": "arm",
                "param": {"post_move": [_world_pose(0.3)]},
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    backend = evaluator._context.backend
    if operation == "pick":
        grasp_checks = iter((False, True, True))
        target_checks = iter((True, False))
        backend.is_operator_grasping = lambda _operator: np.asarray(
            [next(grasp_checks)], dtype=bool
        )
    else:
        target_checks = iter((True, True, False))
        backend.is_operator_grasping = lambda _operator: np.asarray([True], dtype=bool)
    backend.is_object_grasped = lambda _operator, _object: np.asarray(
        [next(target_checks)], dtype=bool
    )
    try:
        update = _run_config_policy(evaluator, policy)

        assert update.success.tolist() == [False]
        assert update.details[0]["failure_stage"] == "postcondition"
        assert update.details[0]["failure_category"] == "missing_grasp"
        assert update.details[0]["target_object"] == "target"
        assert update.details[0]["is_operator_grasping"] is True
        assert update.details[0]["is_target_grasped"] is False
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


def test_config_driven_push_can_require_target_grasp_after_eef() -> None:
    config = _task_file(
        "stage_execution_push_required_grasp",
        [
            {
                "name": "push_with_required_grasp",
                "object": "handle",
                "operation": "push",
                "operator": "arm",
                "param": {
                    "pre_move": [_world_pose(0.2)],
                    "eef": {"close": True, "require_grasp": True},
                },
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
        assert update.details[0]["target_object"] == "handle"
        assert update.details[0]["is_target_grasped"] is False
    finally:
        evaluator.close()


# ---------------------------------------------------------------------------
# Absolute named-joint arc completion
# ---------------------------------------------------------------------------


def _absolute_arc_task_file(
    env_name: str,
    *,
    target_angle: float,
    joint_tolerance: float = 0.01,
    timeout_steps: int = 8,
    batch_size: int = 1,
) -> TaskFileConfig:
    """Build a one-waypoint task whose EEF target follows a named joint."""

    ComponentRegistry.register_env(
        env_name, {"kind": "mock_env", "batch_size": batch_size}
    )
    return TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {
                "env_name": env_name,
                "stages": [
                    {
                        "name": "turn_joint",
                        "object": "",
                        "operation": "move",
                        "operator": "arm",
                        "param": {
                            "pre_move": [
                                {
                                    "reference": "world",
                                    "arc": {
                                        "pivot": "door_hinge",
                                        "axis": [0.0, 0.0, 1.0],
                                        "angle": target_angle,
                                        "absolute": True,
                                        "max_step": 0.2,
                                        "joint_tolerance": joint_tolerance,
                                        "timeout_steps": timeout_steps,
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
            "task_operators": {"arm": {}},
        }
    )


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"joint_tolerance": 0.0}, id="zero-joint-tolerance"),
        pytest.param({"joint_tolerance": -0.01}, id="negative-joint-tolerance"),
        pytest.param({"timeout_steps": 0}, id="zero-timeout"),
        pytest.param({"timeout_steps": -1}, id="negative-timeout"),
    ],
)
def test_absolute_arc_completion_limits_must_be_positive(
    overrides: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        ArcControlConfig(
            pivot="door_hinge",
            axis=(0.0, 0.0, 1.0),
            angle=0.6,
            absolute=True,
            **overrides,
        )


def _immediate_reached_motion(
    backend: Any,
    *,
    initial_angles: tuple[float, ...],
    angles_after_reach: tuple[tuple[float, ...], ...],
    raw_signal: ControlSignal = ControlSignal.REACHED,
) -> SimpleNamespace:
    """Install an EEF-reached controller while a separate joint evolves per tick.

    This deliberately models the failure mode behind absolute arcs: Cartesian
    IK can report its target reached although the mechanism has not yet
    reached its measured joint target.  Joint reads remain pure reads because
    runtime diagnostics may read them more than once per tick.
    """

    batch_size = backend.batch_size
    assert len(initial_angles) == batch_size
    assert len(angles_after_reach) == batch_size
    tracker = SimpleNamespace(
        angles=np.asarray(initial_angles, dtype=np.float64),
        schedules=[list(schedule) for schedule in angles_after_reach],
        ticks=np.zeros(batch_size, dtype=np.int64),
        masks=[],
        raw_signals=[],
    )

    def get_joint_angle(name: str, env_index: int = 0) -> float:
        assert name == "door_hinge"
        return float(tracker.angles[env_index])

    def get_element_pose(name: str, env_index: int = 0) -> PoseState:
        assert name == "door_hinge"
        _ = env_index
        return PoseState(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

    backend.get_joint_angle = get_joint_angle
    backend.get_element_pose = get_element_pose
    operator = backend.get_operator_handler("arm")

    def move_to_pose(
        self: Any,
        pose: Any,
        _target: Any,
        env_mask: Any = None,
    ) -> ControlResult:
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        assert mask.shape == (batch_size,)
        tracker.masks.append(mask.copy())
        signals = np.asarray([ControlSignal.RUNNING] * batch_size, dtype=object)
        details = [{} for _ in range(batch_size)]
        for env_index in np.flatnonzero(mask):
            index = int(env_index)
            self.end_effector_pose.position[index] = np.asarray(
                pose.position, dtype=np.float64
            )
            self.end_effector_pose.orientation[index] = np.asarray(
                pose.orientation, dtype=np.float64
            )
            tick = int(tracker.ticks[index])
            schedule = tracker.schedules[index]
            if schedule:
                tracker.angles[index] = schedule[min(tick, len(schedule) - 1)]
            tracker.ticks[index] += 1
            signals[index] = raw_signal
            details[index] = {"event": "eef_reached", "raw_signal": raw_signal.value}
            tracker.raw_signals.append(raw_signal)
        return ControlResult(signals=signals, details=details)

    operator.move_to_pose = MethodType(move_to_pose, operator)
    return tracker


def _reset_arc_executor(
    execution_path: str,
    config: TaskFileConfig,
) -> tuple[TaskRunner | PolicyEvaluator, ConfigDrivenDemoPolicy | None, Any]:
    if execution_path == "demo":
        runner = TaskRunner().from_config(config)
        return runner, None, runner.reset()
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    return evaluator, policy, evaluator.reset()


def _arc_executor_update(
    execution_path: str,
    executor: TaskRunner | PolicyEvaluator,
    policy: ConfigDrivenDemoPolicy | None,
    update: Any,
    env_mask: np.ndarray | None = None,
) -> Any:
    if execution_path == "demo":
        assert isinstance(executor, TaskRunner)
        return executor.update(env_mask)
    assert isinstance(executor, PolicyEvaluator)
    assert policy is not None
    action = policy.act({}, update, executor)
    return executor.update(action, env_mask)


@pytest.mark.parametrize("execution_path", ["demo", "policy"])
@pytest.mark.parametrize(
    "target_angle,initial_angle,angles_after_reach,joint_tolerance,expected_ticks",
    [
        pytest.param(0.6, 0.0, (0.2, 0.4, 0.6), 0.01, 3, id="positive"),
        pytest.param(0.0, 0.6, (0.4, 0.2, 0.0), 0.01, 3, id="negative"),
        pytest.param(0.6, 0.0, (0.3, 0.75, 0.6), 0.01, 3, id="overshoot"),
        pytest.param(
            0.5,
            0.4921875,
            (0.4921875,),
            0.0078125,
            1,
            id="exact-tolerance-boundary",
        ),
    ],
)
def test_absolute_arc_waits_for_measured_joint_target(
    execution_path: str,
    target_angle: float,
    initial_angle: float,
    angles_after_reach: tuple[float, ...],
    joint_tolerance: float,
    expected_ticks: int,
) -> None:
    """EEF REACHED alone must not advance an absolute named-joint arc."""

    config = _absolute_arc_task_file(
        f"absolute_arc_{execution_path}_{target_angle}_{initial_angle}",
        target_angle=target_angle,
        joint_tolerance=joint_tolerance,
    )
    executor, policy, update = _reset_arc_executor(execution_path, config)
    try:
        backend = executor._require_context().backend
        tracker = _immediate_reached_motion(
            backend,
            initial_angles=(initial_angle,),
            angles_after_reach=(angles_after_reach,),
        )

        for tick in range(expected_ticks):
            update = _arc_executor_update(execution_path, executor, policy, update)
            is_final_tick = tick == expected_ticks - 1
            assert update.done.tolist() == [is_final_tick]
            assert update.success.tolist() == [is_final_tick]
            if not is_final_tick:
                assert update.status.tolist() == [StageExecutionStatus.RUNNING]
                state = executor._env_states[0]
                assert state.active is not None
                # The only compiled absolute-arc primitive must be retried.
                assert state.active.action_index == 0

        assert tracker.raw_signals == [ControlSignal.REACHED] * expected_ticks
        assert [mask.tolist() for mask in tracker.masks] == [[True]] * expected_ticks
        assert abs(float(tracker.angles[0]) - target_angle) <= joint_tolerance
        assert [record.status for record in executor.records] == [
            StageExecutionStatus.SUCCEEDED
        ]
    finally:
        executor.close()


@pytest.mark.parametrize("execution_path", ["demo", "policy"])
@pytest.mark.parametrize(
    ("raw_signal", "failure_category"),
    [
        (ControlSignal.FAILED, "controller_failure"),
        (ControlSignal.TIMED_OUT, "controller_timeout"),
    ],
)
def test_absolute_arc_preserves_raw_controller_terminal_signal(
    execution_path: str,
    raw_signal: ControlSignal,
    failure_category: str,
) -> None:
    config = _absolute_arc_task_file(
        f"absolute_arc_raw_{execution_path}_{raw_signal.value}",
        target_angle=0.6,
    )
    executor, policy, update = _reset_arc_executor(execution_path, config)
    try:
        tracker = _immediate_reached_motion(
            executor._require_context().backend,
            initial_angles=(0.0,),
            angles_after_reach=((0.2,),),
            raw_signal=raw_signal,
        )
        update = _arc_executor_update(execution_path, executor, policy, update)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.status.tolist() == [StageExecutionStatus.FAILED]
        assert update.details[0]["failure_category"] == failure_category
        assert tracker.raw_signals == [raw_signal]
    finally:
        executor.close()


@pytest.mark.parametrize("execution_path", ["demo", "policy"])
def test_absolute_arc_times_out_when_measured_joint_stalls(
    execution_path: str,
) -> None:
    config = _absolute_arc_task_file(
        f"absolute_arc_timeout_{execution_path}",
        target_angle=0.6,
        timeout_steps=2,
    )
    executor, policy, update = _reset_arc_executor(execution_path, config)
    try:
        tracker = _immediate_reached_motion(
            executor._require_context().backend,
            initial_angles=(0.0,),
            angles_after_reach=((0.1, 0.2),),
        )

        update = _arc_executor_update(execution_path, executor, policy, update)
        assert update.done.tolist() == [False]
        assert executor._env_states[0].active is not None
        assert executor._env_states[0].active.action_index == 0

        update = _arc_executor_update(execution_path, executor, policy, update)
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == "controller_timeout"
        assert tracker.raw_signals == [ControlSignal.REACHED, ControlSignal.REACHED]
    finally:
        executor.close()


@pytest.mark.parametrize("execution_path", ["demo", "policy"])
@pytest.mark.parametrize("kind,expected_reaches", [("pose", 1), ("relative_arc", 3)])
def test_non_absolute_pose_actions_keep_reached_progression(
    execution_path: str,
    kind: str,
    expected_reaches: int,
) -> None:
    if kind == "pose":
        stages = [_move_stage("pose", _world_pose(0.3))]
    else:
        stages = [
            {
                "name": "relative_arc",
                "object": "",
                "operation": "move",
                "operator": "arm",
                "param": {
                    "pre_move": [
                        {
                            "reference": "world",
                            "arc": {
                                "pivot": [0.0, 0.0, 0.0],
                                "axis": [0.0, 0.0, 1.0],
                                "angle": 0.5,
                                "max_step": 0.2,
                            },
                        }
                    ]
                },
            }
        ]
    config = _task_file(f"non_absolute_{execution_path}_{kind}", stages)
    executor, policy, update = _reset_arc_executor(execution_path, config)
    try:
        tracker = _immediate_reached_motion(
            executor._require_context().backend,
            initial_angles=(0.0,),
            angles_after_reach=((),),
        )
        for tick in range(expected_reaches):
            update = _arc_executor_update(execution_path, executor, policy, update)
            if tick < expected_reaches - 1:
                assert update.done.tolist() == [False]
                assert executor._env_states[0].active is not None
                assert executor._env_states[0].active.action_index == tick + 1

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert len(tracker.raw_signals) == expected_reaches
    finally:
        executor.close()


@pytest.mark.parametrize("execution_path", ["demo", "policy"])
def test_absolute_arc_partial_batch_masks_are_independent(
    execution_path: str,
) -> None:
    config = _absolute_arc_task_file(
        f"absolute_arc_partial_batch_{execution_path}",
        target_angle=0.6,
        batch_size=2,
    )
    executor, policy, update = _reset_arc_executor(execution_path, config)
    try:
        tracker = _immediate_reached_motion(
            executor._require_context().backend,
            initial_angles=(0.0, 0.0),
            angles_after_reach=((0.2, 0.4, 0.6), (0.2, 0.4, 0.6)),
        )

        update = _arc_executor_update(
            execution_path,
            executor,
            policy,
            update,
            np.asarray([True, False], dtype=bool),
        )
        assert update.status.tolist() == [StageExecutionStatus.RUNNING, "pending"]
        np.testing.assert_allclose(tracker.angles, [0.2, 0.0])
        assert executor._env_states[0].active is not None
        assert executor._env_states[0].active.action_index == 0
        assert executor._env_states[1].active is None

        update = _arc_executor_update(
            execution_path,
            executor,
            policy,
            update,
            np.asarray([False, True], dtype=bool),
        )
        assert update.status.tolist() == [StageExecutionStatus.RUNNING] * 2
        np.testing.assert_allclose(tracker.angles, [0.2, 0.2])
        assert executor._env_states[0].active is not None
        assert executor._env_states[0].active.action_index == 0
        assert executor._env_states[1].active is not None
        assert executor._env_states[1].active.action_index == 0
        assert [mask.tolist() for mask in tracker.masks] == [
            [True, False],
            [False, True],
        ]
    finally:
        executor.close()
