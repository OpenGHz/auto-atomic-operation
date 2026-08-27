"""Red-path regressions for object-centric policy execution boundaries."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from auto_atom.framework import TaskFileConfig
from auto_atom.mock import MockObjectHandler
from auto_atom.policy_eval import (
    ConfigDrivenDemoPolicy,
    PolicyActionFeedback,
    PolicyEvaluator,
)
from auto_atom.runtime import (
    ComponentRegistry,
    ControlSignal,
    PoseState,
    StageExecutionStatus,
)


def _held_waypoint(
    *,
    reference: str = "object",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    return {
        "controlled_frame": {"kind": "held_object"},
        "position": list(position),
        "orientation_goal": {
            "kind": "fixed",
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "reference": reference,
    }


def _task_file(
    env_name: str,
    stages: list[dict[str, object]],
) -> TaskFileConfig:
    ComponentRegistry.register_env(env_name, {"kind": "mock_env", "batch_size": 1})
    return TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {"env_name": env_name, "stages": stages},
            "task_operators": {"arm": {}},
        }
    )


def _ordinary_feedback(
    signal: ControlSignal,
    *,
    sequence_done: bool,
    event: str,
) -> PolicyActionFeedback:
    return PolicyActionFeedback(
        signals=[signal],
        details=[{"event": event}],
        stage_action_sequence_done=[sequence_done],
    )


def _ordinary_evaluator(
    config: TaskFileConfig,
    action_applier: Callable[..., PolicyActionFeedback],
) -> PolicyEvaluator:
    return PolicyEvaluator(action_applier=action_applier).from_config(config)


@pytest.fixture(autouse=True)
def _clear_component_registry() -> None:
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_config_policy_bootstraps_initial_place_before_first_held_pose() -> None:
    """An initially held PLACE must bind before its first config-driven action."""
    config = _task_file(
        "object_policy_initial_place",
        [
            {
                "name": "place_initially_held_plate",
                "object": "rack",
                "operation": "place",
                "operator": "arm",
                "param": {"pre_move": [_held_waypoint(reference="object")]},
            }
        ],
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    backend = evaluator._context.backend
    backend.objects["plate"] = MockObjectHandler(
        name="plate",
        pose=PoseState(position=(0.1, 0.2, 0.15)),
    )
    backend.is_operator_grasping = lambda _operator: np.asarray([True], dtype=bool)
    backend.get_grasped_object_name = lambda _operator, _env_index: "plate"
    try:
        update = evaluator.reset()
        action = policy.act({}, update, evaluator)
        update = evaluator.update(action)

        binding = evaluator._context.get_grasp_binding(0, "arm")
        assert binding is not None
        assert binding.object_name == "plate"
        assert update.done.tolist() == [False]
        assert update.status.tolist() == [StageExecutionStatus.RUNNING]
        assert update.details[0].get("failure_category") != (
            "motion_goal_resolution_failed"
        )
    finally:
        evaluator.close()


def test_external_object_centric_place_far_from_goal_cannot_succeed() -> None:
    """Release alone is insufficient when the held object misses its semantic goal."""
    config = _task_file(
        "object_policy_external_place_far",
        [
            {
                "name": "place_plate_far_from_goal",
                "object": "rack",
                "operation": "place",
                "operator": "arm",
                "param": {
                    "pre_move": [_held_waypoint(reference="object")],
                    "placed_reference": "pre_move",
                    "placed_tolerance": {
                        "position": 0.01,
                        "orientation": 0.05,
                    },
                },
            }
        ],
    )
    grasped = {"value": True}
    calls = {"count": 0}

    def action_applier(*_args: object, **_kwargs: object) -> PolicyActionFeedback:
        calls["count"] += 1
        if calls["count"] == 1:
            return _ordinary_feedback(
                ControlSignal.RUNNING,
                sequence_done=False,
                event="external_place_running",
            )
        grasped["value"] = False
        return _ordinary_feedback(
            ControlSignal.REACHED,
            sequence_done=True,
            event="external_place_released",
        )

    evaluator = _ordinary_evaluator(config, action_applier)
    backend = evaluator._context.backend
    backend.objects["plate"] = MockObjectHandler(
        name="plate",
        pose=PoseState(position=(-0.8, 0.7, 0.5)),
    )
    backend.is_operator_grasping = lambda _operator: np.asarray(
        [grasped["value"]], dtype=bool
    )
    backend.get_grasped_object_name = lambda _operator, _env_index: (
        "plate" if grasped["value"] else None
    )
    try:
        evaluator.reset()
        running = evaluator.update({"phase": "approach"})
        assert running.done.tolist() == [False]

        released = evaluator.update({"phase": "release"})

        assert released.done.tolist() == [True]
        assert released.success.tolist() == [False]
        assert released.status.tolist() == [StageExecutionStatus.FAILED]
        assert released.details[0]["failure_category"] == "placement_failed"
    finally:
        evaluator.close()


def test_external_pick_establishes_binding_before_next_held_stage() -> None:
    """A successful ordinary-policy PICK must authorize the next held goal."""
    config = _task_file(
        "object_policy_external_pick_then_move",
        [
            {
                "name": "pick_plate",
                "object": "plate",
                "operation": "pick",
                "operator": "arm",
                "param": {},
            },
            {
                "name": "reorient_held_plate",
                "object": "",
                "operation": "move",
                "operator": "arm",
                "param": {
                    "pre_move": [
                        _held_waypoint(
                            reference="world",
                            position=(0.4, 0.0, 0.35),
                        )
                    ]
                },
            },
        ],
    )
    grasped = {"value": False}
    calls = {"count": 0}

    def action_applier(*_args: object, **_kwargs: object) -> PolicyActionFeedback:
        calls["count"] += 1
        if calls["count"] == 1:
            return _ordinary_feedback(
                ControlSignal.RUNNING,
                sequence_done=False,
                event="external_pick_running",
            )
        if calls["count"] == 2:
            grasped["value"] = True
            return _ordinary_feedback(
                ControlSignal.REACHED,
                sequence_done=True,
                event="external_pick_succeeded",
            )
        return _ordinary_feedback(
            ControlSignal.RUNNING,
            sequence_done=False,
            event="external_held_move_running",
        )

    evaluator = _ordinary_evaluator(config, action_applier)
    backend = evaluator._context.backend
    backend.is_operator_grasping = lambda _operator: np.asarray(
        [grasped["value"]], dtype=bool
    )
    backend.is_object_grasped = lambda _operator, object_name: np.asarray(
        [grasped["value"] and object_name == "plate"],
        dtype=bool,
    )
    backend.get_grasped_object_name = lambda _operator, _env_index: (
        "plate" if grasped["value"] else None
    )
    try:
        evaluator.reset()
        evaluator.update({"phase": "pick"})
        picked = evaluator.update({"phase": "pick_done"})

        binding = evaluator._context.get_grasp_binding(0, "arm")
        assert picked.done.tolist() == [False]
        assert picked.success.tolist() == [False]
        assert binding is not None
        assert binding.object_name == "plate"

        next_stage = evaluator.update({"phase": "held_move"})
        assert next_stage.done.tolist() == [False]
        assert next_stage.status.tolist() == [StageExecutionStatus.RUNNING]
    finally:
        evaluator.close()


def test_external_place_release_clears_bootstrapped_binding() -> None:
    """An ordinary-policy PLACE must revoke held-object control on release."""
    config = _task_file(
        "object_policy_external_place_clear_binding",
        [
            {
                "name": "place_plate",
                "object": "rack",
                "operation": "place",
                "operator": "arm",
                "param": {
                    "pre_move": [_held_waypoint(reference="object")],
                    "placed_reference": "object",
                    "placed_tolerance": {
                        "position": 0.01,
                        "orientation": 0.05,
                    },
                },
            }
        ],
    )
    grasped = {"value": True}
    calls = {"count": 0}

    def action_applier(*_args: object, **_kwargs: object) -> PolicyActionFeedback:
        calls["count"] += 1
        if calls["count"] == 1:
            return _ordinary_feedback(
                ControlSignal.RUNNING,
                sequence_done=False,
                event="external_place_running",
            )
        grasped["value"] = False
        return _ordinary_feedback(
            ControlSignal.REACHED,
            sequence_done=True,
            event="external_place_released",
        )

    evaluator = _ordinary_evaluator(config, action_applier)
    backend = evaluator._context.backend
    rack_pose = backend.get_object_handler("rack").get_pose().select(0)
    backend.objects["plate"] = MockObjectHandler(name="plate", pose=rack_pose)
    backend.is_operator_grasping = lambda _operator: np.asarray(
        [grasped["value"]], dtype=bool
    )
    backend.get_grasped_object_name = lambda _operator, _env_index: (
        "plate" if grasped["value"] else None
    )
    try:
        evaluator.reset()
        running = evaluator.update({"phase": "place"})
        assert running.done.tolist() == [False]
        assert evaluator._context.get_grasp_binding(0, "arm") is not None

        released = evaluator.update({"phase": "release"})

        assert released.done.tolist() == [True]
        assert released.success.tolist() == [True]
        assert evaluator._context.get_grasp_binding(0, "arm") is None
    finally:
        evaluator.close()


def test_external_pick_does_not_resolve_held_post_move_before_grasp() -> None:
    """Stage start must not resolve a post-grasp held frame before PICK succeeds."""
    config = _task_file(
        "object_policy_external_pick_held_post_move",
        [
            {
                "name": "pick_then_lift_plate",
                "object": "plate",
                "operation": "pick",
                "operator": "arm",
                "param": {
                    "post_move": [
                        _held_waypoint(
                            reference="world",
                            position=(0.35, 0.0, 0.45),
                        )
                    ]
                },
            }
        ],
    )

    def action_applier(*_args: object, **_kwargs: object) -> PolicyActionFeedback:
        return _ordinary_feedback(
            ControlSignal.RUNNING,
            sequence_done=False,
            event="external_pick_approach",
        )

    evaluator = _ordinary_evaluator(config, action_applier)
    backend = evaluator._context.backend
    backend.is_operator_grasping = lambda _operator: np.asarray([False], dtype=bool)
    backend.is_object_grasped = lambda _operator, _object: np.asarray(
        [False], dtype=bool
    )
    backend.get_grasped_object_name = lambda _operator, _env_index: None
    try:
        evaluator.reset()

        running = evaluator.update({"phase": "pre_grasp"})

        assert running.done.tolist() == [False]
        assert running.status.tolist() == [StageExecutionStatus.RUNNING]
        assert evaluator._context.get_grasp_binding(0, "arm") is None
    finally:
        evaluator.close()
