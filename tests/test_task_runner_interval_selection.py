from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from auto_atom.framework import (
    IntervalSelectionConfig,
    PoseControlConfig,
    StageConfig,
    TaskFileConfig,
    TaskKeypointConfig,
    TaskPhase,
)
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runner.common import ExampleLoopHooks, run_example_rounds
from auto_atom.runtime import (
    ArcExecutionSnapshot,
    ComponentRegistry,
    PrimitiveAction,
    TaskFlowBuilder,
    TaskRunner,
)


def _pose(x: float) -> dict:
    return {
        "reference": "world",
        "position": [x, 0.0, 0.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


def _move_stage(name: str, *positions: float) -> dict:
    return {
        "name": name,
        "object": "block",
        "operation": "move",
        "operator": "arm",
        "param": {"pre_move": [_pose(position) for position in positions]},
    }


def _task_payload(*, batch_size: int = 1, interval: dict | None = None) -> dict:
    env_name = f"interval_mock_{batch_size}"
    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": batch_size},
    )
    payload = {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {
            "env_name": env_name,
            "stages": [
                {
                    "name": "selected",
                    "object": "block",
                    "operation": "move",
                    "operator": "arm",
                    "param": {
                        "pre_move": [_pose(0.1), _pose(0.2)],
                        "post_move": [_pose(0.3), _pose(0.4)],
                    },
                },
                {
                    "name": "excluded",
                    "object": "block",
                    "operation": "move",
                    "operator": "arm",
                    "param": {"pre_move": [_pose(0.9)]},
                },
            ],
        },
        "task_operators": {"arm": {}},
    }
    if interval is not None:
        payload["execution"] = {"interval_selection": interval}
    return payload


def _keypoint(
    stage: str,
    phase: str,
    waypoint: int,
    *,
    side: str | None = None,
) -> dict:
    point = {"stage": stage, "phase": phase, "waypoint": waypoint}
    if side is not None:
        point["side"] = side
    return point


def _operator_positions(runner: TaskRunner) -> np.ndarray:
    assert runner._context is not None
    return (
        runner._context.backend.get_operator_handler("arm")
        .get_end_effector_pose()
        .position
    )


@pytest.fixture(autouse=True)
def _clear_component_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_interval_selection_absent_preserves_reset_behavior() -> None:
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(_task_payload()))
    try:
        update = runner.reset()

        assert _operator_positions(runner)[0, 0] == pytest.approx(0.2)
        assert update.phase == [None]
        assert update.phase_step.tolist() == [-1]
        assert update.done.tolist() == [False]
        assert update.status.tolist() == ["pending"]
    finally:
        runner.close()


def test_interval_selection_absent_preserves_legacy_eef_phase_step() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        {
            "name": "grasp",
            "object": "block",
            "operation": "grasp",
            "operator": "arm",
            "param": {"eef": {"close": True}},
        }
    ]
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        runner.reset()
        update = runner.update()

        assert update.phase == ["eef"]
        assert update.phase_step.tolist() == [-1]
    finally:
        runner.close()


def test_top_level_interval_selection_is_rejected_with_migration_path() -> None:
    payload = _task_payload()
    point = _keypoint("selected", "post_move", 0)
    payload["interval_selection"] = {"start": point, "stop": point}

    with pytest.raises(
        ValidationError,
        match="use execution.interval_selection instead",
    ):
        TaskFileConfig.model_validate(payload)


def test_task_keypoint_side_defaults_to_none_outside_interval_selection() -> None:
    point = TaskKeypointConfig(
        stage="selected",
        phase=TaskPhase.PRE_MOVE,
        waypoint=0,
    )

    assert point.side is None
    assert point.model_dump(mode="json")["side"] is None


def test_interval_endpoint_sides_have_role_specific_defaults() -> None:
    config = TaskFileConfig.model_validate(
        _task_payload(
            interval={
                "start": _keypoint("selected", "pre_move", 0),
                "stop": _keypoint("selected", "post_move", 1),
            }
        )
    )

    selection = config.execution.interval_selection
    assert selection is not None
    assert selection.start.__class__ is TaskKeypointConfig
    assert selection.stop.__class__ is TaskKeypointConfig
    assert selection.start.side is not None
    assert selection.stop.side is not None
    assert selection.start.side.value == "before"
    assert selection.stop.side.value == "after"
    assert selection.model_dump(mode="json") == {
        "start": {
            "stage": "selected",
            "phase": "pre_move",
            "waypoint": 0,
            "side": "before",
        },
        "stop": {
            "stage": "selected",
            "phase": "post_move",
            "waypoint": 1,
            "side": "after",
        },
        "max_fast_forward_updates": 10_000,
    }


def test_interval_resolves_task_keypoint_config_instance_sides_by_role() -> None:
    point = TaskKeypointConfig(
        stage="selected",
        phase=TaskPhase.PRE_MOVE,
        waypoint=0,
    )

    selection = IntervalSelectionConfig(start=point, stop=point)

    assert selection.start.side is not None
    assert selection.stop.side is not None
    assert selection.start.side.value == "before"
    assert selection.stop.side.value == "after"


def test_interval_treats_explicit_none_side_as_role_adaptive() -> None:
    point = {
        "stage": "selected",
        "phase": "pre_move",
        "waypoint": 0,
        "side": None,
    }

    selection = IntervalSelectionConfig(start=point, stop=point)

    assert selection.start.side is not None
    assert selection.stop.side is not None
    assert selection.start.side.value == "before"
    assert selection.stop.side.value == "after"


def test_interval_preserves_explicit_endpoint_sides() -> None:
    selection = IntervalSelectionConfig(
        start=TaskKeypointConfig(
            stage="selected",
            phase=TaskPhase.PRE_MOVE,
            waypoint=0,
            side="after",
        ),
        stop=TaskKeypointConfig(
            stage="selected",
            phase=TaskPhase.POST_MOVE,
            waypoint=0,
            side="before",
        ),
    )

    assert selection.start.side is not None
    assert selection.stop.side is not None
    assert selection.start.side.value == "after"
    assert selection.stop.side.value == "before"
    dumped = selection.model_dump(mode="json")
    assert dumped["start"]["side"] == "after"
    assert dumped["stop"]["side"] == "before"


def test_default_start_before_first_keypoint_performs_no_fast_forward_ticks() -> None:
    point = _keypoint("selected", "pre_move", 0)
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(interval={"start": point, "stop": point})
        )
    )
    try:
        update = runner.reset()

        assert _operator_positions(runner)[0, 0] == pytest.approx(0.2)
        assert update.done.tolist() == [False]
        assert update.stage_name == ["selected"]
        assert update.phase == ["pre_move"]
        assert update.phase_step.tolist() == [0]
        assert update.details[0]["interval_selection"]["fast_forward_updates"] == 0
        assert update.details[0]["interval_selection"]["start"]["side"] == ("before")
    finally:
        runner.close()


def test_explicit_after_start_is_reached_and_default_stop_runs_through_keypoint() -> (
    None
):
    interval = {
        "start": _keypoint("selected", "post_move", 0, side="after"),
        "stop": _keypoint("selected", "post_move", 1),
    }
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload(interval=interval))
    )
    try:
        reset_update = runner.reset()

        assert _operator_positions(runner)[0, 0] == pytest.approx(0.3)
        assert reset_update.stage_name == ["selected"]
        assert reset_update.phase == ["post_move"]
        assert reset_update.phase_step.tolist() == [0]
        assert reset_update.done.tolist() == [False]
        assert reset_update.status.tolist() == ["pending"]
        assert "initial_poses" in reset_update.details[0]
        assert reset_update.details[0]["action"] == "pose"
        assert reset_update.details[0]["action_index"] == 2
        assert (
            reset_update.details[0]["interval_selection"]["event"]
            == "interval_start_reached"
        )
        assert runner.records == []

        running_update = runner.update()
        assert running_update.done.tolist() == [False]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.3)

        final_update = runner.update()
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.4)
        assert final_update.stage_name == ["selected"]
        assert final_update.phase == ["post_move"]
        assert final_update.phase_step.tolist() == [1]
        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert final_update.status.tolist() == ["succeeded"]
        assert (
            final_update.details[0]["interval_selection"]["event"]
            == "interval_stop_reached"
        )
    finally:
        runner.close()


def test_stop_before_finishes_without_executing_the_target_keypoint() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                interval={
                    "start": _keypoint("selected", "pre_move", 0),
                    "stop": _keypoint("selected", "pre_move", 1, side="before"),
                }
            )
        )
    )
    try:
        reset_update = runner.reset()
        assert reset_update.done.tolist() == [False]

        running_update = runner.update()
        assert running_update.done.tolist() == [False]
        final_update = runner.update()

        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.1)
        assert final_update.stage_name == ["selected"]
        assert final_update.phase == ["pre_move"]
        assert final_update.phase_step.tolist() == [1]
        assert final_update.details[0]["interval_selection"]["stop"]["side"] == (
            "before"
        )
        assert "action" not in final_update.details[0]
        assert "action_index" not in final_update.details[0]
    finally:
        runner.close()


def test_same_keypoint_before_to_after_executes_exactly_that_keypoint() -> None:
    point = _keypoint("selected", "pre_move", 0)
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(interval={"start": point, "stop": point})
        )
    )
    try:
        reset_update = runner.reset()
        assert reset_update.done.tolist() == [False]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.2)

        running_update = runner.update()
        assert running_update.done.tolist() == [False]
        final_update = runner.update()

        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.1)
        assert final_update.phase == ["pre_move"]
        assert final_update.phase_step.tolist() == [0]
    finally:
        runner.close()


def test_same_keypoint_before_to_before_finishes_at_reset_state() -> None:
    point = _keypoint("selected", "pre_move", 0, side="before")
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(interval={"start": point, "stop": point})
        )
    )
    try:
        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.2)
        assert update.phase == ["pre_move"]
        assert update.phase_step.tolist() == [0]
        assert update.details[0]["interval_selection"]["fast_forward_updates"] == 0
        assert update.details[0]["execution"]["event"] == "interval_stop_reached"
    finally:
        runner.close()


def test_adjacent_after_to_before_finishes_at_the_shared_state() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                interval={
                    "start": _keypoint("selected", "pre_move", 0, side="after"),
                    "stop": _keypoint("selected", "pre_move", 1, side="before"),
                }
            )
        )
    )
    try:
        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.1)
        assert update.phase == ["pre_move"]
        assert update.phase_step.tolist() == [1]
        assert update.details[0]["interval_selection"]["fast_forward_updates"] == 2
        assert "action" not in update.details[0]
        assert runner.records == []
    finally:
        runner.close()


def test_interval_crosses_stage_boundaries_without_entering_stage_after_stop() -> None:
    payload = _task_payload()
    payload["task"]["stages"].append(
        {
            "name": "never_run",
            "object": "block",
            "operation": "move",
            "operator": "arm",
            "param": {"pre_move": [_pose(0.8)]},
        }
    )
    payload["execution"] = {
        "interval_selection": {
            "start": _keypoint("selected", "post_move", 1, side="after"),
            "stop": _keypoint("excluded", "pre_move", 0),
        }
    }
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        reset_update = runner.reset()
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.4)
        assert reset_update.stage_name == ["selected"]
        assert reset_update.phase == ["post_move"]
        assert reset_update.phase_step.tolist() == [1]

        running_update = runner.update()
        assert running_update.done.tolist() == [False]
        final_update = runner.update()

        assert _operator_positions(runner)[0, 0] == pytest.approx(0.9)
        assert final_update.stage_name == ["excluded"]
        assert final_update.phase == ["pre_move"]
        assert final_update.phase_step.tolist() == [0]
        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert [record.stage_name for record in runner.records] == ["excluded"]
    finally:
        runner.close()


def test_same_keypoint_after_to_after_finishes_during_reset() -> None:
    point = _keypoint("selected", "post_move", 0, side="after")
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(interval={"start": point, "stop": point})
        )
    )
    try:
        reset_update = runner.reset()

        assert _operator_positions(runner)[0, 0] == pytest.approx(0.3)
        assert reset_update.done.tolist() == [True]
        assert reset_update.success.tolist() == [True]
        assert reset_update.details[0]["execution"]["event"] == (
            "interval_stop_reached"
        )
        assert reset_update.phase == ["post_move"]
        assert reset_update.phase_step.tolist() == [0]
        assert runner.records == []

        repeated_update = runner.update()
        assert repeated_update.done.tolist() == [True]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.3)
    finally:
        runner.close()


def test_example_loop_does_not_step_an_interval_completed_by_reset() -> None:
    point = _keypoint("selected", "post_move", 0, side="after")
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(interval={"start": point, "stop": point})
        )
    )
    step_calls = 0

    def step_fn(_step: int, _update):
        nonlocal step_calls
        step_calls += 1
        return runner.update()

    try:
        summaries = run_example_rounds(
            rounds=1,
            use_input=False,
            hooks=ExampleLoopHooks(
                reset_fn=runner.reset,
                step_fn=step_fn,
                summarize_fn=lambda update, steps, maximum, elapsed: runner.summarize(
                    update,
                    updates_used=steps,
                    max_updates=maximum,
                    elapsed_time_sec=elapsed,
                ),
                records_fn=lambda: runner.records,
                max_updates=10,
            ),
        )

        assert step_calls == 0
        assert summaries[0].updates_used == 0
        assert summaries[0].env_completion_steps.tolist() == [0]
        assert summaries[0].env_completion_time_sec.tolist() == [0.0]
        assert summaries[0].env_completion_sim_time_sec.tolist() == [0.0]
    finally:
        runner.close()


def test_eef_is_selectable_as_singleton_waypoint_zero() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        {
            "name": "push",
            "object": "block",
            "operation": "push",
            "operator": "arm",
            "param": {
                "pre_move": [_pose(0.1)],
                "eef": {"close": True},
                "post_move": [_pose(0.2)],
            },
        }
    ]
    point = _keypoint("push", "eef", 0, side="after")
    payload["execution"] = {"interval_selection": {"start": point, "stop": point}}
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.phase == ["eef"]
        assert update.phase_step.tolist() == [0]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.1)
    finally:
        runner.close()


def test_eef_before_to_after_executes_the_eef_keypoint() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        {
            "name": "push",
            "object": "block",
            "operation": "push",
            "operator": "arm",
            "param": {
                "pre_move": [_pose(0.1)],
                "eef": {"close": True},
                "post_move": [_pose(0.2)],
            },
        }
    ]
    point = _keypoint("push", "eef", 0)
    payload["execution"] = {"interval_selection": {"start": point, "stop": point}}
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        reset_update = runner.reset()

        assert reset_update.done.tolist() == [False]
        assert reset_update.phase == ["eef"]
        assert reset_update.phase_step.tolist() == [0]
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.1)

        running_update = runner.update()
        assert running_update.done.tolist() == [False]
        final_update = runner.update()

        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert final_update.phase == ["eef"]
        assert final_update.phase_step.tolist() == [0]
    finally:
        runner.close()


def test_fast_forward_condition_failure_is_not_overwritten_as_success() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        {
            "name": "pick",
            "object": "block",
            "operation": "pick",
            "operator": "arm",
            "param": {
                "pre_move": [_pose(0.1)],
                "post_move": [_pose(0.2)],
            },
        }
    ]
    payload["execution"] = {
        "interval_selection": {
            "start": _keypoint("pick", "eef", 0, side="after"),
            "stop": _keypoint("pick", "post_move", 0),
        }
    }
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.status.tolist() == ["failed"]
        assert update.details[0]["failure_category"] == "missing_grasp"
        assert (
            update.details[0]["interval_selection"]["event"]
            == "interval_fast_forward_failed"
        )
        assert len(runner.records) == 1
        assert runner.records[0].status == "failed"
    finally:
        runner.close()


def test_configurable_fast_forward_limit_fails_instead_of_hanging_reset() -> None:
    point = _keypoint("selected", "post_move", 0, side="after")
    interval = {
        "start": point,
        "stop": point,
        "max_fast_forward_updates": 1,
    }
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload(interval=interval))
    )
    try:
        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == (
            "interval_fast_forward_timeout"
        )
        assert update.details[0]["fast_forward_updates"] == 1
        assert update.details[0]["interval_selection"]["max_fast_forward_updates"] == 1
        assert len(runner.records) == 1
    finally:
        runner.close()


def test_failed_fast_forward_does_not_expose_successful_prefix_stage_records() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        _move_stage("prefix", 0.1),
        _move_stage("selected", 0.2),
    ]
    payload["execution"] = {
        "interval_selection": {
            "start": _keypoint("selected", "pre_move", 0, side="after"),
            "stop": _keypoint("selected", "pre_move", 0),
            "max_fast_forward_updates": 3,
        }
    }
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        update = runner.reset()
        summary = runner.summarize(update)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.details[0]["failure_category"] == (
            "interval_fast_forward_timeout"
        )
        assert [record.stage_name for record in runner.records] == ["selected"]
        assert [record.status for record in runner.records] == ["failed"]
        assert summary.completed_stage_count.tolist() == [0]
    finally:
        runner.close()


def test_stop_condition_failure_is_not_overwritten_as_interval_success() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        {
            "name": "pick",
            "object": "block",
            "operation": "pick",
            "operator": "arm",
            "param": {
                "pre_move": [_pose(0.1)],
                "post_move": [_pose(0.2)],
            },
        }
    ]
    payload["execution"] = {
        "interval_selection": {
            "start": _keypoint("pick", "pre_move", 0, side="after"),
            "stop": _keypoint("pick", "eef", 0),
        }
    }
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        reset_update = runner.reset()
        assert reset_update.done.tolist() == [False]

        running_update = runner.update()
        assert running_update.done.tolist() == [False]
        assert running_update.phase == ["eef"]
        assert running_update.phase_step.tolist() == [0]
        assert (
            running_update.details[0]["interval_selection"]["event"]
            == "interval_running"
        )

        failed_update = runner.update()
        assert failed_update.done.tolist() == [True]
        assert failed_update.success.tolist() == [False]
        assert failed_update.status.tolist() == ["failed"]
        assert failed_update.details[0]["failure_category"] == "missing_grasp"
        assert (
            failed_update.details[0]["interval_selection"]["event"] == "interval_failed"
        )
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("interval", "message"),
    [
        (
            {
                "start": _keypoint("missing", "pre_move", 0),
                "stop": _keypoint("selected", "post_move", 0),
            },
            "does not match a task stage",
        ),
        (
            {
                "start": _keypoint("selected", "eef", 0),
                "stop": _keypoint("selected", "post_move", 0),
            },
            "does not execute that phase",
        ),
        (
            {
                "start": _keypoint("selected", "pre_move", 0),
                "stop": _keypoint("selected", "post_move", 2),
            },
            "is out of range",
        ),
        (
            {
                "start": _keypoint("selected", "post_move", 1),
                "stop": _keypoint("selected", "pre_move", 0),
            },
            "start must not come after",
        ),
        (
            {
                "start": _keypoint("selected", "pre_move", 0, side="after"),
                "stop": _keypoint("selected", "pre_move", 0, side="before"),
            },
            "start must not come after",
        ),
        (
            {
                "start": _keypoint("selected", "pre_move", 0, side="during"),
                "stop": _keypoint("selected", "post_move", 0),
            },
            "before|after",
        ),
    ],
)
def test_invalid_interval_points_are_rejected(interval: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        TaskFileConfig.model_validate(_task_payload(interval=interval))


def test_duplicate_selected_stage_name_is_rejected() -> None:
    payload = _task_payload(
        interval={
            "start": _keypoint("selected", "pre_move", 0),
            "stop": _keypoint("selected", "post_move", 0),
        }
    )
    duplicate = deepcopy(payload["task"]["stages"][0])
    payload["task"]["stages"].append(duplicate)

    with pytest.raises(ValidationError, match="ambiguous"):
        TaskFileConfig.model_validate(payload)


def test_policy_evaluator_rejects_task_runner_interval_selection() -> None:
    point = _keypoint("selected", "post_move", 0)
    config = TaskFileConfig.model_validate(
        _task_payload(interval={"start": point, "stop": point})
    )
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="TaskRunner/aao-demo only"):
        evaluator.from_config(config)


def test_active_builder_must_emit_configured_interval_keypoints() -> None:
    class NoPostMoveBuilder(TaskFlowBuilder):
        def build_actions(self, stage, last_orientation=None):
            actions, orientation = super().build_actions(stage, last_orientation)
            return [
                action for action in actions if action.phase != TaskPhase.POST_MOVE
            ], orientation

    interval = {
        "start": _keypoint("selected", "post_move", 0, side="after"),
        "stop": _keypoint("selected", "post_move", 1),
    }
    config = TaskFileConfig.model_validate(_task_payload(interval=interval))
    runner = TaskRunner(builder=NoPostMoveBuilder())
    try:
        with pytest.raises(ValueError, match="is not emitted by NoPostMoveBuilder"):
            runner.from_config(config)
    finally:
        runner.close()


def test_interval_rejects_invalid_builder_keypoint_completion_marker() -> None:
    class NeverCompletesKeypointBuilder(TaskFlowBuilder):
        def build_actions(self, stage, last_orientation=None):
            actions, orientation = super().build_actions(stage, last_orientation)
            for action in actions:
                action.completes_keypoint = False
            return actions, orientation

    interval = {
        "start": _keypoint("selected", "pre_move", 0),
        "stop": _keypoint("selected", "post_move", 1),
    }
    config = TaskFileConfig.model_validate(_task_payload(interval=interval))
    runner = TaskRunner(builder=NeverCompletesKeypointBuilder())
    try:
        with pytest.raises(ValueError, match="mark only the final primitive"):
            runner.from_config(config)
    finally:
        runner.close()


def test_interval_fast_forward_respects_partial_reset_mask() -> None:
    interval = {
        "start": _keypoint("selected", "post_move", 0, side="after"),
        "stop": _keypoint("selected", "post_move", 1),
    }
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload(batch_size=2, interval=interval))
    )
    try:
        first_update = runner.reset(np.asarray([True, False], dtype=bool))
        assert _operator_positions(runner)[:, 0].tolist() == pytest.approx([0.3, 0.2])
        assert (
            first_update.details[1]["interval_selection"]["event"] == "interval_pending"
        )

        with pytest.raises(RuntimeError, match="have not been reset"):
            runner.update(np.asarray([False, True], dtype=bool))

        second_update = runner.reset(np.asarray([False, True], dtype=bool))
        assert _operator_positions(runner)[:, 0].tolist() == pytest.approx([0.3, 0.3])
        assert second_update.details[0]["execution"]["event"] == (
            "interval_start_reached"
        )
    finally:
        runner.close()


def test_arc_subactions_keep_the_configured_waypoint_identity() -> None:
    stage = StageConfig.model_validate(
        {
            "name": "arc_move",
            "object": "block",
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
    )

    actions, _ = TaskFlowBuilder.build_actions(stage)

    assert len(actions) == 3
    assert [action.phase for action in actions] == [TaskPhase.PRE_MOVE] * 3
    assert [action.waypoint for action in actions] == [0, 0, 0]
    assert [action.completes_keypoint for action in actions] == [False, False, True]


def test_arc_before_to_after_executes_all_primitives_as_one_keypoint() -> None:
    payload = _task_payload()
    payload["task"]["stages"] = [
        {
            "name": "arc_move",
            "object": "block",
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
    point = _keypoint("arc_move", "pre_move", 0)
    payload["execution"] = {"interval_selection": {"start": point, "stop": point}}
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(payload))
    try:
        update = runner.reset()
        assert update.done.tolist() == [False]
        assert update.details[0]["interval_selection"]["fast_forward_updates"] == 0

        updates = 0
        while not bool(update.done[0]):
            update = runner.update()
            updates += 1

        assert updates == 6
        assert update.success.tolist() == [True]
        assert update.phase == ["pre_move"]
        assert update.phase_step.tolist() == [0]
    finally:
        runner.close()


def test_primitive_action_preserves_existing_positional_constructor_order() -> None:
    pose = PoseControlConfig(position=(0.1, 0.0, 0.3))
    resolved_pose = PoseControlConfig(position=(0.2, 0.0, 0.3))
    arc_snapshot = ArcExecutionSnapshot()

    action = PrimitiveAction(
        "pose",
        pose,
        None,
        resolved_pose,
        arc_snapshot,
        0.25,
    )

    assert action.resolved_pose is resolved_pose
    assert action.arc_snapshot is arc_snapshot
    assert action.arc_cumulative_angle == pytest.approx(0.25)
    assert action.phase is None
    assert action.waypoint is None
