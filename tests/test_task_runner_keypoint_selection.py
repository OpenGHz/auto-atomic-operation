from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from auto_atom.config.execution import TaskPhase
from auto_atom.config.task import TaskFileConfig
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runtime import ComponentRegistry, TaskFlowBuilder, TaskRunner


def _pose(x: float) -> dict:
    return {
        "reference": "world",
        "position": [x, 0.0, 0.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


def _move_stage(
    name: str,
    *positions: float,
    post_move: Tuple[float, ...] = (),
) -> dict:
    param: dict = {"pre_move": [_pose(position) for position in positions]}
    if post_move:
        param["post_move"] = [_pose(position) for position in post_move]
    return {
        "name": name,
        "object": "block",
        "operation": "move",
        "operator": "arm",
        "param": param,
    }


def _object_only_pick(name: str, object_name: str) -> dict:
    return {
        "name": name,
        "object": object_name,
        "operation": "pick",
        "operator": "arm",
        "param": {
            "pre_move": [_pose(0.0)],
            "post_move": [
                {
                    "reference": "world",
                    "position": [0.0, 0.0, 0.4],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                }
            ],
            "eef": {"close": True},
        },
    }


def _object_only_place(name: str, target: str) -> dict:
    return {
        "name": name,
        "object": target,
        "operation": "place",
        "operator": "arm",
        "param": {
            "pre_move": [
                {
                    "reference": "world",
                    "position": [0.52, 0.0, 0.3],
                    "orientation": [0.0, 0.0, 1.0, 0.0],
                    "controlled_frame": {"kind": "held_object"},
                }
            ],
            "post_move": [
                {
                    "reference": "world",
                    "position": [0.52, 0.0, 0.5],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                }
            ],
            "eef": {"close": False},
            "placed_tolerance": {"position": 0.01, "orientation": 0.1},
        },
    }


def _task_payload(
    *,
    batch_size: int = 1,
    selection: list | None = None,
    stages: List[dict] | None = None,
    mode: str | None = None,
) -> dict:
    env_name = f"keypoint_selection_mock_{batch_size}"
    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": batch_size},
    )
    payload = {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {
            "env_name": env_name,
            "stages": (
                stages
                if stages is not None
                else [
                    _move_stage("first", 0.11),
                    _move_stage("second", 0.55),
                    _move_stage("third", 0.91),
                ]
            ),
        },
        "task_operators": {"arm": {}},
    }
    execution: dict = {}
    if mode is not None:
        execution["mode"] = mode
    if selection is not None:
        execution["keypoint_selection"] = selection
    if execution:
        payload["execution"] = execution
    return payload


def _operator_positions(runner: TaskRunner) -> np.ndarray:
    assert runner._context is not None
    return (
        runner._context.backend.get_operator_handler("arm")
        .get_end_effector_pose()
        .position
    )


def _drive(runner: TaskRunner, limit: int = 200) -> Tuple[Any, List[float]]:
    """Reset, then run public updates until every environment is terminal."""
    update = runner.reset()
    visited = [float(_operator_positions(runner)[0, 0])]
    for _ in range(limit):
        if bool(np.all(update.done)):
            break
        update = runner.update()
        visited.append(float(_operator_positions(runner)[0, 0]))
    return update, visited


@pytest.fixture(autouse=True)
def _clear_component_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_selection_absent_still_runs_the_complete_task() -> None:
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(_task_payload()))
    try:
        update, visited = _drive(runner)

        assert update.success.tolist() == [True]
        assert 0.11 in visited
        assert 0.55 in visited
        assert visited[-1] == pytest.approx(0.91)
        assert [record.stage_name for record in runner.records] == [
            "first",
            "second",
            "third",
        ]
    finally:
        runner.close()


def test_stage_selection_executes_only_selected_stages_in_order() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(selection=[{"stage": "first"}, {"stage": "third"}])
        )
    )
    try:
        update, visited = _drive(runner)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert visited[-1] == pytest.approx(0.91)
        assert all(abs(position - 0.55) > 1e-9 for position in visited)
        assert [record.stage_name for record in runner.records] == ["first", "third"]
        assert [plan.stage_name for plan in runner._plan] == ["first", "third"]
    finally:
        runner.close()


def test_selection_starts_from_the_untouched_reset_state() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload(selection=[{"stage": "third"}]))
    )
    try:
        reset_update = runner.reset()

        # No prefix is fast-forwarded: the mid-task selection begins from the
        # backend reset state and performs no keypoint work during reset().
        assert _operator_positions(runner)[0, 0] == pytest.approx(0.2)
        assert runner.records == []
        assert reset_update.done.tolist() == [False]
        assert reset_update.stage_name == ["third"]
        assert (
            reset_update.details[0]["keypoint_selection"]["event"]
            == "keypoint_selection_running"
        )

        final_update, visited = _drive(runner)
        assert final_update.success.tolist() == [True]
        assert all(abs(position - 0.11) > 1e-9 for position in visited)
        assert all(abs(position - 0.55) > 1e-9 for position in visited)
        assert visited[-1] == pytest.approx(0.91)
    finally:
        runner.close()


def test_phase_selection_keeps_only_that_phase() -> None:
    stages = [_move_stage("only", 0.11, 0.22, post_move=(0.77, 0.88))]
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                stages=stages,
                selection=[{"stage": "only", "phase": "post_move"}],
            )
        )
    )
    try:
        first_update = runner.reset()
        assert first_update.phase == [None]

        first_update = runner.update()
        assert first_update.phase == ["post_move"]
        assert first_update.phase_step.tolist() == [0]

        final_update, visited = _drive(runner)
        assert final_update.success.tolist() == [True]
        assert all(abs(position - 0.11) > 1e-9 for position in visited)
        assert all(abs(position - 0.22) > 1e-9 for position in visited)
        assert visited[-1] == pytest.approx(0.88)
    finally:
        runner.close()


def test_waypoint_selection_keeps_a_single_keypoint() -> None:
    stages = [_move_stage("only", 0.11, 0.22, post_move=(0.77, 0.88))]
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                stages=stages,
                selection=[
                    {"stage": "only", "phase": "pre_move", "waypoint": 1},
                ],
            )
        )
    )
    try:
        final_update, visited = _drive(runner)

        assert final_update.success.tolist() == [True]
        assert visited[-1] == pytest.approx(0.22)
        assert all(abs(position - 0.11) > 1e-9 for position in visited)
        assert all(abs(position - 0.77) > 1e-9 for position in visited)
    finally:
        runner.close()


def test_ordinal_selection_keeps_the_first_and_last_keypoint() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload(selection=[0, -1]))
    )
    try:
        update, visited = _drive(runner)

        assert update.success.tolist() == [True]
        assert visited[-1] == pytest.approx(0.91)
        assert all(abs(position - 0.55) > 1e-9 for position in visited)
        assert [record.stage_name for record in runner.records] == ["first", "third"]
    finally:
        runner.close()


def test_scoped_negative_index_addresses_the_end_of_the_scope() -> None:
    stages = [_move_stage("only", 0.11, 0.22, post_move=(0.77, 0.88))]
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                stages=stages,
                selection=[{"stage": "only", "waypoint": -1}],
            )
        )
    )
    try:
        final_update, visited = _drive(runner)

        # The scope is the whole stage: its last keypoint is post_move[1].
        assert final_update.success.tolist() == [True]
        assert visited[-1] == pytest.approx(0.88)
        assert all(abs(position - 0.11) > 1e-9 for position in visited)
        assert all(abs(position - 0.77) > 1e-9 for position in visited)
    finally:
        runner.close()


def test_phase_relative_negative_index_addresses_the_phase_end() -> None:
    stages = [_move_stage("only", 0.11, 0.22, post_move=(0.77, 0.88))]
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                stages=stages,
                selection=[{"stage": "only", "phase": "pre_move", "waypoint": -1}],
            )
        )
    )
    try:
        final_update, visited = _drive(runner)

        assert final_update.success.tolist() == [True]
        assert visited[-1] == pytest.approx(0.22)
        assert all(abs(position - 0.11) > 1e-9 for position in visited)
        assert all(abs(position - 0.88) > 1e-9 for position in visited)
    finally:
        runner.close()


def test_ordinal_and_scoped_entries_can_be_mixed() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                selection=[0, {"stage": "third", "phase": "pre_move", "waypoint": 0}]
            )
        )
    )
    try:
        update, visited = _drive(runner)

        assert update.success.tolist() == [True]
        assert visited[-1] == pytest.approx(0.91)
        assert [record.stage_name for record in runner.records] == ["first", "third"]
    finally:
        runner.close()


def test_selection_applies_to_every_environment() -> None:
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(batch_size=2, selection=[{"stage": "third"}])
        )
    )
    try:
        update, _ = _drive(runner)

        assert update.success.tolist() == [True, True]
        assert _operator_positions(runner)[:, 0].tolist() == pytest.approx([0.91, 0.91])
        assert [record.stage_name for record in runner.records] == ["third", "third"]
    finally:
        runner.close()


def test_object_only_mode_keeps_only_the_selected_stages() -> None:
    stages = [
        _object_only_pick("pick_a", "item_a"),
        _object_only_place("place_a", "target_a"),
        _object_only_pick("pick_b", "item_b"),
        _object_only_place("place_b", "target_b"),
    ]
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(
            _task_payload(
                stages=stages,
                selection=[{"stage": "pick_b"}, {"stage": "place_b"}],
                mode="object_only",
            )
        )
    )
    try:
        update = runner.reset()
        for _ in range(100):
            if bool(np.all(update.done)):
                break
            update = runner.update()

        assert update.success.tolist() == [True]
        assert [plan.stage_name for plan in runner._plan] == ["pick_b", "place_b"]
        assert [record.stage_name for record in runner.records] == [
            "pick_b",
            "place_b",
        ]
    finally:
        runner.close()


def test_selection_details_report_configured_entries() -> None:
    selection = [
        0,
        {"stage": "third", "phase": "pre_move", "waypoint": 0},
    ]
    runner = TaskRunner().from_config(
        TaskFileConfig.model_validate(_task_payload(selection=selection))
    )
    try:
        update, _ = _drive(runner)

        details = update.details[0]["keypoint_selection"]
        assert details["event"] == "keypoint_selection_succeeded"
        assert details["keypoints"] == [
            0,
            {"stage": "third", "phase": "pre_move", "waypoint": 0},
        ]
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ([{"stage": "third"}, {"stage": "first"}], "after the previous entry"),
        ([{"stage": "first"}, {"stage": "first"}], "after the previous entry"),
        ([{"stage": "missing"}], "does not match a task stage"),
        ([{"stage": "first", "side": "before"}], "has no side"),
        ([{"stage": "first", "waypoint": 3}], "out of range"),
        ([{"stage": "first", "waypoint": -2}], "out of range"),
        ([3], "out of range"),
        ([-4], "out of range"),
        ([-1, 0], "after the previous entry"),
        ([0, 0], "after the previous entry"),
        ([1, 0], "after the previous entry"),
        ([{"stage": "first", "phase": "eef"}], "does not execute that phase"),
        (
            [{"stage": "first", "phase": "pre_move", "waypoint": 4}],
            "out of range",
        ),
    ],
)
def test_invalid_selections_are_rejected(
    selection: list,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        TaskFileConfig.model_validate(_task_payload(selection=selection))


def test_phase_selection_order_is_validated_against_the_task_order() -> None:
    stages = [_move_stage("only", 0.11, post_move=(0.77,))]
    with pytest.raises(ValidationError, match="after the previous entry"):
        TaskFileConfig.model_validate(
            _task_payload(
                stages=stages,
                selection=[
                    {"stage": "only", "phase": "post_move"},
                    {"stage": "only", "phase": "pre_move"},
                ],
            )
        )


def test_empty_selection_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one keypoint"):
        TaskFileConfig.model_validate(_task_payload(selection=[]))


def test_selection_conflicts_with_interval_selection() -> None:
    payload = _task_payload(selection=[{"stage": "first"}])
    payload["execution"]["interval_selection"] = {
        "start": {"stage": "first", "phase": "pre_move", "waypoint": 0},
        "stop": {"stage": "third", "phase": "pre_move", "waypoint": 0},
    }

    with pytest.raises(
        ValidationError,
        match="mutually exclusive",
    ):
        TaskFileConfig.model_validate(payload)


def test_top_level_keypoint_selection_is_rejected_with_migration_path() -> None:
    payload = _task_payload()
    payload["keypoint_selection"] = [{"stage": "first"}]

    with pytest.raises(
        ValidationError,
        match="use execution.keypoint_selection instead",
    ):
        TaskFileConfig.model_validate(payload)


def test_policy_evaluator_rejects_keypoint_selection() -> None:
    config = TaskFileConfig.model_validate(
        _task_payload(selection=[{"stage": "first"}])
    )
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="TaskRunner/aao-demo only"):
        evaluator.from_config(config)


def test_active_builder_must_emit_selected_keypoints() -> None:
    class NoPostMoveBuilder(TaskFlowBuilder):
        def build_actions(self, stage, last_orientation=None):
            actions, orientation = super().build_actions(stage, last_orientation)
            return [
                action for action in actions if action.phase != TaskPhase.POST_MOVE
            ], orientation

    stages = [_move_stage("only", 0.11, post_move=(0.77,))]
    config = TaskFileConfig.model_validate(
        _task_payload(
            stages=stages,
            selection=[{"stage": "only", "phase": "post_move"}],
        )
    )
    runner = TaskRunner(builder=NoPostMoveBuilder())
    try:
        with pytest.raises(ValueError, match="is not emitted by NoPostMoveBuilder"):
            runner.from_config(config)
    finally:
        runner.close()
