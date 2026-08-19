from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from auto_atom.framework import TaskFileConfig
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runner.common import ExampleLoopHooks, run_example_rounds
from auto_atom.runtime import (
    ComponentRegistry,
    ControlResult,
    ControlSignal,
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


def _task_payload(
    *,
    env_name: str,
    stages: list[dict],
    batch_size: int = 1,
    update_boundary: str | None = None,
    max_internal_updates: int | None = None,
    interval_selection: dict | None = None,
) -> dict:
    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": batch_size},
    )
    payload = {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {"env_name": env_name, "stages": stages},
        "task_operators": {"arm": {}},
    }
    execution = {}
    if update_boundary is not None:
        execution["update_boundary"] = update_boundary
    if max_internal_updates is not None:
        execution["max_internal_updates_per_update"] = max_internal_updates
    if interval_selection is not None:
        execution["interval_selection"] = interval_selection
    if execution:
        payload["execution"] = execution
    return payload


def _runner(payload: dict) -> TaskRunner:
    return TaskRunner().from_config(TaskFileConfig.model_validate(payload))


def _x_positions(runner: TaskRunner) -> list[float]:
    assert runner._context is not None
    position = (
        runner._context.backend.get_operator_handler("arm")
        .get_end_effector_pose()
        .position
    )
    return position[:, 0].tolist()


@pytest.fixture(autouse=True)
def _clear_component_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_default_update_boundary_matches_explicit_control_tick() -> None:
    default_runner = _runner(
        _task_payload(
            env_name="boundary_default",
            stages=[_move_stage("move", 0.1, 0.4)],
        )
    )
    explicit_runner = _runner(
        _task_payload(
            env_name="boundary_explicit_tick",
            stages=[_move_stage("move", 0.1, 0.4)],
            update_boundary="control_tick",
        )
    )
    try:
        default_runner.reset()
        explicit_runner.reset()

        for _ in range(4):
            default_update = default_runner.update()
            explicit_update = explicit_runner.update()

            assert default_update.status.tolist() == explicit_update.status.tolist()
            assert default_update.done.tolist() == explicit_update.done.tolist()
            assert default_update.success.tolist() == explicit_update.success.tolist()
            assert default_update.phase == explicit_update.phase
            assert default_update.phase_step.tolist() == (
                explicit_update.phase_step.tolist()
            )
            assert _x_positions(default_runner) == pytest.approx(
                _x_positions(explicit_runner)
            )
            assert (
                default_update.details[0]["execution"]
                == (explicit_update.details[0]["execution"])
            )
            assert default_update.details[0]["execution"]["internal_updates"] == 1
            assert (
                default_update.details[0]["execution"]["update_boundary"]
                == "control_tick"
            )
    finally:
        default_runner.close()
        explicit_runner.close()


@pytest.mark.parametrize(
    ("field_name", "target"),
    [
        ("update_boundary", "execution.update_boundary"),
        (
            "max_internal_updates_per_update",
            "execution.max_internal_updates_per_update",
        ),
        ("render_internal_updates", "execution.render_internal_updates"),
        (
            "max_fast_forward_updates",
            "execution.interval_selection.max_fast_forward_updates",
        ),
    ],
)
def test_top_level_execution_fields_are_rejected_with_migration_path(
    field_name: str,
    target: str,
) -> None:
    payload = _task_payload(
        env_name=f"boundary_top_level_{field_name}",
        stages=[_move_stage("move", 0.1)],
    )
    if field_name == "update_boundary":
        payload[field_name] = "keypoint"
    elif field_name == "render_internal_updates":
        payload[field_name] = False
    else:
        payload[field_name] = 1

    with pytest.raises(ValidationError, match=target):
        TaskFileConfig.model_validate(payload)


def test_top_level_execution_field_is_rejected_from_mapping_config() -> None:
    payload = _task_payload(
        env_name="boundary_top_level_mapping",
        stages=[_move_stage("move", 0.1)],
    )
    payload["update_boundary"] = "keypoint"

    with pytest.raises(ValidationError, match="execution.update_boundary"):
        TaskFileConfig.model_validate(OmegaConf.create(payload))


def test_primitive_boundary_completes_exactly_one_primitive_per_update() -> None:
    runner = _runner(
        _task_payload(
            env_name="boundary_primitive",
            stages=[_move_stage("move", 0.1, 0.4)],
            update_boundary="primitive",
        )
    )
    try:
        runner.reset()

        first_update = runner.update()
        state = runner._env_states[0]
        assert state.active is not None
        assert state.active.action_index == 1
        assert _x_positions(runner) == pytest.approx([0.1])
        assert first_update.done.tolist() == [False]
        assert first_update.details[0]["execution"]["event"] == "primitive_reached"
        assert first_update.details[0]["execution"]["internal_updates"] == 2

        final_update = runner.update()
        assert _x_positions(runner) == pytest.approx([0.4])
        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("boundary", "expected_action_index", "expected_updates", "expected_event"),
    [
        ("primitive", 1, 2, "primitive_reached"),
        ("keypoint", 3, 6, "keypoint_reached"),
    ],
)
def test_arc_is_three_primitive_boundaries_but_one_keypoint_boundary(
    boundary: str,
    expected_action_index: int,
    expected_updates: int,
    expected_event: str,
) -> None:
    arc_waypoint = {
        "reference": "world",
        "arc": {
            "pivot": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "angle": 0.5,
            "max_step": 0.2,
        },
    }
    stage = _move_stage("arc_move", 0.8)
    stage["param"]["pre_move"].insert(0, arc_waypoint)
    runner = _runner(
        _task_payload(
            env_name=f"boundary_arc_{boundary}",
            stages=[stage],
            update_boundary=boundary,
        )
    )
    try:
        runner.reset()
        update = runner.update()

        state = runner._env_states[0]
        assert state.active is not None
        assert len(state.active.actions) == 4
        assert [action.waypoint for action in state.active.actions[:3]] == [0, 0, 0]
        assert [action.completes_keypoint for action in state.active.actions[:3]] == [
            False,
            False,
            True,
        ]
        assert state.active.action_index == expected_action_index
        assert update.done.tolist() == [False]
        assert update.details[0]["execution"]["event"] == expected_event
        assert update.details[0]["execution"]["internal_updates"] == expected_updates
    finally:
        runner.close()


def test_stage_boundary_does_not_enter_the_next_stage() -> None:
    runner = _runner(
        _task_payload(
            env_name="boundary_stage",
            stages=[
                _move_stage("first", 0.1, 0.4),
                _move_stage("second", 0.9),
            ],
            update_boundary="stage",
        )
    )
    try:
        runner.reset()

        first_update = runner.update()
        assert _x_positions(runner) == pytest.approx([0.4])
        assert first_update.done.tolist() == [False]
        assert [record.stage_name for record in runner.records] == ["first"]
        assert runner._env_states[0].stage_cursor == 1
        assert runner._env_states[0].active is None
        assert first_update.details[0]["execution"]["event"] == "stage_succeeded"
        assert first_update.details[0]["execution"]["internal_updates"] == 4

        final_update = runner.update()
        assert _x_positions(runner) == pytest.approx([0.9])
        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert [record.stage_name for record in runner.records] == ["first", "second"]
    finally:
        runner.close()


def test_batch_waits_for_each_envs_first_boundary_without_advancing_fast_env() -> None:
    runner = _runner(
        _task_payload(
            env_name="boundary_batch",
            stages=[_move_stage("move", 0.1, 0.4)],
            batch_size=2,
            update_boundary="primitive",
        )
    )
    call_counts = np.zeros(2, dtype=np.int64)
    required_calls = np.asarray([1, 4], dtype=np.int64)
    try:
        assert runner._context is not None
        handler = runner._context.backend.get_operator_handler("arm")

        def uneven_move_to_pose(self, pose, target, env_mask=None):
            del target
            mask = (
                np.ones(self.batch_size, dtype=bool)
                if env_mask is None
                else np.asarray(env_mask, dtype=bool).reshape(-1)
            )
            signals = np.asarray(
                [ControlSignal.RUNNING] * self.batch_size,
                dtype=object,
            )
            details = [{} for _ in range(self.batch_size)]
            for env_index in np.flatnonzero(mask):
                index = int(env_index)
                call_counts[index] += 1
                details[index] = {"event": "moving", "operator": self.name}
                if call_counts[index] < required_calls[index]:
                    continue
                self.end_effector_pose.position[index] = np.asarray(
                    pose.position,
                    dtype=np.float64,
                )
                self.end_effector_pose.orientation[index] = np.asarray(
                    pose.orientation,
                    dtype=np.float64,
                )
                signals[index] = ControlSignal.REACHED
                details[index]["event"] = "pose_reached"
            return ControlResult(signals=signals, details=details)

        handler.move_to_pose = MethodType(uneven_move_to_pose, handler)
        runner.reset()
        update = runner.update()

        assert call_counts.tolist() == [1, 4]
        assert _x_positions(runner) == pytest.approx([0.1, 0.1])
        assert [state.active.action_index for state in runner._env_states] == [1, 1]
        assert update.done.tolist() == [False, False]
        assert [
            details["execution"]["internal_updates"] for details in update.details
        ] == [
            1,
            4,
        ]
    finally:
        runner.close()


def test_internal_update_cap_is_an_explicit_terminal_failure() -> None:
    runner = _runner(
        _task_payload(
            env_name="boundary_cap",
            stages=[_move_stage("move", 0.1)],
            update_boundary="primitive",
            max_internal_updates=1,
        )
    )
    try:
        runner.reset()
        update = runner.update()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.status.tolist() == ["failed"]
        assert update.details[0]["failure_stage"] == "execution"
        assert update.details[0]["failure_category"] == "internal_update_limit_exceeded"
        assert update.details[0]["execution"] == {
            "event": "internal_update_limit_exceeded",
            "update_boundary": "primitive",
            "render_internal_updates": True,
            "internal_updates": 1,
            "max_internal_updates_per_update": 1,
        }
        assert len(runner.records) == 1
        assert runner.records[0].status == "failed"
    finally:
        runner.close()


def test_keypoint_boundary_rejects_builder_without_keypoint_metadata() -> None:
    class LegacyBuilder(TaskFlowBuilder):
        def build_actions(self, stage, last_orientation=None):
            actions, orientation = super().build_actions(stage, last_orientation)
            for action in actions:
                action.phase = None
                action.waypoint = None
            return actions, orientation

    config = TaskFileConfig.model_validate(
        _task_payload(
            env_name="boundary_legacy_builder",
            stages=[_move_stage("move", 0.1, 0.4)],
            update_boundary="keypoint",
        )
    )
    runner = TaskRunner(builder=LegacyBuilder())
    try:
        with pytest.raises(
            ValueError,
            match="requires every action emitted by LegacyBuilder",
        ):
            runner.from_config(config)
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("phase_type", "TaskPhase phase and integer waypoint"),
        ("waypoint_range", "emitted invalid keypoint"),
        ("completion_marker", "mark only the final primitive"),
    ],
)
def test_keypoint_boundary_rejects_invalid_builder_metadata(
    defect: str,
    message: str,
) -> None:
    class InvalidMetadataBuilder(TaskFlowBuilder):
        def build_actions(self, stage, last_orientation=None):
            actions, orientation = super().build_actions(stage, last_orientation)
            if defect == "phase_type":
                actions[0].phase = "pre_move"
            elif defect == "waypoint_range":
                actions[0].waypoint = 99
            else:
                actions[0].completes_keypoint = False
            return actions, orientation

    config = TaskFileConfig.model_validate(
        _task_payload(
            env_name=f"boundary_invalid_builder_{defect}",
            stages=[_move_stage("move", 0.1)],
            update_boundary="keypoint",
        )
    )
    runner = TaskRunner(builder=InvalidMetadataBuilder())
    try:
        with pytest.raises(ValueError, match=message):
            runner.from_config(config)
    finally:
        runner.close()


@pytest.mark.parametrize("boundary", ["control_tick", "primitive", "keypoint", "stage"])
def test_empty_task_finishes_without_exhausting_macro_update_limit(
    boundary: str,
) -> None:
    runner = _runner(
        _task_payload(
            env_name=f"boundary_empty_{boundary}",
            stages=[],
            update_boundary=boundary,
            max_internal_updates=1,
        )
    )
    try:
        runner.reset()
        update = runner.update()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["execution"]["event"] == "task_succeeded"
        assert update.details[0]["execution"]["internal_updates"] == 0
    finally:
        runner.close()


def test_macro_summary_uses_each_envs_internal_update_count(monkeypatch) -> None:
    runner = _runner(
        _task_payload(
            env_name="boundary_summary_batch",
            stages=[_move_stage("move", 0.4)],
            batch_size=2,
            update_boundary="primitive",
        )
    )
    call_counts = np.zeros(2, dtype=np.int64)
    required_calls = np.asarray([1, 4], dtype=np.int64)
    try:
        assert runner._context is not None
        backend = runner._context.backend
        handler = backend.get_operator_handler("arm")

        def uneven_move_to_pose(self, pose, target, env_mask=None):
            del target
            mask = (
                np.ones(self.batch_size, dtype=bool)
                if env_mask is None
                else np.asarray(env_mask, dtype=bool).reshape(-1)
            )
            signals = np.asarray(
                [ControlSignal.RUNNING] * self.batch_size,
                dtype=object,
            )
            details = [{} for _ in range(self.batch_size)]
            for env_index in np.flatnonzero(mask):
                index = int(env_index)
                call_counts[index] += 1
                if call_counts[index] < required_calls[index]:
                    continue
                self.end_effector_pose.position[index] = np.asarray(
                    pose.position,
                    dtype=np.float64,
                )
                signals[index] = ControlSignal.REACHED
            return ControlResult(signals=signals, details=details)

        handler.move_to_pose = MethodType(uneven_move_to_pose, handler)
        monkeypatch.setattr(
            type(backend),
            "dt_per_update",
            property(lambda _backend: 0.25),
        )

        summaries = run_example_rounds(
            rounds=1,
            use_input=False,
            hooks=ExampleLoopHooks(
                reset_fn=runner.reset,
                step_fn=lambda _step, _update: runner.update(),
                summarize_fn=(
                    lambda update, steps, maximum, elapsed: runner.summarize(
                        update,
                        updates_used=steps,
                        max_updates=maximum,
                        elapsed_time_sec=elapsed,
                    )
                ),
                records_fn=lambda: runner.records,
                max_updates=2,
            ),
        )

        summary = summaries[0]
        assert call_counts.tolist() == [1, 4]
        assert summary.updates_used == 1
        assert summary.sim_time_sec == pytest.approx(1.0)
        assert summary.env_completion_sim_time_sec.tolist() == pytest.approx(
            [0.25, 1.0]
        )
    finally:
        runner.close()


def test_precondition_failure_does_not_count_a_simulated_control_tick(
    monkeypatch,
) -> None:
    place_stage = {
        "name": "place_without_grasp",
        "object": "block",
        "operation": "place",
        "operator": "arm",
        "param": {"pre_move": [_pose(0.4)]},
    }
    runner = _runner(
        _task_payload(
            env_name="boundary_precondition_failure",
            stages=[place_stage],
            update_boundary="stage",
        )
    )
    try:
        assert runner._context is not None
        monkeypatch.setattr(
            type(runner._context.backend),
            "dt_per_update",
            property(lambda _backend: 0.25),
        )

        runner.reset()
        update = runner.update()
        summary = runner.summarize(update, updates_used=1)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [False]
        assert update.details[0]["execution"]["internal_updates"] == 0
        assert summary.sim_time_sec == 0.0
        assert summary.env_completion_sim_time_sec.tolist() == [0.0]
    finally:
        runner.close()


def test_interval_stop_preempts_stage_boundary_and_truncates_stage() -> None:
    interval = {
        "start": {"stage": "selected", "phase": "pre_move", "waypoint": 0},
        "stop": {"stage": "selected", "phase": "pre_move", "waypoint": 1},
    }
    runner = _runner(
        _task_payload(
            env_name="boundary_interval_stop",
            stages=[_move_stage("selected", 0.1, 0.4, 0.9)],
            update_boundary="stage",
            interval_selection=interval,
        )
    )
    try:
        reset_update = runner.reset()
        assert _x_positions(runner) == pytest.approx([0.1])
        assert reset_update.done.tolist() == [False]

        final_update = runner.update()
        assert _x_positions(runner) == pytest.approx([0.4])
        assert final_update.done.tolist() == [True]
        assert final_update.success.tolist() == [True]
        assert runner.records == []
        assert (
            final_update.details[0]["interval_selection"]["event"]
            == "interval_stop_reached"
        )
        assert final_update.details[0]["execution"]["event"] == "interval_stop_reached"
        assert final_update.details[0]["execution"]["internal_updates"] == 2
    finally:
        runner.close()


@pytest.mark.parametrize("boundary", ["primitive", "keypoint", "stage"])
def test_policy_evaluator_rejects_non_control_tick_update_boundaries(
    boundary: str,
) -> None:
    config = TaskFileConfig.model_validate(
        _task_payload(
            env_name=f"policy_boundary_{boundary}",
            stages=[_move_stage("move", 0.1)],
            update_boundary=boundary,
        )
    )
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)

    with pytest.raises(
        ValueError,
        match=(
            "PolicyEvaluator only supports execution.update_boundary='control_tick'"
        ),
    ):
        evaluator.from_config(config)
