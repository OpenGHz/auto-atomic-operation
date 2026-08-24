from __future__ import annotations

import numpy as np
import pytest

from auto_atom.execution_timeline import ExecutionTimeline
from auto_atom.framework import TaskFileConfig
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.runtime import ComponentRegistry, TaskFlowBuilder, TaskRunner


def _pose(x: float) -> dict:
    return {
        "reference": "world",
        "position": [x, 0.0, 0.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


def _config(env_name: str, *, execution: dict | None = None) -> TaskFileConfig:
    ComponentRegistry.register_env(env_name, {"kind": "mock_env", "batch_size": 2})
    payload = {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {
            "env_name": env_name,
            "stages": [
                {
                    "name": "first",
                    "object": "block",
                    "operation": "move",
                    "operator": "arm",
                    "param": {"pre_move": [_pose(0.1), _pose(0.2)]},
                },
                {
                    "name": "second",
                    "object": "block",
                    "operation": "move",
                    "operator": "arm",
                    "param": {"pre_move": [_pose(0.3)]},
                },
            ],
        },
        "task_operators": {"arm": {}},
    }
    if execution is not None:
        payload["execution"] = execution
    return TaskFileConfig.model_validate(payload)


class _CountingBuilder(TaskFlowBuilder):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def build_actions(self, stage, last_orientation=None):
        self.calls.append(stage.name or "")
        return super().build_actions(stage, last_orientation)


class _LegacyFilteringBuilder(_CountingBuilder):
    """Exercise the pre-timeline ``build`` extension seam."""

    def build(self, context):
        return super().build(context)[:1]


class _SparseLegacyBuilder(_CountingBuilder):
    def build(self, context):
        plans = super().build(context)
        plans[0].stage_index = 10
        return plans


def test_task_runner_compiles_each_stage_once_and_reuses_templates() -> None:
    ComponentRegistry.clear()
    builder = _CountingBuilder()
    runner = TaskRunner(builder=builder).from_config(
        _config(
            "timeline_counting",
            execution={"update_boundary": "keypoint"},
        )
    )
    try:
        assert builder.calls == ["first", "second"]
        assert isinstance(runner._timeline, ExecutionTimeline)

        runner.reset()
        runner.update()
        runner.update()
        runner.reset(np.asarray([True, False], dtype=bool))

        assert builder.calls == ["first", "second"]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_legacy_build_override_shapes_timeline_without_duplicate_action_builds() -> (
    None
):
    ComponentRegistry.clear()
    builder = _LegacyFilteringBuilder()
    runner = TaskRunner(builder=builder).from_config(_config("timeline_legacy_build"))
    try:
        assert [plan.stage_name for plan in runner._plan] == ["first"]
        assert builder.calls == ["first", "second"]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_legacy_build_override_must_preserve_stage_index_ordinals() -> None:
    ComponentRegistry.clear()
    runner = TaskRunner(builder=_SparseLegacyBuilder())
    try:
        with pytest.raises(ValueError, match="preserve contiguous stage indices"):
            runner.from_config(_config("timeline_sparse_legacy_build"))
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_timeline_clone_isolates_runtime_state_but_preserves_arc_snapshot_alias() -> (
    None
):
    ComponentRegistry.clear()
    ComponentRegistry.register_env(
        "timeline_arc",
        {"kind": "mock_env", "batch_size": 1},
    )
    config = TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {
                "env_name": "timeline_arc",
                "stages": [
                    {
                        "name": "arc",
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
                ],
            },
            "task_operators": {"arm": {}},
            "execution": {"update_boundary": "keypoint"},
        }
    )
    runner = TaskRunner().from_config(config)
    try:
        timeline = runner._timeline
        assert timeline is not None
        first = timeline.clone_stage_actions(0)
        second = timeline.clone_stage_actions(0)

        assert len(first) == 3
        assert first[0].arc_snapshot is first[1].arc_snapshot
        assert first[0].arc_snapshot is not second[0].arc_snapshot
        first[0].resolved_pose = first[0].pose
        assert second[0].resolved_pose is None
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_config_driven_policy_reuses_evaluator_timeline() -> None:
    ComponentRegistry.clear()
    builder = _CountingBuilder()
    policy = ConfigDrivenDemoPolicy(builder=builder)
    evaluator = PolicyEvaluator(
        action_applier=policy.action_applier,
        builder=builder,
    ).from_config(_config("timeline_policy"))
    try:
        assert builder.calls == ["first", "second"]
        update = evaluator.reset()
        for _ in range(8):
            if bool(update.done.all()):
                break
            update = evaluator.update(policy.act({}, update, evaluator))
        assert bool(update.done.all())
        assert builder.calls == ["first", "second"]
    finally:
        evaluator.close()
        ComponentRegistry.clear()


def test_default_config_driven_policy_uses_evaluator_timeline() -> None:
    ComponentRegistry.clear()
    evaluator_builder = _CountingBuilder()
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(
        action_applier=policy.action_applier,
        builder=evaluator_builder,
    ).from_config(_config("timeline_default_policy"))
    try:
        update = evaluator.reset()
        action = policy.act({}, update, evaluator)
        assert action.env_actions[0] is not None
        assert evaluator_builder.calls == ["first", "second"]
    finally:
        evaluator.close()
        ComponentRegistry.clear()


def test_timeline_boundary_lookup_distinguishes_order_and_state_indices() -> None:
    ComponentRegistry.clear()
    config = _config(
        "timeline_boundaries",
        execution={
            "interval_selection": {
                "start": {
                    "stage": "first",
                    "phase": "pre_move",
                    "waypoint": 0,
                    "side": "after",
                },
                "stop": {
                    "stage": "first",
                    "phase": "pre_move",
                    "waypoint": 1,
                    "side": "before",
                },
            }
        },
    )
    runner = TaskRunner().from_config(config)
    try:
        timeline = runner._timeline
        assert timeline is not None
        start = config.execution.interval_selection.start
        stop = config.execution.interval_selection.stop
        assert timeline.boundary_order_index(start) < timeline.boundary_order_index(
            stop
        )
        assert timeline.boundary_state_index(start) == timeline.boundary_state_index(
            stop
        )
    finally:
        runner.close()
        ComponentRegistry.clear()
