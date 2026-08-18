from pathlib import Path
import sys
from types import MethodType

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom import StartAfterWaypointConfig as PublicStartAfterWaypointConfig
from auto_atom import StopAtConfig as PublicStopAtConfig
from auto_atom.framework import (
    AutoAtomConfig,
    PhysicalReplayConfig,
    PhysicalReplayPresentationConfig,
    StartAfterWaypointConfig,
    StopAtConfig,
    TaskFileConfig,
)
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.runtime import (
    ComponentRegistry,
    ControlResult,
    ControlSignal,
    PoseState,
    TaskRunner,
)


def _move_stage(name: str, positions: list[float]) -> dict:
    return {
        "name": name,
        "object": "",
        "operation": "move",
        "operator": "arm",
        "param": {
            "pre_move": [
                {
                    "position": [position, 0.0, 0.2],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "reference": "world",
                }
                for position in positions
            ]
        },
    }


def _linear_then_arc_stage(name: str) -> dict:
    return {
        "name": name,
        "object": "",
        "operation": "move",
        "operator": "arm",
        "param": {
            "pre_move": [
                {
                    "position": [0.1, 0.0, 0.2],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "reference": "world",
                },
                {
                    "arc": {
                        "pivot": [0.0, 0.0, 0.0],
                        "axis": [0.0, 0.0, 1.0],
                        "angle": 0.2,
                        "max_step": 0.05,
                    },
                    "reference": "world",
                },
            ]
        },
    }


def _mock_task(
    physical_replay: dict | None,
    *,
    stages: list[dict] | None = None,
    batch_size: int = 1,
    start_after: dict | None = None,
    stop_at: dict | None = None,
) -> TaskFileConfig:
    ComponentRegistry.clear()
    ComponentRegistry.register_env(
        "physical_replay_mock",
        {"kind": "mock_env", "batch_size": batch_size},
    )
    task = {
        "env_name": "physical_replay_mock",
        "stages": stages or [_move_stage("move_0", [0.1])],
    }
    if physical_replay is not None:
        task["physical_replay"] = physical_replay
    if start_after is not None:
        task["start_after"] = start_after
    if stop_at is not None:
        task["stop_at"] = stop_at
    return TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": task,
            "task_operators": {"arm": {}},
        }
    )


def _spy_on_operator(runner: TaskRunner, *, reject_teleport: bool = False) -> list:
    operator = runner._context.backend.get_operator_handler("arm")
    move_calls = []
    original_move = operator.move_to_pose

    def _move(self, pose, target, env_mask=None):
        move_calls.append(np.asarray(env_mask, dtype=bool).copy())
        return original_move(pose, target, env_mask=env_mask)

    operator.move_to_pose = MethodType(_move, operator)
    if reject_teleport:

        def _teleport(self, pose, target=None, env_mask=None):
            raise AssertionError("physical replay must not teleport")

        operator.teleport_end_effector = MethodType(_teleport, operator)
    return move_calls


def _spy_on_replay_presentation(runner: TaskRunner) -> tuple[list[bool], list[float]]:
    backend = runner._context.backend
    animation_calls: list[bool] = []
    keyframe_calls: list[float] = []

    def _set_animation(self, enabled):
        animation_calls.append(bool(enabled))

    def _present_keyframe(self, hold_seconds):
        keyframe_calls.append(float(hold_seconds))

    backend.set_physical_replay_animation = MethodType(
        _set_animation, backend
    )
    backend.present_physical_replay_keyframe = MethodType(
        _present_keyframe, backend
    )
    return animation_calls, keyframe_calls


def _base_auto_atom_payload() -> dict:
    return {
        "env_name": "validation",
        "stages": [_move_stage("move_0", [0.1])],
    }


def test_start_after_waypoint_type_schema_and_dump_are_unchanged() -> None:
    selector = StartAfterWaypointConfig(
        stage="move_0",
        phase="pre_move",
        waypoint=0,
    )

    assert PublicStartAfterWaypointConfig is StartAfterWaypointConfig
    assert type(selector) is StartAfterWaypointConfig
    assert selector.__class__.__name__ == "StartAfterWaypointConfig"
    assert selector.__class__.__module__ == "auto_atom.framework"
    assert tuple(selector.__class__.model_fields) == (
        "stage",
        "phase",
        "waypoint",
    )
    assert all(
        field.is_required()
        for field in selector.__class__.model_fields.values()
    )
    assert selector.model_dump() == {
        "stage": "move_0",
        "phase": "pre_move",
        "waypoint": 0,
    }
    assert selector.model_dump(mode="json") == {
        "stage": "move_0",
        "phase": "pre_move",
        "waypoint": 0,
    }
    assert (
        selector.model_dump_json()
        == '{"stage":"move_0","phase":"pre_move","waypoint":0}'
    )
    assert StartAfterWaypointConfig.model_json_schema() == {
        "additionalProperties": False,
        "description": (
            "Reset-time fast-forward target expressed in YAML waypoint coordinates."
        ),
        "properties": {
            "stage": {"title": "Stage", "type": "string"},
            "phase": {
                "enum": ["pre_move", "post_move"],
                "title": "Phase",
                "type": "string",
            },
            "waypoint": {
                "minimum": 0,
                "title": "Waypoint",
                "type": "integer",
            },
        },
        "required": ["stage", "phase", "waypoint"],
        "title": "StartAfterWaypointConfig",
        "type": "object",
    }

    base = _base_auto_atom_payload()
    legacy = AutoAtomConfig.model_validate(
        {
            **base,
            "start_after": {
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
            },
        }
    )
    assert type(legacy.start_after) is StartAfterWaypointConfig
    assert legacy.start_after.model_dump() == selector.model_dump()
    assert legacy.physical_replay is None

    schema = AutoAtomConfig.model_json_schema()
    assert (
        schema["properties"]["start_after"]["anyOf"][0]["$ref"]
        == "#/$defs/StartAfterWaypointConfig"
    )
    assert "StartAfterConfig" not in schema["$defs"]


@pytest.mark.parametrize(
    ("raw_waypoint", "expected"),
    [(True, 1), (1.0, 1), ("2", 2)],
)
def test_start_after_waypoint_preserves_legacy_integer_coercion(
    raw_waypoint: object, expected: int
) -> None:
    selector = StartAfterWaypointConfig.model_validate(
        {
            "stage": "move_0",
            "phase": "pre_move",
            "waypoint": raw_waypoint,
        }
    )
    assert selector.waypoint == expected


def test_physical_replay_schema_accepts_absolute_and_waypoint_targets() -> None:
    base = _base_auto_atom_payload()

    absolute = AutoAtomConfig.model_validate(
        {**base, "physical_replay": {"frame": 0}}
    )
    assert type(absolute.physical_replay) is PhysicalReplayConfig
    assert absolute.physical_replay.frame == 0
    assert absolute.start_after is None

    waypoint = AutoAtomConfig.model_validate(
        {
            **base,
            "physical_replay": {
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "frame_offset": 3,
            },
        }
    )
    assert waypoint.physical_replay.frame_offset == 3


def test_physical_replay_presentation_schema_and_defaults() -> None:
    selector = PhysicalReplayConfig(frame=0)
    assert type(selector.presentation) is PhysicalReplayPresentationConfig
    assert selector.presentation.mode == "waypoint"
    assert selector.presentation.preserve_arcs is False
    assert selector.presentation.keyframe_hold_seconds == pytest.approx(0.05)
    assert selector.model_dump(exclude_none=True, exclude_defaults=True) == {
        "frame": 0
    }

    configured = PhysicalReplayConfig.model_validate(
        {
            "frame": 1,
            "presentation": {
                "mode": "waypoint",
                "preserve_arcs": True,
                "keyframe_hold_seconds": 0.2,
            },
        }
    )
    assert configured.presentation.preserve_arcs is True
    assert configured.presentation.keyframe_hold_seconds == pytest.approx(0.2)


@pytest.mark.parametrize(
    "presentation, expected",
    [
        ({"mode": "keyframe"}, "Input should be"),
        ({"keyframe_hold_seconds": -0.1}, "greater than or equal to 0"),
        ({"animate_lines": True}, "Extra inputs are not permitted"),
    ],
)
def test_physical_replay_presentation_schema_rejects_invalid_values(
    presentation: dict, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        PhysicalReplayConfig.model_validate(
            {"frame": 0, "presentation": presentation}
        )


@pytest.mark.parametrize(
    "selector, expected",
    [
        ({}, "requires either frame"),
        ({"frame": 1, "stage": "move_0"}, "not both"),
        ({"frame": 1, "frame_offset": 0}, "only valid with a waypoint"),
        (
            {
                "stage": "move_0",
                "phase": "pre_move",
            },
            "requires either frame",
        ),
        ({"frame": -1}, "greater than or equal to 0"),
        ({"frame": True}, "valid integer"),
        ({"mode": "physical", "frame": 1}, "Extra inputs are not permitted"),
    ],
)
def test_physical_replay_schema_rejects_ambiguous_targets(
    selector: dict, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        AutoAtomConfig.model_validate(
            {**_base_auto_atom_payload(), "physical_replay": selector}
        )


def test_stop_at_schema_accepts_absolute_and_waypoint_targets() -> None:
    base = _base_auto_atom_payload()

    absolute = AutoAtomConfig.model_validate({**base, "stop_at": {"frame": 7}})
    assert PublicStopAtConfig is StopAtConfig
    assert type(absolute.stop_at) is StopAtConfig
    assert absolute.stop_at.frame == 7

    waypoint = AutoAtomConfig.model_validate(
        {
            **base,
            "stop_at": {
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "frame_offset": 3,
            },
        }
    )
    assert waypoint.stop_at.frame_offset == 3


@pytest.mark.parametrize(
    "selector, expected",
    [
        ({}, "requires either frame"),
        ({"frame": 1, "stage": "move_0"}, "not both"),
        ({"frame": 1, "frame_offset": 0}, "only valid with a waypoint"),
        (
            {"stage": "move_0", "phase": "pre_move"},
            "requires either frame",
        ),
        ({"frame": -1}, "greater than or equal to 0"),
        ({"frame": True}, "valid integer"),
    ],
)
def test_stop_at_schema_rejects_ambiguous_targets(
    selector: dict, expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        AutoAtomConfig.model_validate(
            {**_base_auto_atom_payload(), "stop_at": selector}
        )


def test_start_after_requires_a_later_waypoint_stop_target() -> None:
    stages = [_move_stage("move_0", [0.1, 0.2])]
    with pytest.raises(ValueError, match="stop_at waypoint must be after"):
        _mock_task(
            None,
            stages=stages,
            start_after={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
            },
            stop_at={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
            },
        )

    config = _mock_task(
        None,
        stages=stages,
        start_after={
            "stage": "move_0",
            "phase": "pre_move",
            "waypoint": 0,
        },
        stop_at={
            "stage": "move_0",
            "phase": "pre_move",
            "waypoint": 1,
        },
    )
    assert config.task.stop_at.waypoint == 1


def test_start_after_and_physical_replay_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="start_after.*physical_replay.*mutually"):
        AutoAtomConfig.model_validate(
            {
                **_base_auto_atom_payload(),
                "start_after": {
                    "stage": "move_0",
                    "phase": "pre_move",
                    "waypoint": 0,
                },
                "physical_replay": {"frame": 0},
            }
        )


def test_place_only_examples_compose_to_independent_replay_fields() -> None:
    config_dir = str((ROOT / "aao_configs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        teleport_raw = OmegaConf.to_container(
            compose(config_name="pick_and_place_place_only"),
            resolve=True,
        )
        physical_raw = OmegaConf.to_container(
            compose(
                config_name="pick_and_place_place_only_physical_replay"
            ),
            resolve=True,
        )
        segment_raw = OmegaConf.to_container(
            compose(config_name="pick_and_place_physical_segment"),
            resolve=True,
        )
        block_segment_raw = OmegaConf.to_container(
            compose(
                config_name="place_blocks_on_disk_airbot_play_g2_segment"
            ),
            resolve=True,
        )

    teleport = TaskFileConfig.model_validate(teleport_raw)
    physical = TaskFileConfig.model_validate(physical_raw)
    segment = TaskFileConfig.model_validate(segment_raw)
    block_segment = TaskFileConfig.model_validate(block_segment_raw)

    assert type(teleport.task.start_after) is StartAfterWaypointConfig
    assert teleport.task.physical_replay is None
    assert physical.task.start_after is None
    assert type(physical.task.physical_replay) is PhysicalReplayConfig
    assert [stage.name for stage in physical.task.stages] == [
        "pick_source",
        "place_source",
    ]
    assert type(segment.task.physical_replay) is PhysicalReplayConfig
    assert type(segment.task.stop_at) is StopAtConfig
    assert segment.task.stop_at.model_dump(exclude_defaults=True) == {
        "stage": "place_source",
        "phase": "pre_move",
        "waypoint": 1,
    }
    assert block_segment.task.physical_replay.model_dump(
        exclude_defaults=True
    ) == {
        "stage": "pick_cube_yellow_2",
        "phase": "pre_move",
        "waypoint": 1,
    }
    assert block_segment.task.physical_replay.presentation.mode == "waypoint"
    assert block_segment.task.physical_replay.presentation.preserve_arcs is False
    assert block_segment.task.stop_at.model_dump(exclude_defaults=True) == {
        "stage": "place_cube_orange_3_in_disk",
        "phase": "post_move",
        "waypoint": 0,
    }


def test_start_after_schema_rejects_physical_replay_fields_as_extras() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        StartAfterWaypointConfig.model_validate(
            {
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "mode": "physical",
            }
        )


def test_absolute_frame_zero_does_not_execute_control() -> None:
    runner = TaskRunner().from_config(_mock_task({"frame": 0}))
    try:
        move_calls = _spy_on_operator(runner, reject_teleport=True)
        update = runner.reset()

        assert move_calls == []
        assert update.stage_name == ["move_0"]
        assert update.status.tolist() == ["pending"]
        assert "fast_forward" not in update.details[0]
        details = update.details[0]["physical_replay"]
        assert details["target"] == {"frame": 0}
        assert details["replayed_frames"] == 0
        assert runner.records == []
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_waypoint_presentation_fast_forwards_linear_motion_between_keyframes() -> None:
    selector = {
        "stage": "move_0",
        "phase": "pre_move",
        "waypoint": 1,
        "presentation": {
            "mode": "waypoint",
            "keyframe_hold_seconds": 0.0,
        },
    }
    runner = TaskRunner().from_config(
        _mock_task(selector, stages=[_move_stage("move_0", [0.1, 0.2])])
    )
    try:
        animation_calls, keyframe_calls = _spy_on_replay_presentation(runner)
        runner.reset()

        # Two mock controller ticks per linear waypoint. The reset-time prefix
        # before the selected start waypoint is hidden; that start keyframe is
        # presented once when reset returns.
        assert animation_calls == [False]
        assert keyframe_calls == [0.0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_full_presentation_hides_prefix_and_animates_every_visible_tick() -> None:
    runner = TaskRunner().from_config(
        _mock_task(
            {
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "presentation": {
                    "mode": "full",
                    "keyframe_hold_seconds": 0.0,
                },
            },
            stages=[_move_stage("move_0", [0.1, 0.2])],
            stop_at={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 1,
            },
        )
    )
    try:
        animation_calls, keyframe_calls = _spy_on_replay_presentation(runner)
        runner.reset()
        runner.update()
        update = runner.update()

        assert update.done.tolist() == [True]
        assert animation_calls == [False, True, True]
        assert keyframe_calls == [0.0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_waypoint_presentation_continues_through_visible_suffix_until_stop_at() -> None:
    runner = TaskRunner().from_config(
        _mock_task(
            {
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "presentation": {
                    "mode": "waypoint",
                    "keyframe_hold_seconds": 0.0,
                },
            },
            stages=[_move_stage("move_0", [0.1, 0.2])],
            stop_at={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 1,
            },
        )
    )
    try:
        animation_calls, keyframe_calls = _spy_on_replay_presentation(runner)
        runner.reset()

        # The first waypoint is the visible start. The second waypoint is
        # presented by normal rollout before stop_at marks the task complete.
        assert animation_calls == [False]
        assert keyframe_calls == [0.0]

        update = runner.update()
        assert update.done.tolist() == [False]
        update = runner.update()
        assert update.done.tolist() == [True]
        assert animation_calls == [False, False, False]
        assert keyframe_calls == [0.0, 0.0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_physical_replay_can_fast_forward_all_the_way_to_stop_at() -> None:
    endpoint = {
        "stage": "move_0",
        "phase": "pre_move",
        "waypoint": 1,
    }
    runner = TaskRunner().from_config(
        _mock_task(
            {
                **endpoint,
                "presentation": {
                    "mode": "waypoint",
                    "keyframe_hold_seconds": 0.0,
                },
            },
            stages=[_move_stage("move_0", [0.1, 0.2])],
            stop_at=endpoint,
        )
    )
    try:
        animation_calls, keyframe_calls = _spy_on_replay_presentation(runner)
        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["stop_at"]["task_frame"] == 4
        # The prefix is hidden and the endpoint has priority as the final
        # visible keyframe, even though the task ends during reset.
        assert animation_calls == [False]
        assert keyframe_calls == [0.0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_waypoint_presentation_can_preserve_arc_animation() -> None:
    stage = _linear_then_arc_stage("arc_move")
    runner = TaskRunner().from_config(
        _mock_task(
            {
                "stage": "arc_move",
                "phase": "pre_move",
                "waypoint": 0,
                "presentation": {
                    "mode": "waypoint",
                    "preserve_arcs": True,
                    "keyframe_hold_seconds": 0.0,
                },
            },
            stages=[stage],
            stop_at={
                "stage": "arc_move",
                "phase": "pre_move",
                "waypoint": 1,
            },
        )
    )
    try:
        animation_calls, keyframe_calls = _spy_on_replay_presentation(runner)
        runner.reset()
        for _ in range(8):
            update = runner.update()

        # The prefix before waypoint 0 is hidden. The visible arc has four
        # sub-primitives and two mock controller ticks per primitive.
        assert update.done.tolist() == [True]
        assert animation_calls == [False] + [True] * 8
        assert keyframe_calls == [0.0, 0.0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_waypoint_presentation_can_skip_arc_animation() -> None:
    stage = _linear_then_arc_stage("arc_move")
    runner = TaskRunner().from_config(
        _mock_task(
            {
                "stage": "arc_move",
                "phase": "pre_move",
                "waypoint": 0,
                "presentation": {
                    "mode": "waypoint",
                    "preserve_arcs": False,
                    "keyframe_hold_seconds": 0.0,
                },
            },
            stages=[stage],
            stop_at={
                "stage": "arc_move",
                "phase": "pre_move",
                "waypoint": 1,
            },
        )
    )
    try:
        animation_calls, keyframe_calls = _spy_on_replay_presentation(runner)
        runner.reset()
        for _ in range(8):
            update = runner.update()

        # The complete four-part arc still takes eight visible-window controller
        # ticks; all are hidden and only the reached endpoint is presented.
        assert update.done.tolist() == [True]
        assert animation_calls == [False] * 9
        assert keyframe_calls == [0.0, 0.0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_stop_at_frame_zero_finishes_successfully_without_control() -> None:
    runner = TaskRunner().from_config(
        _mock_task(None, stop_at={"frame": 0})
    )
    try:
        move_calls = _spy_on_operator(runner)
        update = runner.reset()

        assert move_calls == []
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.status.tolist() == ["succeeded"]
        assert update.details[0]["stop_at"] == {
            "event": "stop_at_reached",
            "target": {"frame": 0},
            "task_frame": 0,
            "task_sim_time_sec": 0.0,
            "stage": "move_0",
        }
        assert runner.records == []
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_physical_replay_and_stop_at_share_absolute_task_frames() -> None:
    stages = [_move_stage("move_0", [0.1, 0.2])]
    runner = TaskRunner().from_config(
        _mock_task(
            {"frame": 1},
            stages=stages,
            stop_at={"frame": 3},
        )
    )
    try:
        move_calls = _spy_on_operator(runner, reject_teleport=True)
        reset_update = runner.reset()

        assert len(move_calls) == 1
        assert reset_update.done.tolist() == [False]
        assert runner._env_states[0].task_frame == 1

        update = runner.update()
        assert update.done.tolist() == [False]
        update = runner.update()

        assert len(move_calls) == 3
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["stop_at"]["task_frame"] == 3
        assert runner._env_states[0].active.action_index == 1
        assert runner.records == []
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_stop_at_waypoint_offset_counts_after_reached_tick() -> None:
    stages = [_move_stage("move_0", [0.1, 0.2])]
    runner = TaskRunner().from_config(
        _mock_task(
            None,
            stages=stages,
            stop_at={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "frame_offset": 1,
            },
        )
    )
    try:
        move_calls = _spy_on_operator(runner)
        runner.reset()
        runner.update()
        reached_update = runner.update()
        assert reached_update.done.tolist() == [False]
        assert runner._env_states[0].stop_waypoint_reached_frame == 2

        update = runner.update()

        assert len(move_calls) == 3
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["stop_at"]["waypoint_reached_frame"] == 2
        assert update.details[0]["stop_at"]["task_frame"] == 3
        assert runner._env_states[0].active.action_index == 1
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_stop_at_at_final_waypoint_skips_remaining_stage_completion() -> None:
    runner = TaskRunner().from_config(
        _mock_task(
            None,
            stop_at={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
            },
        )
    )
    try:
        runner.reset()
        runner.update()
        update = runner.update()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.stage_name == ["move_0"]
        assert update.details[0]["stop_at"]["task_frame"] == 2
        assert runner.records == []
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_stop_at_before_physical_replay_target_is_rejected() -> None:
    runner = TaskRunner().from_config(
        _mock_task(
            {"frame": 2},
            stages=[_move_stage("move_0", [0.1, 0.2])],
            stop_at={"frame": 1},
        )
    )
    try:
        with pytest.raises(RuntimeError, match="resolved to task frame 1"):
            runner.reset()
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_stop_at_beyond_task_end_is_rejected() -> None:
    runner = TaskRunner().from_config(
        _mock_task(None, stop_at={"frame": 3})
    )
    try:
        runner.reset()
        runner.update()
        with pytest.raises(RuntimeError, match="end after 2 frame.*stop_at.*frame 3"):
            runner.update()
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_absolute_frame_stops_mid_action_and_resumes_without_replanning() -> None:
    runner = TaskRunner().from_config(_mock_task({"frame": 1}))
    try:
        move_calls = _spy_on_operator(runner, reject_teleport=True)
        update = runner.reset()
        active = runner._env_states[0].active

        assert len(move_calls) == 1
        assert active is not None
        assert active.action_index == 0
        assert update.phase == ["pre_move"]
        assert "fast_forward" not in update.details[0]
        assert update.details[0]["physical_replay"]["replayed_frames"] == 1
        assert runner.records == []

        update = runner.update()
        assert len(move_calls) == 2
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_absolute_frame_matches_normal_rollout_state() -> None:
    stages = [_move_stage("move_0", [0.1, 0.2])]
    normal = TaskRunner().from_config(_mock_task(None, stages=stages))
    try:
        normal_update = normal.reset()
        for _ in range(3):
            normal_update = normal.update()
        normal_state = normal._env_states[0]
        normal_pose = (
            normal._context.backend.get_operator_handler("arm")
            .get_end_effector_pose()
            .position.copy()
        )
        normal_action_index = normal_state.active.action_index
    finally:
        normal.close()
        ComponentRegistry.clear()

    replay = TaskRunner().from_config(
        _mock_task({"frame": 3}, stages=stages)
    )
    try:
        replay_update = replay.reset()
        replay_state = replay._env_states[0]
        replay_pose = (
            replay._context.backend.get_operator_handler("arm")
            .get_end_effector_pose()
            .position.copy()
        )

        np.testing.assert_allclose(replay_pose, normal_pose, atol=1e-12)
        assert replay_update.phase == normal_update.phase
        assert replay_state.stage_cursor == normal_state.stage_cursor
        assert replay_state.active.action_index == normal_action_index
    finally:
        replay.close()
        ComponentRegistry.clear()


def test_physical_replay_respects_reset_env_mask() -> None:
    runner = TaskRunner().from_config(
        _mock_task({"frame": 1}, batch_size=2)
    )
    try:
        move_calls = _spy_on_operator(runner, reject_teleport=True)
        runner.reset(np.asarray([True, False], dtype=bool))

        assert len(move_calls) == 1
        assert move_calls[0].tolist() == [True, False]
        assert [state.episode_index for state in runner._env_states] == [0, -1]

        runner.reset(np.asarray([False, True], dtype=bool))
        assert len(move_calls) == 2
        assert move_calls[1].tolist() == [False, True]
        assert [state.episode_index for state in runner._env_states] == [0, 0]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_empty_reset_mask_does_not_change_summary_start_cursor() -> None:
    stages = [_move_stage("move_0", [0.1]), _move_stage("move_1", [0.2])]
    runner = TaskRunner().from_config(
        _mock_task({"frame": 0}, stages=stages)
    )
    try:
        update = runner.reset()
        assert runner.summarize(update).total_stages == 2

        update = runner.update()
        update = runner.update()
        assert runner._env_states[0].stage_cursor == 1

        update = runner.reset(np.asarray([False], dtype=bool))
        assert runner.summarize(update).total_stages == 2
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_failed_replay_marks_selected_env_uninitialized() -> None:
    runner = TaskRunner().from_config(_mock_task({"frame": 1}))
    try:
        runner.reset()
        assert runner._has_reset.tolist() == [True]

        runner._context.config.physical_replay.frame = 3
        with pytest.raises(RuntimeError, match="end of the task"):
            runner.reset()

        assert runner._has_reset.tolist() == [False]
        with pytest.raises(RuntimeError, match="have not been reset"):
            runner.update()
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_last_action_completion_error_preserves_original_failure_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_completion_error(*, env_index, context, active):
        _ = env_index, context, active
        raise RuntimeError("completion condition exploded")

    monkeypatch.setattr(
        TaskRunner,
        "_stage_completion_failure",
        staticmethod(_raise_completion_error),
    )
    runner = TaskRunner().from_config(_mock_task({"frame": 2}))
    try:
        with pytest.raises(RuntimeError) as exc_info:
            runner.reset()

        message = str(exc_info.value)
        assert "completion condition exploded" in message
        assert "frame 2" in message
        assert "move_0.pre_move[0]" in message
        assert "IndexError" not in message
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_later_batch_replay_failure_invalidates_all_selected_without_commit() -> None:
    runner = TaskRunner().from_config(
        _mock_task({"frame": 1}, batch_size=2)
    )
    try:
        operator = runner._context.backend.get_operator_handler("arm")
        original_move = operator.move_to_pose
        fail_later_env = [False]

        def _move(self, pose, target, env_mask=None):
            selected = np.flatnonzero(np.asarray(env_mask, dtype=bool))
            assert selected.size == 1
            if fail_later_env[0] and int(selected[0]) == 1:
                raise RuntimeError("env 1 replay exploded")
            return original_move(pose, target, env_mask=env_mask)

        operator.move_to_pose = MethodType(_move, operator)
        runner.reset()
        assert runner._has_reset.tolist() == [True, True]
        previous_states = list(runner._env_states)
        assert [state.episode_index for state in previous_states] == [0, 0]

        fail_later_env[0] = True
        with pytest.raises(RuntimeError, match="env 1 replay exploded"):
            runner.reset()

        assert runner._has_reset.tolist() == [False, False]
        assert all(
            current is previous
            for current, previous in zip(runner._env_states, previous_states)
        )
        assert [state.episode_index for state in runner._env_states] == [0, 0]
        with pytest.raises(RuntimeError, match="have not been reset"):
            runner.update(np.asarray([True, False], dtype=bool))
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_waypoint_frame_offset_counts_after_reached_tick() -> None:
    selector = {
        "stage": "move_0",
        "phase": "pre_move",
        "waypoint": 0,
        "frame_offset": 1,
    }
    runner = TaskRunner().from_config(
        _mock_task(selector, stages=[_move_stage("move_0", [0.1, 0.2])])
    )
    try:
        move_calls = _spy_on_operator(runner, reject_teleport=True)
        update = runner.reset()
        active = runner._env_states[0].active
        assert "fast_forward" not in update.details[0]
        details = update.details[0]["physical_replay"]

        # Mock pose actions need two controller ticks. The waypoint completes
        # on frame 2; offset 1 executes the first tick of waypoint 1.
        assert len(move_calls) == 3
        assert details["waypoint_reached_frame"] == 2
        assert details["replayed_frames"] == 3
        assert details["resume_waypoint"] == 1
        assert active is not None
        assert active.action_index == 1
        assert runner.records == []

        update = runner.update()
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_waypoint_offset_can_cross_stage_without_recording_prefix() -> None:
    selector = {
        "stage": "move_0",
        "phase": "pre_move",
        "waypoint": 0,
        "frame_offset": 1,
    }
    stages = [_move_stage("move_0", [0.1]), _move_stage("move_1", [0.2])]
    runner = TaskRunner().from_config(_mock_task(selector, stages=stages))
    try:
        update = runner.reset()
        active = runner._env_states[0].active

        assert update.stage_name == ["move_1"]
        assert active is not None and active.plan.stage_name == "move_1"
        assert active.action_index == 0
        assert "fast_forward" not in update.details[0]
        assert update.details[0]["physical_replay"]["completed_stages"] == [
            "move_0"
        ]
        assert runner.records == []
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_arc_waypoint_anchor_waits_for_last_internal_primitive() -> None:
    selector = {
        "stage": "arc_move",
        "phase": "pre_move",
        "waypoint": 0,
    }
    stage = {
        "name": "arc_move",
        "object": "",
        "operation": "move",
        "operator": "arm",
        "param": {
            "pre_move": [
                {
                    "arc": {
                        "pivot": [0.0, 0.0, 0.0],
                        "axis": [0.0, 0.0, 1.0],
                        "angle": 0.2,
                        "max_step": 0.05,
                    },
                    "reference": "world",
                }
            ]
        },
    }
    runner = TaskRunner().from_config(_mock_task(selector, stages=[stage]))
    try:
        move_calls = _spy_on_operator(runner, reject_teleport=True)
        update = runner.reset()
        assert "fast_forward" not in update.details[0]
        details = update.details[0]["physical_replay"]

        # Four arc sub-primitives, two mock controller ticks each.
        assert len(move_calls) == 8
        assert details["waypoint_reached_frame"] == 8
        assert details["replayed_frames"] == 8
        assert update.done.tolist() == [True]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_absolute_arc_waypoint_waits_for_final_joint_angle() -> None:
    selector = {
        "stage": "absolute_arc",
        "phase": "pre_move",
        "waypoint": 0,
    }
    stage = {
        "name": "absolute_arc",
        "object": "",
        "operation": "move",
        "operator": "arm",
        "param": {
            "pre_move": [
                {
                    "arc": {
                        "pivot": "hinge",
                        "axis": [0.0, 0.0, 1.0],
                        "angle": 0.2,
                        "absolute": True,
                        "max_step": 0.05,
                        "joint_tolerance": 1e-6,
                    },
                    "reference": "world",
                }
            ]
        },
    }
    runner = TaskRunner().from_config(_mock_task(selector, stages=[stage]))
    try:
        backend = runner._context.backend
        operator = backend.get_operator_handler("arm")
        joint_angle = [0.0]

        def _get_joint_angle(name, env_index=0):
            assert name == "hinge" and env_index == 0
            return joint_angle[0]

        def _get_element_pose(name, env_index=0):
            assert name == "hinge" and env_index == 0
            return PoseState(position=[0.0, 0.0, 0.0])

        def _move(self, pose, target, env_mask=None):
            joint_angle[0] = min(0.2, joint_angle[0] + 0.05)
            self.end_effector_pose.position[0] = np.asarray(pose.position)
            self.end_effector_pose.orientation[0] = np.asarray(pose.orientation)
            return ControlResult.filled(1, ControlSignal.REACHED)

        backend.get_joint_angle = _get_joint_angle
        backend.get_element_pose = _get_element_pose
        operator.move_to_pose = MethodType(_move, operator)

        update = runner.reset()
        assert "fast_forward" not in update.details[0]
        details = update.details[0]["physical_replay"]

        assert joint_angle[0] == pytest.approx(0.2)
        assert details["waypoint_reached_frame"] == 4
        assert details["replayed_frames"] == 4
        assert update.done.tolist() == [True]
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_absolute_arc_finishes_when_joint_is_reached_while_controller_runs() -> None:
    selector = {
        "stage": "absolute_arc",
        "phase": "pre_move",
        "waypoint": 0,
    }
    stage = {
        "name": "absolute_arc",
        "object": "",
        "operation": "move",
        "operator": "arm",
        "param": {
            "pre_move": [
                {
                    "arc": {
                        "pivot": "hinge",
                        "axis": [0.0, 0.0, 1.0],
                        "angle": 0.2,
                        "absolute": True,
                        "max_step": 0.05,
                        "joint_tolerance": 1e-6,
                    },
                    "reference": "world",
                }
            ]
        },
    }
    runner = TaskRunner().from_config(_mock_task(selector, stages=[stage]))
    try:
        backend = runner._context.backend
        operator = backend.get_operator_handler("arm")
        joint_angle = [0.0]

        def _get_joint_angle(name, env_index=0):
            assert name == "hinge" and env_index == 0
            return joint_angle[0]

        def _get_element_pose(name, env_index=0):
            assert name == "hinge" and env_index == 0
            return PoseState(position=[0.0, 0.0, 0.0])

        def _move(self, pose, target, env_mask=None):
            joint_angle[0] = 0.2
            self.end_effector_pose.position[0] = np.asarray(pose.position)
            self.end_effector_pose.orientation[0] = np.asarray(pose.orientation)
            return ControlResult.filled(1, ControlSignal.RUNNING)

        backend.get_joint_angle = _get_joint_angle
        backend.get_element_pose = _get_element_pose
        operator.move_to_pose = MethodType(_move, operator)

        update = runner.reset()

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["physical_replay"]["replayed_frames"] == 1
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_absolute_frame_beyond_task_end_is_rejected() -> None:
    runner = TaskRunner().from_config(_mock_task({"frame": 3}))
    try:
        with pytest.raises(
            RuntimeError,
            match="end of the task.*before target frame 3",
        ):
            runner.reset()
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_legacy_start_after_uses_teleport_and_preserves_details_shape() -> None:
    selector = {"stage": "move_0", "phase": "pre_move", "waypoint": 0}
    runner = TaskRunner().from_config(_mock_task(None, start_after=selector))
    try:
        backend = runner._context.backend
        assert type(runner._context.config.start_after) is StartAfterWaypointConfig

        def _reject_physical_replay(self):
            raise AssertionError("legacy start_after must not enter physical replay")

        backend.physical_replay_context = MethodType(
            _reject_physical_replay, backend
        )
        operator = backend.get_operator_handler("arm")
        teleport_calls = []
        original_teleport = operator.teleport_end_effector

        def _teleport(self, pose, target=None, env_mask=None):
            teleport_calls.append(1)
            return original_teleport(pose, target=target, env_mask=env_mask)

        operator.teleport_end_effector = MethodType(_teleport, operator)
        update = runner.reset()

        assert teleport_calls == [1]
        assert update.done.tolist() == [True]
        assert "physical_replay" not in update.details[0]
        assert update.details[0]["fast_forward"] == {
            "target": selector,
            "completed_stages": ["move_0"],
            "resume_stage": "",
            "resume_phase": None,
            "resume_waypoint": None,
            "held_objects": {"arm": []},
        }
    finally:
        runner.close()
        ComponentRegistry.clear()


def test_policy_evaluator_reuses_mid_action_physical_replay_state() -> None:
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        _mock_task({"frame": 1})
    )
    try:
        policy.reset()
        update = evaluator.reset()
        resume = evaluator._resume_stages[0]

        assert resume is not None
        assert resume.action_index == 0
        action = policy.act({}, update, evaluator)
        assert action.env_actions[0].action is resume.actions[0]

        update = evaluator.update(action)
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        evaluator.close()
        ComponentRegistry.clear()


def test_policy_evaluator_uses_full_task_frame_for_stop_at() -> None:
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        _mock_task(
            {"frame": 1},
            stages=[_move_stage("move_0", [0.1, 0.2])],
            stop_at={"frame": 3},
        )
    )
    try:
        policy.reset()
        update = evaluator.reset()
        assert evaluator._env_states[0].task_frame == 1

        action = policy.act({}, update, evaluator)
        update = evaluator.update(action)
        assert update.done.tolist() == [False]

        action = policy.act({}, update, evaluator)
        update = evaluator.update(action)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["stop_at"]["task_frame"] == 3
        assert evaluator.records == []
    finally:
        evaluator.close()
        ComponentRegistry.clear()


def test_policy_evaluator_observes_waypoint_stop_metadata() -> None:
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        _mock_task(
            None,
            stages=[_move_stage("move_0", [0.1, 0.2])],
            stop_at={
                "stage": "move_0",
                "phase": "pre_move",
                "waypoint": 0,
                "frame_offset": 1,
            },
        )
    )
    try:
        policy.reset()
        update = evaluator.reset()
        for _ in range(3):
            action = policy.act({}, update, evaluator)
            update = evaluator.update(action)

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.details[0]["stop_at"]["waypoint_reached_frame"] == 2
        assert update.details[0]["stop_at"]["task_frame"] == 3
    finally:
        evaluator.close()
        ComponentRegistry.clear()


def test_config_policy_invalidates_same_stage_cache_on_reset() -> None:
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        _mock_task({"frame": 1})
    )
    try:
        first_update = evaluator.reset()
        first_resume = evaluator._resume_stages[0]
        first_action = policy.act({}, first_update, evaluator).env_actions[0].action
        assert first_action is first_resume.actions[0]

        second_update = evaluator.reset()
        second_resume = evaluator._resume_stages[0]
        second_action = policy.act({}, second_update, evaluator).env_actions[0].action

        assert second_resume is not first_resume
        assert second_action is second_resume.actions[0]
        assert second_action is not first_action
    finally:
        evaluator.close()
        ComponentRegistry.clear()
