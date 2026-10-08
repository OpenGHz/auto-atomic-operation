"""Episode records, row recording, and StreamConfig validation (no simulator)."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from pydantic import ValidationError

from auto_atom.data import EpisodeArrays, StreamConfig, Transition
from auto_atom.data.recording import (
    EpisodeRecorder,
    is_command_key,
    policy_action_row,
    split_observation,
)
from auto_atom.policy_eval import ConfigDrivenPolicyAction
from auto_atom.runtime import TaskUpdate


def _arrays(length: int) -> EpisodeArrays:
    return EpisodeArrays(
        obs={"cam/color": np.arange(length * 2).reshape(length, 2)},
        action={"action/arm/pose": np.arange(length, dtype=np.float64)},
        tick=np.arange(length, dtype=np.int64),
        stage_index=np.zeros(length, dtype=np.int64),
        stage_name=np.asarray(["pick"] * length),
        phase=np.asarray(["pre_move"] * (length - 1) + [""]),
        phase_step=np.asarray([0] * (length - 1) + [-1], dtype=np.int64),
        sim_time=np.linspace(0.0, 1.0, length),
    )


def _update(
    batch_size: int, stage: int = 0, phase: str | None = "pre_move"
) -> TaskUpdate:
    return TaskUpdate(
        stage_index=np.full(batch_size, stage, dtype=np.int64),
        stage_name=[f"stage_{stage}"] * batch_size,
        status=np.asarray([None] * batch_size, dtype=object),
        done=np.zeros(batch_size, dtype=bool),
        success=np.zeros(batch_size, dtype=bool),
        details=[{} for _ in range(batch_size)],
        phase=[phase] * batch_size,
        phase_step=np.full(batch_size, 3, dtype=np.int64),
    )


def test_episode_arrays_reject_columns_of_different_length() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        EpisodeArrays(
            obs={"cam/color": np.zeros((3, 2))},
            action={},
            tick=np.arange(4),
            stage_index=np.zeros(4, dtype=np.int64),
            stage_name=np.asarray(["a"] * 4),
            phase=np.asarray([""] * 4),
            phase_step=np.zeros(4, dtype=np.int64),
            sim_time=np.zeros(4),
        )


def test_integer_index_is_a_transition_with_sentinels_mapped_to_none() -> None:
    arrays = _arrays(3)

    first = arrays[0]
    last = arrays[-1]

    assert isinstance(first, Transition)
    assert first.phase == "pre_move" and first.phase_step == 0
    assert last.tick == 2
    assert last.phase is None and last.phase_step is None
    np.testing.assert_array_equal(last.obs["cam/color"], [4, 5])
    with pytest.raises(IndexError):
        arrays[3]


def test_slices_are_views_and_windows_are_full_only() -> None:
    arrays = _arrays(5)

    windows = list(arrays.window(2, 2))

    assert [window.tick.tolist() for window in windows] == [[0, 1], [2, 3]]
    assert np.shares_memory(windows[0].obs["cam/color"], arrays.obs["cam/color"])
    assert list(_arrays(1).window(2, 1)) == []
    with pytest.raises(ValueError, match="positive"):
        list(arrays.window(0, 1))


def test_records_survive_pickle() -> None:
    arrays = _arrays(3)

    restored = pickle.loads(pickle.dumps(arrays))

    np.testing.assert_array_equal(restored.sim_time, arrays.sim_time)
    assert restored[1].stage_name == "pick"


def test_command_keys_are_recognized_with_a_prefix() -> None:
    assert is_command_key("action/arm/pose/position")
    assert is_command_key("/robot/action/arm/pose")
    assert not is_command_key("camera/action_cam/color")
    assert not is_command_key("arm/pose/position")


def test_split_observation_separates_commands_and_filters_measurements() -> None:
    observation = {
        "arm/pose/position": {
            "data": np.arange(6).reshape(2, 3),
            "t": np.array([1, 2]),
        },
        "cam/color": {"data": np.zeros((2, 4, 4, 3)), "t": np.array([1, 5])},
        "action/arm/pose/position": {"data": np.ones((2, 3)), "t": np.array([1, 2])},
    }

    measured, commands, sim_time = split_observation(
        observation, 1, observation_keys={"arm/pose/position"}
    )

    assert list(measured) == ["arm/pose/position"]
    np.testing.assert_array_equal(measured["arm/pose/position"], [3, 4, 5])
    assert list(commands) == ["action/arm/pose/position"]
    assert sim_time == 2.0


def test_split_observation_rejects_a_reshaped_observation() -> None:
    with pytest.raises(TypeError, match="capture_observation"):
        split_observation(np.zeros((2, 3)), 0)
    with pytest.raises(TypeError, match="'data', 't'"):
        split_observation({"arm": np.zeros((2, 3))}, 0)


def test_policy_action_rows_follow_the_default_applier_layout() -> None:
    batched = policy_action_row(np.arange(6.0).reshape(2, 3), 1, 2)
    single = policy_action_row(np.arange(3.0), 0, 1)
    mapping = policy_action_row({"action": [[1.0], [2.0]]}, 1, 2)

    np.testing.assert_array_equal(batched["policy"], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(single["policy"], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(mapping["policy/action"], [2.0])
    assert policy_action_row(ConfigDrivenPolicyAction(env_actions=[None]), 0, 1) == {}
    assert policy_action_row(None, 0, 1) == {}
    with pytest.raises(ValueError, match="batch size"):
        policy_action_row(np.zeros((3, 2)), 0, 2)


def test_recorder_stacks_rows_and_keeps_ragged_values_as_objects() -> None:
    recorder = EpisodeRecorder()
    for tick, ragged in enumerate(([1], [1, 2])):
        recorder.append(
            tick=tick,
            obs={"pos": np.full(3, tick), "ragged": ragged},
            action={"action/arm": np.zeros(2)},
            update=_update(2, stage=tick, phase=None if tick else "pre_move"),
            env_index=1,
            sim_time=0.5 * tick,
        )

    arrays = recorder.arrays()

    assert len(arrays) == 2
    assert arrays.obs["pos"].shape == (2, 3)
    assert arrays.obs["ragged"].dtype == object
    assert arrays.stage_name.tolist() == ["stage_0", "stage_1"]
    assert arrays.phase.tolist() == ["pre_move", ""]
    assert arrays.phase_step.tolist() == [3, 3]
    assert len(EpisodeRecorder().arrays()) == 0


def test_stream_config_requires_an_explicit_seed_and_accepts_zero() -> None:
    with pytest.raises(ValidationError):
        StreamConfig(task="mock")
    with pytest.raises(ValidationError):
        StreamConfig(task="mock", base_seed=None)

    config = StreamConfig(task="mock", base_seed=0, overrides=["env.viewer=null"])

    assert config.base_seed == 0
    assert config.overrides == ("env.viewer=null",)
    assert config.model_dump(mode="json")["overrides"] == ["env.viewer=null"]


@pytest.mark.parametrize(
    "fields",
    [
        {"queue_size": 0},
        {"retry_budget": -1},
        {"max_consecutive_invalid": 0},
        {"on_invalid": "skip"},
        {"determinism": "random"},
        {"policy": "my_policy"},
        {"unknown_field": 1},
    ],
)
def test_stream_config_rejects_invalid_fields(fields: dict) -> None:
    with pytest.raises(ValidationError):
        StreamConfig(task="mock", base_seed=0, **fields)


@pytest.mark.parametrize(
    "override",
    ["task.seed=1", "++task.seed=1", "+env.batch_size=4", "env.batch_size=2"],
)
def test_stream_config_reserves_seed_and_batch_size_overrides(override: str) -> None:
    with pytest.raises(ValidationError, match="use StreamConfig"):
        StreamConfig(task="mock", base_seed=0, overrides=[override])


def test_stream_config_is_frozen() -> None:
    config = StreamConfig(task="mock", base_seed=0)

    with pytest.raises(ValidationError):
        config.base_seed = 1
