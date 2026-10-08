"""EvaluatorEpisodeSource on the simulator-free mock tasks.

``policy_eval_mock`` (one move stage) always succeeds; ``mock`` always fails,
because its pick needs a grasp the mock backend never reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

import auto_atom.mock as mock
from auto_atom.data import (
    EvaluatorEpisodeSource,
    InvalidEpisodeError,
    StreamConfig,
)
from auto_atom.execution_model import StageExecutionStatus
from auto_atom.runtime import ComponentRegistry

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "aao_configs")


@pytest.fixture(autouse=True)
def captures(monkeypatch) -> Dict[str, int]:
    """Make the mock env report EEF poses, an EEF command, and a capture clock.

    The EEF command channel reports the EEF pose, which the mock operator sets
    on reaching it. Returns the count of full captures.
    """
    counts = {"full": 0}
    original = mock.MockSceneBackend.__post_init__

    def post_init(self: Any) -> None:
        original(self)
        self.env.backend = self

    def capture_observation(env: Any) -> Dict[str, Dict[str, Any]]:
        counts["full"] += 1
        backend = env.backend
        operator = sorted(backend.operators)[0]
        eef = backend.get_operator_handler(operator).get_end_effector_pose()
        stamp = np.full(backend.batch_size, float(counts["full"]))
        return {
            "arm/eef/position": {"data": eef.position.copy(), "t": stamp},
            "arm/eef/orientation": {"data": eef.orientation.copy(), "t": stamp},
            "action/arm/eef/position": {"data": eef.position.copy(), "t": stamp},
        }

    monkeypatch.setattr(mock.MockSceneBackend, "__post_init__", post_init)
    monkeypatch.setattr(mock.MockEnv, "capture_observation", capture_observation)
    return counts


@pytest.fixture
def command_captures(monkeypatch) -> List[int]:
    """Give the mock env a render-free ``capture_commands``; logs each call."""
    calls: List[int] = []

    def capture_commands(env: Any) -> Dict[str, Dict[str, Any]]:
        calls.append(1)
        backend = env.backend
        operator = sorted(backend.operators)[0]
        eef = backend.get_operator_handler(operator).get_end_effector_pose()
        stamp = np.full(backend.batch_size, -1.0)
        return {"action/arm/eef/position": {"data": eef.position.copy(), "t": stamp}}

    monkeypatch.setattr(
        mock.MockEnv, "capture_commands", capture_commands, raising=False
    )
    return calls


def _config(task: str, **fields: Any) -> StreamConfig:
    fields.setdefault("determinism", "sequential")
    fields.setdefault("num_episodes", 1)
    return StreamConfig(task=task, base_seed=0, config_dir=CONFIG_DIR, **fields)


def _run(config: StreamConfig) -> List[Any]:
    source = EvaluatorEpisodeSource(config)
    try:
        return list(source.episodes())
    finally:
        source.close()


def test_each_row_pairs_the_observation_before_a_tick_with_its_command() -> None:
    (episode,) = _run(_config("policy_eval_mock"))

    arrays = episode.transitions
    steps = episode.metadata["steps"]
    assert episode.success and not episode.truncated
    assert episode.failure_reason is None
    assert len(episode) == steps == 2
    assert arrays.tick.tolist() == list(range(steps))
    assert arrays.stage_name.tolist() == ["observe_home_pose"] * steps
    assert arrays.stage_index.tolist() == [0] * steps
    # Capture 1 follows the reset and capture t + 1 follows tick t - 1, so the
    # observation tick t decides on is capture t + 1.
    np.testing.assert_array_equal(arrays.sim_time, np.arange(steps) + 1.0)
    # Tick 1 reaches the commanded pose: its command is what the observation
    # after it shows, the final observation.
    position = arrays.obs["arm/eef/position"]
    commands = arrays.action["action/arm/eef/position"]
    np.testing.assert_array_equal(commands[0], position[1])
    np.testing.assert_array_equal(
        commands[-1], episode.final_observation["arm/eef/position"]
    )
    np.testing.assert_allclose(position[0], [0.2, 0.0, 0.3])
    assert set(episode.final_observation) == {
        "arm/eef/position",
        "arm/eef/orientation",
    }
    assert list(arrays.action) == ["action/arm/eef/position"]


def test_successful_episode_carries_scene_records_and_metadata() -> None:
    config = _config("policy_eval_mock")

    (episode,) = _run(config)

    assert episode.episode_index == 0
    assert episode.seed == 0
    assert episode.task == "policy_eval_mock"
    assert set(episode.scene) == {"objects", "operators", "cameras"}
    np.testing.assert_allclose(
        episode.scene["operators"]["arm"]["eef"]["position"], [0.2, 0.0, 0.3]
    )
    assert [record.status for record in episode.records] == [
        StageExecutionStatus.SUCCEEDED
    ]
    metadata = episode.metadata
    assert metadata["config"] == config.model_dump(mode="json")
    assert metadata["retry"] == 0
    assert metadata["reset_index"] == 1
    assert (metadata["worker_id"], metadata["num_workers"], metadata["slot"]) == (
        0,
        1,
        0,
    )
    assert metadata["outcome"] == "success"
    assert metadata["invalid_attempts"] == []


def test_failed_episode_is_kept_with_its_reason() -> None:
    (episode,) = _run(_config("mock", on_invalid="keep"))

    assert not episode.success and not episode.truncated
    assert "grasp" in episode.failure_reason
    assert episode.metadata["failure_category"] == "missing_grasp"
    assert episode.records[-1].status == StageExecutionStatus.FAILED
    assert len(episode) == episode.metadata["steps"]
    assert set(episode.scene["objects"]) == {"cup", "shelf", "tray"}


def test_episode_cut_by_max_updates_is_truncated() -> None:
    (episode,) = _run(_config("mock", on_invalid="keep", max_updates=2))

    assert episode.truncated and not episode.success
    assert episode.failure_reason == "reached max_updates=2 before task completion"
    assert episode.metadata["failure_category"] == "max_updates_reached"
    assert episode.metadata["steps"] == 2


def test_sample_stride_records_ticks_zero_k_2k(
    captures: Dict[str, int], command_captures: List[int]
) -> None:
    (episode,) = _run(_config("mock", on_invalid="keep", sample_stride=3))

    steps = episode.metadata["steps"]
    assert steps == 4
    assert episode.transitions.tick.tolist() == [0, 3]
    # Full captures: after the reset, before tick 3, and the final one. Tick
    # 0's command needs one more capture, and it renders nothing.
    assert captures["full"] == 3
    assert len(command_captures) == 1
    np.testing.assert_array_equal(episode.transitions.sim_time, [1.0, 2.0])


def test_without_capture_commands_a_command_costs_a_full_capture(
    captures: Dict[str, int],
) -> None:
    (episode,) = _run(_config("mock", on_invalid="keep", sample_stride=3))

    assert episode.transitions.tick.tolist() == [0, 3]
    assert captures["full"] == 4
    # The extra capture follows tick 0; the rows still hold their own frames.
    np.testing.assert_array_equal(episode.transitions.sim_time, [1.0, 3.0])


def test_observation_keys_filter_measurements_but_keep_commands() -> None:
    (episode,) = _run(
        _config("policy_eval_mock", observation_keys=["arm/eef/position"])
    )

    assert list(episode.transitions.obs) == ["arm/eef/position"]
    assert list(episode.final_observation) == ["arm/eef/position"]
    assert list(episode.transitions.action) == ["action/arm/eef/position"]


def test_unknown_observation_keys_fail_before_the_first_tick() -> None:
    with pytest.raises(KeyError, match="not produced"):
        _run(_config("policy_eval_mock", observation_keys=["camera/missing"]))


def test_on_invalid_raise_raises_with_the_outcome() -> None:
    with pytest.raises(InvalidEpisodeError) as error:
        _run(_config("mock", on_invalid="raise"))

    assert error.value.outcome.kind == "failed"
    assert error.value.stats.failures == 1


def test_episode_determinism_fails_closed_without_reset_addressing(
    monkeypatch,
) -> None:
    monkeypatch.delattr(mock.MockSceneBackend, "set_reset_address")

    with pytest.raises(TypeError, match="AddressableResetHost"):
        _run(_config("policy_eval_mock", determinism="episode"))


def test_close_tears_down_and_clears_the_registry() -> None:
    source = EvaluatorEpisodeSource(_config("policy_eval_mock"))
    episodes = source.episodes()
    next(episodes)

    assert ComponentRegistry.has_env("mock_policy_eval")
    source.close()

    assert not ComponentRegistry.has_env("mock_policy_eval")
