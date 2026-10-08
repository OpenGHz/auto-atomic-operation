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
from auto_atom.data.source import resolve_policy_factory
from auto_atom.execution_model import StageExecutionStatus
from auto_atom.policy_eval import ConfigDrivenDemoPolicy
from auto_atom.runtime import ComponentRegistry

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "aao_configs")


class ObservingDemoPolicy(ConfigDrivenDemoPolicy):
    """Demo policy whose observation getter reports EEF poses and a capture clock."""

    def __init__(self) -> None:
        super().__init__()
        self.captures = 0

    def observation_getter(self, context: Any) -> Dict[str, Dict[str, Any]]:
        self.captures += 1
        backend = context.backend
        operator = sorted(backend.operators)[0]
        eef = backend.get_operator_handler(operator).get_end_effector_pose()
        stamp = np.full(backend.batch_size, float(self.captures))
        return {
            "arm/eef/position": {"data": eef.position.copy(), "t": stamp},
            "arm/eef/orientation": {"data": eef.orientation.copy(), "t": stamp},
            "action/arm/eef/position": {"data": eef.position.copy(), "t": stamp},
        }


class ZeroActionPolicy:
    """External numeric policy; the mock never completes a stage with it."""

    def act(self, observation: Any, update: Any, evaluator: Any) -> np.ndarray:
        return np.zeros((evaluator.batch_size, 3))


def _config(task: str, **fields: Any) -> StreamConfig:
    fields.setdefault("determinism", "sequential")
    fields.setdefault("num_episodes", 1)
    return StreamConfig(task=task, base_seed=0, config_dir=CONFIG_DIR, **fields)


def _run(config: StreamConfig, policy: Any = ObservingDemoPolicy) -> List[Any]:
    source = EvaluatorEpisodeSource(config, policy=policy)
    try:
        return list(source.episodes())
    finally:
        source.close()


def test_successful_episode_rows_align_with_ticks_labels_and_observations() -> None:
    (episode,) = _run(_config("policy_eval_mock"))

    arrays = episode.transitions
    steps = episode.metadata["steps"]
    assert episode.success and not episode.truncated
    assert episode.failure_reason is None
    assert len(episode) == steps == 2
    assert arrays.tick.tolist() == list(range(steps))
    assert arrays.obs["arm/eef/position"].shape == (steps, 3)
    assert arrays.stage_name.tolist() == ["observe_home_pose"] * steps
    assert arrays.stage_index.tolist() == [0] * steps
    # The capture right after the reset is the initial observation, so the
    # observation of tick t is capture t + 2.
    np.testing.assert_array_equal(arrays.sim_time, np.arange(steps) + 2.0)
    assert set(episode.initial_observation) == {
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


def test_truncated_episode_records_the_numeric_policy_action() -> None:
    (episode,) = _run(
        _config("mock", on_invalid="keep", max_updates=5), policy=ZeroActionPolicy
    )

    assert episode.truncated and not episode.success
    assert episode.failure_reason == "reached max_updates=5 before task completion"
    assert episode.metadata["failure_category"] == "max_updates_reached"
    assert episode.transitions.action["policy"].shape == (5, 3)


def test_sample_stride_keeps_every_kth_tick_and_the_last() -> None:
    (episode,) = _run(_config("mock", on_invalid="keep", sample_stride=3))

    steps = episode.metadata["steps"]
    expected = [tick for tick in range(steps) if (tick + 1) % 3 == 0]
    if steps - 1 not in expected:
        expected.append(steps - 1)
    assert episode.transitions.tick.tolist() == expected


def test_observation_keys_filter_measurements_but_keep_commands() -> None:
    (episode,) = _run(
        _config("policy_eval_mock", observation_keys=["arm/eef/position"])
    )

    assert list(episode.transitions.obs) == ["arm/eef/position"]
    assert list(episode.initial_observation) == ["arm/eef/position"]
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


def test_batched_stream_rejects_a_policy_whose_reset_takes_no_mask() -> None:
    class AllSlotsPolicy(ConfigDrivenDemoPolicy):
        def reset(self) -> None:
            super().reset()

    with pytest.raises(TypeError, match="env_mask"):
        _run(_config("policy_eval_mock", batch_size=2), policy=AllSlotsPolicy)


def test_close_tears_down_and_clears_the_registry() -> None:
    source = EvaluatorEpisodeSource(
        _config("policy_eval_mock"), policy=ObservingDemoPolicy
    )
    episodes = source.episodes()
    next(episodes)

    assert ComponentRegistry.has_env("mock_policy_eval")
    source.close()

    assert not ComponentRegistry.has_env("mock_policy_eval")


def test_policy_names_resolve_to_factories() -> None:
    assert resolve_policy_factory("demo") is ConfigDrivenDemoPolicy
    assert (
        resolve_policy_factory("auto_atom.policy_eval:ConfigDrivenDemoPolicy")
        is ConfigDrivenDemoPolicy
    )
    with pytest.raises(TypeError, match="not callable"):
        resolve_policy_factory("auto_atom.data.recording:POLICY_ACTION_KEY")
