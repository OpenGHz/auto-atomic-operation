"""EpisodeStream scheduling: failure policy, slots, shards, and backpressure."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List

import pytest

import auto_atom.mock as mock
from auto_atom.data import (
    AttemptOutcome,
    AttemptScheduler,
    EpisodeAttempt,
    EpisodeStream,
    InvalidEpisodeError,
    RetryBudgetExhaustedError,
    StreamConfig,
    StreamUnhealthyError,
    stream,
)
from auto_atom.policy_eval import ConfigDrivenDemoPolicy
from auto_atom.randomization import RandomizationFailureError

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "aao_configs")

SUCCESS = AttemptOutcome(kind="success", category="succeeded", steps=3)
FAILED = AttemptOutcome(kind="failed", category="missing_grasp", reason="no grasp")
TRUNCATED = AttemptOutcome(kind="truncated", category="max_updates_reached")
UNPLACEABLE = AttemptOutcome(
    kind="randomization_failed", category="randomization_failed"
)


def _config(**fields: Any) -> StreamConfig:
    fields.setdefault("task", "mock")
    fields.setdefault("determinism", "sequential")
    return StreamConfig(base_seed=0, config_dir=CONFIG_DIR, **fields)


class StallingDemoPolicy(ConfigDrivenDemoPolicy):
    """Withholds slot 1's action every other tick, so slot 0 finishes first."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def act(self, observation: Any, update: Any, evaluator: Any) -> Any:
        action = super().act(observation, update, evaluator)
        self.calls += 1
        if self.calls % 2 == 0 and len(action.env_actions) > 1:
            action.env_actions[1] = None
        return action


# ---------------------------------------------------------------------------
#  Scheduler rules
# ---------------------------------------------------------------------------


def test_shards_partition_the_episode_indices() -> None:
    def indices(worker_id: int) -> List[int]:
        scheduler = AttemptScheduler(
            _config(num_episodes=10), worker_id=worker_id, num_workers=3
        )
        found = []
        while (attempt := scheduler.next_attempt()) is not None:
            found.append(attempt.episode_index)
        return found

    assert [indices(worker) for worker in range(3)] == [
        [0, 3, 6, 9],
        [1, 4, 7],
        [2, 5, 8],
    ]


def test_resample_retries_the_same_index_before_new_ones() -> None:
    scheduler = AttemptScheduler(_config(num_episodes=3))
    first = scheduler.next_attempt()

    verdict = scheduler.judge(first, FAILED)
    retry = scheduler.next_attempt()
    emitted = scheduler.judge(retry, SUCCESS)

    assert not verdict.emit
    assert retry == EpisodeAttempt(0, retry=1)
    assert emitted.emit
    assert emitted.invalid_attempts == [
        {
            "retry": 0,
            "kind": "failed",
            "category": "missing_grasp",
            "reason": "no grasp",
        }
    ]
    assert scheduler.next_attempt() == EpisodeAttempt(1)
    stats = scheduler.snapshot()
    assert (stats.attempts, stats.invalid_skipped, stats.retries) == (2, 1, 1)
    assert stats.invalid_reasons == {"failed:missing_grasp": 1}
    assert stats.success_rate == 0.5


def test_exhausted_retry_budget_raises() -> None:
    scheduler = AttemptScheduler(_config(retry_budget=1))

    scheduler.judge(scheduler.next_attempt(), TRUNCATED)
    with pytest.raises(RetryBudgetExhaustedError, match="retry_budget=1") as error:
        scheduler.judge(scheduler.next_attempt(), TRUNCATED)

    assert error.value.attempt == EpisodeAttempt(0, retry=1)
    assert error.value.stats.truncations == 2


def test_keep_yields_failed_rollouts_but_retries_failed_resets() -> None:
    scheduler = AttemptScheduler(_config(on_invalid="keep"))

    unplaceable = scheduler.judge(scheduler.next_attempt(), UNPLACEABLE)
    kept = scheduler.judge(scheduler.next_attempt(), FAILED)

    assert not unplaceable.emit
    assert kept.emit
    assert [entry["kind"] for entry in kept.invalid_attempts] == [
        "randomization_failed"
    ]
    assert scheduler.snapshot().episodes_emitted == 1


def test_consecutive_invalid_attempts_trip_the_breaker() -> None:
    scheduler = AttemptScheduler(_config(on_invalid="keep", max_consecutive_invalid=3))

    scheduler.judge(scheduler.next_attempt(), FAILED)
    scheduler.judge(scheduler.next_attempt(), SUCCESS)
    scheduler.judge(scheduler.next_attempt(), FAILED)
    scheduler.judge(scheduler.next_attempt(), FAILED)
    with pytest.raises(StreamUnhealthyError, match="3 consecutive") as error:
        scheduler.judge(scheduler.next_attempt(), FAILED)

    assert error.value.stats.consecutive_invalid == 3
    assert error.value.stats.valid_rate == pytest.approx(1 / 5)


def test_raise_policy_raises_on_the_first_invalid_attempt() -> None:
    scheduler = AttemptScheduler(_config(on_invalid="raise"))

    with pytest.raises(InvalidEpisodeError, match="episode 0 retry 0 failed"):
        scheduler.judge(scheduler.next_attempt(), FAILED)


def test_stats_snapshot_is_detached_and_json_ready() -> None:
    scheduler = AttemptScheduler(_config())
    snapshot = scheduler.snapshot()

    scheduler.judge(scheduler.next_attempt(), SUCCESS)

    assert snapshot.attempts == 0
    data = scheduler.snapshot().to_dict()
    assert data["recent_steps"] == [3]
    assert data["steps_p50"] == 3.0


# ---------------------------------------------------------------------------
#  Stream
# ---------------------------------------------------------------------------


def test_slots_restart_independently_without_crosstalk() -> None:
    config = _config(task="policy_eval_mock", num_episodes=6, batch_size=2)

    with EpisodeStream.from_config(config, policy=StallingDemoPolicy) as episodes:
        collected = list(episodes)

    assert sorted(episode.episode_index for episode in collected) == list(range(6))
    by_slot = {
        slot: [e for e in collected if e.metadata["slot"] == slot] for slot in (0, 1)
    }
    # Slot 0 is never stalled, so it runs more of the episodes.
    assert len(by_slot[0]) > len(by_slot[1]) >= 1
    for episode in collected:
        assert episode.success
        assert episode.transitions.tick.tolist() == list(
            range(episode.metadata["steps"])
        )
        assert len(episode.records) == 1
        assert episode.records[0].env_index == episode.metadata["slot"]
    assert [len(e) for e in by_slot[0]] == [2] * len(by_slot[0])
    assert all(len(e) > 2 for e in by_slot[1])


def test_shards_together_yield_every_episode_once() -> None:
    base = EpisodeStream.from_config(_config(task="policy_eval_mock", num_episodes=7))
    seen: List[int] = []

    for worker_id in range(3):
        with base.shard(worker_id, 3) as shard:
            seen.extend(episode.episode_index for episode in shard)

    assert sorted(seen) == list(range(7))


def test_shard_of_a_shard_splits_it_further() -> None:
    base = EpisodeStream.from_config(_config(task="policy_eval_mock"))

    nested = base.shard(1, 2).shard(1, 3)

    assert (nested.worker_id, nested.num_workers) == (1 + 1 * 2, 6)


def test_full_queue_holds_the_producer_back() -> None:
    episodes = stream(
        "policy_eval_mock",
        base_seed=0,
        config_dir=CONFIG_DIR,
        determinism="sequential",
        queue_size=1,
        num_episodes=200,
    )
    with episodes:
        next(episodes)
        time.sleep(0.5)
        produced = episodes.stats.attempts

    # One consumed, one queued, one waiting to be queued, one in flight.
    assert produced <= 4
    assert episodes._thread is not None and not episodes._thread.is_alive()


def test_close_mid_stream_tears_the_backend_down(monkeypatch) -> None:
    teardowns: List[str] = []
    original = mock.MockSceneBackend.teardown

    def teardown(self: Any) -> None:
        teardowns.append(self.env_name)
        original(self)

    monkeypatch.setattr(mock.MockSceneBackend, "teardown", teardown)
    episodes = EpisodeStream.from_config(_config(task="policy_eval_mock"))

    next(episodes)
    episodes.close()

    assert teardowns == ["mock_policy_eval"]
    with pytest.raises(StopIteration):
        next(episodes)


def test_failed_reset_randomization_is_resampled(monkeypatch) -> None:
    resets: List[int] = []
    original = mock.MockSceneBackend.reset

    def reset(self: Any, env_mask: Any = None) -> None:
        resets.append(len(resets))
        if len(resets) == 1:
            raise RandomizationFailureError(
                target="cup", attempts=4, violations=["collision"], minimum_clearance=0
            )
        original(self, env_mask)

    monkeypatch.setattr(mock.MockSceneBackend, "reset", reset)

    with EpisodeStream.from_config(
        _config(task="policy_eval_mock", num_episodes=1)
    ) as episodes:
        (episode,) = list(episodes)

    assert episode.metadata["retry"] == 1
    assert episode.metadata["invalid_attempts"][0]["kind"] == "randomization_failed"
    stats = episodes.stats
    assert (stats.randomization_failures, stats.retries) == (1, 1)


def test_producer_errors_reach_the_consumer() -> None:
    with (
        EpisodeStream.from_config(_config(num_episodes=5, retry_budget=0)) as episodes,
        pytest.raises(RetryBudgetExhaustedError),
    ):
        next(episodes)


def test_infeasible_task_trips_the_breaker_through_the_stream() -> None:
    config = _config(on_invalid="keep", max_consecutive_invalid=3, batch_size=2)

    with EpisodeStream.from_config(config) as episodes:
        kept = []
        with pytest.raises(StreamUnhealthyError):
            for episode in episodes:
                kept.append(episode)

    assert len(kept) == 2
    assert not any(episode.success for episode in kept)
    assert episodes.stats.failures == 3
