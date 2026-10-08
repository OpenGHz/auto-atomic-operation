"""Which episode attempts run, and what becomes of their outcomes.

Every reset is an *attempt* at one ``episode_index``; ``retry`` counts the
attempts of that index before it. The scheduler hands attempts to the
producer, judges each outcome against the stream's ``on_invalid`` policy, and
keeps the statistics. It owns the failure rules and nothing about how an
attempt is run, so the rules are testable without an evaluator.
"""

from __future__ import annotations

import copy
import itertools
import threading
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Literal, Optional

import numpy as np

from .config import StreamConfig

AttemptKind = Literal["success", "failed", "truncated", "randomization_failed"]


@dataclass(frozen=True)
class EpisodeAttempt:
    """One attempt at an episode: its index and how many attempts preceded it."""

    episode_index: int
    retry: int = 0


@dataclass(frozen=True)
class AttemptOutcome:
    """How one attempt ended."""

    kind: AttemptKind
    category: str
    """Short reason class (``failure_category``), used for counting."""
    reason: Optional[str] = None
    steps: int = 0
    wall_time: float = 0.0
    sim_time: float = 0.0

    @property
    def valid(self) -> bool:
        return self.kind == "success"

    @property
    def is_rollout(self) -> bool:
        """Whether the attempt ran (it has an episode to keep)."""
        return self.kind != "randomization_failed"

    def describe(self, attempt: EpisodeAttempt) -> Dict[str, Any]:
        return {
            "retry": attempt.retry,
            "kind": self.kind,
            "category": self.category,
            "reason": self.reason,
        }


@dataclass
class StreamStats:
    """Counters of a stream. Read through ``EpisodeStream.stats`` (a snapshot)."""

    attempts: int = 0
    """Resets tried, including those whose randomization failed."""
    episodes_emitted: int = 0
    successes: int = 0
    failures: int = 0
    """Rollouts that finished without success."""
    truncations: int = 0
    randomization_failures: int = 0
    invalid_skipped: int = 0
    """Invalid attempts discarded rather than yielded."""
    retries: int = 0
    consecutive_invalid: int = 0
    invalid_reasons: Dict[str, int] = field(default_factory=dict)
    """Invalid attempts by ``kind:category``, kept or not."""
    wall_time_total: float = 0.0
    sim_time_total: float = 0.0
    recent_steps: Deque[int] = field(default_factory=lambda: deque(maxlen=1024))
    """Control ticks of the latest rollouts (bounded)."""

    @property
    def rollouts(self) -> int:
        return self.successes + self.failures + self.truncations

    @property
    def success_rate(self) -> float:
        """Successes over finished rollouts: the task's own success rate."""
        return self.successes / self.rollouts if self.rollouts else float("nan")

    @property
    def valid_rate(self) -> float:
        """Successes over all attempts; a falling value warrants an early stop."""
        return self.successes / self.attempts if self.attempts else float("nan")

    @property
    def steps_p50(self) -> float:
        return _percentile(self.recent_steps, 50)

    @property
    def steps_p95(self) -> float:
        return _percentile(self.recent_steps, 95)

    @property
    def wall_time_per_episode(self) -> float:
        return self.wall_time_total / self.rollouts if self.rollouts else float("nan")

    @property
    def sim_time_per_episode(self) -> float:
        return self.sim_time_total / self.rollouts if self.rollouts else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """Counters and derived rates as plain JSON-ready values."""
        data = asdict(self)
        data["recent_steps"] = list(self.recent_steps)
        for name in (
            "rollouts",
            "success_rate",
            "valid_rate",
            "steps_p50",
            "steps_p95",
            "wall_time_per_episode",
            "sim_time_per_episode",
        ):
            data[name] = getattr(self, name)
        return data


class StreamError(RuntimeError):
    """Base class of the stream's failure-policy errors; carries the stats."""

    def __init__(self, message: str, stats: StreamStats) -> None:
        super().__init__(message)
        self.stats = stats


class InvalidEpisodeError(StreamError):
    """An invalid attempt under ``on_invalid="raise"``."""

    def __init__(
        self,
        message: str,
        stats: StreamStats,
        *,
        attempt: EpisodeAttempt,
        outcome: AttemptOutcome,
    ) -> None:
        super().__init__(message, stats)
        self.attempt = attempt
        self.outcome = outcome


class RetryBudgetExhaustedError(InvalidEpisodeError):
    """An episode index stayed invalid through ``retry_budget`` retries."""


class StreamUnhealthyError(StreamError):
    """``max_consecutive_invalid`` attempts in a row were invalid.

    A task that cannot be done (every reset fails, every rollout times out)
    would otherwise retry forever.
    """


@dataclass(frozen=True)
class Verdict:
    """What to do with a finished attempt."""

    emit: bool
    invalid_attempts: List[Dict[str, Any]]
    """The discarded attempts of this index, for ``Episode.metadata``."""


class AttemptScheduler:
    """Hand out attempts for one shard and judge their outcomes.

    Shard ``worker_id`` of ``num_workers`` runs episode indices
    ``worker_id + k * num_workers``. Retries go before new indices, so an index
    is settled before the stream moves far past it. Thread-safe: the producer
    thread schedules while the consumer reads :meth:`snapshot`.
    """

    def __init__(
        self, config: StreamConfig, *, worker_id: int = 0, num_workers: int = 1
    ) -> None:
        if num_workers < 1 or not 0 <= worker_id < num_workers:
            raise ValueError(
                f"worker_id must be in [0, num_workers); got {worker_id} of "
                f"{num_workers}"
            )
        self.config = config
        self.worker_id = worker_id
        self.num_workers = num_workers
        self._indices = self._shard_indices()
        self._retries: Deque[EpisodeAttempt] = deque()
        self._invalid: Dict[int, List[Dict[str, Any]]] = {}
        self._stats = StreamStats()
        self._lock = threading.Lock()

    def _shard_indices(self) -> Iterator[int]:
        if self.config.num_episodes is None:
            return itertools.count(self.worker_id, self.num_workers)
        return iter(range(self.worker_id, self.config.num_episodes, self.num_workers))

    def snapshot(self) -> StreamStats:
        with self._lock:
            return copy.deepcopy(self._stats)

    def next_attempt(self) -> Optional[EpisodeAttempt]:
        """The next attempt to run, or ``None`` when the shard has no more."""
        with self._lock:
            if self._retries:
                return self._retries.popleft()
            index = next(self._indices, None)
            return None if index is None else EpisodeAttempt(index)

    def judge(self, attempt: EpisodeAttempt, outcome: AttemptOutcome) -> Verdict:
        """Count ``outcome`` and decide whether its episode is yielded.

        Raises :class:`InvalidEpisodeError`, :class:`RetryBudgetExhaustedError`
        or :class:`StreamUnhealthyError` as the failure policy demands.
        """
        with self._lock:
            stats = self._stats
            stats.attempts += 1
            if outcome.kind == "randomization_failed":
                stats.randomization_failures += 1
            else:
                stats.successes += outcome.kind == "success"
                stats.failures += outcome.kind == "failed"
                stats.truncations += outcome.kind == "truncated"
                stats.recent_steps.append(outcome.steps)
                stats.wall_time_total += outcome.wall_time
                stats.sim_time_total += outcome.sim_time

            if outcome.valid:
                stats.consecutive_invalid = 0
                stats.episodes_emitted += 1
                return Verdict(True, self._invalid.pop(attempt.episode_index, []))

            reasons = Counter(stats.invalid_reasons)
            reasons[f"{outcome.kind}:{outcome.category}"] += 1
            stats.invalid_reasons = dict(reasons)
            stats.consecutive_invalid += 1
            message = _describe(attempt, outcome)
            if self.config.on_invalid == "raise":
                raise InvalidEpisodeError(
                    message, copy.deepcopy(stats), attempt=attempt, outcome=outcome
                )
            if stats.consecutive_invalid >= self.config.max_consecutive_invalid:
                raise StreamUnhealthyError(
                    f"{stats.consecutive_invalid} consecutive invalid attempts "
                    f"(max_consecutive_invalid="
                    f"{self.config.max_consecutive_invalid}); last: {message}",
                    copy.deepcopy(stats),
                )

            if self.config.on_invalid == "keep" and outcome.is_rollout:
                stats.episodes_emitted += 1
                return Verdict(True, self._invalid.pop(attempt.episode_index, []))

            stats.invalid_skipped += 1
            history = self._invalid.setdefault(attempt.episode_index, [])
            history.append(outcome.describe(attempt))
            if attempt.retry >= self.config.retry_budget:
                raise RetryBudgetExhaustedError(
                    f"Episode {attempt.episode_index} stayed invalid through "
                    f"retry_budget={self.config.retry_budget} retries; last: "
                    f"{message}",
                    copy.deepcopy(stats),
                    attempt=attempt,
                    outcome=outcome,
                )
            stats.retries += 1
            self._retries.append(
                EpisodeAttempt(attempt.episode_index, attempt.retry + 1)
            )
            return Verdict(False, [])


def _describe(attempt: EpisodeAttempt, outcome: AttemptOutcome) -> str:
    reason = f": {outcome.reason}" if outcome.reason else ""
    return (
        f"episode {attempt.episode_index} retry {attempt.retry} "
        f"{outcome.kind} ({outcome.category}){reason}"
    )


def _percentile(values: Deque[int], q: float) -> float:
    return float(np.percentile(list(values), q)) if values else float("nan")
