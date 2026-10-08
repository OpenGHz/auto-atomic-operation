"""Streaming episodes for training: ``for episode in stream(...)``.

AAO produces episodes on demand, in process, without writing them to disk.
:class:`EpisodeStream` schedules env slots, shards episode indices across
workers, buffers a bounded number of finished episodes, and applies the
failure policy; :class:`EvaluatorEpisodeSource` rolls the episodes out with a
:class:`~auto_atom.policy_eval.PolicyEvaluator`.

See ``docs/design/streaming-data-loader-design.md``.
"""

from .attempts import (
    AttemptOutcome,
    AttemptScheduler,
    EpisodeAttempt,
    InvalidEpisodeError,
    RetryBudgetExhaustedError,
    StreamError,
    StreamStats,
    StreamUnhealthyError,
)
from .config import StreamConfig
from .records import Episode, EpisodeArrays, Transition
from .source import EpisodeSource, EvaluatorEpisodeSource, scene_ground_truth
from .stream import EpisodeStream, stream

__all__ = [
    "AttemptOutcome",
    "AttemptScheduler",
    "Episode",
    "EpisodeArrays",
    "EpisodeAttempt",
    "EpisodeSource",
    "EpisodeStream",
    "EvaluatorEpisodeSource",
    "InvalidEpisodeError",
    "RetryBudgetExhaustedError",
    "StreamConfig",
    "StreamError",
    "StreamStats",
    "StreamUnhealthyError",
    "Transition",
    "scene_ground_truth",
    "stream",
]
