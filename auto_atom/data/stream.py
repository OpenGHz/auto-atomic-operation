"""Iterate episodes produced in the background, with bounded memory.

An :class:`EpisodeStream` runs its producer in one background thread that
builds the simulator, rolls out episodes, and tears the simulator down; the
consumer only takes finished episodes from a bounded queue, so a slow
consumer blocks the producer instead of growing memory.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Union

from .attempts import AttemptScheduler, StreamStats
from .config import StreamConfig
from .records import Episode
from .source import EvaluatorEpisodeSource, PolicyFactory

_POLL_SECONDS = 0.1


class _End:
    """Queue sentinel: the producer finished and released its resources."""


@dataclass(frozen=True)
class _Failure:
    """Queue item: the producer raised ``error``."""

    error: BaseException


_Item = Union[Episode, _End, _Failure]


class EpisodeStream(Iterator[Episode]):
    """Episodes of one task, produced in the background.

    Use it as a context manager, or call :meth:`close`, so the simulator is
    torn down when the consumer stops early. A failure-policy error
    (``StreamUnhealthyError`` and friends) or a simulator error raised by the
    producer is re-raised by ``next()``.
    """

    def __init__(
        self,
        config: StreamConfig,
        *,
        policy: Optional[PolicyFactory] = None,
        worker_id: int = 0,
        num_workers: int = 1,
    ) -> None:
        self.config = config
        self._policy = policy
        self._scheduler = AttemptScheduler(
            config, worker_id=worker_id, num_workers=num_workers
        )
        self._queue: "queue.Queue[_Item]" = queue.Queue(maxsize=config.queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._finished = False

    @classmethod
    def from_config(
        cls, config: StreamConfig, *, policy: Optional[PolicyFactory] = None
    ) -> "EpisodeStream":
        """A stream of ``config``; ``policy`` overrides ``config.policy``.

        ``policy`` must be picklable (a module-level callable) for the stream
        to be rebuilt in a spawned worker.
        """
        return cls(config, policy=policy)

    @property
    def worker_id(self) -> int:
        return self._scheduler.worker_id

    @property
    def num_workers(self) -> int:
        return self._scheduler.num_workers

    def shard(self, worker_id: int, num_workers: int) -> "EpisodeStream":
        """The part of this stream that worker ``worker_id`` of ``num_workers`` runs.

        A worker runs episode indices ``worker_id + k * num_workers``. Sharding
        a shard splits it further, so a rank can shard by process and then by
        DataLoader worker. Must be called before iterating.
        """
        if self._thread is not None:
            raise RuntimeError("shard() must be called before iterating the stream.")
        if num_workers < 1 or not 0 <= worker_id < num_workers:
            raise ValueError(
                f"worker_id must be in [0, num_workers); got {worker_id} of "
                f"{num_workers}"
            )
        return EpisodeStream(
            self.config,
            policy=self._policy,
            worker_id=self.worker_id + worker_id * self.num_workers,
            num_workers=self.num_workers * num_workers,
        )

    @property
    def stats(self) -> StreamStats:
        """A snapshot of the stream's counters."""
        return self._scheduler.snapshot()

    def __iter__(self) -> "EpisodeStream":
        return self

    def __next__(self) -> Episode:
        if self._finished:
            raise StopIteration
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._produce, name="aao-episode-stream", daemon=True
            )
            self._thread.start()
        item = self._queue.get()
        if isinstance(item, Episode):
            return item
        self._finished = True
        self._thread.join()
        if isinstance(item, _Failure):
            raise item.error
        raise StopIteration

    def close(self) -> None:
        """Stop the producer and wait until it has torn the simulator down."""
        self._finished = True
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        while thread.is_alive():
            self._drain()
            thread.join(_POLL_SECONDS)
        self._drain()

    def __enter__(self) -> "EpisodeStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _produce(self) -> None:
        source = EvaluatorEpisodeSource(
            self.config,
            scheduler=self._scheduler,
            policy=self._policy,
            stop_event=self._stop,
        )
        episodes = source.episodes()
        last: _Item = _End()
        try:
            for episode in episodes:
                if not self._put(episode):
                    break
        except BaseException as error:  # handed to the consumer
            last = _Failure(error)
        finally:
            try:
                episodes.close()
                source.close()
            except BaseException as error:
                if isinstance(last, _End):
                    last = _Failure(error)
        self._put(last)

    def _put(self, item: _Item) -> bool:
        """Queue ``item``, waiting for space; ``False`` once the stream is closed."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=_POLL_SECONDS)
                return True
            except queue.Full:
                continue
        return False

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


def stream(
    task: str,
    *,
    base_seed: int,
    policy: Optional[PolicyFactory] = None,
    **options: Any,
) -> EpisodeStream:
    """``EpisodeStream`` of ``task``; ``options`` are further ``StreamConfig`` fields.

    ::

        with stream("pick_and_place", base_seed=0, num_episodes=8) as episodes:
            for episode in episodes:
                ...
    """
    config = StreamConfig(task=task, base_seed=base_seed, **options)
    return EpisodeStream.from_config(config, policy=policy)
