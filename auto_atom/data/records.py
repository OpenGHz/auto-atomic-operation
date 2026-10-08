"""What a stream yields: one episode, reset to reset, stored as columns.

An episode is the dataset-level name for the rollout between two resets.
These are records, not configuration, so they are frozen dataclasses: they
carry large numpy arrays, are built once per episode (once per tick for a
:class:`Transition` view), and cross worker and queue boundaries by pickle.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Iterator, List, Optional, Union, overload

import numpy as np

from auto_atom.execution_model import ExecutionRecord


@dataclass(frozen=True)
class Transition:
    """One recorded control tick: a row of :class:`EpisodeArrays`."""

    obs: Dict[str, np.ndarray]
    """Observation captured after the tick, batch dimension removed."""
    action: Dict[str, np.ndarray]
    """Command issued on the tick; see :attr:`EpisodeArrays.action`."""
    tick: int
    """0-based control tick of the episode that produced this row."""
    stage_index: int
    stage_name: str
    phase: Optional[str]
    phase_step: Optional[int]
    sim_time: float
    """Simulator time of ``obs`` (ns with ``env.stamp_ns``, else s)."""


@dataclass(frozen=True)
class EpisodeArrays:
    """Columnar episode rows: every column has the same leading length ``T``.

    Row ``t`` pairs the command of control tick ``tick[t]`` with the labels
    it was issued under and the observation that followed it:

    - ``action``: the command issued on the tick. Environment command
      channels (observation keys with an ``action/`` segment) keep their
      observation key; a numeric policy return value is stored under
      ``policy`` (``policy/<key>`` for a dict).
    - ``stage_index`` ... ``phase_step``: the ``TaskUpdate`` the policy acted
      on, i.e. the stage the command belongs to. ``-1`` / ``""`` mean none.
    - ``obs`` and ``sim_time``: the observation captured after the tick.

    A ``(obs_t, action_t)`` sample therefore takes ``obs`` from the previous
    row (``Episode.initial_observation`` for row 0); that shift belongs in
    the training adapter. With ``sample_stride > 1`` rows skip ticks, which
    ``tick`` makes explicit; the last tick is always recorded.
    """

    obs: Dict[str, np.ndarray]
    action: Dict[str, np.ndarray]
    tick: np.ndarray
    stage_index: np.ndarray
    stage_name: np.ndarray
    phase: np.ndarray
    phase_step: np.ndarray
    sim_time: np.ndarray

    def __post_init__(self) -> None:
        """Check that every column has the same leading length."""
        lengths = {name: len(column) for name, column in self._columns().items()}
        if len(set(lengths.values())) > 1:
            detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(f"EpisodeArrays columns differ in length: {detail}")

    def __len__(self) -> int:
        return len(self.tick)

    @overload
    def __getitem__(self, index: int) -> Transition: ...

    @overload
    def __getitem__(self, index: slice) -> "EpisodeArrays": ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Transition, "EpisodeArrays"]:
        """A :class:`Transition` for an integer, a view for a slice."""
        if isinstance(index, slice):
            return EpisodeArrays(
                obs={key: value[index] for key, value in self.obs.items()},
                action={key: value[index] for key, value in self.action.items()},
                **{name: getattr(self, name)[index] for name in _LABEL_COLUMNS},
            )
        row = range(len(self))[index]
        phase = str(self.phase[row])
        phase_step = int(self.phase_step[row])
        return Transition(
            obs={key: value[row] for key, value in self.obs.items()},
            action={key: value[row] for key, value in self.action.items()},
            tick=int(self.tick[row]),
            stage_index=int(self.stage_index[row]),
            stage_name=str(self.stage_name[row]),
            phase=phase or None,
            phase_step=None if phase_step < 0 else phase_step,
            sim_time=float(self.sim_time[row]),
        )

    def __iter__(self) -> Iterator[Transition]:
        for row in range(len(self)):
            yield self[row]

    def window(self, length: int, stride: int) -> Iterator["EpisodeArrays"]:
        """Views of ``length`` consecutive rows, starting every ``stride`` rows.

        Only full windows are yielded, so an episode shorter than ``length``
        yields none.
        """
        if length < 1 or stride < 1:
            raise ValueError(
                f"window length and stride must be positive; got {length}, {stride}"
            )
        for start in range(0, len(self) - length + 1, stride):
            yield self[start : start + length]

    def _columns(self) -> Dict[str, Any]:
        columns: Dict[str, Any] = {
            f"obs[{key!r}]": value for key, value in self.obs.items()
        }
        columns.update(
            {f"action[{key!r}]": value for key, value in self.action.items()}
        )
        columns.update({name: getattr(self, name) for name in _LABEL_COLUMNS})
        return columns


_LABEL_COLUMNS = tuple(
    item.name for item in fields(EpisodeArrays) if item.name not in ("obs", "action")
)


@dataclass(frozen=True)
class Episode:
    """One rollout from a reset to its end, with what made it and how it went.

    ``done`` and ``success`` stay independent: a task can finish and fail.
    ``truncated`` means the stream's ``max_updates`` ran out first; it is a
    host-side limit, so a truncated episode has neither succeeded nor failed
    on its own terms.
    """

    episode_index: int
    seed: int
    """Run seed the episode was drawn from (``task.seed``)."""
    task: str
    success: bool
    truncated: bool
    failure_reason: Optional[str]
    transitions: EpisodeArrays
    initial_observation: Dict[str, np.ndarray]
    """Observation captured right after the reset, batch dimension removed."""
    scene: Dict[str, Any]
    """Ground truth after the reset: object, operator, and camera poses."""
    randomization: Dict[str, Any]
    """Sampled reset poses and the backend's reset diagnostics."""
    records: List[ExecutionRecord]
    metadata: Dict[str, Any]
    """The stream config dump, reset address, worker, slot, timings, and the
    discarded attempts of this ``episode_index``."""

    def __len__(self) -> int:
        return len(self.transitions)

    def window(self, length: int, stride: int) -> Iterator[EpisodeArrays]:
        """See :meth:`EpisodeArrays.window`."""
        return self.transitions.window(length, stride)
