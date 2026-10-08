"""Turn batched evaluator output into one env's episode rows.

Everything here is pure: it takes the batched observation and ``TaskUpdate``
an evaluator produced and selects one environment row, so the row layout can
be tested without a simulator.
"""

from __future__ import annotations

from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

import numpy as np

from auto_atom.runtime import TaskUpdate

from .records import EpisodeArrays


def is_command_key(key: str) -> bool:
    """Whether an observation key is a command channel (an ``action/`` segment).

    Command keys may carry the environment's key prefix, e.g.
    ``/robot/action/arm/pose``.
    """
    return "/action/" in f"/{key}"


def split_observation(
    observation: Any,
    env_index: int,
    *,
    observation_keys: Optional[Collection[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """One env's measurements, command channels, and simulator time.

    ``observation`` has the ``capture_observation()`` layout,
    ``{key: {"data": batched, "t": batched}}``. ``observation_keys`` limits
    the measurements; command channels are always kept. The time is the
    latest ``t`` of the selected keys, NaN when none carries one.
    """
    if not isinstance(observation, Mapping):
        raise TypeError(
            "A recorded observation must map keys to {'data', 't'} payloads, as "
            f"capture_observation() does; got {type(observation).__name__}. A "
            "policy observation_getter that changes the layout breaks recording: "
            "transform the observation inside the policy instead."
        )
    measurements: Dict[str, Any] = {}
    commands: Dict[str, Any] = {}
    times: List[float] = []
    for key, payload in observation.items():
        command = is_command_key(key)
        if not command and observation_keys is not None and key not in observation_keys:
            continue
        if not isinstance(payload, Mapping) or "data" not in payload:
            raise TypeError(
                f"Observation key {key!r} must hold a {{'data', 't'}} payload; "
                f"got {type(payload).__name__}."
            )
        value = _select_row(payload["data"], env_index)
        (commands if command else measurements)[key] = value
        stamp = payload.get("t")
        if stamp is not None:
            times.append(float(np.asarray(_select_row(stamp, env_index))))
    return measurements, commands, max(times) if times else float("nan")


def missing_observation_keys(
    observation: Mapping[str, Any], observation_keys: Collection[str]
) -> List[str]:
    """Requested keys the environment does not produce."""
    return sorted(set(observation_keys) - set(observation))


class EpisodeRecorder:
    """Accumulate one env's rows from its reset until its episode ends."""

    def __init__(self) -> None:
        self._obs: List[Dict[str, Any]] = []
        self._action: List[Dict[str, Any]] = []
        self._labels: List[Tuple[int, int, str, str, int, float]] = []

    def __len__(self) -> int:
        return len(self._labels)

    def append(
        self,
        *,
        tick: int,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        update: TaskUpdate,
        env_index: int,
        sim_time: float,
    ) -> None:
        """Record tick ``tick``; ``update`` is the one its command was issued under."""
        phase_step = (
            -1 if update.phase_step is None else int(update.phase_step[env_index])
        )
        phase = update.phase[env_index] if update.phase else None
        self._obs.append(obs)
        self._action.append(action)
        self._labels.append(
            (
                int(tick),
                int(update.stage_index[env_index]),
                str(update.stage_name[env_index]),
                phase or "",
                phase_step,
                float(sim_time),
            )
        )

    def arrays(self) -> EpisodeArrays:
        """Stack the rows into columns."""
        ticks, stages, names, phases, phase_steps, times = (
            zip(*self._labels, strict=True) if self._labels else ((),) * 6
        )
        return EpisodeArrays(
            obs=_stack_rows(self._obs),
            action=_stack_rows(self._action),
            tick=np.asarray(ticks, dtype=np.int64),
            stage_index=np.asarray(stages, dtype=np.int64),
            stage_name=np.asarray(names, dtype=str),
            phase=np.asarray(phases, dtype=str),
            phase_step=np.asarray(phase_steps, dtype=np.int64),
            sim_time=np.asarray(times, dtype=np.float64),
        )


def _select_row(value: Any, env_index: int) -> Any:
    """The env's entry of a batched value; scalars are shared by every env."""
    if isinstance(value, (list, tuple)):
        return value[env_index]
    array = np.asarray(value) if not isinstance(value, np.ndarray) else value
    if array.ndim == 0:
        return array
    return array[env_index]


def _stack_rows(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Columns of per-row dicts; every row must have the same keys."""
    if not rows:
        return {}
    keys = list(rows[0])
    for index, row in enumerate(rows):
        if row.keys() != rows[0].keys():
            raise ValueError(
                f"Row {index} has keys {sorted(row)}, the first row {sorted(keys)}."
            )
    return {key: _stack_column([row[key] for row in rows]) for key in keys}


def _stack_column(values: List[Any]) -> np.ndarray:
    """``np.stack`` the values, or an object column when their shapes differ."""
    try:
        return np.stack([np.asarray(value) for value in values])
    except ValueError:
        column = np.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            column[index] = value
        return column
