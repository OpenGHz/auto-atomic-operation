"""Logical-batch execution policies independent of MuJoCo and rendering."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Iterable, Sequence

import numpy as np


class BatchExecutionMode(str, Enum):
    REPLICATED = "replicated"
    SHARED = "shared"


class BatchExecutionAdapter:
    """Map logical batch rows onto replicated or shared physical environments."""

    def __init__(
        self,
        envs: Sequence[Any],
        batch_size: int,
        mode: BatchExecutionMode = BatchExecutionMode.REPLICATED,
    ) -> None:
        mode = BatchExecutionMode(mode)
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if not envs:
            raise ValueError("Batch execution requires at least one physical env.")
        if mode == BatchExecutionMode.REPLICATED and len(envs) != batch_size:
            raise ValueError(
                "Replicated batch execution requires one physical env per logical "
                f"row; got {len(envs)} env(s) for batch_size={batch_size}."
            )
        self.envs = list(envs)
        self.batch_size = int(batch_size)
        self.mode = mode

    @property
    def physical_envs(self) -> list[Any]:
        unique: list[Any] = []
        seen: set[int] = set()
        for env in self.envs:
            identity = id(env)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(env)
        return unique

    def normalize_mask(self, env_mask: Any = None) -> np.ndarray:
        if env_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape != (self.batch_size,):
            raise ValueError(
                f"env_mask must have shape ({self.batch_size},), got {mask.shape}."
            )
        return mask

    def broadcast_rows(
        self,
        value: Any,
        *,
        label: str,
        dtype: Any = None,
        width: int | None = None,
        allow_scalar: bool = True,
    ) -> np.ndarray:
        rows = np.asarray(value, dtype=dtype)
        if rows.ndim == 0:
            if not allow_scalar:
                raise ValueError(
                    f"{label} expects shape (B, ...), got scalar {rows.shape}."
                )
            rows = np.broadcast_to(rows, (self.batch_size,))
        elif rows.ndim == 1:
            if not allow_scalar:
                raise ValueError(
                    f"{label} expects shape (B, ...), got scalar row {rows.shape}."
                )
            rows = np.broadcast_to(
                rows.reshape((1,) + rows.shape), (self.batch_size,) + rows.shape
            )
        elif rows.ndim < 2 or rows.shape[0] != self.batch_size:
            raise ValueError(
                f"{label} must have leading dimension {self.batch_size}, "
                f"got {rows.shape}."
            )
        if width is not None and rows.shape[-1] != width:
            raise ValueError(
                f"{label} must have final dimension {width}, got {rows.shape}."
            )
        return rows

    def active_indices(self, env_mask: Any = None) -> np.ndarray:
        return np.flatnonzero(self.normalize_mask(env_mask))

    def dispatch(
        self,
        callback: Callable[[Any, int], Any],
        env_mask: Any = None,
    ) -> list[Any]:
        active = self.active_indices(env_mask)
        if active.size == 0:
            return []
        if self.mode == BatchExecutionMode.SHARED:
            return [callback(self.physical_envs[0], 0)]
        return [callback(self.envs[int(index)], int(index)) for index in active]

    def dispatch_rows(
        self,
        callback: Callable[..., Any],
        rows: Iterable[np.ndarray],
        env_mask: Any = None,
    ) -> list[Any]:
        values = tuple(rows)
        for value in values:
            if value.shape[0] != self.batch_size:
                raise ValueError(
                    "All dispatched values must use the adapter's logical batch size."
                )
        return self.dispatch(
            lambda env, index: callback(
                env,
                index,
                *(value[index] for value in values),
            ),
            env_mask,
        )

    def map_rows(
        self,
        callback: Callable[..., Any],
        rows: Iterable[np.ndarray] = (),
    ) -> list[Any]:
        values = tuple(rows)
        for value in values:
            if value.shape[0] != self.batch_size:
                raise ValueError(
                    "All mapped values must use the adapter's logical batch size."
                )
        return [
            callback(
                (
                    self.physical_envs[0]
                    if self.mode == BatchExecutionMode.SHARED
                    else self.envs[index]
                ),
                index,
                *(value[index] for value in values),
            )
            for index in range(self.batch_size)
        ]

    def collect(self, callback: Callable[[Any], Any]) -> list[Any]:
        if self.mode == BatchExecutionMode.SHARED:
            value = callback(self.physical_envs[0])
            return [value for _ in range(self.batch_size)]
        return [callback(env) for env in self.envs]

    def stack_pairs(
        self,
        callback: Callable[[Any], tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = self.collect(callback)
        return (
            np.stack([position for position, _ in values]),
            np.stack([orientation for _, orientation in values]),
        )

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        if self.mode == BatchExecutionMode.SHARED:
            return self.broadcast_observation(
                self.physical_envs[0].capture_observation()
            )
        return self.stack_observations([env.capture_observation() for env in self.envs])

    def stack_observations(
        self,
        observations: Sequence[dict[str, dict[str, Any]]],
    ) -> dict[str, dict[str, Any]]:
        if len(observations) != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} observations, got {len(observations)}."
            )
        keys = set().union(*(observation.keys() for observation in observations))
        batched: dict[str, dict[str, Any]] = {}
        for key in keys:
            items = [
                observation[key] for observation in observations if key in observation
            ]
            if len(items) != self.batch_size:
                raise KeyError(
                    f"Observation key '{key}' missing from some env replicas."
                )
            first_data = items[0]["data"]
            batched[key] = {
                "data": (
                    [item["data"] for item in items]
                    if isinstance(first_data, dict)
                    else np.stack([np.asarray(item["data"]) for item in items], axis=0)
                ),
                "t": np.asarray([item["t"] for item in items]),
            }
        return batched

    def broadcast_observation(
        self,
        observation: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        batched: dict[str, dict[str, Any]] = {}
        for key, entry in observation.items():
            data = entry["data"]
            timestamp = entry["t"]
            if isinstance(data, dict):
                batched_data: Any = [data] * self.batch_size
            else:
                data_array = np.asarray(data)
                batched_data = np.broadcast_to(
                    data_array[None, ...],
                    (self.batch_size,) + data_array.shape,
                ).copy()
            batched[key] = {
                "data": batched_data,
                "t": np.asarray([timestamp] * self.batch_size),
            }
        return batched

    def probe_bool(self, callback: Callable[[Any], Any]) -> np.ndarray:
        if self.mode == BatchExecutionMode.SHARED:
            value = bool(callback(self.physical_envs[0]))
            return np.full(self.batch_size, value, dtype=bool)
        return np.asarray([bool(callback(env)) for env in self.envs], dtype=bool)
