"""Canonical replay trajectory model independent of runner adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Sequence

import numpy as np

ReplayMode = Literal["pose", "ctrl", "joint"]

_REQUIRED_KEYS: dict[ReplayMode, tuple[str, ...]] = {
    "pose": ("position", "orientation"),
    "ctrl": ("ctrl",),
    "joint": ("joint",),
}
_OPTIONAL_ARRAY_KEYS = (
    "gripper",
    "base_position",
    "base_orientation",
    "scene_joint",
)


def _as_frame_array(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError(
            f"{label} must have at least two dimensions (T, ...), got {array.shape}."
        )
    return array


@dataclass
class ReplayTrajectory:
    """Validated time-series channels produced by a replay source.

    Arrays are owned by the caller at construction time.  Preparation methods
    may reuse views or replace only the channel they must transform; the runner
    treats the resulting trajectory as read-only while it emits actions.
    """

    mode: ReplayMode
    arrays: dict[str, np.ndarray]
    joint_names: tuple[str, ...] = ()
    scene_joint_names: tuple[str, ...] = ()
    timestamps_ns: np.ndarray | None = None
    _frame_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in _REQUIRED_KEYS:
            raise ValueError(
                f"Unknown replay mode {self.mode!r}; expected 'pose', 'ctrl', or 'joint'."
            )

        normalized: dict[str, np.ndarray] = {}
        for key, value in self.arrays.items():
            if value is None:
                continue
            normalized[key] = _as_frame_array(value, label=key)
        self.arrays = normalized

        missing = [key for key in _REQUIRED_KEYS[self.mode] if key not in self.arrays]
        if missing:
            raise KeyError(
                f"Replay mode {self.mode!r} is missing required channel(s): {missing}."
            )

        lengths = {key: int(value.shape[0]) for key, value in self.arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Replay channels must share frame count; got {lengths}.")
        self._frame_count = next(iter(lengths.values()))

        has_base_position = "base_position" in self.arrays
        has_base_orientation = "base_orientation" in self.arrays
        if has_base_position != has_base_orientation:
            missing_key = "base_orientation" if has_base_position else "base_position"
            raise KeyError(
                "Base pose replay channels must provide both base_position and "
                f"base_orientation; missing '{missing_key}'."
            )

        has_scene = "scene_joint" in self.arrays
        if has_scene != bool(self.scene_joint_names):
            missing_key = "scene_joint_names" if has_scene else "scene_joint"
            raise KeyError(
                "Scene joint replay channels must provide both scene_joint and "
                f"scene_joint_names; missing '{missing_key}'."
            )
        if has_scene and self.arrays["scene_joint"].shape[-1] != len(
            self.scene_joint_names
        ):
            raise ValueError(
                "scene_joint width does not match scene_joint_names: "
                f"{self.arrays['scene_joint'].shape[-1]} vs {len(self.scene_joint_names)}"
            )

        if self.timestamps_ns is not None:
            times = np.asarray(self.timestamps_ns, dtype=np.int64).reshape(-1)
            if times.shape != (self._frame_count,):
                raise ValueError(
                    "timestamps_ns must have shape (T,), "
                    f"got {times.shape} for T={self._frame_count}."
                )
            if times.size and int(times[0]) != 0:
                raise ValueError(
                    "timestamps_ns must be zero-anchored at the first frame."
                )
            if times.size > 1 and np.any(np.diff(times) < 0):
                raise ValueError("timestamps_ns must be monotonically non-decreasing.")
            self.timestamps_ns = times

        self.joint_names = tuple(str(name) for name in self.joint_names)
        self.scene_joint_names = tuple(str(name) for name in self.scene_joint_names)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def action_key(self) -> str:
        return _REQUIRED_KEYS[self.mode][0]

    @classmethod
    def from_demo_dict(
        cls,
        demo: Mapping[str, Any],
        mode: ReplayMode,
    ) -> "ReplayTrajectory":
        arrays = {
            key: demo[key]
            for key in (*_REQUIRED_KEYS[mode], *_OPTIONAL_ARRAY_KEYS)
            if key in demo and demo[key] is not None
        }
        return cls(
            mode=mode,
            arrays=arrays,
            joint_names=tuple(demo.get("joint_names", ()) or ()),
            scene_joint_names=tuple(demo.get("scene_joint_names", ()) or ()),
            timestamps_ns=demo.get("joint_times"),
        )

    @classmethod
    def from_mcap_demo(cls, demo: Any) -> "ReplayTrajectory":
        """Convert the legacy ``McapDemo`` container without importing it."""
        arrays: dict[str, np.ndarray] = {"joint": demo.joint}
        for key in _OPTIONAL_ARRAY_KEYS:
            value = getattr(demo, key, None)
            if value is not None:
                arrays[key] = value
        return cls(
            mode="joint",
            arrays=arrays,
            joint_names=tuple(getattr(demo, "joint_names", ()) or ()),
            scene_joint_names=tuple(getattr(demo, "scene_joint_names", ()) or ()),
            timestamps_ns=getattr(demo, "joint_times", None),
        )

    def to_demo_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = dict(self.arrays)
        if self.joint_names:
            result["joint_names"] = list(self.joint_names)
        if self.scene_joint_names:
            result["scene_joint_names"] = list(self.scene_joint_names)
        if self.timestamps_ns is not None:
            result["joint_times"] = self.timestamps_ns
        return result

    def first_frame_joint_positions(self) -> Dict[str, float]:
        if "joint" not in self.arrays:
            raise ValueError("first_frame_joint_positions requires joint-mode data.")
        values = self.arrays["joint"][0]
        if values.ndim > 1:
            values = values[0]
        result = {
            name: float(value)
            for name, value in zip(self.joint_names, np.asarray(values).reshape(-1))
        }
        if "scene_joint" in self.arrays:
            scene_values = self.arrays["scene_joint"][0]
            if scene_values.ndim > 1:
                scene_values = scene_values[0]
            result.update(
                {
                    name: float(value)
                    for name, value in zip(
                        self.scene_joint_names,
                        np.asarray(scene_values).reshape(-1),
                    )
                }
            )
        return result

    def align_to_actuators(
        self,
        actuator_names: Sequence[str],
        name_mapping: Mapping[str, str] | None = None,
    ) -> "ReplayTrajectory":
        """Reorder the owned joint channel in place and return ``self``."""
        if "joint" not in self.arrays:
            raise ValueError("Actuator alignment requires joint-mode data.")
        mapping = dict(name_mapping or {})
        mapped_names = [mapping.get(name, name) for name in self.joint_names]
        reorder: list[int] = []
        for actuator_name in actuator_names:
            if actuator_name not in mapped_names:
                raise ValueError(
                    f"Actuator '{actuator_name}' not found in recorded joint names "
                    f"{list(self.joint_names)} (after mapping: {mapped_names})"
                )
            reorder.append(mapped_names.index(actuator_name))
        self.arrays["joint"] = self.arrays["joint"][..., reorder]
        self.joint_names = tuple(str(name) for name in actuator_names)
        return self

    def normalize_for_batch(self, batch_size: int) -> "ReplayTrajectory":
        """Adapt optional recorded batch axes, preferring NumPy views."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        normalized: dict[str, np.ndarray] = {}
        for key, array in self.arrays.items():
            if array.ndim == 2:
                normalized[key] = array
                continue
            if array.ndim < 3:
                raise ValueError(
                    f"{key} must have shape (T, dim) or (T, B, dim), got {array.shape}."
                )
            recorded_batch = array.shape[1]
            if batch_size > recorded_batch:
                raise ValueError(
                    f"{key} recorded with batch_size={recorded_batch}, "
                    f"but replay requires batch_size={batch_size}."
                )
            selected = array[:, :batch_size, ...]
            normalized[key] = selected[:, 0, ...] if batch_size == 1 else selected

        return ReplayTrajectory(
            mode=self.mode,
            arrays=normalized,
            joint_names=self.joint_names,
            scene_joint_names=self.scene_joint_names,
            timestamps_ns=self.timestamps_ns,
        )

    def subsample(self, stride: int) -> "ReplayTrajectory":
        """Keep every ``stride``-th frame and always retain the final frame."""
        if stride <= 1 or self.frame_count == 0:
            return self
        indices = np.arange(0, self.frame_count, int(stride), dtype=np.int64)
        if indices[-1] != self.frame_count - 1:
            indices = np.append(indices, self.frame_count - 1)
        arrays = {key: value[indices] for key, value in self.arrays.items()}
        timestamps = None if self.timestamps_ns is None else self.timestamps_ns[indices]
        return ReplayTrajectory(
            mode=self.mode,
            arrays=arrays,
            joint_names=self.joint_names,
            scene_joint_names=self.scene_joint_names,
            timestamps_ns=timestamps,
        )

    def action_at(self, index: int) -> dict[str, Any]:
        if self.frame_count <= 0:
            raise IndexError("Cannot read an action from an empty replay trajectory.")
        frame = min(max(int(index), 0), self.frame_count - 1)
        if self.mode == "pose":
            action: dict[str, Any] = {
                "position": self.arrays["position"][frame],
                "orientation": self.arrays["orientation"][frame],
            }
            if "gripper" in self.arrays:
                action["gripper"] = self.arrays["gripper"][frame]
        else:
            action = {self.action_key: self.arrays[self.action_key][frame]}
        if "base_position" in self.arrays:
            action["base_position"] = self.arrays["base_position"][frame]
            action["base_orientation"] = self.arrays["base_orientation"][frame]
        if "scene_joint" in self.arrays:
            action["scene_joint_positions"] = self.arrays["scene_joint"][frame]
            action["scene_joint_names"] = list(self.scene_joint_names)
        return action
