"""Canonical recordings and lazy source adapters for data replay.

The replay runner still owns simulator state and control ticks.  This module
owns the recording seam: source-specific readers produce a validated trajectory,
and a timeline exposes one action at a time without eagerly building action
dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping, Optional, Sequence

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


class ReplayTimeline:
    """Lazy frame cursor over a canonical trajectory."""

    def __init__(self, trajectory: ReplayTrajectory) -> None:
        self.trajectory = trajectory
        self._demo = trajectory.to_demo_dict()
        self._step = 0

    @property
    def num_steps(self) -> int:
        return self.trajectory.frame_count

    @property
    def remaining_steps(self) -> int:
        return max(self.num_steps - self._step, 0)

    def current_log_time_ns(self) -> int | None:
        if self.trajectory.timestamps_ns is None or self._step <= 0:
            return None
        index = min(self._step - 1, self.num_steps - 1)
        return int(self.trajectory.timestamps_ns[index])

    def reset(self, start_step: int = 0) -> None:
        self._step = max(0, int(start_step))

    def _action_at(self, index: int) -> dict[str, Any]:
        return self.trajectory.action_at(index)

    def apply_first_frame_as_reset(self) -> dict[str, Any] | None:
        if self.num_steps <= 0:
            return None
        action = self._action_at(0)
        self._step = min(1, self.num_steps)
        return action

    def act(self) -> dict[str, Any]:
        action = self._action_at(self._step)
        self._step += 1
        return action


class ReplayPolicy(ReplayTimeline):
    """Compatibility timeline accepting the legacy ``(demo, mode)`` shape."""

    def __init__(
        self,
        demo: ReplayTrajectory | Mapping[str, Any],
        mode: ReplayMode | None = None,
    ) -> None:
        trajectory = (
            demo
            if isinstance(demo, ReplayTrajectory)
            else ReplayTrajectory.from_demo_dict(demo, mode or _infer_mode(demo))
        )
        super().__init__(trajectory)
        self._mode = trajectory.mode
        self._times = trajectory.timestamps_ns
        self._max = trajectory.frame_count - 1


def _infer_mode(demo: Mapping[str, Any]) -> ReplayMode:
    if "position" in demo:
        return "pose"
    if "joint" in demo:
        return "joint"
    if "ctrl" in demo:
        return "ctrl"
    raise ValueError("Cannot infer replay mode: expected position, joint, or ctrl.")


def normalize_demo_for_batch(
    demo: Mapping[str, Any],
    batch_size: int,
    mode: ReplayMode,
) -> dict[str, Any]:
    """Compatibility wrapper around :meth:`ReplayTrajectory.normalize_for_batch`."""
    return (
        ReplayTrajectory.from_demo_dict(demo, mode)
        .normalize_for_batch(batch_size)
        .to_demo_dict()
    )


def subsample_demo(demo: Mapping[str, Any], stride: int) -> dict[str, Any]:
    """Compatibility wrapper around :meth:`ReplayTrajectory.subsample`."""
    trajectory = ReplayTrajectory.from_demo_dict(demo, _infer_mode(demo))
    if stride <= 1:
        return dict(demo)
    result = trajectory.subsample(stride).to_demo_dict()
    for key, value in demo.items():
        result.setdefault(key, value)
    return result


class NpzRecordingAdapter:
    """Load only the low-dimensional channels required by the selected mode."""

    def load(
        self,
        source: str | Path | np.lib.npyio.NpzFile,
        mode: Literal["pose", "ctrl"],
    ) -> ReplayTrajectory:
        owns_file = not isinstance(source, np.lib.npyio.NpzFile)
        data = np.load(source) if owns_file else source
        try:
            low_dim_keys = self._load_low_dim_keys(data)

            def get(key: str) -> np.ndarray:
                if key not in low_dim_keys:
                    raise KeyError(f"NPZ is missing low-dimensional channel '{key}'.")
                index = low_dim_keys.index(key)
                data_key = f"low_dim_data__{index}"
                if data_key not in data:
                    raise KeyError(
                        f"NPZ missing '{data_key}' for low-dimensional key '{key}'."
                    )
                return np.asarray(data[data_key], dtype=np.float32)

            if mode == "pose":
                arrays: dict[str, np.ndarray] = {
                    "position": get("action/arm/pose/position"),
                    "orientation": get("action/arm/pose/orientation"),
                }
                gripper_key = (
                    "action/gripper/joint_state/position"
                    if "action/gripper/joint_state/position" in low_dim_keys
                    else "action/eef/joint_state/position"
                )
                if gripper_key in low_dim_keys:
                    arrays["gripper"] = get(gripper_key)
            else:
                arm = get("action/arm/joint_state/position")
                eef_key = "action/eef/joint_state/position"
                arrays = {
                    "ctrl": (
                        np.concatenate([arm, get(eef_key)], axis=-1)
                        if eef_key in low_dim_keys
                        else arm
                    )
                }
            self._attach_optional_base_pose(
                {
                    key: get(key)
                    for key in low_dim_keys
                    if "base_pose" in key or key.startswith("action/base/pose/")
                },
                arrays,
            )
            return ReplayTrajectory(mode=mode, arrays=arrays)
        finally:
            if owns_file:
                data.close()

    @staticmethod
    def _load_low_dim_map(
        demo_data: np.lib.npyio.NpzFile,
    ) -> dict[str, np.ndarray]:
        if "low_dim_keys" not in demo_data:
            raise KeyError("NPZ missing 'low_dim_keys'.")
        keys = [str(key) for key in np.asarray(demo_data["low_dim_keys"])]
        result: dict[str, np.ndarray] = {}
        for index, key in enumerate(keys):
            data_key = f"low_dim_data__{index}"
            if data_key not in demo_data:
                raise KeyError(
                    f"NPZ missing '{data_key}' for low-dimensional key '{key}'."
                )
            # Accessing only requested keys keeps unrelated image channels out of
            # the canonical trajectory; NpzFile loads each selected array lazily.
            result[key] = np.asarray(demo_data[data_key], dtype=np.float32)
        return result

    @staticmethod
    def _load_low_dim_keys(demo_data: np.lib.npyio.NpzFile) -> list[str]:
        if "low_dim_keys" not in demo_data:
            raise KeyError("NPZ missing 'low_dim_keys'.")
        return [str(key) for key in np.asarray(demo_data["low_dim_keys"])]

    @staticmethod
    def _attach_optional_base_pose(
        low_dim: Mapping[str, np.ndarray],
        result: dict[str, np.ndarray],
        *,
        operator_name: str = "arm",
    ) -> None:
        candidates = (
            (
                f"action/{operator_name}/base_pose/position",
                f"action/{operator_name}/base_pose/orientation",
            ),
            ("action/base/pose/position", "action/base/pose/orientation"),
        )
        for position_key, orientation_key in candidates:
            has_position = position_key in low_dim
            has_orientation = orientation_key in low_dim
            if not has_position and not has_orientation:
                continue
            if has_position != has_orientation:
                missing_key = orientation_key if has_position else position_key
                raise KeyError(
                    "Base pose replay channels must provide both position and "
                    f"orientation; missing '{missing_key}'."
                )
            result["base_position"] = low_dim[position_key]
            result["base_orientation"] = low_dim[orientation_key]
            return


class Ros2McapRecordingAdapter:
    """Adapter for ROS2 CDR MCAP recordings.

    The reader remains iterator-driven; the selected aligned channels are
    materialized only because replay needs random frame access.
    """

    def load(
        self,
        mcap_path: str,
        arm_topic: str,
        gripper_topic: str,
        base_topic: str | None = None,
        scene_joint_topic: str | None = None,
    ) -> ReplayTrajectory:
        from . import data_replay

        legacy = data_replay._load_mcap_demo_ros2(
            mcap_path,
            arm_topic,
            gripper_topic,
            base_topic,
            scene_joint_topic,
        )
        return ReplayTrajectory.from_mcap_demo(legacy)


class FoxgloveMcapRecordingAdapter:
    """Adapter for Foxglove flatbuffer MCAP recordings."""

    def load(
        self,
        mcap_path: str,
        arm_topic: str | None,
        gripper_topic: str | None,
        base_topic: str | None = None,
        scene_joint_topic: str | None = None,
        *,
        arm_actuator_names: list[str] | None = None,
        eef_actuator_names: list[str] | None = None,
    ) -> ReplayTrajectory:
        from . import data_replay

        metadata = data_replay._read_mcap_channel_meta(mcap_path)
        legacy = data_replay._load_mcap_demo_foxglove(
            mcap_path,
            arm_topic,
            gripper_topic,
            base_topic,
            scene_joint_topic,
            metadata,
            arm_actuator_names=arm_actuator_names,
            eef_actuator_names=eef_actuator_names,
        )
        return ReplayTrajectory.from_mcap_demo(legacy)


def load_mcap_recording(
    mcap_path: str,
    arm_topic: str | None,
    gripper_topic: str | None,
    base_topic: str | None = None,
    scene_joint_topic: str | None = None,
    *,
    arm_actuator_names: list[str] | None = None,
    eef_actuator_names: list[str] | None = None,
) -> ReplayTrajectory:
    """Auto-select the ROS2 or Foxglove recording adapter."""
    from . import data_replay

    metadata = data_replay._read_mcap_channel_meta(mcap_path)
    if data_replay._mcap_is_foxglove(metadata):
        return FoxgloveMcapRecordingAdapter().load(
            mcap_path,
            arm_topic,
            gripper_topic,
            base_topic,
            scene_joint_topic,
            arm_actuator_names=arm_actuator_names,
            eef_actuator_names=eef_actuator_names,
        )
    return Ros2McapRecordingAdapter().load(
        mcap_path,
        arm_topic or "",
        gripper_topic or "",
        base_topic,
        scene_joint_topic,
    )


def prepare_joint_trajectory(
    trajectory: ReplayTrajectory,
    actuator_names: Sequence[str],
    *,
    joint_name_mapping: Mapping[str, str] | None = None,
    joint_axis_scale: Sequence[float] = (),
    joint_clip: Mapping[str, Any] | None = None,
) -> ReplayTrajectory:
    """Apply actuator order, scale, and clip to an owned joint trajectory."""
    if trajectory.mode not in {"joint", "ctrl"}:
        return trajectory
    if trajectory.mode == "joint" and actuator_names:
        trajectory.align_to_actuators(actuator_names, joint_name_mapping)
    if joint_axis_scale:
        scale = np.asarray(joint_axis_scale, dtype=trajectory.arrays["joint"].dtype)
        data = trajectory.arrays["joint"]
        if data.shape[-1] < scale.shape[0]:
            raise ValueError(
                f"joint_axis_scale has length {scale.shape[0]}, but joint data only "
                f"has {data.shape[-1]} column(s)."
            )
        data[..., : scale.shape[0]] *= scale
    if joint_clip:
        data = trajectory.arrays["joint"]
        for index, name in enumerate(trajectory.joint_names):
            bounds = joint_clip.get(name)
            if bounds is None:
                continue
            lower = getattr(bounds, "min", None)
            upper = getattr(bounds, "max", None)
            if isinstance(bounds, Mapping):
                lower = bounds.get("min", lower)
                upper = bounds.get("max", upper)
            if lower is not None:
                data[..., index] = np.maximum(data[..., index], lower)
            if upper is not None:
                data[..., index] = np.minimum(data[..., index], upper)
    return trajectory
