"""Canonical recordings and lazy source adapters for data replay.

The replay runner still owns simulator state and control ticks.  This module
owns the recording seam: source-specific readers produce a validated trajectory,
and a timeline exposes one action at a time without eagerly building action
dictionaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .mcap_sources import (
    _load_mcap_demo_foxglove,
    _load_mcap_demo_ros2,
    _mcap_is_foxglove,
    _read_mcap_channel_meta,
)
from .replay_model import ReplayMode, ReplayTrajectory


class ReplayTimeline:
    """Lazy frame cursor over a canonical trajectory."""

    def __init__(self, trajectory: ReplayTrajectory) -> None:
        self.trajectory = trajectory
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
        legacy = _load_mcap_demo_ros2(
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
        metadata = _read_mcap_channel_meta(mcap_path)
        legacy = _load_mcap_demo_foxglove(
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
    metadata = _read_mcap_channel_meta(mcap_path)
    if _mcap_is_foxglove(metadata):
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
