"""Pose utilities built on top of the bundled transformation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

from ..framework import Orientation, PoseControlConfig, Position, Rotation
from .transformations import (
    concatenate_matrices,
    euler_from_matrix,
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
    translation_matrix,
)


def _as_batched_vector(
    value: Iterable[float] | np.ndarray,
    *,
    width: int,
    default: tuple[float, ...],
) -> np.ndarray:
    raw = default if value is None else value
    arr = np.asarray(raw, dtype=np.float64)
    if arr.size == 0:
        arr = np.asarray(default, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != width:
            raise ValueError(f"Expected shape ({width},), got {arr.shape}")
        arr = arr.reshape(1, width)
    elif arr.ndim == 2:
        if arr.shape[1] != width:
            raise ValueError(f"Expected shape (B, {width}), got {arr.shape}")
    else:
        raise ValueError(f"Expected rank-1 or rank-2 array, got {arr.ndim}")
    return arr


@dataclass
class PoseState:
    """Concrete pose batch used by the runtime for frame conversions."""

    position: np.ndarray | Iterable[float] = (0.0, 0.0, 0.0)
    """Cartesian positions with shape ``(B, 3)``."""
    orientation: np.ndarray | Iterable[float] = (0.0, 0.0, 0.0, 1.0)
    """Quaternion orientations with shape ``(B, 4)`` in ``xyzw`` order."""

    def __post_init__(self) -> None:
        self.position = _as_batched_vector(
            self.position,
            width=3,
            default=(0.0, 0.0, 0.0),
        )
        self.orientation = _as_batched_vector(
            self.orientation,
            width=4,
            default=(0.0, 0.0, 0.0, 1.0),
        )
        if self.position.shape[0] != self.orientation.shape[0]:
            raise ValueError(
                "position and orientation must have the same batch dimension, got "
                f"{self.position.shape[0]} and {self.orientation.shape[0]}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.position.shape[0])

    def select(self, env_index: int) -> "PoseState":
        return PoseState(
            position=self.position[env_index],
            orientation=self.orientation[env_index],
        )

    def broadcast_to(self, batch_size: int) -> "PoseState":
        if self.batch_size == batch_size:
            return self
        if self.batch_size != 1:
            raise ValueError(
                f"Cannot broadcast pose batch of size {self.batch_size} to {batch_size}"
            )
        return PoseState(
            position=np.repeat(self.position, batch_size, axis=0),
            orientation=np.repeat(self.orientation, batch_size, axis=0),
        )

    @classmethod
    def stack(cls, poses: Iterable["PoseState"]) -> "PoseState":
        pose_list = list(poses)
        if not pose_list:
            return cls()
        return cls(
            position=np.concatenate([pose.position for pose in pose_list], axis=0),
            orientation=np.concatenate(
                [pose.orientation for pose in pose_list], axis=0
            ),
        )


def pose_config_to_pose_state(pose: PoseControlConfig) -> PoseState:
    """Convert a pose config into a concrete pose state."""
    position = pose.position if pose.position else (0.0, 0.0, 0.0)
    orientation = resolve_orientation(pose)
    return PoseState(position=position, orientation=orientation)


def resolve_orientation(pose: PoseControlConfig) -> Orientation:
    """Resolve quaternion orientation from explicit xyzw quaternion or rpy Euler angles."""
    if pose.orientation:
        return normalize_quaternion(pose.orientation)
    if pose.rotation:
        return euler_to_quaternion(pose.rotation)
    return (0.0, 0.0, 0.0, 1.0)


def compose_pose(parent: PoseState, child: PoseState) -> PoseState:
    """Compose two pose batches."""
    batch = max(parent.batch_size, child.batch_size)
    parent = parent.broadcast_to(batch)
    child = child.broadcast_to(batch)
    matrices = [
        concatenate_matrices(as_matrix(parent.select(i)), as_matrix(child.select(i)))
        for i in range(batch)
    ]
    return PoseState.stack([pose_state_from_matrix(m) for m in matrices])


def inverse_pose(pose: PoseState) -> PoseState:
    """Invert a pose batch."""
    results = []
    for i in range(pose.batch_size):
        single = pose.select(i)
        inv_orientation = normalize_quaternion(
            tuple(quaternion_inverse(single.orientation[0]))
        )
        inv_rotation = quaternion_matrix(inv_orientation)[:3, :3]
        inv_translation = -inv_rotation.dot(single.position[0])
        results.append(
            PoseState(
                position=tuple(float(v) for v in inv_translation),
                orientation=inv_orientation,
            )
        )
    return PoseState.stack(results)


def euler_to_quaternion(rotation: Rotation) -> Orientation:
    """Convert rpy Euler angles to a normalized xyzw quaternion."""
    quat = quaternion_from_euler(*rotation, axes="sxyz")
    return normalize_quaternion(tuple(float(v) for v in quat))


def quaternion_to_rpy(quat: Orientation | np.ndarray) -> Rotation:
    """Convert an xyzw quaternion to rpy Euler angles."""
    matrix = quaternion_matrix(normalize_quaternion(quat))
    rpy = euler_from_matrix(matrix, axes="sxyz")
    return tuple(float(v) for v in rpy)


def rotate_vector(
    quat: Orientation | np.ndarray,
    vec: Tuple[float, float, float] | np.ndarray,
) -> Position:
    """Rotate a vector by a quaternion."""
    rotation = quaternion_matrix(normalize_quaternion(quat))[:3, :3]
    rotated = rotation.dot(np.asarray(vec, dtype=np.float64))
    return tuple(float(v) for v in rotated)


def rotate_pose_around_axis(
    pose: PoseState,
    pivot: Position,
    axis: Position,
    angle: float,
) -> PoseState:
    """Rotate a single-env pose around an axis passing through a pivot point."""
    if pose.batch_size != 1:
        raise ValueError("rotate_pose_around_axis expects a single-env PoseState")
    pivot_np = np.asarray(pivot, dtype=np.float64)
    axis_np = np.asarray(axis, dtype=np.float64)
    axis_np = axis_np / np.linalg.norm(axis_np)
    pos_np = np.asarray(pose.position[0], dtype=np.float64)

    half = angle / 2.0
    sin_half = np.sin(half)
    cos_half = np.cos(half)
    rot_quat: Orientation = (
        float(axis_np[0] * sin_half),
        float(axis_np[1] * sin_half),
        float(axis_np[2] * sin_half),
        float(cos_half),
    )
    rot_matrix = quaternion_matrix(rot_quat)[:3, :3]

    offset = pos_np - pivot_np
    new_pos = pivot_np + rot_matrix.dot(offset)
    new_position: Position = tuple(float(v) for v in new_pos)
    new_orientation = normalize_quaternion(
        tuple(float(v) for v in quaternion_multiply(rot_quat, pose.orientation[0]))
    )
    return PoseState(position=new_position, orientation=new_orientation)


def normalize_quaternion(quat: Orientation | np.ndarray) -> Orientation:
    """Normalize a quaternion and fall back to identity if zero-length."""
    array = np.asarray(quat, dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    normalized = array / norm
    return tuple(float(v) for v in normalized)


def multiply_quaternions(
    a: Orientation | np.ndarray,
    b: Orientation | np.ndarray,
) -> Orientation:
    """Multiply two quaternions and normalize the result."""
    quat = quaternion_multiply(a, b)
    return normalize_quaternion(tuple(float(v) for v in quat))


def as_matrix(pose: PoseState) -> np.ndarray:
    """Convert a single-env pose state into a homogeneous transform matrix."""
    if pose.batch_size != 1:
        raise ValueError("as_matrix expects a single-env PoseState")
    matrix = quaternion_matrix(normalize_quaternion(pose.orientation[0]))
    matrix = concatenate_matrices(translation_matrix(pose.position[0]), matrix)
    return matrix


def pose_state_from_matrix(matrix: np.ndarray) -> PoseState:
    """Convert a homogeneous transform matrix into a pose state."""
    position = tuple(float(v) for v in matrix[:3, 3])
    rotation = matrix[:3, :3]
    quat = quaternion_from_matrix_3x3(rotation)
    return PoseState(position=position, orientation=quat)


def quaternion_from_matrix_3x3(matrix: np.ndarray) -> Orientation:
    """Convert a 3x3 rotation matrix into a normalized quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion((x, y, z, w))


def quaternion_to_rotation_matrix(quat: Orientation | np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix for the given xyzw quaternion."""
    return quaternion_matrix(normalize_quaternion(quat))[:3, :3]


def quaternion_angular_distance(
    q1: Orientation | np.ndarray,
    q2: Orientation | np.ndarray,
) -> float:
    """Return angular distance between two xyzw quaternions in radians."""
    dot = abs(
        float(
            np.dot(
                np.asarray(q1, dtype=np.float64).reshape(-1),
                np.asarray(q2, dtype=np.float64).reshape(-1),
            )
        )
    )
    dot = min(1.0, dot)
    return 2.0 * np.arccos(dot)


def mujoco_euler_to_quaternion(ax: float, ay: float, az: float) -> Orientation:
    """Convert MuJoCo intrinsic XYZ euler angles (radians) to xyzw quaternion."""
    quat = quaternion_from_euler(ax, ay, az, axes="rxyz")
    return normalize_quaternion(tuple(float(v) for v in quat))
