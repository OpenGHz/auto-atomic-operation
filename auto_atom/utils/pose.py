"""Pose utilities built on top of the bundled transformation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..framework import Orientation, PoseControlConfig, Position, Rotation
from .transformations import (
    concatenate_matrices,
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
    translation_matrix,
)


@dataclass
class PoseState:
    """Concrete world pose used by the runtime for frame conversions."""

    position: Position = (0.0, 0.0, 0.0)
    # Quaternion convention is always xyzw.
    orientation: Orientation = (0.0, 0.0, 0.0, 1.0)


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
    """Compose two poses."""
    matrix = concatenate_matrices(as_matrix(parent), as_matrix(child))
    return pose_state_from_matrix(matrix)


def inverse_pose(pose: PoseState) -> PoseState:
    """Invert a pose."""
    inv_orientation = normalize_quaternion(tuple(quaternion_inverse(pose.orientation)))
    inv_rotation = quaternion_matrix(inv_orientation)[:3, :3]
    inv_translation = -inv_rotation.dot(np.asarray(pose.position, dtype=np.float64))
    return PoseState(
        position=tuple(float(v) for v in inv_translation),
        orientation=inv_orientation,
    )


def euler_to_quaternion(rotation: Rotation) -> Orientation:
    """Convert rpy Euler angles to a normalized xyzw quaternion."""
    quat = quaternion_from_euler(*rotation, axes="sxyz")
    return normalize_quaternion(tuple(float(v) for v in quat))


def rotate_vector(quat: Orientation, vec: Tuple[float, float, float]) -> Position:
    """Rotate a vector by a quaternion."""
    rotation = quaternion_matrix(normalize_quaternion(quat))[:3, :3]
    rotated = rotation.dot(np.asarray(vec, dtype=np.float64))
    return tuple(float(v) for v in rotated)


def normalize_quaternion(quat: Orientation) -> Orientation:
    """Normalize a quaternion and fall back to identity if zero-length."""
    array = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(array)
    if norm == 0.0:
        return (0.0, 0.0, 0.0, 1.0)
    normalized = array / norm
    return tuple(float(v) for v in normalized)


def multiply_quaternions(a: Orientation, b: Orientation) -> Orientation:
    """Multiply two quaternions and normalize the result."""
    quat = quaternion_multiply(a, b)
    return normalize_quaternion(tuple(float(v) for v in quat))


def as_matrix(pose: PoseState) -> np.ndarray:
    """Convert a pose state into a homogeneous transform matrix."""
    matrix = quaternion_matrix(normalize_quaternion(pose.orientation))
    matrix = concatenate_matrices(translation_matrix(pose.position), matrix)
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
