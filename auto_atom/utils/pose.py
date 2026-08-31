"""Pose utilities built on top of the bundled transformation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np

from ..framework import (
    Orientation,
    PoseAxisConfig,
    PoseControlConfig,
    PoseOrientationConfig,
    PoseOverrideConfig,
    PosePositionConfig,
    PoseReference,
    Position,
    Rotation,
)
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


def resolve_pose_override(
    config: PoseOverrideConfig | list[float] | tuple[float, ...],
    fallback_pose_world: PoseState,
    reference_pose_world: PoseState | None = None,
    reference_poses: Mapping[PoseReference | str, PoseState] | None = None,
) -> PoseState:
    """Resolve one initial-pose override into a world-frame pose.

    ``config`` is deliberately independent of the thing being moved.  The
    caller resolves the pose-, component-, and axis-level references through
    its backend seam, then this function applies the optional local position
    and orientation fields.  Missing fields preserve the fallback pose after
    it is transformed into the relevant reference frame.  This gives objects,
    cameras, and operators one consistent partial-override rule.

    The compact six-value operator form is ``[x, y, z, yaw, pitch, roll]`` (or
    the equivalent tuple) and is interpreted as a complete world-frame pose;
    structured overrides remain the canonical model for partial or referenced
    poses.
    """
    if isinstance(config, (list, tuple)):
        if len(config) != 6:
            raise ValueError(
                "legacy EEF pose override must contain exactly six values: "
                "[x, y, z, yaw, pitch, roll]"
            )
        return PoseState(
            position=tuple(float(value) for value in config[:3]),
            orientation=euler_to_quaternion(
                (float(config[5]), float(config[4]), float(config[3]))
            ),
        )
    if not isinstance(config, PoseOverrideConfig):
        raise TypeError(
            "pose override must be a PoseOverrideConfig or the legacy "
            "[x, y, z, yaw, pitch, roll] sequence"
        )

    global_reference = config.reference
    resolved_references = dict(reference_poses or {})
    if reference_pose_world is not None:
        resolved_references.setdefault(global_reference, reference_pose_world)
    reference = resolved_references.get(global_reference, PoseState())
    fallback_local = compose_pose(inverse_pose(reference), fallback_pose_world)
    position = fallback_local.position[0].copy()
    orientation = fallback_local.orientation[0].copy()

    def _reference_pose(pose_reference: PoseReference | str) -> PoseState:
        if pose_reference in resolved_references:
            return resolved_references[pose_reference]
        if pose_reference == PoseReference.WORLD:
            return PoseState()
        if pose_reference == global_reference:
            return reference
        raise ValueError(
            f"No resolved pose was provided for reference {pose_reference!r}"
        )

    component_position_world: np.ndarray | None = None
    component_orientation_world: np.ndarray | None = None
    position_world_overrides: dict[int, np.ndarray] = {}
    orientation_world_overrides: dict[int, float] = {}

    if config.position is not None:
        if isinstance(config.position, PosePositionConfig):
            component_reference = config.position.reference or global_reference
            component_pose = _reference_pose(component_reference)
            component_fallback = compose_pose(
                inverse_pose(component_pose),
                fallback_pose_world,
            )
            component_position = component_fallback.position[0].copy()
            for axis_index, axis_name in enumerate(("x", "y", "z")):
                axis_value = getattr(config.position, axis_name)
                if axis_value is None:
                    continue
                if isinstance(axis_value, PoseAxisConfig):
                    value = axis_value.value
                    axis_reference = axis_value.reference or component_reference
                else:
                    value = axis_value
                    axis_reference = component_reference
                if axis_reference == component_reference:
                    component_position[axis_index] = float(value)
                else:
                    local_point = np.zeros(3, dtype=np.float64)
                    local_point[axis_index] = float(value)
                    transformed = compose_pose(
                        _reference_pose(axis_reference),
                        PoseState(position=local_point),
                    )
                    position_world_overrides[axis_index] = transformed.position[0]
            component_position_world = compose_pose(
                component_pose,
                PoseState(position=component_position),
            ).position[0]
            for axis_index, transformed in position_world_overrides.items():
                component_position_world[axis_index] = transformed[axis_index]
        else:
            if len(config.position) < 3:
                raise ValueError(
                    "pose override position must contain at least three values"
                )
            position = np.asarray(config.position[:3], dtype=np.float64)
    if config.orientation is not None:
        if isinstance(config.orientation, PoseOrientationConfig):
            component_reference = config.orientation.reference or global_reference
            component_pose = _reference_pose(component_reference)
            component_fallback = compose_pose(
                inverse_pose(component_pose),
                fallback_pose_world,
            )
            local_rpy = list(quaternion_to_rpy(component_fallback.orientation[0]))
            for axis_index, axis_name in enumerate(("roll", "pitch", "yaw")):
                axis_value = getattr(config.orientation, axis_name)
                if axis_value is None:
                    continue
                if isinstance(axis_value, PoseAxisConfig):
                    value = axis_value.value
                    axis_reference = axis_value.reference or component_reference
                else:
                    value = axis_value
                    axis_reference = component_reference
                if axis_reference == component_reference:
                    local_rpy[axis_index] = float(value)
                else:
                    axis_rpy = [0.0, 0.0, 0.0]
                    axis_rpy[axis_index] = float(value)
                    transformed = compose_pose(
                        _reference_pose(axis_reference),
                        PoseState(orientation=euler_to_quaternion(tuple(axis_rpy))),
                    )
                    orientation_world_overrides[axis_index] = quaternion_to_rpy(
                        transformed.orientation[0]
                    )[axis_index]
            component_orientation_world = compose_pose(
                component_pose,
                PoseState(orientation=euler_to_quaternion(tuple(local_rpy))),
            ).orientation[0]
        elif len(config.orientation) == 3:
            orientation = np.asarray(
                euler_to_quaternion(
                    tuple(float(value) for value in config.orientation)
                ),
                dtype=np.float64,
            )
        elif len(config.orientation) == 4:
            orientation = np.asarray(
                normalize_quaternion(
                    tuple(float(value) for value in config.orientation)
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(
                "pose override orientation must contain three RPY values or "
                "four quaternion values"
            )

    resolved = compose_pose(
        reference,
        PoseState(position=position, orientation=orientation),
    )
    if (
        component_position_world is not None
        or component_orientation_world is not None
        or position_world_overrides
        or orientation_world_overrides
    ):
        world_position = (
            resolved.position[0].copy()
            if component_position_world is None
            else component_position_world
        )
        world_orientation = (
            resolved.orientation[0]
            if component_orientation_world is None
            else component_orientation_world
        )
        world_rpy = list(quaternion_to_rpy(world_orientation))
        for axis_index, transformed in position_world_overrides.items():
            world_position[axis_index] = transformed[axis_index]
        for axis_index, value in orientation_world_overrides.items():
            world_rpy[axis_index] = value
        resolved = PoseState(
            position=world_position,
            orientation=euler_to_quaternion(tuple(world_rpy)),
        )
    return resolved


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


def position_within_tolerance(
    pos_diff: np.ndarray, tolerance: Union[float, List[float]]
) -> bool:
    """Check whether a position difference is within tolerance.

    Args:
        pos_diff: 3-element array of position differences (x, y, z).
        tolerance: A scalar (L2-norm threshold) or a 3-element list
            ``[x, y, z]`` for per-axis checking.

    Returns:
        True if the position is within tolerance.
    """
    if isinstance(tolerance, (list, np.ndarray)) and len(tolerance) == 3:
        return bool(np.all(np.abs(pos_diff) <= np.asarray(tolerance, dtype=np.float64)))
    return float(np.linalg.norm(pos_diff)) <= float(tolerance)


def position_within_tolerance_nullable(
    pos_diff: np.ndarray,
    tolerance: Union[float, List[Optional[float]], None],
) -> bool:
    """Check position difference against tolerance, with per-axis null support.

    Args:
        pos_diff: 3-element array of position differences (x, y, z).
        tolerance: ``None`` = always pass. Scalar = L2-norm threshold.
            List ``[x, y, z]`` = per-axis thresholds where ``None`` elements
            mean that axis is unchecked.

    Returns:
        True if the position is within tolerance.
    """
    if tolerance is None:
        return True
    if isinstance(tolerance, (list, np.ndarray)) and len(tolerance) == 3:
        if all(t is None for t in tolerance):
            return True
        for i, tol in enumerate(tolerance):
            if tol is not None and abs(float(pos_diff[i])) > float(tol):
                return False
        return True
    return float(np.linalg.norm(pos_diff)) <= float(tolerance)


def orientation_within_tolerance_nullable(
    q1: np.ndarray,
    q2: np.ndarray,
    tolerance: Union[float, List[Optional[float]], None],
) -> bool:
    """Check orientation difference against tolerance.

    Args:
        q1: Current orientation quaternion (xyzw).
        q2: Target orientation quaternion (xyzw).
        tolerance: ``None`` = always pass. Scalar = quaternion angular
            distance threshold. List ``[roll, pitch, yaw]`` = per-axis
            Euler thresholds where ``None`` elements are unchecked.

    Returns:
        True if the orientation is within tolerance.
    """
    if tolerance is None:
        return True
    if isinstance(tolerance, (list, np.ndarray)) and len(tolerance) == 3:
        if all(t is None for t in tolerance):
            return True
        r1, p1, y1 = quaternion_to_rpy(np.asarray(q1, dtype=np.float64))
        r2, p2, y2 = quaternion_to_rpy(np.asarray(q2, dtype=np.float64))
        diffs = [abs(r1 - r2), abs(p1 - p2), abs(y1 - y2)]
        for diff, tol in zip(diffs, tolerance):
            if tol is not None and diff > float(tol):
                return False
        return True
    return quaternion_angular_distance(q1, q2) <= float(tolerance)
