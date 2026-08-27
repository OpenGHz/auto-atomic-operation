"""Backend-independent geometry for partially constrained pose goals.

The task configuration layer describes axes in named coordinate frames, while
motion backends consume concrete world-frame orientations.  This module keeps
the geometry between those layers independent of a simulator or controller.
All quaternions use the AAO ``xyzw`` convention.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from .utils.pose import (
    PoseState,
    multiply_quaternions,
    normalize_quaternion,
    quaternion_to_rotation_matrix,
)

AxisAlignmentDirectionLike = Literal["same", "opposite", "either"]

_AXIS_NORM_EPS = 1e-12
_PARALLEL_CROSS_EPS = 1e-12


def normalize_axis(
    axis: Sequence[float] | np.ndarray,
    *,
    name: str = "axis",
) -> np.ndarray:
    """Return a finite unit axis.

    Args:
        axis: Three-vector to normalize.
        name: Label included in validation errors.

    Raises:
        ValueError: If ``axis`` is not a finite, non-zero three-vector.
    """
    vector = np.asarray(axis, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= _AXIS_NORM_EPS:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def resolve_axis_in_world(
    axis: Sequence[float] | np.ndarray,
    reference_pose: PoseState | None = None,
) -> np.ndarray:
    """Resolve an axis expressed in a reference frame into the world frame.

    Passing ``None`` means that ``axis`` is already expressed in the world
    frame.  A reference pose must select one environment; callers resolve
    batched task state per environment before invoking this function.
    """
    local_axis = normalize_axis(axis)
    if reference_pose is None:
        return local_axis
    if reference_pose.batch_size != 1:
        raise ValueError(
            "resolve_axis_in_world expects a single-env reference PoseState"
        )
    rotation = quaternion_to_rotation_matrix(reference_pose.orientation[0])
    return normalize_axis(rotation @ local_axis, name="resolved world axis")


def axis_alignment_error(
    current_axis: Sequence[float] | np.ndarray,
    target_axis: Sequence[float] | np.ndarray,
    direction: AxisAlignmentDirectionLike | str,
) -> float:
    """Return the angular error for a directional axis constraint.

    ``same`` requires equal directions, ``opposite`` requires opposite
    directions, and ``either`` treats the two polarities as equivalent.  The
    result is in radians; the ``either`` result is therefore in ``[0, pi/2]``.
    """
    current = normalize_axis(current_axis, name="current_axis")
    target = normalize_axis(target_axis, name="target_axis")
    mode = _direction_value(direction)
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    if mode == "opposite":
        dot = -dot
    elif mode == "either":
        dot = abs(dot)
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def resolve_axis_alignment_orientation(
    current_pose: PoseState,
    controlled_axis: Sequence[float] | np.ndarray,
    target_axis_world: Sequence[float] | np.ndarray,
    direction: AxisAlignmentDirectionLike | str,
) -> tuple[float, float, float, float]:
    """Find the nearest orientation satisfying an axis-alignment goal.

    ``controlled_axis`` is expressed in the controlled frame and
    ``target_axis_world`` is already resolved into the world frame.  The
    returned ``xyzw`` quaternion applies only the minimum *swing* needed to
    align those axes.  It adds no rotation around the constrained axis, so the
    controlled frame's existing twist is retained.

    For ``either``, the target polarity requiring the smaller swing is chosen.
    Exact parallel input returns the current orientation unchanged.  Exact
    anti-parallel input is geometrically ambiguous; a deterministic tangent of
    the controlled frame is used as the half-turn axis, retaining a stable
    secondary direction instead of selecting an arbitrary world axis.
    """
    if current_pose.batch_size != 1:
        raise ValueError(
            "resolve_axis_alignment_orientation expects a single-env PoseState"
        )

    local_axis = normalize_axis(controlled_axis, name="controlled_axis")
    target = normalize_axis(target_axis_world, name="target_axis_world")
    current_axis_world = resolve_axis_in_world(local_axis, current_pose)
    mode = _direction_value(direction)

    dot = float(np.clip(np.dot(current_axis_world, target), -1.0, 1.0))
    if mode == "opposite" or (mode == "either" and dot < 0.0):
        target = -target

    tangent_local = _canonical_tangent(local_axis)
    tangent_world = resolve_axis_in_world(tangent_local, current_pose)
    swing = _shortest_arc_quaternion(
        current_axis_world,
        target,
        anti_parallel_axis=tangent_world,
    )
    return multiply_quaternions(swing, current_pose.orientation[0])


def _direction_value(direction: AxisAlignmentDirectionLike | str) -> str:
    raw_value = getattr(direction, "value", direction)
    if raw_value not in {"same", "opposite", "either"}:
        raise ValueError(
            "direction must be one of 'same', 'opposite', or 'either', "
            f"got {raw_value!r}"
        )
    return str(raw_value)


def _canonical_tangent(axis: np.ndarray) -> np.ndarray:
    """Return a deterministic unit tangent in the axis's local frame."""
    basis = np.zeros(3, dtype=np.float64)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    return normalize_axis(np.cross(axis, basis), name="canonical tangent")


def _shortest_arc_quaternion(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
    *,
    anti_parallel_axis: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return the shortest ``xyzw`` rotation from source to target."""
    source = normalize_axis(source_axis, name="source_axis")
    target = normalize_axis(target_axis, name="target_axis")
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    cross = np.cross(source, target)
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm <= _PARALLEL_CROSS_EPS:
        if dot >= 0.0:
            return (0.0, 0.0, 0.0, 1.0)
        half_turn_axis = normalize_axis(
            anti_parallel_axis,
            name="anti_parallel_axis",
        )
        # Remove numerical leakage along the source axis before the half-turn.
        half_turn_axis = half_turn_axis - source * float(np.dot(half_turn_axis, source))
        half_turn_axis = normalize_axis(
            half_turn_axis,
            name="anti_parallel_axis",
        )
        return (
            float(half_turn_axis[0]),
            float(half_turn_axis[1]),
            float(half_turn_axis[2]),
            0.0,
        )

    rotation_axis = cross / cross_norm
    half_angle = 0.5 * float(np.arctan2(cross_norm, dot))
    sine = float(np.sin(half_angle))
    return normalize_quaternion(
        (
            float(rotation_axis[0] * sine),
            float(rotation_axis[1] * sine),
            float(rotation_axis[2] * sine),
            float(np.cos(half_angle)),
        )
    )
