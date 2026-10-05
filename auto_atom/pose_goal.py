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


AngleIntervals = Sequence[tuple[float, float]] | None
"""Alternative inclusive angle intervals in radians; ``None`` means free."""

_ANGLE_EPS = 1e-9


def nearest_feasible_orientations(
    current_orientation: Sequence[float] | np.ndarray,
    reference_orientation: Sequence[float] | np.ndarray,
    primary_axis: Sequence[float] | np.ndarray,
    *,
    primary_elevation: AngleIntervals = None,
    primary_azimuth: AngleIntervals = None,
    constraints: Sequence[
        tuple[Sequence[float] | np.ndarray, AngleIntervals, AngleIntervals]
    ] = (),
    resolution: float = np.radians(5.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Return feasible orientations ordered by rotation from the current one.

    The controlled frame's orientation in the reference frame is written as
    ``Rz(psi) @ Ry(-theta) @ Rx(phi) @ A.T``, where ``A`` maps +x onto
    ``primary_axis`` (controlled frame): ``psi`` and ``theta`` are the primary
    axis's azimuth and elevation in the reference frame and ``phi`` is the
    spin about it. Candidates sample ``psi`` and ``theta`` inside their
    intervals and ``phi`` over the full circle at ``resolution``; every grid
    includes the current value clamped into its intervals, so a current
    orientation that is already feasible is returned unrotated. Candidates
    that violate a constraint are dropped. Each constraint is an
    ``(axis, elevation_intervals, azimuth_intervals)`` triple with the axis
    expressed in the controlled frame.

    Returns world-frame rotation matrices of shape ``(n, 3, 3)`` and their
    rotation angles from the current orientation, ascending (ties keep grid
    order); both are empty when nothing is feasible.
    """
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    current = quaternion_to_rotation_matrix(normalize_quaternion(current_orientation))
    world_from_reference = quaternion_to_rotation_matrix(
        normalize_quaternion(reference_orientation)
    )
    in_reference = world_from_reference.T @ current
    axis = normalize_axis(primary_axis, name="primary_axis")
    primary_from_x = _rotation_from_x(axis)

    # Current (psi, theta, phi) of the parameterization.
    frame = in_reference @ primary_from_x
    direction = frame[:, 0]
    psi0 = float(np.arctan2(direction[1], direction[0]))
    theta0 = float(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    spin = _rot_y(theta0) @ _rot_z(-psi0) @ frame
    phi0 = float(np.arctan2(spin[2, 1], spin[1, 1]))

    psis = _angle_grid(psi0, primary_azimuth, resolution, wrap=True)
    thetas = _angle_grid(
        theta0,
        primary_elevation
        if primary_elevation is not None
        else ((-np.pi / 2.0, np.pi / 2.0),),
        resolution,
        wrap=False,
    )
    phis = phi0 + resolution * np.arange(int(np.ceil(2.0 * np.pi / resolution)))

    psi_grid, theta_grid, phi_grid = (
        values.ravel() for values in np.meshgrid(psis, thetas, phis, indexing="ij")
    )
    rotations = (
        _rot_z_batch(psi_grid)
        @ _rot_y_batch(-theta_grid)
        @ _rot_x_batch(phi_grid)
        @ primary_from_x.T
    )
    keep = np.ones(len(rotations), dtype=bool)
    for constraint_axis, elevation, azimuth in constraints:
        local = normalize_axis(constraint_axis, name="constraint axis")
        directions = rotations @ local
        if elevation is not None:
            angles = np.arcsin(np.clip(directions[:, 2], -1.0, 1.0))
            keep &= _within(angles, elevation, wrap=False)
        if azimuth is not None:
            angles = np.arctan2(directions[:, 1], directions[:, 0])
            keep &= _within(angles, azimuth, wrap=True)
    rotations = rotations[keep]
    if len(rotations) == 0:
        return np.empty((0, 3, 3)), np.empty(0)

    relative = np.einsum("nji,jk->nik", rotations, in_reference)
    cosine = (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0
    angles = np.arccos(np.clip(cosine, -1.0, 1.0))
    order = np.argsort(angles, kind="stable")
    world = np.einsum("ij,njk->nik", world_from_reference, rotations[order])
    return world, angles[order]


AxisConstraint = tuple[Sequence[float] | np.ndarray, AngleIntervals, AngleIntervals]
"""``(axis, elevation_intervals, azimuth_intervals)``, axis in the controlled frame."""


def nearest_feasible_candidates(
    current_orientation: Sequence[float] | np.ndarray,
    reference_orientation: Sequence[float] | np.ndarray,
    primary_axis: Sequence[float] | np.ndarray,
    *,
    primary_elevation: AngleIntervals = None,
    primary_azimuth: AngleIntervals = None,
    constraints: Sequence[AxisConstraint] = (),
    postures: Sequence[Sequence[AxisConstraint]] = (),
    resolution: float = np.radians(5.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feasible orientations over alternative postures, nearest first.

    Each posture adds its constraints to the shared ``constraints``; an empty
    ``postures`` is one posture with none. Returns rotation matrices
    ``(n, 3, 3)``, rotation angles from the current orientation and the
    posture index of each candidate, ordered by angle; exact ties keep
    posture order, then grid order.
    """
    rotations, angles, indices = [], [], []
    for index, extra in enumerate(postures or ((),)):
        posture_rotations, posture_angles = nearest_feasible_orientations(
            current_orientation,
            reference_orientation,
            primary_axis,
            primary_elevation=primary_elevation,
            primary_azimuth=primary_azimuth,
            constraints=[*constraints, *extra],
            resolution=resolution,
        )
        rotations.append(posture_rotations)
        angles.append(posture_angles)
        indices.append(np.full(len(posture_angles), index, dtype=np.int64))
    all_angles = np.concatenate(angles)
    order = np.argsort(all_angles, kind="stable")
    return (
        np.concatenate(rotations)[order],
        all_angles[order],
        np.concatenate(indices)[order],
    )


def _angle_grid(
    current: float,
    intervals: AngleIntervals,
    resolution: float,
    *,
    wrap: bool,
) -> np.ndarray:
    """Samples covering ``intervals``, including the clamped current value."""
    if intervals is None:
        count = int(np.ceil(2.0 * np.pi / resolution))
        return current + resolution * np.arange(count)
    samples: list[float] = []
    nearest, nearest_distance = None, np.inf
    for low, high in intervals:
        count = max(1, int(np.ceil((high - low) / resolution)) + 1)
        samples.extend(np.linspace(low, high, count) if high > low else [low])
        offset = _wrap(current - low) + low if wrap else current
        clamped = float(np.clip(offset, low, high))
        distance = abs(_wrap(clamped - current)) if wrap else abs(clamped - current)
        if distance < nearest_distance:
            nearest, nearest_distance = clamped, distance
    samples.append(float(nearest))
    return np.unique(np.asarray(samples, dtype=np.float64))


def _within(angles: np.ndarray, intervals: AngleIntervals, *, wrap: bool) -> np.ndarray:
    """Whether each angle lies inside one of the intervals."""
    if wrap:
        angles = (angles + np.pi) % (2.0 * np.pi) - np.pi
    inside = np.zeros(angles.shape, dtype=bool)
    for low, high in intervals or ():
        inside |= (angles >= low - _ANGLE_EPS) & (angles <= high + _ANGLE_EPS)
    return inside


def _wrap(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi)``."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _rotation_from_x(axis: np.ndarray) -> np.ndarray:
    """A rotation that maps +x onto ``axis``."""
    swing = _shortest_arc_quaternion(
        np.array([1.0, 0.0, 0.0]),
        axis,
        anti_parallel_axis=np.array([0.0, 0.0, 1.0]),
    )
    return quaternion_to_rotation_matrix(swing)


def _rot_x(angle: float) -> np.ndarray:
    return _rot_x_batch(np.array([angle]))[0]


def _rot_y(angle: float) -> np.ndarray:
    return _rot_y_batch(np.array([angle]))[0]


def _rot_z(angle: float) -> np.ndarray:
    return _rot_z_batch(np.array([angle]))[0]


def _rot_x_batch(angles: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles), np.sin(angles)
    out = np.zeros((len(angles), 3, 3))
    out[:, 0, 0] = 1.0
    out[:, 1, 1], out[:, 1, 2], out[:, 2, 1], out[:, 2, 2] = c, -s, s, c
    return out


def _rot_y_batch(angles: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles), np.sin(angles)
    out = np.zeros((len(angles), 3, 3))
    out[:, 1, 1] = 1.0
    out[:, 0, 0], out[:, 0, 2], out[:, 2, 0], out[:, 2, 2] = c, s, -s, c
    return out


def _rot_z_batch(angles: np.ndarray) -> np.ndarray:
    c, s = np.cos(angles), np.sin(angles)
    out = np.zeros((len(angles), 3, 3))
    out[:, 2, 2] = 1.0
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1] = c, -s, s, c
    return out


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
