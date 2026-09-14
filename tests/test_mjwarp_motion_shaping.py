"""Tests for MJWarp cartesian step shaping and stall scaling.

Shaping decides what command a tick sends, so its failure mode is an arm that
either lurches (no clamping) or crawls and times out (over-clamping). The stall
scaler is a state machine, so its transitions are driven explicitly rather than
inferred from a single happy path.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_atom.backend.mjwarp.motion_shaping import (
    StallScaler,
    clamp_cartesian_step,
    effective_step_bounds,
)
from auto_atom.utils.pose import quaternion_angular_distance

_IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def test_position_is_clamped_along_the_line_to_the_goal():
    current = np.zeros(3)
    desired = np.array([1.0, 0.0, 0.0])

    position, _ = clamp_cartesian_step(
        current, _IDENTITY, desired, _IDENTITY, 0.02, 0.0
    )

    np.testing.assert_allclose(position, [0.02, 0.0, 0.0], atol=1e-12)


def test_position_clamp_keeps_the_direction_for_a_diagonal_goal():
    """Clamping must scale the whole vector, not truncate one axis."""
    current = np.zeros(3)
    desired = np.array([3.0, 4.0, 0.0])  # length 5

    position, _ = clamp_cartesian_step(current, _IDENTITY, desired, _IDENTITY, 0.5, 0.0)

    np.testing.assert_allclose(position, [0.3, 0.4, 0.0], atol=1e-12)
    assert float(np.linalg.norm(position)) == pytest.approx(0.5)


def test_a_goal_inside_the_bound_is_untouched():
    current = np.zeros(3)
    desired = np.array([0.005, 0.0, 0.0])

    position, orientation = clamp_cartesian_step(
        current, _IDENTITY, desired, _IDENTITY, 0.02, 0.2
    )

    np.testing.assert_allclose(position, desired, atol=1e-12)
    np.testing.assert_allclose(orientation, _IDENTITY, atol=1e-12)


def test_non_positive_bounds_disable_clamping():
    """Zero means 'no limit' in the config, not 'no motion'."""
    current = np.zeros(3)
    desired = np.array([5.0, 0.0, 0.0])

    position, _ = clamp_cartesian_step(current, _IDENTITY, desired, _IDENTITY, 0.0, 0.0)

    np.testing.assert_allclose(position, desired, atol=1e-12)


def test_orientation_is_clamped_to_the_angular_bound():
    """A large rotation is approached at a bounded angular rate."""
    # 90 deg about z.
    desired = np.array([0.0, 0.0, 0.70710678, 0.70710678])
    max_angular = 0.2

    _, orientation = clamp_cartesian_step(
        np.zeros(3), _IDENTITY, np.zeros(3), desired, 0.0, max_angular
    )

    angle = quaternion_angular_distance(_IDENTITY, orientation)
    assert angle == pytest.approx(max_angular, abs=1e-6)


def test_orientation_inside_the_bound_is_untouched():
    small = np.array([0.0, 0.0, 0.0174524, 0.9998477])  # 2 deg about z

    _, orientation = clamp_cartesian_step(
        np.zeros(3), _IDENTITY, np.zeros(3), small, 0.0, 0.2
    )

    np.testing.assert_allclose(orientation, small, atol=1e-9)


def test_waypoint_bounds_override_defaults_and_scale_applies():
    linear, angular = effective_step_bounds(0.05, 0.3, 0.02, 0.2, 1.0)
    assert (linear, angular) == (0.05, 0.3)

    # Zero means "not specified at this waypoint", so the default is used.
    linear, angular = effective_step_bounds(0.0, 0.0, 0.02, 0.2, 1.0)
    assert (linear, angular) == (0.02, 0.2)

    linear, angular = effective_step_bounds(0.0, 0.0, 0.02, 0.2, 0.5)
    assert linear == pytest.approx(0.01)
    assert angular == pytest.approx(0.1)


def test_scaler_grows_slowly_on_progress():
    scaler = StallScaler(nworld=1)

    # First call always improves (best starts at inf) but is already capped.
    assert scaler.update(0, 0.5, 0.5) == pytest.approx(1.0)
    assert scaler.stall_count(0) == 0


def test_scaler_halves_after_the_patience_window():
    """Eight stalled ticks halve the step, then the counter re-arms."""
    scaler = StallScaler(nworld=1)
    scaler.update(0, 0.5, 0.5)  # establish a best

    for _ in range(7):
        assert scaler.update(0, 0.5, 0.5) == pytest.approx(1.0)
    assert scaler.stall_count(0) == 7

    assert scaler.update(0, 0.5, 0.5) == pytest.approx(0.5)
    assert scaler.stall_count(0) == 0, "counter re-arms after acting"


def test_scaler_floors_at_a_tenth():
    scaler = StallScaler(nworld=1)
    scaler.update(0, 0.5, 0.5)

    for _ in range(8 * 12):
        scaler.update(0, 0.5, 0.5)

    assert scaler.scale(0) == pytest.approx(0.1)


def test_scaler_recovers_after_progress_resumes():
    scaler = StallScaler(nworld=1)
    scaler.update(0, 0.5, 0.5)
    for _ in range(8):
        scaler.update(0, 0.5, 0.5)
    assert scaler.scale(0) == pytest.approx(0.5)

    # A real improvement (>1e-4 m) grows the scale back by 10%.
    assert scaler.update(0, 0.4, 0.5) == pytest.approx(0.55)


def test_orientation_progress_alone_counts_as_progress():
    """Either axis improving is progress; requiring both would stall on turns."""
    scaler = StallScaler(nworld=1)
    scaler.update(0, 0.5, 0.5)

    scaler.update(0, 0.5, 0.4)  # position flat, orientation improved

    assert scaler.stall_count(0) == 0


def test_improvement_must_exceed_the_threshold():
    """Numerical jitter must not read as progress and mask a real stall."""
    scaler = StallScaler(nworld=1)
    scaler.update(0, 0.5, 0.5)

    scaler.update(0, 0.5 - 1e-6, 0.5 - 1e-6)

    assert scaler.stall_count(0) == 1


def test_worlds_scale_independently():
    """One world stalling must not shrink another world's step."""
    scaler = StallScaler(nworld=2)
    for world in (0, 1):
        scaler.update(world, 0.5, 0.5)

    for _ in range(8):
        scaler.update(0, 0.5, 0.5)

    assert scaler.scale(0) == pytest.approx(0.5)
    assert scaler.scale(1) == pytest.approx(1.0)


def test_reset_forgets_progress_history():
    """A new waypoint's error is unrelated to the old one's best."""
    scaler = StallScaler(nworld=1)
    scaler.update(0, 0.5, 0.5)
    for _ in range(8):
        scaler.update(0, 0.5, 0.5)
    assert scaler.scale(0) == pytest.approx(0.5)

    scaler.reset(0)

    assert scaler.scale(0) == pytest.approx(1.0)
    assert scaler.stall_count(0) == 0
    # A large error right after reset counts as progress, not a stall.
    scaler.update(0, 9.0, 9.0)
    assert scaler.stall_count(0) == 0


def test_disabled_scaler_always_returns_one():
    """Contact-heavy motion opts out, and then the caller's bounds stand."""
    scaler = StallScaler(nworld=1, enabled=False)

    for _ in range(20):
        assert scaler.update(0, 0.5, 0.5) == pytest.approx(1.0)
    assert scaler.scale(0) == pytest.approx(1.0)


def test_world_index_is_bounds_checked():
    scaler = StallScaler(nworld=2)

    with pytest.raises(IndexError, match=r"world must be in \[0, 2\)"):
        scaler.update(2, 0.1, 0.1)


def test_nworld_must_be_positive():
    with pytest.raises(ValueError, match="nworld must be >= 1"):
        StallScaler(nworld=0)


def test_joint_delta_clamp_scales_the_whole_vector():
    """The intermediate pose must stay on the joint-space line to the solution.

    Clipping each joint independently would bend that path and can steer the arm
    somewhere neither the seed nor the solution intended.
    """
    from auto_atom.backend.mjwarp.motion_shaping import clamp_joint_delta

    seed = np.zeros(3)
    solved = np.array([1.0, 0.5, -0.25])  # worst joint moves 1.0 rad

    got = clamp_joint_delta(solved, seed, 0.35)

    np.testing.assert_allclose(got, [0.35, 0.175, -0.0875], atol=1e-12)
    # Direction preserved: got is a scalar multiple of the original delta.
    assert float(np.max(np.abs(got))) == pytest.approx(0.35)


def test_joint_delta_within_the_bound_is_untouched():
    from auto_atom.backend.mjwarp.motion_shaping import clamp_joint_delta

    seed = np.array([0.1, 0.2])
    solved = np.array([0.2, 0.25])

    np.testing.assert_allclose(clamp_joint_delta(solved, seed, 0.35), solved)


def test_joint_delta_clamp_can_be_disabled():
    from auto_atom.backend.mjwarp.motion_shaping import clamp_joint_delta

    seed = np.zeros(2)
    solved = np.array([5.0, 0.0])

    np.testing.assert_allclose(clamp_joint_delta(solved, seed, 0.0), solved)


def test_joint_delta_clamp_handles_an_empty_arm():
    from auto_atom.backend.mjwarp.motion_shaping import clamp_joint_delta

    empty = np.empty(0)
    assert clamp_joint_delta(empty, empty, 0.35).size == 0
