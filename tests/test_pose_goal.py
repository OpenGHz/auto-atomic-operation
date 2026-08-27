import numpy as np
import pytest
from numpy.testing import assert_allclose

from auto_atom.framework import AxisAlignmentDirection
from auto_atom.pose_goal import (
    axis_alignment_error,
    normalize_axis,
    resolve_axis_alignment_orientation,
    resolve_axis_in_world,
)
from auto_atom.utils.pose import (
    PoseState,
    multiply_quaternions,
    quaternion_angular_distance,
)


def _axis_angle_quaternion(
    axis: tuple[float, float, float], angle: float
) -> tuple[float, float, float, float]:
    unit_axis = normalize_axis(axis)
    sine = np.sin(angle / 2.0)
    return (
        float(unit_axis[0] * sine),
        float(unit_axis[1] * sine),
        float(unit_axis[2] * sine),
        float(np.cos(angle / 2.0)),
    )


def _assert_quaternions_equivalent(
    actual: tuple[float, float, float, float] | np.ndarray,
    expected: tuple[float, float, float, float] | np.ndarray,
    *,
    atol: float = 1e-10,
) -> None:
    actual_array = np.asarray(actual, dtype=np.float64)
    expected_array = np.asarray(expected, dtype=np.float64)
    assert abs(float(np.dot(actual_array, expected_array))) == pytest.approx(
        1.0, abs=atol
    )


def test_normalize_axis_returns_unit_vector_without_changing_direction() -> None:
    assert_allclose(normalize_axis((0.0, -3.0, 4.0)), (0.0, -0.6, 0.8))


@pytest.mark.parametrize(
    "axis,error",
    [
        ((0.0, 0.0, 0.0), "non-zero"),
        ((1.0, 2.0), "shape"),
        ((1.0, np.inf, 0.0), "finite"),
    ],
)
def test_normalize_axis_rejects_invalid_vectors(
    axis: tuple[float, ...], error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        normalize_axis(axis)


def test_resolve_axis_in_world_applies_reference_orientation() -> None:
    reference_pose = PoseState(
        orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), np.pi / 2.0)
    )

    assert_allclose(
        resolve_axis_in_world((1.0, 0.0, 0.0), reference_pose),
        (0.0, 1.0, 0.0),
        atol=1e-12,
    )
    assert_allclose(resolve_axis_in_world((0.0, 2.0, 0.0)), (0.0, 1.0, 0.0))


@pytest.mark.parametrize(
    "current,target,direction,expected",
    [
        ((0, 0, 1), (0, 0, 1), "same", 0.0),
        ((0, 0, 1), (0, 0, -1), "same", np.pi),
        ((0, 0, 1), (0, 0, -1), "opposite", 0.0),
        ((0, 0, 1), (0, 0, 1), "opposite", np.pi),
        ((0, 0, 1), (0, 0, -1), "either", 0.0),
        ((0, 0, 1), (0, 1, 0), "either", np.pi / 2.0),
    ],
)
def test_axis_alignment_error_obeys_direction_semantics(
    current: tuple[int, int, int],
    target: tuple[int, int, int],
    direction: str,
    expected: float,
) -> None:
    assert axis_alignment_error(current, target, direction) == pytest.approx(expected)


def test_axis_alignment_math_accepts_configuration_direction_enum() -> None:
    assert axis_alignment_error(
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        AxisAlignmentDirection.EITHER,
    ) == pytest.approx(0.0)


def test_resolve_orientation_applies_only_minimum_swing_and_keeps_twist() -> None:
    twist = 0.63
    current_orientation = _axis_angle_quaternion((0.0, 0.0, 1.0), twist)
    current_pose = PoseState(orientation=current_orientation)

    goal = resolve_axis_alignment_orientation(
        current_pose,
        controlled_axis=(0.0, 0.0, 1.0),
        target_axis_world=(1.0, 0.0, 0.0),
        direction="same",
    )

    goal_pose = PoseState(orientation=goal)
    assert_allclose(
        resolve_axis_in_world((0.0, 0.0, 1.0), goal_pose),
        (1.0, 0.0, 0.0),
        atol=1e-12,
    )
    assert quaternion_angular_distance(current_orientation, goal) == pytest.approx(
        np.pi / 2.0
    )
    expected = multiply_quaternions(
        _axis_angle_quaternion((0.0, 1.0, 0.0), np.pi / 2.0),
        current_orientation,
    )
    _assert_quaternions_equivalent(goal, expected)


def test_parallel_alignment_does_not_rotate_the_controlled_frame() -> None:
    current_orientation = _axis_angle_quaternion((1.0, 2.0, 3.0), 0.71)
    current_pose = PoseState(orientation=current_orientation)
    target_axis = resolve_axis_in_world((0.0, 0.0, 1.0), current_pose)

    goal = resolve_axis_alignment_orientation(
        current_pose,
        controlled_axis=(0.0, 0.0, 1.0),
        target_axis_world=target_axis,
        direction="same",
    )

    _assert_quaternions_equivalent(goal, current_orientation)
    assert quaternion_angular_distance(goal, current_orientation) == pytest.approx(0.0)


def test_either_alignment_selects_existing_polarity_without_flipping() -> None:
    current_pose = PoseState()

    goal = resolve_axis_alignment_orientation(
        current_pose,
        controlled_axis=(0.0, 0.0, 1.0),
        target_axis_world=(0.0, 0.0, -1.0),
        direction="either",
    )

    _assert_quaternions_equivalent(goal, current_pose.orientation[0])


def test_antiparallel_alignment_is_stable_and_retains_a_secondary_axis() -> None:
    twist = 0.47
    current_pose = PoseState(orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), twist))
    tangent_before = resolve_axis_in_world((0.0, 1.0, 0.0), current_pose)

    first_goal = resolve_axis_alignment_orientation(
        current_pose,
        controlled_axis=(0.0, 0.0, 1.0),
        target_axis_world=(0.0, 0.0, -1.0),
        direction="same",
    )
    second_goal = resolve_axis_alignment_orientation(
        current_pose,
        controlled_axis=(0.0, 0.0, 1.0),
        target_axis_world=(0.0, 0.0, -1.0),
        direction="same",
    )
    goal_pose = PoseState(orientation=first_goal)

    assert np.all(np.isfinite(first_goal))
    assert np.linalg.norm(first_goal) == pytest.approx(1.0)
    assert_allclose(first_goal, second_goal, atol=0.0)
    assert_allclose(
        resolve_axis_in_world((0.0, 0.0, 1.0), goal_pose),
        (0.0, 0.0, -1.0),
        atol=1e-12,
    )
    assert_allclose(
        resolve_axis_in_world((0.0, 1.0, 0.0), goal_pose),
        tangent_before,
        atol=1e-12,
    )
    assert quaternion_angular_distance(
        current_pose.orientation[0], first_goal
    ) == pytest.approx(np.pi)


def test_nearly_antiparallel_alignment_remains_numerically_accurate() -> None:
    current_pose = PoseState()
    target_axis = normalize_axis((1e-8, -2e-8, -1.0))

    goal = resolve_axis_alignment_orientation(
        current_pose,
        controlled_axis=(0.0, 0.0, 1.0),
        target_axis_world=target_axis,
        direction="same",
    )

    assert_allclose(
        resolve_axis_in_world((0.0, 0.0, 1.0), PoseState(orientation=goal)),
        target_axis,
        atol=1e-12,
    )


def test_axis_alignment_rejects_unknown_direction_and_batched_pose() -> None:
    with pytest.raises(ValueError, match="direction"):
        axis_alignment_error((1, 0, 0), (1, 0, 0), "parallel")

    batched_pose = PoseState(
        position=np.zeros((2, 3)),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2),
    )
    with pytest.raises(ValueError, match="single-env"):
        resolve_axis_alignment_orientation(
            batched_pose,
            controlled_axis=(0.0, 0.0, 1.0),
            target_axis_world=(0.0, 1.0, 0.0),
            direction="same",
        )
