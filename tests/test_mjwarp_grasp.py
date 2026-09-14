"""Tests for the MJWarp grasp-verdict geometry.

The lateral half of the grasp check is compared against the native
``MujocoOperatorHandler._check_grasp_conditions`` arithmetic, so a divergence
in frame convention (which rotation, transpose or not, which axes are lateral)
fails here rather than as a grasp that reads as centred when it is off to one
side.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_atom.backend.mjwarp.grasp import lateral_grasp_error, lateral_grasp_ok
from auto_atom.utils.pose import quaternion_to_rotation_matrix


def _native_lateral_error(obj_pos, eef_pos, eef_quat, grasp_axis):
    """The exact arithmetic native _check_grasp_conditions runs."""
    rot = quaternion_to_rotation_matrix(eef_quat)
    obj_in_eef = rot.T @ (np.asarray(obj_pos) - np.asarray(eef_pos))
    lateral_indices = [i for i in range(3) if i != grasp_axis]
    return float(np.linalg.norm(obj_in_eef[lateral_indices]))


def test_error_matches_native_arithmetic_for_a_tilted_eef():
    """A non-axis-aligned eef is where a transpose or frame bug would show."""
    obj_pos = np.array([0.31, -0.12, 0.44])
    eef_pos = np.array([0.30, -0.10, 0.40])
    # 30 deg about z, so the lateral plane is genuinely rotated from world.
    eef_quat = np.array([0.0, 0.0, 0.258819, 0.9659258])

    want = _native_lateral_error(obj_pos, eef_pos, eef_quat, grasp_axis=2)
    got = lateral_grasp_error(obj_pos, eef_pos, eef_quat, grasp_axis=2)

    assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("grasp_axis", [0, 1, 2])
def test_only_the_two_non_grasp_axes_count(grasp_axis):
    """Offset purely along the grasp axis is not lateral error.

    With the eef at identity, an offset placed only on ``grasp_axis`` should
    read as zero lateral error; the same offset on either other axis should not.
    """
    eef_pos = np.zeros(3)
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    along = np.zeros(3)
    along[grasp_axis] = 0.05
    assert lateral_grasp_error(eef_pos + along, eef_pos, identity, grasp_axis) == (
        pytest.approx(0.0, abs=1e-12)
    )

    for other in (i for i in range(3) if i != grasp_axis):
        off = np.zeros(3)
        off[other] = 0.05
        assert lateral_grasp_error(
            eef_pos + off, eef_pos, identity, grasp_axis
        ) == pytest.approx(0.05, abs=1e-9)


def test_ok_gate_uses_the_threshold():
    obj_pos = np.array([0.02, 0.0, 0.0])
    eef_pos = np.zeros(3)
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    ok, err = lateral_grasp_ok(
        obj_pos, eef_pos, identity, grasp_axis=2, lateral_threshold=0.03
    )
    assert ok and err == pytest.approx(0.02, abs=1e-9)

    ok, err = lateral_grasp_ok(
        obj_pos, eef_pos, identity, grasp_axis=2, lateral_threshold=0.01
    )
    assert not ok and err == pytest.approx(0.02, abs=1e-9)


def test_non_positive_threshold_disables_the_check():
    """Native treats threshold<=0 as 'unconstrained': ok=True, error=0."""
    far = np.array([1.0, 1.0, 1.0])
    eef_pos = np.zeros(3)
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    for threshold in (0.0, -1.0):
        ok, err = lateral_grasp_ok(
            far, eef_pos, identity, grasp_axis=2, lateral_threshold=threshold
        )
        assert ok and err == 0.0
