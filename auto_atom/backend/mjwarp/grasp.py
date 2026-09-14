"""Grasp-verdict geometry for the MJWarp backend.

The grasp check the native handler runs has two independent halves:

* a **contact** half -- does a left and a right finger geom each touch the
  target -- which lives on the scene-state adapter
  (:meth:`~auto_atom.basis.mjwarp.state.MjWarpSceneState.finger_contacts_with_target`)
  because it reads ``data.contact``;
* a **geometry** half -- is the object centred between the fingers rather than
  merely brushing one -- which is a pure function of two poses and lives here.

Keeping the geometry half as a free function, taking poses as arguments rather
than reaching for an end-effector accessor, lets it be written and tested before
the operator handler (which owns the eef-pose plumbing) exists. The operator
handler in a later round composes the two halves into one verdict.
"""

from __future__ import annotations

import numpy as np

from auto_atom.utils.pose import quaternion_to_rotation_matrix


def lateral_grasp_error(
    object_position: np.ndarray,
    eef_position: np.ndarray,
    eef_orientation_xyzw: np.ndarray,
    grasp_axis: int,
) -> float:
    """Distance from the object to the grasp line, in the end-effector frame.

    The object is transformed into the eef frame and its offset is measured in
    the plane perpendicular to ``grasp_axis`` -- the two axes along which the
    fingers close. A large value means the object sits off to the side of the
    gripper (a glancing touch) rather than between the fingers, which is what
    separates a real grasp from a finger merely grazing the target.

    Mirrors the native ``_check_grasp_conditions`` geometry exactly:
    ``rot.T @ (obj - eef)`` then the norm over the non-grasp axes. Inputs are
    single poses (one world); the caller iterates worlds.
    """
    obj = np.asarray(object_position, dtype=np.float64)
    eef = np.asarray(eef_position, dtype=np.float64)
    rot = quaternion_to_rotation_matrix(eef_orientation_xyzw)
    obj_in_eef = rot.T @ (obj - eef)
    lateral_indices = [axis for axis in range(3) if axis != grasp_axis]
    return float(np.linalg.norm(obj_in_eef[lateral_indices]))


def lateral_grasp_ok(
    object_position: np.ndarray,
    eef_position: np.ndarray,
    eef_orientation_xyzw: np.ndarray,
    grasp_axis: int,
    lateral_threshold: float,
) -> tuple[bool, float]:
    """``(ok, error)`` for the lateral half of the grasp check.

    A non-positive ``lateral_threshold`` disables the check -- the native path
    treats that as "geometry unconstrained", returning ``ok=True`` with zero
    error -- so a config that does not want lateral gating gets the same
    verdict on either backend.
    """
    if lateral_threshold <= 0:
        return True, 0.0
    error = lateral_grasp_error(
        object_position, eef_position, eef_orientation_xyzw, grasp_axis
    )
    return error <= lateral_threshold, error
