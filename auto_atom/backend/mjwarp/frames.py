"""World <-> operator-base frame conversion for the MJWarp backend.

``move_to_pose`` issues its target in the operator's base frame, so every
control tick converts a world-frame goal into base frame and operator
registration converts the eef pose into base frame once to get the fixed tool
offset. Both are pure functions of a base pose, which is why they live here
rather than on the env: they can be written, tested and reasoned about before
any operator plumbing exists.

Quaternions are xyzw throughout, matching the runtime's convention and this
project's ``utils.transformations`` (which is the xyzw variant -- its
``quaternion_multiply`` unpacks ``x, y, z, w`` and its identity is
``[0, 0, 0, 1]``, unlike the classic wxyz library it derives from).

**Deliberate difference from the native path:** the native
``_world_to_base``/``_base_to_world`` cast their results to ``float32``. That
cast is not reproduced here. It buys nothing on this path -- the device model is
float32 regardless, so the quantisation happens at the device boundary anyway --
and doing the host arithmetic in float64 keeps the conversion itself from adding
error on top of that. Equivalence with native is therefore asserted at float32
tolerance, which is the bound the design doc's 3.6 already sets for every
CPU/GPU comparison in this port.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from auto_atom.utils.transformations import (
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)


def world_to_base(
    position_w: np.ndarray,
    orientation_w_xyzw: np.ndarray,
    base_position: np.ndarray,
    base_orientation_xyzw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Express a world-frame pose in the operator base frame.

    Mirrors the native ``_world_to_base``: rotate the world-frame offset by the
    inverse base rotation, and left-multiply the orientation by the inverse base
    quaternion.
    """
    inverse = quaternion_inverse(np.asarray(base_orientation_xyzw, dtype=np.float64))
    inverse_rotation = quaternion_matrix(inverse)[:3, :3]
    position = inverse_rotation @ (
        np.asarray(position_w, dtype=np.float64)
        - np.asarray(base_position, dtype=np.float64)
    )
    orientation = quaternion_multiply(
        inverse, np.asarray(orientation_w_xyzw, dtype=np.float64)
    )
    return position, orientation


def base_to_world(
    position_b: np.ndarray,
    orientation_b_xyzw: np.ndarray,
    base_position: np.ndarray,
    base_orientation_xyzw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Express an operator-base-frame pose in the world frame.

    The exact inverse of :func:`world_to_base`, mirroring the native
    ``_base_to_world``.
    """
    base_quat = np.asarray(base_orientation_xyzw, dtype=np.float64)
    rotation = quaternion_matrix(base_quat)[:3, :3]
    position = rotation @ np.asarray(position_b, dtype=np.float64) + np.asarray(
        base_position, dtype=np.float64
    )
    orientation = quaternion_multiply(
        base_quat, np.asarray(orientation_b_xyzw, dtype=np.float64)
    )
    return position, orientation


def world_to_base_batch(
    positions_w: np.ndarray,
    orientations_w_xyzw: np.ndarray,
    base_positions: np.ndarray,
    base_orientations_xyzw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-world :func:`world_to_base`, shaped ``(nworld, 3)`` / ``(nworld, 4)``.

    Each world converts against **its own** base pose rather than a shared one.
    That matters because randomization can place the operator base differently
    per environment, and MJWarp's ``body_pos``/``body_quat`` are batched per
    world precisely so it can: sharing one base pose across worlds would send
    every world but one a target in the wrong frame.
    """
    positions_w = np.atleast_2d(np.asarray(positions_w, dtype=np.float64))
    orientations_w_xyzw = np.atleast_2d(
        np.asarray(orientations_w_xyzw, dtype=np.float64)
    )
    base_positions = np.atleast_2d(np.asarray(base_positions, dtype=np.float64))
    base_orientations_xyzw = np.atleast_2d(
        np.asarray(base_orientations_xyzw, dtype=np.float64)
    )

    nworld = max(
        positions_w.shape[0],
        orientations_w_xyzw.shape[0],
        base_positions.shape[0],
        base_orientations_xyzw.shape[0],
    )
    out_positions = np.empty((nworld, 3), dtype=np.float64)
    out_orientations = np.empty((nworld, 4), dtype=np.float64)
    for world in range(nworld):
        out_positions[world], out_orientations[world] = world_to_base(
            _row(positions_w, world),
            _row(orientations_w_xyzw, world),
            _row(base_positions, world),
            _row(base_orientations_xyzw, world),
        )
    return out_positions, out_orientations


def _row(array: np.ndarray, index: int) -> np.ndarray:
    """Row ``index``, or the single row when the argument is shared.

    A batch of one is a value shared by every world; anything else is indexed
    per world. Keeping this in one place stops a caller from silently reading
    row 0 for every world when it meant to broadcast.
    """
    if array.shape[0] == 1:
        return array[0]
    if index >= array.shape[0]:
        raise ValueError(
            f"Expected either 1 row or more than {index} rows; got {array.shape[0]}."
        )
    return array[index]
