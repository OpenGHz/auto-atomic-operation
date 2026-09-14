"""Tests for MJWarp world <-> operator-base frame conversion.

Every case compares against the *native* ``UnifiedMujocoEnv._world_to_base`` /
``_base_to_world`` static methods rather than a restatement of the maths, so a
divergence in quaternion order, transpose direction or multiplication order
fails here instead of surfacing later as an arm that moves to a mirrored pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from auto_atom.backend.mjwarp.frames import (
    base_to_world,
    world_to_base,
    world_to_base_batch,
)

mujoco = pytest.importorskip("mujoco")

from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv  # noqa: E402

# A base pose that is neither at the origin nor axis-aligned: an identity base
# would pass even with the rotation applied the wrong way round.
_BASE_POS = np.array([0.35, -0.20, 0.15])
_BASE_QUAT = np.array([0.0, 0.0, 0.3826834, 0.9238795])  # 45 deg about z, xyzw

# Likewise an asymmetric target pose.
_TARGET_POS = np.array([0.52, 0.08, 0.44])
_TARGET_QUAT = np.array([-0.70710678, 0.70710678, 0.0, 0.0])


def test_world_to_base_matches_native():
    want_pos, want_quat = UnifiedMujocoEnv._world_to_base(
        _TARGET_POS, _TARGET_QUAT, _BASE_POS, _BASE_QUAT
    )
    got_pos, got_quat = world_to_base(_TARGET_POS, _TARGET_QUAT, _BASE_POS, _BASE_QUAT)

    # Native casts its result to float32; this path stays float64 on purpose
    # (see the module docstring), so the comparison is at float32 tolerance --
    # the bound the design doc's 3.6 sets for every CPU/GPU comparison here.
    np.testing.assert_allclose(got_pos, want_pos, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got_quat, want_quat, rtol=1e-6, atol=1e-7)


def test_base_to_world_matches_native():
    want_pos, want_quat = UnifiedMujocoEnv._base_to_world(
        _TARGET_POS, _TARGET_QUAT, _BASE_POS, _BASE_QUAT
    )
    got_pos, got_quat = base_to_world(_TARGET_POS, _TARGET_QUAT, _BASE_POS, _BASE_QUAT)

    np.testing.assert_allclose(got_pos, want_pos, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got_quat, want_quat, rtol=1e-6, atol=1e-7)


def test_the_two_conversions_are_inverses():
    """Round-tripping is what a control tick effectively relies on."""
    pos_b, quat_b = world_to_base(_TARGET_POS, _TARGET_QUAT, _BASE_POS, _BASE_QUAT)
    pos_w, quat_w = base_to_world(pos_b, quat_b, _BASE_POS, _BASE_QUAT)

    np.testing.assert_allclose(pos_w, _TARGET_POS, rtol=1e-9, atol=1e-12)
    # q and -q are the same rotation.
    assert np.allclose(quat_w, _TARGET_QUAT, atol=1e-9) or np.allclose(
        quat_w, -_TARGET_QUAT, atol=1e-9
    )


def test_identity_base_is_a_no_op():
    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    pos, quat = world_to_base(_TARGET_POS, _TARGET_QUAT, np.zeros(3), identity_quat)

    np.testing.assert_allclose(pos, _TARGET_POS, atol=1e-12)
    np.testing.assert_allclose(quat, _TARGET_QUAT, atol=1e-12)


def test_batch_gives_each_world_its_own_base_frame():
    """Per-world base poses are the point: randomization can move the base.

    Sharing one base pose across worlds would put every world but one in the
    wrong frame, and this asserts each row equals the single-pose conversion
    against that row's own base.
    """
    base_positions = np.stack([_BASE_POS, _BASE_POS + np.array([0.1, 0.05, -0.02])])
    base_quats = np.stack([_BASE_QUAT, np.array([0.0, 0.0, 0.0, 1.0])])
    targets = np.stack([_TARGET_POS, _TARGET_POS + np.array([0.01, 0.02, 0.03])])
    target_quats = np.stack([_TARGET_QUAT, _TARGET_QUAT])

    got_pos, got_quat = world_to_base_batch(
        targets, target_quats, base_positions, base_quats
    )

    assert got_pos.shape == (2, 3) and got_quat.shape == (2, 4)
    for world in range(2):
        want_pos, want_quat = world_to_base(
            targets[world],
            target_quats[world],
            base_positions[world],
            base_quats[world],
        )
        np.testing.assert_allclose(got_pos[world], want_pos, atol=1e-12)
        np.testing.assert_allclose(got_quat[world], want_quat, atol=1e-12)
    # World 1's base is unrotated but still translated, so its result is a pure
    # subtraction. A shared-base bug would instead rotate it by world 0's 45 deg.
    np.testing.assert_allclose(got_pos[1], targets[1] - base_positions[1], atol=1e-12)


def test_batch_broadcasts_a_shared_base_to_every_world():
    targets = np.stack([_TARGET_POS, _TARGET_POS + np.array([0.05, 0.0, 0.0])])
    target_quats = np.stack([_TARGET_QUAT, _TARGET_QUAT])

    got_pos, _ = world_to_base_batch(targets, target_quats, _BASE_POS, _BASE_QUAT)

    for world in range(2):
        want_pos, _ = world_to_base(
            targets[world], target_quats[world], _BASE_POS, _BASE_QUAT
        )
        np.testing.assert_allclose(got_pos[world], want_pos, atol=1e-12)


def test_batch_rejects_a_row_count_it_cannot_map():
    """Three base rows for two worlds is a caller bug, not a broadcast."""
    targets = np.zeros((3, 3))
    target_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (3, 1))
    bases = np.zeros((2, 3))
    base_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))

    with pytest.raises(ValueError, match="Expected either 1 row"):
        world_to_base_batch(targets, target_quats, bases, base_quats)
