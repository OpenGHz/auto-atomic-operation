"""Equivalence tests for the MJWarp scene-state adapter.

Every frame read is compared against the *native* ``MujocoBasis`` method rather
than against a restatement of what that method does, so a divergence in
convention (quaternion order, matrix layout, dtype) fails here instead of
surfacing later as a subtly wrong grasp pose.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.basis.mjc.mujoco_basis import MujocoBasis  # noqa: E402
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402

# A deliberately asymmetric pose: an identity or axis-aligned orientation would
# pass even with a wrong quaternion component order.
_SCENE_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="holder" pos="0.1 -0.2 0.3" quat="0.9238795 0 0.3826834 0">
      <geom name="holder_geom" type="box" size="0.05 0.04 0.03"/>
      <site name="holder_site" pos="0.01 0.02 0.03"/>
    </body>
    <body name="mover" pos="-0.4 0.15 0.5" quat="0.8446232 0.1913417 0.4619398 0.1913417">
      <freejoint name="mover_free"/>
      <geom name="mover_geom" type="box" size="0.02 0.02 0.02"/>
      <site name="mover_site" pos="0 0.01 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def host_model():
    return mujoco.MjModel.from_xml_string(_SCENE_XML)


@pytest.fixture(scope="module")
def native_basis(host_model):
    """A native basis carrying only what the frame readers touch."""
    basis = MujocoBasis.__new__(MujocoBasis)
    basis.model = host_model
    basis.data = mujoco.MjData(host_model)
    mujoco.mj_forward(basis.model, basis.data)
    return basis


@pytest.fixture(scope="module")
def warp_state(host_model):
    state = MjWarpSceneState(host_model, nworld=2)
    state.forward()
    return state


@pytest.mark.parametrize("body_name", ["holder", "mover"])
def test_body_pose_matches_native_basis(native_basis, warp_state, body_name):
    want_pos, want_quat = native_basis.get_body_pose(body_name)
    got_pos, got_quat = warp_state.get_body_pose(body_name)

    np.testing.assert_allclose(got_pos, want_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat, want_quat, atol=1e-6)
    assert got_pos.dtype == want_pos.dtype
    assert got_quat.dtype == want_quat.dtype


@pytest.mark.parametrize("site_name", ["holder_site", "mover_site"])
def test_site_pose_matches_native_basis(native_basis, warp_state, site_name):
    want_pos, want_quat = native_basis.get_site_pose(site_name)
    got_pos, got_quat = warp_state.get_site_pose(site_name)

    np.testing.assert_allclose(got_pos, want_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat, want_quat, atol=1e-6)


def test_body_orientation_is_xyzw_not_wxyz(native_basis, warp_state):
    """Pin the conversion that MJWarp's wp.quat type invites getting wrong.

    MJWarp stores MuJoCo's wxyz order inside a ``wp.quat`` whose Warp-native
    order is xyzw. Returning the raw array would therefore be wrong while still
    looking plausible, so this asserts the returned xyzw explicitly against the
    host model's authored wxyz.
    """
    _, got_quat = warp_state.get_body_pose("holder")
    authored_wxyz = np.array([0.9238795, 0.0, 0.3826834, 0.0])

    np.testing.assert_allclose(got_quat, authored_wxyz[[1, 2, 3, 0]], atol=1e-6)
    # And the raw device order is *not* what we return.
    assert not np.allclose(got_quat, authored_wxyz, atol=1e-3)


def test_unknown_names_raise(warp_state):
    with pytest.raises(ValueError, match="Body 'nope' not found"):
        warp_state.get_body_pose("nope")
    with pytest.raises(ValueError, match="Site 'nope' not found"):
        warp_state.get_site_pose("nope")


def test_world_index_is_bounds_checked(warp_state):
    with pytest.raises(IndexError, match=r"world_index must be in \[0, 2\)"):
        warp_state.get_body_pose("mover", world_index=2)


def test_free_joint_write_round_trips_through_forward(host_model):
    """A kinematic transport lands where it was asked to, in every world."""
    state = MjWarpSceneState(host_model, nworld=2)
    position = np.array([0.25, -0.35, 0.65])
    orientation_xyzw = np.array([0.1913417, 0.4619398, 0.1913417, 0.8446232])

    state.set_free_joint_pose("mover_free", position, orientation_xyzw)

    for world in range(2):
        got_pos, got_quat = state.get_body_pose("mover", world_index=world)
        np.testing.assert_allclose(got_pos, position, atol=1e-6)
        # A quaternion and its negation are the same rotation.
        assert np.allclose(got_quat, orientation_xyzw, atol=1e-6) or np.allclose(
            got_quat, -orientation_xyzw, atol=1e-6
        )


def test_free_joint_write_matches_native_qpos_write(host_model):
    """The device write agrees with doing the same write on the host."""
    position = np.array([0.11, 0.22, 0.44])
    orientation_xyzw = np.array([0.0, 0.3826834, 0.0, 0.9238795])

    # Native: write qpos directly, exactly as the MuJoCo object handler does.
    native = MujocoBasis.__new__(MujocoBasis)
    native.model = host_model
    native.data = mujoco.MjData(host_model)
    joint = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_JOINT, "mover_free")
    adr = int(host_model.jnt_qposadr[joint])
    native.data.qpos[adr : adr + 3] = position
    native.data.qpos[adr + 3 : adr + 7] = orientation_xyzw[[3, 0, 1, 2]]
    mujoco.mj_forward(native.model, native.data)
    want_pos, want_quat = native.get_body_pose("mover")

    state = MjWarpSceneState(host_model, nworld=1)
    state.set_free_joint_pose("mover_free", position, orientation_xyzw)
    got_pos, got_quat = state.get_body_pose("mover")

    np.testing.assert_allclose(got_pos, want_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat, want_quat, atol=1e-6)


def test_free_joint_write_honours_world_mask(host_model):
    """A masked write moves the selected world and leaves the other alone."""
    state = MjWarpSceneState(host_model, nworld=2)
    before_pos, _ = state.get_body_pose("mover", world_index=1)
    target = np.array([0.3, 0.3, 0.9])

    state.set_free_joint_pose(
        "mover_free",
        target,
        np.array([0.0, 0.0, 0.0, 1.0]),
        world_mask=np.array([True, False]),
    )

    moved, _ = state.get_body_pose("mover", world_index=0)
    untouched, _ = state.get_body_pose("mover", world_index=1)
    np.testing.assert_allclose(moved, target, atol=1e-6)
    np.testing.assert_allclose(untouched, before_pos, atol=1e-6)


def test_free_joint_write_rejects_non_free_joint(host_model):
    state = MjWarpSceneState(host_model, nworld=1)
    with pytest.raises(ValueError, match="Joint 'nope' not found"):
        state.set_free_joint_pose("nope", np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))


def test_nworld_must_be_positive(host_model):
    with pytest.raises(ValueError, match="nworld must be >= 1"):
        MjWarpSceneState(host_model, nworld=0)
