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
      <!-- Static and *nested* below a parent with non-identity pos and quat.
           A world->parent-local conversion bug is invisible on a direct child
           of worldbody, whose parent frame is the identity. -->
      <body name="nested_static" pos="0.06 0.01 0.02" quat="0.7071068 0 0 0.7071068">
        <geom name="nested_geom" type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
    <body name="mover" pos="-0.4 0.15 0.5" quat="0.8446232 0.1913417 0.4619398 0.1913417">
      <freejoint name="mover_free"/>
      <geom name="mover_geom" type="box" size="0.02 0.02 0.02"/>
      <site name="mover_site" pos="0 0.01 0.02"/>
    </body>
    <!-- No geoms: exercises the zero-radius support-geometry branch. -->
    <body name="marker" pos="0.3 0.4 0.6">
      <site name="marker_site"/>
    </body>
    <camera name="side_cam" pos="0.8 -0.9 0.5" xyaxes="1 0 0 0 0.5 0.87" fovy="47"/>
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


# ----------------------------------------------------------------------
# Randomization-constraint reads
# ----------------------------------------------------------------------


def test_camera_pose_matches_native_derivation(host_model, native_basis, warp_state):
    """Camera pose agrees with what the native get_camera_model computes.

    get_camera_model itself also folds in per-camera config (clip ranges,
    enabled streams), which is env-level state rather than simulator state, so
    the comparison targets the simulator reads it performs.
    """
    from auto_atom.utils.pose import quaternion_from_matrix_3x3

    cam_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_CAMERA, "side_cam")
    want_pos = np.asarray(native_basis.data.cam_xpos[cam_id], dtype=np.float64)
    want_quat = quaternion_from_matrix_3x3(
        np.asarray(native_basis.data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    )

    got_pos, got_quat = warp_state.get_camera_pose("side_cam")

    np.testing.assert_allclose(got_pos, want_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat, want_quat, atol=1e-6)


def test_camera_fovy_converts_degrees_to_radians(host_model, warp_state):
    got = warp_state.get_camera_fovy_radians("side_cam")
    assert got == pytest.approx(np.deg2rad(47.0))


def test_default_clip_range_matches_native_derivation(host_model, warp_state):
    """Near/far in metres, derived as the native camera model derives them."""
    want_near = float(host_model.vis.map.znear) * float(host_model.stat.extent)
    want_far = float(host_model.vis.map.zfar) * float(host_model.stat.extent)

    near, far = warp_state.default_clip_range_m()

    assert near == pytest.approx(want_near)
    assert far == pytest.approx(want_far)
    assert 0.0 < near < far


@pytest.mark.parametrize("entity", ["holder", "mover"])
def test_support_geometry_matches_native_basis(native_basis, warp_state, entity):
    want = native_basis.get_support_geometry(entity)
    got = warp_state.get_support_geometry(entity)

    np.testing.assert_allclose(got.center, want.center, atol=1e-6)
    assert got.radius == pytest.approx(want.radius, abs=1e-6)


def test_support_geometry_of_geomless_body_matches_native(native_basis, warp_state):
    """A body with no geoms yields a zero-radius sphere at its origin."""
    want = native_basis.get_support_geometry("marker")
    got = warp_state.get_support_geometry("marker")

    np.testing.assert_allclose(got.center, want.center, atol=1e-6)
    assert got.radius == pytest.approx(want.radius, abs=1e-9)
    assert got.radius == 0.0


def test_randomization_reads_raise_keyerror_like_native(native_basis, warp_state):
    """Unknown names raise KeyError, matching the native contract.

    This differs from the frame readers, which raise ValueError -- the native
    path is inconsistent here and the adapter reproduces it per method rather
    than unifying it, so a caller's except clause keeps working.
    """
    with pytest.raises(KeyError):
        native_basis.get_support_geometry("nope")
    with pytest.raises(KeyError):
        warp_state.get_support_geometry("nope")
    with pytest.raises(KeyError, match="Camera 'nope' not found"):
        warp_state.get_camera_pose("nope")


# ----------------------------------------------------------------------
# Static-body placement (the world -> parent-local path)
# ----------------------------------------------------------------------


def test_free_joint_id_resolution(host_model):
    """Free-joint bodies resolve a joint; static bodies resolve -1."""
    state = MjWarpSceneState(host_model, nworld=1)

    assert state.resolve_free_joint_id("mover") >= 0
    assert state.resolve_free_joint_id("mover", "mover_free") >= 0
    assert state.resolve_free_joint_id("holder") == -1
    assert state.resolve_free_joint_id("nested_static") == -1
    # An override naming a joint that does not exist falls back to inspection.
    assert state.resolve_free_joint_id("mover", "no_such_joint") >= 0


def test_static_pose_round_trips_through_nested_parent(host_model):
    """A nested static body lands at the requested *world* pose.

    This is the assertion that catches a world->parent-local error: the parent
    has both a translation and a rotation, so writing the world pose directly
    into body_pos/body_quat would land somewhere else entirely.
    """
    state = MjWarpSceneState(host_model, nworld=2)
    target_pos = np.array([0.42, -0.13, 0.37])
    target_quat = np.array([0.1913417, 0.4619398, 0.1913417, 0.8446232])

    state.set_static_body_pose("nested_static", target_pos, target_quat)

    for world in range(2):
        got_pos, got_quat = state.get_body_pose("nested_static", world_index=world)
        np.testing.assert_allclose(got_pos, target_pos, atol=1e-6)
        assert np.allclose(got_quat, target_quat, atol=1e-6) or np.allclose(
            got_quat, -target_quat, atol=1e-6
        )


def test_static_pose_matches_native_handler_math(host_model):
    """The conversion agrees with the native object handler's, step for step."""
    target_pos = np.array([0.42, -0.13, 0.37])
    target_quat_xyzw = np.array([0.1913417, 0.4619398, 0.1913417, 0.8446232])

    # Native: reproduce MujocoObjectHandler.set_pose's static branch verbatim.
    native_model = mujoco.MjModel.from_xml_string(_SCENE_XML)
    native_data = mujoco.MjData(native_model)
    mujoco.mj_forward(native_model, native_data)
    bid = mujoco.mj_name2id(native_model, mujoco.mjtObj.mjOBJ_BODY, "nested_static")
    parent_id = int(native_model.body_parentid[bid])
    parent_pos = native_data.xpos[parent_id].astype(np.float64)
    parent_mat = native_data.xmat[parent_id].reshape(3, 3).astype(np.float64)
    native_model.body_pos[bid] = parent_mat.T @ (target_pos - parent_pos)
    world_quat_wxyz = target_quat_xyzw[[3, 0, 1, 2]]
    inverse_parent = np.empty(4, dtype=np.float64)
    mujoco.mju_negQuat(inverse_parent, native_data.xquat[parent_id].astype(np.float64))
    local_quat = np.empty(4, dtype=np.float64)
    mujoco.mju_mulQuat(local_quat, inverse_parent, world_quat_wxyz)
    native_model.body_quat[bid] = local_quat
    mujoco.mj_forward(native_model, native_data)

    state = MjWarpSceneState(host_model, nworld=1)
    state.set_static_body_pose("nested_static", target_pos, target_quat_xyzw)

    np.testing.assert_allclose(
        state.model.body_pos.numpy()[0][bid], native_model.body_pos[bid], atol=1e-9
    )
    np.testing.assert_allclose(
        state.model.body_quat.numpy()[0][bid], native_model.body_quat[bid], atol=1e-9
    )


def test_static_pose_honours_world_mask(host_model):
    """Per-world scenery placement: one shared model, independent worlds.

    The native path needs one MjModel per replica to express this, because
    body_pos is model state there rather than a batched field.
    """
    state = MjWarpSceneState(host_model, nworld=2)
    before_pos, _ = state.get_body_pose("nested_static", world_index=1)
    target = np.array([0.42, -0.13, 0.37])

    state.set_static_body_pose(
        "nested_static",
        target,
        np.array([0.0, 0.0, 0.0, 1.0]),
        world_mask=np.array([True, False]),
    )

    moved, _ = state.get_body_pose("nested_static", world_index=0)
    untouched, _ = state.get_body_pose("nested_static", world_index=1)
    np.testing.assert_allclose(moved, target, atol=1e-6)
    np.testing.assert_allclose(untouched, before_pos, atol=1e-6)
    assert not np.allclose(moved, untouched, atol=1e-6)


# ----------------------------------------------------------------------
# Per-world distinct poses (what randomization actually writes)
# ----------------------------------------------------------------------


def test_free_joint_accepts_one_pose_per_world(host_model):
    """Randomization samples each env independently, so rows must differ.

    The executor fills one batched PoseState slot per env and applies it with a
    single masked call, so a write API that could only broadcast would collapse
    every world onto the same sample.
    """
    state = MjWarpSceneState(host_model, nworld=2)
    positions = np.array([[0.10, 0.20, 0.30], [-0.40, 0.50, 0.60]])
    orientations = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.3826834, 0.0, 0.9238795]])

    state.set_free_joint_pose("mover_free", positions, orientations)

    for world in range(2):
        got_pos, got_quat = state.get_body_pose("mover", world_index=world)
        np.testing.assert_allclose(got_pos, positions[world], atol=1e-6)
        assert np.allclose(got_quat, orientations[world], atol=1e-6) or np.allclose(
            got_quat, -orientations[world], atol=1e-6
        )


def test_static_body_accepts_one_pose_per_world(host_model):
    """Same for scenery: per-world rows survive the parent-local conversion."""
    state = MjWarpSceneState(host_model, nworld=2)
    positions = np.array([[0.42, -0.13, 0.37], [0.05, 0.44, 0.21]])
    orientations = np.array(
        [[0.0, 0.0, 0.0, 1.0], [0.1913417, 0.4619398, 0.1913417, 0.8446232]]
    )

    state.set_static_body_pose("nested_static", positions, orientations)

    for world in range(2):
        got_pos, got_quat = state.get_body_pose("nested_static", world_index=world)
        np.testing.assert_allclose(got_pos, positions[world], atol=1e-6)
        assert np.allclose(got_quat, orientations[world], atol=1e-6) or np.allclose(
            got_quat, -orientations[world], atol=1e-6
        )
    # And the two worlds genuinely disagree, so this is not a broadcast.
    first, _ = state.get_body_pose("nested_static", world_index=0)
    second, _ = state.get_body_pose("nested_static", world_index=1)
    assert not np.allclose(first, second, atol=1e-6)


def test_per_world_rows_index_by_absolute_world_under_mask(host_model):
    """Row w applies to world w, not to the w-th selected world.

    The native handler indexes pose.position[env_index], so a masked write must
    keep using absolute world indices rather than positions within the mask.
    """
    state = MjWarpSceneState(host_model, nworld=2)
    before, _ = state.get_body_pose("mover", world_index=0)
    positions = np.array([[0.10, 0.20, 0.30], [-0.40, 0.50, 0.60]])
    orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))

    # Write world 1 only: it must receive row 1, not row 0.
    state.set_free_joint_pose(
        "mover_free", positions, orientations, world_mask=np.array([False, True])
    )

    untouched, _ = state.get_body_pose("mover", world_index=0)
    written, _ = state.get_body_pose("mover", world_index=1)
    np.testing.assert_allclose(untouched, before, atol=1e-6)
    np.testing.assert_allclose(written, positions[1], atol=1e-6)


def test_single_pose_still_broadcasts(host_model):
    """A batch-1 pose, as a PoseState of batch size 1 yields, broadcasts."""
    state = MjWarpSceneState(host_model, nworld=2)
    target = np.array([[0.15, 0.25, 0.35]])
    orientation = np.array([[0.0, 0.0, 0.0, 1.0]])

    state.set_free_joint_pose("mover_free", target, orientation)

    for world in range(2):
        got_pos, _ = state.get_body_pose("mover", world_index=world)
        np.testing.assert_allclose(got_pos, target[0], atol=1e-6)


def test_pose_batch_shape_is_validated(host_model):
    state = MjWarpSceneState(host_model, nworld=2)
    good_quat = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))

    with pytest.raises(ValueError, match="pose batch must be 1 or nworld"):
        state.set_free_joint_pose(
            "mover_free", np.zeros((3, 3)), np.tile(good_quat[0], (3, 1))
        )
    with pytest.raises(ValueError, match="must share a batch dimension"):
        state.set_free_joint_pose("mover_free", np.zeros((2, 3)), good_quat[:1])
    with pytest.raises(ValueError, match=r"position must be \(3,\)"):
        state.set_free_joint_pose("mover_free", np.zeros((2, 4)), good_quat)


# ----------------------------------------------------------------------
# Batched reads (the shape PoseState wants)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("body_name", ["holder", "mover", "nested_static"])
def test_body_pose_batch_agrees_with_per_world_reads(warp_state, body_name):
    """The batched read is the per-world read stacked, in PoseState's shape."""
    positions, orientations = warp_state.get_body_pose_batch(body_name)

    assert positions.shape == (warp_state.nworld, 3)
    assert orientations.shape == (warp_state.nworld, 4)
    for world in range(warp_state.nworld):
        want_pos, want_quat = warp_state.get_body_pose(body_name, world_index=world)
        np.testing.assert_allclose(positions[world], want_pos, atol=1e-6)
        np.testing.assert_allclose(orientations[world], want_quat, atol=1e-6)


@pytest.mark.parametrize("site_name", ["holder_site", "mover_site"])
def test_site_pose_batch_agrees_with_per_world_reads(warp_state, site_name):
    positions, orientations = warp_state.get_site_pose_batch(site_name)

    assert positions.shape == (warp_state.nworld, 3)
    assert orientations.shape == (warp_state.nworld, 4)
    for world in range(warp_state.nworld):
        want_pos, want_quat = warp_state.get_site_pose(site_name, world_index=world)
        np.testing.assert_allclose(positions[world], want_pos, atol=1e-6)
        np.testing.assert_allclose(orientations[world], want_quat, atol=1e-6)


def test_batch_read_reflects_per_world_distinct_writes(host_model):
    """A batched read must not collapse worlds that hold different poses."""
    state = MjWarpSceneState(host_model, nworld=2)
    positions = np.array([[0.10, 0.20, 0.30], [-0.40, 0.50, 0.60]])
    orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1))

    state.set_free_joint_pose("mover_free", positions, orientations)
    got_positions, _ = state.get_body_pose_batch("mover")

    np.testing.assert_allclose(got_positions, positions, atol=1e-6)
    assert not np.allclose(got_positions[0], got_positions[1], atol=1e-6)


def test_batch_read_matches_native_stacked_read(host_model, native_basis):
    """One world's batched row equals what the native env reports.

    The native batched env stacks per-replica get_body_pose results; with one
    world the stack is a single row, which is the comparison that pins the
    shape and dtype convention against the native path.
    """
    state = MjWarpSceneState(host_model, nworld=1)
    positions, orientations = state.get_body_pose_batch("holder")
    want_pos, want_quat = native_basis.get_body_pose("holder")

    assert positions.shape == (1, 3)
    np.testing.assert_allclose(positions[0], want_pos, atol=1e-6)
    np.testing.assert_allclose(orientations[0], want_quat, atol=1e-6)
    assert positions.dtype == want_pos.dtype


@pytest.mark.parametrize("body", ["mover", "holder", "nested_static"])
def test_set_object_pose_dispatches_by_body_kind(host_model, body):
    """One entry point places a body whichever mechanism it has."""
    state = MjWarpSceneState(host_model, nworld=1)
    target_pos = np.array([0.2, 0.3, 0.45])
    target_quat = np.array([0.0, 0.3826834, 0.0, 0.9238795])

    state.set_object_pose(body, target_pos, target_quat)

    got_pos, got_quat = state.get_body_pose(body)
    np.testing.assert_allclose(got_pos, target_pos, atol=1e-6)
    assert np.allclose(got_quat, target_quat, atol=1e-6) or np.allclose(
        got_quat, -target_quat, atol=1e-6
    )
