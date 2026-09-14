"""Tests for MJWarp per-operator control state.

Registration is the point where an operator's addressing and frames are fixed
for the rest of a reset, so a mistake here surfaces much later as an arm that
moves to a plausible-looking wrong pose. These tests pin the frame relationships
against independent derivations rather than against a restatement of the
registration code.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.frames import base_to_world  # noqa: E402
from auto_atom.backend.mjwarp.operator_state import (  # noqa: E402
    get_base_pose,
    get_eef_pose_in_base,
    get_eef_pose_in_world,
    override_base_pose,
    register_operator,
)
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402

# An arm whose base is deliberately away from the origin and rotated, so a
# registration that ignored the base frame would still produce plausible
# numbers while getting the tool offset wrong.
_ARM_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="arm_base" pos="0.3 -0.2 0.1" quat="0.9238795 0 0 0.3826834">
      <geom name="base_geom" type="box" size="0.04 0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <geom name="l1" type="box" size="0.05 0.02 0.02"/>
        <body name="link2" pos="0.1 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2.5 0.08"/>
          <geom name="l2" type="box" size="0.05 0.02 0.02"/>
          <body name="tool" pos="0.1 0 0">
            <geom name="tool_geom" type="box" size="0.01 0.01 0.01"/>
            <site name="eef_pose" pos="0.02 0 0"/>
            <body name="left_pad" pos="0.03 -0.02 0">
              <geom name="left_finger_pad" type="box" size="0.005 0.005 0.01"/>
            </body>
            <body name="right_pad" pos="0.03 0.02 0">
              <geom name="right_finger_pad" type="box" size="0.005 0.005 0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_j1" joint="j1" kp="60"/>
    <position name="act_j2" joint="j2" kp="60"/>
    <position name="act_grip" joint="j2" kp="10"/>
  </actuator>
</mujoco>
"""


class _StubIK:
    """Registration only needs a non-None solver; it never calls it."""


@pytest.fixture(scope="module")
def arm_model():
    return mujoco.MjModel.from_xml_string(_ARM_XML)


@pytest.fixture
def registered(arm_model):
    state = MjWarpSceneState(arm_model, nworld=2)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1", "act_j2"),
        eef_actuators=("act_grip",),
        ik_solver=_StubIK(),
    )
    return state, operator


def test_registration_resolves_actuators_and_joints(registered):
    state, operator = registered

    np.testing.assert_array_equal(
        operator.arm_actuator_ids, state.actuator_ids(["act_j1", "act_j2"])
    )
    np.testing.assert_array_equal(
        operator.eef_actuator_ids, state.actuator_ids(["act_grip"])
    )
    assert operator.arm_qpos_indices.size == 2
    assert operator.joint_mode


def test_base_pose_is_the_root_body_pose_per_world(registered):
    state, operator = registered

    want_pos, want_quat = state.get_body_pose_batch("arm_base")
    got_pos, got_quat = get_base_pose(operator)

    assert got_pos.shape == (2, 3) and got_quat.shape == (2, 4)
    np.testing.assert_allclose(got_pos, want_pos, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(got_quat, want_quat, rtol=1e-6, atol=1e-7)


def test_tool_offset_reconstructs_the_world_eef_pose(registered):
    """base * tool_offset == world eef pose, which is what makes it a tool offset.

    Checked by composing the stored offset back out to world rather than by
    re-running the same conversion, so a wrong-direction conversion fails.
    """
    state, operator = registered
    want_pos, want_quat = get_eef_pose_in_world(state, operator)

    for world in range(state.nworld):
        got_pos, got_quat = base_to_world(
            operator.tool_offset_position[world],
            operator.tool_offset_orientation[world],
            operator.base_position[world],
            operator.base_orientation[world],
        )
        np.testing.assert_allclose(got_pos, want_pos[world], rtol=1e-5, atol=1e-6)
        assert np.allclose(got_quat, want_quat[world], atol=1e-5) or np.allclose(
            got_quat, -want_quat[world], atol=1e-5
        )


def test_eef_pose_in_base_equals_the_tool_offset_at_rest(registered):
    """At registration the arm has not moved, so the two must agree."""
    state, operator = registered

    pos_b, quat_b = get_eef_pose_in_base(state, operator)

    np.testing.assert_allclose(
        pos_b, operator.tool_offset_position, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        quat_b, operator.tool_offset_orientation, rtol=1e-6, atol=1e-7
    )


def test_eef_pose_in_base_tracks_arm_motion(registered):
    """Moving a joint changes the eef pose in base, not the base itself."""
    state, operator = registered
    before_pos, _ = get_eef_pose_in_base(state, operator)
    base_before, _ = get_base_pose(operator)

    state.set_joint_positions(
        operator.arm_qpos_indices,
        np.array([0.6, -0.4]),
        operator.arm_dof_indices,
    )

    after_pos, _ = get_eef_pose_in_base(state, operator)
    assert np.linalg.norm(after_pos - before_pos) > 1e-3
    # The base frame is a snapshot: an arm move must not redefine it.
    np.testing.assert_array_equal(get_base_pose(operator)[0], base_before)


def test_home_state_is_snapshotted(registered):
    state, operator = registered

    assert operator.home_arm_qpos.shape == (2, 2)
    assert operator.home_ctrl.shape == (2, int(state.host_model.nu))
    np.testing.assert_allclose(operator.home_arm_qpos, 0.0, atol=1e-7)


def test_override_base_pose_refreshes_the_tool_offset(registered):
    """A moved base must not leave a stale offset behind.

    The offset is defined relative to the base, so the control path would
    otherwise convert targets against one frame and measure the tool in another.
    """
    state, operator = registered
    before = operator.tool_offset_position.copy()

    override_base_pose(
        state,
        operator,
        np.array([-0.20, -0.5, 0.075]),
        np.array([0.0, 0.0, 0.70710678, 0.70710678]),
    )

    assert not np.allclose(operator.tool_offset_position, before)
    np.testing.assert_allclose(
        operator.base_position, np.tile([-0.20, -0.5, 0.075], (2, 1)), atol=1e-9
    )
    # And the refreshed offset still composes back to the true world eef pose.
    want_pos, _ = get_eef_pose_in_world(state, operator)
    for world in range(state.nworld):
        got_pos, _ = base_to_world(
            operator.tool_offset_position[world],
            operator.tool_offset_orientation[world],
            operator.base_position[world],
            operator.base_orientation[world],
        )
        np.testing.assert_allclose(got_pos, want_pos[world], rtol=1e-5, atol=1e-6)


def test_override_accepts_per_world_base_poses(registered):
    state, operator = registered

    positions = np.array([[-0.2, -0.5, 0.075], [0.1, 0.2, 0.3]])
    identity = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    override_base_pose(state, operator, positions, identity)

    np.testing.assert_allclose(operator.base_position, positions, atol=1e-9)
    assert not np.allclose(
        operator.tool_offset_position[0], operator.tool_offset_position[1]
    )


def test_override_rejects_an_unmappable_row_count(registered):
    state, operator = registered

    with pytest.raises(ValueError, match="broadcastable"):
        override_base_pose(
            state, operator, np.zeros((3, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
        )


def test_registration_rejects_mocap_mode(arm_model):
    """Mocap mode is not implemented, so it is refused rather than half-done."""
    state = MjWarpSceneState(arm_model, nworld=1)

    with pytest.raises(ValueError, match="joint mode only"):
        register_operator(
            state,
            name="arm",
            root_body="arm_base",
            eef_site="eef_pose",
            arm_actuators=(),
            ik_solver=_StubIK(),
        )


def test_registration_requires_an_ik_solver(arm_model):
    state = MjWarpSceneState(arm_model, nworld=1)

    with pytest.raises(ValueError, match="no ik_solver"):
        register_operator(
            state,
            name="arm",
            root_body="arm_base",
            eef_site="eef_pose",
            arm_actuators=("act_j1",),
            ik_solver=None,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"joint_control_mode": "teleport"}, "Unsupported joint_control_mode"),
        ({"joint_interp_speed": 0.0}, "must be > 0"),
    ],
)
def test_registration_rejects_bad_control_settings(arm_model, kwargs, match):
    state = MjWarpSceneState(arm_model, nworld=1)

    with pytest.raises(ValueError, match=match):
        register_operator(
            state,
            name="arm",
            root_body="arm_base",
            eef_site="eef_pose",
            arm_actuators=("act_j1",),
            ik_solver=_StubIK(),
            **kwargs,
        )
