"""Tests for MJWarp backend-level grasp and contact queries.

These answer stage post-conditions, so the distinction that matters is between
*grasping* (both fingers, centred) and merely *contacting* (any touch). A query
that conflated them would let a press stage report success on a grasp, or fail a
grasp stage that only brushed the object.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.grasp_queries import MjWarpGraspQueries  # noqa: E402
from auto_atom.backend.mjwarp.operator_state import register_operator  # noqa: E402
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402

# Gripper closed on "held": both pads touch it. "brushed" touches only the left
# pad, so it is contacted but not grasped. "loose" touches nothing.
_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="arm_base" pos="0 0 0.5">
      <geom name="base_geom" type="box" size="0.03 0.03 0.01"/>
      <body name="tool" pos="0 0 -0.03">
        <!-- Offset in y from the pads' centre line, so the grasped object has a
             genuine lateral offset in the eef frame for the threshold to judge.
             A site coincident with the object would give zero error and pass
             any threshold, testing nothing. -->
        <site name="eef_pose" pos="0 0.01 -0.03"/>
        <geom name="tool_geom" type="box" size="0.04 0.01 0.01"/>
        <body name="left_pad" pos="-0.028 0 -0.03">
          <geom name="left_finger_pad" type="box" size="0.005 0.01 0.02"/>
        </body>
        <body name="right_pad" pos="0.028 0 -0.03">
          <geom name="right_finger_pad" type="box" size="0.005 0.01 0.02"/>
        </body>
      </body>
    </body>
    <body name="held" pos="0 0 0.44">
      <!-- Free joint, not static: moving a static body does not update
           geom_xpos on MJWarp (design doc 3.9), so a static target could not
           be moved to test that a verdict tracks state. -->
      <freejoint name="held_free"/>
      <body name="held_shell">
        <geom name="held_geom" type="box" size="0.024 0.01 0.02"/>
      </body>
    </body>
    <body name="brushed" pos="-0.040 0 0.44">
      <geom name="brushed_geom" type="box" size="0.009 0.01 0.02"/>
    </body>
    <body name="loose" pos="0.5 0.5 0.1">
      <geom name="loose_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act_dummy" joint="dummy" kp="10"/>
  </actuator>
  <equality/>
</mujoco>
"""

# The scene above needs a joint for its actuator; declare it on the base.
_XML = _XML.replace(
    '<geom name="base_geom" type="box" size="0.03 0.03 0.01"/>',
    '<joint name="dummy" type="slide" axis="0 0 1" range="-0.1 0.1"/>\n'
    '      <geom name="base_geom" type="box" size="0.03 0.03 0.01"/>',
)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(_XML)


@pytest.fixture
def queries(model):
    state = MjWarpSceneState(model, nworld=2)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_dummy",),
        ik_solver=object(),
    )
    return state, MjWarpGraspQueries(state, operator)


_NAMES = {"held": "held", "brushed": "brushed", "loose": "loose"}


def test_a_two_sided_contact_reads_as_grasped(queries):
    state, q = queries

    got = q.is_object_grasped("held")

    assert got.shape == (state.nworld,)
    assert got.all(), "both pads touch it in every world"


def test_a_one_sided_touch_is_not_a_grasp(queries):
    """One finger brushing an object must not satisfy a grasp post-condition."""
    _, q = queries

    assert not q.is_object_grasped("brushed").any()


def test_an_untouched_object_is_neither_grasped_nor_contacted(queries):
    _, q = queries

    assert not q.is_object_grasped("loose").any()
    assert not q.is_operator_contacting("loose").any()


def test_a_one_sided_touch_still_counts_as_contact(queries):
    """This is the press/push post-condition: any touch counts."""
    _, q = queries

    assert q.is_operator_contacting("brushed").all()


def test_grasping_any_object_is_the_union(queries):
    _, q = queries

    assert q.is_operator_grasping(_NAMES).all()
    assert not q.is_operator_grasping({"loose": "loose"}).any()


def test_grasped_object_name_reports_the_logical_name(queries):
    """The runtime keys grasp bindings on the logical name, not the body name."""
    _, q = queries

    assert q.grasped_object_name(_NAMES, 0) == "held"
    assert q.grasped_object_name({"loose": "loose"}, 0) is None


def test_grasped_object_name_bounds_checks_the_world(queries):
    _, q = queries

    with pytest.raises(IndexError, match=r"world_index must be in \[0, 2\)"):
        q.grasped_object_name(_NAMES, 2)


def test_a_moved_object_stops_being_grasped(queries):
    """The verdict tracks state rather than being fixed at construction."""
    state, q = queries
    assert q.is_object_grasped("held").all()

    # Free-joint write: a static-body write would leave geom_xpos stale and the
    # verdict would (correctly, for that stale state) still read as grasped.
    state.set_object_pose(
        "held", np.array([0.6, 0.6, 0.6]), np.array([0.0, 0.0, 0.0, 1.0])
    )
    state.step()

    assert not q.is_object_grasped("held").any()


def test_worlds_are_answered_independently(queries):
    """Moving one world's object must not change the other world's verdict."""
    state, q = queries

    state.set_object_pose(
        "held",
        np.array([0.6, 0.6, 0.6]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        world_mask=np.array([True, False]),
    )
    state.step()

    got = q.is_object_grasped("held")
    assert not got[0] and got[1]


def test_lateral_threshold_can_reject_an_off_centre_grasp(model):
    """Two-sided contact is not enough when the object sits off the grasp line.

    Built as a separate instance because the threshold is construction-time
    config, and this is the case where contact alone would wrongly say yes.
    """
    state = MjWarpSceneState(model, nworld=1)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_dummy",),
        ik_solver=object(),
    )
    strict = MjWarpGraspQueries(state, operator, lateral_threshold=1e-6, grasp_axis=2)
    permissive = MjWarpGraspQueries(state, operator, lateral_threshold=0.0)

    # The eef site sits 0.03 below the tool while the object is centred on the
    # pads, so there is a real lateral offset for the strict threshold to catch.
    assert permissive.is_object_grasped("held").all()
    assert not strict.is_object_grasped("held").any()


def test_target_subtree_is_included(queries):
    """held's collision geom hangs off a child body, and must still count."""
    state, q = queries

    bodies = q._bodies_for("held")
    shell = mujoco.mj_name2id(state.host_model, mujoco.mjtObj.mjOBJ_BODY, "held_shell")
    assert shell in bodies
