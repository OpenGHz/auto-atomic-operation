"""Tests for MJWarp end-effector control.

The acceptance ladder in ``_completion`` is where a gripper either advances the
stage or hangs, so each rung is exercised on its own rather than only through a
happy path: a blocked gripper that never reaches its commanded angle must still
be accepted, and an opening gripper must not be accepted before its release
settling has elapsed.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.operator_handler import MjWarpEefControl  # noqa: E402
from auto_atom.backend.mjwarp.operator_state import register_operator  # noqa: E402
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402
from auto_atom.execution_model import ControlSignal  # noqa: E402

# A one-joint arm carrying a two-finger gripper, plus a graspable box. The
# fingers are driven by a slide joint so a commanded "close" actually travels.
_GRIPPER_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="arm_base" pos="0 0 0.3">
      <geom name="base_geom" type="box" size="0.03 0.03 0.02"/>
      <body name="link1" pos="0 0 0.03">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom name="l1" type="box" size="0.02 0.02 0.01"/>
        <body name="tool" pos="0 0 0.02">
          <site name="eef_pose" pos="0 0 0.02"/>
          <geom name="tool_geom" type="box" size="0.02 0.01 0.01"/>
          <body name="left_pad" pos="0 0 0.04">
            <joint name="j_left" type="slide" axis="1 0 0" range="0 0.06"/>
            <geom name="left_finger_pad" type="box" size="0.004 0.01 0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_j1" joint="j1" kp="60"/>
    <position name="act_grip" joint="j_left" kp="80"/>
  </actuator>
</mujoco>
"""


class _StubIK:
    pass


@pytest.fixture(scope="module")
def gripper_model():
    return mujoco.MjModel.from_xml_string(_GRIPPER_XML)


@pytest.fixture
def control(gripper_model):
    state = MjWarpSceneState(gripper_model, nworld=2)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1",),
        eef_actuators=("act_grip",),
        ik_solver=_StubIK(),
    )
    return MjWarpEefControl(
        state=state,
        operator=operator,
        eef_open_value=0.0,
        eef_close_value=0.05,
        eef_tolerance=0.002,
        timeout_steps=50,
        settle_steps=5,
    )


def test_target_value_prefers_explicit_joint_positions(control):
    """A named angle means that angle, not the gripper's travel limit."""
    assert control.target_value(close=True) == 0.05
    assert control.target_value(close=False) == 0.0
    assert control.target_value(close=True, joint_positions=[0.02]) == 0.02


def test_control_writes_ctrl_and_requests_one_step(control):
    """Per-world commands land, and physics advances once inside a tick.

    This is the deferral contract from design doc 3.8: the handler asks for a
    step per call, and the tick boundary collapses those into one.
    """
    state = control.state
    timestep = float(state.host_model.opt.timestep)
    before = state.data.time.numpy().copy()

    with state.deferred_step():
        for world in range(state.nworld):
            mask = np.zeros(state.nworld, dtype=bool)
            mask[world] = True
            control.control(close=True, world_mask=mask)

    after = state.data.time.numpy()
    np.testing.assert_allclose(after - before, timestep, rtol=1e-6)
    # The gripper actuator carries the close command in every world.
    ctrl = state.get_ctrl()
    for world in range(state.nworld):
        assert ctrl[world][control.operator.eef_actuator_ids[0]] == pytest.approx(0.05)


def test_closing_reaches_when_the_angle_is_met(control):
    """With nothing in the way the gripper reaches its commanded angle."""
    state = control.state
    result = None
    for _ in range(120):
        with state.deferred_step():
            result = control.control(close=True)
        if result.signals[0] == ControlSignal.REACHED:
            break

    assert result.signals[0] == ControlSignal.REACHED
    assert result.details[0]["event"] == "eef_reached"


def test_require_grasp_without_a_target_is_a_config_error(control):
    """Native reports this as a config error, not a failed grasp."""
    with control.state.deferred_step():
        result = control.control(close=True, require_grasp=True)

    for world in range(control.state.nworld):
        assert result.signals[world] == ControlSignal.FAILED
        assert result.details[world]["failure_category"] == "missing_grasp_target"


def test_blocked_gripper_is_accepted_after_settling(control):
    """A gripper stopped by an object may never reach its commanded angle.

    Without this rung a successful grasp on a stiff object reads as a timeout,
    so the test drives the command far past the joint's range to emulate being
    physically blocked and asserts acceptance still happens.
    """
    state = control.state
    control.eef_close_value = 10.0  # unreachable: joint range stops at 0.06
    result = None
    for _ in range(60):
        with state.deferred_step():
            result = control.control(close=True)
        if result.signals[0] == ControlSignal.REACHED:
            break

    assert result.signals[0] == ControlSignal.REACHED
    assert result.details[0]["steps"] >= 30, "must wait for the settle window"


def test_opening_waits_for_release_settling(control):
    """Release settling is a hold, so acceptance cannot happen before it."""
    state = control.state
    control.release_settle_steps = 8

    with state.deferred_step():
        first = control.control(close=False)
    assert first.signals[0] == ControlSignal.RUNNING, "1 step < 8 settle steps"

    for _ in range(20):
        with state.deferred_step():
            result = control.control(close=False)
        if result.signals[0] == ControlSignal.REACHED:
            break
    assert result.signals[0] == ControlSignal.REACHED
    assert result.details[0]["steps"] >= 8


def test_timeout_is_reported_when_nothing_completes(control):
    """A command that never completes times out rather than running forever."""
    state = control.state
    control.timeout_steps = 4
    control.settle_steps = 1000  # keep the grasp rung out of reach
    control.eef_close_value = 10.0
    control.eef_tolerance = 1e-9

    result = None
    for _ in range(10):
        with state.deferred_step():
            result = control.control(close=True)
        if result.signals[0] == ControlSignal.TIMED_OUT:
            break

    assert result.signals[0] == ControlSignal.TIMED_OUT
    assert result.details[0]["event"] == "eef_timeout"


def test_a_changed_command_restarts_the_step_counter(control):
    """Switching command mid-stage must not inherit the old command's progress."""
    state = control.state
    for _ in range(4):
        with state.deferred_step():
            control.control(close=True)
    assert int(control._steps[0]) == 4

    with state.deferred_step():
        result = control.control(close=False)
    assert result.details[0]["steps"] == 1


def test_masked_worlds_keep_their_own_counters(control):
    """Only the selected world advances, since the runtime drives one at a time."""
    state = control.state
    only_world_1 = np.array([False, True])

    for _ in range(3):
        with state.deferred_step():
            control.control(close=True, world_mask=only_world_1)

    assert int(control._steps[0]) == 0
    assert int(control._steps[1]) == 3


def test_mask_shape_is_validated(control):
    with pytest.raises(ValueError, match=r"world_mask must have shape \(2,\)"):
        control.control(close=True, world_mask=np.array([True]))


def test_grasp_details_are_reported_when_a_target_is_given(control):
    """The verdict's two halves are both visible to a failure diagnostic."""
    state = control.state
    with state.deferred_step():
        result = control.control(close=True, target_body_name="tool")

    check = result.details[0]["grasp_check"]
    assert set(check) == {
        "left_contact",
        "right_contact",
        "lateral_ok",
        "lateral_error",
        "lateral_threshold",
    }


def test_missing_eef_actuator_is_refused(gripper_model):
    """An operator with no gripper cannot be commanded, and says so."""
    state = MjWarpSceneState(gripper_model, nworld=1)
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1",),
        eef_actuators=(),
        ik_solver=_StubIK(),
    )
    control = MjWarpEefControl(state=state, operator=operator)

    with pytest.raises(ValueError, match="no eef_actuators"):
        control.control(close=True)
