"""Tests for the MJWarp ``OperatorHandler`` seam.

This layer owns translation, not control, so the tests check that config objects
reach the state machines as the right arguments -- a waypoint's step bounds and
tolerance overrides actually applied, a grasp target resolved to its MJCF body
name rather than its logical name -- and that the contract's own methods are
satisfied.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.arm_control import MjWarpArmControl  # noqa: E402
from auto_atom.backend.mjwarp.operator_handler import (  # noqa: E402
    MjWarpOperatorHandler,
)
from auto_atom.backend.mjwarp.handlers import MjWarpObjectHandler  # noqa: E402
from auto_atom.backend.mjwarp.ik import MjWarpIkCaller  # noqa: E402
from auto_atom.backend.mjwarp.eef_control import MjWarpEefControl  # noqa: E402
from auto_atom.backend.mjwarp.operator_state import register_operator  # noqa: E402
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402
from auto_atom.config.motion import (  # noqa: E402
    EefControlConfig,
    PoseControlConfig,
    WaypointToleranceConfig,
)
from auto_atom.contracts import OperatorHandler  # noqa: E402
from auto_atom.execution_model import ControlSignal  # noqa: E402

_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="arm_base" pos="0.2 -0.1 0.1">
      <geom name="base_geom" type="box" size="0.03 0.03 0.02"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <geom name="l1" type="box" size="0.06 0.02 0.02"/>
        <body name="tool" pos="0.12 0 0">
          <site name="eef_pose" pos="0.02 0 0"/>
          <geom name="tool_geom" type="box" size="0.01 0.01 0.01"/>
          <body name="left_pad" pos="0.02 0 0.02">
            <joint name="j_left" type="slide" axis="1 0 0" range="0 0.06"/>
            <geom name="left_finger_pad" type="box" size="0.004 0.01 0.01"/>
          </body>
        </body>
      </body>
    </body>
    <body name="widget_body" pos="0.6 0.0 0.3">
      <freejoint name="widget_free"/>
      <geom name="widget_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position name="act_j1" joint="j1" kp="200"/>
    <position name="act_grip" joint="j_left" kp="80"/>
  </actuator>
</mujoco>
"""


class _RecordingArm(MjWarpArmControl):
    """Captures the arguments move() was called with."""

    def move(self, *args, **kwargs):  # type: ignore[override]
        self.last_call = (args, kwargs)
        return super().move(*args, **kwargs)


class _RecordingEef(MjWarpEefControl):
    def control(self, **kwargs):  # type: ignore[override]
        self.last_call = kwargs
        return super().control(**kwargs)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(_XML)


@pytest.fixture
def handler(model):
    state = MjWarpSceneState(model, nworld=2)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1",),
        eef_actuators=("act_grip",),
        ik_solver=object(),
    )
    arm = _RecordingArm(
        state=state,
        operator=operator,
        ik=MjWarpIkCaller(solver=_Solver(), nworld=2, operator_name="arm"),
        position_tolerance=0.012,
        orientation_tolerance=0.10,
    )
    eef = _RecordingEef(state=state, operator=operator, eef_close_value=0.05)
    return MjWarpOperatorHandler(state=state, operator=operator, arm=arm, eef=eef)


class _Solver:
    def solve(self, target, seed):
        return np.zeros(np.asarray(seed).shape)


def test_handler_satisfies_the_contract(handler):
    assert isinstance(handler, OperatorHandler)
    assert handler.name == "arm"


def test_batched_pose_getters_report_every_world(handler):
    eef = handler.get_end_effector_pose()
    base = handler.get_base_pose()

    assert np.asarray(eef.position).shape == (2, 3)
    assert np.asarray(eef.orientation).shape == (2, 4)
    assert np.asarray(base.position).shape == (2, 3)


def test_waypoint_step_bounds_and_tolerances_are_forwarded(handler):
    """A waypoint's own limits must reach the state machine, not be dropped."""
    pose = PoseControlConfig(
        position=[0.5, 0.0, 0.3],
        orientation=[0.0, 0.0, 0.0, 1.0],
        max_linear_step=0.02,
        max_angular_step=0.2,
        tolerance=WaypointToleranceConfig(position=0.03, orientation=0.15),
    )

    with handler.control_tick():
        handler.move_to_pose(pose, None, env_mask=np.array([True, False]))

    _, kwargs = handler.arm.last_call
    assert kwargs["waypoint_linear_step"] == pytest.approx(0.02)
    assert kwargs["waypoint_angular_step"] == pytest.approx(0.2)
    assert kwargs["waypoint_position_tolerance"] == pytest.approx(0.03)
    assert kwargs["waypoint_orientation_tolerance"] == pytest.approx(0.15)


def test_absent_waypoint_tolerance_forwards_none(handler):
    """No override means the operator default applies, not zero tolerance."""
    pose = PoseControlConfig(position=[0.5, 0.0, 0.3], orientation=[0, 0, 0, 1.0])

    with handler.control_tick():
        handler.move_to_pose(pose, None, env_mask=np.array([True, False]))

    _, kwargs = handler.arm.last_call
    assert kwargs["waypoint_position_tolerance"] is None
    assert kwargs["waypoint_orientation_tolerance"] is None


def test_a_position_only_waypoint_holds_current_orientation(handler):
    """Snapping to identity would rotate the wrist for no reason."""
    _, current = (
        handler.get_end_effector_pose().position,
        handler.get_end_effector_pose().orientation,
    )
    pose = PoseControlConfig(position=[0.5, 0.0, 0.3])

    with handler.control_tick():
        handler.move_to_pose(pose, None, env_mask=np.array([True, False]))

    args, _ = handler.arm.last_call
    np.testing.assert_allclose(args[1], current[0], atol=1e-6)


def test_the_command_key_changes_with_the_waypoint(handler):
    """Two different waypoints must not share progress state."""
    first = PoseControlConfig(position=[0.5, 0.0, 0.3], orientation=[0, 0, 0, 1.0])
    second = PoseControlConfig(position=[0.4, 0.1, 0.3], orientation=[0, 0, 0, 1.0])
    mask = np.array([True, False])

    with handler.control_tick():
        handler.move_to_pose(first, None, env_mask=mask)
    key_first = handler.arm.last_call[1]["command_key"]
    with handler.control_tick():
        handler.move_to_pose(second, None, env_mask=mask)
    key_second = handler.arm.last_call[1]["command_key"]

    assert key_first != key_second


def test_a_pose_without_a_position_is_rejected(handler):
    """The runtime resolves waypoints before dispatch, so this is a bug signal."""
    with pytest.raises(ValueError, match="without a position"):
        handler.move_to_pose(PoseControlConfig(), None)


def test_grasp_target_is_resolved_to_its_mjcf_body_name(handler):
    """Logical name and body name differ, and contacts are keyed on the body.

    Passing the logical name through would look right and silently find no
    contacts, so the grasp would never confirm.
    """
    target = MjWarpObjectHandler(
        name="widget", state=handler.state, body_name="widget_body"
    )

    with handler.control_tick():
        handler.control_eef(
            EefControlConfig(close=True), target, env_mask=np.array([True, False])
        )

    assert handler.eef.last_call["target_body_name"] == "widget_body"


def test_eef_config_fields_are_forwarded(handler):
    config = EefControlConfig(close=True, joint_positions=[0.02])

    with handler.control_tick():
        handler.control_eef(config, None, env_mask=np.array([True, False]))

    call = handler.eef.last_call
    assert call["close"] is True
    assert call["joint_positions"] == [0.02]
    assert call["require_grasp"] is False


def test_tolerance_getters_report_configured_values(handler):
    position, orientation = handler.get_reached_tolerances()
    assert position == pytest.approx(0.012)
    assert orientation == pytest.approx(0.10)

    # Unconfigured PLACED tolerance is unconstrained, per the contract.
    assert handler.get_placed_tolerances() == (None, None)


def test_home_joint_positions_are_recorded_and_applied(handler):
    handler.set_home_joint_positions({"j1": 0.4})

    np.testing.assert_allclose(handler.operator.home_arm_qpos[:, 0], 0.4)
    got = handler.state.get_joint_positions(handler.operator.arm_qpos_indices)
    np.testing.assert_allclose(got[:, 0], 0.4, atol=1e-5)


def test_home_joint_positions_can_be_recorded_without_applying(handler):
    before = handler.state.get_joint_positions(handler.operator.arm_qpos_indices).copy()

    handler.set_home_joint_positions({"j1": 0.4}, apply_home=False)

    np.testing.assert_allclose(handler.operator.home_arm_qpos[:, 0], 0.4)
    np.testing.assert_allclose(
        handler.state.get_joint_positions(handler.operator.arm_qpos_indices), before
    )


def test_an_unknown_home_joint_is_rejected(handler):
    """Silently dropping it would leave a pose the config did not ask for."""
    with pytest.raises(ValueError, match="no arm joint"):
        handler.set_home_joint_positions({"nonexistent": 0.1})


def test_control_tick_collapses_both_halves_into_one_step(handler):
    """Arm and gripper in one tick must advance physics exactly once.

    If they did not share the boundary they would step twice, and the two halves
    would disagree about how much time a tick represents.
    """
    timestep = float(handler.state.host_model.opt.timestep)
    before = handler.state.data.time.numpy().copy()
    pose = PoseControlConfig(position=[0.5, 0.0, 0.3], orientation=[0, 0, 0, 1.0])
    mask = np.array([True, False])

    with handler.control_tick():
        handler.move_to_pose(pose, None, env_mask=mask)
        handler.control_eef(EefControlConfig(close=True), None, env_mask=mask)

    after = handler.state.data.time.numpy()
    np.testing.assert_allclose(after - before, timestep, rtol=1e-6)


def test_missing_eef_control_is_refused(model):
    state = MjWarpSceneState(model, nworld=1)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1",),
        ik_solver=object(),
    )
    handler = MjWarpOperatorHandler(
        state=state,
        operator=operator,
        arm=MjWarpArmControl(
            state=state,
            operator=operator,
            ik=MjWarpIkCaller(solver=_Solver(), nworld=1),
        ),
        eef=None,
    )

    with pytest.raises(ValueError, match="no end-effector control"):
        handler.control_eef(EefControlConfig(close=True), None)


def test_required_collaborators_are_validated(model):
    state = MjWarpSceneState(model, nworld=1)

    with pytest.raises(ValueError, match="non-None 'arm'"):
        MjWarpOperatorHandler(state=state, operator=object(), arm=None)


def test_signals_come_back_batched(handler):
    """The runtime reads signals[env_index], so every world needs an entry."""
    pose = PoseControlConfig(position=[0.5, 0.0, 0.3], orientation=[0, 0, 0, 1.0])

    with handler.control_tick():
        result = handler.move_to_pose(pose, None, env_mask=np.array([True, False]))

    assert len(result.signals) == 2
    assert len(result.details) == 2
    assert result.signals[1] == ControlSignal.RUNNING
