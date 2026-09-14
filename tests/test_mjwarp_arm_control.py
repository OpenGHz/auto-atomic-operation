"""Tests for MJWarp arm motion.

The load-bearing behaviour here is that shaping affects only the command and
never the completion test: a clamped sub-step must not read as reaching the
waypoint. The IK-failure path matters just as much -- a miss must hold the
previous command rather than writing a fresh one, since a zeroed command would
drop the arm under gravity.

IK is stubbed. A real analytical solver would make these tests about the solver
rather than about the state machine, and the state machine is what this module
owns.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.arm_control import MjWarpArmControl  # noqa: E402
from auto_atom.backend.mjwarp.ik import MjWarpIkCaller  # noqa: E402
from auto_atom.backend.mjwarp.operator_state import (  # noqa: E402
    get_eef_pose_in_world,
    register_operator,
)
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402
from auto_atom.execution_model import ControlSignal  # noqa: E402

_ARM_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="arm_base" pos="0.2 -0.1 0.1">
      <geom name="base_geom" type="box" size="0.03 0.03 0.02"/>
      <body name="link1" pos="0 0 0.04">
        <joint name="j1" type="hinge" axis="0 0 1" range="-2.9 2.9"/>
        <geom name="l1" type="box" size="0.06 0.02 0.02"/>
        <body name="link2" pos="0.12 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
          <geom name="l2" type="box" size="0.06 0.02 0.02"/>
          <body name="tool" pos="0.12 0 0">
            <geom name="tool_geom" type="box" size="0.01 0.01 0.01"/>
            <site name="eef_pose" pos="0.02 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="act_j1" joint="j1" kp="200"/>
    <position name="act_j2" joint="j2" kp="200"/>
  </actuator>
</mujoco>
"""


class _StubSolver:
    """Returns a configurable solution, recording the targets it was asked for."""

    def __init__(self, solution=None):
        self.solution = solution
        self.targets = []

    def solve(self, target, seed):
        self.targets.append(
            (
                np.asarray(target.position, dtype=np.float64).reshape(-1),
                np.asarray(target.orientation, dtype=np.float64).reshape(-1),
            )
        )
        if callable(self.solution):
            return self.solution(target, seed)
        return self.solution


@pytest.fixture(scope="module")
def arm_model():
    return mujoco.MjModel.from_xml_string(_ARM_XML)


def _build(arm_model, solution=None, nworld=2, **kwargs):
    state = MjWarpSceneState(arm_model, nworld=nworld)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1", "act_j2"),
        ik_solver=object(),
    )
    solver = _StubSolver(solution)
    ik = MjWarpIkCaller(solver=solver, nworld=nworld, operator_name="arm")
    control = MjWarpArmControl(state=state, operator=operator, ik=ik, **kwargs)
    return state, operator, control, solver


def test_ik_target_is_expressed_in_the_operator_base_frame(arm_model):
    """The solver must receive a base-frame target, not the world-frame goal.

    The base sits at (0.2, -0.1, 0.1), so a world goal converted correctly
    cannot equal the world goal itself.
    """
    state, operator, control, solver = _build(arm_model, np.zeros(2))
    goal = np.array([0.5, 0.0, 0.3])

    with state.deferred_step():
        control.move(
            goal, np.array([0.0, 0.0, 0.0, 1.0]), world_mask=np.array([True, False])
        )

    asked_position = solver.targets[0][0]
    assert not np.allclose(asked_position, goal), "target was not converted"
    np.testing.assert_allclose(
        asked_position, goal - operator.base_position[0], atol=1e-5
    )


def test_a_reached_goal_reports_reached(arm_model):
    """Commanding the pose the tool already holds completes immediately."""
    state, operator, control, _ = _build(arm_model, np.zeros(2))
    positions, orientations = get_eef_pose_in_world(state, operator)

    with state.deferred_step():
        result = control.move(
            positions[0], orientations[0], world_mask=np.array([True, False])
        )

    assert result.signals[0] == ControlSignal.REACHED
    assert result.details[0]["event"] == "pose_reached"


def test_a_clamped_substep_does_not_read_as_reaching_the_waypoint(arm_model):
    """Completion is measured against the final waypoint, never the sub-target.

    With a 2 mm step bound and a far goal, the commanded sub-target is reachable
    while the waypoint is not, so a state machine testing the wrong one would
    wrongly report REACHED on the first tick.
    """
    state, _, control, _ = _build(
        arm_model,
        np.zeros(2),
        max_linear_step=0.002,
        position_tolerance=0.01,
    )
    far_goal = np.array([0.9, 0.4, 0.5])

    with state.deferred_step():
        result = control.move(
            far_goal, np.array([0.0, 0.0, 0.0, 1.0]), world_mask=np.array([True, False])
        )

    assert result.signals[0] == ControlSignal.RUNNING
    assert result.details[0]["position_error"] > 0.01


def test_ik_failure_holds_the_previous_command(arm_model):
    """A miss must not write ctrl; a zeroed command would drop the arm."""
    state, operator, control, _ = _build(arm_model, None)
    state.set_ctrl(operator.arm_actuator_ids, np.array([0.3, -0.2]))
    before = state.get_ctrl().copy()

    with state.deferred_step():
        result = control.move(
            np.array([0.5, 0.0, 0.3]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            world_mask=np.array([True, False]),
        )

    np.testing.assert_allclose(
        state.get_ctrl()[0][operator.arm_actuator_ids],
        before[0][operator.arm_actuator_ids],
    )
    assert result.signals[0] == ControlSignal.RUNNING
    assert result.details[0]["ik_failure_streak"] == 1


def test_persistent_ik_failure_fails_with_a_specific_category(arm_model):
    """An unreachable target fails fast, distinguishable from slow motion."""
    state, _, control, _ = _build(
        arm_model, None, ik_unreachable_threshold=3, timeout_steps=1000
    )
    mask = np.array([True, False])

    for _ in range(3):
        with state.deferred_step():
            result = control.move(
                np.array([9.0, 9.0, 9.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                command_key="wp",
                world_mask=mask,
            )

    assert result.signals[0] == ControlSignal.FAILED
    assert result.details[0]["failure_category"] == "ik_unreachable"


def test_timeout_is_reported_before_the_unreachable_threshold(arm_model):
    """A solvable but unreachable-in-time goal times out rather than failing."""
    state, _, control, _ = _build(
        arm_model,
        np.zeros(2),
        timeout_steps=3,
        max_linear_step=0.0005,
        ik_unreachable_threshold=1000,
    )
    mask = np.array([True, False])

    for _ in range(3):
        with state.deferred_step():
            result = control.move(
                np.array([0.9, 0.4, 0.5]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                command_key="wp",
                world_mask=mask,
            )

    assert result.signals[0] == ControlSignal.TIMED_OUT
    assert result.details[0]["event"] == "move_timeout"


def test_a_new_waypoint_resets_progress(arm_model):
    """The previous waypoint's step count and stall history do not carry over."""
    state, _, control, _ = _build(arm_model, np.zeros(2), max_linear_step=0.001)
    mask = np.array([True, False])
    goal = np.array([0.9, 0.4, 0.5])

    for _ in range(4):
        with state.deferred_step():
            control.move(
                goal,
                np.array([0.0, 0.0, 0.0, 1.0]),
                command_key="first",
                world_mask=mask,
            )
    assert int(control._steps[0]) == 4

    with state.deferred_step():
        result = control.move(
            goal, np.array([0.0, 0.0, 0.0, 1.0]), command_key="second", world_mask=mask
        )
    assert result.details[0]["steps"] == 1


def test_one_step_per_tick_across_worlds(arm_model):
    """Both worlds commanded in a tick still advance physics exactly once."""
    state, _, control, _ = _build(arm_model, np.zeros(2))
    timestep = float(arm_model.opt.timestep)
    before = state.data.time.numpy().copy()

    with state.deferred_step():
        for world in range(state.nworld):
            mask = np.zeros(state.nworld, dtype=bool)
            mask[world] = True
            control.move(
                np.array([0.4, 0.0, 0.3]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                world_mask=mask,
            )

    after = state.data.time.numpy()
    np.testing.assert_allclose(after - before, timestep, rtol=1e-6)


def test_joint_command_is_delta_clamped(arm_model):
    """A branch-jumping solution is approached over ticks, not lunged at."""
    state, operator, control, _ = _build(arm_model, np.array([2.5, -2.0]))
    operator.max_joint_delta = 0.1

    with state.deferred_step():
        control.move(
            np.array([0.4, 0.0, 0.3]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            world_mask=np.array([True, False]),
        )

    commanded = state.get_ctrl()[0][operator.arm_actuator_ids]
    # Read back through float32 device storage, so the bound is checked at
    # float32 tolerance (design doc 3.6) rather than exactly: the clamp computes
    # 0.1 in float64 and the device returns 0.10000000149011612.
    assert float(np.max(np.abs(commanded))) <= 0.1 + 1e-6
    # Direction preserved: the whole delta scaled by the worst joint (0.1/2.5).
    np.testing.assert_allclose(commanded, [0.1, -0.08], rtol=1e-6, atol=1e-7)


def test_worlds_are_controlled_independently(arm_model):
    """An unmasked world keeps its command untouched."""
    state, operator, control, _ = _build(arm_model, np.array([0.2, 0.1]))
    state.set_ctrl(operator.arm_actuator_ids, np.array([[0.9, 0.9], [0.8, 0.8]]))

    with state.deferred_step():
        control.move(
            np.array([0.4, 0.0, 0.3]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            world_mask=np.array([True, False]),
        )

    ctrl = state.get_ctrl()
    assert not np.allclose(ctrl[0][operator.arm_actuator_ids], [0.9, 0.9])
    np.testing.assert_allclose(ctrl[1][operator.arm_actuator_ids], [0.8, 0.8])


def test_waypoint_tolerance_overrides_the_operator_default(arm_model):
    """A waypoint that asks for looser tolerance gets it."""
    state, operator, control, _ = _build(
        arm_model, np.zeros(2), position_tolerance=1e-6, orientation_tolerance=1e-6
    )
    positions, orientations = get_eef_pose_in_world(state, operator)
    nudged = positions[0] + np.array([0.004, 0.0, 0.0])
    mask = np.array([True, False])

    with state.deferred_step():
        strict = control.move(nudged, orientations[0], world_mask=mask)
    assert strict.signals[0] == ControlSignal.RUNNING

    with state.deferred_step():
        loose = control.move(
            nudged,
            orientations[0],
            waypoint_position_tolerance=0.01,
            waypoint_orientation_tolerance=0.5,
            world_mask=mask,
        )
    assert loose.signals[0] == ControlSignal.REACHED


def test_reset_clears_progress_and_streaks(arm_model):
    state, _, control, _ = _build(arm_model, None)
    mask = np.array([True, False])
    for _ in range(3):
        with state.deferred_step():
            control.move(
                np.array([9.0, 9.0, 9.0]),
                np.array([0.0, 0.0, 0.0, 1.0]),
                command_key="wp",
                world_mask=mask,
            )
    assert control.ik.failure_streak(0) == 3

    control.reset()

    assert control.ik.failure_streak(0) == 0
    assert int(control._steps[0]) == 0


def test_mask_shape_is_validated(arm_model):
    state, _, control, _ = _build(arm_model, np.zeros(2))

    with pytest.raises(ValueError, match=r"world_mask must have shape \(2,\)"):
        control.move(
            np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), world_mask=np.array([True])
        )


def test_solve_once_interpolate_is_refused(arm_model):
    """Refused explicitly rather than silently behaving like per_step_ik."""
    state = MjWarpSceneState(arm_model, nworld=1)
    state.forward()
    operator = register_operator(
        state,
        name="arm",
        root_body="arm_base",
        eef_site="eef_pose",
        arm_actuators=("act_j1", "act_j2"),
        ik_solver=object(),
        joint_control_mode="solve_once_interpolate",
    )

    with pytest.raises(ValueError, match="does not implement"):
        MjWarpArmControl(
            state=state,
            operator=operator,
            ik=MjWarpIkCaller(solver=_StubSolver(), nworld=1),
        )
