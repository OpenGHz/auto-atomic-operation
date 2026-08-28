"""Focused regression tests for object-centric motion goals."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from auto_atom.framework import (
    EefControlConfig,
    PoseControlConfig,
    StageConfig,
    TaskFileConfig,
)
from auto_atom.mock import (
    MockObjectHandler,
    MockOperatorHandler,
    MockSceneBackend,
)
from auto_atom.policy_eval import (
    ConfigDrivenDemoPolicy,
    ConfigDrivenEnvAction,
    ConfigDrivenPolicyAction,
)
from auto_atom.pose_goal import (
    resolve_axis_alignment_orientation,
    resolve_axis_in_world,
)
from auto_atom.runtime import (
    ActiveStageState,
    ControlResult,
    ControlSignal,
    ExecutionContext,
    GraspBinding,
    PrimitiveAction,
    ResolvedMotionGoal,
    StageExecutionPlan,
    TaskFlowBuilder,
    TaskRunner,
)
from auto_atom.stage_execution import StageExecution, _placed_condition_satisfied
from auto_atom.utils.pose import (
    PoseState,
    compose_pose,
    inverse_pose,
    multiply_quaternions,
    quaternion_angular_distance,
)


def _axis_angle_quaternion(
    axis: tuple[float, float, float],
    angle: float,
) -> tuple[float, float, float, float]:
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= np.linalg.norm(unit_axis)
    sine = np.sin(angle / 2.0)
    return (
        float(unit_axis[0] * sine),
        float(unit_axis[1] * sine),
        float(unit_axis[2] * sine),
        float(np.cos(angle / 2.0)),
    )


def _assert_pose_equivalent(
    actual: PoseState,
    expected: PoseState,
    *,
    atol: float = 1.0e-10,
) -> None:
    np.testing.assert_allclose(actual.position, expected.position, atol=atol)
    assert abs(
        float(np.dot(actual.orientation[0], expected.orientation[0]))
    ) == pytest.approx(1.0, abs=atol)


def _task_file(operator_names: tuple[str, ...] = ("arm",)) -> TaskFileConfig:
    return TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {"env_name": "object_centric_motion_goal", "stages": []},
            "task_operators": {name: {} for name in operator_names},
        }
    )


def _context(
    *,
    batch_size: int = 1,
    operator_names: tuple[str, ...] = ("arm",),
    object_poses: dict[str, PoseState] | None = None,
) -> tuple[ExecutionContext, MockSceneBackend]:
    operators = {
        name: MockOperatorHandler(
            operator_name=name,
            batch_size=batch_size,
            end_effector_pose=PoseState(
                position=np.asarray(
                    [
                        [0.1 + 0.2 * env_index, 0.03 * operator_index, 0.4]
                        for env_index in range(batch_size)
                    ],
                    dtype=np.float64,
                ),
                orientation=np.asarray(
                    [[0.0, 0.0, 0.0, 1.0]] * batch_size,
                    dtype=np.float64,
                ),
            ),
        )
        for operator_index, name in enumerate(operator_names)
    }
    objects = {
        name: MockObjectHandler(name=name, pose=pose)
        for name, pose in (object_poses or {}).items()
    }
    backend = MockSceneBackend(
        env_name="object_centric_motion_goal",
        batch_size=batch_size,
        operators=operators,
        objects=objects,
    )
    task_file = _task_file(operator_names)
    context = ExecutionContext(
        config=task_file.task,
        backend=backend,
        task_file=task_file,
    )
    return context, backend


def _held_object_pose(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation_goal: dict[str, object] | None = None,
) -> PoseControlConfig:
    payload: dict[str, object] = {
        "controlled_frame": {"kind": "held_object"},
        "position": position,
        "reference": "world",
    }
    if orientation_goal is not None:
        payload["orientation_goal"] = orientation_goal
    return PoseControlConfig.model_validate(payload)


def _axis_alignment_goal() -> dict[str, object]:
    return {
        "kind": "axis_alignment",
        "controlled_axis": [0.0, 0.0, 1.0],
        "target_axis": {
            "vector": [0.0, 1.0, 0.0],
            "reference": "world",
        },
        "direction": "same",
    }


def test_nonzero_grasp_transform_retargets_object_goal_to_eef_in_se3() -> None:
    """The runtime must invert the measured EEF-to-object rigid transform."""
    eef_from_object = PoseState(
        position=(0.11, -0.04, 0.07),
        orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), 0.63),
    )
    current_world_from_eef = PoseState(
        position=(0.25, -0.13, 0.42),
        orientation=_axis_angle_quaternion((1.0, 0.0, 0.0), -0.31),
    )
    current_world_from_object = compose_pose(
        current_world_from_eef,
        eef_from_object,
    )
    context, backend = _context(
        object_poses={"plate": current_world_from_object},
    )
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = current_world_from_eef
    backend.get_grasped_object_name = (
        lambda operator_name, env_index: "plate"
        if operator_name == "arm" and env_index == 0
        else None
    )
    binding = GraspBinding(
        env_index=0,
        operator_name="arm",
        object_name="plate",
        eef_from_object=eef_from_object,
    )
    target_world_from_object = PoseState(
        position=(0.62, 0.18, 0.37),
        orientation=_axis_angle_quaternion((0.0, 1.0, 0.0), 1.17),
    )
    pose = _held_object_pose(
        position=tuple(target_world_from_object.position[0]),
        orientation_goal={
            "kind": "fixed",
            "quaternion_xyzw": target_world_from_object.orientation[0].tolist(),
        },
    )

    goal = TaskRunner._resolve_motion_goal(
        env_index=0,
        operator=operator,
        pose=pose,
        target=None,
        backend=backend,
        grasp_binding=binding,
    )

    expected_world_from_eef = compose_pose(
        target_world_from_object,
        inverse_pose(eef_from_object),
    )
    command_world_from_eef = PoseState(
        position=goal.command_pose.position,
        orientation=goal.command_pose.orientation,
    )
    _assert_pose_equivalent(goal.controlled_world_pose, target_world_from_object)
    _assert_pose_equivalent(command_world_from_eef, expected_world_from_eef)
    _assert_pose_equivalent(
        compose_pose(command_world_from_eef, eef_from_object),
        target_world_from_object,
    )
    assert goal.controlled_object_name == "plate"
    assert context.grasp_bindings == {}


@pytest.mark.parametrize("twist", [-2.1, 0.47, 2.35])
def test_axis_alignment_and_placed_ignore_twist_about_plate_normal(
    twist: float,
) -> None:
    """Only the plate's local Z direction matters; in-plane twist stays free."""
    target_position = (0.52, -0.08, 0.31)
    swing = _axis_angle_quaternion((1.0, 0.0, 0.0), -np.pi / 2.0)
    twist_about_local_z = _axis_angle_quaternion((0.0, 0.0, 1.0), twist)
    current_orientation = multiply_quaternions(swing, twist_about_local_z)
    current_plate_pose = PoseState(
        position=target_position,
        orientation=current_orientation,
    )
    context, backend = _context(
        object_poses={
            "plate": current_plate_pose,
            "rack": PoseState(position=target_position),
        }
    )
    pose = PoseControlConfig.model_validate(
        {
            **_held_object_pose(
                position=target_position,
                orientation_goal=_axis_alignment_goal(),
            ).model_dump(mode="json"),
            "tolerance": {"position": 0.001, "orientation": 0.01},
        }
    )
    semantic_goal = ResolvedMotionGoal(
        configured_pose=pose,
        controlled_world_pose=PoseState(
            position=target_position,
            orientation=swing,
        ),
        command_pose=PoseControlConfig(
            position=target_position,
            orientation=swing,
            reference="world",
        ),
        controlled_object_name="plate",
        target_axis_world=np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    )

    position_error, orientation_error, _ = TaskRunner.motion_goal_errors(
        env_index=0,
        operator=backend.get_operator_handler("arm"),
        backend=backend,
        goal=semantic_goal,
        require_held=False,
    )

    np.testing.assert_allclose(position_error, np.zeros(3), atol=1.0e-12)
    assert orientation_error == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(
        resolve_axis_in_world((0.0, 0.0, 1.0), current_plate_pose),
        (0.0, 1.0, 0.0),
        atol=1.0e-12,
    )
    assert quaternion_angular_distance(current_orientation, swing) == pytest.approx(
        abs(twist),
        abs=1.0e-12,
    )

    stage = StageConfig.model_validate(
        {
            "name": "place_plate",
            "object": "rack",
            "operation": "place",
            "operator": "arm",
            "param": {
                "pre_move": [pose.model_dump(mode="json")],
                "placed_reference": "pre_move",
                "placed_tolerance": {
                    "position": 0.001,
                    "orientation": 0.01,
                },
            },
        }
    )
    plan = StageExecutionPlan(stage_index=0, stage=stage, operator_name="arm")
    assert _placed_condition_satisfied(
        env_index=0,
        context=context,
        plan=plan,
        is_grasping=False,
        target_object_pose=semantic_goal.controlled_world_pose,
        target_motion_goal=semantic_goal,
        held_object_name="plate",
    )


def test_grasp_bindings_are_isolated_and_partial_reset_clears_one_env() -> None:
    plate_pose = PoseState(
        position=np.asarray([[0.4, 0.0, 0.2], [0.7, -0.1, 0.25]]),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2),
    )
    context, backend = _context(
        batch_size=2,
        operator_names=("arm", "helper"),
        object_poses={"plate": plate_pose},
    )

    env0_arm = context.capture_grasp_binding(0, "arm", "plate")
    env1_arm = context.capture_grasp_binding(1, "arm", "plate")
    env0_helper = context.capture_grasp_binding(0, "helper", "plate")

    assert context.get_grasp_binding(0, "arm") is env0_arm
    assert context.get_grasp_binding(1, "arm") is env1_arm
    assert context.get_grasp_binding(0, "helper") is env0_helper
    assert not np.allclose(
        env0_arm.eef_from_object.position,
        env1_arm.eef_from_object.position,
    )

    execution = StageExecution(
        context,
        [],
        actions_factory=lambda _plan: [],
    )
    execution.reset(
        np.asarray([True, False]),
        details_factory=lambda env_index: {"env_index": env_index},
    )

    assert context.get_grasp_binding(0, "arm") is None
    assert context.get_grasp_binding(0, "helper") is None
    assert context.get_grasp_binding(1, "arm") is env1_arm
    assert backend.batch_size == 2


def test_verified_release_clears_only_its_operator_binding() -> None:
    context, backend = _context(
        batch_size=2,
        object_poses={
            "plate": PoseState(
                position=np.asarray([[0.4, 0.0, 0.2], [0.6, 0.0, 0.2]]),
                orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2),
            ),
            "rack": PoseState().broadcast_to(2),
        },
    )
    env0_binding = context.capture_grasp_binding(0, "arm", "plate")
    env1_binding = context.capture_grasp_binding(1, "arm", "plate")
    stage = StageConfig.model_validate(
        {
            "name": "release_plate",
            "object": "rack",
            "operation": "place",
            "operator": "arm",
            "param": {},
        }
    )
    plan = StageExecutionPlan(stage_index=0, stage=stage, operator_name="arm")
    release = PrimitiveAction(kind="eef", eef=EefControlConfig(close=False))
    active = ActiveStageState(
        plan=plan,
        operator=backend.get_operator_handler("arm"),
        target=backend.get_object_handler("rack"),
        actions=[release],
    )
    execution = StageExecution(
        context,
        [plan],
        actions_factory=lambda _plan: [release],
    )

    failure = execution._update_grasp_binding_after_eef(0, active, release)

    assert failure is None
    assert context.get_grasp_binding(0, "arm") is None
    assert context.get_grasp_binding(1, "arm") is env1_binding
    assert env0_binding.object_name == "plate"


def test_held_object_goal_without_binding_fails_fast() -> None:
    _, backend = _context(object_poses={"plate": PoseState()})
    pose = _held_object_pose(position=(0.2, 0.1, 0.3))

    with pytest.raises(RuntimeError, match="verified grasp binding"):
        TaskRunner._resolve_motion_goal(
            env_index=0,
            operator=backend.get_operator_handler("arm"),
            pose=pose,
            target=None,
            backend=backend,
            grasp_binding=None,
        )


def test_held_object_goal_rejects_changed_object_identity() -> None:
    _, backend = _context(object_poses={"plate": PoseState(), "cup": PoseState()})
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "cup" if operator_name == "arm" and env_index == 0 else None
    )
    binding = GraspBinding(
        env_index=0,
        operator_name="arm",
        object_name="plate",
        eef_from_object=PoseState(),
    )
    pose = _held_object_pose(position=(0.2, 0.1, 0.3))

    with pytest.raises(RuntimeError, match="expected 'plate', got 'cup'"):
        TaskRunner._resolve_motion_goal(
            env_index=0,
            operator=backend.get_operator_handler("arm"),
            pose=pose,
            target=None,
            backend=backend,
            grasp_binding=binding,
        )


def test_held_object_slip_gets_stable_follow_up_command_without_rebinding() -> None:
    context, backend = _context(
        object_poses={"plate": PoseState(position=(0.0, 0.0, 0.0))}
    )
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = PoseState(position=(0.0, 0.0, 0.0))
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {"kind": "held_object"},
            "position": [1.0, 0.0, 0.0],
            "orientation_goal": {
                "kind": "fixed",
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reference": "world",
            "static": True,
            "tolerance": {"position": 0.001, "orientation": 0.001},
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)
    env_mask = np.asarray([True])

    first = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    semantic_goal = action.resolved_motion_goal
    assert first.signals[0] == ControlSignal.RUNNING
    assert semantic_goal is not None
    initial_command = semantic_goal.command_pose

    # The EEF reaches the original command while the plate lags by 20 cm.
    # Passing an incompatible remeasurement proves that the snapshotted goal
    # and original binding identity are retained instead of silently rebound.
    backend.get_object_handler("plate").set_pose(PoseState(position=(0.8, 0.0, 0.0)))
    replacement_binding = GraspBinding(
        env_index=0,
        operator_name="arm",
        object_name="plate",
        eef_from_object=PoseState(position=(99.0, 0.0, 0.0)),
    )
    second = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=replacement_binding,
    )

    assert second.signals[0] == ControlSignal.RUNNING
    assert second.details[0]["event"] == "controlled_frame_correction"
    assert action.resolved_motion_goal is semantic_goal
    assert semantic_goal.controlled_object_name == "plate"
    np.testing.assert_allclose(
        semantic_goal.controlled_world_pose.position[0],
        [1.0, 0.0, 0.0],
    )
    assert semantic_goal.command_pose is not initial_command
    np.testing.assert_allclose(
        semantic_goal.command_pose.position,
        [1.2, 0.0, 0.0],
        atol=1.0e-12,
    )

    # A changed correction command takes the mock controller two updates.  It
    # must remain byte-for-byte stable on the RUNNING tick, otherwise real
    # backends would reset their per-command timeout/progress state forever.
    correction_command = semantic_goal.command_pose
    correction_payload = correction_command.model_dump(mode="json")
    third = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert third.signals[0] == ControlSignal.RUNNING
    assert semantic_goal.command_pose is not correction_command
    assert semantic_goal.command_pose.model_dump(mode="json") == correction_payload
    assert operator._progress[0] == 1

    backend.get_object_handler("plate").set_pose(PoseState(position=(1.0, 0.0, 0.0)))
    fourth = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert fourth.signals[0] == ControlSignal.REACHED


def test_live_held_reference_resumes_after_one_stable_correction_leg() -> None:
    context, backend = _context(
        object_poses={
            "plate": PoseState(position=(0.0, 0.0, 0.0)),
            "rack": PoseState(position=(1.0, 0.0, 0.0)),
        }
    )
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = PoseState(position=(0.0, 0.0, 0.0))
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {"kind": "held_object"},
            "position": [0.0, 0.0, 0.0],
            "orientation_goal": {
                "kind": "fixed",
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reference": "object",
            "tolerance": {"position": 0.001, "orientation": 0.001},
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)
    env_mask = np.asarray([True])
    rack = backend.get_object_handler("rack")

    TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    first_goal = action.resolved_motion_goal
    assert first_goal is not None
    np.testing.assert_allclose(first_goal.controlled_world_pose.position, [[1.0, 0, 0]])

    backend.get_object_handler("plate").set_pose(PoseState(position=(0.8, 0.0, 0.0)))
    second = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert second.signals[0] == ControlSignal.RUNNING
    correction_goal = action.resolved_motion_goal
    assert correction_goal is not None
    assert correction_goal is not first_goal
    correction_payload = correction_goal.command_pose.model_dump(mode="json")

    rack.set_pose(PoseState(position=(2.0, 0.0, 0.0)))
    third = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert third.signals[0] == ControlSignal.RUNNING
    assert action.resolved_motion_goal is correction_goal
    assert correction_goal.command_pose.model_dump(mode="json") == correction_payload
    np.testing.assert_allclose(
        correction_goal.controlled_world_pose.position,
        [[1.0, 0, 0]],
    )

    backend.get_object_handler("plate").set_pose(PoseState(position=(1.0, 0.0, 0.0)))
    fourth = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert fourth.signals[0] == ControlSignal.RUNNING
    assert fourth.details[0]["event"] == "controlled_frame_correction"
    refreshed_goal = action.resolved_motion_goal
    assert refreshed_goal is not None
    assert refreshed_goal is not correction_goal
    assert action.resolved_pose is correction_goal.command_pose
    assert action.resolved_pose is not refreshed_goal.command_pose
    np.testing.assert_allclose(
        refreshed_goal.controlled_world_pose.position,
        [[2.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        refreshed_goal.command_pose.position,
        [2.2, 0.0, 0.0],
        atol=1.0e-12,
    )

    # The refreshed semantic error creates a correction from the current
    # plate/EEF state.  It must not issue the nominal grasp-binding command,
    # which would move the already-corrected plate back toward the old pose.
    backend.get_object_handler("plate").set_pose(PoseState(position=(2.0, 0.0, 0.0)))
    fifth = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert fifth.signals[0] == ControlSignal.RUNNING
    assert action.resolved_motion_goal is refreshed_goal

    sixth = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert sixth.signals[0] == ControlSignal.REACHED


def test_axis_correction_preserves_current_free_twist() -> None:
    context, backend = _context(object_poses={"plate": PoseState()})
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = PoseState()
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {"kind": "held_object"},
            "position": [0.0, 0.0, 0.0],
            "orientation_goal": _axis_alignment_goal(),
            "reference": "world",
            "static": True,
            "tolerance": {"position": 0.001, "orientation": 0.001},
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)
    env_mask = np.asarray([True])

    TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    goal = action.resolved_motion_goal
    assert goal is not None

    slipped_plate = PoseState(orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), 0.7))
    backend.get_object_handler("plate").set_pose(slipped_plate)
    result = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )

    assert result.signals[0] == ControlSignal.RUNNING
    current_eef = operator.get_end_effector_pose().select(0)
    corrected_eef = PoseState(
        position=goal.command_pose.position,
        orientation=goal.command_pose.orientation,
    )
    world_correction = compose_pose(corrected_eef, inverse_pose(current_eef))
    corrected_plate = compose_pose(world_correction, slipped_plate)
    expected_orientation = resolve_axis_alignment_orientation(
        slipped_plate,
        (0.0, 0.0, 1.0),
        np.asarray([0.0, 1.0, 0.0]),
        "same",
    )
    _assert_pose_equivalent(
        corrected_plate,
        PoseState(orientation=expected_orientation),
    )
    np.testing.assert_allclose(
        resolve_axis_in_world((0.0, 0.0, 1.0), corrected_plate),
        [0.0, 1.0, 0.0],
        atol=1.0e-12,
    )
    assert quaternion_angular_distance(
        corrected_plate.orientation[0],
        goal.controlled_world_pose.orientation[0],
    ) == pytest.approx(0.7, abs=1.0e-12)


def test_live_target_axis_refreshes_after_correction_and_converges() -> None:
    context, backend = _context(
        object_poses={"plate": PoseState(), "rack": PoseState()}
    )
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = PoseState()
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    rack = backend.get_object_handler("rack")
    pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {"kind": "held_object"},
            "position": [0.0, 0.0, 0.0],
            "reference": "eef_world",
            "orientation_goal": {
                "kind": "axis_alignment",
                "controlled_axis": [0.0, 0.0, 1.0],
                "target_axis": {
                    "vector": [1.0, 0.0, 0.0],
                    "reference": "object",
                },
                "direction": "same",
            },
            "tolerance": {"position": 0.001, "orientation": 0.001},
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)
    env_mask = np.asarray([True])

    first = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert first.signals[0] == ControlSignal.RUNNING
    initial_goal = action.resolved_motion_goal
    assert initial_goal is not None
    np.testing.assert_allclose(initial_goal.target_axis_world, [1.0, 0.0, 0.0])

    second = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert second.signals[0] == ControlSignal.RUNNING
    assert second.details[0]["event"] == "controlled_frame_correction"
    frozen_goal = action.resolved_motion_goal
    assert frozen_goal is not None

    rack.set_pose(
        PoseState(
            orientation=_axis_angle_quaternion(
                (0.0, 0.0, 1.0),
                np.pi / 2.0,
            )
        )
    )
    third = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert third.signals[0] == ControlSignal.RUNNING
    assert action.resolved_motion_goal is frozen_goal

    plate_aligned_with_old_axis = PoseState(
        orientation=resolve_axis_alignment_orientation(
            PoseState(),
            (0.0, 0.0, 1.0),
            np.asarray([1.0, 0.0, 0.0]),
            "same",
        )
    )
    backend.get_object_handler("plate").set_pose(plate_aligned_with_old_axis)
    fourth = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert fourth.signals[0] == ControlSignal.RUNNING
    assert fourth.details[0]["event"] == "controlled_frame_correction"
    refreshed_goal = action.resolved_motion_goal
    assert refreshed_goal is not None
    assert refreshed_goal is not frozen_goal
    np.testing.assert_allclose(refreshed_goal.target_axis_world, [0.0, 1.0, 0.0])

    corrected_plate = PoseState(
        orientation=resolve_axis_alignment_orientation(
            plate_aligned_with_old_axis,
            (0.0, 0.0, 1.0),
            np.asarray([0.0, 1.0, 0.0]),
            "same",
        )
    )
    backend.get_object_handler("plate").set_pose(corrected_plate)
    fifth = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert fifth.signals[0] == ControlSignal.RUNNING
    sixth = TaskRunner._run_action(
        0,
        operator,
        action,
        rack,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert sixth.signals[0] == ControlSignal.REACHED
    np.testing.assert_allclose(
        resolve_axis_in_world((0.0, 0.0, 1.0), corrected_plate),
        [0.0, 1.0, 0.0],
        atol=1.0e-12,
    )


def test_held_object_identity_change_fails_before_next_motion_command() -> None:
    context, backend = _context(object_poses={"plate": PoseState()})
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = PoseState()
    held_name = {"value": "plate"}
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        held_name["value"] if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    pose = PoseControlConfig.model_validate(
        {
            **_held_object_pose(position=(0.2, 0.0, 0.0)).model_dump(mode="json"),
            "static": True,
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)
    env_mask = np.asarray([True])

    first = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )
    assert first.signals[0] == ControlSignal.RUNNING
    progress_before_identity_change = int(operator._progress[0])

    held_name["value"] = "cup"
    second = TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        env_mask,
        grasp_binding=binding,
    )

    assert second.signals[0] == ControlSignal.FAILED
    assert second.details[0]["event"] == "controlled_frame_validation_failed"
    assert "expected 'plate', got 'cup'" in second.details[0]["failure_reason"]
    assert int(operator._progress[0]) == progress_before_identity_change


def test_task_flow_does_not_inject_legacy_orientation_into_orientation_goal() -> None:
    legacy_orientation = _axis_angle_quaternion((0.0, 0.0, 1.0), 0.81)
    stage = StageConfig.model_validate(
        {
            "name": "orient_held_plate",
            "object": "",
            "operation": "move",
            "operator": "arm",
            "param": {
                "pre_move": [
                    {
                        "position": [0.2, 0.0, 0.3],
                        "orientation": legacy_orientation,
                        "reference": "world",
                    },
                    {
                        "controlled_frame": {"kind": "held_object"},
                        "position": [0.5, 0.0, 0.3],
                        "orientation_goal": _axis_alignment_goal(),
                        "reference": "world",
                    },
                ]
            },
        }
    )

    actions, last_orientation = TaskFlowBuilder.build_actions(stage)

    assert len(actions) == 2
    assert actions[0].pose is not None
    assert actions[0].pose.orientation == pytest.approx(legacy_orientation)
    assert actions[1].pose is not None
    assert actions[1].pose.orientation is None
    assert actions[1].pose.rotation is None
    assert actions[1].pose.orientation_goal is not None
    assert last_orientation is None


def test_object_goal_breaks_legacy_orientation_inheritance_for_later_eef() -> None:
    legacy_orientation = _axis_angle_quaternion((0.0, 0.0, 1.0), 0.81)
    stage = StageConfig.model_validate(
        {
            "name": "orient_then_retreat",
            "object": "rack",
            "operation": "place",
            "operator": "arm",
            "param": {
                "pre_move": [
                    {
                        "position": [0.2, 0.0, 0.3],
                        "orientation": legacy_orientation,
                        "reference": "world",
                    },
                    {
                        "controlled_frame": {"kind": "held_object"},
                        "position": [0.5, 0.0, 0.3],
                        "orientation_goal": {
                            "kind": "fixed",
                            "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
                        },
                        "reference": "world",
                    },
                ],
                "post_move": [
                    {
                        "position": [0.0, 0.0, 0.2],
                        "reference": "eef_world",
                    }
                ],
            },
        }
    )

    actions, last_orientation = TaskFlowBuilder.build_actions(stage)

    assert actions[-1].pose is not None
    assert actions[-1].pose.orientation is None
    assert actions[-1].pose.rotation is None
    assert last_orientation is None


def test_relative_held_object_goal_is_snapshotted_on_first_tick() -> None:
    context, backend = _context(
        object_poses={"plate": PoseState(position=(0.0, 0.0, 0.0))}
    )
    operator = backend.get_operator_handler("arm")
    operator.end_effector_pose = PoseState(position=(0.0, 0.0, 0.0))
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {"kind": "held_object"},
            "position": [0.1, 0.0, 0.0],
            "orientation_goal": {
                "kind": "fixed",
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "reference": "world",
            "relative": True,
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)

    TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        np.asarray([True]),
        grasp_binding=binding,
    )
    first_goal = action.resolved_motion_goal
    assert first_goal is not None
    backend.get_object_handler("plate").set_pose(PoseState(position=(0.04, 0.0, 0.0)))
    operator.end_effector_pose = PoseState(position=(0.04, 0.0, 0.0))

    TaskRunner._run_action(
        0,
        operator,
        action,
        None,
        backend,
        np.asarray([True]),
        grasp_binding=binding,
    )

    assert action.resolved_motion_goal is first_goal
    np.testing.assert_allclose(
        action.resolved_motion_goal.controlled_world_pose.position[0],
        [0.1, 0.0, 0.0],
    )


@pytest.mark.parametrize("static", [False, True])
def test_eef_snapshot_keeps_independent_object_axis_live_unless_static(
    static: bool,
) -> None:
    _, backend = _context(object_poses={"rack": PoseState()})
    operator = backend.get_operator_handler("arm")
    target = backend.get_object_handler("rack")
    pose = PoseControlConfig.model_validate(
        {
            "position": [0.0, 0.0, 0.0],
            "reference": "eef_world",
            "static": static,
            "orientation_goal": {
                "kind": "axis_alignment",
                "controlled_axis": [0.0, 0.0, 1.0],
                "target_axis": {
                    "vector": [1.0, 0.0, 0.0],
                    "reference": "object",
                },
                "direction": "same",
            },
        }
    )
    action = PrimitiveAction(kind="pose", pose=pose)

    TaskRunner._run_action(
        0,
        operator,
        action,
        target,
        backend,
        np.asarray([True]),
    )
    first_goal = action.resolved_motion_goal
    assert first_goal is not None
    first_position = first_goal.controlled_world_pose.position.copy()
    target.set_pose(
        PoseState(
            orientation=_axis_angle_quaternion(
                (0.0, 0.0, 1.0),
                np.pi / 2.0,
            )
        )
    )

    TaskRunner._run_action(
        0,
        operator,
        action,
        target,
        backend,
        np.asarray([True]),
    )
    second_goal = action.resolved_motion_goal
    assert second_goal is not None

    np.testing.assert_allclose(
        second_goal.controlled_world_pose.position,
        first_position,
    )
    expected_axis = [1.0, 0.0, 0.0] if static else [0.0, 1.0, 0.0]
    np.testing.assert_allclose(
        second_goal.target_axis_world,
        expected_axis,
        atol=1.0e-12,
    )
    assert (second_goal is first_goal) is static


def test_held_object_named_frame_requires_rigid_object_ownership() -> None:
    context, backend = _context(
        object_poses={
            "plate": PoseState(position=(0.3, 0.0, 0.2)),
            "rack": PoseState(position=(0.6, 0.0, 0.2)),
        }
    )
    operator = backend.get_operator_handler("arm")
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    backend.element_poses.update(
        {
            "plate_site": PoseState(position=(0.31, 0.0, 0.2)),
            "rack_site": PoseState(position=(0.61, 0.0, 0.2)),
        }
    )
    backend.element_owners.update({"plate_site": "plate", "rack_site": "rack"})
    binding = context.capture_grasp_binding(0, "arm", "plate")

    legal_pose = PoseControlConfig.model_validate(
        {
            "controlled_frame": {
                "kind": "held_object",
                "frame": "plate_site",
            },
            "position": [0.5, 0.0, 0.3],
            "reference": "world",
        }
    )
    legal_goal = TaskRunner._resolve_motion_goal(
        0,
        operator,
        legal_pose,
        None,
        backend,
        grasp_binding=binding,
    )
    assert legal_goal.controlled_object_name == "plate"

    unrelated_pose = legal_pose.model_copy(
        update={
            "controlled_frame": legal_pose.controlled_frame.model_copy(
                update={"frame": "rack_site"}
            )
        }
    )
    with pytest.raises(ValueError, match="not rigidly attached"):
        TaskRunner._resolve_motion_goal(
            0,
            operator,
            unrelated_pose,
            None,
            backend,
            grasp_binding=binding,
        )


def test_verified_pick_captures_binding_before_first_held_object_post_move() -> None:
    """The verified close boundary must authorize the immediately following move."""
    current_eef = PoseState(
        position=(0.18, -0.06, 0.36),
        orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), 0.21),
    )
    current_plate = PoseState(
        position=(0.31, -0.02, 0.28),
        orientation=_axis_angle_quaternion((1.0, 0.0, 0.0), 0.34),
    )
    context, backend = _context(object_poses={"plate": current_plate})
    backend.get_operator_handler("arm").end_effector_pose = current_eef
    grasp_state = {"verified": False}
    backend.is_operator_grasping = lambda operator_name: np.asarray(
        [grasp_state["verified"] and operator_name == "arm"],
        dtype=bool,
    )
    backend.is_object_grasped = lambda operator_name, object_name: np.asarray(
        [grasp_state["verified"] and operator_name == "arm" and object_name == "plate"],
        dtype=bool,
    )
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate"
        if grasp_state["verified"] and operator_name == "arm" and env_index == 0
        else None
    )
    stage = StageConfig.model_validate(
        {
            "name": "pick_and_reorient_plate",
            "object": "plate",
            "operation": "pick",
            "operator": "arm",
            "param": {
                "post_move": [
                    {
                        "controlled_frame": {"kind": "held_object"},
                        "position": [0.48, 0.03, 0.44],
                        "orientation_goal": _axis_alignment_goal(),
                        "reference": "world",
                    }
                ]
            },
        }
    )
    plan = StageExecutionPlan(stage_index=0, stage=stage, operator_name="arm")
    actions, _ = TaskFlowBuilder.build_actions(stage)
    binding_seen_by_post_move: list[GraspBinding | None] = []

    def run_action(
        env_index: int,
        stage_plan: StageExecutionPlan,
        action: PrimitiveAction,
        env_mask: np.ndarray,
    ) -> ControlResult:
        if action.kind == "eef":
            grasp_state["verified"] = True
            return ControlResult.filled(backend.batch_size, ControlSignal.REACHED)
        binding = context.get_grasp_binding(env_index, stage_plan.operator_name)
        binding_seen_by_post_move.append(binding)
        return TaskRunner._run_stage_action(
            env_index=env_index,
            plan=stage_plan,
            action=action,
            backend=backend,
            env_mask=env_mask,
            grasp_binding=binding,
        )

    execution = StageExecution(
        context,
        [plan],
        actions_factory=lambda _plan: actions,
        action_runner=run_action,
    )

    close_event = execution.advance_control(0, use_configured_identity=True)

    binding = context.get_grasp_binding(0, "arm")
    assert close_event.primitive_reached
    assert binding is not None
    assert binding.object_name == "plate"
    assert execution.states[0].active is not None
    assert execution.states[0].active.action_index == 1
    expected_binding = compose_pose(inverse_pose(current_eef), current_plate)
    _assert_pose_equivalent(binding.eef_from_object, expected_binding)

    post_move_event = execution.advance_control(0, use_configured_identity=True)
    active = execution.states[0].active

    assert post_move_event.control_tick
    assert active is not None
    assert active.action_index == 1
    assert binding_seen_by_post_move == [binding]
    assert actions[1].resolved_motion_goal is not None
    assert actions[1].resolved_motion_goal.controlled_object_name == "plate"
    assert actions[1].resolved_pose is not None


def test_repeated_close_for_same_object_does_not_rebind_after_slip() -> None:
    current_eef = PoseState(
        position=(0.2, 0.0, 0.4),
        orientation=_axis_angle_quaternion((0.0, 1.0, 0.0), 0.25),
    )
    initial_plate = PoseState(
        position=(0.33, 0.04, 0.3),
        orientation=_axis_angle_quaternion((1.0, 0.0, 0.0), -0.42),
    )
    context, backend = _context(object_poses={"plate": initial_plate})
    backend.get_operator_handler("arm").end_effector_pose = current_eef
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    original = context.capture_grasp_binding(0, "arm", "plate")
    original_transform = PoseState(
        position=original.eef_from_object.position.copy(),
        orientation=original.eef_from_object.orientation.copy(),
    )
    backend.get_object_handler("plate").set_pose(
        PoseState(
            position=(0.57, -0.18, 0.49),
            orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), 1.03),
        )
    )
    stage = StageConfig.model_validate(
        {
            "name": "repeat_close",
            "object": "plate",
            "operation": "pick",
            "operator": "arm",
            "param": {},
        }
    )
    plan = StageExecutionPlan(stage_index=0, stage=stage, operator_name="arm")
    close = PrimitiveAction(
        kind="eef",
        eef=EefControlConfig(close=True, require_grasp=True),
    )
    active = ActiveStageState(
        plan=plan,
        operator=backend.get_operator_handler("arm"),
        target=backend.get_object_handler("plate"),
        actions=[close],
    )
    execution = StageExecution(
        context,
        [plan],
        actions_factory=lambda _plan: [close],
    )

    failure = execution._update_grasp_binding_after_eef(0, active, close)
    retained = context.get_grasp_binding(0, "arm")

    assert failure is None
    assert retained is original
    _assert_pose_equivalent(retained.eef_from_object, original_transform)
    slipped_measurement = compose_pose(
        inverse_pose(current_eef),
        backend.get_object_handler("plate").get_pose(),
    )
    assert not np.allclose(
        retained.eef_from_object.position,
        slipped_measurement.position,
    )


def test_legacy_eef_orientation_config_preserves_world_command_semantics() -> None:
    rack_pose = PoseState(
        position=(0.41, -0.16, 0.23),
        orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), 0.58),
    )
    _, backend = _context(object_poses={"rack": rack_pose})
    local_orientation = _axis_angle_quaternion((1.0, 0.0, 0.0), -0.37)
    local_position = (0.06, -0.03, 0.12)
    task_file = TaskFileConfig.model_validate(
        {
            "backend": "auto_atom.mock.build_mock_backend",
            "task": {
                "env_name": "legacy_eef_orientation",
                "stages": [
                    {
                        "name": "legacy_move",
                        "object": "rack",
                        "operation": "move",
                        "operator": "arm",
                        "param": {
                            "pre_move": [
                                {
                                    "position": local_position,
                                    "orientation": local_orientation,
                                    "reference": "object",
                                }
                            ]
                        },
                    }
                ],
            },
            "task_operators": {"arm": {}},
        }
    )
    stage = task_file.task.stages[0]
    plan = StageExecutionPlan(stage_index=0, stage=stage, operator_name="arm")
    actions, _ = TaskFlowBuilder.build_actions(stage)

    result = TaskRunner._run_stage_action(
        env_index=0,
        plan=plan,
        action=actions[0],
        backend=backend,
        env_mask=np.asarray([True]),
    )

    assert result.signals[0] == ControlSignal.RUNNING
    assert actions[0].pose is not None
    assert actions[0].pose.controlled_frame.kind.value == "eef"
    assert actions[0].pose.orientation_goal is None
    assert actions[0].resolved_motion_goal is not None
    assert actions[0].resolved_motion_goal.controlled_object_name is None
    expected_world_from_eef = compose_pose(
        rack_pose,
        PoseState(position=local_position, orientation=local_orientation),
    )
    command_world_from_eef = PoseState(
        position=actions[0].resolved_pose.position,
        orientation=actions[0].resolved_pose.orientation,
    )
    _assert_pose_equivalent(command_world_from_eef, expected_world_from_eef)


def test_task_runner_and_config_policy_share_binding_and_eef_command() -> None:
    current_eef = PoseState(
        position=(0.14, -0.11, 0.39),
        orientation=_axis_angle_quaternion((0.0, 1.0, 0.0), -0.29),
    )
    current_plate = PoseState(
        position=(0.29, -0.05, 0.32),
        orientation=_axis_angle_quaternion((1.0, 0.0, 0.0), 0.46),
    )
    rack_pose = PoseState(
        position=(0.56, 0.07, 0.21),
        orientation=_axis_angle_quaternion((0.0, 0.0, 1.0), 0.38),
    )
    context, backend = _context(
        object_poses={"plate": current_plate, "rack": rack_pose}
    )
    backend.get_operator_handler("arm").end_effector_pose = current_eef
    backend.get_grasped_object_name = lambda operator_name, env_index: (
        "plate" if operator_name == "arm" and env_index == 0 else None
    )
    binding = context.capture_grasp_binding(0, "arm", "plate")
    stage = StageConfig.model_validate(
        {
            "name": "place_plate",
            "object": "rack",
            "operation": "place",
            "operator": "arm",
            "param": {
                "pre_move": [
                    {
                        "controlled_frame": {"kind": "held_object"},
                        "position": [0.0, 0.0, 0.08],
                        "orientation_goal": {
                            "kind": "axis_alignment",
                            "controlled_axis": [0.0, 0.0, 1.0],
                            "target_axis": {
                                "vector": [0.0, 1.0, 0.0],
                                "reference": "object",
                            },
                            "direction": "same",
                        },
                        "reference": "object",
                    }
                ]
            },
        }
    )
    plan = StageExecutionPlan(stage_index=0, stage=stage, operator_name="arm")
    context.plan = [plan]
    template_actions, _ = TaskFlowBuilder.build_actions(stage)
    direct_action = deepcopy(template_actions[0])
    policy_action = deepcopy(template_actions[0])

    direct_result = TaskRunner._run_stage_action(
        env_index=0,
        plan=plan,
        action=direct_action,
        backend=backend,
        env_mask=np.asarray([True]),
        grasp_binding=binding,
    )
    backend.reset(np.asarray([True]))
    assert context.get_grasp_binding(0, "arm") is binding

    policy = ConfigDrivenDemoPolicy()
    policy._ensure_capacity(backend.batch_size)
    policy._cached_stage_indices[0] = 0
    policy._cached_actions[0] = [policy_action]
    feedback = policy.action_applier(
        context,
        ConfigDrivenPolicyAction(
            env_actions=[
                ConfigDrivenEnvAction(stage_index=0, action=policy_action),
            ]
        ),
        env_mask=np.asarray([True]),
    )

    assert direct_result.signals[0] == ControlSignal.RUNNING
    assert feedback.signals[0] == ControlSignal.RUNNING
    assert direct_action.resolved_motion_goal is not None
    assert policy_action.resolved_motion_goal is not None
    assert direct_action.resolved_motion_goal.controlled_object_name == "plate"
    assert policy_action.resolved_motion_goal.controlled_object_name == "plate"
    direct_command = PoseState(
        position=direct_action.resolved_pose.position,
        orientation=direct_action.resolved_pose.orientation,
    )
    policy_command = PoseState(
        position=policy_action.resolved_pose.position,
        orientation=policy_action.resolved_pose.orientation,
    )
    _assert_pose_equivalent(policy_command, direct_command)
    _assert_pose_equivalent(
        policy_action.resolved_motion_goal.controlled_world_pose,
        direct_action.resolved_motion_goal.controlled_world_pose,
    )
