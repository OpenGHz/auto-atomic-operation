"""Backend-neutral semantic motion-goal resolution and error measurement.

Stage execution and the concrete runtime both need the same controlled-frame
geometry.  This module owns that semantic contract and talks to the backend
through the small set of methods it needs, so the state machine does not need
to import the TaskRunner that happens to host one implementation.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .execution_model import ResolvedMotionGoal, ResolvedObjectMotionGoal
from .framework import (
    AxisAlignmentOrientationGoalConfig,
    AxisReference,
    ControlledFrameKind,
    FixedOrientationGoalConfig,
    PoseControlConfig,
    PoseReference,
)
from .pose_goal import (
    axis_alignment_error,
    resolve_axis_alignment_orientation,
    resolve_axis_in_world,
)
from .utils.pose import (
    PoseState,
    compose_pose,
    inverse_pose,
    normalize_quaternion,
    pose_config_to_pose_state,
    quaternion_angular_distance,
)


def pose_config_to_local_pose(pose: PoseControlConfig) -> PoseState:
    """Convert a pose control declaration into a concrete local pose."""

    if isinstance(pose.orientation_goal, FixedOrientationGoalConfig):
        return PoseState(
            position=pose.position or (0.0, 0.0, 0.0),
            orientation=normalize_quaternion(pose.orientation_goal.quaternion_xyzw),
        )
    return pose_config_to_pose_state(pose)


def object_target_reference_pose(
    *,
    env_index: int,
    target: Any,
    backend: Any,
    reference_site: Optional[str],
) -> PoseState:
    """Return the world pose used as an object-relative reference."""

    if reference_site is not None:
        return backend.get_element_pose(reference_site, env_index)
    if target is None:
        raise ValueError("Object reference requires a stage target object.")
    return target.get_pose().select(env_index)


def resolve_object_reference_pose(
    *,
    env_index: int,
    pose: PoseControlConfig,
    target: Any,
    backend: Any,
    reference_site: Optional[str],
) -> PoseState:
    """Resolve a reference accepted by object-only motion."""

    reference = pose.reference
    if reference == PoseReference.AUTO:
        reference = (
            PoseReference.OBJECT_WORLD if target is not None else PoseReference.WORLD
        )
    if reference == PoseReference.WORLD:
        return PoseState()
    object_reference = object_target_reference_pose(
        env_index=env_index,
        target=target,
        backend=backend,
        reference_site=reference_site,
    )
    if reference == PoseReference.OBJECT:
        return object_reference
    if reference == PoseReference.OBJECT_WORLD:
        return PoseState(position=object_reference.position[0])
    raise ValueError(
        "Object-only motion cannot resolve operator-dependent reference "
        f"{reference.value!r}."
    )


def resolve_object_motion_goal(
    *,
    env_index: int,
    object_name: str,
    pose: PoseControlConfig,
    target: Any,
    backend: Any,
    reference_site: Optional[str],
    current_object_pose: Optional[PoseState] = None,
) -> ResolvedObjectMotionGoal:
    """Resolve a held-object waypoint without constructing an EEF command."""

    if pose.controlled_frame.kind != ControlledFrameKind.HELD_OBJECT:
        raise ValueError("Object-only motion requires controlled_frame='held_object'.")
    if pose.arc is not None:
        raise ValueError("Object-only motion does not support arc waypoints.")
    handler = backend.get_object_handler(object_name)
    if handler is None:
        raise KeyError(f"Unknown carried object {object_name!r}.")
    actual_world_from_object = handler.get_pose().select(env_index)
    world_from_object = (
        actual_world_from_object if current_object_pose is None else current_object_pose
    )
    frame_name = pose.controlled_frame.frame
    if frame_name is None:
        world_from_controlled = world_from_object
        object_from_controlled = PoseState()
    else:
        if not backend.is_element_rigidly_attached_to_object(
            frame_name,
            object_name,
            env_index,
        ):
            raise ValueError(
                f"Controlled frame {frame_name!r} is not rigidly attached "
                f"to carried object {object_name!r}."
            )
        actual_world_from_controlled = backend.get_element_pose(
            frame_name,
            env_index,
        )
        object_from_controlled = compose_pose(
            inverse_pose(actual_world_from_object),
            actual_world_from_controlled,
        )
        world_from_controlled = compose_pose(
            world_from_object,
            object_from_controlled,
        )

    reference_pose = resolve_object_reference_pose(
        env_index=env_index,
        pose=pose,
        target=target,
        backend=backend,
        reference_site=reference_site,
    )
    local_pose = pose_config_to_local_pose(pose)
    current_local = compose_pose(
        inverse_pose(reference_pose),
        world_from_controlled,
    )
    has_fixed_orientation = bool(pose.orientation or pose.rotation) or isinstance(
        pose.orientation_goal,
        FixedOrientationGoalConfig,
    )
    if pose.relative:
        target_local_pose = compose_pose(current_local, local_pose)
    elif has_fixed_orientation:
        target_local_pose = local_pose
    else:
        target_local_pose = PoseState(
            position=local_pose.position[0],
            orientation=current_local.orientation[0],
        )

    controlled_world_pose = compose_pose(reference_pose, target_local_pose)
    target_axis_world: Optional[np.ndarray] = None
    orientation_goal = pose.orientation_goal
    if isinstance(orientation_goal, AxisAlignmentOrientationGoalConfig):
        if orientation_goal.target_axis.reference == AxisReference.WORLD:
            axis_reference_pose = None
        elif orientation_goal.target_axis.reference == AxisReference.OBJECT:
            axis_reference_pose = object_target_reference_pose(
                env_index=env_index,
                target=target,
                backend=backend,
                reference_site=reference_site,
            )
        else:
            raise ValueError("Object-only axis alignment cannot use BASE.")
        target_axis_world = resolve_axis_in_world(
            orientation_goal.target_axis.vector,
            axis_reference_pose,
        )
        controlled_world_pose = PoseState(
            position=controlled_world_pose.position[0],
            orientation=resolve_axis_alignment_orientation(
                world_from_controlled,
                orientation_goal.controlled_axis,
                target_axis_world,
                orientation_goal.direction,
            ),
        )

    return ResolvedObjectMotionGoal(
        configured_pose=pose,
        controlled_world_pose=controlled_world_pose,
        object_world_pose=compose_pose(
            controlled_world_pose,
            inverse_pose(object_from_controlled),
        ),
        controlled_object_name=object_name,
        target_axis_world=target_axis_world,
    )


def motion_goal_errors(
    *,
    env_index: int,
    operator: Any,
    backend: Any,
    goal: ResolvedMotionGoal,
    require_held: bool,
) -> tuple[np.ndarray, float, PoseState]:
    """Return position and orientation error for a resolved semantic goal."""

    configured = goal.configured_pose
    if configured.controlled_frame.kind == ControlledFrameKind.EEF:
        current_pose = operator.get_end_effector_pose().select(env_index)
    else:
        if not goal.controlled_object_name:
            raise RuntimeError("Held-object motion goal has no controlled identity.")
        if require_held:
            actual_name = backend.get_grasped_object_name(operator.name, env_index)
            if actual_name != goal.controlled_object_name:
                raise RuntimeError(
                    "Held-object identity changed while executing motion goal: "
                    f"expected {goal.controlled_object_name!r}, got {actual_name!r}."
                )
        handler = backend.get_object_handler(goal.controlled_object_name)
        if handler is None:
            raise RuntimeError(
                f"Unknown controlled object {goal.controlled_object_name!r}."
            )
        frame_name = configured.controlled_frame.frame
        current_pose = (
            handler.get_pose().select(env_index)
            if frame_name is None
            else backend.get_element_pose(frame_name, env_index)
        )

    position_error = np.asarray(
        current_pose.position[0],
        dtype=np.float64,
    ) - np.asarray(goal.controlled_world_pose.position[0], dtype=np.float64)
    orientation_goal = configured.orientation_goal
    if isinstance(orientation_goal, AxisAlignmentOrientationGoalConfig):
        if goal.target_axis_world is None:
            raise RuntimeError(
                "Axis-alignment motion goal has no resolved target axis."
            )
        current_axis_world = resolve_axis_in_world(
            orientation_goal.controlled_axis,
            current_pose,
        )
        orientation_error = axis_alignment_error(
            current_axis_world,
            goal.target_axis_world,
            orientation_goal.direction,
        )
    else:
        orientation_error = quaternion_angular_distance(
            current_pose.orientation[0],
            goal.controlled_world_pose.orientation[0],
        )
    return position_error, float(orientation_error), current_pose


__all__ = [
    "motion_goal_errors",
    "object_target_reference_pose",
    "pose_config_to_local_pose",
    "resolve_object_motion_goal",
    "resolve_object_reference_pose",
]
