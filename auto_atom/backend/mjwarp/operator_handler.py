"""``OperatorHandler`` implementation for the MJWarp backend.

This is the seam the runtime actually calls. It owns no control logic of its
own: :class:`~auto_atom.backend.mjwarp.arm_control.MjWarpArmControl` and
:class:`~auto_atom.backend.mjwarp.eef_control.MjWarpEefControl` hold the
two state machines, and this class translates between the runtime's
config-object vocabulary (``PoseControlConfig``, ``EefControlConfig``,
``env_mask``) and their argument vocabulary.

Keeping the translation here rather than inside the control classes is what let
those be written and tested against native's arithmetic before any config
plumbing existed, and it keeps this file free of physics.

Both control halves synchronously advance selected worlds by a complete control
update. Explicit step deferral remains available for callers that assemble a
batched command before stepping; the runtime's one-world dispatch works without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Optional

import numpy as np

from auto_atom.backend.mjwarp.arm_control import MjWarpArmControl
from auto_atom.backend.mjwarp.eef_control import MjWarpEefControl
from auto_atom.backend.mjwarp.operator_state import (
    MjWarpOperatorState,
    get_base_pose,
    get_eef_pose_in_world,
)
from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.config.motion import EefControlConfig, PoseControlConfig
from auto_atom.contracts import ObjectHandler, OperatorHandler
from auto_atom.execution_model import ControlResult
from auto_atom.utils.pose import PoseState


@dataclass
class MjWarpOperatorHandler(OperatorHandler):
    """One operator, satisfying the runtime's handler contract.

    ``arm`` and ``eef`` are the two control state machines; either may be
    absent in principle. Mocap operators use the pose-control half without IK.
    """

    state: MjWarpSceneState = None  # type: ignore[assignment]
    operator: MjWarpOperatorState = None  # type: ignore[assignment]
    arm: MjWarpArmControl = None  # type: ignore[assignment]
    eef: Optional[MjWarpEefControl] = None

    # PLACED post-condition tolerances. ``None`` means that component is
    # unconstrained unless a Stage supplies its own, which is the contract's
    # documented default rather than a placeholder.
    placed_position_tolerance: Any = None
    placed_orientation_tolerance: Optional[float] = None

    def __post_init__(self) -> None:
        # OperatorHandler is a plain ABC, not a dataclass, so there is no base
        # __post_init__ to chain to (unlike ObjectHandler).
        for field_name in ("state", "operator", "arm"):
            if getattr(self, field_name) is None:
                raise ValueError(
                    f"MjWarpOperatorHandler requires a non-None '{field_name}'."
                )

    @property
    def name(self) -> str:
        return self.operator.name

    # ------------------------------------------------------------------
    # Tick boundary
    # ------------------------------------------------------------------

    def control_tick(self) -> Iterator[None]:
        """Context manager spanning one control tick's per-env calls.

        Explicit callers can coalesce equal-duration commands for multiple worlds.
        Completion observations inside this block precede its deferred step.
        """
        return self.state.deferred_step()

    # ------------------------------------------------------------------
    # OperatorHandler contract
    # ------------------------------------------------------------------

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],  # noqa: ARG002 - arm motion ignores it
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the arm toward ``pose`` for the selected environments.

        ``target`` is unused: the arm's goal is fully described by the waypoint
        the runtime resolved, and the grasp target only matters to the gripper
        half. Native takes it to cache a handler-side reference for its own
        grasp bookkeeping, which lives in the eef path here instead.
        """
        if pose.position is None:
            raise ValueError(
                f"Operator '{self.name}' received a pose waypoint without a "
                "position; the runtime resolves waypoints before dispatch, so "
                "this indicates an unresolved goal."
            )
        orientation = pose.orientation
        if orientation is None:
            # Hold the current orientation when the waypoint constrains only
            # position, rather than snapping to identity.
            _, current = get_eef_pose_in_world(self.state, self.operator)
            orientation = current[0]

        waypoint_tolerance = pose.tolerance
        return self.arm.move(
            np.asarray(pose.position, dtype=np.float64),
            np.asarray(orientation, dtype=np.float64),
            waypoint_linear_step=float(pose.max_linear_step),
            waypoint_angular_step=float(pose.max_angular_step),
            waypoint_position_tolerance=(
                waypoint_tolerance.position if waypoint_tolerance else None
            ),
            waypoint_orientation_tolerance=(
                waypoint_tolerance.orientation if waypoint_tolerance else None
            ),
            command_key=str(pose.model_dump(mode="json")),
            world_mask=env_mask,
        )

    def control_eef(
        self,
        eef: EefControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the gripper toward ``eef`` for the selected environments."""
        if self.eef is None:
            raise ValueError(
                f"Operator '{self.name}' has no end-effector control configured."
            )
        return self.eef.control(
            close=eef.close,
            require_grasp=eef.require_grasp,
            joint_positions=eef.joint_positions,
            target_body_name=self._target_body_name(target),
            world_mask=env_mask,
        )

    def get_end_effector_pose(self) -> PoseState:
        positions, orientations = get_eef_pose_in_world(self.state, self.operator)
        return PoseState(position=positions, orientation=orientations)

    def get_base_pose(self) -> PoseState:
        positions, orientations = get_base_pose(self.operator)
        return PoseState(position=positions, orientation=orientations)

    def get_reached_tolerances(self) -> tuple[Any, Any]:
        return self.arm.position_tolerance, self.arm.orientation_tolerance

    def get_placed_tolerances(self) -> tuple[Any, Any]:
        return self.placed_position_tolerance, self.placed_orientation_tolerance

    def set_home_joint_positions(
        self,
        joint_positions: Mapping[str, object],
        env_mask: Optional[np.ndarray] = None,
        *,
        apply_home: bool = True,
    ) -> None:
        """Record arm-joint home angles, optionally applying them now.

        The mapping is joint name -> angle, so an operator's initial joints stay
        out of the generic env config. Unnamed joints are rejected rather than
        ignored: silently dropping one would leave the arm in a pose the config
        did not ask for.
        """
        import mujoco

        arm_indices = self.operator.arm_qpos_indices
        if arm_indices.size == 0:
            raise NotImplementedError(
                f"Operator '{self.name}' has no arm joints to home."
            )

        joint_names = [
            mujoco.mj_id2name(
                self.state.host_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                int(self.state.host_model.actuator_trnid[int(actuator), 0]),
            )
            for actuator in self.operator.arm_actuator_ids
        ]
        angles = np.array(
            self.operator.home_arm_qpos[0][: len(joint_names)], dtype=np.float64
        )
        unknown = set(joint_positions) - set(joint_names)
        if unknown:
            raise ValueError(
                f"Operator '{self.name}' has no arm joint(s) named "
                f"{sorted(unknown)}. Known arm joints: {joint_names}."
            )
        for position, name in enumerate(joint_names):
            if name in joint_positions:
                angles[position] = float(joint_positions[name])  # type: ignore[arg-type]

        self.operator.home_arm_qpos = np.tile(angles, (self.state.nworld, 1))
        if apply_home:
            self.home(env_mask=env_mask)

    def home(self, env_mask: Optional[np.ndarray] = None) -> None:
        """Restore the recorded home joint angles and clear motion progress."""
        mask = None if env_mask is None else np.asarray(env_mask, dtype=bool)
        if self.operator.joint_mode:
            self.state.set_joint_positions(
                self.operator.arm_qpos_indices,
                self.operator.home_arm_qpos,
                self.operator.arm_dof_indices,
                actuator_ids=self.operator.arm_actuator_ids,
                world_mask=mask,
            )
        else:
            from auto_atom.backend.mjwarp.frames import base_to_world

            self.state.set_mocap_pose(
                self.operator.mocap_body_name,
                self.operator.home_mocap_position,
                self.operator.home_mocap_orientation,
                world_mask=mask,
            )
            root_positions = np.empty((self.state.nworld, 3))
            root_orientations = np.empty((self.state.nworld, 4))
            for world in range(self.state.nworld):
                root_positions[world], root_orientations[world] = base_to_world(
                    self.operator.mocap_to_root_position,
                    self.operator.mocap_to_root_orientation,
                    self.operator.home_mocap_position[world],
                    self.operator.home_mocap_orientation[world],
                )
            self.state.set_free_joint_pose(
                self.operator.freejoint_name,
                root_positions,
                root_orientations,
                world_mask=mask,
            )
        worlds = None if mask is None else list(np.flatnonzero(mask))
        self.arm.reset(worlds)

    def set_home_end_effector_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Resolve and realize per-world home targets during reset/randomization."""
        from auto_atom.backend.mjwarp.frames import world_to_base
        from auto_atom.backend.mjwarp.operator_state import eef_to_root_pose

        mask = (
            np.ones(self.state.nworld, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool)
        )
        positions = np.asarray(pose.position)
        orientations = np.asarray(pose.orientation)
        seeds = self.state.get_joint_positions(self.operator.arm_qpos_indices)
        for world in np.flatnonzero(mask):
            row = 0 if positions.shape[0] == 1 else world
            position, orientation = positions[row], orientations[row]
            if self.operator.joint_mode:
                pos_b, quat_b = world_to_base(
                    position,
                    orientation,
                    self.operator.base_position[world],
                    self.operator.base_orientation[world],
                )
                solution = self.arm.ik.solve(world, pos_b, quat_b, seeds[world], "home")
                if solution is None:
                    raise ValueError(
                        f"Home EEF pose is unreachable for '{self.name}' world {world}."
                    )
                self.operator.home_arm_qpos[world] = solution
            else:
                root_position, root_orientation = eef_to_root_pose(
                    self.operator, world, position, orientation
                )
                self.operator.home_mocap_position[world] = root_position
                self.operator.home_mocap_orientation[world] = root_orientation
        self.home(mask)

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        """Move a base and its current EEF rigidly, preserving their relative pose."""
        from auto_atom.backend.mjwarp.frames import base_to_world, world_to_base

        mask = (
            np.ones(self.state.nworld, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool)
        )
        positions = np.asarray(pose.position)
        orientations = np.asarray(pose.orientation)
        eef = self.get_end_effector_pose()
        moved_positions, moved_orientations = (
            np.asarray(eef.position).copy(),
            np.asarray(eef.orientation).copy(),
        )
        for world in np.flatnonzero(mask):
            row = 0 if positions.shape[0] == 1 else world
            pos_b, quat_b = world_to_base(
                eef.position[world],
                eef.orientation[world],
                self.operator.base_position[world],
                self.operator.base_orientation[world],
            )
            self.operator.base_position[world] = positions[row]
            self.operator.base_orientation[world] = orientations[row]
            moved_positions[world], moved_orientations[world] = base_to_world(
                pos_b, quat_b, positions[row], orientations[row]
            )
        if self.operator.joint_mode:
            self.state.set_object_pose(
                self.operator.root_body_name,
                self.operator.base_position,
                self.operator.base_orientation,
                world_mask=mask,
            )
        else:
            self.set_home_end_effector_pose(
                PoseState(position=moved_positions, orientation=moved_orientations),
                mask,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _target_body_name(target: Optional[ObjectHandler]) -> Optional[str]:
        """Body name of a grasp target, when one was supplied.

        Read off the handler rather than assumed equal to its logical name: the
        two differ whenever a config names an object differently from its MJCF
        body, which the object handler keeps separate for that reason.
        """
        if target is None:
            return None
        return getattr(target, "body_name", None) or target.name
