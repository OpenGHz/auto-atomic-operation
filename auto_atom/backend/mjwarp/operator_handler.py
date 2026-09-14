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

**The step boundary is the caller's.** Both halves request a step rather than
taking one, so a control tick must be wrapped in
:meth:`~auto_atom.basis.mjwarp.state.MjWarpSceneState.deferred_step` by whoever
drives the tick. This handler exposes :meth:`control_tick` for that, and the
reason it cannot do it internally is that the runtime calls the handler once per
environment with a one-hot mask -- so the boundary has to span all of those
calls, not sit inside one (design doc 3.8).
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
    absent in principle, but an operator with no arm cannot be driven by this
    backend (:func:`register_operator` refuses that at registration).
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

        The runtime calls control primitives once per environment with a one-hot
        mask; MJWarp's step advances every world at once. Wrapping the whole
        sweep in one of these makes a tick advance physics exactly once instead
        of ``batch_size`` times under mismatched commands.
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
        self.state.set_joint_positions(
            self.operator.arm_qpos_indices,
            self.operator.home_arm_qpos,
            self.operator.arm_dof_indices,
            actuator_ids=self.operator.arm_actuator_ids,
            world_mask=mask,
        )
        worlds = None if mask is None else list(np.flatnonzero(mask))
        self.arm.reset(worlds)

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
