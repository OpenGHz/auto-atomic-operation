"""Mujoco backend adapting the generic task runner to the basis environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np
from pydantic import BaseModel

from ...framework import (
    ArmPoseConfig,
    AutoAtomConfig,
    EefControlConfig,
    OperatorConfig,
    PoseControlConfig,
)
from ...runtime import (
    ComponentRegistry,
    ControlResult,
    ControlSignal,
    ObjectHandler,
    OperatorHandler,
    PoseRandomRange,
    SceneBackend,
)
from ...utils.pose import PoseState, compose_pose, inverse_pose
from ...utils.transformations import quaternion_slerp
from ...basis.mujoco_env import EnvConfig, UnifiedMujocoEnv


class MujocoToleranceConfig(BaseModel):
    """Tolerance thresholds for pose and gripper control."""

    position: float = 0.01
    """Position error threshold (meters) for pose control."""
    orientation: float = 0.08
    """Orientation error threshold (radians) for pose control."""
    eef: float = 0.03
    """Gripper position tolerance for eef control."""


class MujocoGraspConfig(BaseModel):
    """Grasp detection parameters."""

    lateral_threshold: float = 0.0
    """Max lateral distance (meters) perpendicular to grasp direction.
    Measured in EEF frame on the plane perpendicular to grasp_axis.
    Set to 0 or negative to disable lateral distance check."""
    grasp_axis: int = 2
    """Grasp direction axis in EEF frame: 0=X, 1=Y, 2=Z (default).
    Lateral distance is computed on the plane perpendicular to this axis."""
    settle_steps: int = 5
    """Minimum simulation steps before checking grasp, allowing fingers to fully clamp."""


class MujocoControlConfig(BaseModel):
    """Control loop parameters."""

    timeout_steps: int = 600
    """Maximum simulation steps per primitive action before timeout."""
    tolerance: MujocoToleranceConfig = MujocoToleranceConfig()
    """Tolerance thresholds for control."""
    grasp: MujocoGraspConfig = MujocoGraspConfig()
    """Grasp detection parameters."""


@dataclass
class MujocoObjectHandler(ObjectHandler):
    """Object handle backed by a Mujoco body."""

    env: UnifiedMujocoEnv
    """The shared Mujoco basis environment used to query object state."""
    body_name: str
    """The Mujoco body name that stores this object's pose."""
    freejoint_name: Optional[str] = None
    """The optional free-joint name associated with the object for direct manipulation."""

    def get_pose(self) -> PoseState:
        pos, quat = self.env.get_body_pose(self.body_name)
        return PoseState(
            position=tuple(float(v) for v in pos),
            orientation=tuple(float(v) for v in quat),
        )

    def set_pose(self, pose: PoseState) -> None:
        """Force-set the object world pose via its free joint.

        No-op when the object has no free joint (i.e. it is a static body).
        After writing qpos the method calls ``mj_forward`` so that derived
        quantities (xpos, xquat, …) are immediately consistent.
        """
        if self.freejoint_name is None:
            return
        jid = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, self.freejoint_name
        )
        if jid < 0:
            return
        qpos_adr = int(self.env.model.jnt_qposadr[jid])
        dof_adr = int(self.env.model.jnt_dofadr[jid])
        x, y, z = pose.position
        qx, qy, qz, qw = pose.orientation  # xyzw → mujoco wxyz
        self.env.data.qpos[qpos_adr : qpos_adr + 7] = [x, y, z, qw, qx, qy, qz]
        self.env.data.qvel[dof_adr : dof_adr + 6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)


@dataclass
class MujocoOperatorHandler(OperatorHandler):
    """Operator controller backed by the free-flying gripper model."""

    operator_name: str
    """The runtime-visible operator name for this controller."""
    env: UnifiedMujocoEnv
    """The shared Mujoco basis environment used to step the simulation."""
    root_body_name: str = "robotiq_interface"
    """The root body name used to read the operator base pose."""
    eef_site_name: str = "eef_pose"
    """The site name used to read the operator end-effector pose."""
    mocap_body_name: str = "robotiq_mocap"
    """The mocap body name used to drive the operator base pose."""
    freejoint_name: str = "robotiq_freejoint"
    """The freejoint name for the physical base body."""
    eef_ctrl_index: int = 0
    """The control index corresponding to the gripper or end-effector actuator."""
    control: MujocoControlConfig = field(default_factory=MujocoControlConfig)
    """Control parameters including tolerances, grasp detection, and timeouts."""
    _tool_pose_in_base: PoseState = field(init=False)
    """The fixed transform from the operator base frame to the tool frame."""
    _mocap_id: int = field(init=False, repr=False)
    """Index into data.mocap_pos / data.mocap_quat arrays."""
    _fj_qpos_adr: int = field(init=False, repr=False)
    """Start index of the freejoint in the qpos array (7 DOF: pos3 + quat_wxyz4)."""
    _fj_dof_adr: int = field(init=False, repr=False)
    """Start index of the freejoint in the qvel array (6 DOF)."""
    _home_mocap_pos: np.ndarray = field(init=False, repr=False)
    """Home position snapshot from keyframe for the mocap body."""
    _home_mocap_quat: np.ndarray = field(init=False, repr=False)
    """Home quaternion (wxyz) snapshot from keyframe for the mocap body."""
    _last_move_key: Optional[str] = None
    """The serialized pose command currently tracked by the motion controller."""
    _last_eef_key: Optional[str] = None
    """The serialized end-effector command currently tracked by the controller."""
    _last_target: Optional[MujocoObjectHandler] = None
    """The last target object involved in a primitive action, if any."""
    _move_steps: int = 0
    """The number of simulation steps consumed by the active pose command."""
    _move_start_orientation: Optional[tuple] = None
    """Starting orientation (xyzw quaternion) for SLERP interpolation."""
    _move_target_orientation: Optional[tuple] = None
    """Target orientation (xyzw quaternion) for SLERP interpolation."""
    _eef_steps: int = 0
    """The number of simulation steps consumed by the active eef command."""
    _home_ctrl: np.ndarray = field(init=False, repr=False)
    """The nominal home control vector (gripper only) from the keyframe."""

    @property
    def name(self) -> str:
        return self.operator_name

    def __post_init__(self) -> None:
        # Resolve mocap body index.
        body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.mocap_body_name
        )
        if body_id < 0:
            raise ValueError(f"Mocap body '{self.mocap_body_name}' not found.")
        self._mocap_id = int(self.env.model.body_mocapid[body_id])
        if self._mocap_id < 0:
            raise ValueError(
                f"Body '{self.mocap_body_name}' is not a mocap body "
                f"(body_mocapid={self._mocap_id})."
            )

        # Resolve freejoint addresses.
        jid = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_JOINT, self.freejoint_name
        )
        if jid < 0:
            raise ValueError(f"Freejoint '{self.freejoint_name}' not found.")
        self._fj_qpos_adr = int(self.env.model.jnt_qposadr[jid])
        self._fj_dof_adr = int(self.env.model.jnt_dofadr[jid])

        # Snapshot home poses from keyframe.
        self._home_mocap_pos = self.env.data.mocap_pos[self._mocap_id].copy()
        self._home_mocap_quat = self.env.data.mocap_quat[self._mocap_id].copy()
        self._home_ctrl = np.asarray(
            self.env.data.ctrl[: self.env.model.nu], dtype=np.float64
        ).copy()
        self._tool_pose_in_base = self._compute_tool_pose_in_base()

    def _set_mocap_pose(self, position: tuple, orientation_xyzw: tuple) -> None:
        """Write a desired base pose to the mocap body (xyzw → wxyz conversion)."""
        qx, qy, qz, qw = orientation_xyzw
        self.env.data.mocap_pos[self._mocap_id] = np.asarray(position, dtype=np.float64)
        self.env.data.mocap_quat[self._mocap_id] = np.array(
            [qw, qx, qy, qz], dtype=np.float64
        )

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        key = str(pose.model_dump(mode="json"))
        if self._last_move_key != key:
            self._last_move_key = key
            self._move_steps = 0
            # Save start and target orientations for SLERP interpolation
            current_eef = self.get_end_effector_pose()
            self._move_start_orientation = current_eef.orientation
            self._move_target_orientation = pose.orientation
        if isinstance(target, MujocoObjectHandler):
            self._last_target = target

        # Use SLERP interpolation for smooth orientation changes (if enabled)
        if (
            pose.use_slerp
            and self._move_start_orientation
            and self._move_target_orientation
        ):
            current_eef = self.get_end_effector_pose()
            # Calculate interpolation fraction based on orientation error
            ori_error = self._orientation_error(
                current_eef.orientation, self._move_target_orientation
            )
            # Use smaller steps when error is large for smoother motion
            alpha = min(0.1, 0.05 / max(ori_error, 0.01))
            interpolated_ori = quaternion_slerp(
                current_eef.orientation,
                self._move_target_orientation,
                alpha,
                shortestpath=True,
            )
            desired_eef_pose = PoseState(
                position=pose.position, orientation=interpolated_ori
            )
        else:
            desired_eef_pose = PoseState(
                position=pose.position, orientation=pose.orientation
            )

        desired_base_pose = compose_pose(
            desired_eef_pose, inverse_pose(self._tool_pose_in_base)
        )
        self._set_mocap_pose(desired_base_pose.position, desired_base_pose.orientation)
        self.env.update()
        self._move_steps += 1

        current_pose = self.get_end_effector_pose()
        pos_error = float(
            np.linalg.norm(
                np.asarray(current_pose.position) - np.asarray(pose.position)
            )
        )
        ori_error = self._orientation_error(current_pose.orientation, pose.orientation)

        details = {
            "event": "moving"
            if pos_error > self.control.tolerance.position
            or ori_error > self.control.tolerance.orientation
            else "pose_reached",
            "operator": self.name,
            "target": target.name if target else "",
            "target_pose": pose.model_dump(mode="json"),
            "current_pose": {
                "position": list(current_pose.position),
                "orientation": list(current_pose.orientation),
            },
            "position_error": pos_error,
            "orientation_error": float(ori_error),
        }
        if (
            pos_error <= self.control.tolerance.position
            and ori_error <= self.control.tolerance.orientation
        ):
            return ControlResult(signal=ControlSignal.REACHED, details=details)
        if self._move_steps >= self.control.timeout_steps:
            details["event"] = "move_timeout"
            return ControlResult(signal=ControlSignal.TIMED_OUT, details=details)
        return ControlResult(signal=ControlSignal.RUNNING, details=details)

    def control_eef(
        self,
        eef: EefControlConfig,
    ) -> ControlResult:
        target_ctrl = self._eef_target(eef)
        key = f"{target_ctrl:.6f}:{eef.model_dump(mode='json')}"
        if self._last_eef_key != key:
            self._last_eef_key = key
            self._eef_steps = 0

        ctrl = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        ctrl[self.eef_ctrl_index] = target_ctrl
        self.env.step(ctrl)
        self._eef_steps += 1

        current = float(np.asarray(self.env.data.ctrl)[self.eef_ctrl_index])
        eef_qidx = self.env._op_eef_qidx[self.operator_name]
        actual = float(self.env.data.qpos[eef_qidx[0]]) if len(eef_qidx) > 0 else 0.0
        error = abs(actual - target_ctrl)
        grasped_name = ""
        reached = False
        event = "eef_moving"
        if (
            eef.close
            and self._eef_steps >= self.control.grasp.settle_steps
            and self._last_target is not None
            and self._is_target_grasped(self._last_target)
        ):
            reached = True
            event = "eef_grasped"
            grasped_name = self._last_target.name
        elif eef.close and actual >= max(
            target_ctrl - self.control.tolerance.eef, 0.45
        ):
            reached = True
            event = "eef_reached"
        elif not eef.close and actual <= max(self.control.tolerance.eef, 0.05):
            reached = True
            event = "eef_reached"
        details = {
            "event": event,
            "operator": self.name,
            "eef": eef.model_dump(mode="json"),
            "target_ctrl": target_ctrl,
            "actual_qpos": actual,
            "actual_ctrl": current,
            "error": error,
            "grasped_object": grasped_name,
        }

        # Add grasp detection details when closing gripper
        if eef.close and self._last_target is not None:
            grasp_check = self._check_grasp_conditions(self._last_target)
            details["grasp_check"] = grasp_check

        if reached:
            return ControlResult(signal=ControlSignal.REACHED, details=details)
        if self._eef_steps >= self.control.timeout_steps:
            details["event"] = "eef_timeout"
            return ControlResult(signal=ControlSignal.TIMED_OUT, details=details)
        return ControlResult(signal=ControlSignal.RUNNING, details=details)

    def get_end_effector_pose(self) -> PoseState:
        pos, quat = self.env.get_site_pose(self.eef_site_name)
        return PoseState(
            position=tuple(float(v) for v in pos),
            orientation=tuple(float(v) for v in quat),
        )

    def _orientation_error(self, quat1: tuple, quat2: tuple) -> float:
        """Calculate angular error between two quaternions in radians."""
        quat_dot = abs(float(np.dot(np.asarray(quat1), np.asarray(quat2))))
        quat_dot = min(1.0, max(-1.0, quat_dot))
        return 2.0 * np.arccos(quat_dot)

    def get_base_pose(self) -> PoseState:
        pos, quat = self.env.get_body_pose(self.root_body_name)
        return PoseState(
            position=tuple(float(v) for v in pos),
            orientation=tuple(float(v) for v in quat),
        )

    def _is_target_grasped(self, target: "MujocoObjectHandler") -> bool:
        grasp_check = self._check_grasp_conditions(target)
        return (
            grasp_check["left_contact"]
            and grasp_check["right_contact"]
            and grasp_check["lateral_ok"]
        )

    def _check_grasp_conditions(self, target: "MujocoObjectHandler") -> dict:
        """Check individual grasp conditions and return detailed status."""
        target_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name
        )
        if target_body_id < 0:
            return {
                "left_contact": False,
                "right_contact": False,
                "lateral_ok": False,
                "lateral_error": float("inf"),
                "lateral_threshold": 0.03,
            }

        left_contact = False
        right_contact = False
        for idx in range(self.env.data.ncon):
            contact = self.env.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.env.model.geom_bodyid[geom1])
            body2 = int(self.env.model.geom_bodyid[geom2])
            if target_body_id not in {body1, body2}:
                continue
            other_geom = geom2 if body1 == target_body_id else geom1
            other_name = (
                mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                or ""
            )
            if other_name.startswith("left_"):
                left_contact = True
            if other_name.startswith("right_"):
                right_contact = True

        target_pose = target.get_pose()
        eef_pose = self.get_end_effector_pose()

        lateral_threshold = self.control.grasp.lateral_threshold
        if lateral_threshold <= 0:
            # Distance check disabled
            lateral_ok = True
            lateral_error = 0.0
        else:
            # Convert object position to EEF frame
            obj_pos = np.asarray(target_pose.position, dtype=np.float64)
            eef_pos = np.asarray(eef_pose.position, dtype=np.float64)
            eef_quat = np.asarray(eef_pose.orientation, dtype=np.float64)  # xyzw

            # Quaternion to rotation matrix (xyzw format)
            qx, qy, qz, qw = eef_quat
            R = np.array(
                [
                    [
                        1 - 2 * (qy**2 + qz**2),
                        2 * (qx * qy - qz * qw),
                        2 * (qx * qz + qy * qw),
                    ],
                    [
                        2 * (qx * qy + qz * qw),
                        1 - 2 * (qx**2 + qz**2),
                        2 * (qy * qz - qx * qw),
                    ],
                    [
                        2 * (qx * qz - qy * qw),
                        2 * (qy * qz + qx * qw),
                        1 - 2 * (qx**2 + qy**2),
                    ],
                ]
            )

            # Transform to EEF frame and compute lateral error perpendicular to grasp axis
            obj_in_eef = R.T @ (obj_pos - eef_pos)
            grasp_axis = self.control.grasp.grasp_axis
            lateral_indices = [i for i in range(3) if i != grasp_axis]
            lateral_error = float(np.linalg.norm(obj_in_eef[lateral_indices]))
            lateral_ok = lateral_error <= lateral_threshold

        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "lateral_ok": lateral_ok,
            "lateral_error": lateral_error,
            "lateral_threshold": lateral_threshold,
        }

    def reset_state(self) -> None:
        self._last_move_key = None
        self._last_eef_key = None
        self._last_target = None
        self._move_steps = 0
        self._eef_steps = 0

    def home(self) -> None:
        self.reset_state()
        # Restore mocap target to home pose.
        self.env.data.mocap_pos[self._mocap_id] = self._home_mocap_pos.copy()
        self.env.data.mocap_quat[self._mocap_id] = self._home_mocap_quat.copy()
        # Teleport physical body to match (instant, no weld drift).
        adr = self._fj_qpos_adr
        self.env.data.qpos[adr : adr + 3] = self._home_mocap_pos
        self.env.data.qpos[adr + 3 : adr + 7] = self._home_mocap_quat
        self.env.data.qvel[self._fj_dof_adr : self._fj_dof_adr + 6] = 0.0
        # Restore gripper ctrl and joint state.
        _, eef_qidx, _, eef_vidx, _, eef_aidx = (
            self.env._split_component_joint_state_indices(self.operator_name)
        )
        n = min(len(self._home_ctrl), self.env.model.nu)
        for i, aidx in enumerate(eef_aidx):
            ai = int(aidx)
            if ai < n:
                self.env.data.ctrl[ai] = self._home_ctrl[ai]
                if i < len(eef_qidx):
                    self.env.data.qpos[eef_qidx[i]] = self._home_ctrl[ai]
        if len(eef_vidx) > 0:
            self.env.data.qvel[eef_vidx] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def set_home_end_effector_pose(self, pose: PoseState) -> None:
        """Update the home mocap pose from a desired EEF world pose."""
        desired_base_pose = compose_pose(pose, inverse_pose(self._tool_pose_in_base))
        self._home_mocap_pos = np.asarray(desired_base_pose.position, dtype=np.float64)
        qx, qy, qz, qw = desired_base_pose.orientation
        self._home_mocap_quat = np.array([qw, qx, qy, qz], dtype=np.float64)

    def set_pose(self, pose: PoseState) -> None:
        """Force-set the operator base world pose in one step.

        Sets both the mocap target and the freejoint qpos so the physical body
        is instantly teleported.  Zeros velocities and calls ``mj_forward`` so
        the simulation state is immediately consistent.

        ``pose`` is interpreted as the desired **base-body** world pose, which
        is the same frame returned by ``get_base_pose``.
        """
        qx, qy, qz, qw = pose.orientation
        pos = np.asarray(pose.position, dtype=np.float64)
        quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
        # Set mocap target.
        self.env.data.mocap_pos[self._mocap_id] = pos
        self.env.data.mocap_quat[self._mocap_id] = quat_wxyz
        # Teleport physical body.
        adr = self._fj_qpos_adr
        self.env.data.qpos[adr : adr + 3] = pos
        self.env.data.qpos[adr + 3 : adr + 7] = quat_wxyz
        self.env.data.qvel[self._fj_dof_adr : self._fj_dof_adr + 6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.reset_state()

    def _compute_tool_pose_in_base(self) -> PoseState:
        base_pose = self.get_base_pose()
        eef_pose = self.get_end_effector_pose()
        return compose_pose(inverse_pose(base_pose), eef_pose)

    def _eef_target(self, eef: EefControlConfig) -> float:
        if eef.joint_positions:
            return float(eef.joint_positions[0])
        return 0.82 if eef.close else 0.0


@dataclass
class MujocoTaskBackend(SceneBackend):
    """Framework backend implemented on top of ``UnifiedMujocoEnv``."""

    env: UnifiedMujocoEnv
    """The registered Mujoco basis environment used by this backend instance."""
    operator_handlers: Dict[str, MujocoOperatorHandler]
    """The operator handlers available to execute task stages."""
    object_handlers: Dict[str, MujocoObjectHandler]
    """The object handlers available for stage target lookup and state queries."""
    randomization: Dict[str, PoseRandomRange] = field(default_factory=dict)
    """Per-entity randomization ranges keyed by object or operator name.

    Applied at the end of every ``reset()`` call.  Each key must match a name
    in ``object_handlers`` or ``operator_handlers``.  Unknown keys are ignored
    with a warning.
    """
    random_seed: Optional[int] = None
    """Seed for the internal NumPy random generator.  ``None`` means non-deterministic."""
    randomization_debug: bool = False
    """When True the first resets cycle through extreme poses for debugging."""
    _rng: np.random.Generator = field(init=False, repr=False)
    _default_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )
    _debug_extreme_queue: Optional[List[Any]] = field(
        init=False, repr=False, default=None
    )
    _debug_extreme_index: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.random_seed)

    def setup(self, config: AutoAtomConfig) -> None:
        for operator in self.operator_handlers.values():
            operator.home()
        self._record_default_poses()

    def reset(self) -> None:
        self.env.reset()
        for operator in self.operator_handlers.values():
            operator.home()
        if not self._default_poses:
            self._record_default_poses()
        if self.randomization:
            self._apply_randomization()
        self.env.refresh_viewer()

    def teardown(self) -> None:
        self.env.close()

    def get_operator_handler(self, name: str) -> MujocoOperatorHandler:
        try:
            return self.operator_handlers[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.operator_handlers)) or "<empty>"
            raise KeyError(
                f"Unknown operator '{name}'. Known operators: {known}"
            ) from exc

    def get_object_handler(self, name: str) -> Optional[MujocoObjectHandler]:
        if not name:
            return None
        try:
            return self.object_handlers[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.object_handlers)) or "<empty>"
            raise KeyError(f"Unknown object '{name}'. Known objects: {known}") from exc

    def get_element_pose(self, name: str) -> PoseState:
        """Look up a named site, body, or joint and return its world pose.

        Resolution order: site → body → joint.  For joints the anchor
        position in world frame is computed from the parent body pose and
        the joint's local ``pos`` attribute.
        """
        model, data = self.env.model, self.env.data
        # Try site
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            pos, quat = self.env.get_site_pose(name)
            return PoseState(
                position=tuple(float(v) for v in pos),
                orientation=tuple(float(v) for v in quat),
            )
        # Try body
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            pos, quat = self.env.get_body_pose(name)
            return PoseState(
                position=tuple(float(v) for v in pos),
                orientation=tuple(float(v) for v in quat),
            )
        # Try joint — compute world anchor from the PARENT body's frame.
        # The joint anchor is fixed in the parent frame; using the joint's own
        # body would give wrong results when the joint has already rotated.
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            joint_bid = model.jnt_bodyid[jid]
            parent_bid = model.body_parentid[joint_bid]
            parent_pos = data.xpos[parent_bid]
            parent_rot = data.xmat[parent_bid].reshape(3, 3)
            # Anchor in parent frame = body's local offset + joint's local pos
            body_local = model.body_pos[joint_bid]
            anchor_in_parent = body_local + model.jnt_pos[jid]
            world_pos = parent_pos + parent_rot @ anchor_in_parent
            parent_quat_wxyz = data.xquat[parent_bid]
            qx, qy, qz, qw = (
                parent_quat_wxyz[1],
                parent_quat_wxyz[2],
                parent_quat_wxyz[3],
                parent_quat_wxyz[0],
            )
            return PoseState(
                position=tuple(float(v) for v in world_pos),
                orientation=(float(qx), float(qy), float(qz), float(qw)),
            )
        raise KeyError(
            f"No site, body, or joint named '{name}' found in the MuJoCo model."
        )

    def get_joint_angle(self, name: str) -> float:
        model, data = self.env.model, self.env.data
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"No joint named '{name}' found in the MuJoCo model.")
        qadr = model.jnt_qposadr[jid]
        return float(data.qpos[qadr])

    def is_object_grasped(self, operator_name: str, object_name: str) -> bool:
        operator = self.get_operator_handler(operator_name)
        target = self.get_object_handler(object_name)
        if target is None:
            return False

        target_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name
        )
        if target_body_id < 0:
            return False

        left_contact = False
        right_contact = False
        for idx in range(self.env.data.ncon):
            contact = self.env.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.env.model.geom_bodyid[geom1])
            body2 = int(self.env.model.geom_bodyid[geom2])
            if target_body_id not in {body1, body2}:
                continue
            other_geom = geom2 if body1 == target_body_id else geom1
            other_name = (
                mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom)
                or ""
            )
            if other_name.startswith("left_"):
                left_contact = True
            if other_name.startswith("right_"):
                right_contact = True

        if not (left_contact and right_contact):
            return False

        target_pose = target.get_pose()
        eef_pose = operator.get_end_effector_pose()

        # Convert to EEF frame
        obj_pos = np.asarray(target_pose.position, dtype=np.float64)
        eef_pos = np.asarray(eef_pose.position, dtype=np.float64)
        qx, qy, qz, qw = np.asarray(eef_pose.orientation, dtype=np.float64)

        # Rotation matrix from quaternion
        R = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qz * qw),
                    2 * (qx * qz + qy * qw),
                ],
                [
                    2 * (qx * qy + qz * qw),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qx * qw),
                ],
                [
                    2 * (qx * qz - qy * qw),
                    2 * (qy * qz + qx * qw),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )

        obj_in_eef = R.T @ (obj_pos - eef_pos)
        grasp_axis = 2  # Default Z-axis
        lateral_indices = [i for i in range(3) if i != grasp_axis]
        lateral_error = np.linalg.norm(obj_in_eef[lateral_indices])
        return bool(lateral_error <= 0.03)

    def is_operator_grasping(self, operator_name: str) -> bool:
        _ = self.get_operator_handler(operator_name)
        for object_name in self.object_handlers:
            if self.is_object_grasped(operator_name, object_name):
                return True
        return False

    def is_operator_contacting(self, operator_name: str, object_name: str) -> bool:
        """Return True if any geom of the operator is in contact with the target object."""
        operator = self.get_operator_handler(operator_name)
        target = self.get_object_handler(object_name)
        if target is None:
            return False
        target_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name
        )
        if target_body_id < 0:
            return False
        # Collect all body IDs that belong to this operator (root + descendants).
        root_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, operator.root_body_name
        )
        operator_body_ids: set[int] = set()
        for bid in range(self.env.model.nbody):
            parent = int(self.env.model.body_parentid[bid])
            if bid == root_body_id or (parent in operator_body_ids and bid != 0):
                operator_body_ids.add(bid)
        for idx in range(self.env.data.ncon):
            contact = self.env.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.env.model.geom_bodyid[geom1])
            body2 = int(self.env.model.geom_bodyid[geom2])
            if target_body_id in {body1, body2}:
                other_body = body2 if body1 == target_body_id else body1
                if other_body in operator_body_ids:
                    return True
        return False

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        """Not implemented now"""
        # self.env.set_interest_objects_and_operations(object_names, operation_names)


def create_mujoco_env(
    env_name: str,
    config: EnvConfig,
) -> UnifiedMujocoEnv:
    env = UnifiedMujocoEnv(config)
    ComponentRegistry.register_env(env_name, env)
    return env


def build_mujoco_backend(
    task: AutoAtomConfig | Dict[str, Any],
    operators: List[OperatorConfig] | List[Dict[str, Any]],
) -> MujocoTaskBackend:
    config = (
        task
        if isinstance(task, AutoAtomConfig)
        else AutoAtomConfig.model_validate(task)
    )
    operator_configs = [
        item
        if isinstance(item, OperatorConfig)
        else OperatorConfig.model_validate(item)
        for item in operators
    ]
    env = ComponentRegistry.get_env(config.env_name)
    if not isinstance(env, UnifiedMujocoEnv):
        raise TypeError(
            f"Registered environment '{config.env_name}' must be a UnifiedMujocoEnv, got {type(env).__name__}."
        )

    operator_handlers = {
        operator.name: MujocoOperatorHandler(operator_name=operator.name, env=env)
        for operator in operator_configs
    }
    for operator in operator_configs:
        if operator.initial_state is not None:
            handler = operator_handlers[operator.name]
            if operator.initial_state.arm is not None:
                arm_config = operator.initial_state.arm
                pose = handler.get_end_effector_pose()

                # Handle both old flat list format and new structured format
                if isinstance(arm_config, list):
                    # Old format: [x, y, z, yaw, pitch, roll]
                    if len(arm_config) >= 6:
                        from scipy.spatial.transform import Rotation as R

                        pos = np.array(arm_config[:3], dtype=np.float64)
                        quat_xyzw = R.from_euler(
                            "ZYX", [arm_config[3], arm_config[4], arm_config[5]]
                        ).as_quat()
                        pose = PoseState(
                            position=tuple(float(v) for v in pos),
                            orientation=tuple(float(v) for v in quat_xyzw),
                        )
                else:
                    # New structured format: {position: [...], orientation: [...]}.
                    # This pose is interpreted in the EEF world frame so that
                    # runner.reset() reports the configured values back verbatim.
                    from scipy.spatial.transform import Rotation as R

                    # Override position if provided
                    if arm_config.position is not None:
                        if len(arm_config.position) >= 3:
                            pose = PoseState(
                                position=tuple(
                                    float(v) for v in arm_config.position[:3]
                                ),
                                orientation=pose.orientation,
                            )

                    # Override orientation if provided
                    if arm_config.orientation is not None:
                        ori = arm_config.orientation
                        if len(ori) == 3:
                            # Euler angles: [yaw, pitch, roll]
                            quat_xyzw = R.from_euler("ZYX", ori).as_quat()
                        elif len(ori) == 4:
                            # Quaternion: [x, y, z, w]
                            quat_xyzw = np.array(ori, dtype=np.float64)
                        else:
                            raise ValueError(
                                f"orientation must be 3 floats (Euler) or 4 floats (quaternion), got {len(ori)}"
                            )
                        pose = PoseState(
                            position=pose.position,
                            orientation=tuple(float(v) for v in quat_xyzw),
                        )
                handler.set_home_end_effector_pose(pose)
            if operator.initial_state.eef is not None:
                handler._home_ctrl[handler.eef_ctrl_index] = operator.initial_state.eef
    object_names = {stage.object for stage in config.stages if stage.object}
    object_handlers = {
        object_name: MujocoObjectHandler(
            name=object_name,
            env=env,
            body_name=object_name,
            freejoint_name=f"{object_name}_joint"
            if mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint"
            )
            >= 0
            else (
                f"{object_name}_joint0"
                if mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint0"
                )
                >= 0
                else None
            ),
        )
        for object_name in object_names
    }
    return MujocoTaskBackend(
        env=env,
        operator_handlers=operator_handlers,
        object_handlers=object_handlers,
        randomization=dict(config.randomization),
        random_seed=config.seed if config.seed != 0 else None,
        randomization_debug=config.randomization_debug,
    )
