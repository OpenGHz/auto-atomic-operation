"""Mujoco backend adapting the generic task runner to the basis environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np
from pydantic import BaseModel

from ...framework import (
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
from ...utils.pose import PoseState, compose_pose, inverse_pose, quaternion_to_rpy
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

    horizontal_threshold: float = 0.03
    """Max horizontal distance (meters) between object center and EEF for valid grasp.
    Increase for side grasps (e.g., 0.06-0.10 for cup handles)."""
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
    eef_ctrl_index: int = 6
    """The control index corresponding to the gripper or end-effector actuator."""
    control: MujocoControlConfig = field(default_factory=MujocoControlConfig)
    """Control parameters including tolerances, grasp detection, and timeouts."""
    _tool_pose_in_base: PoseState = field(init=False)
    """The fixed transform from the operator base frame to the tool frame."""
    _last_move_key: Optional[str] = None
    """The serialized pose command currently tracked by the motion controller."""
    _last_eef_key: Optional[str] = None
    """The serialized end-effector command currently tracked by the controller."""
    _last_target: Optional[MujocoObjectHandler] = None
    """The last target object involved in a primitive action, if any."""
    _move_steps: int = 0
    """The number of simulation steps consumed by the active pose command."""
    _eef_steps: int = 0
    """The number of simulation steps consumed by the active eef command."""
    _home_ctrl: np.ndarray = field(init=False, repr=False)
    """The nominal home control vector used to settle the operator at reset time.
    Initialised from the environment's current ctrl state (i.e. the keyframe)
    so that home() always returns the arm to the scene-defined initial pose."""

    @property
    def name(self) -> str:
        return self.operator_name

    def __post_init__(self) -> None:
        # Snapshot the env ctrl at construction time (env has already applied the
        # keyframe in its __init__).  This ensures home() brings the arm back to
        # the scene-defined initial position rather than a hard-coded constant.
        self._home_ctrl = np.asarray(
            self.env.data.ctrl[: self.env.model.nu], dtype=np.float64
        ).copy()
        self._tool_pose_in_base = self._compute_tool_pose_in_base()

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        key = str(pose.model_dump(mode="json"))
        if self._last_move_key != key:
            self._last_move_key = key
            self._move_steps = 0
        if isinstance(target, MujocoObjectHandler):
            self._last_target = target

        desired_eef_pose = PoseState(
            position=pose.position, orientation=pose.orientation
        )
        desired_base_pose = compose_pose(
            desired_eef_pose, inverse_pose(self._tool_pose_in_base)
        )
        roll, pitch, yaw = quaternion_to_rpy(desired_base_pose.orientation)

        ctrl = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        ctrl[:6] = np.asarray(
            [
                desired_base_pose.position[0],
                desired_base_pose.position[1],
                desired_base_pose.position[2],
                yaw,
                pitch,
                roll,
            ],
            dtype=np.float64,
        )

        self.env.step(ctrl)
        self._move_steps += 1

        current_pose = self.get_end_effector_pose()
        pos_error = float(
            np.linalg.norm(
                np.asarray(current_pose.position) - np.asarray(pose.position)
            )
        )
        quat_dot = abs(
            float(
                np.dot(
                    np.asarray(current_pose.orientation), np.asarray(pose.orientation)
                )
            )
        )
        quat_dot = min(1.0, max(-1.0, quat_dot))
        ori_error = 2.0 * np.arccos(quat_dot)

        details = {
            "event": "moving"
            if pos_error > self.control.tolerance.position
            or ori_error > self.control.tolerance.orientation
            else "pose_reached",
            "operator": self.name,
            "target": target.name if target else "",
            "pose": pose.model_dump(mode="json"),
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
        actual = float(np.asarray(self.env.data.qpos)[6])
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
            and grasp_check["horizontal_ok"]
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
                "horizontal_ok": False,
                "horizontal_error": float("inf"),
                "horizontal_threshold": 0.03,
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
        horizontal_error = float(
            np.linalg.norm(
                np.asarray(target_pose.position[:2], dtype=np.float64)
                - np.asarray(eef_pose.position[:2], dtype=np.float64)
            )
        )
        horizontal_threshold = self.control.grasp.horizontal_threshold
        horizontal_ok = horizontal_error <= horizontal_threshold

        return {
            "left_contact": left_contact,
            "right_contact": right_contact,
            "horizontal_ok": horizontal_ok,
            "horizontal_error": horizontal_error,
            "horizontal_threshold": horizontal_threshold,
        }

    def reset_state(self) -> None:
        self._last_move_key = None
        self._last_eef_key = None
        self._last_target = None
        self._move_steps = 0
        self._eef_steps = 0

    def home(self) -> None:
        self.reset_state()
        arm_qidx, eef_qidx, arm_vidx, eef_vidx, arm_aidx, eef_aidx = (
            self.env._split_component_joint_state_indices(self.operator_name)
        )
        n = min(len(self._home_ctrl), self.env.model.nu)
        ctrl = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        ctrl[:n] = self._home_ctrl[:n]
        low = self.env.model.actuator_ctrlrange[:n, 0]
        high = self.env.model.actuator_ctrlrange[:n, 1]
        ctrl[:n] = np.clip(ctrl[:n], low, high)
        self.env.data.ctrl[:n] = ctrl[:n]
        for i, aidx in enumerate(arm_aidx):
            if i < len(arm_qidx) and int(aidx) < n:
                self.env.data.qpos[arm_qidx[i]] = ctrl[int(aidx)]
        for i, aidx in enumerate(eef_aidx):
            if i < len(eef_qidx) and int(aidx) < n:
                self.env.data.qpos[eef_qidx[i]] = ctrl[int(aidx)]
        all_vidx = np.concatenate([arm_vidx, eef_vidx])
        if len(all_vidx) > 0:
            self.env.data.qvel[all_vidx] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def set_pose(self, pose: PoseState) -> None:
        """Force-set the operator base world pose in one step.

        Writes the desired position and orientation directly into both the
        control vector and the corresponding qpos entries, zeros velocities,
        and calls ``mj_forward`` so the simulation state is immediately
        consistent.  The controller state is reset so the next ``move_to_pose``
        call starts fresh.

        ``pose`` is interpreted as the desired **base-body** world pose, which
        is the same frame returned by ``get_base_pose``.
        """
        roll, pitch, yaw = quaternion_to_rpy(pose.orientation)
        arm_qidx, _, arm_vidx, eef_vidx, arm_aidx, _ = (
            self.env._split_component_joint_state_indices(self.operator_name)
        )
        n = min(len(self._home_ctrl), self.env.model.nu)
        new_ctrl = np.asarray(self.env.data.ctrl, dtype=np.float64).copy()
        # ctrl layout: [x, y, z, yaw, pitch, roll, gripper, ...]
        target = np.asarray(
            [pose.position[0], pose.position[1], pose.position[2], yaw, pitch, roll],
            dtype=np.float64,
        )
        n_arm = min(len(arm_aidx), len(target))
        for i in range(n_arm):
            aidx = int(arm_aidx[i])
            if aidx < n:
                new_ctrl[aidx] = target[i]
        low = self.env.model.actuator_ctrlrange[:n, 0]
        high = self.env.model.actuator_ctrlrange[:n, 1]
        new_ctrl[:n] = np.clip(new_ctrl[:n], low, high)
        self.env.data.ctrl[:n] = new_ctrl[:n]
        for i, aidx in enumerate(arm_aidx):
            if i < len(arm_qidx) and int(aidx) < n:
                self.env.data.qpos[arm_qidx[i]] = new_ctrl[int(aidx)]
        all_vidx = np.concatenate([arm_vidx, eef_vidx])
        if len(all_vidx) > 0:
            self.env.data.qvel[all_vidx] = 0.0
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
        horizontal_error = np.linalg.norm(
            np.asarray(target_pose.position[:2], dtype=np.float64)
            - np.asarray(eef_pose.position[:2], dtype=np.float64)
        )
        return bool(horizontal_error <= 0.03)

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
                vals = operator.initial_state.arm
                handler._home_ctrl[: len(vals)] = vals
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
