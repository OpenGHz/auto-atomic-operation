"""Mujoco backend adapting the generic task runner to the basis environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mujoco
import numpy as np

from ...framework import AutoAtomConfig, EefControlConfig, PoseControlConfig, TaskFileConfig
from ...runtime import (
    ControlResult,
    ControlSignal,
    ObjectHandler,
    OperatorHandler,
    SimulatorBackend,
)
from ...utils.pose import PoseState, compose_pose, inverse_pose, quaternion_to_rpy
from ..basis.mujoco_env import EnvConfig, UnifiedMujocoEnv


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
        return PoseState(position=tuple(float(v) for v in pos), orientation=tuple(float(v) for v in quat))

@dataclass
class MujocoOperatorHandler(OperatorHandler):
    """Operator controller backed by the free-flying gripper model."""

    operator_name: str
    """The runtime-visible operator name for this controller."""
    env: UnifiedMujocoEnv
    """The shared Mujoco basis environment used to step the simulation."""
    component: str = "arm"
    """The environment component name this operator is bound to."""
    root_body_name: str = "robotiq_interface"
    """The root body name used to read the operator base pose."""
    eef_site_name: str = "eef_pose"
    """The site name used to read the operator end-effector pose."""
    eef_ctrl_index: int = 6
    """The control index corresponding to the gripper or end-effector actuator."""
    position_tolerance: float = 0.01
    """The Euclidean position tolerance for considering a pose command reached."""
    orientation_tolerance: float = 0.08
    """The angular tolerance in radians for considering a pose orientation reached."""
    eef_tolerance: float = 0.03
    """The tolerance used to determine whether the gripper target has been reached."""
    command_timeout_steps: int = 600
    """The maximum number of simulation steps allowed for one primitive command."""
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
    _home_ctrl: np.ndarray = field(default_factory=lambda: np.asarray([0.0, -0.25, 0.35, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
    """The nominal home control vector used to settle the operator at reset time."""

    @property
    def name(self) -> str:
        return self.operator_name

    def __post_init__(self) -> None:
        self._tool_pose_in_base = self._compute_tool_pose_in_base()

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        simulator: SimulatorBackend,
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        key = str(pose.model_dump(mode="json"))
        if self._last_move_key != key:
            self._last_move_key = key
            self._move_steps = 0
        if isinstance(target, MujocoObjectHandler):
            self._last_target = target

        desired_eef_pose = PoseState(position=pose.position, orientation=pose.orientation)
        desired_base_pose = compose_pose(desired_eef_pose, inverse_pose(self._tool_pose_in_base))
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

        current_pose = self.get_end_effector_pose(simulator)
        pos_error = float(np.linalg.norm(np.asarray(current_pose.position) - np.asarray(pose.position)))
        quat_dot = abs(float(np.dot(np.asarray(current_pose.orientation), np.asarray(pose.orientation))))
        quat_dot = min(1.0, max(-1.0, quat_dot))
        ori_error = 2.0 * np.arccos(quat_dot)

        details = {
            "event": "moving" if pos_error > self.position_tolerance or ori_error > self.orientation_tolerance else "pose_reached",
            "operator": self.name,
            "target": target.name if target else "",
            "pose": pose.model_dump(mode="json"),
            "position_error": pos_error,
            "orientation_error": float(ori_error),
        }
        if pos_error <= self.position_tolerance and ori_error <= self.orientation_tolerance:
            return ControlResult(signal=ControlSignal.REACHED, details=details)
        if self._move_steps >= self.command_timeout_steps:
            details["event"] = "move_timeout"
            return ControlResult(signal=ControlSignal.TIMED_OUT, details=details)
        return ControlResult(signal=ControlSignal.RUNNING, details=details)

    def control_eef(
        self,
        eef: EefControlConfig,
        simulator: SimulatorBackend,
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
            and self._last_target is not None
            and simulator.is_object_grasped(self.name, self._last_target.name)
        ):
            reached = True
            event = "eef_grasped"
            grasped_name = self._last_target.name
        elif eef.close and actual >= max(target_ctrl - self.eef_tolerance, 0.45):
            reached = True
            event = "eef_reached"
        elif not eef.close and actual <= max(self.eef_tolerance, 0.05):
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
        if reached:
            return ControlResult(signal=ControlSignal.REACHED, details=details)
        if self._eef_steps >= self.command_timeout_steps:
            details["event"] = "eef_timeout"
            return ControlResult(signal=ControlSignal.TIMED_OUT, details=details)
        return ControlResult(signal=ControlSignal.RUNNING, details=details)

    def get_end_effector_pose(self, simulator: SimulatorBackend) -> PoseState:
        pos, quat = self.env.get_site_pose(self.eef_site_name)
        return PoseState(position=tuple(float(v) for v in pos), orientation=tuple(float(v) for v in quat))

    def get_base_pose(self, simulator: SimulatorBackend) -> PoseState:
        pos, quat = self.env.get_body_pose(self.root_body_name)
        return PoseState(position=tuple(float(v) for v in pos), orientation=tuple(float(v) for v in quat))

    def reset_state(self) -> None:
        self._last_move_key = None
        self._last_eef_key = None
        self._last_target = None
        self._move_steps = 0
        self._eef_steps = 0

    def home(self, settle_steps: int = 200) -> None:
        self.reset_state()
        for _ in range(settle_steps):
            self.env.step(self._home_ctrl)

    def _compute_tool_pose_in_base(self) -> PoseState:
        base_pose = self.get_base_pose(None)  # type: ignore[arg-type]
        eef_pose = self.get_end_effector_pose(None)  # type: ignore[arg-type]
        return compose_pose(inverse_pose(base_pose), eef_pose)

    def _eef_target(self, eef: EefControlConfig) -> float:
        if eef.joint_positions:
            return float(eef.joint_positions[0])
        return 0.82 if eef.close else 0.0


@dataclass
class MujocoTaskBackend(SimulatorBackend):
    """Framework backend implemented on top of ``UnifiedMujocoEnv``."""

    env: UnifiedMujocoEnv
    """The registered Mujoco basis environment used by this backend instance."""
    operator_handlers: Dict[str, MujocoOperatorHandler]
    """The operator handlers available to execute task stages."""
    object_handlers: Dict[str, MujocoObjectHandler]
    """The object handlers available for stage target lookup and state queries."""

    def setup(self, config: AutoAtomConfig) -> None:
        for operator in self.operator_handlers.values():
            operator.home(settle_steps=120)

    def reset(self) -> None:
        self.env.reset()
        for operator in self.operator_handlers.values():
            operator.home(settle_steps=120)

    def teardown(self) -> None:
        self.env.close()

    def get_operator_handler(self, name: str) -> MujocoOperatorHandler:
        try:
            return self.operator_handlers[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.operator_handlers)) or "<empty>"
            raise KeyError(f"Unknown operator '{name}'. Known operators: {known}") from exc

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

        target_body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name)
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
            other_name = mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_name.startswith("left_"):
                left_contact = True
            if other_name.startswith("right_"):
                right_contact = True

        if not (left_contact and right_contact):
            return False

        target_pose = target.get_pose()
        eef_pose = operator.get_end_effector_pose(self)
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

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        """Not implemented now"""
        # self.env.set_interest_objects_and_operations(object_names, operation_names)


def create_mujoco_env(
    registry,
    env_name: str,
    config: EnvConfig,
) -> UnifiedMujocoEnv:
    env = UnifiedMujocoEnv(config)
    registry.register_env(env_name, env)
    return env


def build_mujoco_backend(task_file: TaskFileConfig, registry) -> MujocoTaskBackend:
    config = task_file.task
    env = registry.get_env(config.env_name)
    if not isinstance(env, UnifiedMujocoEnv):
        raise TypeError(
            f"Registered environment '{config.env_name}' must be a UnifiedMujocoEnv, got {type(env).__name__}."
        )

    operator_handlers = {
        operator.name: MujocoOperatorHandler(operator_name=operator.name, env=env)
        for operator in task_file.operators
    }
    object_names = {stage.object for stage in config.stages if stage.object}
    object_handlers = {
        object_name: MujocoObjectHandler(
            name=object_name,
            env=env,
            body_name=object_name,
            freejoint_name=f"{object_name}_joint" if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint") >= 0 else (
                f"{object_name}_joint0" if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint0") >= 0 else None
            ),
        )
        for object_name in object_names
    }
    return MujocoTaskBackend(
        env=env,
        operator_handlers=operator_handlers,
        object_handlers=object_handlers,
    )
