"""Mujoco backend adapting the generic task runner to batched basis envs."""

from __future__ import annotations

import logging
import mujoco
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from ...basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv, EnvConfig
from ...framework import (
    ArmPoseConfig,
    AutoAtomConfig,
    EefControlConfig,
    OperatorRandomizationConfig,
    OperatorConfig,
    PoseControlConfig,
    PoseRandomRange,
    PoseReference,
)
from ...runtime import (
    ComponentRegistry,
    ControlResult,
    ControlSignal,
    IKSolver,
    ObjectHandler,
    OperatorHandler,
    SceneBackend,
)
from ...utils.pose import (
    PoseState,
    euler_to_quaternion,
    quaternion_angular_distance,
    quaternion_to_rpy,
    quaternion_to_rotation_matrix,
)
from ...utils.transformations import quaternion_slerp


class MujocoToleranceConfig(BaseModel):
    position: float = 0.01
    orientation: float = 0.08
    eef: float = 0.03


class MujocoGraspConfig(BaseModel):
    lateral_threshold: float = 0.0
    grasp_axis: int = 2
    settle_steps: int = 5


class MujocoControlConfig(BaseModel):
    timeout_steps: int = 600
    tolerance: MujocoToleranceConfig = MujocoToleranceConfig()
    grasp: MujocoGraspConfig = MujocoGraspConfig()
    cartesian_max_linear_step: float = 0.0
    cartesian_max_angular_step: float = 0.0
    adaptive_step_scaling: bool = False
    """When True, automatically reduce step scale on stall and recover on progress.
    Set to False for contact-heavy tasks (e.g. door pushing) where stall detection
    causes unnecessary slowdown."""


@dataclass
class MujocoObjectHandler(ObjectHandler):
    env: BatchedUnifiedMujocoEnv
    body_name: str
    freejoint_name: Optional[str] = None

    def get_pose(self) -> PoseState:
        pos, quat = self.env.get_body_pose(self.body_name)
        return PoseState(position=pos, orientation=quat)

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        if self.freejoint_name is None:
            return
        pose = pose.broadcast_to(self.env.batch_size)
        mask = (
            np.ones(self.env.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, single_env in enumerate(self.env.envs):
            if not mask[env_index]:
                continue
            jid = mujoco.mj_name2id(
                single_env.model, mujoco.mjtObj.mjOBJ_JOINT, self.freejoint_name
            )
            if jid < 0:
                continue
            qpos_adr = int(single_env.model.jnt_qposadr[jid])
            dof_adr = int(single_env.model.jnt_dofadr[jid])
            x, y, z = pose.position[env_index]
            qx, qy, qz, qw = pose.orientation[env_index]
            single_env.data.qpos[qpos_adr : qpos_adr + 7] = [x, y, z, qw, qx, qy, qz]
            single_env.data.qvel[dof_adr : dof_adr + 6] = 0.0
            mujoco.mj_forward(single_env.model, single_env.data)


@dataclass
class MujocoOperatorHandler(OperatorHandler):
    operator_name: str
    env: BatchedUnifiedMujocoEnv
    root_body_name: str = "robotiq_interface"
    eef_site_name: str = "eef_pose"
    mocap_body_name: str = "robotiq_mocap"
    freejoint_name: str = "robotiq_freejoint"
    eef_ctrl_index: int = 0
    eef_open_value: float = 0.0
    eef_close_value: float = 0.82
    control: MujocoControlConfig = field(default_factory=MujocoControlConfig)
    ik_solver: Optional[IKSolver] = None
    joint_control_mode: str = "per_step_ik"
    joint_interp_speed: float = 0.05

    _last_move_key: List[str | None] = field(init=False, repr=False)
    _last_eef_key: List[str | None] = field(init=False, repr=False)
    _last_target: List[Optional[MujocoObjectHandler]] = field(init=False, repr=False)
    _move_steps: np.ndarray = field(init=False, repr=False)
    _move_best_pos_error: np.ndarray = field(init=False, repr=False)
    _move_best_ori_error: np.ndarray = field(init=False, repr=False)
    _move_stall_count: np.ndarray = field(init=False, repr=False)
    _move_step_scale: np.ndarray = field(init=False, repr=False)
    _eef_steps: np.ndarray = field(init=False, repr=False)
    _home_ctrl: np.ndarray = field(init=False, repr=False)

    @property
    def name(self) -> str:
        return self.operator_name

    def __post_init__(self) -> None:
        self._last_move_key = [None] * self.env.batch_size
        self._last_eef_key = [None] * self.env.batch_size
        self._last_target = [None] * self.env.batch_size
        self._move_steps = np.zeros(self.env.batch_size, dtype=np.int64)
        self._move_best_pos_error = np.full(
            self.env.batch_size, np.inf, dtype=np.float64
        )
        self._move_best_ori_error = np.full(
            self.env.batch_size, np.inf, dtype=np.float64
        )
        self._move_stall_count = np.zeros(self.env.batch_size, dtype=np.int64)
        self._move_step_scale = np.ones(self.env.batch_size, dtype=np.float64)
        self._eef_steps = np.zeros(self.env.batch_size, dtype=np.int64)
        self._home_ctrl = np.stack(
            [
                np.asarray(
                    single_env.data.ctrl[: single_env.model.nu], dtype=np.float64
                ).copy()
                for single_env in self.env.envs
            ],
            axis=0,
        )
        self.env.register_operator(
            self.operator_name,
            root_body=self.root_body_name,
            eef_site=self.eef_site_name,
            ik_solver=self.ik_solver,
            mocap_body=self.mocap_body_name,
            freejoint=self.freejoint_name,
            joint_control_mode=self.joint_control_mode,
            joint_interp_speed=self.joint_interp_speed,
        )

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        mask = self._normalize_mask(env_mask)
        current_eef = self.get_end_effector_pose()
        desired_pos = np.repeat(
            np.asarray(pose.position, dtype=np.float64).reshape(1, 3),
            self.env.batch_size,
            axis=0,
        )
        desired_ori = np.repeat(
            np.asarray(pose.orientation, dtype=np.float64).reshape(1, 4),
            self.env.batch_size,
            axis=0,
        )
        signals = np.asarray(
            [ControlSignal.RUNNING] * self.env.batch_size, dtype=object
        )
        details = [{} for _ in range(self.env.batch_size)]
        for env_index in range(self.env.batch_size):
            if not mask[env_index]:
                continue
            key = str(pose.model_dump(mode="json"))
            if self._last_move_key[env_index] != key:
                self._last_move_key[env_index] = key
                self._move_steps[env_index] = 0
                self._move_best_pos_error[env_index] = float("inf")
                self._move_best_ori_error[env_index] = float("inf")
                self._move_stall_count[env_index] = 0
                self._move_step_scale[env_index] = 1.0
            if isinstance(target, MujocoObjectHandler):
                self._last_target[env_index] = target

            pos_err = float(
                np.linalg.norm(current_eef.position[env_index] - desired_pos[env_index])
            )
            ori_err = quaternion_angular_distance(
                current_eef.orientation[env_index], desired_ori[env_index]
            )
            if self.control.adaptive_step_scaling:
                improved = pos_err < (
                    self._move_best_pos_error[env_index] - 1e-4
                ) or ori_err < (self._move_best_ori_error[env_index] - 1e-3)
                if improved:
                    self._move_best_pos_error[env_index] = min(
                        self._move_best_pos_error[env_index], pos_err
                    )
                    self._move_best_ori_error[env_index] = min(
                        self._move_best_ori_error[env_index], ori_err
                    )
                    self._move_stall_count[env_index] = 0
                    self._move_step_scale[env_index] = min(
                        1.0, self._move_step_scale[env_index] * 1.1
                    )
                else:
                    self._move_stall_count[env_index] += 1
                    if self._move_stall_count[env_index] >= 8:
                        self._move_step_scale[env_index] = max(
                            0.1, self._move_step_scale[env_index] * 0.5
                        )
                        self._move_stall_count[env_index] = 0

            max_linear_step = (
                float(
                    pose.max_linear_step
                    if pose.max_linear_step > 0.0
                    else self.control.cartesian_max_linear_step
                )
                * self._move_step_scale[env_index]
            )
            max_angular_step = (
                float(
                    pose.max_angular_step
                    if pose.max_angular_step > 0.0
                    else self.control.cartesian_max_angular_step
                )
                * self._move_step_scale[env_index]
            )
            pos_goal = desired_pos[env_index].copy()
            ori_goal = desired_ori[env_index].copy()
            if max_linear_step > 0.0:
                pos_delta = pos_goal - current_eef.position[env_index]
                pos_dist = float(np.linalg.norm(pos_delta))
                if pos_dist > max_linear_step:
                    pos_goal = current_eef.position[env_index] + pos_delta * (
                        max_linear_step / pos_dist
                    )
            if max_angular_step > 0.0 and ori_err > max_angular_step:
                ori_goal = quaternion_slerp(
                    current_eef.orientation[env_index],
                    ori_goal,
                    fraction=max_angular_step / ori_err,
                )

            target_pos_b, target_quat_b = self.env.world_to_base(
                self.operator_name, pos_goal, ori_goal
            )
            batched_pos_b = np.zeros((self.env.batch_size, 3), dtype=np.float32)
            batched_quat_b = np.zeros((self.env.batch_size, 4), dtype=np.float32)
            batched_pos_b[env_index] = target_pos_b[env_index]
            batched_quat_b[env_index] = target_quat_b[env_index]
            self.env.step_operator_toward_target(
                self.operator_name,
                batched_pos_b,
                batched_quat_b,
                env_mask=np.eye(self.env.batch_size, dtype=bool)[env_index],
            )
            self._move_steps[env_index] += 1
            eef_world_after = self.get_end_effector_pose()
            pos_err_after = float(
                np.linalg.norm(eef_world_after.position[env_index] - pos_goal)
            )
            ori_err_after = quaternion_angular_distance(
                eef_world_after.orientation[env_index], ori_goal
            )
            event = (
                "moving"
                if pos_err_after > self.control.tolerance.position
                or ori_err_after > self.control.tolerance.orientation
                else "pose_reached"
            )
            details[env_index] = {
                "event": event,
                "operator": self.name,
                "target": target.name if target else "",
                "target_pose": pose.model_dump(mode="json"),
                "current_pose": {
                    "position": [float(v) for v in eef_world_after.position[env_index]],
                    "orientation": [
                        float(v) for v in eef_world_after.orientation[env_index]
                    ],
                },
                "position_error": pos_err_after,
                "orientation_error": ori_err_after,
                "steps": int(self._move_steps[env_index]),
            }
            if (
                pos_err_after <= self.control.tolerance.position
                and ori_err_after <= self.control.tolerance.orientation
            ):
                signals[env_index] = ControlSignal.REACHED
                self._move_steps[env_index] = 0
            elif self._move_steps[env_index] >= self.control.timeout_steps:
                details[env_index]["event"] = "move_timeout"
                signals[env_index] = ControlSignal.TIMED_OUT
            else:
                signals[env_index] = ControlSignal.RUNNING
        return ControlResult(signals=signals, details=details)

    def control_eef(
        self,
        eef: EefControlConfig,
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        mask = self._normalize_mask(env_mask)
        target_value = self._eef_target(eef)
        signals = np.asarray(
            [ControlSignal.RUNNING] * self.env.batch_size, dtype=object
        )
        details = [{} for _ in range(self.env.batch_size)]
        for env_index, single_env in enumerate(self.env.envs):
            if not mask[env_index]:
                continue
            command_key = f"{eef.close}:{eef.joint_positions}"
            if self._last_eef_key[env_index] != command_key:
                self._last_eef_key[env_index] = command_key
                self._eef_steps[env_index] = 0
            ctrl = np.asarray(single_env.data.ctrl, dtype=np.float64).copy()
            ctrl[self.eef_ctrl_index] = target_value
            self.env.step(
                np.vstack(
                    [
                        ctrl
                        if i == env_index
                        else np.asarray(env.data.ctrl[: env.model.nu], dtype=np.float64)
                        for i, env in enumerate(self.env.envs)
                    ]
                ),
                env_mask=np.eye(self.env.batch_size, dtype=bool)[env_index],
            )
            self._eef_steps[env_index] += 1
            current = float(np.asarray(single_env.data.ctrl)[self.eef_ctrl_index])
            eef_qidx = single_env._op_eef_qidx[self.operator_name]
            actual = (
                float(single_env.data.qpos[eef_qidx[0]]) if len(eef_qidx) > 0 else 0.0
            )
            error = abs(actual - target_value)
            grasped_name = ""
            reached = False
            settle_ready = self._eef_steps[env_index] >= self.control.grasp.settle_steps
            event = "eef_moving"
            if (
                eef.close
                and settle_ready
                and self._last_target[env_index] is not None
                and self._is_target_grasped(env_index, self._last_target[env_index])
            ):
                reached = True
                grasped_name = self._last_target[env_index].name
                event = "eef_grasped"
            elif eef.close and actual >= (target_value - self.control.tolerance.eef):
                reached = True
                event = "eef_reached"
            elif (
                eef.close
                and self._eef_steps[env_index]
                >= max(self.control.grasp.settle_steps, 30)
                and actual > self.eef_open_value + self.control.tolerance.eef * 0.1
            ):
                # Gripper commanded to close but physically blocked by an
                # object — qpos may never reach the target.  Accept when
                # the actuator has had enough time and qpos has moved
                # noticeably away from the fully-open position.
                reached = True
                event = "eef_reached"
            elif not eef.close and actual <= (
                self.eef_open_value + self.control.tolerance.eef
            ):
                reached = True
                event = "eef_reached"
            details[env_index] = {
                "event": event,
                "operator": self.name,
                "eef": eef.model_dump(mode="json"),
                "target_ctrl": target_value,
                "actual_qpos": actual,
                "actual_ctrl": current,
                "error": error,
                "eef_target": target_value,
                "eef_command": current,
                "eef_actual": actual,
                "eef_error": error,
                "settle_ready": settle_ready,
                "steps": int(self._eef_steps[env_index]),
                "grasped_object": grasped_name,
            }
            if eef.close and self._last_target[env_index] is not None:
                details[env_index]["grasp_check"] = self._check_grasp_conditions(
                    env_index, self._last_target[env_index]
                )
            if reached:
                signals[env_index] = ControlSignal.REACHED
                self._eef_steps[env_index] = 0
            elif self._eef_steps[env_index] >= self.control.timeout_steps:
                details[env_index]["event"] = "eef_timeout"
                signals[env_index] = ControlSignal.TIMED_OUT
            else:
                signals[env_index] = ControlSignal.RUNNING
        return ControlResult(signals=signals, details=details)

    def get_end_effector_pose(self) -> PoseState:
        pos, quat = self.env.get_operator_eef_pose_in_world(self.operator_name)
        return PoseState(position=pos, orientation=quat)

    def get_base_pose(self) -> PoseState:
        pos, quat = self.env.get_operator_base_pose(self.operator_name)
        return PoseState(position=pos, orientation=quat)

    def reset_state(self, env_mask: Optional[np.ndarray] = None) -> None:
        mask = self._normalize_mask(env_mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._last_move_key[env_index] = None
                self._last_eef_key[env_index] = None
                self._last_target[env_index] = None
                self._move_steps[env_index] = 0
                self._eef_steps[env_index] = 0
                self._move_best_pos_error[env_index] = float("inf")
                self._move_best_ori_error[env_index] = float("inf")
                self._move_stall_count[env_index] = 0
                self._move_step_scale[env_index] = 1.0

    def home(self, env_mask: Optional[np.ndarray] = None) -> None:
        self.reset_state(env_mask)
        self.env.home_operator(self.operator_name, env_mask=env_mask)

    def set_home_end_effector_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        pose = pose.broadcast_to(self.env.batch_size)
        self.env.set_operator_home_eef_pose(
            self.operator_name,
            pose.position,
            pose.orientation,
            env_mask=env_mask,
        )
        self.home(env_mask)
        mask = self._normalize_mask(env_mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._home_ctrl[env_index, self.eef_ctrl_index] = self.env.envs[
                    env_index
                ].data.ctrl[self.eef_ctrl_index]

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        self.reset_state(env_mask)
        pose = pose.broadcast_to(self.env.batch_size)
        # ``OperatorHandler.set_pose`` is the runtime-facing "base pose" API.
        # For mocap operators this must only update the virtual base frame used
        # for world/base conversions and diagnostics, not physically move the
        # mocap/root body.
        self.env.override_operator_base_pose(
            self.operator_name,
            pose.position,
            pose.orientation,
            env_mask=env_mask,
        )

    def _eef_target(self, eef: EefControlConfig) -> float:
        if eef.joint_positions:
            return float(eef.joint_positions[0])
        return self.eef_close_value if eef.close else self.eef_open_value

    def _normalize_mask(self, env_mask: Optional[np.ndarray]) -> np.ndarray:
        if env_mask is None:
            return np.ones(self.env.batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.env.batch_size:
            raise ValueError(
                f"env_mask must have shape ({self.env.batch_size},), got {mask.shape}"
            )
        return mask

    def _is_target_grasped(
        self,
        env_index: int,
        target: "MujocoObjectHandler",
    ) -> bool:
        grasp_check = self._check_grasp_conditions(env_index, target)
        return bool(
            grasp_check["left_contact"]
            and grasp_check["right_contact"]
            and grasp_check["lateral_ok"]
        )

    def _check_grasp_conditions(
        self,
        env_index: int,
        target: "MujocoObjectHandler",
    ) -> Dict[str, Any]:
        single_env = self.env.envs[env_index]
        target_body_id = mujoco.mj_name2id(
            single_env.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name
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
        for idx in range(single_env.data.ncon):
            contact = single_env.data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(single_env.model.geom_bodyid[geom1])
            body2 = int(single_env.model.geom_bodyid[geom2])
            if target_body_id not in {body1, body2}:
                continue
            other_geom = geom2 if body1 == target_body_id else geom1
            other_name = (
                mujoco.mj_id2name(
                    single_env.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                )
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
            lateral_ok = True
            lateral_error = 0.0
        else:
            obj_pos = np.asarray(target_pose.position[env_index], dtype=np.float64)
            eef_pos = np.asarray(eef_pose.position[env_index], dtype=np.float64)
            rot = quaternion_to_rotation_matrix(eef_pose.orientation[env_index])
            obj_in_eef = rot.T @ (obj_pos - eef_pos)
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


@dataclass
class MujocoTaskBackend(SceneBackend):
    env: BatchedUnifiedMujocoEnv
    operator_handlers: Dict[str, MujocoOperatorHandler]
    object_handlers: Dict[str, MujocoObjectHandler]
    randomization: Dict[str, PoseRandomRange | OperatorRandomizationConfig] = field(
        default_factory=dict
    )
    random_seed: Optional[int] = None
    randomization_debug: bool = False
    _rng: np.random.Generator = field(init=False, repr=False)
    _default_object_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )
    _default_operator_base_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )
    _default_operator_eef_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        logging.getLogger(MujocoTaskBackend.__name__).info(
            "MujocoTaskBackend random_seed=%s", self.random_seed
        )
        self._rng = np.random.default_rng(self.random_seed)

    @property
    def batch_size(self) -> int:
        return self.env.batch_size

    def setup(self, config: AutoAtomConfig) -> None:
        for operator in self.operator_handlers.values():
            operator.home()
        self._record_default_poses()

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        mask = self._normalize_mask(env_mask)
        self.env.reset(mask)
        for operator in self.operator_handlers.values():
            operator.home(mask)
        if not (
            self._default_object_poses
            or self._default_operator_base_poses
            or self._default_operator_eef_poses
        ):
            self._record_default_poses()
        if self.randomization:
            self._apply_randomization(mask)
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

    def _record_default_poses(self) -> None:
        for name, handler in self.object_handlers.items():
            self._default_object_poses[name] = handler.get_pose()
        for name, handler in self.operator_handlers.items():
            self._default_operator_base_poses[name] = handler.get_base_pose()
            self._default_operator_eef_poses[name] = handler.get_end_effector_pose()

    def _apply_randomization(self, env_mask: np.ndarray) -> None:
        for name, rand_range in self.randomization.items():
            if name in self.object_handlers:
                if isinstance(rand_range, OperatorRandomizationConfig):
                    raise TypeError(
                        f"Object '{name}' randomization must be a PoseRandomRange, "
                        "not an operator randomization config."
                    )
                base_pose = self._default_object_poses.get(
                    name, self.object_handlers[name].get_pose()
                )
                sampled = self._sample_random_pose(base_pose, rand_range, env_mask)
                self.object_handlers[name].set_pose(sampled, env_mask=env_mask)
            elif name in self.operator_handlers:
                handler = self.operator_handlers[name]
                if isinstance(rand_range, OperatorRandomizationConfig):
                    if rand_range.base is not None:
                        default_base_pose = self._default_operator_base_poses.get(
                            name, handler.get_base_pose()
                        )
                        sampled_base = self._sample_random_pose(
                            default_base_pose, rand_range.base, env_mask
                        )
                        handler.set_pose(sampled_base, env_mask=env_mask)
                    if rand_range.eef is not None:
                        default_eef_pose = self._default_operator_eef_poses.get(
                            name, handler.get_end_effector_pose()
                        )
                        sampled_eef = self._sample_random_pose(
                            default_eef_pose, rand_range.eef, env_mask
                        )
                        handler.set_home_end_effector_pose(
                            sampled_eef,
                            env_mask=env_mask,
                        )
                else:
                    default_base_pose = self._default_operator_base_poses.get(
                        name, handler.get_base_pose()
                    )
                    sampled_base = self._sample_random_pose(
                        default_base_pose, rand_range, env_mask
                    )
                    handler.set_pose(sampled_base, env_mask=env_mask)

    def _sample_random_pose(
        self, base_pose: PoseState, rand_range: PoseRandomRange, env_mask: np.ndarray
    ) -> PoseState:
        base_pose = base_pose.broadcast_to(self.batch_size)
        position = base_pose.position.copy()
        orientation = base_pose.orientation.copy()
        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            position[env_index, 0] += float(self._rng.uniform(*rand_range.x))
            position[env_index, 1] += float(self._rng.uniform(*rand_range.y))
            position[env_index, 2] += float(self._rng.uniform(*rand_range.z))
            r, p, y = quaternion_to_rpy(orientation[env_index])
            orientation[env_index] = euler_to_quaternion(
                (
                    r + float(self._rng.uniform(*rand_range.roll)),
                    p + float(self._rng.uniform(*rand_range.pitch)),
                    y + float(self._rng.uniform(*rand_range.yaw)),
                )
            )
        return PoseState(position=position, orientation=orientation)

    def get_element_pose(self, name: str, env_index: int = 0) -> PoseState:
        single_env = self.env.envs[env_index]
        model, data = single_env.model, single_env.data
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            pos, quat = single_env.get_site_pose(name)
            return PoseState(position=pos, orientation=quat)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            pos, quat = single_env.get_body_pose(name)
            return PoseState(position=pos, orientation=quat)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            joint_bid = model.jnt_bodyid[jid]
            parent_bid = model.body_parentid[joint_bid]
            parent_pos = data.xpos[parent_bid]
            parent_rot = data.xmat[parent_bid].reshape(3, 3)
            body_local = model.body_pos[joint_bid]
            anchor_in_parent = body_local + model.jnt_pos[jid]
            world_pos = parent_pos + parent_rot @ anchor_in_parent
            parent_quat_wxyz = data.xquat[parent_bid]
            return PoseState(
                position=world_pos,
                orientation=(
                    float(parent_quat_wxyz[1]),
                    float(parent_quat_wxyz[2]),
                    float(parent_quat_wxyz[3]),
                    float(parent_quat_wxyz[0]),
                ),
            )
        raise KeyError(
            f"No site, body, or joint named '{name}' found in the MuJoCo model."
        )

    def get_joint_angle(self, name: str, env_index: int = 0) -> float:
        single_env = self.env.envs[env_index]
        jid = mujoco.mj_name2id(single_env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"No joint named '{name}' found in the MuJoCo model.")
        qadr = single_env.model.jnt_qposadr[jid]
        return float(single_env.data.qpos[qadr])

    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        operator = self.get_operator_handler(operator_name)
        target = self.get_object_handler(object_name)
        if target is None:
            return np.zeros(self.batch_size, dtype=bool)
        result = np.zeros(self.batch_size, dtype=bool)
        for env_index in range(self.batch_size):
            result[env_index] = operator._is_target_grasped(env_index, target)
        return result

    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        result = np.zeros(self.batch_size, dtype=bool)
        for object_name in self.object_handlers:
            result |= self.is_object_grasped(operator_name, object_name)
        return result

    def is_operator_contacting(
        self, operator_name: str, object_name: str
    ) -> np.ndarray:
        operator = self.get_operator_handler(operator_name)
        target = self.get_object_handler(object_name)
        if target is None:
            return np.zeros(self.batch_size, dtype=bool)
        result = np.zeros(self.batch_size, dtype=bool)
        for env_index, single_env in enumerate(self.env.envs):
            target_body_id = mujoco.mj_name2id(
                single_env.model, mujoco.mjtObj.mjOBJ_BODY, target.body_name
            )
            if target_body_id < 0:
                continue
            root_body_id = mujoco.mj_name2id(
                single_env.model, mujoco.mjtObj.mjOBJ_BODY, operator.root_body_name
            )
            operator_body_ids: set[int] = set()
            for bid in range(single_env.model.nbody):
                parent = int(single_env.model.body_parentid[bid])
                if bid == root_body_id or (parent in operator_body_ids and bid != 0):
                    operator_body_ids.add(bid)
            for idx in range(single_env.data.ncon):
                contact = single_env.data.contact[idx]
                geom1 = int(contact.geom1)
                geom2 = int(contact.geom2)
                body1 = int(single_env.model.geom_bodyid[geom1])
                body2 = int(single_env.model.geom_bodyid[geom2])
                if target_body_id in {body1, body2}:
                    other_body = body2 if body1 == target_body_id else body1
                    if other_body in operator_body_ids:
                        result[env_index] = True
                        break
        return result

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        for env_index, single_env in enumerate(self.env.envs):
            object_name = (
                object_names[env_index] if env_index < len(object_names) else ""
            )
            operation_name = (
                operation_names[env_index] if env_index < len(operation_names) else ""
            )
            if object_name and operation_name:
                single_env.set_interest_objects_and_operations(
                    [object_name], [operation_name]
                )
            else:
                single_env.set_interest_objects_and_operations([], [])

    def _normalize_mask(self, env_mask: Optional[np.ndarray]) -> np.ndarray:
        if env_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != self.batch_size:
            raise ValueError(
                f"env_mask must have shape ({self.batch_size},), got {mask.shape}"
            )
        return mask


def create_mujoco_env(
    env_name: str,
    config: EnvConfig,
) -> BatchedUnifiedMujocoEnv:
    return BatchedUnifiedMujocoEnv(config.model_copy(update={"name": env_name}))


def _resolve_arm_pose(arm_config, fallback_pose: PoseState) -> PoseState:
    pose = fallback_pose
    if isinstance(arm_config, list):
        if len(arm_config) >= 6:
            pose = PoseState(
                position=tuple(float(v) for v in arm_config[:3]),
                orientation=euler_to_quaternion(
                    (arm_config[5], arm_config[4], arm_config[3])
                ),
            )
    else:
        if arm_config.position is not None and len(arm_config.position) >= 3:
            pose = PoseState(
                position=tuple(float(v) for v in arm_config.position[:3]),
                orientation=pose.orientation,
            )
        if arm_config.orientation is not None:
            ori = arm_config.orientation
            if len(ori) == 3:
                quat_xyzw = euler_to_quaternion((ori[2], ori[1], ori[0]))
            elif len(ori) == 4:
                quat_xyzw = np.array(ori, dtype=np.float64)
            else:
                raise ValueError(
                    f"orientation must be 3 floats (Euler) or 4 floats (quaternion), got {len(ori)}"
                )
            pose = PoseState(position=pose.position, orientation=quat_xyzw)
    return pose


def build_mujoco_backend(
    task: AutoAtomConfig | Dict[str, Any],
    operators: List[OperatorConfig] | List[Dict[str, Any]],
    ik_solver: Optional[IKSolver] = None,
    handler_kwargs: Optional[Dict[str, Any]] = None,
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
    if not isinstance(env, BatchedUnifiedMujocoEnv):
        raise TypeError(
            f"Registered environment '{config.env_name}' must be a BatchedUnifiedMujocoEnv, got {type(env).__name__}."
        )

    extra = handler_kwargs or {}
    operator_handlers: Dict[str, MujocoOperatorHandler] = {}
    for operator in operator_configs:
        op_extra = operator.model_extra or {}
        ik_extra = (
            op_extra.get("ik", {}) if isinstance(op_extra.get("ik"), dict) else {}
        )
        control_extra = (
            op_extra.get("control", {})
            if isinstance(op_extra.get("control"), dict)
            else {}
        )
        control_cfg = MujocoControlConfig.model_validate(control_extra)
        operator_handlers[operator.name] = MujocoOperatorHandler(
            operator_name=operator.name,
            env=env,
            control=control_cfg,
            ik_solver=ik_solver,
            joint_control_mode=str(
                ik_extra.get(
                    "joint_control_mode",
                    extra.get("joint_control_mode", "per_step_ik"),
                )
            ),
            joint_interp_speed=float(
                ik_extra.get(
                    "joint_interp_speed",
                    extra.get("joint_interp_speed", 0.1),
                )
            ),
            **{
                k: v
                for k, v in extra.items()
                if k not in {"joint_control_mode", "joint_interp_speed"}
            },
        )

    for operator in operator_configs:
        if operator.initial_state is None:
            continue
        handler = operator_handlers[operator.name]
        if operator.initial_state.base_pose is not None:
            bp = operator.initial_state.base_pose
            base_ps = _resolve_arm_pose(bp, handler.get_base_pose().select(0))
            handler.env.override_operator_base_pose(
                operator.name,
                base_ps.broadcast_to(env.batch_size).position,
                base_ps.broadcast_to(env.batch_size).orientation,
            )

        if operator.initial_state.arm is not None:
            arm_config = operator.initial_state.arm
            pose = _resolve_arm_pose(
                arm_config, handler.get_end_effector_pose().select(0)
            )
            if (
                isinstance(arm_config, ArmPoseConfig)
                and arm_config.reference == PoseReference.BASE
            ):
                pos_w, quat_w = handler.env.base_to_world(
                    operator.name,
                    np.asarray(pose.position, dtype=np.float32),
                    np.asarray(pose.orientation, dtype=np.float32),
                )
                pose = PoseState(position=pos_w, orientation=quat_w)
            handler.set_home_end_effector_pose(pose)

        if operator.initial_state.eef is not None:
            handler._home_ctrl[:, handler.eef_ctrl_index] = operator.initial_state.eef

    object_names = {stage.object for stage in config.stages if stage.object}
    object_handlers: Dict[str, MujocoObjectHandler] = {}
    model = env.envs[0].model
    for object_name in object_names:
        freejoint_name: Optional[str] = None
        if (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint")
            >= 0
        ):
            freejoint_name = f"{object_name}_joint"
        elif (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{object_name}_joint0")
            >= 0
        ):
            freejoint_name = f"{object_name}_joint0"
        object_handlers[object_name] = MujocoObjectHandler(
            name=object_name,
            env=env,
            body_name=object_name,
            freejoint_name=freejoint_name,
        )

    return MujocoTaskBackend(
        env=env,
        operator_handlers=operator_handlers,
        object_handlers=object_handlers,
        randomization=dict(config.randomization),
        random_seed=config.seed if config.seed != 0 else None,
        randomization_debug=config.randomization_debug,
    )
