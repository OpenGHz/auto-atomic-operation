"""Mujoco backend adapting the generic task runner to batched basis envs."""

from __future__ import annotations

import logging
import mujoco
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel
from ...basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv, EnvConfig
from ...framework import (
    ArmPoseConfig,
    AutoAtomConfig,
    EefControlConfig,
    InitialPoseConfig,
    OperatorRandomizationConfig,
    OperatorConfig,
    PlacedToleranceConfig,
    PoseControlConfig,
    PoseRandomRange,
    PoseReference,
    RandomizationReference,
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
    compose_pose,
    euler_to_quaternion,
    inverse_pose,
    orientation_within_tolerance_nullable,
    position_within_tolerance,
    position_within_tolerance_nullable,
    quaternion_angular_distance,
    quaternion_to_rpy,
    quaternion_to_rotation_matrix,
)
from ...utils.transformations import quaternion_slerp


class MujocoToleranceConfig(BaseModel):
    position: Union[float, List[float]] = 0.01
    """Position tolerance. A scalar applies as an L2-norm threshold;
    a 3-element list ``[x, y, z]`` checks each axis independently."""
    orientation: float = 0.08
    eef: float = 0.03
    placed: Optional[PlacedToleranceConfig] = None
    """Operator-level tolerance for the PLACED post-condition.

    When ``None``, placement tolerance falls back to the stage-level
    ``placed_tolerance`` only; if neither level is configured, the placement
    check degrades to released-only.
    """


class MujocoGraspConfig(BaseModel):
    lateral_threshold: float = 0.0
    grasp_axis: int = 2
    settle_steps: int = 5


class MujocoControlConfig(BaseModel):
    timeout_steps: int = 100
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
    _descendant_body_ids: Optional[Dict[int, frozenset]] = field(
        init=False, repr=False, default=None
    )

    def get_descendant_body_ids(self, model: Any) -> frozenset:
        """Return a frozenset of body IDs that are the target body or its
        descendants. Cached per model (model topology is static)."""
        model_id = id(model)
        if (
            self._descendant_body_ids is not None
            and model_id in self._descendant_body_ids
        ):
            return self._descendant_body_ids[model_id]
        target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.body_name)
        ids: set = set()
        if target_bid >= 0:
            ids.add(target_bid)
            for bid in range(model.nbody):
                parent = int(model.body_parentid[bid])
                if parent in ids and bid != 0:
                    ids.add(bid)
        result = frozenset(ids)
        if self._descendant_body_ids is None:
            self._descendant_body_ids = {}
        self._descendant_body_ids[model_id] = result
        return result

    def get_pose(self) -> PoseState:
        pos, quat = self.env.get_body_pose(self.body_name)
        return PoseState(position=pos, orientation=quat)

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        pose = pose.broadcast_to(self.env.batch_size)
        mask = (
            np.ones(self.env.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, single_env in enumerate(self.env.envs):
            if not mask[env_index]:
                continue
            x, y, z = pose.position[env_index]
            qx, qy, qz, qw = pose.orientation[env_index]
            if self.freejoint_name is not None:
                jid = mujoco.mj_name2id(
                    single_env.model, mujoco.mjtObj.mjOBJ_JOINT, self.freejoint_name
                )
                if jid >= 0:
                    qpos_adr = int(single_env.model.jnt_qposadr[jid])
                    dof_adr = int(single_env.model.jnt_dofadr[jid])
                    single_env.data.qpos[qpos_adr : qpos_adr + 7] = [
                        x,
                        y,
                        z,
                        qw,
                        qx,
                        qy,
                        qz,
                    ]
                    single_env.data.qvel[dof_adr : dof_adr + 6] = 0.0
                    mujoco.mj_forward(single_env.model, single_env.data)
                    continue

            bid = mujoco.mj_name2id(
                single_env.model, mujoco.mjtObj.mjOBJ_BODY, self.body_name
            )
            if bid < 0:
                continue
            single_env.model.body_pos[bid] = [x, y, z]
            single_env.model.body_quat[bid] = [qw, qx, qy, qz]
            mujoco.mj_forward(single_env.model, single_env.data)

    def is_at_target(
        self,
        target_pose: PoseState,
        position_tolerance: Union[float, List[Optional[float]], None] = 0.02,
        orientation_tolerance: Union[float, List[Optional[float]], None] = None,
    ) -> np.ndarray:
        """Return a bool per env whether the object is within tolerance of the
        target pose."""
        current = self.get_pose()
        target = target_pose.broadcast_to(self.env.batch_size)
        result = np.zeros(self.env.batch_size, dtype=bool)
        for i in range(self.env.batch_size):
            pos_diff = np.asarray(current.position[i], dtype=np.float64) - np.asarray(
                target.position[i], dtype=np.float64
            )
            pos_ok = position_within_tolerance_nullable(pos_diff, position_tolerance)
            ori_ok = orientation_within_tolerance_nullable(
                current.orientation[i], target.orientation[i], orientation_tolerance
            )
            result[i] = pos_ok and ori_ok
        return result


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
    max_joint_delta: float = 0.35

    _operator_body_ids_cache: Optional[Dict[int, frozenset]] = field(
        init=False, repr=False, default=None
    )
    _left_right_geom_cache: Optional[Dict[int, Dict[int, str]]] = field(
        init=False, repr=False, default=None
    )
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

    def get_operator_body_ids(self, model: Any) -> frozenset:
        """Return all body IDs belonging to this operator (root + descendants).
        Cached per model (topology is static)."""
        model_id = id(model)
        if (
            self._operator_body_ids_cache is not None
            and model_id in self._operator_body_ids_cache
        ):
            return self._operator_body_ids_cache[model_id]
        root_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.root_body_name
        )
        ids: set = set()
        if root_bid >= 0:
            ids.add(root_bid)
            for bid in range(model.nbody):
                parent = int(model.body_parentid[bid])
                if parent in ids and bid != 0:
                    ids.add(bid)
        result = frozenset(ids)
        if self._operator_body_ids_cache is None:
            self._operator_body_ids_cache = {}
        self._operator_body_ids_cache[model_id] = result
        return result

    def get_left_right_geom_ids(self, model: Any) -> Dict[int, str]:
        """Return a dict mapping geom_id → 'left' or 'right' for gripper
        finger geoms. Cached per model."""
        model_id = id(model)
        if (
            self._left_right_geom_cache is not None
            and model_id in self._left_right_geom_cache
        ):
            return self._left_right_geom_cache[model_id]
        mapping: Dict[int, str] = {}
        operator_bodies = self.get_operator_body_ids(model)
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) not in operator_bodies:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith("left_"):
                mapping[gid] = "left"
            elif name.startswith("right_"):
                mapping[gid] = "right"
        if self._left_right_geom_cache is None:
            self._left_right_geom_cache = {}
        self._left_right_geom_cache[model_id] = mapping
        return mapping

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
            max_joint_delta=self.max_joint_delta,
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
            pos_diff_after = eef_world_after.position[env_index] - pos_goal
            pos_err_after = float(np.linalg.norm(pos_diff_after))
            ori_err_after = quaternion_angular_distance(
                eef_world_after.orientation[env_index], ori_goal
            )
            # Resolve effective tolerance: per-waypoint override > operator default
            wp_tol = pose.tolerance
            eff_pos_tol = (
                wp_tol.position
                if wp_tol is not None and wp_tol.position is not None
                else self.control.tolerance.position
            )
            eff_ori_tol = (
                wp_tol.orientation
                if wp_tol is not None and wp_tol.orientation is not None
                else self.control.tolerance.orientation
            )
            pos_ok = position_within_tolerance(pos_diff_after, eff_pos_tol)
            ori_ok = ori_err_after <= eff_ori_tol
            event = "pose_reached" if pos_ok and ori_ok else "moving"
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
            if pos_ok and ori_ok:
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
        # Apply the desired eef ctrl value from _home_ctrl.  home_operator()
        # restores from the env-level home_ctrl snapshot (captured at
        # registration time), which does not reflect initial_state.eef
        # changes.  We set ctrl here and step the simulation so the gripper
        # linkage settles physically rather than jumping qpos directly.
        mask = self._normalize_mask(env_mask)
        needs_settle = False
        for env_index, enabled in enumerate(mask):
            if not enabled:
                continue
            single_env = self.env.envs[env_index]
            current = float(single_env.data.ctrl[self.eef_ctrl_index])
            target = float(self._home_ctrl[env_index, self.eef_ctrl_index])
            if abs(current - target) > 1e-6:
                single_env.data.ctrl[self.eef_ctrl_index] = target
                needs_settle = True
        if needs_settle:
            for env_index, enabled in enumerate(mask):
                if enabled:
                    se = self.env.envs[env_index]
                    for _ in range(200):
                        mujoco.mj_step(se.model, se.data)
                    # Clear residual velocities and reset time so the
                    # settle phase is invisible to the rest of the sim.
                    se.data.qvel[:] = 0.0
                    se.data.time = 0.0
                    mujoco.mj_forward(se.model, se.data)

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
        model = single_env.model
        target_bodies = target.get_descendant_body_ids(model)
        if not target_bodies:
            return {
                "left_contact": False,
                "right_contact": False,
                "lateral_ok": False,
                "lateral_error": float("inf"),
                "lateral_threshold": 0.03,
            }

        left_right_geoms = self.get_left_right_geom_ids(model)
        left_contact = False
        right_contact = False
        geom_bodyid = model.geom_bodyid
        data = single_env.data
        for idx in range(data.ncon):
            contact = data.contact[idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(geom_bodyid[geom1])
            body2 = int(geom_bodyid[geom2])
            b1_match = body1 in target_bodies
            b2_match = body2 in target_bodies
            if not b1_match and not b2_match:
                continue
            other_geom = geom2 if b1_match else geom1
            side = left_right_geoms.get(other_geom)
            if side == "left":
                left_contact = True
            elif side == "right":
                right_contact = True
            if left_contact and right_contact:
                break

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
    initial_poses: Dict[str, InitialPoseConfig] = field(default_factory=dict)
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

    @property
    def dt_per_update(self) -> float:
        e = self.env.envs[0]
        return e.model.opt.timestep * e._n_substeps

    def setup(self, config: AutoAtomConfig) -> None:
        for operator in self.operator_handlers.values():
            operator.home()
        if self.initial_poses:
            self._apply_initial_poses()
        self._record_default_poses()

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        mask = self._normalize_mask(env_mask)
        self.env.reset(mask)
        for operator in self.operator_handlers.values():
            operator.home(mask)
        if self.initial_poses:
            self._apply_initial_poses(mask)
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

    def _apply_initial_poses(self, env_mask: np.ndarray | None = None) -> None:
        """Apply per-object initial pose overrides from config.

        Called after keyframe reset and operator homing, before default-pose
        recording and randomization.  Only the specified components (position
        and/or orientation) are overridden; the rest keep their keyframe value.

        After setting each object's pose the recorded default is updated so
        that subsequent randomization uses the new pose as its baseline.  This
        allows callers to mutate ``self.initial_poses`` between resets for
        per-episode initial conditions.
        """
        for name, cfg in self.initial_poses.items():
            handler = self.object_handlers.get(name)
            if handler is None:
                continue
            current = handler.get_pose()
            pos = current.position
            ori = current.orientation
            if cfg.position is not None and len(cfg.position) >= 3:
                pos = np.array(cfg.position[:3], dtype=np.float64)
            if cfg.orientation is not None:
                if len(cfg.orientation) == 3:
                    ori = euler_to_quaternion(
                        (cfg.orientation[2], cfg.orientation[1], cfg.orientation[0])
                    )
                elif len(cfg.orientation) == 4:
                    ori = np.array(cfg.orientation, dtype=np.float64)
                else:
                    raise ValueError(
                        f"initial_pose[{name!r}].orientation must be 3 floats "
                        f"(Euler) or 4 floats (quaternion), got {len(cfg.orientation)}"
                    )
            handler.set_pose(
                PoseState(position=pos, orientation=ori), env_mask=env_mask
            )
            # Keep recorded defaults in sync so randomization offsets from
            # the (possibly dynamic) initial pose, not the stale keyframe.
            self._default_object_poses[name] = handler.get_pose()

    # ------------------------------------------------------------------
    #  Randomization: ordering, reference resolution, and application
    # ------------------------------------------------------------------

    def _randomization_order(self) -> List[str]:
        """Return randomization keys in dependency order (referenced first).

        An entry whose ``reference`` is another entity name depends on that
        entity being sampled first. Cycles raise ``ValueError``.
        """
        deps: Dict[str, Set[str]] = {name: set() for name in self.randomization}
        for name, rand in self.randomization.items():
            sub_refs: list = []
            if isinstance(rand, OperatorRandomizationConfig):
                if rand.base is not None:
                    sub_refs.append(rand.base.reference)
                if rand.eef is not None:
                    sub_refs.append(rand.eef.reference)
            else:
                sub_refs.append(rand.reference)
            for ref in sub_refs:
                if isinstance(ref, RandomizationReference):
                    continue
                if ref in self.randomization:
                    deps[name].add(ref)
        order: List[str] = []
        visited: Set[str] = set()
        visiting: Set[str] = set()

        def _visit(n: str) -> None:
            if n in visited:
                return
            if n in visiting:
                raise ValueError(f"Circular randomization reference involving '{n}'")
            visiting.add(n)
            for dep in deps[n]:
                _visit(dep)
            visiting.remove(n)
            visited.add(n)
            order.append(n)

        for name in self.randomization:
            _visit(name)
        return order

    def _resolve_reference_base_pose(
        self,
        reference: Union[RandomizationReference, str],
        sampled_poses: Dict[str, PoseState],
        default_pose: PoseState,
    ) -> PoseState:
        """Resolve the base pose to feed ``_sample_random_pose``.

        For enum modes the entity's own ``default_pose`` is returned.

        For an entity-name reference, the delta-carry algorithm is applied:
        ``delta = ref_sampled * ref_default⁻¹``, then ``delta * default_pose``
        so the current entity moves with the referenced entity while
        preserving their original spatial relationship.
        """
        if isinstance(reference, RandomizationReference):
            return default_pose
        # --- Entity-name reference: delta-carry ---
        ref_sampled = sampled_poses.get(reference)
        # Resolve the reference entity's default pose.
        if reference in self._default_object_poses:
            ref_default = self._default_object_poses[reference]
        elif reference in self._default_operator_eef_poses:
            ref_default = self._default_operator_eef_poses[reference]
        elif reference in self.object_handlers:
            ref_default = self.object_handlers[reference].get_pose()
        elif reference in self.operator_handlers:
            ref_default = self.operator_handlers[reference].get_end_effector_pose()
        else:
            raise ValueError(
                f"Randomization reference '{reference}' is not a known mode "
                "('relative', 'absolute_world', 'absolute_base') nor an "
                "existing object/operator name."
            )
        if ref_sampled is None:
            return default_pose  # entity not randomized → no delta
        delta = compose_pose(ref_sampled, inverse_pose(ref_default))
        return compose_pose(delta, default_pose)

    def _apply_randomization(self, env_mask: np.ndarray) -> None:
        sampled_poses: Dict[str, PoseState] = {}
        for name in self._randomization_order():
            rand_range = self.randomization[name]
            if name in self.object_handlers:
                if isinstance(rand_range, OperatorRandomizationConfig):
                    raise TypeError(
                        f"Object '{name}' randomization must be a "
                        "PoseRandomRange, not an operator randomization "
                        "config."
                    )
                if rand_range.reference == RandomizationReference.ABSOLUTE_BASE:
                    raise ValueError(
                        f"Object '{name}' randomization cannot use "
                        "'absolute_base' — only operator end-effector "
                        "randomization is defined in a base frame."
                    )
                default_pose = self._default_object_poses.get(
                    name, self.object_handlers[name].get_pose()
                )
                base_pose = self._resolve_reference_base_pose(
                    rand_range.reference, sampled_poses, default_pose
                )
                sampled = self._sample_random_pose(base_pose, rand_range, env_mask)
                self.object_handlers[name].set_pose(sampled, env_mask=env_mask)
                sampled_poses[name] = sampled
            elif name in self.operator_handlers:
                handler = self.operator_handlers[name]
                if isinstance(rand_range, OperatorRandomizationConfig):
                    if rand_range.base is not None:
                        if (
                            rand_range.base.reference
                            == RandomizationReference.ABSOLUTE_BASE
                        ):
                            raise ValueError(
                                f"Operator '{name}' base randomization "
                                "cannot use 'absolute_base' — the base IS "
                                "the frame."
                            )
                        default_base = self._default_operator_base_poses.get(
                            name, handler.get_base_pose()
                        )
                        base_pose = self._resolve_reference_base_pose(
                            rand_range.base.reference,
                            sampled_poses,
                            default_base,
                        )
                        sampled_base = self._sample_random_pose(
                            base_pose, rand_range.base, env_mask
                        )
                        handler.set_pose(sampled_base, env_mask=env_mask)
                        sampled_poses[name] = sampled_base
                    if rand_range.eef is not None:
                        sampled_eef = self._sample_operator_eef_pose(
                            handler,
                            name,
                            rand_range.eef,
                            env_mask,
                            sampled_poses,
                        )
                        handler.set_home_end_effector_pose(
                            sampled_eef,
                            env_mask=env_mask,
                        )
                        sampled_poses[name] = sampled_eef
                else:
                    sampled_eef = self._sample_operator_eef_pose(
                        handler,
                        name,
                        rand_range,
                        env_mask,
                        sampled_poses,
                    )
                    handler.set_home_end_effector_pose(sampled_eef, env_mask=env_mask)
                    sampled_poses[name] = sampled_eef
            else:
                logging.getLogger(MujocoTaskBackend.__name__).warning(
                    "Randomization key '%s' does not match any object or "
                    "operator handler — skipping.",
                    name,
                )

    def _sample_operator_eef_pose(
        self,
        handler: MujocoOperatorHandler,
        name: str,
        rand_range: PoseRandomRange,
        env_mask: np.ndarray,
        sampled_poses: Dict[str, PoseState],
    ) -> PoseState:
        """Sample a new home EEF world pose.

        Handles entity-name references (delta-carry) and ``absolute_base``
        mode (sample in operator base frame, transform back to world).
        """
        default_eef_world = self._default_operator_eef_poses.get(
            name, handler.get_end_effector_pose()
        )
        base_pose_for_sampler = self._resolve_reference_base_pose(
            rand_range.reference, sampled_poses, default_eef_world
        )
        if rand_range.reference != RandomizationReference.ABSOLUTE_BASE:
            return self._sample_random_pose(base_pose_for_sampler, rand_range, env_mask)
        base_world = handler.get_base_pose()
        default_in_base = compose_pose(inverse_pose(base_world), base_pose_for_sampler)
        sampled_in_base = self._sample_random_pose(
            default_in_base, rand_range, env_mask
        )
        return compose_pose(base_world, sampled_in_base)

    def _sample_random_pose(
        self, base_pose: PoseState, rand_range: PoseRandomRange, env_mask: np.ndarray
    ) -> PoseState:
        base_pose = base_pose.broadcast_to(self.batch_size)
        position = base_pose.position.copy()
        orientation = base_pose.orientation.copy()
        is_absolute = rand_range.reference in (
            RandomizationReference.ABSOLUTE_WORLD,
            RandomizationReference.ABSOLUTE_BASE,
        )
        pos_ranges = (rand_range.x, rand_range.y, rand_range.z)
        rot_ranges = (rand_range.roll, rand_range.pitch, rand_range.yaw)
        rot_any = any(r is not None for r in rot_ranges)
        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            for axis, rng_pair in enumerate(pos_ranges):
                if rng_pair is None:
                    continue
                sampled = float(self._rng.uniform(*rng_pair))
                if is_absolute:
                    position[env_index, axis] = sampled
                else:
                    position[env_index, axis] += sampled
            if rot_any:
                r0, p0, y0 = quaternion_to_rpy(orientation[env_index])
                if is_absolute:
                    r_val = (
                        r0
                        if rot_ranges[0] is None
                        else float(self._rng.uniform(*rot_ranges[0]))
                    )
                    p_val = (
                        p0
                        if rot_ranges[1] is None
                        else float(self._rng.uniform(*rot_ranges[1]))
                    )
                    y_val = (
                        y0
                        if rot_ranges[2] is None
                        else float(self._rng.uniform(*rot_ranges[2]))
                    )
                else:
                    r_val = r0 + (
                        0.0
                        if rot_ranges[0] is None
                        else float(self._rng.uniform(*rot_ranges[0]))
                    )
                    p_val = p0 + (
                        0.0
                        if rot_ranges[1] is None
                        else float(self._rng.uniform(*rot_ranges[1]))
                    )
                    y_val = y0 + (
                        0.0
                        if rot_ranges[2] is None
                        else float(self._rng.uniform(*rot_ranges[2]))
                    )
                orientation[env_index] = euler_to_quaternion((r_val, p_val, y_val))
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
            model = single_env.model
            target_bodies = target.get_descendant_body_ids(model)
            if not target_bodies:
                continue
            operator_bodies = operator.get_operator_body_ids(model)
            geom_bodyid = model.geom_bodyid
            data = single_env.data
            for idx in range(data.ncon):
                contact = data.contact[idx]
                body1 = int(geom_bodyid[int(contact.geom1)])
                body2 = int(geom_bodyid[int(contact.geom2)])
                b1_target = body1 in target_bodies
                b2_target = body2 in target_bodies
                if not b1_target and not b2_target:
                    continue
                other_body = body2 if b1_target else body1
                if other_body in operator_bodies:
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
            max_joint_delta=float(
                op_extra.get(
                    "max_joint_delta",
                    ik_extra.get(
                        "max_joint_delta",
                        extra.get("max_joint_delta", 0.35),
                    ),
                )
            ),
            **{
                k: v
                for k, v in extra.items()
                if k
                not in {
                    "joint_control_mode",
                    "joint_interp_speed",
                    "max_joint_delta",
                }
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
    # Also register objects mentioned in the randomization dict (they may not
    # appear in any stage but still need handlers for pose get/set).
    model = env.envs[0].model

    def _body_exists(name: str) -> bool:
        """Check if a body (or its _gs variant) exists in the MuJoCo model."""
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return True
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_gs") >= 0:
            return True
        return False

    _rand_candidate_names: set = set()
    for rand_name in config.randomization:
        if rand_name not in operator_handlers:
            _rand_candidate_names.add(rand_name)
    for rand_range in config.randomization.values():
        refs: list = []
        if isinstance(rand_range, OperatorRandomizationConfig):
            if rand_range.base is not None:
                refs.append(rand_range.base.reference)
            if rand_range.eef is not None:
                refs.append(rand_range.eef.reference)
        else:
            refs.append(rand_range.reference)
        for ref in refs:
            if isinstance(ref, str) and not isinstance(ref, RandomizationReference):
                if ref not in operator_handlers:
                    _rand_candidate_names.add(ref)
    # Also register objects mentioned in initial_pose.
    for ip_name in config.initial_pose:
        if ip_name not in operator_handlers:
            _rand_candidate_names.add(ip_name)
    for cand in _rand_candidate_names:
        if _body_exists(cand):
            object_names.add(cand)

    object_handlers: Dict[str, MujocoObjectHandler] = {}
    for object_name in object_names:
        body_name = object_name
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{object_name}_gs") >= 0:
            body_name = f"{object_name}_gs"
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
            body_name=body_name,
            freejoint_name=freejoint_name,
        )

    return MujocoTaskBackend(
        env=env,
        operator_handlers=operator_handlers,
        object_handlers=object_handlers,
        randomization=dict(config.randomization),
        initial_poses=dict(config.initial_pose),
        random_seed=config.seed if config.seed != 0 else None,
        randomization_debug=config.randomization_debug,
    )
