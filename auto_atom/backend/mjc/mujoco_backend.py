"""Mujoco backend adapting the generic task runner to batched basis envs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import mujoco
import numpy as np
from pydantic import BaseModel

from ...basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv, EnvConfig
from ...framework import (
    AutoAtomConfig,
    EefControlConfig,
    OperatorConfig,
    OperatorInitialState,
    OperatorRandomizationConfig,
    PlacedToleranceConfig,
    PoseControlConfig,
    PoseOverrideConfig,
    PoseRandomizationSpec,
    PoseRandomRange,
    PoseReference,
    RandomizationConstraintConfig,
    RandomizationGeneratorConfig,
    RandomizationGeneratorKind,
    RandomizationGroupConfig,
    RandomizationInput,
    RandomizationPoissonDiskConfig,
    RandomizationReference,
    RandomizationSelectorKind,
    RandomizationSpec,
    RandomizationStrategy,
    canonical_randomization_spec,
    pose_randomization_regions,
)
from ...randomization import (
    PoissonDiskCandidateStream,
    RandomizationFailureError,
    RandomizationPlan,
    compile_randomization_plan,
    maximin_select,
    unit_candidate,
)
from ...runtime import (
    CameraModel,
    ComponentRegistry,
    ContactObservation,
    ControlResult,
    ControlSignal,
    IKSolver,
    ObjectHandler,
    OperatorHandler,
    RandomizationConstraintReport,
    SceneBackend,
    SupportGeometry,
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
    quaternion_from_matrix_3x3,
    quaternion_to_rotation_matrix,
    quaternion_to_rpy,
    resolve_pose_override,
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
    release_settle_steps: int = 0
    """Control updates to wait after opening before the arm retreats."""


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
    ik_unreachable_threshold: int = 30
    """Number of consecutive IK failures after which ``move_to_pose`` declares the
    waypoint unreachable and fails the stage immediately, rather than waiting for
    ``timeout_steps`` to elapse with the arm frozen. At ``update_freq=50`` the
    default ≈0.6 s of trying-and-failing is plenty to rule out a transient miss."""


_MAX_COLLISION_REJECTION_ATTEMPTS = 100
_RandomizationAncestors = Set[str] | List[Set[str]]


def _mujoco_element_name(
    model: Any,
    object_type: Any,
    element_id: int,
    fallback_prefix: str,
) -> str:
    name = mujoco.mj_id2name(model, object_type, element_id)
    return name if name is not None else f"{fallback_prefix}#{element_id}"


def _randomization_references(
    spec: PoseRandomizationSpec,
) -> tuple[Union[RandomizationReference, str], ...]:
    """Return every reference declared by a randomization spec."""
    references: list[Union[RandomizationReference, str]] = []
    for region in pose_randomization_regions(spec):
        references.extend(region.references())
    return tuple(dict.fromkeys(references))


def _copy_randomization_ancestors(
    ancestors: _RandomizationAncestors,
) -> _RandomizationAncestors:
    """Copy scalar or per-environment reference-ancestor sets."""
    if isinstance(ancestors, list):
        return [set(values) for values in ancestors]
    return set(ancestors)


def _stateful_pose_indices(
    env: Any,
    pose: PoseState,
    env_mask: Optional[np.ndarray],
    *,
    label: str,
) -> tuple[int, ...]:
    """Return physical rows for a pose mutation.

    Replicated batches have one physical row per logical environment.  GS
    shared-physics batches expose aliases of one physical row, so a stateful
    pose must be identical for every logical row participating in the update;
    otherwise a loop over aliases would silently leave the last row's value in
    the model while callers still observe a different logical batch.  The
    shared-batch contract uses logical row 0 as the canonical state-changing
    value (the same rule as :class:`BatchExecutionAdapter`), even when a
    partial mask selects another row.  The helper validates that contract and
    returns one physical representative row.
    """
    batch_size = int(env.batch_size)
    mask = (
        np.ones(batch_size, dtype=bool)
        if env_mask is None
        else np.asarray(env_mask, dtype=bool).reshape(-1)
    )
    if mask.shape != (batch_size,):
        raise ValueError(f"env_mask must have shape ({batch_size},), got {mask.shape}")
    active = np.flatnonzero(mask)
    if active.size == 0:
        return ()
    if not bool(getattr(env, "_share_physics", False)):
        return tuple(int(index) for index in active)

    representative = 0
    ref_position = np.asarray(pose.position[representative], dtype=np.float64)
    ref_orientation = np.asarray(pose.orientation[representative], dtype=np.float64)
    ref_norm = float(np.linalg.norm(ref_orientation))
    # ``representative`` is fixed at logical row 0 to match the shared
    # BatchExecutionAdapter contract.  Do not use ``active[1:]`` here: a
    # partial mask such as ``[False, True, True]`` would otherwise skip the
    # first active row and fail to validate it against the canonical value.
    for index_raw in active:
        index = int(index_raw)
        if index == representative:
            continue
        if not np.allclose(pose.position[index], ref_position, atol=1e-7, rtol=1e-7):
            raise ValueError(
                f"{label} differs across logical rows backed by shared physics; "
                "use one shared pose or disable gaussian_render.share_physics."
            )
        orientation = np.asarray(pose.orientation[index], dtype=np.float64)
        orientation_norm = float(np.linalg.norm(orientation))
        dot = (
            abs(float(np.dot(orientation, ref_orientation)))
            / (orientation_norm * ref_norm)
            if orientation_norm > 0.0 and ref_norm > 0.0
            else 0.0
        )
        if not np.isclose(dot, 1.0, atol=1e-7, rtol=1e-7):
            raise ValueError(
                f"{label} differs across logical rows backed by shared physics; "
                "use one shared pose or disable gaussian_render.share_physics."
            )
    return (representative,)


@dataclass
class _CollisionParticipant:
    owner: str
    label: str
    pose: PoseState
    radius: float | np.ndarray
    ancestors: _RandomizationAncestors = field(default_factory=set)


@dataclass
class _PendingRandomizationAction:
    kind: str
    owner: str
    label: str
    pose: PoseState
    radius: float | np.ndarray
    references: tuple[Union[RandomizationReference, str], ...] = ()
    ancestors: _RandomizationAncestors = field(default_factory=set)
    constraints: Optional[RandomizationConstraintConfig] = None


@dataclass(frozen=True)
class _RandomizationActionSpec:
    kind: str
    owner: str
    label: str
    randomization: RandomizationSpec


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
        env_indices = _stateful_pose_indices(
            self.env,
            pose,
            env_mask,
            label=f"Object '{self.name}' pose",
        )
        for env_index in env_indices:
            single_env = self.env.envs[env_index]
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
            # ``body_pos``/``body_quat`` are stored in the parent body's local
            # frame.  Static (non-freejoint) assets are often nested below a
            # scene body, so writing the requested world pose directly here
            # silently places them at the wrong location.  Convert world →
            # parent-local just as the operator base-pose path does.
            parent_id = int(single_env.model.body_parentid[bid])
            parent_pos = single_env.data.xpos[parent_id].astype(np.float64)
            parent_mat = (
                single_env.data.xmat[parent_id].reshape(3, 3).astype(np.float64)
            )
            single_env.model.body_pos[bid] = parent_mat.T @ (
                np.asarray([x, y, z], dtype=np.float64) - parent_pos
            )
            world_quat_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float64)
            parent_quat_wxyz = single_env.data.xquat[parent_id].astype(np.float64)
            inverse_parent_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_negQuat(inverse_parent_quat, parent_quat_wxyz)
            local_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(local_quat, inverse_parent_quat, world_quat_wxyz)
            single_env.model.body_quat[bid] = local_quat
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
            # Match both unprefixed (``left_finger_pad``) and prefixed
            # (``eef_left_finger_pad_upper``) names, so grasp detection works
            # when a gripper model is attached under an ``<attach prefix=...>``.
            if name.startswith("left_") or "_left_" in name:
                mapping[gid] = "left"
            elif name.startswith("right_") or "_right_" in name:
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
        # In ``solve_once_interpolate`` mode the env solves IK once for the
        # final waypoint pose and then interpolates joint targets, so the
        # handler must hand the unmodified target down — running cartesian
        # step clamping (or adaptive step scaling) here would shift
        # ``target_pos_in_base`` every frame and force the env to replan
        # every step, defeating the single-shot semantics.
        interp_mode = self.joint_control_mode == "solve_once_interpolate"
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

            if interp_mode:
                # Pass the final waypoint pose straight through; the env
                # owns motion shaping via joint-space interpolation.
                pos_goal = desired_pos[env_index].copy()
                ori_goal = desired_ori[env_index].copy()
            else:
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
            step_pos_diff_after = eef_world_after.position[env_index] - pos_goal
            step_pos_err_after = float(np.linalg.norm(step_pos_diff_after))
            step_ori_err_after = quaternion_angular_distance(
                eef_world_after.orientation[env_index], ori_goal
            )
            # Step shaping changes only the command sent on this control tick.
            # Primitive completion must still be measured against the final
            # waypoint; otherwise reaching the first clamped sub-step would
            # incorrectly advance the Stage after only a few millimetres.
            final_pos_diff_after = (
                eef_world_after.position[env_index] - desired_pos[env_index]
            )
            final_pos_err_after = float(np.linalg.norm(final_pos_diff_after))
            final_ori_err_after = quaternion_angular_distance(
                eef_world_after.orientation[env_index], desired_ori[env_index]
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
            pos_ok = position_within_tolerance(final_pos_diff_after, eff_pos_tol)
            ori_ok = final_ori_err_after <= eff_ori_tol
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
                "position_error": final_pos_err_after,
                "orientation_error": final_ori_err_after,
                "command_step_position_error": step_pos_err_after,
                "command_step_orientation_error": step_ori_err_after,
                "steps": int(self._move_steps[env_index]),
            }
            ik_streak = int(
                self.env.envs[env_index].get_operator_ik_failure_streak(
                    self.operator_name
                )
            )
            if pos_ok and ori_ok:
                signals[env_index] = ControlSignal.REACHED
                self._move_steps[env_index] = 0
            elif ik_streak >= int(self.control.ik_unreachable_threshold):
                # Persistent IK failure: don't burn the rest of the stage
                # timeout watching a frozen arm. Fail the stage now with a
                # specific category so the user can tell unreachable targets
                # from genuine motion timeouts.
                details[env_index]["event"] = "ik_unreachable"
                details[env_index]["failure_category"] = "ik_unreachable"
                details[env_index]["failure_reason"] = (
                    f"IK failed for {ik_streak} consecutive control steps; "
                    f"target pose is outside the arm's reachable workspace"
                )
                details[env_index]["ik_failure_streak"] = ik_streak
                signals[env_index] = ControlSignal.FAILED
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
        target: Optional[ObjectHandler],
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
            self._last_target[env_index] = (
                target if isinstance(target, MujocoObjectHandler) else None
            )
            command_key = f"{eef.close}:{eef.joint_positions}:{eef.require_grasp}"
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
            target_grasped = (
                eef.close
                and settle_ready
                and self._last_target[env_index] is not None
                and self._is_target_grasped(env_index, self._last_target[env_index])
            )
            if eef.close and eef.require_grasp and self._last_target[env_index] is None:
                event = "grasp_target_required"
                details[env_index] = {
                    "event": event,
                    "failure_category": "missing_grasp_target",
                    "failure_reason": (
                        "require_grasp=true needs a non-empty Stage target object"
                    ),
                }
                signals[env_index] = ControlSignal.FAILED
                self._eef_steps[env_index] = 0
                continue
            if target_grasped:
                reached = True
                grasped_name = self._last_target[env_index].name
                event = "eef_grasped"
            elif (
                eef.close
                and not eef.require_grasp
                and actual >= (target_value - self.control.tolerance.eef)
            ):
                reached = True
                event = "eef_reached"
            elif (
                eef.close
                and not eef.require_grasp
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
            elif (
                not eef.close
                and self._eef_steps[env_index]
                >= self.control.grasp.release_settle_steps
                and actual <= (self.eef_open_value + self.control.tolerance.eef)
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

    def get_reached_tolerances(self) -> tuple[Any, Any]:
        tolerance = self.control.tolerance
        return tolerance.position, tolerance.orientation

    def get_placed_tolerances(self) -> tuple[Any, Any]:
        placed = self.control.tolerance.placed
        if placed is None:
            return None, None
        return placed.position, placed.orientation

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
        mask = self._normalize_mask(env_mask)
        shared = bool(getattr(self.env, "_share_physics", False))

        # ``home_operator`` writes the registered home controls (and the
        # corresponding EEF qpos) immediately.  Capture the physical control
        # value first so a newly configured ``initial_state.eef`` still gets
        # the same controller-driven linkage settle as the historical path;
        # comparing only after ``home_operator`` would make the values look
        # equal and skip the settle entirely.  Keep one entry per physical
        # replica because shared-physics exposes aliases for logical rows.
        pre_home_ctrl: dict[int, float] = {}
        for env_index, enabled in enumerate(mask):
            if not enabled:
                continue
            physical_index = 0 if shared else env_index
            if physical_index in pre_home_ctrl:
                continue
            pre_home_ctrl[physical_index] = float(
                self.env.envs[physical_index].data.ctrl[self.eef_ctrl_index]
            )

        self.env.home_operator(self.operator_name, env_mask=env_mask)
        # ``_home_ctrl`` is a per-logical-row view used only for the settling
        # pass below.  The authoritative value lives in the env's
        # ``_OperatorState.home_ctrl``; refresh the view after every reset so
        # stale initial-state/randomization values cannot leak across episodes.
        for env_index, enabled in enumerate(mask):
            if not enabled:
                continue
            physical_index = 0 if shared else env_index
            states = getattr(self.env.envs[physical_index], "_operator_states", None)
            if states is None or self.operator_name not in states:
                # Lightweight handler test doubles may not expose the
                # concrete MuJoCo operator-state registry.  Production envs
                # always do; retain the existing cache when the seam is absent.
                continue
            state = states[self.operator_name]
            self._home_ctrl[env_index] = state.home_ctrl

        # Apply the desired eef ctrl value from _home_ctrl.  home_operator()
        # restores the env-level home_ctrl snapshot first; this second pass
        # handles a configured EEF override and lets the linkage settle
        # physically rather than jumping qpos directly.
        needs_settle = False
        for env_index, enabled in enumerate(mask):
            if not enabled:
                continue
            physical_index = 0 if shared else env_index
            single_env = self.env.envs[physical_index]
            current = float(single_env.data.ctrl[self.eef_ctrl_index])
            target = float(self._home_ctrl[env_index, self.eef_ctrl_index])
            # ``current`` is normally already ``target`` because
            # ``home_operator`` restores the control snapshot.  The
            # pre-home comparison is what detects a newly requested EEF
            # control and preserves the physical settle step.
            if (
                abs(current - target) > 1e-6
                or abs(pre_home_ctrl[physical_index] - target) > 1e-6
            ):
                single_env.data.ctrl[self.eef_ctrl_index] = target
                needs_settle = True
        if needs_settle:
            settled_physical: set[int] = set()
            for env_index, enabled in enumerate(mask):
                if enabled:
                    physical_index = 0 if shared else env_index
                    if physical_index in settled_physical:
                        continue
                    settled_physical.add(physical_index)
                    se = self.env.envs[physical_index]
                    for _ in range(200):
                        mujoco.mj_step(se.model, se.data)
                    state = se._operator_states[self.operator_name]
                    if state.joint_mode:
                        if state.home_arm_qpos is not None:
                            se.data.qpos[se._op_arm_qidx[self.operator_name]] = (
                                state.home_arm_qpos
                            )
                    else:
                        q = state.home_mocap_quat
                        se._write_mocap_pose(
                            state,
                            state.home_mocap_pos,
                            np.asarray([q[1], q[2], q[3], q[0]]),
                            sync_freejoint=True,
                        )
                    # Clear residual velocities and reset time so the
                    # settle phase is invisible to the rest of the sim.
                    se.data.qvel[:] = 0.0
                    se.data.time = 0.0
                    mujoco.mj_forward(se.model, se.data)

    def set_home_end_effector_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
        *,
        apply_home: bool = True,
    ) -> None:
        """Set the EEF pose restored by :meth:`home`.

        ``apply_home`` lets a lifecycle coordinator stage several home-state
        fields (base, EEF, and gripper) and perform one final homing pass.  The
        default preserves the immediate-apply behavior used by randomization
        and direct callers.
        """
        pose = pose.broadcast_to(self.env.batch_size)
        _stateful_pose_indices(
            self.env,
            pose,
            env_mask,
            label=f"Operator '{self.operator_name}' home EEF pose",
        )
        self.env.set_operator_home_eef_pose(
            self.operator_name,
            pose.position,
            pose.orientation,
            env_mask=env_mask,
        )
        if apply_home:
            self.home(env_mask)
        mask = self._normalize_mask(env_mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._home_ctrl[env_index, self.eef_ctrl_index] = self.env.envs[
                    env_index
                ].data.ctrl[self.eef_ctrl_index]

    def set_home_joint_positions(
        self,
        joint_positions: Mapping[str, object],
        env_mask: Optional[np.ndarray] = None,
        *,
        apply_home: bool = True,
    ) -> None:
        """Stage raw qpos values for this operator's home joint state."""
        self.env.set_operator_home_joint_positions(
            self.operator_name,
            joint_positions,
            env_mask=env_mask,
        )
        if apply_home:
            self.home(env_mask)

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        self.reset_state(env_mask)
        pose = pose.broadcast_to(self.env.batch_size)
        _stateful_pose_indices(
            self.env,
            pose,
            env_mask,
            label=f"Operator '{self.operator_name}' base pose",
        )
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
    randomization: Dict[
        str,
        RandomizationInput | OperatorRandomizationConfig,
    ] = field(default_factory=dict)
    randomization_strategy: RandomizationStrategy = RandomizationStrategy.RSA
    randomization_groups: Dict[str, RandomizationGroupConfig] = field(
        default_factory=dict
    )
    camera_randomization: Dict[str, RandomizationInput] = field(default_factory=dict)
    initial_poses: Dict[str, PoseOverrideConfig] = field(default_factory=dict)
    camera_initial_poses: Dict[str, PoseOverrideConfig] = field(default_factory=dict)
    operator_initial_states: Dict[str, OperatorInitialState] = field(
        default_factory=dict
    )
    random_seed: Optional[int] = None
    randomization_debug: bool = False
    _rng: np.random.Generator = field(init=False, repr=False)
    _randomization_reset_index: int = field(init=False, repr=False, default=0)
    _last_randomization_diagnostics: Dict[int, List[Dict[str, Any]]] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _space_filling_history: Dict[Tuple[str, ...], List[np.ndarray]] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _poisson_streams: Dict[Tuple[object, ...], PoissonDiskCandidateStream] = field(
        init=False,
        repr=False,
        default_factory=dict,
    )
    _default_object_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )
    _default_operator_base_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )
    _default_operator_eef_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )
    _default_camera_poses: Dict[str, PoseState] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        logging.getLogger(MujocoTaskBackend.__name__).info(
            "MujocoTaskBackend random_seed=%s", self.random_seed
        )
        self._rng = np.random.default_rng(self.random_seed)
        set_noise_seed = getattr(self.env, "set_camera_noise_seed", None)
        if set_noise_seed is not None:
            set_noise_seed(self.random_seed)
        self._validate_initial_joint_position_ownership()

    def _validate_initial_joint_position_ownership(self) -> None:
        """Reject duplicate env/operator definitions for arm home joints.

        The low-level environment map remains the scene-level default, while
        ``task_operators.<name>.initial_state.joint_positions`` is the task
        override. A joint appearing in both maps would make reset order the
        hidden source of truth, so fail at backend construction with a clear
        migration hint instead.
        """
        env_config = getattr(self.env, "config", None)
        env_initial = getattr(env_config, "initial_joint_positions", None)
        if not env_initial or not self.operator_initial_states:
            return
        envs = getattr(self.env, "envs", ())
        if not envs:
            return
        single_env = envs[0]
        model = getattr(single_env, "model", None)
        if model is None:
            return
        arm_names_by_operator: dict[str, set[str]] = {}
        for operator_name, initial_state in self.operator_initial_states.items():
            if not initial_state.joint_positions:
                continue
            actuator_indices = getattr(single_env, "_op_arm_aidx", {}).get(
                operator_name, ()
            )
            names: set[str] = set()
            for actuator_index in actuator_indices:
                joint_id = int(model.actuator_trnid[int(actuator_index), 0])
                if joint_id < 0:
                    continue
                joint_name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                if joint_name:
                    names.add(joint_name)
            arm_names_by_operator[operator_name] = names
        for operator_name, arm_names in arm_names_by_operator.items():
            duplicated = sorted(set(env_initial).intersection(arm_names))
            if duplicated:
                raise ValueError(
                    f"Initial joint(s) {duplicated} for operator "
                    f"'{operator_name}' are defined in both env.initial_joint_positions "
                    "and task_operators.<name>.initial_state.joint_positions; "
                    "remove the env entries or set env.initial_joint_positions: null."
                )

    def get_env(self) -> BatchedUnifiedMujocoEnv:
        return self.env

    @property
    def batch_size(self) -> int:
        return self.env.batch_size

    @property
    def dt_per_update(self) -> float:
        e = self.env.envs[0]
        return e.model.opt.timestep * e._n_substeps

    def get_random_generator(self) -> np.random.Generator:
        return self._rng

    def get_camera_reset_poses(self, env_index: int) -> Dict[str, PoseState]:
        if not 0 <= env_index < self.batch_size:
            raise IndexError(
                f"env_index must be in [0, {self.batch_size}), got {env_index}"
            )
        poses: Dict[str, PoseState] = {}
        for camera_name in self.camera_randomization:
            try:
                poses[camera_name] = self._get_camera_pose(camera_name).select(
                    env_index
                )
            except KeyError:
                continue
        return poses

    def get_randomization_diagnostics(self, env_index: int = 0) -> Dict[str, Any]:
        """Return diagnostics from the most recent constrained reset."""
        if not 0 <= env_index < self.batch_size:
            raise IndexError(
                f"env_index must be in [0, {self.batch_size}), got {env_index}"
            )
        entries = self._last_randomization_diagnostics.get(env_index, ())
        return {"attempts": [dict(entry) for entry in entries]} if entries else {}

    def get_camera_model(self, camera_name: str, env_index: int = 0) -> CameraModel:
        if not 0 <= env_index < self.batch_size:
            raise IndexError(
                f"env_index must be in [0, {self.batch_size}), got {env_index}"
            )
        physical_index = (
            0 if bool(getattr(self.env, "_share_physics", False)) else env_index
        )
        return self.env.envs[physical_index].get_camera_model(camera_name)

    def get_support_geometry(
        self,
        entity_name: str,
        env_index: int = 0,
    ) -> SupportGeometry:
        if not 0 <= env_index < self.batch_size:
            raise IndexError(
                f"env_index must be in [0, {self.batch_size}), got {env_index}"
            )
        handler = self.get_object_handler(entity_name)
        if handler is None:
            raise KeyError(f"Unknown randomization entity '{entity_name}'.")
        physical_index = (
            0 if bool(getattr(self.env, "_share_physics", False)) else env_index
        )
        return self.env.envs[physical_index].get_support_geometry(handler.body_name)

    def evaluate_randomization_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int = 0,
        constraints: Any = None,
        ancestors: Optional[Mapping[str, Set[str]]] = None,
        target_names: Optional[Set[str]] = None,
    ) -> RandomizationConstraintReport:
        physical_index = (
            0 if bool(getattr(self.env, "_share_physics", False)) else env_index
        )
        mapped_poses = {
            self.object_handlers[name].body_name: pose
            for name, pose in candidate_poses.items()
            if name in self.object_handlers
        }
        mapped_ancestors = None
        if ancestors is not None:
            mapped_ancestors = {
                self.object_handlers[name].body_name: {
                    self.object_handlers[ancestor].body_name
                    for ancestor in values
                    if ancestor in self.object_handlers
                }
                for name, values in ancestors.items()
                if name in self.object_handlers
            }
        return self.env.envs[physical_index].evaluate_randomization_constraints(
            mapped_poses,
            constraints=constraints,
            ancestors=mapped_ancestors,
            target_names={
                self.object_handlers[name].body_name
                for name in (target_names or set(mapped_poses))
                if name in self.object_handlers
            },
        )

    def setup(self, config: AutoAtomConfig) -> None:
        for operator in self.operator_handlers.values():
            operator.home()
        if self.initial_poses:
            self._apply_initial_poses()
        self.apply_operator_initial_states(home=True)
        if self.camera_initial_poses:
            self._apply_camera_initial_poses()
        self._record_default_poses()

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        mask = self._normalize_mask(env_mask)
        self._randomization_reset_index += 1
        self._last_randomization_diagnostics.clear()
        self.env.reset(env_mask)
        for operator in self.operator_handlers.values():
            operator.home(mask)
        if self.initial_poses:
            self._apply_initial_poses(mask)
        self.apply_operator_initial_states(mask, home=True)
        if self.camera_initial_poses:
            self._apply_camera_initial_poses(mask)
        if not (
            self._default_object_poses
            or self._default_operator_base_poses
            or self._default_operator_eef_poses
            or self._default_camera_poses
        ):
            self._record_default_poses()
        if self.randomization or self.camera_randomization:
            self._apply_randomization(mask)
        self.env.refresh_viewer()

    @contextmanager
    def defer_viewer_updates(self) -> Iterator[None]:
        with self.env.defer_viewer_updates():
            yield

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

    def apply_object_pose(
        self,
        object_name: str,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Apply a kinematic object pose and refresh the passive viewer."""
        super().apply_object_pose(object_name, pose, env_mask=env_mask)
        self.env.refresh_viewer()

    def _record_default_poses(self) -> None:
        for name, handler in self.object_handlers.items():
            self._default_object_poses[name] = handler.get_pose()
        for name, handler in self.operator_handlers.items():
            self._default_operator_base_poses[name] = handler.get_base_pose()
            self._default_operator_eef_poses[name] = handler.get_end_effector_pose()
        for cam_name in self.camera_randomization:
            self._default_camera_poses[cam_name] = self._get_camera_pose(cam_name)

    def _resolve_initial_reference_pose(
        self,
        reference: PoseReference | str,
        env_index: int,
        *,
        operator_name: str | None = None,
        allow_operator_base: bool = False,
        context: str,
    ) -> PoseState:
        """Resolve one initial-pose reference through the backend seam.

        Built-in ``world`` is the identity frame.  ``base`` is available only
        when resolving an operator EEF pose.  Any other non-built-in string is
        a named scene element (site/body/geom/joint).  Operator-frame aliases
        are likewise limited to EEF overrides; a base override must anchor to
        the world or a scene element so its resolution does not depend on
        another operator's initialization order.
        """
        if isinstance(reference, PoseReference):
            if reference == PoseReference.WORLD:
                return PoseState()
            if reference == PoseReference.BASE and allow_operator_base:
                if operator_name is None:
                    raise ValueError(f"{context} reference 'base' requires an operator")
                return (
                    self.operator_handlers[operator_name]
                    .get_base_pose()
                    .select(env_index)
                )
            raise ValueError(
                f"{context} reference {reference.value!r} is not valid here; "
                "use 'world' or a named scene element"
            )

        if not isinstance(reference, str) or not reference:
            raise ValueError(
                f"{context} reference must be 'world' or a scene element name"
            )

        # A reference that exactly names another configured object is resolved
        # through its handler.  This takes precedence over a same-named site or
        # body and, importantly, observes that object's already-applied initial
        # pose (the caller orders such objects topologically).
        if reference in self.object_handlers:
            return self.object_handlers[reference].get_pose().select(env_index)

        bare, attr = self._parse_entity_reference(reference)
        if attr is not None:
            if operator_name is None:
                raise ValueError(
                    f"{context} reference {reference!r} is an operator frame; "
                    "operator frame aliases are only valid for operator poses"
                )
            if bare not in self.operator_handlers:
                raise ValueError(
                    f"{context} reference {reference!r} names an unknown operator"
                )
            if attr == "base":
                return self.operator_handlers[bare].get_base_pose().select(env_index)
            return (
                self.operator_handlers[bare].get_end_effector_pose().select(env_index)
            )

        try:
            return self.get_element_pose(reference, env_index)
        except KeyError as exc:
            raise ValueError(
                f"{context} reference {reference!r} does not name a scene "
                "site, body, geom, or joint"
            ) from exc

    def _resolve_initial_pose_batch(
        self,
        config: PoseOverrideConfig | list[float] | tuple[float, ...],
        fallback_pose: PoseState,
        env_mask: np.ndarray,
        *,
        operator_name: str | None = None,
        allow_operator_base: bool = False,
        context: str,
    ) -> PoseState:
        """Resolve an initial-pose config independently for each env row."""
        if fallback_pose.batch_size != self.batch_size:
            fallback_pose = fallback_pose.broadcast_to(self.batch_size)
        positions = fallback_pose.position.copy()
        orientations = fallback_pose.orientation.copy()
        reference = (
            PoseReference.WORLD
            if isinstance(config, (list, tuple))
            else config.reference
        )
        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            reference_poses: Dict[PoseReference | str, PoseState] = {}
            if isinstance(config, PoseOverrideConfig):
                for axis_reference in config.axis_references():
                    reference_poses[axis_reference] = (
                        self._resolve_initial_reference_pose(
                            axis_reference,
                            env_index,
                            operator_name=operator_name,
                            allow_operator_base=allow_operator_base,
                            context=context,
                        )
                    )
            else:
                reference_poses[reference] = self._resolve_initial_reference_pose(
                    reference,
                    env_index,
                    operator_name=operator_name,
                    allow_operator_base=allow_operator_base,
                    context=context,
                )
            resolved = resolve_pose_override(
                config,
                fallback_pose.select(env_index),
                reference_poses.get(reference),
                reference_poses,
            )
            positions[env_index] = resolved.position[0]
            orientations[env_index] = resolved.orientation[0]
        return PoseState(position=positions, orientation=orientations)

    def apply_operator_initial_states(
        self,
        env_mask: np.ndarray | None = None,
        *,
        home: bool = False,
    ) -> None:
        """Apply configured operator initial states through one pose resolver.

        Base references are resolved after scene initial poses have been
        applied.  ``home=False`` is used only while staging state changes;
        setup/reset pass ``home=True`` so the physical arm state is homed once
        after all configured fields have been resolved.
        """
        if not self.operator_initial_states:
            return
        mask = self._normalize_mask(env_mask)

        # Resolve all bases first: an EEF override expressed in ``base`` must
        # see the newly selected base frame.
        for name, initial_state in self.operator_initial_states.items():
            handler = self.operator_handlers.get(name)
            if handler is None:
                raise KeyError(f"Unknown operator in initial_state: {name!r}")
            if initial_state is None or initial_state.base_pose is None:
                continue
            resolved = self._resolve_initial_pose_batch(
                initial_state.base_pose,
                handler.get_base_pose(),
                mask,
                context=f"operator {name} base_pose",
            )
            # Route the resolved pose through the handler seam so shared
            # physics row validation, controller-state invalidation, and
            # backend-specific base handling stay in one place.
            handler.set_pose(resolved, env_mask=mask)

        for name, initial_state in self.operator_initial_states.items():
            handler = self.operator_handlers.get(name)
            if handler is None:
                raise KeyError(f"Unknown operator in initial_state: {name!r}")
            if initial_state is None:
                continue
            has_override = (
                initial_state.base_pose is not None
                or initial_state.eef_pose is not None
                or bool(initial_state.joint_positions)
                or initial_state.eef is not None
            )
            if not has_override:
                continue
            if initial_state.joint_positions:
                handler.set_home_joint_positions(
                    initial_state.joint_positions,
                    env_mask=mask,
                    apply_home=False,
                )
            if initial_state.eef_pose is not None:
                resolved = self._resolve_initial_pose_batch(
                    initial_state.eef_pose,
                    handler.get_end_effector_pose(),
                    mask,
                    operator_name=name,
                    allow_operator_base=True,
                    context=f"operator {name} eef_pose",
                )
                handler.set_home_end_effector_pose(
                    resolved,
                    env_mask=mask,
                    apply_home=False,
                )
            if initial_state.eef is not None:
                eef_value = float(initial_state.eef)
                shared = bool(getattr(handler.env, "_share_physics", False))
                for env_index, enabled in enumerate(mask):
                    if not enabled:
                        continue
                    # Shared physics has one physical operator state; update
                    # it once while keeping the handler's logical view in
                    # sync for the subsequent home/settle pass.
                    physical_index = 0 if shared else env_index
                    state = handler.env.envs[physical_index]._operator_states[name]
                    state.home_ctrl[handler.eef_ctrl_index] = eef_value
                    handler._home_ctrl[env_index, handler.eef_ctrl_index] = eef_value

            if home:
                handler.home(mask)

    def _apply_initial_poses(self, env_mask: np.ndarray | None = None) -> None:
        """Apply per-object initial pose overrides from config.

        Called after keyframe reset and operator homing, before default-pose
        recording and randomization.  Only the specified components (position
        and/or orientation) are overridden; the rest keep their keyframe value.

        After setting each object's pose the selected rows of the recorded
        default are updated so subsequent randomization uses the effective
        initial pose as its baseline.  Unselected rows retain their existing
        baselines, which keeps masked resets from absorbing a prior episode's
        random sample.  Callers may mutate ``self.initial_poses`` between
        resets for per-episode initial conditions.
        """
        mask = self._normalize_mask(env_mask)
        for name in self._initial_pose_order():
            cfg = self.initial_poses[name]
            handler = self.object_handlers.get(name)
            if handler is None:
                continue
            current = handler.get_pose()
            # Keep one stable randomization baseline per logical environment.
            # A masked reset must not copy an unselected row's episode pose
            # (which may already include random offsets) into that baseline.
            baseline = self._default_object_poses.get(name)
            if baseline is None:
                baseline = current
            else:
                try:
                    baseline = baseline.broadcast_to(self.batch_size)
                except ValueError:
                    baseline = current
            resolved = self._resolve_initial_pose_batch(
                cfg,
                current,
                mask,
                context=f"initial_pose[{name!r}]",
            )
            handler.set_pose(resolved, env_mask=mask)
            # Keep recorded defaults in sync so randomization offsets from
            # the (possibly dynamic) initial pose, not the stale keyframe.
            effective = handler.get_pose()
            positions = baseline.position.copy()
            orientations = baseline.orientation.copy()
            positions[mask] = effective.position[mask]
            orientations[mask] = effective.orientation[mask]
            self._default_object_poses[name] = PoseState(
                position=positions,
                orientation=orientations,
            )

    def _initial_pose_order(self) -> list[str]:
        """Return configured initial-pose keys in dependency order.

        A string reference that exactly matches another ``initial_poses`` key
        depends on that key's effective pose.  Resolve those dependencies
        before applying the dependent override, while retaining declaration
        order for independent entries.  Cycles are rejected before any model
        mutation so a failed configuration cannot leave a half-applied scene.
        References that do not name an ``initial_poses`` key remain ordinary
        MuJoCo scene-frame references and are resolved at application time.
        """
        names = list(self.initial_poses)
        declaration_index = {name: index for index, name in enumerate(names)}
        dependencies: dict[str, set[str]] = {name: set() for name in names}
        for name in names:
            config = self.initial_poses[name]
            references = (
                config.axis_references()
                if isinstance(config, PoseOverrideConfig)
                else ()
            )
            for reference in references:
                if (
                    isinstance(reference, str)
                    and not isinstance(reference, PoseReference)
                    and reference in dependencies
                ):
                    dependencies[name].add(reference)

        order: list[str] = []
        visited: set[str] = set()
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Circular initial pose reference involving {name!r}")
            visiting.add(name)
            for dependency in sorted(
                dependencies[name], key=declaration_index.__getitem__
            ):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            order.append(name)

        for name in names:
            visit(name)
        return order

    def _apply_camera_initial_poses(self, env_mask: np.ndarray | None = None) -> None:
        """Apply per-camera initial pose overrides from config.

        Runs after the model reset and before ``_record_default_poses``
        so camera_randomization samples around the overridden pose. Only
        specified components (position and/or orientation) are changed;
        the rest keep their XML value.
        """
        mask = self._normalize_mask(env_mask)
        for cam_name, cfg in self.camera_initial_poses.items():
            current = self._get_camera_pose(cam_name)
            baseline = self._default_camera_poses.get(cam_name)
            if baseline is None:
                baseline = current
            else:
                try:
                    baseline = baseline.broadcast_to(self.batch_size)
                except ValueError:
                    baseline = current
            resolved = self._resolve_initial_pose_batch(
                cfg,
                current,
                mask,
                context=f"camera_initial_pose[{cam_name!r}]",
            )
            self._set_camera_pose(cam_name, resolved, mask)
            # Keep only selected rows in sync so camera_randomization offsets
            # from the overridden pose without absorbing unselected samples.
            effective = self._get_camera_pose(cam_name)
            positions = baseline.position.copy()
            orientations = baseline.orientation.copy()
            positions[mask] = effective.position[mask]
            orientations[mask] = effective.orientation[mask]
            self._default_camera_poses[cam_name] = PoseState(
                position=positions,
                orientation=orientations,
            )

    # ------------------------------------------------------------------
    #  Randomization: ordering, reference resolution, and application
    # ------------------------------------------------------------------

    def _select_randomization_region(
        self,
        spec: RandomizationInput,
        *,
        candidate_index: int = 0,
    ) -> PoseRandomRange:
        """Select one region from a possibly multi-region randomization spec.

        A wrapper's regions are equiprobable.  Selection is deliberately made
        at the point where a target is sampled so collision-rejection retries
        naturally draw a fresh region on every attempt.  The legacy single
        ``PoseRandomRange`` path does not consume an extra random value.
        """
        canonical = canonical_randomization_spec(spec)
        regions = pose_randomization_regions(canonical)
        if not regions:
            raise ValueError("Randomization region lists must not be empty")
        if len(regions) == 1:
            return regions[0]
        distribution = canonical.distribution
        if distribution.region_weighting == "volume":
            volumes = []
            for region in regions:
                volume = 1.0
                for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
                    axis_range = region.axis_range(axis)
                    if axis_range is not None:
                        volume *= max(float(axis_range[1]) - float(axis_range[0]), 0.0)
                volumes.append(volume)
            total_volume = float(sum(volumes))
            if total_volume > 0.0:
                sampled = float(self._rng.uniform(0.0, total_volume))
                cumulative = 0.0
                for region, volume in zip(regions, volumes):
                    cumulative += volume
                    if sampled <= cumulative:
                        return region
        sampled_index = int(self._rng.uniform(0.0, float(len(regions))))
        sampled_index = max(0, min(sampled_index, len(regions) - 1))
        return regions[sampled_index]

    @staticmethod
    def _parse_entity_reference(ref: str) -> Tuple[str, Optional[str]]:
        """Split an entity-name reference into ``(name, attr)``.

        ``'arm.base'`` → ``('arm', 'base')``; ``'arm.eef'`` → ``('arm', 'eef')``;
        plain ``'vase'`` → ``('vase', None)``. Only ``.base`` / ``.eef`` suffixes
        are recognized; any other dotted form is returned unchanged.
        """
        if "." in ref:
            name, attr = ref.split(".", 1)
            if attr in ("base", "eef"):
                return name, attr
        return ref, None

    def _randomization_action_specs(self) -> Dict[str, _RandomizationActionSpec]:
        """Expand public entity configs into independently ordered actions."""
        plan = self._randomization_plan()
        return {
            label: _RandomizationActionSpec(
                kind=action.kind,
                owner=action.owner,
                label=action.label,
                randomization=action.randomization,
            )
            for label, action in plan.actions.items()
        }

    def _randomization_plan(self) -> RandomizationPlan:
        """Compile the backend's public randomization mapping into a plan."""
        return compile_randomization_plan(
            self.randomization,
            object_names=set(self.object_handlers),
            operator_names=set(self.operator_handlers),
            randomization_groups=(
                self.randomization_groups
                if self.randomization_strategy == RandomizationStrategy.RSA
                else {}
            ),
            strategy=self.randomization_strategy,
        )

    def _randomization_dependencies(self) -> Dict[str, Set[str]]:
        return {
            label: set(dependencies)
            for label, dependencies in self._randomization_plan().dependencies.items()
        }

    @staticmethod
    def _distribution_uses_space_filling_history(distribution: Any) -> bool:
        """Return whether a generator participates in cross-reset coverage.

        ``selector`` is intentionally absent from this decision.  Selectors
        operate on the current candidate group only; non-IID generators own
        the persistent sequence semantics that make accepted samples from
        earlier resets unavailable to later resets.
        """
        generator = getattr(distribution, "generator", RandomizationGeneratorKind.IID)
        return generator != RandomizationGeneratorKind.IID

    @staticmethod
    def _history_clearance(vector: np.ndarray, history: list[np.ndarray]) -> float:
        """Return distance from a candidate to its nearest accepted sample."""
        if not history:
            return float("inf")
        points = np.vstack(history)
        return float(
            np.linalg.norm(
                np.asarray(vector, dtype=np.float64) - points,
                axis=1,
            ).min()
        )

    def _record_space_filling_sample(
        self,
        history_key: Tuple[str, ...],
        vector: np.ndarray,
    ) -> None:
        """Record only the pose selected/applied for a reset component."""
        history = self._space_filling_history.get(history_key, [])
        history = (history + [np.asarray(vector, dtype=np.float64).copy()])[-256:]
        self._space_filling_history[history_key] = history

    def _poisson_stream_for_action(
        self,
        action_spec: _RandomizationActionSpec,
        env_index: int,
        rand_range: PoseRandomRange,
        distribution: Any,
        working_poses: Dict[str, PoseState],
    ) -> PoissonDiskCandidateStream | None:
        """Return the persistent physical-position stream for one proposal range."""
        generator = getattr(
            distribution,
            "generator",
            RandomizationGeneratorKind.IID,
        )
        if isinstance(generator, RandomizationGeneratorConfig):
            poisson_config = generator.poisson_disk
        elif generator == RandomizationGeneratorKind.POISSON_DISK:
            poisson_config = RandomizationPoissonDiskConfig()
        else:
            return None

        position_axes = tuple(
            axis for axis in ("x", "y", "z") if rand_range.axis_range(axis) is not None
        )
        if not position_axes:
            return None
        min_distance = float(getattr(distribution, "min_distance", 0.0))
        if min_distance <= 0.0:
            raise ValueError(
                f"Poisson-disk randomization '{action_spec.label}' requires "
                "distribution.min_distance > 0"
            )
        lower_bounds = tuple(
            float(rand_range.axis_range(axis)[0]) for axis in position_axes
        )
        upper_bounds = tuple(
            float(rand_range.axis_range(axis)[1]) for axis in position_axes
        )
        reference_context: list[float] = []
        for reference in rand_range.references():
            if isinstance(reference, RandomizationReference):
                if reference == RandomizationReference.ABSOLUTE_WORLD:
                    continue
                if reference == RandomizationReference.ABSOLUTE_BASE:
                    key = f"{action_spec.owner}.base"
                    pose = working_poses.get(key)
                    if pose is None:
                        pose = self._default_operator_base_poses.get(
                            action_spec.owner,
                            self.operator_handlers[action_spec.owner].get_base_pose(),
                        )
                else:
                    pose = working_poses.get(action_spec.label)
                    if pose is None:
                        pose = self._current_pose_for_action(
                            action_spec.kind, action_spec.owner
                        )
            else:
                bare, attr = self._parse_entity_reference(reference)
                if attr is None and bare in self.operator_handlers:
                    attr = "base"
                key = f"{bare}.{attr}" if attr is not None else bare
                pose = working_poses.get(key)
                if pose is None and attr is None:
                    pose = working_poses.get(bare)
                if pose is None:
                    if attr == "base":
                        pose = self._default_operator_base_poses.get(
                            bare, self.operator_handlers[bare].get_base_pose()
                        )
                    elif attr == "eef":
                        pose = self._default_operator_eef_poses.get(
                            bare, self.operator_handlers[bare].get_end_effector_pose()
                        )
                    else:
                        pose = self._default_object_poses.get(
                            bare, self.object_handlers[bare].get_pose()
                        )
            selected_pose = pose.select(env_index)
            reference_context.extend(
                np.asarray(selected_pose.position[0], dtype=np.float64).tolist()
            )
            reference_context.extend(
                np.asarray(selected_pose.orientation[0], dtype=np.float64).tolist()
            )
        key = (
            env_index,
            action_spec.label,
            position_axes,
            lower_bounds,
            upper_bounds,
            min_distance,
            tuple(reference_context),
            poisson_config.hypersphere,
            int(poisson_config.ncandidates),
            poisson_config.optimization,
        )
        stream = self._poisson_streams.get(key)
        if stream is None:
            label_seed = sum(
                (index + 1) * ord(character)
                for index, character in enumerate(action_spec.label)
            )
            stream = PoissonDiskCandidateStream(
                poisson_config,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                radius=min_distance,
                seed=int((self.random_seed or 0) + env_index * 10_007 + label_seed),
            )
            self._poisson_streams[key] = stream
        return stream

    def _randomization_order(self) -> List[str]:
        """Return randomization keys in dependency order (referenced first).

        An entry whose ``reference`` is another entity name depends on that
        entity being sampled first. Cycles raise ``ValueError``.
        """
        return list(self._randomization_plan().order)

    def _reference_ancestors(
        self,
        references: tuple[Union[RandomizationReference, str], ...],
        selected_ancestors: Dict[str, Set[str]],
    ) -> Set[str]:
        ancestors: Set[str] = set()
        for reference in references:
            if isinstance(reference, RandomizationReference):
                continue
            bare, attr = self._parse_entity_reference(reference)
            if attr is None and bare in self.operator_handlers:
                attr = "base"
            reference_key = f"{bare}.{attr}" if attr is not None else bare
            ancestors.add(bare)
            ancestors.update(selected_ancestors.get(reference_key, ()))
        return ancestors

    def _validate_pose_randomization_spec(
        self,
        label: str,
        spec: PoseRandomizationSpec,
        *,
        allow_absolute_base: bool,
    ) -> None:
        """Validate every region against its target context before sampling."""
        for region_index, region in enumerate(pose_randomization_regions(spec)):
            references = region.references()
            if RandomizationReference.ABSOLUTE_BASE in references:
                if not allow_absolute_base:
                    raise ValueError(
                        f"{label} randomization region {region_index} cannot use "
                        "'absolute_base' — only operator end-effector "
                        "randomization is defined in a base frame."
                    )
                if any(
                    reference != RandomizationReference.ABSOLUTE_BASE
                    for reference in references
                ):
                    raise ValueError(
                        f"{label} randomization region {region_index} cannot mix "
                        "'absolute_base' with references in other frames."
                    )
            for reference in references:
                if isinstance(reference, RandomizationReference):
                    continue
                bare, attr = self._parse_entity_reference(reference)
                if attr is not None and bare not in self.operator_handlers:
                    raise ValueError(
                        f"{label} randomization region {region_index} reference "
                        f"'{reference}' uses '.{attr}', but '{bare}' is not a "
                        "known operator."
                    )
                if (
                    attr is None
                    and bare not in self.object_handlers
                    and bare not in self.operator_handlers
                ):
                    raise ValueError(
                        f"{label} randomization region {region_index} reference "
                        f"'{reference}' is not a known object or operator."
                    )

    def _validate_randomization_configuration(self) -> None:
        """Validate target-specific rules for all configured regions."""
        for name, spec in self.randomization.items():
            if name in self.object_handlers:
                if isinstance(spec, OperatorRandomizationConfig):
                    raise TypeError(
                        f"Object '{name}' randomization must use a direct "
                        "single- or multi-region pose specification, not an "
                        "operator randomization config."
                    )
                self._validate_pose_randomization_spec(
                    f"Object '{name}'",
                    spec,
                    allow_absolute_base=False,
                )
                continue
            if name not in self.operator_handlers:
                logging.getLogger(MujocoTaskBackend.__name__).warning(
                    "Randomization key '%s' does not match any object or "
                    "operator handler — skipping.",
                    name,
                )
                continue
            if not isinstance(spec, OperatorRandomizationConfig):
                raise TypeError(
                    f"Operator '{name}' randomization must use the nested form "
                    "with explicit `base:` and/or `eef:` sub-entries (i.e. an "
                    "OperatorRandomizationConfig). Direct pose randomization "
                    "specifications are not supported."
                )
            if spec.base is not None:
                self._validate_pose_randomization_spec(
                    f"Operator '{name}' base",
                    spec.base,
                    allow_absolute_base=False,
                )
            if spec.eef is not None:
                self._validate_pose_randomization_spec(
                    f"Operator '{name}' end effector",
                    spec.eef,
                    allow_absolute_base=True,
                )

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
        bare, attr = self._parse_entity_reference(reference)
        if attr is None and bare in self.operator_handlers:
            attr = "base"  # plain operator name defaults to its base
        if attr is not None:
            if bare not in self.operator_handlers:
                raise ValueError(
                    f"Randomization reference '{reference}' — '.{attr}' is only "
                    f"valid for operator names, but '{bare}' is not a known operator."
                )
            handler = self.operator_handlers[bare]
            if attr == "base":
                ref_default = self._default_operator_base_poses.get(
                    bare, handler.get_base_pose()
                )
            else:
                ref_default = self._default_operator_eef_poses.get(
                    bare, handler.get_end_effector_pose()
                )
            ref_sampled = sampled_poses.get(f"{bare}.{attr}")
        else:
            ref_sampled = sampled_poses.get(bare)
            if bare in self._default_object_poses:
                ref_default = self._default_object_poses[bare]
            elif bare in self.object_handlers:
                ref_default = self.object_handlers[bare].get_pose()
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
        self._validate_randomization_configuration()
        plan = self._randomization_plan()
        order = list(plan.order)
        components = [list(component) for component in plan.components]
        action_specs = self._randomization_action_specs()
        hard_sphere_groups = {
            frozenset(group.members): (group_name, group)
            for group_name, group in plan.groups.items()
        }
        sampled_poses: Dict[str, PoseState] = {}
        collision_participants: List[_CollisionParticipant] = []

        def apply_actions(actions: List[_PendingRandomizationAction]) -> None:
            for action in actions:
                if action.kind == "object":
                    self.object_handlers[action.owner].set_pose(action.pose, env_mask)
                elif action.kind == "operator_base":
                    self.operator_handlers[action.owner].set_pose(action.pose, env_mask)
                elif action.kind == "operator_eef":
                    self.operator_handlers[action.owner].set_home_end_effector_pose(
                        action.pose,
                        env_mask=env_mask,
                    )
                else:
                    raise ValueError(
                        f"Unknown randomization action kind: {action.kind}"
                    )
                collision_participants.append(
                    _CollisionParticipant(
                        owner=action.owner,
                        label=action.label,
                        pose=action.pose,
                        radius=action.radius,
                        ancestors=_copy_randomization_ancestors(action.ancestors),
                    )
                )

        # Operators form the context for mounted cameras and object references.
        # Sampling them first makes the camera state final before visibility
        # constraints are evaluated for objects.
        operator_labels = {
            label for label, action in action_specs.items() if action.kind != "object"
        }
        for component in components:
            operator_component = [
                label for label in component if label in operator_labels
            ]
            if not operator_component:
                continue
            component_poses, component_actions = self._sample_randomization_component(
                operator_component,
                env_mask,
                sampled_poses,
                collision_participants,
            )
            apply_actions(component_actions)
            sampled_poses.update(component_poses)

        if self.camera_randomization:
            self._apply_camera_randomization(env_mask)

        # Object components retain reference-connected and separated joint
        # sampling, but now see the already-final operator/camera context.
        for component in components:
            object_component = [
                label for label in component if action_specs[label].kind == "object"
            ]
            if not object_component:
                continue
            hard_sphere_group = hard_sphere_groups.get(frozenset(object_component))
            component_poses, component_actions = self._sample_randomization_component(
                object_component,
                env_mask,
                sampled_poses,
                collision_participants,
                hard_sphere_rsa_group=hard_sphere_group,
                use_rsa=(
                    self.randomization_strategy == RandomizationStrategy.RSA
                    and len(object_component) > 1
                ),
            )
            apply_actions(component_actions)
            sampled_poses.update(component_poses)

    def _randomization_components(
        self,
        order: List[str],
        deps: Dict[str, Set[str]],
    ) -> List[List[str]]:
        adjacency: Dict[str, Set[str]] = {name: set() for name in deps}
        for name, parents in deps.items():
            for parent in parents:
                adjacency[name].add(parent)
                adjacency[parent].add(name)
        order_index = {name: index for index, name in enumerate(order)}
        visited: Set[str] = set()
        components: List[List[str]] = []
        for name in order:
            if name in visited:
                continue
            stack = [name]
            component: Set[str] = set()
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                stack.extend(adjacency[current] - visited)
            components.append(sorted(component, key=order_index.__getitem__))
        return components

    def _sample_randomization_component(
        self,
        component: List[str],
        env_mask: np.ndarray,
        accepted_sampled_poses: Dict[str, PoseState],
        accepted_participants: List[_CollisionParticipant],
        hard_sphere_rsa_group: tuple[str, RandomizationGroupConfig] | None = None,
        use_rsa: bool = False,
    ) -> tuple[Dict[str, PoseState], List[_PendingRandomizationAction]]:
        key_buffers = {
            name: self._current_pose_for_randomization_key(name) for name in component
        }
        action_buffers: Dict[str, _PendingRandomizationAction] = {}
        action_order: List[str] = []

        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            if not use_rsa and hard_sphere_rsa_group is None:
                env_sampled_poses, env_actions, failure = (
                    self._sample_component_for_env(
                        component,
                        env_index,
                        accepted_sampled_poses,
                        accepted_participants,
                    )
                )
            else:
                env_sampled_poses, env_actions, failure = (
                    self._sample_hard_sphere_rsa_component_for_env(
                        component,
                        env_index,
                        accepted_sampled_poses,
                        accepted_participants,
                        group_name=(
                            hard_sphere_rsa_group[0]
                            if hard_sphere_rsa_group is not None
                            else f"component:{','.join(component)}"
                        ),
                        group=(
                            hard_sphere_rsa_group[1]
                            if hard_sphere_rsa_group is not None
                            else None
                        ),
                    )
                )
            if failure is not None:
                logger = logging.getLogger(MujocoTaskBackend.__name__)
                failed_label, blocking_label = failure
                if not use_rsa and hard_sphere_rsa_group is None:
                    logger.warning(
                        "Collision rejection exhausted for '%s' after %d attempts; "
                        "keeping the last overlapping sample against '%s'.",
                        failed_label,
                        _MAX_COLLISION_REJECTION_ATTEMPTS,
                        blocking_label,
                    )
                else:
                    logger.warning(
                        "Hard-sphere RSA exhausted for '%s' after %d attempts; "
                        "keeping the best-effort sample against '%s'.",
                        failed_label,
                        int(
                            hard_sphere_rsa_group[1].failure.max_attempts
                            if hard_sphere_rsa_group is not None
                            else _MAX_COLLISION_REJECTION_ATTEMPTS
                        ),
                        blocking_label,
                    )

            for name, pose in env_sampled_poses.items():
                if name not in key_buffers:
                    continue
                key_buffers[name].position[env_index] = pose.position[0]
                key_buffers[name].orientation[env_index] = pose.orientation[0]
            for action in env_actions:
                if action.label not in action_buffers:
                    template = self._current_pose_for_action(action.kind, action.owner)
                    buffered_radius: float | np.ndarray = float(action.radius)
                    action_ancestors = self._collision_ancestors_for_env(
                        action.ancestors,
                        env_index,
                    )
                    buffered_ancestors: _RandomizationAncestors = set(action_ancestors)
                    if self.batch_size > 1:
                        buffered_radius = np.full(
                            self.batch_size,
                            float(action.radius),
                            dtype=np.float64,
                        )
                        buffered_ancestors = [
                            set(action_ancestors) for _ in range(self.batch_size)
                        ]
                    action_buffers[action.label] = _PendingRandomizationAction(
                        kind=action.kind,
                        owner=action.owner,
                        label=action.label,
                        pose=template,
                        radius=buffered_radius,
                        ancestors=buffered_ancestors,
                        constraints=action.constraints,
                    )
                    action_order.append(action.label)
                action_buffers[action.label].pose.position[env_index] = (
                    action.pose.position[0]
                )
                action_buffers[action.label].pose.orientation[env_index] = (
                    action.pose.orientation[0]
                )
                if isinstance(action_buffers[action.label].radius, np.ndarray):
                    action_buffers[action.label].radius[env_index] = float(
                        action.radius
                    )
                else:
                    action_buffers[action.label].radius = float(action.radius)
                buffered_action_ancestors = action_buffers[action.label].ancestors
                env_action_ancestors = self._collision_ancestors_for_env(
                    action.ancestors,
                    env_index,
                )
                if isinstance(buffered_action_ancestors, list):
                    buffered_action_ancestors[env_index] = set(env_action_ancestors)
                else:
                    action_buffers[action.label].ancestors = set(env_action_ancestors)

        component_actions = [action_buffers[label] for label in action_order]
        return key_buffers, component_actions

    def _sample_hard_sphere_rsa_component_for_env(
        self,
        component: List[str],
        env_index: int,
        accepted_sampled_poses: Dict[str, PoseState],
        accepted_participants: List[_CollisionParticipant],
        *,
        group_name: str,
        group: RandomizationGroupConfig | None = None,
    ) -> tuple[
        Dict[str, PoseState],
        List[_PendingRandomizationAction],
        Optional[tuple[str, str]],
    ]:
        """Place group members sequentially with heterogeneous hard spheres.

        RSA is deliberately local: a rejected proposal is discarded, while
        already accepted members remain fixed for the current reset.  Member
        order is independently shuffled for every environment/reset, so the
        YAML declaration order does not become a spatial bias.
        """
        action_specs = self._randomization_action_specs()
        accepted_env_poses = {
            name: pose.select(env_index)
            for name, pose in accepted_sampled_poses.items()
        }
        working_poses = dict(accepted_env_poses)
        sampled_poses: Dict[str, PoseState] = {}
        actions: List[_PendingRandomizationAction] = []
        local_participants: List[_CollisionParticipant] = []
        dependencies = self._randomization_plan().dependencies
        remaining = set(component)
        order: list[str] = []
        while remaining:
            ready = [
                label
                for label in component
                if label in remaining
                and not (set(dependencies.get(label, ())) & remaining)
            ]
            if not ready:
                raise ValueError(
                    f"Circular randomization reference in RSA component {component!r}"
                )
            if len(ready) > 1 and hasattr(self._rng, "permutation"):
                ready = [str(value) for value in self._rng.permutation(ready)]
            order.extend(ready)
            remaining.difference_update(ready)
        configured_attempts = [
            int(action_specs[label].randomization.failure.max_attempts)
            for label in component
        ]
        max_attempts = (
            int(group.failure.max_attempts)
            if group is not None
            else min(configured_attempts, default=_MAX_COLLISION_REJECTION_ATTEMPTS)
        )
        failure_mode = (
            group.failure.mode.value
            if group is not None
            else (
                "error"
                if any(
                    action_specs[label].randomization.failure.mode.value == "error"
                    for label in component
                )
                else "best_effort"
            )
        )
        pair_clearance = max(
            (
                float(
                    action_specs[label].randomization.constraints.separated.min_distance
                )
                for label in component
                if action_specs[label].randomization.constraints.separated is not None
            ),
            default=0.0,
        )
        selected_ancestors: Dict[str, Set[str]] = {}
        for member in order:
            action_spec = action_specs[member]
            best: tuple[float, PoseState, _PendingRandomizationAction, str] | None = (
                None
            )
            accepted = False
            for attempt in range(max_attempts):
                candidate_poses, candidate_actions = (
                    self._sample_randomization_target_for_env(
                        action_spec,
                        env_index,
                        working_poses,
                        candidate_index=(
                            self._randomization_reset_index * 1009
                            + attempt * 17
                            + sum(ord(c) for c in member)
                        ),
                    )
                )
                candidate = next(
                    (item for item in candidate_actions if item.label == member),
                    None,
                )
                if candidate is None:
                    raise RuntimeError(
                        f"Randomization group '{group_name}' member '{member}' "
                        "did not produce a candidate"
                    )
                candidate.ancestors = self._reference_ancestors(
                    candidate.references,
                    selected_ancestors,
                )
                blocking = self._find_collision_participant(
                    owner_name=candidate.owner,
                    env_index=env_index,
                    candidate_pose=candidate.pose,
                    collision_radius=candidate.radius,
                    ancestors=candidate.ancestors,
                    collision_participants=accepted_participants,
                )
                if blocking is None:
                    blocking = self._find_collision_participant(
                        owner_name=candidate.owner,
                        env_index=env_index,
                        candidate_pose=candidate.pose,
                        collision_radius=candidate.radius,
                        ancestors=candidate.ancestors,
                        collision_participants=local_participants,
                        extra_clearance=pair_clearance,
                    )
                constraint_report: RandomizationConstraintReport | None = None
                if candidate.constraints is not None and (
                    candidate.constraints.visible_in is not None
                    or candidate.constraints.separated is not None
                ):
                    candidate_poses_for_constraints = dict(working_poses)
                    candidate_poses_for_constraints.update(candidate_poses)
                    candidate_poses_for_constraints[member] = candidate.pose
                    candidate_constraint_ancestors = {
                        participant.owner: self._collision_ancestors_for_env(
                            participant.ancestors,
                            env_index,
                        )
                        for participant in accepted_participants + local_participants
                    }
                    candidate_constraint_ancestors[candidate.owner] = set(
                        candidate.ancestors
                    )
                    constraint_report = self.evaluate_randomization_constraints(
                        candidate_poses_for_constraints,
                        env_index=env_index,
                        constraints=candidate.constraints,
                        ancestors=candidate_constraint_ancestors,
                        target_names={candidate.owner},
                    )
                violation = (
                    f"{candidate.label}:collides:{blocking.label}"
                    if blocking is not None
                    else (
                        constraint_report.violations[0]
                        if constraint_report is not None and not constraint_report.valid
                        else ""
                    )
                )
                candidate_row = 0 if candidate.pose.batch_size == 1 else env_index
                candidate_pos = np.asarray(
                    candidate.pose.position[candidate_row], dtype=np.float64
                )
                clearance = float("inf")
                if blocking is not None:
                    other_row = 0 if blocking.pose.batch_size == 1 else env_index
                    other_pos = np.asarray(
                        blocking.pose.position[other_row], dtype=np.float64
                    )
                    local_blocking = any(
                        blocking is participant for participant in local_participants
                    )
                    clearance = float(
                        np.linalg.norm(candidate_pos - other_pos)
                        - candidate.radius
                        - self._collision_radius_for_env(blocking.radius, env_index)
                        - (pair_clearance if local_blocking else 0.0)
                    )
                elif constraint_report is not None and not constraint_report.valid:
                    clearance = float(constraint_report.minimum_clearance)
                if best is None or clearance > best[0]:
                    best = (clearance, candidate.pose, candidate, violation)
                if blocking is not None or (
                    constraint_report is not None and not constraint_report.valid
                ):
                    continue
                candidate_poses.update({member: candidate.pose})
                working_poses.update(candidate_poses)
                candidate.ancestors = set(candidate.ancestors)
                local_participants.append(
                    _CollisionParticipant(
                        owner=candidate.owner,
                        label=candidate.label,
                        pose=candidate.pose,
                        radius=candidate.radius,
                        ancestors=set(candidate.ancestors),
                    )
                )
                selected_ancestors[member] = set(candidate.ancestors)
                if candidate.kind in ("object", "operator_base"):
                    selected_ancestors[candidate.owner] = set(candidate.ancestors)
                sampled_poses.update(candidate_poses)
                actions.append(candidate)
                accepted = True
                break
            if accepted:
                continue
            if best is None:
                raise RandomizationFailureError(
                    target=member,
                    attempts=max_attempts,
                    violations=[f"{member}:no_candidate"],
                    minimum_clearance=float("-inf"),
                )
            _, best_pose, best_action, violation = best
            diagnostics = {
                "group": group_name,
                "generator": (
                    group.distribution.generator.value
                    if group is not None
                    else "hard_sphere_rsa"
                ),
                "member": member,
                "attempts": max_attempts,
                "violations": [violation],
                "minimum_clearance": best[0],
                "mode": failure_mode,
            }
            self._last_randomization_diagnostics.setdefault(env_index, []).append(
                diagnostics
            )
            if failure_mode == "error":
                raise RandomizationFailureError(
                    target=member,
                    attempts=max_attempts,
                    violations=diagnostics["violations"],
                    minimum_clearance=best[0],
                )
            working_poses[member] = best_pose
            sampled_poses[member] = best_pose
            best_action.ancestors = set(best_action.ancestors)
            local_participants.append(
                _CollisionParticipant(
                    owner=best_action.owner,
                    label=best_action.label,
                    pose=best_action.pose,
                    radius=best_action.radius,
                    ancestors=set(best_action.ancestors),
                )
            )
            actions.append(best_action)
        return sampled_poses, actions, None

    def _sample_component_for_env(
        self,
        component: List[str],
        env_index: int,
        accepted_sampled_poses: Dict[str, PoseState],
        accepted_participants: List[_CollisionParticipant],
    ) -> tuple[
        Dict[str, PoseState],
        List[_PendingRandomizationAction],
        Optional[tuple[str, str]],
    ]:
        accepted_env_poses = {
            name: pose.select(env_index)
            for name, pose in accepted_sampled_poses.items()
        }
        last_sampled_poses: Dict[str, PoseState] = {}
        last_actions: List[_PendingRandomizationAction] = []
        last_failure: Optional[tuple[str, str]] = None
        last_violations: List[str] = []
        last_minimum_clearance = float("inf")
        best_failure: Optional[
            tuple[
                tuple[int, float],
                Dict[str, PoseState],
                List[_PendingRandomizationAction],
                Optional[tuple[str, str]],
                List[str],
                float,
            ]
        ] = None
        action_specs = self._randomization_action_specs()
        configured_attempts = [
            int(action_specs[label].randomization.failure.max_attempts)
            for label in component
        ]
        attempt_budget = min(
            configured_attempts, default=_MAX_COLLISION_REJECTION_ATTEMPTS
        )
        # Keep the historical module constant useful for existing callers and
        # tests that override the legacy 100-attempt default, while allowing a
        # canonical spec to explicitly request a larger budget.
        if (
            configured_attempts
            and all(value == 100 for value in configured_attempts)
            and _MAX_COLLISION_REJECTION_ATTEMPTS != 100
        ):
            attempt_budget = _MAX_COLLISION_REJECTION_ATTEMPTS
        # Candidate generation owns cross-reset coverage.  The selector only
        # decides how this reset's feasible candidate group is reduced to one
        # sample; it must not decide whether accepted history is consulted.
        history_enabled = any(
            self._distribution_uses_space_filling_history(
                action_specs[label].randomization.distribution
            )
            for label in component
        )
        collect_candidate_group = any(
            action_specs[label].randomization.distribution.selector
            == RandomizationSelectorKind.MAXIMIN
            for label in component
        )
        history_key = tuple(sorted(component))
        history = self._space_filling_history.get(history_key, [])
        history_min_distance = max(
            (
                float(action_specs[label].randomization.distribution.min_distance)
                for label in component
                if self._distribution_uses_space_filling_history(
                    action_specs[label].randomization.distribution
                )
            ),
            default=0.0,
        )
        candidate_limit = max(
            (
                int(action_specs[label].randomization.distribution.candidate_count)
                for label in component
            ),
            default=1,
        )
        valid_candidates: list[
            tuple[Dict[str, PoseState], List[_PendingRandomizationAction], np.ndarray]
        ] = []
        stable_labels = {label: index for index, label in enumerate(sorted(component))}
        for attempt in range(attempt_budget):
            working_poses = dict(accepted_env_poses)
            env_sampled_poses: Dict[str, PoseState] = {}
            env_actions: List[_PendingRandomizationAction] = []
            env_participants: List[_CollisionParticipant] = []
            selected_ancestors: Dict[str, Set[str]] = {}
            failure: Optional[tuple[str, str]] = None
            violations: List[str] = []
            minimum_clearance = float("inf")
            for action_label in component:
                sampled_poses, actions = self._sample_randomization_target_for_env(
                    action_specs[action_label],
                    env_index,
                    working_poses,
                    candidate_index=(
                        self._randomization_reset_index * 1009
                        + attempt * 17
                        + stable_labels[action_label]
                    ),
                )
                for key, pose in sampled_poses.items():
                    working_poses[key] = pose
                    env_sampled_poses[key] = pose
                for action in actions:
                    if not action.references:
                        raise ValueError(
                            f"Sampled action '{action.label}' has no references"
                        )
                    action.ancestors = self._reference_ancestors(
                        action.references,
                        selected_ancestors,
                    )
                    action_ancestors = set(action.ancestors)
                    selected_ancestors[action.label] = action_ancestors
                    if action.kind in ("object", "operator_base"):
                        selected_ancestors[action.owner] = action_ancestors
                    blocking = self._find_collision_participant(
                        owner_name=action.owner,
                        env_index=env_index,
                        candidate_pose=action.pose,
                        collision_radius=action.radius,
                        ancestors=action.ancestors,
                        collision_participants=accepted_participants + env_participants,
                    )
                    env_actions.append(action)
                    env_participants.append(
                        _CollisionParticipant(
                            owner=action.owner,
                            label=action.label,
                            pose=action.pose,
                            radius=action.radius,
                            ancestors=set(action.ancestors),
                        )
                    )
                    if failure is None and blocking is not None:
                        failure = (action.label, blocking.label)
                        violations.append(f"{action.label}:collides:{blocking.label}")
                        candidate_row = 0 if action.pose.batch_size == 1 else env_index
                        other_row = 0 if blocking.pose.batch_size == 1 else env_index
                        candidate_position = np.asarray(
                            action.pose.position[candidate_row], dtype=np.float64
                        )
                        other_position = np.asarray(
                            blocking.pose.position[other_row], dtype=np.float64
                        )
                        minimum_clearance = min(
                            minimum_clearance,
                            float(
                                np.linalg.norm(candidate_position - other_position)
                                - float(action.radius)
                                - self._collision_radius_for_env(
                                    blocking.radius,
                                    env_index,
                                )
                            ),
                        )
            constraint_groups: dict[
                RandomizationConstraintConfig, list[_PendingRandomizationAction]
            ] = {}
            for action in env_actions:
                if action.constraints is None or (
                    action.constraints.visible_in is None
                    and action.constraints.separated is None
                ):
                    continue
                constraint_groups.setdefault(action.constraints, []).append(action)
            for constraints, constrained_actions in constraint_groups.items():
                constrained_names = {action.owner for action in constrained_actions}
                candidate_poses = {
                    name: pose
                    for name, pose in accepted_env_poses.items()
                    if name in self.object_handlers
                }
                candidate_poses.update(
                    {
                        action.owner: action.pose
                        for action in env_actions
                        if action.kind == "object"
                    }
                )
                candidate_ancestors = {
                    participant.owner: self._collision_ancestors_for_env(
                        participant.ancestors,
                        env_index,
                    )
                    for participant in accepted_participants
                    if participant.owner in candidate_poses
                }
                candidate_ancestors.update(
                    {
                        action.owner: set(action.ancestors)
                        for action in env_actions
                        if action.kind == "object" and action.owner in candidate_poses
                    }
                )
                report = self.evaluate_randomization_constraints(
                    candidate_poses,
                    env_index=env_index,
                    constraints=constraints,
                    ancestors=candidate_ancestors,
                    target_names=constrained_names,
                )
                if not report.valid and failure is None:
                    failure = (constrained_actions[0].label, report.violations[0])
                if not report.valid:
                    violations.extend(report.violations)
                    minimum_clearance = min(
                        minimum_clearance,
                        float(report.minimum_clearance),
                    )
            vector: np.ndarray | None = None
            if failure is None and (history_enabled or collect_candidate_group):
                vector = np.concatenate(
                    [
                        np.asarray(
                            next(
                                action.pose
                                for action in env_actions
                                if action.label == label
                            ).position[0],
                            dtype=np.float64,
                        )
                        for label in sorted(component)
                    ]
                )
                if history_enabled:
                    history_clearance = self._history_clearance(
                        vector,
                        history,
                    )
                    duplicate = history_clearance <= 1e-12
                    if duplicate or history_clearance < history_min_distance:
                        failure = (component[0], "history:min_distance")
                        violations.append(f"{component[0]}:history:min_distance")
                        required = max(history_min_distance, 1e-12)
                        minimum_clearance = min(
                            minimum_clearance,
                            history_clearance - required,
                        )

            last_sampled_poses = env_sampled_poses
            last_actions = env_actions
            last_failure = failure
            last_violations = violations
            last_minimum_clearance = minimum_clearance
            if failure is None:
                if not collect_candidate_group:
                    if history_enabled and vector is not None:
                        self._record_space_filling_sample(history_key, vector)
                    return env_sampled_poses, env_actions, None
                if vector is None:
                    raise RuntimeError(
                        "Randomization candidate group was not vectorized"
                    )
                valid_candidates.append((env_sampled_poses, env_actions, vector))
                if len(valid_candidates) >= candidate_limit:
                    break
            else:
                unique_violations = list(dict.fromkeys(violations))
                score = (len(unique_violations), -minimum_clearance)
                if best_failure is None or score < best_failure[0]:
                    best_failure = (
                        score,
                        env_sampled_poses,
                        env_actions,
                        failure,
                        unique_violations,
                        minimum_clearance,
                    )

        if valid_candidates:
            vectors = np.vstack([candidate[2] for candidate in valid_candidates])
            distribution = action_specs[component[0]].randomization.distribution
            selected = maximin_select(
                vectors,
                count=1,
                min_distance=float(distribution.min_distance),
                # History is prepared independently of the selector.  It is
                # the scoring baseline for maximin within this reset's group.
                seed_points=np.vstack(history) if history else None,
            )
            if len(selected) == 0:
                selected = maximin_select(
                    vectors,
                    count=1,
                    min_distance=0.0,
                    seed_points=np.vstack(history) if history else None,
                )
            selected_index = int(
                np.flatnonzero(np.all(np.isclose(vectors, selected[0]), axis=1))[0]
            )
            chosen_poses, chosen_actions, chosen_vector = valid_candidates[
                selected_index
            ]
            if history_enabled:
                self._record_space_filling_sample(history_key, chosen_vector)
            return chosen_poses, chosen_actions, None

        if best_failure is not None:
            (
                _,
                best_poses,
                best_actions,
                best_failure_reason,
                best_violations,
                best_clearance,
            ) = best_failure
            error_actions = [
                action_specs[label]
                for label in component
                if action_specs[label].randomization.failure.mode.value == "error"
            ]
            diagnostics = {
                "labels": list(component),
                "attempts": attempt_budget,
                "violations": best_violations,
                "minimum_clearance": best_clearance,
                "mode": "error" if error_actions else "best_effort",
            }
            self._last_randomization_diagnostics.setdefault(env_index, []).append(
                diagnostics
            )
            if error_actions:
                raise RandomizationFailureError(
                    target=error_actions[0].label,
                    attempts=attempt_budget,
                    violations=diagnostics["violations"],
                    minimum_clearance=best_clearance,
                )
            logger = logging.getLogger(MujocoTaskBackend.__name__)
            logger.warning(
                "Randomization constraints exhausted for '%s' after %d attempts; "
                "applying best-effort candidate. violations=%s minimum_clearance=%s",
                component[0],
                attempt_budget,
                diagnostics["violations"],
                best_clearance,
            )
            if history_enabled:
                best_vector = np.concatenate(
                    [
                        np.asarray(
                            next(
                                action.pose
                                for action in best_actions
                                if action.label == label
                            ).position[0],
                            dtype=np.float64,
                        )
                        for label in sorted(component)
                    ]
                )
                self._record_space_filling_sample(history_key, best_vector)
            return best_poses, best_actions, best_failure_reason
        return last_sampled_poses, last_actions, last_failure

    def _sample_randomization_target_for_env(
        self,
        action_spec: _RandomizationActionSpec,
        env_index: int,
        working_poses: Dict[str, PoseState],
        *,
        candidate_index: int = 0,
    ) -> tuple[Dict[str, PoseState], List[_PendingRandomizationAction]]:
        name = action_spec.owner
        if action_spec.kind == "unknown" or (
            action_spec.kind != "object" and name not in self.operator_handlers
        ):
            logging.getLogger(MujocoTaskBackend.__name__).warning(
                "Randomization key '%s' does not match any object or operator "
                "handler — skipping.",
                name,
            )
            return {}, []

        selected_range = self._select_randomization_region(
            action_spec.randomization,
            candidate_index=candidate_index,
        )
        distribution = action_spec.randomization.distribution
        poisson_stream = self._poisson_stream_for_action(
            action_spec,
            env_index,
            selected_range,
            distribution,
            working_poses,
        )
        if action_spec.kind == "object":
            if RandomizationReference.ABSOLUTE_BASE in selected_range.references():
                raise ValueError(
                    f"Object '{name}' randomization cannot use 'absolute_base' — "
                    "only operator end-effector randomization is defined in a "
                    "base frame."
                )
            sampled = self._sample_object_pose_for_env(
                name,
                selected_range,
                env_index,
                working_poses,
                distribution=distribution,
                sample_index=candidate_index,
                poisson_stream=poisson_stream,
            )
            return {action_spec.label: sampled}, [
                _PendingRandomizationAction(
                    kind=action_spec.kind,
                    owner=name,
                    label=action_spec.label,
                    pose=sampled,
                    radius=float(selected_range.collision_radius),
                    references=selected_range.references(),
                    constraints=action_spec.randomization.constraints,
                )
            ]

        handler = self.operator_handlers[name]
        if action_spec.kind == "operator_base":
            if RandomizationReference.ABSOLUTE_BASE in selected_range.references():
                raise ValueError(
                    f"Operator '{name}' base randomization cannot use "
                    "'absolute_base' — the base IS the frame."
                )
            sampled = self._sample_operator_base_pose_for_env(
                name,
                handler,
                selected_range,
                env_index,
                working_poses,
                distribution=distribution,
                sample_index=candidate_index,
                poisson_stream=poisson_stream,
            )
        elif action_spec.kind == "operator_eef":
            sampled = self._sample_operator_eef_pose_for_env(
                name,
                handler,
                selected_range,
                env_index,
                working_poses,
                distribution=distribution,
                sample_index=candidate_index,
                poisson_stream=poisson_stream,
            )
        else:
            raise ValueError(f"Unknown randomization action kind: {action_spec.kind}")
        return {
            name: sampled,
            action_spec.label: sampled,
        }, [
            _PendingRandomizationAction(
                kind=action_spec.kind,
                owner=name,
                label=action_spec.label,
                pose=sampled,
                radius=float(selected_range.collision_radius),
                references=selected_range.references(),
                constraints=action_spec.randomization.constraints,
            )
        ]

    def _sample_object_pose_for_env(
        self,
        name: str,
        rand_range: PoseRandomRange,
        env_index: int,
        sampled_poses: Dict[str, PoseState],
        *,
        distribution: Any = None,
        sample_index: int = 0,
        poisson_stream: PoissonDiskCandidateStream | None = None,
    ) -> PoseState:
        default_pose = self._default_object_poses.get(
            name,
            self.object_handlers[name].get_pose(),
        ).select(env_index)
        reference_poses = self._resolve_reference_poses_for_env(
            rand_range,
            sampled_poses,
            default_pose,
            env_index,
        )
        return self._sample_random_pose_single(
            default_pose,
            rand_range,
            0,
            reference_poses=reference_poses,
            distribution=distribution,
            sample_index=sample_index,
            poisson_stream=poisson_stream,
        )

    def _sample_operator_base_pose_for_env(
        self,
        name: str,
        handler: MujocoOperatorHandler,
        rand_range: PoseRandomRange,
        env_index: int,
        sampled_poses: Dict[str, PoseState],
        *,
        distribution: Any = None,
        sample_index: int = 0,
        poisson_stream: PoissonDiskCandidateStream | None = None,
    ) -> PoseState:
        default_base = self._default_operator_base_poses.get(
            name,
            handler.get_base_pose(),
        ).select(env_index)
        reference_poses = self._resolve_reference_poses_for_env(
            rand_range,
            sampled_poses,
            default_base,
            env_index,
        )
        return self._sample_random_pose_single(
            default_base,
            rand_range,
            0,
            reference_poses=reference_poses,
            distribution=distribution,
            sample_index=sample_index,
            poisson_stream=poisson_stream,
        )

    def _operator_default_eef_following_base(
        self,
        name: str,
        handler: MujocoOperatorHandler,
        env_index: int,
        sampled_poses: Optional[Dict[str, PoseState]] = None,
    ) -> tuple[PoseState, PoseState]:
        """Return the operator's default EEF pose **rigidly tracking the
        operator's current base**, plus the resolved current base pose.

        ``_record_default_poses`` snapshots the EEF in **world** frame at
        backend init. If the operator's base is later randomized (by this
        sampler or by external tooling), naïvely reusing that world-frame
        default leaves the EEF target sitting at its old absolute position,
        and the IK chain has to bridge a base-induced offset that grows
        with the base randomization range — quickly becoming unreachable
        and surfacing as ``ik_unreachable`` failures even when the EEF
        offset itself is small.

        Re-anchoring to the current base preserves the original eef-in-base
        relative pose, so randomizing the base does not implicitly enlarge
        the EEF reach budget. Sampling logic on top of this helper then
        operates on a default that is already sensible for the current
        base placement.

        ``sampled_poses`` (if given) is consulted for an in-flight base
        sample so eef sampling sees the base that was just decided in the
        same iteration; otherwise the helper falls back to the handler's
        live base pose.
        """
        default_eef_world = self._default_operator_eef_poses.get(
            name,
            handler.get_end_effector_pose(),
        ).select(env_index)
        default_base_world = self._default_operator_base_poses.get(
            name,
            handler.get_base_pose(),
        ).select(env_index)
        current_base_world: Optional[PoseState] = None
        if sampled_poses is not None:
            current_base_world = sampled_poses.get(name)
        if current_base_world is None:
            current_base_world = handler.get_base_pose().select(env_index)
        eef_in_default_base = compose_pose(
            inverse_pose(default_base_world), default_eef_world
        )
        return (
            compose_pose(current_base_world, eef_in_default_base),
            current_base_world,
        )

    def _sample_operator_eef_pose_for_env(
        self,
        name: str,
        handler: MujocoOperatorHandler,
        rand_range: PoseRandomRange,
        env_index: int,
        sampled_poses: Dict[str, PoseState],
        *,
        distribution: Any = None,
        sample_index: int = 0,
        poisson_stream: PoissonDiskCandidateStream | None = None,
    ) -> PoseState:
        following_base_default, base_world = self._operator_default_eef_following_base(
            name,
            handler,
            env_index,
            sampled_poses,
        )
        references = rand_range.references()
        if references == (RandomizationReference.ABSOLUTE_BASE,):
            default_in_base = compose_pose(
                inverse_pose(base_world),
                following_base_default,
            )
            sampled_in_base = self._sample_random_pose_single(
                default_in_base,
                rand_range,
                0,
                distribution=distribution,
                sample_index=sample_index,
                poisson_stream=poisson_stream,
            )
            return compose_pose(base_world, sampled_in_base)

        snapshot_default = self._default_operator_eef_poses.get(
            name,
            handler.get_end_effector_pose(),
        ).select(env_index)
        reference_poses: Dict[Union[RandomizationReference, str], PoseState] = {}
        for reference in references:
            if isinstance(reference, RandomizationReference):
                reference_poses[reference] = following_base_default
            else:
                reference_poses[reference] = self._resolve_reference_base_pose_for_env(
                    reference,
                    sampled_poses,
                    snapshot_default,
                    env_index,
                )
        return self._sample_random_pose_single(
            following_base_default,
            rand_range,
            0,
            reference_poses=reference_poses,
            distribution=distribution,
            sample_index=sample_index,
            poisson_stream=poisson_stream,
        )

    def _resolve_reference_poses_for_env(
        self,
        rand_range: PoseRandomRange,
        sampled_poses: Dict[str, PoseState],
        default_pose: PoseState,
        env_index: int,
    ) -> Dict[Union[RandomizationReference, str], PoseState]:
        """Resolve every reference used by one range to its baseline pose."""
        return {
            reference: self._resolve_reference_base_pose_for_env(
                reference,
                sampled_poses,
                default_pose,
                env_index,
            )
            for reference in rand_range.references()
        }

    def _resolve_reference_base_pose_for_env(
        self,
        reference: Union[RandomizationReference, str],
        sampled_poses: Dict[str, PoseState],
        default_pose: PoseState,
        env_index: int,
    ) -> PoseState:
        if isinstance(reference, RandomizationReference):
            return default_pose
        bare, attr = self._parse_entity_reference(reference)
        if attr is None and bare in self.operator_handlers:
            attr = "base"  # plain operator name defaults to its base
        if attr is not None:
            if bare not in self.operator_handlers:
                raise ValueError(
                    f"Randomization reference '{reference}' — '.{attr}' is only "
                    f"valid for operator names, but '{bare}' is not a known operator."
                )
            handler = self.operator_handlers[bare]
            if attr == "base":
                ref_default = self._default_operator_base_poses.get(
                    bare, handler.get_base_pose()
                ).select(env_index)
            else:
                ref_default = self._default_operator_eef_poses.get(
                    bare, handler.get_end_effector_pose()
                ).select(env_index)
            ref_sampled = sampled_poses.get(f"{bare}.{attr}")
        else:
            ref_sampled = sampled_poses.get(bare)
            if bare in self._default_object_poses:
                ref_default = self._default_object_poses[bare].select(env_index)
            elif bare in self.object_handlers:
                ref_default = self.object_handlers[bare].get_pose().select(env_index)
            else:
                raise ValueError(
                    f"Randomization reference '{reference}' is not a known mode "
                    "('relative', 'absolute_world', 'absolute_base') nor an existing "
                    "object/operator name."
                )
        if ref_sampled is None:
            return default_pose
        delta = compose_pose(ref_sampled, inverse_pose(ref_default))
        return compose_pose(delta, default_pose)

    def _current_pose_for_randomization_key(self, name: str) -> PoseState:
        action = self._randomization_action_specs().get(name)
        if (
            action is None
            or action.kind == "unknown"
            or (action.kind != "object" and action.owner not in self.operator_handlers)
        ):
            return PoseState().broadcast_to(self.batch_size)
        return self._current_pose_for_action(action.kind, action.owner)

    def _current_pose_for_action(self, kind: str, owner: str) -> PoseState:
        if kind == "object":
            return self.object_handlers[owner].get_pose()
        if kind == "operator_base":
            return self.operator_handlers[owner].get_base_pose()
        if kind == "operator_eef":
            return self.operator_handlers[owner].get_end_effector_pose()
        raise ValueError(f"Unknown randomization action kind: {kind}")

    def _sample_random_pose(
        self,
        base_pose: PoseState,
        rand_range: PoseRandomRange,
        env_mask: np.ndarray,
        *,
        distribution: Any = None,
    ) -> PoseState:
        return self._sample_pose_batch(
            base_pose=base_pose,
            env_mask=env_mask,
            sampler=lambda env_index: self._sample_random_pose_single(
                base_pose,
                rand_range,
                env_index,
                distribution=distribution,
                sample_index=self._randomization_reset_index * 1009 + env_index,
            ),
        )

    def _sample_pose_batch(
        self,
        base_pose: PoseState,
        env_mask: np.ndarray,
        sampler: Callable[[int], PoseState],
    ) -> PoseState:
        base_pose = base_pose.broadcast_to(self.batch_size)
        position = base_pose.position.copy()
        orientation = base_pose.orientation.copy()
        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            sampled = sampler(env_index)
            position[env_index] = sampled.position[0]
            orientation[env_index] = sampled.orientation[0]
        return PoseState(position=position, orientation=orientation)

    def _find_collision_participant(
        self,
        *,
        owner_name: str,
        env_index: int,
        candidate_pose: PoseState,
        collision_radius: float,
        ancestors: Set[str],
        collision_participants: List[_CollisionParticipant],
        extra_clearance: float = 0.0,
    ) -> Optional[_CollisionParticipant]:
        candidate_radius = float(collision_radius)
        if candidate_radius <= 0.0:
            return None
        candidate_row = 0 if candidate_pose.batch_size == 1 else env_index
        candidate_pos = np.asarray(
            candidate_pose.position[candidate_row], dtype=np.float64
        )
        for participant in collision_participants:
            participant_radius = self._collision_radius_for_env(
                participant.radius,
                env_index,
            )
            if participant_radius <= 0.0:
                continue
            if participant.owner == owner_name:
                continue
            participant_ancestors = self._collision_ancestors_for_env(
                participant.ancestors,
                env_index,
            )
            if participant.owner in ancestors or owner_name in participant_ancestors:
                continue
            other_row = 0 if participant.pose.batch_size == 1 else env_index
            other_pos = np.asarray(
                participant.pose.position[other_row],
                dtype=np.float64,
            )
            if np.linalg.norm(
                candidate_pos - other_pos
            ) < candidate_radius + participant_radius + float(extra_clearance):
                return participant
        return None

    @staticmethod
    def _collision_radius_for_env(
        radius: float | np.ndarray,
        env_index: int,
    ) -> float:
        """Resolve a scalar or batched collision radius for one environment."""
        if isinstance(radius, np.ndarray):
            values = np.asarray(radius, dtype=np.float64).reshape(-1)
            if values.size == 0:
                return 0.0
            if values.size == 1:
                return float(values[0])
            return float(values[env_index])
        return float(radius)

    @staticmethod
    def _collision_ancestors_for_env(
        ancestors: _RandomizationAncestors,
        env_index: int,
    ) -> Set[str]:
        """Resolve scalar or batched reference ancestors for one environment."""
        if isinstance(ancestors, list):
            if not ancestors:
                return set()
            if len(ancestors) == 1:
                return ancestors[0]
            return ancestors[env_index]
        return ancestors

    def _sample_random_pose_single(
        self,
        base_pose: PoseState,
        rand_range: PoseRandomRange,
        env_index: int,
        *,
        reference_poses: Optional[
            Mapping[Union[RandomizationReference, str], PoseState]
        ] = None,
        distribution: Any = None,
        sample_index: int = 0,
        poisson_stream: PoissonDiskCandidateStream | None = None,
    ) -> PoseState:
        base_pose = base_pose.broadcast_to(self.batch_size)
        pose_by_reference = {
            reference: pose.broadcast_to(self.batch_size)
            for reference, pose in (reference_poses or {}).items()
        }

        def _baseline(reference: Union[RandomizationReference, str]) -> PoseState:
            return pose_by_reference.get(reference, base_pose)

        generator = getattr(
            distribution,
            "generator",
            RandomizationGeneratorKind.IID,
        )
        is_poisson_generator = generator == RandomizationGeneratorKind.POISSON_DISK or (
            isinstance(generator, RandomizationGeneratorConfig)
        )
        candidate_count = int(getattr(distribution, "candidate_count", 1))
        poisson_position_axes = (
            tuple(
                axis
                for axis in ("x", "y", "z")
                if rand_range.axis_range(axis) is not None
            )
            if poisson_stream is not None
            else ()
        )
        poisson_values = poisson_stream.next() if poisson_stream is not None else None
        qmc_values = None
        if generator != RandomizationGeneratorKind.IID:
            orientation_generator = (
                RandomizationGeneratorKind.SOBOL if is_poisson_generator else generator
            )
            qmc_values = unit_candidate(
                rng=self._rng,
                dimension=3 if is_poisson_generator else 6,
                generator=orientation_generator,
                index=sample_index,
                candidate_count=candidate_count,
                seed=int(self._randomization_reset_index * 1_000_003 + sample_index),
            )

        position = np.empty(3, dtype=np.float64)
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            reference = rand_range.axis_reference(axis_name)
            baseline = _baseline(reference)
            value = float(baseline.position[env_index, axis_index])
            rng_pair = rand_range.axis_range(axis_name)
            if rng_pair is not None:
                if poisson_values is not None:
                    sampled = float(
                        poisson_values[poisson_position_axes.index(axis_name)]
                    )
                elif qmc_values is None:
                    sampled = float(self._rng.uniform(*rng_pair))
                else:
                    sampled = float(
                        rng_pair[0]
                        + qmc_values[axis_index] * (rng_pair[1] - rng_pair[0])
                    )
                if reference in (
                    RandomizationReference.ABSOLUTE_WORLD,
                    RandomizationReference.ABSOLUTE_BASE,
                ):
                    value = sampled
                else:
                    value += sampled
            position[axis_index] = value

        rotation_axes = ("roll", "pitch", "yaw")
        rotation_references = tuple(
            rand_range.axis_reference(axis_name) for axis_name in rotation_axes
        )
        if (
            all(rand_range.axis_range(axis_name) is None for axis_name in rotation_axes)
            and len(set(rotation_references)) == 1
        ):
            orientation = np.asarray(
                _baseline(rotation_references[0]).orientation[env_index],
                dtype=np.float64,
            ).copy()
            return PoseState(position=position, orientation=orientation)

        rotation = np.empty(3, dtype=np.float64)
        for axis_index, axis_name in enumerate(rotation_axes):
            reference = rand_range.axis_reference(axis_name)
            baseline = _baseline(reference)
            baseline_rpy = quaternion_to_rpy(baseline.orientation[env_index])
            value = float(baseline_rpy[axis_index])
            rng_pair = rand_range.axis_range(axis_name)
            if rng_pair is not None:
                if qmc_values is None:
                    sampled = float(self._rng.uniform(*rng_pair))
                else:
                    sampled = float(
                        rng_pair[0]
                        + qmc_values[
                            axis_index if is_poisson_generator else 3 + axis_index
                        ]
                        * (rng_pair[1] - rng_pair[0])
                    )
                if reference in (
                    RandomizationReference.ABSOLUTE_WORLD,
                    RandomizationReference.ABSOLUTE_BASE,
                ):
                    value = sampled
                else:
                    value += sampled
            rotation[axis_index] = value
        orientation = np.asarray(
            euler_to_quaternion(tuple(rotation)),
            dtype=np.float64,
        )
        return PoseState(position=position, orientation=orientation)

    # ------------------------------------------------------------------
    #  Camera randomization
    # ------------------------------------------------------------------

    def _get_camera_pose(self, cam_name: str) -> PoseState:
        """Read a camera's world pose across all envs.

        MuJoCo stores ``cam_pos``/``cam_quat`` in the attached body's local
        frame.  The task-level pose contract is world-frame, so use the
        derived ``cam_xpos``/``cam_xmat`` values here.  This matters for wrist
        cameras and any camera mounted below an articulated scene body.
        """
        positions = np.zeros((self.batch_size, 3), dtype=np.float64)
        orientations = np.zeros((self.batch_size, 4), dtype=np.float64)
        for env_index, single_env in enumerate(self.env.envs):
            cam_id = mujoco.mj_name2id(
                single_env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
            )
            if cam_id < 0:
                raise KeyError(f"Camera '{cam_name}' not found in the MuJoCo model.")
            mujoco.mj_forward(single_env.model, single_env.data)
            positions[env_index] = np.asarray(
                single_env.data.cam_xpos[cam_id], dtype=np.float64
            )
            orientations[env_index] = quaternion_from_matrix_3x3(
                np.asarray(single_env.data.cam_xmat[cam_id], dtype=np.float64).reshape(
                    3, 3
                )
            )
        return PoseState(position=positions, orientation=orientations)

    def _set_camera_pose(
        self,
        cam_name: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None:
        """Write world-frame camera poses as parent-local MuJoCo extrinsics."""
        pose = pose.broadcast_to(self.batch_size)
        env_indices = _stateful_pose_indices(
            self.env,
            pose,
            env_mask,
            label=f"Camera '{cam_name}' pose",
        )
        for env_index in env_indices:
            single_env = self.env.envs[env_index]
            cam_id = mujoco.mj_name2id(
                single_env.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name
            )
            if cam_id < 0:
                continue
            model = single_env.model
            data = single_env.data
            mujoco.mj_forward(model, data)
            parent_id = int(model.cam_bodyid[cam_id])
            parent_pos = np.asarray(data.xpos[parent_id], dtype=np.float64)
            parent_rot = np.asarray(data.xmat[parent_id], dtype=np.float64).reshape(
                3, 3
            )
            world_pos = np.asarray(pose.position[env_index], dtype=np.float64)
            model.cam_pos[cam_id] = parent_rot.T @ (world_pos - parent_pos)

            qx, qy, qz, qw = pose.orientation[env_index]
            world_quat_wxyz = np.asarray([qw, qx, qy, qz], dtype=np.float64)
            parent_quat_wxyz = np.asarray(data.xquat[parent_id], dtype=np.float64)
            inverse_parent_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_negQuat(inverse_parent_quat, parent_quat_wxyz)
            local_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mulQuat(local_quat, inverse_parent_quat, world_quat_wxyz)
            model.cam_quat[cam_id] = local_quat
            mujoco.mj_forward(single_env.model, single_env.data)

    def _apply_camera_randomization(self, env_mask: np.ndarray) -> None:
        """Sample and apply pose randomization for configured cameras."""
        for cam_name, randomization in self.camera_randomization.items():
            canonical = canonical_randomization_spec(randomization)
            rand_range = self._select_randomization_region(canonical)
            for reference in rand_range.references():
                if reference == RandomizationReference.ABSOLUTE_BASE:
                    raise ValueError(
                        f"Camera '{cam_name}' randomization cannot use "
                        "'absolute_base' — cameras have no operator base frame."
                    )
                if isinstance(reference, str) and not isinstance(
                    reference,
                    RandomizationReference,
                ):
                    raise ValueError(
                        f"Camera '{cam_name}' randomization cannot use entity "
                        f"reference '{reference}' — cameras do not participate in "
                        "entity dependency ordering."
                    )
            default_pose = self._default_camera_poses.get(cam_name)
            if default_pose is None:
                logging.getLogger(MujocoTaskBackend.__name__).warning(
                    "Camera '%s' has no recorded default pose — skipping "
                    "randomization.",
                    cam_name,
                )
                continue
            sampled = self._sample_random_pose(
                default_pose,
                rand_range,
                env_mask,
                distribution=canonical.distribution,
            )
            self._set_camera_pose(cam_name, sampled, env_mask)

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
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            return PoseState(
                position=data.geom_xpos[gid],
                orientation=quaternion_from_matrix_3x3(
                    data.geom_xmat[gid].reshape(3, 3)
                ),
            )
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            # MuJoCo publishes the articulated joint anchor and axis in world
            # coordinates after ``mj_forward``.  Using the parent body's
            # static transform here loses both the joint anchor offset and
            # the current hinge/ball/free-joint orientation.
            joint_bid = int(model.jnt_bodyid[jid])
            world_pos = np.asarray(data.xanchor[jid], dtype=np.float64)
            joint_rot = np.asarray(data.xmat[joint_bid], dtype=np.float64).reshape(3, 3)
            return PoseState(
                position=world_pos,
                orientation=quaternion_from_matrix_3x3(joint_rot),
            )
        raise KeyError(
            f"No site, body, geom, or joint named '{name}' found in the MuJoCo model."
        )

    def is_element_rigidly_attached_to_object(
        self,
        element_name: str,
        object_name: str,
        env_index: int = 0,
    ) -> bool:
        single_env = self.env.envs[env_index]
        model = single_env.model
        object_handler = self.get_object_handler(object_name)
        if object_handler is None:
            raise KeyError(f"Unknown object {object_name!r}.")
        object_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_handler.body_name,
        )
        if object_body_id < 0:
            raise KeyError(
                f"Object {object_name!r} refers to missing body "
                f"{object_handler.body_name!r}."
            )
        element_body_id = self._named_element_body_id(model, element_name)

        current_body_id = element_body_id
        while current_body_id != object_body_id:
            if current_body_id <= 0:
                return False
            # A joint on a descendant body makes its frame movable relative to
            # the object's root even though it remains in the same subtree.
            if int(model.body_jntnum[current_body_id]) > 0:
                return False
            current_body_id = int(model.body_parentid[current_body_id])
        return True

    @staticmethod
    def _named_element_body_id(model: Any, name: str) -> int:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return int(model.site_bodyid[site_id])
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            return int(model.geom_bodyid[geom_id])
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            joint_body_id = int(model.jnt_bodyid[joint_id])
            return int(model.body_parentid[joint_body_id])
        raise KeyError(
            f"No site, body, geom, or joint named {name!r} found in the MuJoCo model."
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

    def get_grasped_object_name(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[str]:
        if not 0 <= env_index < self.batch_size:
            raise IndexError(
                f"env_index must be in [0, {self.batch_size}), got {env_index}"
            )
        operator = self.get_operator_handler(operator_name)
        for object_name, target in self.object_handlers.items():
            if operator._is_target_grasped(env_index, target):
                return object_name
        return None

    def is_operator_contacting(
        self, operator_name: str, object_name: str
    ) -> np.ndarray:
        target = self.get_object_handler(object_name)
        if target is None:
            return np.zeros(self.batch_size, dtype=bool)
        result = np.zeros(self.batch_size, dtype=bool)
        for env_index, single_env in enumerate(self.env.envs):
            target_bodies = target.get_descendant_body_ids(single_env.model)
            if not target_bodies:
                continue
            for _, _, _, _, other_body in self._iter_operator_external_contacts(
                operator_name,
                env_index,
            ):
                if other_body in target_bodies:
                    result[env_index] = True
                    break
        return result

    def get_operator_contacts(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[List[ContactObservation]]:
        observations: List[ContactObservation] = []
        single_env = self.env.envs[env_index]
        model = single_env.model
        data = single_env.data
        for (
            contact_index,
            operator_geom,
            operator_body,
            other_geom,
            other_body,
        ) in self._iter_operator_external_contacts(operator_name, env_index):
            contact = data.contact[contact_index]
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, force)
            observations.append(
                ContactObservation(
                    operator_body=_mujoco_element_name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        operator_body,
                        "body",
                    ),
                    operator_geom=_mujoco_element_name(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        operator_geom,
                        "geom",
                    ),
                    other_body=_mujoco_element_name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        other_body,
                        "body",
                    ),
                    other_geom=_mujoco_element_name(
                        model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        other_geom,
                        "geom",
                    ),
                    position_world_m=tuple(float(value) for value in contact.pos),
                    signed_distance_m=float(contact.dist),
                    penetration_depth_m=max(0.0, -float(contact.dist)),
                    normal_force_n=abs(float(force[0])),
                    tangential_force_n=float(np.linalg.norm(force[1:3])),
                )
            )
        observations.sort(
            key=lambda item: (
                item.operator_body,
                item.operator_geom,
                item.other_body,
                item.other_geom,
                item.signed_distance_m,
            )
        )
        return observations

    def _iter_operator_external_contacts(
        self,
        operator_name: str,
        env_index: int,
    ) -> Iterator[Tuple[int, int, int, int, int]]:
        if not 0 <= env_index < self.batch_size:
            raise IndexError(
                f"env_index must be in [0, {self.batch_size}), got {env_index}"
            )
        operator = self.get_operator_handler(operator_name)
        single_env = self.env.envs[env_index]
        model = single_env.model
        data = single_env.data
        operator_bodies = operator.get_operator_body_ids(model)
        geom_bodyid = model.geom_bodyid
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if int(contact.efc_address) < 0:
                # MuJoCo keeps contacts inside the broad collision margin in
                # ``data.contact`` even when ``gap`` leaves their constraint
                # inactive.  They are proximity candidates, not physical
                # contacts, and must not satisfy CONTACTED or enter failure
                # diagnostics as zero-force collisions.
                continue
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(geom_bodyid[geom1])
            body2 = int(geom_bodyid[geom2])
            body1_is_operator = body1 in operator_bodies
            body2_is_operator = body2 in operator_bodies
            if body1_is_operator == body2_is_operator:
                continue
            if body1_is_operator:
                yield contact_index, geom1, body1, geom2, body2
            else:
                yield contact_index, geom2, body2, geom1, body1

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


def build_mujoco_backend(
    task: AutoAtomConfig | Dict[str, Any],
    operators: Dict[str, OperatorConfig],
    ik_solver: Optional[IKSolver] = None,
    handler_kwargs: Optional[Dict[str, Any]] = None,
) -> MujocoTaskBackend:
    config = (
        task
        if isinstance(task, AutoAtomConfig)
        else AutoAtomConfig.model_validate(task)
    )
    operator_configs = list(operators.values())
    env = ComponentRegistry.get_env(config.env_name)
    if not isinstance(env, BatchedUnifiedMujocoEnv):
        raise TypeError(
            f"Registered environment '{config.env_name}' must be a BatchedUnifiedMujocoEnv, got {type(env).__name__}."
        )

    extra = handler_kwargs or {}
    operator_handlers: Dict[str, MujocoOperatorHandler] = {}
    env_op_bindings = getattr(env.envs[0], "_operators", {}) if env.envs else {}
    for operator in operator_configs:
        op_extra = operator.model_extra or {}
        ik_extra = op_extra.get("ik") or {}
        control_extra = op_extra.get("control") or {}
        control_cfg = MujocoControlConfig.model_validate(control_extra)
        # Inherit per-operator body/site/actuator names from the env's
        # OperatorBinding. This keeps the handler in sync with the env
        # (which already auto-registered the operator using these names).
        # Explicit handler_kwargs still take precedence.
        binding = env_op_bindings.get(operator.name)
        binding_defaults: Dict[str, Any] = {}
        if binding is not None:
            if binding.root_body:
                binding_defaults["root_body_name"] = binding.root_body
            if binding.pose_site:
                binding_defaults["eef_site_name"] = binding.pose_site
            if binding.mocap_body:
                binding_defaults["mocap_body_name"] = binding.mocap_body
            if binding.freejoint:
                binding_defaults["freejoint_name"] = binding.freejoint
        # Auto-detect the EEF actuator index and its ctrlrange so open/close
        # targets match the physical actuator limits (robotiq's 0/0.82 doesn't
        # carry over to, e.g., XF9600 which uses 0/0.02).
        eef_aidx = env.envs[0]._op_eef_aidx.get(operator.name, np.array([]))
        if len(eef_aidx) > 0:
            ctrl_idx = int(eef_aidx[0])
            binding_defaults.setdefault("eef_ctrl_index", ctrl_idx)
            low, high = env.envs[0].model.actuator_ctrlrange[ctrl_idx]
            binding_defaults.setdefault("eef_open_value", float(low))
            binding_defaults.setdefault("eef_close_value", float(high))
            # Clamp the eef tolerance so it's meaningful for narrow-travel
            # grippers (XF9600 has close=0.02; the default 0.03 tolerance would
            # treat a fully-open gripper as "reached" on the first step and
            # bypass the actual grasp).
            ctrl_span = float(high) - float(low)
            if ctrl_span > 0:
                control_extra = dict(control_extra)
                tol_block = control_extra.setdefault("tolerance", {})
                if isinstance(tol_block, Mapping) and "eef" not in tol_block:
                    tol_block = dict(tol_block)
                    tol_block["eef"] = min(0.03, max(1e-4, ctrl_span * 0.2))
                    control_extra["tolerance"] = tol_block
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
                **binding_defaults,
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
            },
        )

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
                refs.extend(_randomization_references(rand_range.base))
            if rand_range.eef is not None:
                refs.extend(_randomization_references(rand_range.eef))
        else:
            refs.extend(_randomization_references(rand_range))
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

    backend = MujocoTaskBackend(
        env=env,
        operator_handlers=operator_handlers,
        object_handlers=object_handlers,
        randomization=dict(config.randomization),
        randomization_strategy=config.randomization_strategy,
        camera_randomization=dict(config.camera_randomization),
        initial_poses=dict(config.initial_pose),
        camera_initial_poses=dict(config.camera_initial_pose),
        operator_initial_states={
            operator.name: operator.initial_state
            for operator in operator_configs
            if operator.initial_state is not None
        },
        random_seed=config.seed if config.seed != 0 else None,
        randomization_debug=config.randomization_debug,
    )
    return backend
