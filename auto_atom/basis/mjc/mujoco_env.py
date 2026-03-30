"""High-level MuJoCo environment with pose control and observation capture.

``UnifiedMujocoEnv`` extends :class:`MujocoBasis` with:

* ``step(action)`` — apply control inputs and advance physics.
* ``capture_observation()`` — collect all sensor data into a nested dict.
* Operator pose interfaces — register, query, and control operator poses.
  All frame conversions, IK solving, and MuJoCo data writes are handled
  here so that upper layers never duplicate these operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING
import numpy as np
import mujoco
from auto_atom.utils.transformations import (
    euler_from_matrix,
    quaternion_inverse,
    quaternion_matrix,
    quaternion_multiply,
)
from auto_atom.utils.pose import PoseState
from auto_atom.basis.mjc.mujoco_basis import (
    DataType,
    CameraSpec,
    ViewerConfig,
    OperatorBinding,
    EnvConfig,
    MujocoBasis,
)

if TYPE_CHECKING:
    from auto_atom.runtime import IKSolver

# Re-export config classes so existing imports and Hydra _target_ references work.
__all__ = [
    "DataType",
    "CameraSpec",
    "ViewerConfig",
    "OperatorBinding",
    "EnvConfig",
    "UnifiedMujocoEnv",
    "BatchedUnifiedMujocoEnv",
]


@dataclass
class _OperatorState:
    """Per-operator control state, created by :meth:`UnifiedMujocoEnv.register_operator`."""

    joint_mode: bool
    ik_solver: Optional["IKSolver"]
    eef_site_name: str

    # Base frame (world, fixed for the episode).
    base_pos: np.ndarray  # float32, shape (3,)
    base_quat: np.ndarray  # float32, xyzw, shape (4,)
    # Fixed EEF offset in base frame (base_T_eef).
    tool_offset_pos: np.ndarray  # float32, shape (3,)
    tool_offset_quat: np.ndarray  # float32, xyzw, shape (4,)

    # Home state.
    home_arm_qpos: Optional[np.ndarray]  # joint mode only
    home_mocap_pos: np.ndarray  # float64, wxyz for mocap
    home_mocap_quat: np.ndarray  # float64, wxyz for mocap
    home_ctrl: np.ndarray  # float64

    # Mocap mode IDs (-1 when joint mode).
    mocap_id: int = -1
    fj_qpos_adr: int = 0
    fj_dof_adr: int = 0

    # Observable target pose in base frame (updated on every control step).
    target_pos_in_base: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    target_quat_in_base: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 0, 1], dtype=np.float32)
    )

    # Joint-mode execution strategy.
    joint_control_mode: str = "per_step_ik"
    joint_interp_speed: float = 0.05
    planned_joint_start_qpos: Optional[np.ndarray] = None
    planned_joint_target_qpos: Optional[np.ndarray] = None
    planned_joint_progress: int = 0
    planned_joint_steps_total: int = 1
    planned_target_pos_in_base: Optional[np.ndarray] = None
    planned_target_quat_in_base: Optional[np.ndarray] = None


class UnifiedMujocoEnv(MujocoBasis):
    """MuJoCo environment with operator pose control and observation capture."""

    def __init__(self, config: EnvConfig):
        super().__init__(config)
        self._operator_states: dict[str, _OperatorState] = {}

    # ==================================================================
    # Operator registration
    # ==================================================================

    def register_operator(
        self,
        op_name: str,
        *,
        root_body: str,
        eef_site: str,
        ik_solver: Optional["IKSolver"] = None,
        mocap_body: str = "",
        freejoint: str = "",
        joint_control_mode: str = "per_step_ik",
        joint_interp_speed: float = 0.05,
    ) -> None:
        """Register an operator and snapshot its home state.

        After this call the env owns all frame-conversion data and
        MuJoCo IDs needed to control the operator.  The handler no
        longer needs to cache ``_base_pose`` or write to ``data.*``
        directly.
        """
        arm_aidx = self._op_arm_aidx.get(op_name, np.array([]))
        joint_mode = len(arm_aidx) > 0

        if joint_mode and ik_solver is None:
            raise ValueError(
                f"Operator '{op_name}' has arm_actuators but no ik_solver was provided."
            )

        # --- Base pose (world frame) ---
        # Joint-mode operators use the physical robot base. Pure mocap operators
        # expose a virtual base frame whose default is the world origin.
        physical_base_pos, physical_base_quat = self.get_body_pose(root_body)
        physical_base_pos = physical_base_pos.astype(np.float32)
        physical_base_quat = physical_base_quat.astype(np.float32)
        if joint_mode:
            base_pos = physical_base_pos
            base_quat = physical_base_quat
        else:
            base_pos = np.zeros(3, dtype=np.float32)
            base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # --- EEF pose → tool offset (base_T_eef) ---
        eef_pos_w, eef_quat_w = self.get_site_pose(eef_site)
        tool_base_pos = physical_base_pos if not joint_mode else base_pos
        tool_base_quat = physical_base_quat if not joint_mode else base_quat
        tool_pos, tool_quat = self._world_to_base(
            eef_pos_w, eef_quat_w, tool_base_pos, tool_base_quat
        )

        # --- Home state ---
        home_arm_qpos: Optional[np.ndarray] = None
        home_mocap_pos = np.zeros(3, dtype=np.float64)
        home_mocap_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        mocap_id = -1
        fj_qpos_adr = 0
        fj_dof_adr = 0

        if joint_mode:
            arm_qidx = self._op_arm_qidx[op_name]
            home_arm_qpos = self.data.qpos[arm_qidx].copy()
        else:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, mocap_body
            )
            if body_id < 0:
                raise ValueError(f"Mocap body '{mocap_body}' not found.")
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(
                    f"Body '{mocap_body}' is not a mocap body "
                    f"(body_mocapid={mocap_id})."
                )
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, freejoint)
            if jid < 0:
                raise ValueError(f"Freejoint '{freejoint}' not found.")
            fj_qpos_adr = int(self.model.jnt_qposadr[jid])
            fj_dof_adr = int(self.model.jnt_dofadr[jid])
            home_mocap_pos = self.data.mocap_pos[mocap_id].copy()
            home_mocap_quat = self.data.mocap_quat[mocap_id].copy()

        home_ctrl = np.asarray(self.data.ctrl[: self.model.nu], dtype=np.float64).copy()

        control_mode = str(joint_control_mode)
        if control_mode not in {"per_step_ik", "solve_once_interpolate"}:
            raise ValueError(
                f"Unsupported joint_control_mode '{joint_control_mode}' for operator '{op_name}'."
            )
        interp_speed = float(joint_interp_speed)
        if interp_speed <= 0.0:
            raise ValueError(
                f"joint_interp_speed must be > 0 for operator '{op_name}', got {joint_interp_speed}."
            )

        state = _OperatorState(
            joint_mode=joint_mode,
            ik_solver=ik_solver,
            eef_site_name=eef_site,
            base_pos=base_pos,
            base_quat=base_quat,
            tool_offset_pos=tool_pos,
            tool_offset_quat=tool_quat,
            home_arm_qpos=home_arm_qpos,
            home_mocap_pos=home_mocap_pos,
            home_mocap_quat=home_mocap_quat,
            home_ctrl=home_ctrl,
            mocap_id=mocap_id,
            fj_qpos_adr=fj_qpos_adr,
            fj_dof_adr=fj_dof_adr,
            target_pos_in_base=tool_pos.copy(),
            target_quat_in_base=tool_quat.copy(),
            joint_control_mode=control_mode,
            joint_interp_speed=interp_speed,
            planned_joint_start_qpos=home_arm_qpos.copy()
            if home_arm_qpos is not None
            else None,
            planned_joint_target_qpos=home_arm_qpos.copy()
            if home_arm_qpos is not None
            else None,
            planned_joint_steps_total=1,
            planned_target_pos_in_base=tool_pos.copy(),
            planned_target_quat_in_base=tool_quat.copy(),
        )
        self._operator_states[op_name] = state

    def _get_op(self, op_name: str) -> _OperatorState:
        try:
            return self._operator_states[op_name]
        except KeyError:
            raise ValueError(
                f"Operator '{op_name}' not registered. Call register_operator first."
            )

    # ==================================================================
    # Frame conversion
    # ==================================================================

    def world_to_base(
        self, op_name: str, pos_w: np.ndarray, quat_w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """World frame → operator base frame."""
        s = self._get_op(op_name)
        return self._world_to_base(pos_w, quat_w, s.base_pos, s.base_quat)

    def base_to_world(
        self, op_name: str, pos_b: np.ndarray, quat_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Operator base frame → world frame."""
        s = self._get_op(op_name)
        return self._base_to_world(pos_b, quat_b, s.base_pos, s.base_quat)

    @staticmethod
    def _world_to_base(
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World → base (all quaternions xyzw)."""
        inv_quat = quaternion_inverse(base_quat).astype(np.float32)
        inv_rot = quaternion_matrix(inv_quat)[:3, :3]
        pos = (inv_rot @ (np.asarray(pos_w, dtype=np.float32) - base_pos)).astype(
            np.float32
        )
        quat = quaternion_multiply(
            inv_quat, np.asarray(quat_w, dtype=np.float32)
        ).astype(np.float32)
        return pos, quat

    @staticmethod
    def _base_to_world(
        pos_b: np.ndarray,
        quat_b: np.ndarray,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Base → world (all quaternions xyzw)."""
        rot = quaternion_matrix(base_quat)[:3, :3]
        pos = (rot @ np.asarray(pos_b, dtype=np.float32) + base_pos).astype(np.float32)
        quat = quaternion_multiply(
            np.asarray(base_quat, dtype=np.float32),
            np.asarray(quat_b, dtype=np.float32),
        ).astype(np.float32)
        return pos, quat

    # ==================================================================
    # EEF pose queries
    # ==================================================================

    def get_operator_eef_pose_in_base(
        self, op_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the current EEF pose in the operator's base frame."""
        s = self._get_op(op_name)
        pos_w, quat_w = self.get_site_pose(s.eef_site_name)
        return self._world_to_base(pos_w, quat_w, s.base_pos, s.base_quat)

    def get_operator_eef_pose_in_world(
        self, op_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the current EEF pose in world frame."""
        s = self._get_op(op_name)
        return self.get_site_pose(s.eef_site_name)

    def get_operator_base_pose(self, op_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored base pose in world frame."""
        s = self._get_op(op_name)
        return s.base_pos, s.base_quat

    def override_operator_base_pose(
        self,
        op_name: str,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
    ) -> None:
        """Override the stored operator base frame and refresh cached offsets.

        This is intended for setup-time configuration such as
        ``initial_state.base_pose``.  The MuJoCo model state is not mutated.
        """
        s = self._get_op(op_name)
        s.base_pos = np.asarray(pos_w, dtype=np.float32)
        s.base_quat = np.asarray(quat_w, dtype=np.float32)
        if not s.joint_mode:
            eef_pos_w, eef_quat_w = self.get_site_pose(s.eef_site_name)
            target_pos, target_quat = self._world_to_base(
                eef_pos_w, eef_quat_w, s.base_pos, s.base_quat
            )
            s.target_pos_in_base = target_pos
            s.target_quat_in_base = target_quat
            if s.planned_target_pos_in_base is not None:
                s.planned_target_pos_in_base = target_pos.copy()
            if s.planned_target_quat_in_base is not None:
                s.planned_target_quat_in_base = target_quat.copy()
            return
        eef_pos_w, eef_quat_w = self.get_site_pose(s.eef_site_name)
        tool_pos, tool_quat = self._world_to_base(
            eef_pos_w, eef_quat_w, s.base_pos, s.base_quat
        )
        s.tool_offset_pos = tool_pos
        s.tool_offset_quat = tool_quat
        s.target_pos_in_base = tool_pos.copy()
        s.target_quat_in_base = tool_quat.copy()

    def set_operator_base_pose(
        self,
        op_name: str,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
    ) -> None:
        """Set the operator base pose with consistent semantics across backends.

        Joint-mode operators only update the virtual base frame because the
        physical robot base is fixed in the model. Pure mocap operators also
        move the physical body rigidly so the existing base->EEF tool offset is
        preserved under the new base frame.
        """
        s = self._get_op(op_name)
        if s.joint_mode:
            self.override_operator_base_pose(op_name, pos_w, quat_w)
            return

        s.base_pos = np.asarray(pos_w, dtype=np.float32)
        s.base_quat = np.asarray(quat_w, dtype=np.float32)
        base_body_pos, base_body_quat_xyzw = self._eef_in_base_to_base_body_world(
            s,
            s.tool_offset_pos,
            s.tool_offset_quat,
        )
        self._write_mocap_pose(
            s,
            base_body_pos,
            np.asarray(base_body_quat_xyzw, dtype=np.float32),
            sync_freejoint=True,
        )
        mujoco.mj_forward(self.model, self.data)
        s.target_pos_in_base = s.tool_offset_pos.copy()
        s.target_quat_in_base = s.tool_offset_quat.copy()

    # ==================================================================
    # Pose control (actual physics interaction)
    # ==================================================================

    def step_operator_toward_target(
        self, op_name: str, target_pos_b: np.ndarray, target_quat_b: np.ndarray
    ) -> None:
        """Advance one control step toward the target EEF pose (base frame).

        Joint mode: solve IK every step from current qpos. The solver's
        ``max_joint_delta`` clamp limits per-step joint displacement, so
        the arm smoothly converges without branch jumps.
        Mocap mode: convert to world base-body pose → write mocap → update.
        """
        s = self._get_op(op_name)
        new_pos = np.asarray(target_pos_b, dtype=np.float32)
        new_quat = np.asarray(target_quat_b, dtype=np.float32)
        s.target_pos_in_base = new_pos
        s.target_quat_in_base = new_quat

        if s.joint_mode:
            eef_in_base = PoseState(
                position=tuple(float(v) for v in new_pos),
                orientation=tuple(float(v) for v in new_quat),
            )
            arm_qidx = self._op_arm_qidx[op_name]
            current_arm_qpos = self.data.qpos[arm_qidx].copy()

            if s.joint_control_mode == "solve_once_interpolate":
                target_changed = (
                    s.planned_target_pos_in_base is None
                    or not np.allclose(
                        new_pos,
                        s.planned_target_pos_in_base,
                        atol=1e-6,
                        rtol=0.0,
                    )
                    or s.planned_target_quat_in_base is None
                    or abs(
                        float(
                            np.dot(
                                new_quat.astype(np.float64),
                                s.planned_target_quat_in_base.astype(np.float64),
                            )
                        )
                    )
                    < (1.0 - 1e-6)
                )
                need_plan = (
                    target_changed
                    or s.planned_joint_target_qpos is None
                    or s.planned_joint_start_qpos is None
                    or s.planned_joint_progress >= s.planned_joint_steps_total
                )

                if need_plan:
                    joint_targets = s.ik_solver.solve(eef_in_base, current_arm_qpos)
                    if joint_targets is None:
                        self.update()
                        return
                    s.planned_joint_start_qpos = current_arm_qpos.copy()
                    s.planned_joint_target_qpos = np.asarray(
                        joint_targets, dtype=np.float64
                    ).copy()
                    s.planned_joint_progress = 0
                    max_abs_delta = float(
                        np.max(
                            np.abs(
                                s.planned_joint_target_qpos - s.planned_joint_start_qpos
                            )
                        )
                    )
                    s.planned_joint_steps_total = max(
                        1, int(np.ceil(max_abs_delta / s.joint_interp_speed))
                    )
                    s.planned_target_pos_in_base = new_pos.copy()
                    s.planned_target_quat_in_base = new_quat.copy()

                start_qpos = np.asarray(s.planned_joint_start_qpos, dtype=np.float64)
                final_qpos = np.asarray(s.planned_joint_target_qpos, dtype=np.float64)
                s.planned_joint_progress = min(
                    s.planned_joint_progress + 1, s.planned_joint_steps_total
                )
                alpha = float(s.planned_joint_progress) / float(
                    s.planned_joint_steps_total
                )
                joint_targets = (1.0 - alpha) * start_qpos + alpha * final_qpos
            else:
                joint_targets = s.ik_solver.solve(eef_in_base, current_arm_qpos)
                if joint_targets is None:
                    self.update()
                    return

            arm_aidx = self._op_arm_aidx[op_name]
            ctrl = np.asarray(self.data.ctrl, dtype=np.float64).copy()
            ctrl[arm_aidx] = joint_targets
            self.step(ctrl)
        else:
            base_body_pos, base_body_quat_xyzw = self._eef_in_base_to_base_body_world(
                s, target_pos_b, target_quat_b
            )
            self._write_mocap_pose(s, base_body_pos, base_body_quat_xyzw)
            self.update()

    def teleport_operator(
        self, op_name: str, pos_w: np.ndarray, quat_w: np.ndarray
    ) -> None:
        """Instantly set operator pose (world frame). For reset/randomization.

        Joint mode: ``pos_w/quat_w`` is the desired EEF world pose.
        Mocap mode: ``pos_w/quat_w`` is the desired base-body world pose.
        Calls ``mj_forward`` afterwards.
        """
        s = self._get_op(op_name)
        pos_w = np.asarray(pos_w, dtype=np.float64)
        quat_w_f32 = np.asarray(quat_w, dtype=np.float32)

        if s.joint_mode:
            eef_pos_b, eef_quat_b = self._world_to_base(
                pos_w.astype(np.float32), quat_w_f32, s.base_pos, s.base_quat
            )
            eef_in_base = PoseState(
                position=tuple(float(v) for v in eef_pos_b),
                orientation=tuple(float(v) for v in eef_quat_b),
            )
            arm_qidx = self._op_arm_qidx[op_name]
            arm_vidx = self._op_arm_vidx[op_name]
            current_arm_qpos = self.data.qpos[arm_qidx].copy()
            joint_targets = s.ik_solver.solve(eef_in_base, current_arm_qpos)
            if joint_targets is not None:
                self.data.qpos[arm_qidx] = joint_targets
                s.planned_joint_start_qpos = np.asarray(
                    joint_targets, dtype=np.float64
                ).copy()
                s.planned_joint_target_qpos = np.asarray(
                    joint_targets, dtype=np.float64
                ).copy()
                s.planned_joint_progress = 1
                s.planned_joint_steps_total = 1
            if len(arm_vidx) > 0:
                self.data.qvel[arm_vidx] = 0.0
            s.target_pos_in_base = eef_pos_b.astype(np.float32)
            s.target_quat_in_base = eef_quat_b.astype(np.float32)
            s.planned_target_pos_in_base = eef_pos_b.astype(np.float32)
            s.planned_target_quat_in_base = eef_quat_b.astype(np.float32)
        else:
            self._write_mocap_pose(s, pos_w, quat_w_f32, sync_freejoint=True)

        mujoco.mj_forward(self.model, self.data)
        if not s.joint_mode:
            eef_pos_w, eef_quat_w = self.get_site_pose(s.eef_site_name)
            eef_b, eef_bq = self._world_to_base(
                eef_pos_w, eef_quat_w, s.base_pos, s.base_quat
            )
            s.target_pos_in_base = eef_b
            s.target_quat_in_base = eef_bq

    def home_operator(self, op_name: str) -> None:
        """Restore operator to its registered home state."""
        s = self._get_op(op_name)
        if s.joint_mode:
            arm_qidx = self._op_arm_qidx[op_name]
            arm_vidx = self._op_arm_vidx[op_name]
            arm_aidx = self._op_arm_aidx[op_name]
            if s.home_arm_qpos is not None:
                self.data.qpos[arm_qidx] = s.home_arm_qpos
                s.planned_joint_start_qpos = s.home_arm_qpos.copy()
                s.planned_joint_target_qpos = s.home_arm_qpos.copy()
                s.planned_joint_progress = 1
                s.planned_joint_steps_total = 1
            if len(arm_vidx) > 0:
                self.data.qvel[arm_vidx] = 0.0
            for i, aidx in enumerate(arm_aidx):
                ai = int(aidx)
                if s.home_arm_qpos is not None and i < len(s.home_arm_qpos):
                    self.data.ctrl[ai] = s.home_arm_qpos[i]
        else:
            self._write_mocap_pose(
                s,
                s.home_mocap_pos,
                np.array(
                    [
                        s.home_mocap_quat[1],
                        s.home_mocap_quat[2],
                        s.home_mocap_quat[3],
                        s.home_mocap_quat[0],
                    ],
                    dtype=np.float32,
                ),
                sync_freejoint=True,
            )
        # Restore gripper.
        _, eef_qidx, _, eef_vidx, _, eef_aidx = (
            self._split_component_joint_state_indices(op_name)
        )
        n = min(len(s.home_ctrl), self.model.nu)
        for i, aidx in enumerate(eef_aidx):
            ai = int(aidx)
            if ai < n:
                self.data.ctrl[ai] = s.home_ctrl[ai]
                if i < len(eef_qidx):
                    self.data.qpos[eef_qidx[i]] = s.home_ctrl[ai]
        if len(eef_vidx) > 0:
            self.data.qvel[eef_vidx] = 0.0
        mujoco.mj_forward(self.model, self.data)
        home_eef_pos_b, home_eef_quat_b = self.get_operator_eef_pose_in_base(op_name)
        s.target_pos_in_base = home_eef_pos_b
        s.target_quat_in_base = home_eef_quat_b
        s.planned_target_pos_in_base = home_eef_pos_b.copy()
        s.planned_target_quat_in_base = home_eef_quat_b.copy()

    def set_operator_home_eef_pose(
        self, op_name: str, pos_w: np.ndarray, quat_w: np.ndarray
    ) -> None:
        """Update the operator's home EEF pose (world frame input)."""
        s = self._get_op(op_name)
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)

        if s.joint_mode:
            eef_pos_b, eef_quat_b = self._world_to_base(
                pos_w, quat_w, s.base_pos, s.base_quat
            )
            eef_in_base = PoseState(
                position=tuple(float(v) for v in eef_pos_b),
                orientation=tuple(float(v) for v in eef_quat_b),
            )
            arm_qidx = self._op_arm_qidx[op_name]
            current_arm_qpos = self.data.qpos[arm_qidx].copy()
            joint_targets = s.ik_solver.solve(eef_in_base, current_arm_qpos)
            if joint_targets is not None:
                s.home_arm_qpos = joint_targets.copy()
                s.planned_joint_start_qpos = joint_targets.copy()
                s.planned_joint_target_qpos = joint_targets.copy()
                s.planned_joint_progress = 1
                s.planned_joint_steps_total = 1
        else:
            base_body_pos, base_body_quat_xyzw = self._eef_in_base_to_base_body_world(
                s, *self._world_to_base(pos_w, quat_w, s.base_pos, s.base_quat)
            )
            s.home_mocap_pos = np.asarray(base_body_pos, dtype=np.float64)
            qx, qy, qz, qw_ = base_body_quat_xyzw
            s.home_mocap_quat = np.array([qw_, qx, qy, qz], dtype=np.float64)

    def _eef_in_base_to_base_body_world(
        self, state: _OperatorState, eef_pos_b: np.ndarray, eef_quat_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        eef_pos_w, eef_quat_w = self._base_to_world(
            eef_pos_b, eef_quat_b, state.base_pos, state.base_quat
        )
        inv_tool_quat = quaternion_inverse(state.tool_offset_quat).astype(np.float32)
        inv_tool_rot = quaternion_matrix(inv_tool_quat)[:3, :3]
        inv_tool_pos = (-inv_tool_rot @ state.tool_offset_pos).astype(np.float32)
        eef_rot = quaternion_matrix(eef_quat_w)[:3, :3]
        base_body_pos = (eef_pos_w + eef_rot @ inv_tool_pos).astype(np.float64)
        base_body_quat_xyzw = quaternion_multiply(eef_quat_w, inv_tool_quat)
        return base_body_pos, base_body_quat_xyzw.astype(np.float32)

    def _write_mocap_pose(
        self,
        state: _OperatorState,
        pos_w: np.ndarray,
        quat_xyzw: np.ndarray,
        *,
        sync_freejoint: bool = False,
    ) -> None:
        qx, qy, qz, qw = np.asarray(quat_xyzw, dtype=np.float64)
        quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
        self.data.mocap_pos[state.mocap_id] = np.asarray(pos_w, dtype=np.float64)
        self.data.mocap_quat[state.mocap_id] = quat_wxyz
        if sync_freejoint:
            adr = state.fj_qpos_adr
            self.data.qpos[adr : adr + 3] = np.asarray(pos_w, dtype=np.float64)
            self.data.qpos[adr + 3 : adr + 7] = quat_wxyz
            self.data.qvel[state.fj_dof_adr : state.fj_dof_adr + 6] = 0.0

    # ==================================================================
    # Step / observation
    # ==================================================================

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        n = min(len(action), self.model.nu)
        if n > 0:
            ctrl = np.asarray(self.data.ctrl, dtype=np.float64)
            ctrl[:n] = action[:n]
            if self.model.nu > 0:
                low = self.model.actuator_ctrlrange[:n, 0]
                high = self.model.actuator_ctrlrange[:n, 1]
                ctrl[:n] = np.clip(ctrl[:n], low, high)
            self.data.ctrl[:n] = ctrl[:n]
        self.update()

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        return self._collect_obs(self.config.structured)

    def _collect_obs(self, structured: bool) -> dict[str, dict[str, Any]]:
        t = int(self.data.time * 1e9) if self.config.stamp_ns else float(self.data.time)
        obs: dict[str, dict[str, Any]] = {}

        for op in self._operators:
            arm_qidx = self._op_arm_qidx[op.name]
            eef_qidx = self._op_eef_qidx[op.name]
            arm_vidx = self._op_arm_vidx[op.name]
            eef_vidx = self._op_eef_vidx[op.name]
            arm_aidx = self._op_arm_aidx[op.name]
            eef_aidx = self._op_eef_aidx[op.name]
            arm_name, eef_name = self._op_output_names[op.name]

            if DataType.JOINT_POSITION in self.config.enabled_sensors:
                if structured:
                    for prefix, limb, qidx, vidx, aidx in [
                        ("enc", arm_name, arm_qidx, arm_vidx, arm_aidx),
                        ("enc", eef_name, eef_qidx, eef_vidx, eef_aidx),
                        ("action", arm_name, arm_qidx, arm_vidx, arm_aidx),
                        ("action", eef_name, eef_qidx, eef_vidx, eef_aidx),
                    ]:
                        obs[f"{prefix}/{limb}/joint_state"] = {
                            "data": {
                                "position": np.asarray(
                                    self.data.qpos[qidx], dtype=np.float32
                                ),
                                "velocity": np.asarray(
                                    self.data.qvel[vidx], dtype=np.float32
                                ),
                                "effort": np.asarray(
                                    self.data.ctrl[aidx], dtype=np.float32
                                ),
                            },
                            "t": t,
                        }
                else:
                    for limb, qidx in [(arm_name, arm_qidx), (eef_name, eef_qidx)]:
                        obs[f"{limb}/joint_state/position"] = {
                            "data": np.asarray(self.data.qpos[qidx], dtype=np.float32),
                            "t": t,
                        }
                    for limb, aidx in [(arm_name, arm_aidx), (eef_name, eef_aidx)]:
                        obs[f"action/{limb}/joint_state/position"] = {
                            "data": np.asarray(self.data.ctrl[aidx], dtype=np.float32),
                            "t": t,
                        }

            if not structured:
                if DataType.JOINT_VELOCITY in self.config.enabled_sensors:
                    for limb, vidx in [(arm_name, arm_vidx), (eef_name, eef_vidx)]:
                        obs[f"{limb}/joint_state/velocity"] = {
                            "data": np.asarray(self.data.qvel[vidx], dtype=np.float32),
                            "t": t,
                        }
                if DataType.JOINT_EFFORT in self.config.enabled_sensors:
                    for limb, aidx in [(arm_name, arm_aidx), (eef_name, eef_aidx)]:
                        obs[f"{limb}/joint_state/effort"] = {
                            "data": np.asarray(self.data.ctrl[aidx], dtype=np.float32),
                            "t": t,
                        }

            if DataType.POSE in self.config.enabled_sensors:
                site_id = self._pose_site_ids.get(op.name, -1)
                if site_id >= 0:
                    # Validate sensor vs site (once).
                    pos_id = self._pose_ids[op.name]["pos"]
                    quat_id = self._pose_ids[op.name]["quat"]
                    if (
                        op.name not in self._pose_validated_components
                        and pos_id >= 0
                        and quat_id >= 0
                    ):
                        pos_w = self._sensor_data(pos_id)
                        quat_w = self._quat_wxyz_to_xyzw(self._sensor_data(quat_id))
                        self._validate_pose_sensor_matches_site(op.name, pos_w, quat_w)

                    # Current EEF pose in base frame (or world if not registered).
                    state = self._operator_states.get(op.name)
                    if state is not None:
                        pos, quat = self.get_operator_eef_pose_in_base(op.name)
                    else:
                        pos_w2, quat_w2 = self._site_pose(op.name)
                        pos, quat = (
                            pos_w2.astype(np.float32),
                            quat_w2.astype(np.float32),
                        )

                    rot9d = quaternion_matrix(quat)[:3, :3].ravel()
                    if structured:
                        obs[f"{op.name}/pose"] = {
                            "data": {
                                "position": pos.astype(np.float32),
                                "orientation": quat.astype(np.float32),
                            },
                            "t": t,
                        }
                    else:
                        obs[f"{op.name}/pose/position"] = {
                            "data": pos.astype(np.float32),
                            "t": t,
                        }
                        obs[f"{op.name}/pose/orientation"] = {
                            "data": quat.astype(np.float32),
                            "t": t,
                        }
                    obs[f"{op.name}/pose/rotation"] = {
                        "data": euler_from_matrix(rot9d.reshape(3, 3)),
                        "t": t,
                    }
                    obs[f"{op.name}/pose/rotation_6d"] = {
                        "data": rot9d[:6].astype(np.float32),
                        "t": t,
                    }

                    # Target pose in base frame.
                    if state is not None:
                        tgt_pos = state.target_pos_in_base
                        tgt_ori = state.target_quat_in_base
                    else:
                        tgt_pos = pos.astype(np.float32)
                        tgt_ori = quat.astype(np.float32)
                    if structured:
                        obs[f"action/{op.name}/pose"] = {
                            "data": {
                                "position": tgt_pos,
                                "orientation": tgt_ori,
                            },
                            "t": t,
                        }
                    else:
                        obs[f"action/{op.name}/pose/position"] = {
                            "data": tgt_pos,
                            "t": t,
                        }
                        obs[f"action/{op.name}/pose/orientation"] = {
                            "data": tgt_ori,
                            "t": t,
                        }
                # else:
                #     raise ValueError(
                #         f"Operator '{op.name}' has no pose site registered but "
                #         f"DataType.POSE is enabled. Call register_operator with a valid "
                #         f"eef_site or disable DataType.POSE for this operator."
                #     )

            if DataType.IMU in self.config.enabled_sensors:
                acc_id = self._imu_ids[op.name]["acc"]
                gyro_id = self._imu_ids[op.name]["gyro"]
                quat_id = self._imu_ids[op.name]["quat"]
                if acc_id >= 0 and gyro_id >= 0 and quat_id >= 0:
                    acc = self._sensor_data(acc_id)
                    gyro = self._sensor_data(gyro_id)
                    imu_quat = self._sensor_data(quat_id)
                    if structured:
                        obs[f"{op.name}/imu"] = {
                            "data": {
                                "linear_acceleration": acc,
                                "angular_velocity": gyro,
                                "orientation": imu_quat,
                            },
                            "t": t,
                        }
                    else:
                        obs[f"{op.name}/imu/linear_acceleration"] = {
                            "data": acc,
                            "t": t,
                        }
                        obs[f"{op.name}/imu/angular_velocity"] = {"data": gyro, "t": t}
                        obs[f"{op.name}/imu/orientation"] = {"data": imu_quat, "t": t}

            if DataType.WRENCH in self.config.enabled_sensors:
                force = self._sensor_data(self._wrench_ids[op.name]["force"])
                torque = self._sensor_data(self._wrench_ids[op.name]["torque"])
                if force.size == 0 or torque.size == 0:
                    force, torque = self._wrench_from_tactile(op)
                force = np.asarray(force, dtype=np.float32)
                torque = np.asarray(torque, dtype=np.float32)
                if structured:
                    obs[f"{op.name}/wrench"] = {
                        "data": {"force": force, "torque": torque},
                        "t": t,
                    }
                else:
                    obs[f"{op.name}/wrench/force"] = {"data": force, "t": t}
                    obs[f"{op.name}/wrench/torque"] = {"data": torque, "t": t}

        if (
            DataType.TACTILE in self.config.enabled_sensors
            and self._tactile_manager is not None
        ):
            tactile_data = self._tactile_manager.get_data().get("tactile")
            if tactile_data is not None:
                for component, data in self._group_tactile_by_component(
                    tactile_data
                ).items():
                    key_component = (
                        component.replace("_", "/", 1) if structured else component
                    )
                    obs[f"{key_component}/tactile/point_cloud2"] = {
                        "data": data,
                        "t": t,
                    }

        if DataType.CAMERA in self.config.enabled_sensors:
            for cam_name, renderer in self._renderers.items():
                cam_id = self._camera_ids[cam_name]
                spec = self._camera_specs[cam_name]
                renderer.update_scene(
                    self.data,
                    camera=cam_id,
                    scene_option=self._renderer_scene_option,
                )
                renderer.disable_depth_rendering()
                renderer.disable_segmentation_rendering()
                obs_cam_name = (
                    "camera/" + cam_name.split("_")[0] if structured else cam_name
                )
                if spec.enable_color:
                    obs[f"{obs_cam_name}/color/image_raw"] = {
                        "data": np.asarray(renderer.render(), dtype=np.uint8),
                        "t": t,
                    }
                if spec.enable_depth:
                    renderer.enable_depth_rendering()
                    depth = np.asarray(renderer.render(), dtype=np.float32)
                    renderer.disable_depth_rendering()
                    depth[depth > spec.depth_max] = 0.0
                    obs[f"{obs_cam_name}/aligned_depth_to_color/image_raw"] = {
                        "data": depth,
                        "t": t,
                    }
                if spec.enable_mask or spec.enable_heat_map:
                    renderer.enable_segmentation_rendering()
                    segmentation = np.asarray(renderer.render(), dtype=np.int32)
                    renderer.disable_segmentation_rendering()
                    if spec.enable_mask:
                        obs[f"{obs_cam_name}/mask/image_raw"] = {
                            "data": self._build_binary_mask(segmentation),
                            "t": t,
                        }
                    if spec.enable_heat_map:
                        obs[f"{obs_cam_name}/mask/heat_map"] = {
                            "data": self._build_operation_mask(segmentation),
                            "t": t,
                        }

        if structured:
            return {f"/robot/{key}": value for key, value in obs.items()}
        return obs

    # ------------------------------------------------------------------
    # Tactile / wrench helpers for observation
    # ------------------------------------------------------------------

    def _wrench_from_tactile(
        self, op: OperatorBinding
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._tactile_manager is None:
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        wrenches = self._tactile_manager.get_finger_wrenches()
        force = np.zeros(3, dtype=np.float64)
        torque = np.zeros(3, dtype=np.float64)
        for panel_name, panel_wrench in wrenches.items():
            if not op.tactile_prefixes:
                matches = len(self._operators) == 1
            else:
                matches = any(panel_name.startswith(pfx) for pfx in op.tactile_prefixes)
            if not matches:
                continue
            panel_wrench = np.asarray(panel_wrench, dtype=np.float64).reshape(-1)
            if panel_wrench.shape[0] >= 6:
                force += panel_wrench[:3]
                torque += panel_wrench[3:6]
        return force.astype(np.float32), torque.astype(np.float32)

    def _find_operator_for_tactile_panel(
        self, panel_prefix: str
    ) -> OperatorBinding | None:
        for op in self._operators:
            if not op.tactile_prefixes:
                if len(self._operators) == 1:
                    return op
            elif any(panel_prefix.startswith(pfx) for pfx in op.tactile_prefixes):
                return op
        return None

    def _group_tactile_by_component(
        self, tactile_tensor: np.ndarray
    ) -> Dict[str, Dict[str, Any]]:
        tactile_tensor = np.asarray(tactile_tensor, dtype=np.float32)
        grouped = {}
        rows = 8
        cols = 5
        max_points = rows * cols
        for i, panel_prefix in enumerate(self._tactile_manager.panel_order):
            op = self._find_operator_for_tactile_panel(panel_prefix)
            if op is None:
                continue
            _, eef_name = self._op_output_names[op.name]
            panel_data = tactile_tensor[i]
            packed_points = np.zeros((max_points, 6), dtype=np.float32)

            for j in range(min(len(panel_data), max_points)):
                row = j // cols
                col = j % cols
                packed_points[j, 0] = col * 0.005
                packed_points[j, 1] = row * 0.005
                packed_points[j, 2] = 0.0
                packed_points[j, 3] = panel_data[j, 0]
                packed_points[j, 4] = panel_data[j, 1]
                packed_points[j, 5] = panel_data[j, 2]

            feats = ("x", "y", "z", "fx", "fy", "fz")
            field_size = 4
            fields = [
                {
                    "name": name,
                    "offset": field_size * idx,
                    "datatype": "FLOAT32",
                    "count": 1,
                }
                for idx, name in enumerate(feats)
            ]

            panel_label = panel_prefix.rstrip("_")
            key = f"{eef_name}_{panel_label}" if panel_label else eef_name

            sim_time = self.data.time
            sec = int(sim_time)
            nanosec = int((sim_time - sec) * 1e9)
            grouped[key] = {
                "header": {
                    "frame_id": f"{key}_tactile",
                    "stamp": {"sec": sec, "nanosec": nanosec},
                },
                "height": rows,
                "width": cols,
                "fields": fields,
                "is_bigendian": False,
                "point_step": field_size * len(feats),
                "data": packed_points.tobytes(),
                "is_dense": True,
            }

        return dict(grouped)


class BatchedUnifiedMujocoEnv:
    """Aggregate multiple homogeneous ``UnifiedMujocoEnv`` replicas."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.batch_size = int(config.batch_size)
        self.envs: list[UnifiedMujocoEnv] = []
        for env_index in range(self.batch_size):
            viewer = config.viewer if env_index == config.viewer_env_index else None
            env_cfg = config.model_copy(update={"batch_size": 1, "viewer": viewer})
            self.envs.append(UnifiedMujocoEnv(env_cfg))

    def register_operator(self, *args, **kwargs) -> None:
        for env in self.envs:
            env.register_operator(*args, **kwargs)

    def world_to_base(
        self, op_name: str, pos_w: np.ndarray, quat_w: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)
        if pos_w.ndim == 1:
            pos_w = np.repeat(pos_w.reshape(1, 3), self.batch_size, axis=0)
        if quat_w.ndim == 1:
            quat_w = np.repeat(quat_w.reshape(1, 4), self.batch_size, axis=0)
        pos_out = []
        quat_out = []
        for env_index, env in enumerate(self.envs):
            p, q = env.world_to_base(op_name, pos_w[env_index], quat_w[env_index])
            pos_out.append(p)
            quat_out.append(q)
        return np.stack(pos_out), np.stack(quat_out)

    def base_to_world(
        self, op_name: str, pos_b: np.ndarray, quat_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_b = np.asarray(pos_b, dtype=np.float32)
        quat_b = np.asarray(quat_b, dtype=np.float32)
        if pos_b.ndim == 1:
            pos_b = np.repeat(pos_b.reshape(1, 3), self.batch_size, axis=0)
        if quat_b.ndim == 1:
            quat_b = np.repeat(quat_b.reshape(1, 4), self.batch_size, axis=0)
        pos_out = []
        quat_out = []
        for env_index, env in enumerate(self.envs):
            p, q = env.base_to_world(op_name, pos_b[env_index], quat_b[env_index])
            pos_out.append(p)
            quat_out.append(q)
        return np.stack(pos_out), np.stack(quat_out)

    def get_operator_eef_pose_in_base(
        self, op_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = [env.get_operator_eef_pose_in_base(op_name) for env in self.envs]
        return np.stack([p for p, _ in poses]), np.stack([q for _, q in poses])

    def get_operator_eef_pose_in_world(
        self, op_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        poses = [env.get_operator_eef_pose_in_world(op_name) for env in self.envs]
        return np.stack([p for p, _ in poses]), np.stack([q for _, q in poses])

    def get_operator_base_pose(self, op_name: str) -> tuple[np.ndarray, np.ndarray]:
        poses = [env.get_operator_base_pose(op_name) for env in self.envs]
        return np.stack([p for p, _ in poses]), np.stack([q for _, q in poses])

    def override_operator_base_pose(
        self,
        op_name: str,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)
        if pos_w.ndim == 1:
            pos_w = np.repeat(pos_w.reshape(1, 3), self.batch_size, axis=0)
        if quat_w.ndim == 1:
            quat_w = np.repeat(quat_w.reshape(1, 4), self.batch_size, axis=0)
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.override_operator_base_pose(
                    op_name,
                    pos_w[env_index],
                    quat_w[env_index],
                )

    def set_operator_base_pose(
        self,
        op_name: str,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)
        if pos_w.ndim == 1:
            pos_w = np.repeat(pos_w.reshape(1, 3), self.batch_size, axis=0)
        if quat_w.ndim == 1:
            quat_w = np.repeat(quat_w.reshape(1, 4), self.batch_size, axis=0)
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.set_operator_base_pose(
                    op_name,
                    pos_w[env_index],
                    quat_w[env_index],
                )

    def step_operator_toward_target(
        self,
        op_name: str,
        target_pos_b: np.ndarray,
        target_quat_b: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.step_operator_toward_target(
                    op_name,
                    np.asarray(target_pos_b[env_index], dtype=np.float32),
                    np.asarray(target_quat_b[env_index], dtype=np.float32),
                )

    def teleport_operator(
        self,
        op_name: str,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.teleport_operator(op_name, pos_w[env_index], quat_w[env_index])

    def home_operator(self, op_name: str, env_mask: np.ndarray | None = None) -> None:
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.home_operator(op_name)

    def set_operator_home_eef_pose(
        self,
        op_name: str,
        pos_w: np.ndarray,
        quat_w: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pos_w = np.asarray(pos_w, dtype=np.float32)
        quat_w = np.asarray(quat_w, dtype=np.float32)
        if pos_w.ndim == 1:
            pos_w = np.repeat(pos_w.reshape(1, 3), self.batch_size, axis=0)
        if quat_w.ndim == 1:
            quat_w = np.repeat(quat_w.reshape(1, 4), self.batch_size, axis=0)
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.set_operator_home_eef_pose(
                    op_name,
                    pos_w[env_index],
                    quat_w[env_index],
                )

    def get_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        poses = [env.get_body_pose(body_name) for env in self.envs]
        return np.stack([p for p, _ in poses]), np.stack([q for _, q in poses])

    def get_site_pose(self, site_name: str) -> tuple[np.ndarray, np.ndarray]:
        poses = [env.get_site_pose(site_name) for env in self.envs]
        return np.stack([p for p, _ in poses]), np.stack([q for _, q in poses])

    def step(self, action: np.ndarray, env_mask: np.ndarray | None = None) -> None:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            raise ValueError(
                f"Batched step expects shape (B, action_dim), got {action.shape}"
            )
        if action.shape[0] != self.batch_size:
            raise ValueError(
                f"Expected action shape ({self.batch_size}, action_dim), got {action.shape}"
            )
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.step(action[env_index])

    def update(self, env_mask: np.ndarray | None = None) -> None:
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.update()

    def reset(self, env_mask: np.ndarray | None = None) -> None:
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for env_index, env in enumerate(self.envs):
            if mask[env_index]:
                env.reset()

    def set_interest_objects_and_operations(
        self,
        object_names: list[str],
        operation_names: list[str],
    ) -> None:
        if len(object_names) != len(operation_names):
            raise ValueError(
                "object_names and operation_names must have the same length."
            )
        if len(object_names) not in {1, self.batch_size}:
            raise ValueError(
                "Expected either broadcast length 1 or per-env length equal to batch_size."
            )
        broadcast = len(object_names) == 1
        for env_index, env in enumerate(self.envs):
            obj = object_names[0] if broadcast else object_names[env_index]
            op = operation_names[0] if broadcast else operation_names[env_index]
            if obj and op:
                env.set_interest_objects_and_operations([obj], [op])
            else:
                env.set_interest_objects_and_operations([], [])

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        obs_per_env = [env.capture_observation() for env in self.envs]
        keys = set().union(*(obs.keys() for obs in obs_per_env))
        batched: dict[str, dict[str, Any]] = {}
        for key in keys:
            items = [obs[key] for obs in obs_per_env if key in obs]
            if len(items) != self.batch_size:
                raise KeyError(
                    f"Observation key '{key}' missing from some env replicas."
                )
            data_batch = []
            for item in items:
                data = item["data"]
                if isinstance(data, dict):
                    merged = batched.setdefault(key, {"data": {}, "t": []})
                    for subkey, subval in data.items():
                        merged["data"].setdefault(subkey, []).append(np.asarray(subval))
                    merged["t"].append(item["t"])
                else:
                    data_batch.append(np.asarray(data))
            if data_batch:
                batched[key] = {
                    "data": np.stack(data_batch, axis=0),
                    "t": np.asarray([item["t"] for item in items]),
                }
        for key, item in list(batched.items()):
            if isinstance(item["data"], dict):
                item["data"] = {
                    subkey: np.stack(values, axis=0)
                    for subkey, values in item["data"].items()
                }
                item["t"] = np.asarray(item["t"])
        return batched

    def get_info(self) -> dict[str, Any]:
        info = self.envs[0].get_info()
        info["batch_size"] = self.batch_size
        return info

    def refresh_viewer(self) -> None:
        self.envs[self.config.viewer_env_index].refresh_viewer()

    def close(self) -> None:
        for env in self.envs:
            env.close()
