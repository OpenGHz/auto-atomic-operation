"""Analytical IK backend for the AIRBOT Play 6-DOF arm + XF9600 gripper.

Wraps :class:`auto_atom.backend.mjc.ik.third_party_ik.arm_kdl.ArmKdl` behind
the framework's ``IKSolver`` interface, mirroring the structure of
``p7_analytical_ik_solver.py``.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

from auto_atom.backend.mjc.ik.third_party_ik.arm_kdl import ArmKdl
from auto_atom.utils.pose import PoseState, quaternion_to_rotation_matrix


class AirbotKdlIKSolver:
    """Wrap ``ArmKdl`` behind the framework ``IKSolver`` interface.

    The underlying ``ArmKdl`` solves IK in the **arm base body frame** (DH
    chain starts at joint1).  The end-effector offset baked into ``ArmKdl``
    (``d[6]`` plus ``end_convert_matrix``) is overridden at construction
    time so that the solver targets the **actual TCP site** in the MuJoCo
    model rather than the hardcoded gripper offset.

    Parameters
    ----------
    model:
        Loaded MuJoCo model used to introspect the home keyframe and the
        TCP site pose.
    arm_joint_names:
        Names of the 6 arm joints, in DH order.
    arm_base_body:
        Name of the body that the DH chain origin is attached to (typically
        ``arm_base``).
    tcp_site_name:
        Name of the site treated as the operator end-effector.
    arm_type:
        ``ArmKdl`` arm type key (``play_short`` / ``play_long`` / ``play_pro``
        / ``play_lite``).  Default ``play_short``.
    eef_type:
        ``ArmKdl`` end-effector key.  ``"none"`` works because the actual TCP
        offset is recomputed from the model.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_joint_names: List[str],
        arm_base_body: str = "arm_base",
        tcp_site_name: str = "eef_pose",
        arm_type: str = "play_short",
        eef_type: str = "none",
    ) -> None:
        if len(arm_joint_names) != 6:
            raise ValueError(
                f"AirbotKdlIKSolver requires 6 arm joints, got {len(arm_joint_names)}"
            )
        self._arm_joint_names = list(arm_joint_names)
        self._kdl = ArmKdl(arm_type=arm_type, eef_type=eef_type)
        self._configure_tcp_from_model(model, arm_base_body, tcp_site_name)

    # ------------------------------------------------------------------
    # IKSolver interface
    # ------------------------------------------------------------------

    def solve(
        self,
        target_pose_in_base: PoseState,
        current_qpos: np.ndarray,
    ) -> Optional[np.ndarray]:
        if target_pose_in_base.batch_size != 1:
            raise ValueError(
                "AirbotKdlIKSolver.solve expects a single-env PoseState, "
                f"got batch_size={target_pose_in_base.batch_size}"
            )

        q_seed = np.asarray(current_qpos, dtype=np.float64).reshape(-1)
        n_joints = len(self._arm_joint_names)
        if q_seed.shape[0] < n_joints:
            raise ValueError(
                f"Expected at least {n_joints} arm joints in current_qpos, "
                f"got {q_seed.shape[0]}"
            )
        q_seed = q_seed[:n_joints]

        pos_b = np.asarray(target_pose_in_base.position[0], dtype=np.float64)
        quat_b = np.asarray(target_pose_in_base.orientation[0], dtype=np.float64)

        target_T = np.eye(4, dtype=np.float64)
        target_T[:3, :3] = quaternion_to_rotation_matrix(
            tuple(float(v) for v in quat_b)
        )
        target_T[:3, 3] = pos_b

        # ArmKdl prints diagnostic info on degenerate cases — silence it.
        with contextlib.redirect_stdout(io.StringIO()):
            solutions = self._kdl.inverse_kinematics(
                target_T,
                ref_pos=q_seed.copy(),
                force_calculate=False,
                use_clip=False,
            )

        if not solutions:
            return None

        # ArmKdl already sorts by (limit_bias, ref_bias) — pick the first.
        return np.asarray(solutions[0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _site_transform(
        data: mujoco.MjData, site_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(data.site_xpos[site_id], dtype=np.float64).copy(),
            np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy(),
        )

    def _configure_tcp_from_model(
        self,
        model: mujoco.MjModel,
        arm_base_body: str,
        tcp_site_name: str,
    ) -> None:
        """Override the ArmKdl end-effector offset to match the actual TCP.

        Strategy: with the model in its home keyframe, compute the TCP pose
        in the arm base frame, run ArmKdl FK at the same joint configuration
        with ``end_convert_matrix = I`` and ``d[6] = 0``, then derive an
        ``end_convert_matrix`` such that ``KDL_FK @ ECM == TCP_in_base``.
        """
        arm_base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm_base_body)
        if arm_base_bid < 0:
            raise ValueError(f"Arm base body '{arm_base_body}' not found in model.")
        tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)
        if tcp_sid < 0:
            raise ValueError(f"TCP site '{tcp_site_name}' not found in model.")

        joint_ids = []
        for name in self._arm_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Arm joint '{name}' not found in model.")
            joint_ids.append(jid)
        qpos_addrs = [int(model.jnt_qposadr[j]) for j in joint_ids]

        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        home_q = np.array([data.qpos[a] for a in qpos_addrs], dtype=np.float64)

        tcp_w_pos, tcp_w_R = self._site_transform(data, tcp_sid)
        base_w_pos = np.asarray(data.xpos[arm_base_bid], dtype=np.float64).copy()
        base_w_R = (
            np.asarray(data.xmat[arm_base_bid], dtype=np.float64).reshape(3, 3).copy()
        )

        T_tcp_in_base = np.eye(4)
        T_tcp_in_base[:3, :3] = base_w_R.T @ tcp_w_R
        T_tcp_in_base[:3, 3] = base_w_R.T @ (tcp_w_pos - base_w_pos)

        # Reset DH end-effector offsets so the FK output ends at joint7's
        # frame (no end-effector translation), then derive the new ECM.
        self._kdl.dh.d = self._kdl.dh.d.copy()
        self._kdl.dh.d[6] = 0.0
        self._kdl.dh.end_convert_matrix = np.eye(4)
        T_kdl_no_eef = self._kdl.forward_kinematics(home_q)

        ecm = np.linalg.inv(T_kdl_no_eef) @ T_tcp_in_base
        self._kdl.dh.end_convert_matrix = ecm


# ----------------------------------------------------------------------
# Backend builder for AIRBOT Play + XF9600
# ----------------------------------------------------------------------

_AIRBOT_PLAY_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]
_AIRBOT_ROOT_BODY = "arm_base"
_AIRBOT_TCP_SITE = "eef_pose"


def build_airbot_play_xf9600_backend(
    task: Any,
    operators: Any,
) -> Any:
    """Build a Mujoco backend with the AIRBOT Play analytical IK solver."""
    from auto_atom.backend.mjc.mujoco_backend import build_mujoco_backend
    from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv
    from auto_atom.framework import AutoAtomConfig, OperatorConfig  # noqa: PLC0415
    from auto_atom.runtime import ComponentRegistry  # noqa: PLC0415

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
            f"Environment '{config.env_name}' must be a BatchedUnifiedMujocoEnv, "
            f"got {type(env).__name__}."
        )
    first_env = env.envs[0]

    ik_params: Dict[str, Any] = {}
    for op in operator_configs:
        op_extra = op.model_extra or {}
        if "ik" in op_extra and isinstance(op_extra["ik"], dict):
            ik_params = op_extra["ik"]
            break

    ik_solver = AirbotKdlIKSolver(
        model=first_env.model,
        arm_joint_names=_AIRBOT_PLAY_JOINTS,
        arm_base_body=_AIRBOT_ROOT_BODY,
        tcp_site_name=_AIRBOT_TCP_SITE,
        arm_type=str(ik_params.get("arm_type", "play_short")),
        eef_type=str(ik_params.get("eef_type", "none")),
    )

    eef_aidx = first_env._op_eef_aidx.get("arm", np.array([]))
    eef_ctrl_index = int(eef_aidx[0]) if len(eef_aidx) > 0 else 0
    if 0 <= eef_ctrl_index < first_env.model.nu:
        eef_open_value = float(first_env.model.actuator_ctrlrange[eef_ctrl_index, 0])
        eef_close_value = float(first_env.model.actuator_ctrlrange[eef_ctrl_index, 1])
    else:
        eef_open_value = 0.0
        eef_close_value = 1.0

    return build_mujoco_backend(
        task,
        operators,
        ik_solver=ik_solver,
        handler_kwargs={
            "root_body_name": _AIRBOT_ROOT_BODY,
            "eef_site_name": _AIRBOT_TCP_SITE,
            "eef_ctrl_index": eef_ctrl_index,
            "eef_open_value": eef_open_value,
            "eef_close_value": eef_close_value,
            "max_joint_delta": float(ik_params.get("max_joint_delta", 0.35)),
        },
    )
