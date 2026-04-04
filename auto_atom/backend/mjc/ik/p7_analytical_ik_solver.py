"""Analytical IK backend for the P7 arm + XF9600 gripper."""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

try:
    from third_party.p7_arm_analytical_ik import KDL_7DOF
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from third_party.p7_arm_analytical_ik import KDL_7DOF

from auto_atom.utils.pose import PoseState, quaternion_to_rotation_matrix


class P7AnalyticalIKSolver:
    """Wrap ``third_party.p7_arm_analytical_ik.KDL_7DOF`` behind ``IKSolver``."""

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_joint_names: List[str],
        flange_site_name: str = "tool_site",
        tcp_site_name: str = "tcp_site",
        max_joint_delta: float = 0.35,
    ) -> None:
        self._arm_joint_names = arm_joint_names
        self._max_joint_delta = max_joint_delta
        self._solver = KDL_7DOF()
        self._configure_tcp_from_model(model, flange_site_name, tcp_site_name)

    def solve(
        self,
        target_pose_in_base: PoseState,
        current_qpos: np.ndarray,
    ) -> Optional[np.ndarray]:
        if target_pose_in_base.batch_size != 1:
            raise ValueError(
                "P7AnalyticalIKSolver.solve expects a single-env PoseState, "
                f"got batch_size={target_pose_in_base.batch_size}"
            )

        q_seed = np.asarray(current_qpos, dtype=np.float64).reshape(-1)
        n_joints = len(self._arm_joint_names)
        if q_seed.shape[0] < n_joints:
            raise ValueError(
                f"Expected at least {n_joints} arm joints in current_qpos, got {q_seed.shape[0]}"
            )

        pos_b = np.asarray(target_pose_in_base.position[0], dtype=np.float64)
        quat_b = np.asarray(target_pose_in_base.orientation[0], dtype=np.float64)

        target_T = np.eye(4, dtype=np.float64)
        target_T[:3, :3] = quaternion_to_rotation_matrix(
            tuple(float(v) for v in quat_b)
        )
        target_T[:3, 3] = pos_b

        # The reference seed keeps the analytical solver on the nearest branch.
        with contextlib.redirect_stdout(io.StringIO()):
            solutions = self._solver.ik(
                target_T,
                reference_angles=q_seed[:n_joints].tolist(),
                use_tcp=True,
            )

        if not solutions:
            return None

        solved = np.asarray(solutions[0], dtype=np.float64)
        delta = solved - q_seed[:n_joints]
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > self._max_joint_delta:
            solved = q_seed[:n_joints] + delta * (self._max_joint_delta / max_delta)
        return solved

    @staticmethod
    def _site_transform(data: mujoco.MjData, site_id: int) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
        T[:3, 3] = data.site_xpos[site_id]
        return T

    def _configure_tcp_from_model(
        self,
        model: mujoco.MjModel,
        flange_site_name: str,
        tcp_site_name: str,
    ) -> None:
        flange_sid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, flange_site_name
        )
        if flange_sid < 0:
            raise ValueError(
                f"Flange site '{flange_site_name}' not found in the model."
            )
        tcp_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)
        if tcp_sid < 0:
            raise ValueError(f"TCP site '{tcp_site_name}' not found in the model.")

        data = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        flange_T = self._site_transform(data, flange_sid)
        tcp_T = self._site_transform(data, tcp_sid)
        self._solver.T_tcp = np.linalg.inv(flange_T) @ tcp_T
        self._solver.inv_tcp = np.linalg.inv(self._solver.T_tcp)


_P7_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_P7_ROOT_BODY = "p7_mount"
_P7_FLANGE_SITE = "tool_site"
_P7_TCP_SITE = "tcp_site"


def build_p7_xf9600_backend(
    task: Any,
    operators: Any,
) -> Any:
    from auto_atom.backend.mjc.mujoco_backend import build_mujoco_backend
    from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv
    from auto_atom.framework import AutoAtomConfig, OperatorConfig
    from auto_atom.runtime import ComponentRegistry

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

    ik_solver = P7AnalyticalIKSolver(
        model=first_env.model,
        arm_joint_names=_P7_ARM_JOINTS,
        flange_site_name=_P7_FLANGE_SITE,
        tcp_site_name=_P7_TCP_SITE,
        max_joint_delta=float(ik_params.get("max_joint_delta", 0.35)),
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
            "root_body_name": _P7_ROOT_BODY,
            "eef_site_name": _P7_TCP_SITE,
            "eef_ctrl_index": eef_ctrl_index,
            "eef_open_value": eef_open_value,
            "eef_close_value": eef_close_value,
        },
    )
