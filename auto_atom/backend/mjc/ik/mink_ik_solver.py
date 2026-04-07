"""MinkIKSolver — wraps the ``mink`` library to implement the ``IKSolver`` protocol.

Usage
-----
Create a solver once per session (after the MuJoCo model is available), then
pass it to :func:`~auto_atom.backend.mjc.mujoco_backend.build_mujoco_backend`::

    from auto_atom.backend.mjc.ik.mink_ik_solver import MinkIKSolver, build_franka_backend

    # Directly:
    ik_solver = MinkIKSolver(env.model, ["joint1", ..., "joint7"], "eef_pose")
    backend   = build_mujoco_backend(task, operators, ik_solver=ik_solver)

    # Or via the provided factory (used as YAML ``backend`` target):
    backend = build_franka_backend(task, operators)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mujoco
import numpy as np

import mink

from auto_atom.utils.pose import PoseState, quaternion_to_rotation_matrix
from auto_atom.utils.transformations import quaternion_multiply


class MinkIKSolver:
    """IK solver backed by the ``mink`` differential-IK library.

    The solver receives the desired EEF pose in the **operator base frame** and
    returns the arm joint positions that achieve that pose.

    Parameters
    ----------
    model:
        MuJoCo model for the scene. A separate ``mink.Configuration`` (and its
        own ``MjData``) is created internally — the simulation data is never
        mutated by this class.
    arm_joint_names:
        Ordered list of revolute joint names that form the robot arm (e.g.
        ``["joint1", ..., "joint7"]`` for Franka Panda). The returned joint
        targets follow this order.
    frame_name:
        Name of the MuJoCo *site* that represents the desired end-effector
        frame (e.g. ``"eef_pose"``).
    root_body_name:
        Name of the robot's root (base) body in the model. Used to compute the
        world-frame pose of the base so that the base-frame target can be
        converted back to world frame before being passed to mink.
    n_iterations:
        Number of IK integration steps per ``solve()`` call.
    dt:
        Virtual time-step for each IK integration step (seconds).
    position_cost:
        Weighting on EEF position tracking in the task.
    orientation_cost:
        Weighting on EEF orientation tracking in the task.
    posture_cost:
        Weighting on the posture regularization task (keeps joints near their
        initial configuration to avoid singularities).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        arm_joint_names: List[str],
        frame_name: str,
        root_body_name: str = "link0",
        n_iterations: int = 20,
        dt: float = 1e-2,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-3,
    ) -> None:
        self._arm_joint_names = arm_joint_names
        self._frame_name = frame_name
        self._n_iterations = n_iterations
        self._dt = dt

        # mink configuration owns a private MjData copy — never touches sim data.
        self._configuration = mink.Configuration(model)

        # Arm joint qpos addresses (into the full model qpos vector).
        self._arm_qidx: np.ndarray = np.array(
            [
                model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                ]
                for j in arm_joint_names
            ],
            dtype=np.intp,
        )

        # EEF frame task.
        self._eef_task = mink.FrameTask(
            frame_name=frame_name,
            frame_type="site",
            position_cost=position_cost,
            orientation_cost=orientation_cost,
        )

        # Posture task — hold joints near the initial keyframe pose.
        self._posture_task = mink.PostureTask(model, cost=posture_cost)
        self._posture_task.set_target_from_configuration(self._configuration)

        # Limits.
        self._limits = [mink.ConfigurationLimit(model)]

        # Base pose in world frame (fixed for this scene).
        root_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
        if root_bid < 0:
            raise ValueError(
                f"Root body '{root_body_name}' not found in the MuJoCo model."
            )
        self._base_pos: np.ndarray = model.body_pos[root_bid].astype(np.float64).copy()
        wxyz = model.body_quat[root_bid].astype(np.float64)
        # Store as xyzw (our internal convention).
        self._base_quat_xyzw: np.ndarray = np.array(
            [wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64
        )

    # ------------------------------------------------------------------
    # IKSolver protocol
    # ------------------------------------------------------------------

    def solve(
        self,
        target_pose_in_base: PoseState,
        current_qpos: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Solve IK for the given EEF target (base frame) and return arm qpos.

        Parameters
        ----------
        target_pose_in_base:
            Desired EEF pose expressed in the operator's base frame.
        current_qpos:
            Current arm joint positions (one value per joint in
            ``arm_joint_names``).

        Returns
        -------
        np.ndarray | None
            Solved arm joint positions in the same order as
            ``arm_joint_names``, or ``None`` if the solve diverged.
        """
        # --- Convert base-frame target → world-frame SE3 for mink ---
        if target_pose_in_base.batch_size != 1:
            raise ValueError(
                "MinkIKSolver.solve expects a single-env PoseState, "
                f"got batch_size={target_pose_in_base.batch_size}"
            )

        R_base = quaternion_to_rotation_matrix(
            tuple(float(v) for v in self._base_quat_xyzw)
        )
        pos_b = np.asarray(target_pose_in_base.position[0], dtype=np.float64)
        quat_b = np.asarray(target_pose_in_base.orientation[0], dtype=np.float64)

        pos_w = R_base @ pos_b + self._base_pos
        quat_w = quaternion_multiply(self._base_quat_xyzw, quat_b)

        R_w = quaternion_to_rotation_matrix(tuple(float(v) for v in quat_w))
        target_se3 = mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(R_w), pos_w
        )
        self._eef_task.set_target(target_se3)

        # --- Seed the IK configuration with the current arm joints ---
        q = self._configuration.q.copy()
        n = min(len(current_qpos), len(self._arm_qidx))
        q[self._arm_qidx[:n]] = current_qpos[:n]
        self._configuration.update(q)

        # Update posture target to current seed so IK stays near the seed.
        self._posture_task.set_target_from_configuration(self._configuration)

        # --- Run IK iterations ---
        tasks = [self._eef_task, self._posture_task]
        for _ in range(self._n_iterations):
            vel = mink.solve_ik(
                self._configuration,
                tasks,
                self._dt,
                solver="quadprog",
                limits=self._limits,
            )
            self._configuration.integrate_inplace(vel, self._dt)

        return self._configuration.q[self._arm_qidx].copy()


# ---------------------------------------------------------------------------
# Convenience factory — used as the YAML ``backend`` target for Franka tasks
# ---------------------------------------------------------------------------

_FRANKA_ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]
_FRANKA_EEF_SITE = "eef_pose"
_FRANKA_ROOT_BODY = "link0"


def build_franka_backend(
    task: Any,
    operators: Any,
) -> Any:
    """Build a :class:`~auto_atom.backend.mjc.mujoco_backend.MujocoTaskBackend`
    for the Franka Panda + Robotiq 2F-85 arm using mink IK.

    This function is intended to be used as the ``backend`` entry in task YAML
    files::

        backend: auto_atom.backend.mjc.ik.mink_ik_solver.build_franka_backend

    It looks up the already-registered ``UnifiedMujocoEnv`` (registered by
    ``instantiate(cfg.env)`` before this function is called), creates a
    :class:`MinkIKSolver`, then delegates to
    :func:`~auto_atom.backend.mjc.mujoco_backend.build_mujoco_backend`.

    IK solver parameters can be overridden per-operator via the ``ik`` key
    in the operator YAML config::

        operators:
          - name: arm
            ik:
              n_iterations: 300
              dt: 0.1
              position_cost: 1.0
              orientation_cost: 1.0
              posture_cost: 1e-4
              max_joint_delta: 0.8
    """
    from auto_atom.framework import AutoAtomConfig, OperatorConfig
    from auto_atom.runtime import ComponentRegistry
    from auto_atom.backend.mjc.mujoco_backend import build_mujoco_backend
    from auto_atom.basis.mjc.mujoco_env import BatchedUnifiedMujocoEnv

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

    # Merge IK params from the first operator that has an ``ik`` section.
    ik_params: Dict[str, Any] = {}
    for op in operator_configs:
        op_extra = op.model_extra or {}
        if "ik" in op_extra and isinstance(op_extra["ik"], dict):
            ik_params = op_extra["ik"]
            break

    ik_solver = MinkIKSolver(
        model=first_env.model,
        arm_joint_names=_FRANKA_ARM_JOINTS,
        frame_name=_FRANKA_EEF_SITE,
        root_body_name=_FRANKA_ROOT_BODY,
        n_iterations=ik_params.get("n_iterations", 300),
        dt=ik_params.get("dt", 0.1),
        position_cost=ik_params.get("position_cost", 1.0),
        orientation_cost=ik_params.get("orientation_cost", 1.0),
        posture_cost=ik_params.get("posture_cost", 1e-4),
    )

    eef_aidx = first_env._op_eef_aidx.get("arm", np.array([]))
    eef_ctrl_index = int(eef_aidx[0]) if len(eef_aidx) > 0 else 0

    return build_mujoco_backend(
        task,
        operators,
        ik_solver=ik_solver,
        handler_kwargs={
            "root_body_name": _FRANKA_ROOT_BODY,
            "eef_site_name": _FRANKA_EEF_SITE,
            "eef_ctrl_index": eef_ctrl_index,
            "max_joint_delta": float(ik_params.get("max_joint_delta", 0.8)),
        },
    )
