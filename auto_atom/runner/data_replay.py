"""DataReplayRunner – replays recorded demonstration data through PolicyEvaluator.

This module extracts the core replay logic from ``examples/replay_demo.py`` into
a runner class that conforms to the :class:`RunnerBase` interface, so it can be
driven by an external manager via the standard ``reset`` / ``update`` / ``close``
lifecycle.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

from auto_atom.framework import TaskFileConfig
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runtime import ExecutionContext, TaskUpdate

from .base import RunnerBase


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class DataReplayConfig(BaseModel):
    """Replay-specific settings (subset of the original ``ReplayConfig``)."""

    demo_name: str | None = Field(default=None)
    mode: str = Field(default="pose")
    reset_from_first_frame: bool = Field(default=True)
    steps_per_action: int = Field(default=1)
    """Number of physics steps to repeat each action before advancing to the
    next one.  Set to >1 when the simulation timestep is smaller than the
    demo recording interval (e.g. 10 for MuJoCo sub-stepping)."""
    mcap_path: str | None = Field(default=None)
    arm_topic: str = Field(default="/robot/right_arm/joint_state")
    gripper_topic: str = Field(default="/robot/right_gripper/joint_state")
    joint_name_mapping: Dict[str, str] = {"gripper": "xfg_claw_joint"}
    gripper_range: List[float] = Field(default=[0.0, 0.09])
    """Real gripper distance range [closed, open] in metres from mcap data."""
    kinematic: bool = Field(default=False)
    """If ``True`` the replay sets joint positions directly (no physics);
    if ``False`` the replay drives actuators through the physics engine."""
    demo_dir: str | None = Field(default=None)
    """Directory containing npz demo files. Defaults to ``outputs/records/demos``."""


class DataReplayTaskFileConfig(TaskFileConfig):
    """TaskFileConfig with an embedded :class:`DataReplayConfig`."""

    replay: DataReplayConfig = DataReplayConfig()


# ---------------------------------------------------------------------------
# Gripper rescaling
# ---------------------------------------------------------------------------


def _rescale_gripper_distance_to_ctrl(
    values: np.ndarray,
    src_range: list[float],
    ctrl_range: tuple[float, float] | list[float],
) -> np.ndarray:
    """Map finger distance [closed, open] to actuator ctrl [open, closed]."""
    src_closed, src_open = float(src_range[0]), float(src_range[1])
    if abs(src_open - src_closed) < 1e-12:
        return np.asarray(values, dtype=np.float32).copy()

    ctrl_open = float(ctrl_range[0])
    ctrl_closed = float(ctrl_range[1])
    ctrl_min = min(ctrl_open, ctrl_closed)
    ctrl_max = max(ctrl_open, ctrl_closed)

    arr = np.asarray(values, dtype=np.float32)
    scaled = ctrl_closed + (arr - src_closed) / (src_open - src_closed) * (
        ctrl_open - ctrl_closed
    )
    return np.clip(scaled, ctrl_min, ctrl_max)


# ---------------------------------------------------------------------------
# Demo loading
# ---------------------------------------------------------------------------


def _load_low_dim_map(demo_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    if "low_dim_keys" not in demo_data:
        raise KeyError("NPZ missing 'low_dim_keys'.")
    low_dim_keys = [str(key) for key in np.asarray(demo_data["low_dim_keys"])]
    low_dim_map: Dict[str, np.ndarray] = {}
    for idx, key in enumerate(low_dim_keys):
        data_key = f"low_dim_data__{idx}"
        if data_key not in demo_data:
            raise KeyError(f"NPZ missing '{data_key}' for low-dimensional key '{key}'.")
        low_dim_map[key] = np.asarray(demo_data[data_key], dtype=np.float32)
    return low_dim_map


def _load_pose_demo(demo_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    """Load pose + gripper arrays for pose-mode replay."""
    low_dim = _load_low_dim_map(demo_data)
    result: Dict[str, np.ndarray] = {
        "position": low_dim["action/arm/pose/position"],
        "orientation": low_dim["action/arm/pose/orientation"],
    }
    gripper_key = (
        "action/gripper/joint_state/position"
        if "action/gripper/joint_state/position" in low_dim
        else "action/eef/joint_state/position"
    )
    if gripper_key in low_dim:
        result["gripper"] = low_dim[gripper_key]
    return result


def _load_ctrl_demo(demo_data: np.lib.npyio.NpzFile) -> Dict[str, np.ndarray]:
    """Load joint ctrl arrays for ctrl-mode replay."""
    low_dim = _load_low_dim_map(demo_data)
    arm = low_dim["action/arm/joint_state/position"]
    eef_key = "action/eef/joint_state/position"
    if eef_key in low_dim:
        ctrl = np.concatenate([arm, low_dim[eef_key]], axis=-1)
    else:
        ctrl = arm
    return {"ctrl": ctrl}


# ---------------------------------------------------------------------------
# McapDemo
# ---------------------------------------------------------------------------


class McapDemo:
    """Container for data loaded from a ROS2 mcap file."""

    joint: np.ndarray  # (T, n_arm + n_grip)
    joint_names: list[str]

    def __init__(self, joint: np.ndarray, joint_names: list[str]) -> None:
        self.joint = joint
        self.joint_names = joint_names

    def first_frame_joint_positions(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.joint_names, self.joint[0])}

    def align_to_actuators(
        self,
        actuator_names: list[str],
        name_mapping: Dict[str, str] | None = None,
    ) -> None:
        mapping = name_mapping or {}
        mapped_names = [mapping.get(n, n) for n in self.joint_names]
        reorder: list[int] = []
        for act_name in actuator_names:
            if act_name not in mapped_names:
                raise ValueError(
                    f"Actuator '{act_name}' not found in mcap joint names "
                    f"{self.joint_names} (after mapping: {mapped_names})"
                )
            reorder.append(mapped_names.index(act_name))
        self.joint = self.joint[:, reorder]
        self.joint_names = [actuator_names[i] for i in range(len(actuator_names))]

    def rescale_gripper(
        self,
        eef_actuator_names: list[str],
        src_range: list[float],
        env_cfg: Any,
        base_dir: str | None = None,
    ) -> None:
        """Rescale real gripper distance to MuJoCo actuator ctrl."""
        import mujoco as mj
        from omegaconf import OmegaConf

        env_raw = (
            OmegaConf.to_container(env_cfg, resolve=True)
            if hasattr(env_cfg, "_metadata")
            else env_cfg
        )
        model_path = env_raw.get("model_path")
        if model_path is None:
            model_path = env_raw.get("config", {}).get("model_path")
        if model_path is None:
            print("Warning: cannot resolve model_path; skipping gripper rescale.")
            return
        model_path = str(model_path)
        if not os.path.isabs(model_path):
            model_path = os.path.join(base_dir or os.getcwd(), model_path)
        model = mj.MjModel.from_xml_path(str(model_path))

        src_lo, src_hi = float(src_range[0]), float(src_range[1])
        if abs(src_hi - src_lo) < 1e-12:
            return

        for act_name in eef_actuator_names:
            if act_name not in self.joint_names:
                continue
            col = self.joint_names.index(act_name)
            aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)
            if aid < 0:
                continue
            dst_lo = float(model.actuator_ctrlrange[aid, 0])
            dst_hi = float(model.actuator_ctrlrange[aid, 1])
            self.joint[:, col] = _rescale_gripper_distance_to_ctrl(
                self.joint[:, col],
                src_range=[src_lo, src_hi],
                ctrl_range=[dst_lo, dst_hi],
            )


def _load_mcap_demo(mcap_path: str, arm_topic: str, gripper_topic: str) -> McapDemo:
    """Load arm joint + gripper arrays from a ROS2 mcap file."""
    from mcap.reader import make_reader
    from mcap_ros2idl_support import Ros2DecodeFactory

    factory = Ros2DecodeFactory()
    arm_names: list[str] | None = None
    gripper_names: list[str] | None = None
    arm_positions: list[list[float]] = []
    gripper_positions: list[list[float]] = []
    arm_times: list[int] = []
    gripper_times: list[int] = []

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[factory])
        for decoded in reader.iter_decoded_messages(topics=[arm_topic, gripper_topic]):
            topic = decoded.channel.topic
            msg = decoded.decoded_message
            t = decoded.message.log_time
            if topic == arm_topic:
                if arm_names is None:
                    arm_names = list(msg["name"])
                arm_positions.append(list(msg["position"]))
                arm_times.append(t)
            elif topic == gripper_topic:
                if gripper_names is None:
                    gripper_names = list(msg["name"])
                gripper_positions.append(list(msg["position"]))
                gripper_times.append(t)

    if not arm_positions:
        raise ValueError(f"No arm messages found on topic '{arm_topic}' in {mcap_path}")
    if not gripper_positions:
        raise ValueError(
            f"No gripper messages found on topic '{gripper_topic}' in {mcap_path}"
        )

    arm = np.array(arm_positions, dtype=np.float32)
    grip = np.array(gripper_positions, dtype=np.float32)
    arm_t = np.array(arm_times, dtype=np.int64)
    grip_t = np.array(gripper_times, dtype=np.int64)

    if len(arm_t) != len(grip_t) or np.any(np.abs(arm_t - grip_t) > 50_000_000):
        indices = np.searchsorted(grip_t, arm_t, side="left")
        indices = np.clip(indices, 0, len(grip_t) - 1)
        grip = grip[indices]

    joint = np.concatenate([arm, grip], axis=-1)
    joint_names = (arm_names or []) + (gripper_names or [])
    return McapDemo(joint=joint, joint_names=joint_names)


# ---------------------------------------------------------------------------
# Batch normalisation
# ---------------------------------------------------------------------------


def normalize_demo_for_batch(
    demo: Dict[str, np.ndarray],
    batch_size: int,
    mode: str,
) -> Dict[str, np.ndarray]:
    """Slice a (T, B_rec, dim) demo to match the replay *batch_size*."""
    if mode == "pose":
        position = demo["position"]
        orientation = demo["orientation"]
        rec_bs = position.shape[1]
        if batch_size > rec_bs:
            raise ValueError(
                f"Demo recorded with batch_size={rec_bs}, "
                f"but replay requires batch_size={batch_size}."
            )
        position = position[:, :batch_size, :]
        orientation = orientation[:, :batch_size, :]
        result: Dict[str, np.ndarray] = {
            "position": position[:, 0, :] if batch_size == 1 else position,
            "orientation": orientation[:, 0, :] if batch_size == 1 else orientation,
        }
        if "gripper" in demo:
            gripper = demo["gripper"][:, :batch_size, :]
            result["gripper"] = gripper[:, 0, :] if batch_size == 1 else gripper
        return result

    if mode == "joint":
        joint = demo["joint"]
        if joint.ndim == 2:
            return {"joint": joint}
        rec_bs = joint.shape[1]
        if batch_size > rec_bs:
            raise ValueError(
                f"Demo recorded with batch_size={rec_bs}, "
                f"but replay requires batch_size={batch_size}."
            )
        joint = joint[:, :batch_size, :]
        return {"joint": joint[:, 0, :] if batch_size == 1 else joint}

    ctrl = demo["ctrl"]
    rec_bs = ctrl.shape[1]
    if batch_size > rec_bs:
        raise ValueError(
            f"Demo recorded with batch_size={rec_bs}, "
            f"but replay requires batch_size={batch_size}."
        )
    ctrl = ctrl[:, :batch_size, :]
    return {"ctrl": ctrl[:, 0, :] if batch_size == 1 else ctrl}


# ---------------------------------------------------------------------------
# ReplayPolicy
# ---------------------------------------------------------------------------


class ReplayPolicy:
    """Replays recorded actions step by step (index-based, no observation)."""

    def __init__(self, demo: Dict[str, np.ndarray], mode: str) -> None:
        self._demo = demo
        self._mode = mode
        if mode == "pose":
            self._max = len(demo["position"]) - 1
        elif mode == "joint":
            self._max = len(demo["joint"]) - 1
        else:
            self._max = len(demo["ctrl"]) - 1
        self._step = 0

    def reset(self, start_step: int = 0) -> None:
        self._step = max(0, int(start_step))

    @property
    def num_steps(self) -> int:
        return self._max + 1

    @property
    def remaining_steps(self) -> int:
        return max(self._max - self._step + 1, 0)

    def _action_at(self, index: int) -> Dict[str, Any]:
        i = min(max(0, index), self._max)
        if self._mode == "pose":
            action: Dict[str, Any] = {
                "position": self._demo["position"][i],
                "orientation": self._demo["orientation"][i],
            }
            if "gripper" in self._demo:
                action["gripper"] = self._demo["gripper"][i]
            return action
        if self._mode == "joint":
            return {"joint": self._demo["joint"][i]}
        return {"ctrl": self._demo["ctrl"][i]}

    def apply_first_frame_as_reset(self) -> Dict[str, Any] | None:
        if self.num_steps <= 0:
            return None
        action = self._action_at(0)
        self._step = min(1, self.num_steps)
        return action

    def act(self) -> Dict[str, Any]:
        action = self._action_at(self._step)
        self._step += 1
        return action


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------


def _apply_reset_action(context: ExecutionContext, action: Any) -> None:
    """Apply a recorded action as an exact reset state when possible."""
    if action is None:
        return
    env = context.backend.env
    if "joint" in action:
        env.apply_joint_action("arm", action["joint"], kinematic=True)
    elif "ctrl" in action:
        ctrl = np.asarray(action["ctrl"], dtype=np.float64)
        if ctrl.ndim == 1:
            ctrl = ctrl.reshape(1, -1).repeat(env.batch_size, axis=0)
        env.step(ctrl)
    else:
        env.apply_pose_action(
            "arm",
            action["position"],
            action["orientation"],
            action.get("gripper"),
            kinematic=True,
        )


def _make_replay_action_applier(kinematic: bool = False):
    """Return an action applier closure with the configured *kinematic* flag."""

    def replay_action_applier(
        context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
    ) -> None:
        if action is None:
            return
        env = context.backend.env
        if "joint" in action:
            env.apply_joint_action(
                "arm", action["joint"], env_mask=env_mask, kinematic=kinematic
            )
        elif "ctrl" in action:
            ctrl = np.asarray(action["ctrl"], dtype=np.float64)
            if ctrl.ndim == 1:
                ctrl = ctrl.reshape(1, -1).repeat(env.batch_size, axis=0)
            env.step(ctrl, env_mask=env_mask)
        else:
            env.apply_pose_action(
                "arm",
                action["position"],
                action["orientation"],
                action.get("gripper"),
                env_mask=env_mask,
                kinematic=kinematic,
            )

    return replay_action_applier


def _apply_first_frame_reset(
    evaluator: PolicyEvaluator,
    policy: ReplayPolicy,
) -> Dict[str, Any] | None:
    """Apply the first recorded action as the post-reset initial state."""
    reset_action = policy.apply_first_frame_as_reset()
    if reset_action is None:
        return None
    context = evaluator._context
    if context is None:
        raise RuntimeError("PolicyEvaluator must be initialized before applying reset.")
    with evaluator.sim_lock:
        _apply_reset_action(context, reset_action)
    return reset_action


# ---------------------------------------------------------------------------
# DataReplayRunner
# ---------------------------------------------------------------------------


def preprocess_replay_dictconfig(
    cfg: Any,
    replay_cfg: DataReplayConfig,
    project_root: Optional[str] = None,
) -> None:
    """Pre-process a Hydra DictConfig for mcap replay **before** ``prepare_task_file``.

    This injects ``initial_joint_positions`` into ``cfg.env`` so the backend
    resets the robot at the recorded starting configuration.  It also disables
    task randomization for exact trajectory reproduction.

    Call this *before* ``prepare_task_file(cfg)`` when ``replay_cfg.mcap_path``
    is set.  For npz demos no pre-processing is needed.

    Parameters
    ----------
    cfg:
        A Hydra ``DictConfig`` (with ``open_dict`` support).
    replay_cfg:
        The replay settings – ``mcap_path`` must be non-``None``.
    project_root:
        Base directory for resolving relative paths.  Defaults to ``os.getcwd()``.
    """
    from omegaconf import open_dict

    mcap_path = replay_cfg.mcap_path
    if mcap_path is None:
        return

    root = project_root or os.getcwd()
    if not os.path.isabs(mcap_path):
        mcap_path = os.path.join(root, mcap_path)
    if not os.path.exists(mcap_path):
        raise FileNotFoundError(f"MCAP file not found: {mcap_path}")

    mcap_demo = _load_mcap_demo(
        mcap_path, replay_cfg.arm_topic, replay_cfg.gripper_topic
    )
    replay_cfg.mode = "joint"

    op_cfg = cfg.env.operators.arm
    actuator_names = list(op_cfg.arm_actuators) + list(op_cfg.eef_actuators)
    mcap_demo.align_to_actuators(actuator_names, replay_cfg.joint_name_mapping or None)
    mcap_demo.rescale_gripper(
        eef_actuator_names=list(op_cfg.eef_actuators),
        src_range=replay_cfg.gripper_range,
        env_cfg=cfg.env,
        base_dir=root,
    )

    init_jpos = mcap_demo.first_frame_joint_positions()
    with open_dict(cfg):
        if "initial_joint_positions" not in cfg.env:
            cfg.env.initial_joint_positions = {}
        cfg.env.initial_joint_positions.update(init_jpos)
    print(f"Injected initial_joint_positions: {init_jpos}")

    # Disable randomization for exact trajectory reproduction.
    with open_dict(cfg):
        if "task" in cfg and "randomization" in cfg.task:
            cfg.task.randomization = {}


class DataReplayRunner(RunnerBase):
    """Runner that replays recorded demonstration data.

    Wraps :class:`PolicyEvaluator` and :class:`ReplayPolicy` behind the
    standard ``RunnerBase`` interface so it can be driven by a manager in the
    same way as :class:`TaskRunner`.
    """

    def __init__(
        self,
        observation_getter: Optional[Any] = None,
    ) -> None:
        self._observation_getter = observation_getter
        self._evaluator: Optional[PolicyEvaluator] = None
        self._policy: Optional[ReplayPolicy] = None
        self._replay_cfg: Optional[DataReplayConfig] = None
        self._current_action: Optional[Dict[str, Any]] = None
        self._action_step: int = 0

    def from_config(self, config: DataReplayTaskFileConfig) -> DataReplayRunner:
        # config may come from TaskFileConfig.model_validate() which keeps
        # extra fields as plain dicts; coerce to DataReplayConfig if needed.
        replay_raw = config.replay
        if isinstance(replay_raw, DataReplayConfig):
            self._replay_cfg = replay_raw
        else:
            self._replay_cfg = DataReplayConfig.model_validate(replay_raw)
        rcfg = self._replay_cfg

        # --- Load demo data ---
        if rcfg.mcap_path is not None:
            mcap_path = rcfg.mcap_path
            if not os.path.isabs(mcap_path):
                mcap_path = os.path.join(os.getcwd(), mcap_path)
            if not os.path.exists(mcap_path):
                raise FileNotFoundError(f"MCAP file not found: {mcap_path}")
            mcap_demo = _load_mcap_demo(mcap_path, rcfg.arm_topic, rcfg.gripper_topic)
            rcfg.mode = "joint"

            # Align mcap column order to the YAML actuator declaration order
            # when task_file contains the env operator config.
            task_file_raw = config.model_dump() if hasattr(config, "model_dump") else {}
            env_cfg = task_file_raw.get("env", {})
            op_cfg = env_cfg.get("operators", {}).get("arm", {})
            if op_cfg:
                actuator_names = list(op_cfg.get("arm_actuators", [])) + list(
                    op_cfg.get("eef_actuators", [])
                )
                if actuator_names:
                    mcap_demo.align_to_actuators(
                        actuator_names, rcfg.joint_name_mapping or None
                    )
                    eef_actuator_names = list(op_cfg.get("eef_actuators", []))
                    if eef_actuator_names:
                        mcap_demo.rescale_gripper(
                            eef_actuator_names=eef_actuator_names,
                            src_range=rcfg.gripper_range,
                            env_cfg=env_cfg,
                        )

            demo: Dict[str, np.ndarray] = {"joint": mcap_demo.joint}
        else:
            demo_name = rcfg.demo_name or "demo"
            demo_dir = rcfg.demo_dir or os.path.join(
                os.getcwd(), "outputs", "records", "demos"
            )
            demo_npz_path = os.path.join(demo_dir, f"{demo_name}.npz")
            if not os.path.exists(demo_npz_path):
                raise FileNotFoundError(f"Demo file not found: {demo_npz_path}")
            demo_data = np.load(demo_npz_path)
            if rcfg.mode == "pose":
                demo = _load_pose_demo(demo_data)
            elif rcfg.mode == "ctrl":
                demo = _load_ctrl_demo(demo_data)
            else:
                raise ValueError(
                    f"Unknown replay mode: {rcfg.mode!r} "
                    f"(expected 'pose', 'ctrl', or set mcap_path)"
                )

        # --- Build evaluator ---
        self._evaluator = PolicyEvaluator(
            action_applier=_make_replay_action_applier(rcfg.kinematic),
            observation_getter=self._observation_getter,
        ).from_config(config)

        # --- Normalise demo for batch size ---
        demo = normalize_demo_for_batch(
            demo, batch_size=self._evaluator.batch_size, mode=rcfg.mode
        )
        self._policy = ReplayPolicy(demo, rcfg.mode)
        return self

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        evaluator = self._require_evaluator()
        policy = self._require_policy()
        policy.reset()
        self._current_action = None
        self._action_step = 0
        update = evaluator.reset(env_mask)
        if self._replay_cfg is not None and self._replay_cfg.reset_from_first_frame:
            _apply_first_frame_reset(evaluator, policy)
        return update

    def update(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        evaluator = self._require_evaluator()
        policy = self._require_policy()
        steps_per_action = self._replay_cfg.steps_per_action if self._replay_cfg else 1

        # Advance to the next recorded action when the sub-step budget for
        # the current action is exhausted.
        if self._current_action is None or self._action_step >= steps_per_action:
            if policy.remaining_steps > 0:
                self._current_action = policy.act()
            else:
                self._current_action = None
            self._action_step = 0

        self._action_step += 1
        return evaluator.update(self._current_action, env_mask)

    def close(self) -> None:
        if self._evaluator is not None:
            self._evaluator.close()
            self._evaluator = None
        self._policy = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._require_evaluator().batch_size

    @property
    def remaining_steps(self) -> int:
        return self._require_policy().remaining_steps

    def get_observation(self) -> Any:
        return self._require_evaluator().get_observation()

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
        elapsed_time_sec: float = 0.0,
    ) -> Any:
        return self._require_evaluator().summarize(
            update,
            max_updates=max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed_time_sec,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_evaluator(self) -> PolicyEvaluator:
        if self._evaluator is None:
            raise RuntimeError(
                "DataReplayRunner is not initialized. Call from_config() first."
            )
        return self._evaluator

    def _require_policy(self) -> ReplayPolicy:
        if self._policy is None:
            raise RuntimeError(
                "DataReplayRunner is not initialized. Call from_config() first."
            )
        return self._policy
