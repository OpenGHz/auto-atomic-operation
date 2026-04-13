"""Replay recorded low-level actions with the PolicyEvaluator.

By default this script loads ``outputs/records/demos/<config_name>.npz``
produced by ``record_demo.py`` and replays the saved sequence step by step.
It also supports loading ROS2 mcap files for joint-level replay.

Examples:

    # NPZ demos
    python examples/replay_demo.py --config-name press_three_buttons
    python examples/replay_demo.py --config-name pick_and_place +replay.mode=ctrl
    python examples/replay_demo.py --config-name pick_and_place +replay.save_gif=true
    python examples/replay_demo.py --config-name pick_and_place +replay.demo_name=my_demo

    # ROS2 mcap replay (joint mode, auto-selected when mcap_path is set)
    python examples/replay_demo.py --config-name pick_and_place \
        +replay.mcap_path=data/recording_20260401_185226.mcap
    python examples/replay_demo.py --config-name pick_and_place \
        +replay.mcap_path=data/recording.mcap \
        +replay.arm_topic=/robot/right_arm/joint_state \
        +replay.gripper_topic=/robot/right_gripper/distance
"""

from __future__ import annotations
from typing import Any, Optional

import os
import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from pydantic import BaseModel, Field

from auto_atom import ExecutionContext, PolicyEvaluator
from auto_atom.runner.common import get_config_dir, prepare_task_file


class ReplayConfig(BaseModel):
    demo_name: str | None = Field(default=None)
    mode: str = Field(default="pose")
    reset_from_first_frame: bool = Field(default=True)
    camera: str = Field(default="env1_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)
    save_gif: bool = Field(default=True)
    save_mp4: bool = Field(default=False)
    mcap_path: str | None = Field(default=None)
    arm_topic: str = Field(default="/robot/right_arm/joint_state")
    gripper_topic: str = Field(default="/robot/right_gripper/joint_state")
    joint_name_mapping: dict[str, str] = {"gripper": "xfg_claw_joint"}
    gripper_range: list[float] = Field(default=[0.0, 0.09])
    """Real gripper distance range [closed, open] in metres from mcap data.
    Used to rescale the recorded gripper distance to the MuJoCo actuator
    ctrlrange.  Set to the actuator ctrlrange to skip rescaling."""


def _rescale_gripper_distance_to_ctrl(
    values: np.ndarray,
    src_range: list[float],
    ctrl_range: tuple[float, float] | list[float],
) -> np.ndarray:
    """Map finger distance [closed, open] to actuator ctrl [open, closed].

    Real-robot recordings publish the distance between the two fingers, so a
    larger value means the gripper is more open. In the supported MuJoCo
    gripper models used by replay, ``ctrlrange[0]`` is the open command and
    ``ctrlrange[1]`` is the closed command, so the mapping must be inverted to
    preserve the distance semantics.
    """
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


def _load_low_dim_map(demo_data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    if "low_dim_keys" not in demo_data:
        raise KeyError("NPZ missing 'low_dim_keys'.")

    low_dim_keys = [str(key) for key in np.asarray(demo_data["low_dim_keys"])]
    low_dim_map: dict[str, np.ndarray] = {}
    for idx, key in enumerate(low_dim_keys):
        data_key = f"low_dim_data__{idx}"
        if data_key not in demo_data:
            raise KeyError(f"NPZ missing '{data_key}' for low-dimensional key '{key}'.")
        low_dim_map[key] = np.asarray(demo_data[data_key], dtype=np.float32)
    return low_dim_map


def _load_pose_demo(demo_data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Load pose + gripper arrays for pose-mode replay."""
    low_dim = _load_low_dim_map(demo_data)
    result: dict[str, np.ndarray] = {
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


def _load_ctrl_demo(demo_data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    """Load joint ctrl arrays for ctrl-mode replay."""
    low_dim = _load_low_dim_map(demo_data)
    arm = low_dim["action/arm/joint_state/position"]
    eef_key = "action/eef/joint_state/position"
    if eef_key in low_dim:
        ctrl = np.concatenate([arm, low_dim[eef_key]], axis=-1)
    else:
        ctrl = arm
    return {"ctrl": ctrl}


class McapDemo:
    """Container for data loaded from a ROS2 mcap file."""

    joint: np.ndarray  # (T, n_arm + n_grip)
    joint_names: list[str]  # ordered joint names matching columns of *joint*

    def __init__(self, joint: np.ndarray, joint_names: list[str]) -> None:
        self.joint = joint
        self.joint_names = joint_names

    def first_frame_joint_positions(self) -> dict[str, float]:
        """Return {joint_name: value} for the first timestep."""
        return {n: float(v) for n, v in zip(self.joint_names, self.joint[0])}

    def align_to_actuators(
        self,
        actuator_names: list[str],
        name_mapping: dict[str, str] | None = None,
    ) -> None:
        """Reorder columns to match the actuator declaration order.

        Parameters
        ----------
        actuator_names:
            Target ordering, e.g. ``arm_actuators + eef_actuators`` from YAML.
        name_mapping:
            Optional mcap_name → actuator_name mapping for names that differ
            across domains (e.g. ``{"gripper": "xfg_claw_joint"}``).
        """
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
        print(f"Aligned mcap columns to actuator order: {self.joint_names}")

    def rescale_gripper(
        self,
        eef_actuator_names: list[str],
        src_range: list[float],
        env_cfg: DictConfig,
        base_dir: str | None = None,
    ) -> None:
        """Rescale real gripper distance to MuJoCo actuator ctrl.

        The real gripper publishes finger *distance* (closed → open), while
        the MuJoCo actuator ctrl for the supported grippers in this repo uses
        the opposite direction (open → closed). This method preserves the
        distance semantics, so a larger recorded value still means "more open"
        after converting to actuator ctrl.
        """
        import mujoco as mj
        from omegaconf import OmegaConf

        # Resolve model to read actuator ctrlrange.
        env_raw = OmegaConf.to_container(env_cfg, resolve=True)
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
            old_vals = self.joint[:, col].copy()
            self.joint[:, col] = _rescale_gripper_distance_to_ctrl(
                old_vals,
                src_range=[src_lo, src_hi],
                ctrl_range=[dst_lo, dst_hi],
            )
            print(
                f"Rescaled '{act_name}': "
                f"distance [{src_lo:.4f}, {src_hi:.4f}] (closed->open) -> "
                f"ctrl [{dst_hi:.4f}, {dst_lo:.4f}] (closed->open) "
                f"(was [{old_vals.min():.4f}, {old_vals.max():.4f}] → "
                f"[{self.joint[:, col].min():.4f}, {self.joint[:, col].max():.4f}])"
            )


def _load_mcap_demo(mcap_path: str, arm_topic: str, gripper_topic: str) -> McapDemo:
    """Load arm joint + gripper arrays from a ROS2 mcap file for joint-mode replay."""
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

    arm = np.array(arm_positions, dtype=np.float32)  # (T_arm, n_arm)
    grip = np.array(gripper_positions, dtype=np.float32)  # (T_grip, n_grip)
    arm_t = np.array(arm_times, dtype=np.int64)
    grip_t = np.array(gripper_times, dtype=np.int64)

    # Align gripper to arm timestamps via nearest-neighbour interpolation
    if len(arm_t) != len(grip_t) or np.any(np.abs(arm_t - grip_t) > 50_000_000):
        indices = np.searchsorted(grip_t, arm_t, side="left")
        indices = np.clip(indices, 0, len(grip_t) - 1)
        grip = grip[indices]

    joint = np.concatenate([arm, grip], axis=-1)
    joint_names = (arm_names or []) + (gripper_names or [])
    print(
        f"Loaded mcap: {len(joint)} steps, "
        f"arm={arm.shape[1]} joints, gripper={grip.shape[1]} joints, "
        f"names={joint_names}"
    )
    return McapDemo(joint=joint, joint_names=joint_names)


def normalize_demo_for_batch(
    demo: dict[str, np.ndarray],
    batch_size: int,
    mode: str,
) -> dict[str, np.ndarray]:
    """Slice a (T, B_rec, dim) demo to match the replay batch_size.

    Returns (T, B, dim) arrays, or (T, dim) when batch_size == 1.
    """
    if mode == "pose":
        position = demo["position"]  # (T, B_rec, 3)
        orientation = demo["orientation"]  # (T, B_rec, 4)
        rec_bs = position.shape[1]
        if batch_size > rec_bs:
            raise ValueError(
                f"Demo recorded with batch_size={rec_bs}, "
                f"but replay requires batch_size={batch_size}."
            )
        position = position[:, :batch_size, :]
        orientation = orientation[:, :batch_size, :]
        result: dict[str, np.ndarray] = {
            "position": position[:, 0, :] if batch_size == 1 else position,
            "orientation": orientation[:, 0, :] if batch_size == 1 else orientation,
        }
        if "gripper" in demo:
            gripper = demo["gripper"][:, :batch_size, :]
            result["gripper"] = gripper[:, 0, :] if batch_size == 1 else gripper
        return result

    if mode == "joint":
        joint = demo["joint"]  # (T, dim) from mcap — no batch dim
        if joint.ndim == 2:
            # mcap data is always single-env; broadcast if needed
            return {"joint": joint}
        rec_bs = joint.shape[1]
        if batch_size > rec_bs:
            raise ValueError(
                f"Demo recorded with batch_size={rec_bs}, "
                f"but replay requires batch_size={batch_size}."
            )
        joint = joint[:, :batch_size, :]
        return {"joint": joint[:, 0, :] if batch_size == 1 else joint}

    ctrl = demo["ctrl"]  # (T, B_rec, ctrl_dim)
    rec_bs = ctrl.shape[1]
    if batch_size > rec_bs:
        raise ValueError(
            f"Demo recorded with batch_size={rec_bs}, "
            f"but replay requires batch_size={batch_size}."
        )
    ctrl = ctrl[:, :batch_size, :]
    return {"ctrl": ctrl[:, 0, :] if batch_size == 1 else ctrl}


# ---------------------------------------------------------------------------
# Camera helpers
# ---------------------------------------------------------------------------


def _resolve_camera(
    obs: dict[str, dict], requested_camera: str
) -> tuple[str | None, str | None]:
    requested_full = requested_camera
    requested_short = requested_camera.removesuffix("_cam")
    candidates = (
        (requested_full, f"{requested_full}/color/image_raw"),
        (requested_short, f"{requested_short}_cam/color/image_raw"),
        (requested_short, f"/robot/camera/{requested_short}/color/image_raw"),
    )
    for camera_name, key in candidates:
        if obs.get(key, {}).get("data") is not None:
            return camera_name, key

    for key, payload in obs.items():
        if not key.endswith("/color/image_raw") or payload.get("data") is None:
            continue
        if key.startswith("/robot/camera/"):
            parts = key.split("/")
            if len(parts) >= 5 and "front" in parts[3]:
                return parts[3], key
        elif "front" in key.split("/")[0]:
            return key.split("/")[0], key
    return None, None


def _extract_frame(obs: dict[str, dict], camera_key: str) -> np.ndarray | None:
    data = obs.get(camera_key, {}).get("data")
    if data is None:
        return None
    frame = np.asarray(data, dtype=np.uint8)
    return frame[0] if frame.ndim >= 4 else frame


# ---------------------------------------------------------------------------
# Replay policy
# ---------------------------------------------------------------------------


class ReplayPolicy:
    """Replays recorded actions step by step (index-based, no observation)."""

    def __init__(self, demo: dict[str, np.ndarray], mode: str) -> None:
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

    def _action_at(self, index: int) -> dict[str, Any]:
        i = min(max(0, index), self._max)
        if self._mode == "pose":
            action: dict[str, Any] = {
                "position": self._demo["position"][i],
                "orientation": self._demo["orientation"][i],
            }
            if "gripper" in self._demo:
                action["gripper"] = self._demo["gripper"][i]
            return action
        if self._mode == "joint":
            return {"joint": self._demo["joint"][i]}
        return {"ctrl": self._demo["ctrl"][i]}

    def apply_first_frame_as_reset(self) -> dict[str, Any] | None:
        if self.num_steps <= 0:
            return None
        action = self._action_at(0)
        self._step = min(1, self.num_steps)
        return action

    def act(self) -> dict[str, Any]:
        action = self._action_at(self._step)
        self._step += 1
        return action


def apply_first_frame_reset(
    evaluator: PolicyEvaluator,
    policy: ReplayPolicy,
) -> dict[str, Any] | None:
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


# ---------------------------------------------------------------------------
# Action applier / observation getter
# ---------------------------------------------------------------------------


def action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    if action is None:
        return
    env = context.backend.env
    if "joint" in action:
        env.apply_joint_action(
            "arm", action["joint"], env_mask=env_mask, kinematic=False
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
        )


def make_observation_getter(frames: list[np.ndarray], camera: str) -> tuple[Any, ...]:
    """Build an observation_getter that captures camera frames as a side effect.

    Returns (observation_getter_fn, state_dict). The state_dict holds the
    resolved camera key (populated on first call).
    """
    state: dict[str, str | None] = {"camera_key": None, "camera_name": None}

    def observation_getter(context: ExecutionContext) -> dict:
        obs = context.backend.env.capture_observation()
        if state["camera_key"] is None:
            cam_name, cam_key = _resolve_camera(obs, camera)
            if cam_key is None:
                raise RuntimeError(
                    f"No usable camera found for '{camera}'. "
                    f"Available keys: {list(obs.keys())}"
                )
            state["camera_key"] = cam_key
            state["camera_name"] = cam_name
            if cam_name != camera:
                print(f"Camera '{camera}' not found, using '{cam_name}' instead.")
        frame = _extract_frame(obs, state["camera_key"])
        if frame is not None:
            frames.append(frame)
        return obs

    return observation_getter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    replay_cfg = ReplayConfig.model_validate(raw.pop("replay", {}))

    config_name = HydraConfig.get().job.config_name
    demo_name = replay_cfg.demo_name or config_name
    project_root = hydra.utils.get_original_cwd()
    video_dir = os.path.join(project_root, "outputs", "records", "videos")
    os.makedirs(video_dir, exist_ok=True)
    mp4_path = os.path.join(video_dir, f"{demo_name}_replay.mp4")
    gif_path = os.path.join(video_dir, f"{demo_name}_replay.gif")

    # --- Load demo data from mcap or npz ---
    mcap_demo: McapDemo | None = None
    if replay_cfg.mcap_path is not None:
        mcap_path = replay_cfg.mcap_path
        if not os.path.isabs(mcap_path):
            mcap_path = os.path.join(project_root, mcap_path)
        if not os.path.exists(mcap_path):
            raise FileNotFoundError(f"MCAP file not found: {mcap_path}")
        mcap_demo = _load_mcap_demo(
            mcap_path, replay_cfg.arm_topic, replay_cfg.gripper_topic
        )
        replay_cfg.mode = "joint"
        # Align mcap column order to the YAML actuator declaration order.
        op_cfg = cfg.env.operators.arm
        actuator_names = list(op_cfg.arm_actuators) + list(op_cfg.eef_actuators)
        mcap_demo.align_to_actuators(
            actuator_names, replay_cfg.joint_name_mapping or None
        )
        # Rescale gripper columns from real-robot distance range to the
        # MuJoCo actuator ctrlrange.
        mcap_demo.rescale_gripper(
            eef_actuator_names=list(op_cfg.eef_actuators),
            src_range=replay_cfg.gripper_range,
            env_cfg=cfg.env,
            base_dir=project_root,
        )
        # Inject first frame as initial_joint_positions so reset() places
        # the robot at the recorded starting configuration.
        init_jpos = mcap_demo.first_frame_joint_positions()
        with open_dict(cfg):
            if "initial_joint_positions" not in cfg.env:
                cfg.env.initial_joint_positions = {}
            cfg.env.initial_joint_positions.update(init_jpos)
        print(f"Injected initial_joint_positions: {init_jpos}")
        demo: dict[str, np.ndarray] = {"joint": mcap_demo.joint}
    else:
        demo_npz_path = os.path.join(
            project_root, "outputs", "records", "demos", f"{demo_name}.npz"
        )
        if not os.path.exists(demo_npz_path):
            raise FileNotFoundError(f"Demo file not found: {demo_npz_path}")
        demo_data = np.load(demo_npz_path)
        if replay_cfg.mode == "pose":
            demo = _load_pose_demo(demo_data)
        elif replay_cfg.mode == "ctrl":
            demo = _load_ctrl_demo(demo_data)
        else:
            raise ValueError(
                f"Unknown replay mode: {replay_cfg.mode!r} "
                f"(expected 'pose', 'ctrl', or set +replay.mcap_path=)"
            )

    # Disable randomization — replay must reproduce the exact recorded trajectory.
    with open_dict(cfg):
        if "task" in cfg and "randomization" in cfg.task:
            cfg.task.randomization = {}

    task_file = prepare_task_file(cfg)

    frames: list[np.ndarray] = []
    observation_getter = make_observation_getter(frames, replay_cfg.camera)

    evaluator = PolicyEvaluator(
        action_applier=action_applier,
        observation_getter=observation_getter,
    ).from_config(task_file)
    demo = normalize_demo_for_batch(
        demo,
        batch_size=evaluator.batch_size,
        mode=replay_cfg.mode,
    )

    policy = ReplayPolicy(demo, replay_cfg.mode)

    try:
        policy.reset()
        update = evaluator.reset()
        if replay_cfg.reset_from_first_frame:
            reset_action = apply_first_frame_reset(evaluator, policy)
            if reset_action is not None:
                print("Applied first recorded frame as reset state.")
                evaluator.get_observation()
        # evaluator.get_observation()  # initial frame
        print(f"Stages: {[s.name for s in task_file.task.stages]}")

        max_updates = policy.remaining_steps
        updates_used = 0
        for step in range(max_updates):
            input("Press Enter to continue to the next step...")
            action = policy.act()
            print(f"{action=}")
            for _ in range(10):
                update = evaluator.update(action)
            updates_used += 1
            # evaluator.get_observation()  # capture frame
            print(f"Replay step {step}: {update.stage_name}")
            if bool(np.all(update.done)):
                break

        summary = evaluator.summarize(
            update, max_updates=max_updates, updates_used=updates_used
        )
        print(
            f"\nCompleted {summary.completed_stage_count} stages "
            f"in {summary.updates_used} steps"
        )
        for r in evaluator.records:
            print(f"  {r.stage_name}: {r.status.value}")
        print(f"Success: {list(summary.final_success)}")
    finally:
        evaluator.close()

    if not frames:
        print("No frames captured during replay.")
        return

    if replay_cfg.save_mp4:
        iio.imwrite(mp4_path, frames, fps=replay_cfg.fps, codec="libx264", quality=8)
        print(f"Saved replay MP4 ({len(frames)} frames): {mp4_path}")

    if replay_cfg.save_gif:
        h, w = frames[0].shape[:2]
        gif_height = int(replay_cfg.gif_width * h / w)
        gif_frames = [
            np.array(Image.fromarray(f).resize((replay_cfg.gif_width, gif_height)))
            for f in frames
        ]
        gif_fps = min(replay_cfg.fps, 15)
        iio.imwrite(gif_path, gif_frames, fps=gif_fps, loop=0)
        print(
            f"Saved replay GIF ({len(gif_frames)} frames @ {gif_fps} fps): {gif_path}"
        )


if __name__ == "__main__":
    main()
