"""Replay recorded low-level actions with the PolicyEvaluator.

By default this script loads ``outputs/records/demos/<config_name>.npz``
produced by ``record_demo.py`` and replays the saved sequence step by step.

Examples:

    python examples/replay_demo.py --config-name press_three_buttons
    python examples/replay_demo.py --config-name pick_and_place +replay.mode=ctrl
    python examples/replay_demo.py --config-name pick_and_place +replay.save_gif=true
    python examples/replay_demo.py --config-name pick_and_place +replay.demo_name=my_demo
"""

from __future__ import annotations

import os
from typing import Any, Optional

import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, Field

from auto_atom import ExecutionContext, PolicyEvaluator, TaskUpdate
from auto_atom.runner.common import get_config_dir, prepare_task_file


class ReplayConfig(BaseModel):
    demo_name: str | None = Field(default=None)
    mode: str = Field(default="pose")
    camera: str = Field(default="env1_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)
    save_gif: bool = Field(default=True)
    save_mp4: bool = Field(default=False)


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
        else:
            self._max = len(demo["ctrl"]) - 1
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    @property
    def num_steps(self) -> int:
        return self._max + 1

    def act(self) -> dict[str, Any]:
        i = min(self._step, self._max)
        self._step += 1
        if self._mode == "pose":
            action: dict[str, Any] = {
                "position": self._demo["position"][i],
                "orientation": self._demo["orientation"][i],
            }
            if "gripper" in self._demo:
                action["gripper"] = self._demo["gripper"][i]
            return action
        return {"ctrl": self._demo["ctrl"][i]}


# ---------------------------------------------------------------------------
# Action applier / observation getter
# ---------------------------------------------------------------------------


def action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    if action is None:
        return
    env = context.backend.env
    if "ctrl" in action:
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
    task_file = prepare_task_file(cfg)

    config_name = HydraConfig.get().job.config_name
    demo_name = replay_cfg.demo_name or config_name
    project_root = hydra.utils.get_original_cwd()
    demo_npz_path = os.path.join(
        project_root, "outputs", "records", "demos", f"{demo_name}.npz"
    )
    demo_json_path = os.path.join(
        project_root, "outputs", "records", "demos", f"{demo_name}.json"
    )
    video_dir = os.path.join(project_root, "outputs", "records", "videos")
    os.makedirs(video_dir, exist_ok=True)
    mp4_path = os.path.join(video_dir, f"{demo_name}_replay.mp4")
    gif_path = os.path.join(video_dir, f"{demo_name}_replay.gif")

    if not os.path.exists(demo_npz_path):
        raise FileNotFoundError(f"Demo file not found: {demo_npz_path}")

    demo_data = np.load(demo_npz_path)
    if replay_cfg.mode == "pose":
        demo = _load_pose_demo(demo_data)
    elif replay_cfg.mode == "ctrl":
        demo = _load_ctrl_demo(demo_data)
    else:
        raise ValueError(
            f"Unknown replay mode: {replay_cfg.mode!r} (expected 'pose' or 'ctrl')"
        )

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
        # evaluator.get_observation()  # initial frame
        print(f"Stages: {[s.name for s in task_file.task.stages]}")

        step = -1
        for step in range(policy.num_steps):
            action = policy.act()
            update = evaluator.update(action)
            # evaluator.get_observation()  # capture frame
            print(f"Replay step {step}: {update.stage_name}")
            if bool(np.all(update.done)):
                break

        summary = evaluator.summarize(
            update, max_updates=policy.num_steps, updates_used=step + 1
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
