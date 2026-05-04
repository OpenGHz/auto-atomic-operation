"""Replay recorded low-level actions with the DataReplayRunner.

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
        +replay.gripper_topic=/robot/right_gripper/distance \
        +replay.base_topic=/robot/base_pose \
        +replay.scene_joint_topic=/scene/door/joint_states
"""

from __future__ import annotations

import os
from collections import defaultdict

import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from PIL import Image
from pydantic import Field

from auto_atom import ExecutionContext
from auto_atom.runner.common import get_config_dir
from auto_atom.runner.data_replay import (
    DataReplayConfig,
    DataReplayRunner,
    ReplayPolicy,
    normalize_demo_for_batch,
)
from auto_atom.runner.data_replay import (
    _apply_first_frame_reset as apply_first_frame_reset,
)

# ---------------------------------------------------------------------------
# Script-level config (extends DataReplayConfig with video options)
# ---------------------------------------------------------------------------


class ReplayScriptConfig(DataReplayConfig):
    """Replay settings including video/gif output options."""

    steps_per_action: int = Field(default=1)
    camera: str | list[str] = Field(default="env1_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)
    save_gif: bool = Field(default=False)
    save_mp4: bool = Field(default=False)


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


def _extract_frames(obs: dict[str, dict], camera_key: str) -> list[np.ndarray]:
    data = obs.get(camera_key, {}).get("data")
    if data is None:
        return []
    frame = np.asarray(data, dtype=np.uint8)
    if frame.ndim >= 4:
        return [np.asarray(env_frame, dtype=np.uint8) for env_frame in frame]
    return [frame]


def make_observation_getter(
    frames_by_camera_env: dict[str, dict[int, list[np.ndarray]]], cameras: list[str]
) -> tuple:
    """Build an observation_getter that captures camera frames as a side effect.

    ``frames_by_camera_env`` is keyed by requested camera name, then by env
    index. The resolved camera key/name is cached after the first observation.
    """
    resolved: dict[str, dict[str, str | None]] = {
        cam: {"camera_key": None, "camera_name": None} for cam in cameras
    }

    def observation_getter(context: ExecutionContext) -> dict:
        obs = context.backend.env.capture_observation()
        for cam in cameras:
            state = resolved[cam]
            if state["camera_key"] is None:
                cam_name, cam_key = _resolve_camera(obs, cam)
                if cam_key is None:
                    raise RuntimeError(
                        f"No usable camera found for '{cam}'. "
                        f"Available keys: {list(obs.keys())}"
                    )
                state["camera_key"] = cam_key
                state["camera_name"] = cam_name
                if cam_name != cam:
                    print(f"Camera '{cam}' not found, using '{cam_name}' instead.")
            for env_index, frame in enumerate(
                _extract_frames(obs, state["camera_key"])
            ):
                frames_by_camera_env[cam][env_index].append(frame)
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

    script_cfg = ReplayScriptConfig.model_validate(raw.pop("replay", {}))

    config_name = HydraConfig.get().job.config_name
    demo_name = script_cfg.demo_name or config_name
    project_root = hydra.utils.get_original_cwd()
    video_dir = os.path.join(project_root, "outputs", "records", "videos")
    os.makedirs(video_dir, exist_ok=True)

    cameras = (
        [script_cfg.camera]
        if isinstance(script_cfg.camera, str)
        else list(script_cfg.camera)
    )

    # --- Set replay overrides on DictConfig ---
    script_only_keys = ("camera", "fps", "gif_width", "save_gif", "save_mp4")
    with open_dict(cfg):
        if "replay" not in cfg:
            cfg.replay = {}
        for key in script_only_keys:
            cfg.replay.pop(key, None)
        cfg.replay.demo_name = demo_name
        cfg.replay.demo_dir = os.path.join(project_root, "outputs", "records", "demos")

    # --- Camera frame capture ---
    frames_by_camera_env: dict[str, dict[int, list[np.ndarray]]] = {
        cam: defaultdict(list) for cam in cameras
    }
    observation_getter = make_observation_getter(frames_by_camera_env, cameras)

    # --- Run replay via DataReplayRunner ---
    runner = DataReplayRunner(
        observation_getter=observation_getter,
    ).from_config(cfg)

    try:
        runner.reset()
        total_actions = runner.remaining_steps
        spa = script_cfg.steps_per_action
        max_updates = total_actions * spa
        print(
            f"Replay mode: {script_cfg.mode}, "
            f"actions: {total_actions}, "
            f"steps_per_action: {spa}"
        )

        capture_frames = script_cfg.save_gif or script_cfg.save_mp4
        updates_used = 0
        for step in range(max_updates):
            update = runner.update()
            if capture_frames:
                runner.get_observation()
            updates_used += 1
            print(f"Replay step {step}: {update.stage_name}")
            if bool(np.all(update.done)):
                break

        summary = runner.summarize(
            update, max_updates=max_updates, updates_used=updates_used
        )
        print(
            f"\nCompleted {summary.completed_stage_count} stages "
            f"in {summary.updates_used} steps"
        )
        print(f"Success: {list(summary.final_success)}")
    finally:
        runner.close()

    # --- Save video (one file per camera/env pair) ---
    total_frames = sum(
        len(frames)
        for frames_by_env in frames_by_camera_env.values()
        for frames in frames_by_env.values()
    )
    if total_frames == 0:
        print("No frames captured during replay.")
        return

    suffix_per_cam = len(cameras) > 1
    suffix_per_env = any(
        len(frames_by_env) > 1 for frames_by_env in frames_by_camera_env.values()
    )
    for cam, frames_by_env in frames_by_camera_env.items():
        if not frames_by_env:
            print(f"No frames captured for camera '{cam}'.")
            continue

        for env_index, frames in sorted(frames_by_env.items()):
            if not frames:
                print(f"No frames captured for camera '{cam}' env {env_index}.")
                continue

            env_tag = f"_env{env_index}" if suffix_per_env else ""
            cam_tag = f"_{cam}" if suffix_per_cam else ""
            tag = f"{env_tag}{cam_tag}"
            mp4_path = os.path.join(video_dir, f"{demo_name}{tag}_replay.mp4")
            gif_path = os.path.join(video_dir, f"{demo_name}{tag}_replay.gif")

            if script_cfg.save_mp4:
                iio.imwrite(
                    mp4_path, frames, fps=script_cfg.fps, codec="libx264", quality=8
                )
                print(f"Saved replay MP4 ({len(frames)} frames): {mp4_path}")

            if script_cfg.save_gif:
                h, w = frames[0].shape[:2]
                gif_height = int(script_cfg.gif_width * h / w)
                gif_frames = [
                    np.array(
                        Image.fromarray(f).resize((script_cfg.gif_width, gif_height))
                    )
                    for f in frames
                ]
                gif_fps = min(script_cfg.fps, 15)
                iio.imwrite(gif_path, gif_frames, fps=gif_fps, loop=0)
                print(
                    f"Saved replay GIF ({len(gif_frames)} frames @ {gif_fps} fps): "
                    f"{gif_path}"
                )


if __name__ == "__main__":
    main()
