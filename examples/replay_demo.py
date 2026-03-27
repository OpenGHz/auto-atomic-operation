"""Replay recorded low-level actions with the MuJoCo backend.

By default this script loads ``assets/demos/<config_name>.npz`` produced by
``record_demo.py`` and replays the saved control sequence step by step.

Examples:

    python examples/replay_demo.py --config-name open_hinge_door
    python examples/replay_demo.py --config-name pick_and_place +replay.save_gif=true
    python examples/replay_demo.py --config-name pick_and_place +replay.demo_name=my_demo
"""

import os
import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, Field
from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.utils.pose import PoseState
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


class ReplayConfig(BaseModel):
    demo_name: str | None = Field(default=None)
    mode: str = Field(default="ctrl")
    camera: str = Field(default="front_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)
    save_gif: bool = Field(default=True)
    save_mp4: bool = Field(default=False)


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


def _load_pose_trace(
    demo_data: np.lib.npyio.NpzFile,
) -> list[dict[str, dict[str, object]]]:
    low_dim_map = _load_low_dim_map(demo_data)
    operators: dict[str, dict[str, np.ndarray]] = {}

    for key, values in low_dim_map.items():
        parts = key.split("/")
        if len(parts) >= 4 and parts[0] == "action" and parts[2] == "pose":
            operator = parts[1]
            field_name = parts[3]
        elif (
            len(parts) >= 6
            and parts[0] == ""
            and parts[1] == "robot"
            and parts[2] == "action"
            and parts[4] == "pose"
        ):
            operator = parts[3]
            field_name = parts[5]
        else:
            continue
        if field_name not in {"position", "orientation"}:
            continue
        operators.setdefault(operator, {})[field_name] = values

    complete_ops = sorted(
        operator
        for operator, fields in operators.items()
        if "position" in fields and "orientation" in fields
    )
    if not complete_ops:
        raise KeyError("NPZ does not contain action pose position/orientation series.")

    num_steps = operators[complete_ops[0]]["position"].shape[0]
    pose_trace: list[dict[str, dict[str, object]]] = []
    for step_idx in range(num_steps):
        step_pose: dict[str, dict[str, object]] = {}
        for operator in complete_ops:
            step_pose[operator] = {
                "position": operators[operator]["position"][step_idx],
                "orientation": operators[operator]["orientation"][step_idx],
            }
        pose_trace.append(step_pose)
    return pose_trace


def _apply_pose_targets(
    backend: MujocoTaskBackend,
    pose_targets: dict[str, dict[str, object]],
    ctrl_action: np.ndarray | None,
) -> None:
    env = backend.env
    joint_ctrl = np.asarray(env.data.ctrl[: env.model.nu], dtype=np.float64).copy()
    if ctrl_action is not None:
        n = min(len(ctrl_action), env.model.nu)
        if n > 0:
            joint_ctrl[:n] = np.asarray(ctrl_action[:n], dtype=np.float64)

    needs_step = False
    needs_update = False

    for operator_name, target in pose_targets.items():
        position = np.asarray(target.get("position"), dtype=np.float32)
        orientation = np.asarray(target.get("orientation"), dtype=np.float32)
        state = env._get_op(operator_name)
        state.target_pos_in_base = position.copy()
        state.target_quat_in_base = orientation.copy()

        if state.joint_mode:
            arm_qidx = env._op_arm_qidx[operator_name]
            current_arm_qpos = env.data.qpos[arm_qidx].copy()
            joint_targets = state.ik_solver.solve(
                PoseState(
                    position=tuple(float(v) for v in position),
                    orientation=tuple(float(v) for v in orientation),
                ),
                current_arm_qpos,
            )
            if joint_targets is not None:
                arm_aidx = env._op_arm_aidx[operator_name]
                joint_ctrl[arm_aidx] = joint_targets
            needs_step = True
        else:
            base_body_pos, base_body_quat_xyzw = env._eef_in_base_to_base_body_world(
                state, position, orientation
            )
            env._write_mocap_pose(state, base_body_pos, base_body_quat_xyzw)
            needs_update = True

    if ctrl_action is not None and env.model.nu > 0:
        low = env.model.actuator_ctrlrange[: env.model.nu, 0]
        high = env.model.actuator_ctrlrange[: env.model.nu, 1]
        joint_ctrl[: env.model.nu] = np.clip(joint_ctrl[: env.model.nu], low, high)
        env.data.ctrl[: env.model.nu] = joint_ctrl[: env.model.nu]

    if needs_step:
        env.step(joint_ctrl)
    elif needs_update or ctrl_action is not None:
        env.update()


@hydra.main(config_path="mujoco", config_name="pick_and_place", version_base=None)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)
    ComponentRegistry.clear()
    instantiate(cfg)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    replay_cfg = ReplayConfig.model_validate(raw.pop("replay", {}))
    task_file = TaskFileConfig.model_validate(raw)
    runner = TaskRunner().from_config(task_file)

    config_name = HydraConfig.get().job.config_name
    demo_name = replay_cfg.demo_name or config_name
    project_root = hydra.utils.get_original_cwd()
    demo_npz_path = os.path.join(project_root, "assets", "demos", f"{demo_name}.npz")
    video_dir = os.path.join(project_root, "assets", "videos")
    os.makedirs(video_dir, exist_ok=True)
    mp4_path = os.path.join(video_dir, f"{demo_name}_replay.mp4")
    gif_path = os.path.join(video_dir, f"{demo_name}_replay.gif")

    if not os.path.exists(demo_npz_path):
        raise FileNotFoundError(f"Demo action file not found: {demo_npz_path}")

    demo_data = np.load(demo_npz_path)
    actions = np.asarray(demo_data["actions"], dtype=np.float32)
    pose_trace = _load_pose_trace(demo_data) if replay_cfg.mode == "pose" else []

    frames: list[np.ndarray] = []
    resolved_camera: str | None = None
    resolved_camera_key: str | None = None

    def capture() -> None:
        backend = runner._context and runner._context.backend
        if not isinstance(backend, MujocoTaskBackend):
            return
        obs = backend.env.capture_observation()
        nonlocal resolved_camera, resolved_camera_key
        if resolved_camera_key is None:
            resolved_camera, resolved_camera_key = _resolve_camera(
                obs, replay_cfg.camera
            )
            if resolved_camera_key is None:
                raise RuntimeError(
                    f"No usable camera found for '{replay_cfg.camera}'. "
                    f"Available keys: {list(obs.keys())}"
                )
            if resolved_camera != replay_cfg.camera:
                print(
                    f"Camera '{replay_cfg.camera}' not found, using '{resolved_camera}' instead."
                )
        data = obs.get(resolved_camera_key, {}).get("data")
        if data is not None:
            frames.append(np.asarray(data, dtype=np.uint8))

    try:
        print("Reset task for replay")
        print(runner.reset())
        backend = runner._context and runner._context.backend
        if not isinstance(backend, MujocoTaskBackend):
            raise TypeError("Replay currently only supports MujocoTaskBackend.")

        capture()
        num_steps = len(pose_trace) if replay_cfg.mode == "pose" else len(actions)
        for i in range(num_steps):
            action = actions[i] if i < len(actions) else None
            if replay_cfg.mode == "pose":
                pose_targets = pose_trace[i]
                _apply_pose_targets(backend, pose_targets, action)
                print(
                    f"Replay step {i}: mode=pose operators={sorted(pose_targets.keys())}"
                )
            else:
                if action is None:
                    break
                backend.env.step(action)
                print(f"Replay step {i}: mode=ctrl action_dim={len(action)}")
            capture()
    finally:
        runner.close()

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
