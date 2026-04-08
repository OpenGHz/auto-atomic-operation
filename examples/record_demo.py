"""Record a demo as MP4/GIF plus replayable low-dimensional data.

Uses the same config files as ``aao_demo``. Switch tasks with ``--config-name``
and override any value via Hydra:

    python examples/record_demo.py --config-name pick_and_place
    python examples/record_demo.py --config-name cup_on_coaster
    python examples/record_demo.py --config-name stack_color_blocks
    python examples/record_demo.py --config-name press_three_buttons

Video files are written to ``assets/videos/<config_name>.mp4`` and
``assets/videos/<config_name>.gif`` for single-env recording, or
``assets/videos/<config_name>_env<idx>.mp4`` / ``.gif`` when recording
multiple envs in parallel. Replay data is written to
``assets/demos/<config_name>.npz`` and ``assets/demos/<config_name>.json``.

Extra Hydra overrides:

    python examples/record_demo.py +recorder.camera=env2_cam
    python examples/record_demo.py +recorder.fps=15
    python examples/record_demo.py +recorder.gif_width=480
    python examples/record_demo.py +recorder.max_updates=200
"""

import json
import os
from dataclasses import asdict
import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, Field
from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import TaskRunner


class RecorderConfig(BaseModel):
    camera: str = Field(default="env1_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)
    max_updates: int | None = Field(default=None, ge=0)
    save_gif: bool = Field(default=False)
    save_mp4: bool = Field(default=False)
    save_demo: bool = Field(default=True)


def _is_low_dim_value(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim <= 2
    if isinstance(value, np.generic):
        return True
    if isinstance(value, (bool, int, float, str)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        return arr.ndim <= 1
    if isinstance(value, dict):
        return all(_is_low_dim_value(v) for v in value.values())
    return False


def _to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _extract_low_dim_observation(obs: dict[str, dict]) -> dict[str, dict]:
    low_dim: dict[str, dict] = {}
    excluded_suffixes = (
        "/color/image_raw",
        "/aligned_depth_to_color/image_raw",
        "/mask/image_raw",
        "/mask/heat_map",
        "/tactile/point_cloud2",
    )
    for key, payload in obs.items():
        if key.endswith(excluded_suffixes):
            continue
        data = payload.get("data")
        if not _is_low_dim_value(data):
            continue
        low_dim[key] = {
            "data": _to_jsonable(data),
            "t": _to_jsonable(payload.get("t")),
        }
    return low_dim


def _iter_low_dim_leaf_items(
    low_dim_step: dict[str, dict],
) -> list[tuple[str, object, object]]:
    items: list[tuple[str, object, object]] = []
    for key, payload in low_dim_step.items():
        data = payload.get("data")
        t = payload.get("t")
        if isinstance(data, dict):
            for field_name, field_value in data.items():
                items.append((f"{key}/{field_name}", field_value, t))
        else:
            items.append((key, data, t))
    return items


def _build_low_dim_npz_payload(
    low_dim_observations: list[dict[str, dict]],
) -> dict[str, np.ndarray]:
    leaf_keys = sorted(
        {
            leaf_key
            for step in low_dim_observations
            for leaf_key, _, _ in _iter_low_dim_leaf_items(step)
        }
    )
    payload: dict[str, np.ndarray] = {"low_dim_keys": np.asarray(leaf_keys, dtype=str)}
    for idx, leaf_key in enumerate(leaf_keys):
        values: list[np.ndarray] = []
        times: list[float] = []
        for step in low_dim_observations:
            leaf_items = {k: (v, t) for k, v, t in _iter_low_dim_leaf_items(step)}
            if leaf_key not in leaf_items:
                raise ValueError(f"Missing low-dimensional key '{leaf_key}' in trace.")
            value, t = leaf_items[leaf_key]
            values.append(np.asarray(value, dtype=np.float32).reshape(-1))
            t_scalar = t[0] if isinstance(t, (list, np.ndarray)) else t
            times.append(float(t_scalar))
        payload[f"low_dim_data__{idx}"] = np.stack(values).astype(np.float32)
        payload[f"low_dim_t__{idx}"] = np.asarray(times, dtype=np.float64)
    return payload


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


def _split_frame_batch(data: object) -> list[np.ndarray]:
    frame = np.asarray(data, dtype=np.uint8)
    if frame.ndim == 3:
        return [frame]
    if frame.ndim == 4:
        return [np.asarray(single, dtype=np.uint8) for single in frame]
    raise TypeError(
        f"Expected camera frame with shape (H, W, C) or (B, H, W, C), got {frame.shape}"
    )


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    # Recorder settings (injectable via Hydra: recorder.camera=..., etc.)
    rec_cfg = RecorderConfig.model_validate(raw.pop("recorder", {}))
    task_file = prepare_task_file(cfg)
    runner = TaskRunner().from_config(task_file)

    frames_by_env: list[list[np.ndarray]] = []
    low_dim_observations: list[dict[str, dict]] = []
    action_trace: list[np.ndarray] = []
    update_trace: list[dict] = []
    resolved_camera: str | None = None
    resolved_camera_key: str | None = None
    batch_size = 0
    updates_used = 0
    stopped_due_to_max_updates = False

    def capture() -> None:
        backend = runner._context and runner._context.backend
        if not isinstance(backend, MujocoTaskBackend):
            return
        obs = backend.env.capture_observation()
        low_dim_observations.append(_extract_low_dim_observation(obs))
        nonlocal resolved_camera, resolved_camera_key
        if resolved_camera_key is None:
            resolved_camera, resolved_camera_key = _resolve_camera(obs, rec_cfg.camera)
            if resolved_camera_key is None:
                raise RuntimeError(
                    f"No usable camera found for '{rec_cfg.camera}'. "
                    f"Available keys: {list(obs.keys())}"
                )
            if resolved_camera != rec_cfg.camera:
                print(
                    f"Camera '{rec_cfg.camera}' not found, using '{resolved_camera}' instead."
                )
        data = obs.get(resolved_camera_key, {}).get("data")
        if data is not None:
            batch_frames = _split_frame_batch(data)
            nonlocal batch_size
            if batch_size == 0:
                batch_size = len(batch_frames)
                frames_by_env.extend([] for _ in range(batch_size))
            elif len(batch_frames) != batch_size:
                raise RuntimeError(
                    "Observed camera batch size changed during recording: "
                    f"expected {batch_size}, got {len(batch_frames)}"
                )
            for env_index, env_frame in enumerate(batch_frames):
                frames_by_env[env_index].append(env_frame)

    try:
        print("Reset task")
        reset_update = runner.reset()
        print(reset_update)
        capture()

        while True:
            if rec_cfg.max_updates is not None and updates_used >= rec_cfg.max_updates:
                stopped_due_to_max_updates = True
                print(f"Reached recorder.max_updates={rec_cfg.max_updates}, stopping.")
                break

            update = runner.update()
            updates_used += 1
            backend = runner._context and runner._context.backend
            if isinstance(backend, MujocoTaskBackend):
                action_trace.append(
                    np.stack(
                        [
                            np.asarray(
                                env.data.ctrl[: env.model.nu], dtype=np.float32
                            ).copy()
                            for env in backend.env.envs
                        ],
                        axis=0,
                    )
                )
            update_trace.append(_to_jsonable(asdict(update)))
            capture()
            print(update)
            if bool(np.all(update.done)):
                break

        print()
        print("Execution records:")
        for record in runner.records:
            print(record)
    finally:
        runner.close()

    if not frames_by_env:
        print("No frames captured — is the backend a MujocoTaskBackend?")
        return

    config_name = HydraConfig.get().job.config_name
    project_root = hydra.utils.get_original_cwd()
    video_dir = os.path.join(project_root, "outputs/records", "videos")
    demo_dir = os.path.join(project_root, "outputs/records", "demos")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)
    demo_npz_path = os.path.join(demo_dir, f"{config_name}.npz")
    demo_json_path = os.path.join(demo_dir, f"{config_name}.json")

    def build_video_path(ext: str, env_index: int) -> str:
        suffix = "" if batch_size == 1 else f"_env{env_index}"
        return os.path.join(video_dir, f"{config_name}{suffix}.{ext}")

    if rec_cfg.save_mp4:
        for env_index, env_frames in enumerate(frames_by_env):
            mp4_path = build_video_path("mp4", env_index)
            iio.imwrite(
                mp4_path, env_frames, fps=rec_cfg.fps, codec="libx264", quality=8
            )
            print(
                f"\nSaved MP4 for env {env_index} "
                f"({len(env_frames)} frames @ {rec_cfg.fps} fps): {mp4_path}"
            )

    if rec_cfg.save_gif:
        gif_fps = min(rec_cfg.fps, 15)
        for env_index, env_frames in enumerate(frames_by_env):
            h, w = env_frames[0].shape[:2]
            gif_height = int(rec_cfg.gif_width * h / w)
            gif_frames = [
                np.array(Image.fromarray(f).resize((rec_cfg.gif_width, gif_height)))
                for f in env_frames
            ]
            gif_path = build_video_path("gif", env_index)
            iio.imwrite(gif_path, gif_frames, fps=gif_fps, loop=0)
            print(
                f"Saved GIF for env {env_index} "
                f"({len(gif_frames)} frames @ {gif_fps} fps): {gif_path}"
            )

    if rec_cfg.save_demo:
        action_array = (
            np.stack(action_trace).astype(np.float32)
            if action_trace
            else np.zeros((0, 0), dtype=np.float32)
        )
        npz_payload: dict[str, np.ndarray] = {"actions": action_array}
        npz_payload.update(_build_low_dim_npz_payload(low_dim_observations))
        np.savez_compressed(demo_npz_path, **npz_payload)
        with open(demo_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config_name": config_name,
                    "camera": resolved_camera,
                    "fps": rec_cfg.fps,
                    "max_updates": rec_cfg.max_updates,
                    "updates_used": updates_used,
                    "stopped_due_to_max_updates": stopped_due_to_max_updates,
                    "num_frames": len(frames_by_env[0]) if frames_by_env else 0,
                    "num_frames_per_env": [
                        len(env_frames) for env_frames in frames_by_env
                    ],
                    "num_actions": int(action_array.shape[0]),
                    "batch_size": batch_size,
                    "action_dim": int(action_array.shape[2])
                    if action_array.ndim == 3
                    else 0,
                    "reset_update": _to_jsonable(asdict(reset_update)),
                    "step_updates": update_trace,
                    "low_dim_observations": low_dim_observations,
                    "execution_records": [
                        _to_jsonable(asdict(record)) for record in runner.records
                    ],
                },
                f,
                indent=2,
            )
        print(
            f"Saved demo data ({len(low_dim_observations)} observations, "
            f"{len(action_trace)} actions): {demo_npz_path}"
        )
        print(f"Saved demo metadata: {demo_json_path}")


if __name__ == "__main__":
    main()
