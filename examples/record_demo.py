"""Record a demo as MP4/GIF plus replayable low-dimensional data.

Uses the same config files as ``run_demo.py``. Switch tasks with ``--config-name``
and override any value via Hydra:

    python examples/record_demo.py --config-name pick_and_place
    python examples/record_demo.py --config-name cup_on_coaster
    python examples/record_demo.py --config-name stack_color_blocks
    python examples/record_demo.py --config-name press_three_buttons

Video files are written to ``assets/videos/<config_name>.mp4`` and
``assets/videos/<config_name>.gif``. Replay data is written to
``assets/demos/<config_name>.npz`` and ``assets/demos/<config_name>.json``.

Extra Hydra overrides:

    python examples/record_demo.py +recorder.camera=side_cam
    python examples/record_demo.py +recorder.fps=15
    python examples/record_demo.py +recorder.gif_width=480
"""

import json
import os
from dataclasses import asdict
import hydra
import imageio.v3 as iio
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, Field
from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


class RecorderConfig(BaseModel):
    camera: str = Field(default="front_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)
    save_gif: bool = Field(default=False)
    save_mp4: bool = Field(default=False)
    save_demo: bool = Field(default=True)


def _is_low_dim_value(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim <= 1
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
            values.append(np.asarray(value, dtype=np.float32))
            times.append(float(t))
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


@hydra.main(config_path="mujoco", config_name="pick_and_place", version_base=None)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)
    ComponentRegistry.clear()
    instantiate(cfg)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    # Recorder settings (injectable via Hydra: recorder.camera=..., etc.)
    rec_cfg = RecorderConfig.model_validate(raw.pop("recorder", {}))
    task_file = TaskFileConfig.model_validate(raw)
    runner = TaskRunner().from_config(task_file)

    frames: list[np.ndarray] = []
    low_dim_observations: list[dict[str, dict]] = []
    action_trace: list[np.ndarray] = []
    update_trace: list[dict] = []
    resolved_camera: str | None = None
    resolved_camera_key: str | None = None

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
            frame = np.asarray(data, dtype=np.uint8)
            frames.append(frame[0] if frame.ndim >= 4 else frame)

    try:
        print("Reset task")
        reset_update = runner.reset()
        print(reset_update)
        capture()

        while True:
            update = runner.update()
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

    if not frames:
        print("No frames captured — is the backend a MujocoTaskBackend?")
        return

    config_name = HydraConfig.get().job.config_name
    project_root = hydra.utils.get_original_cwd()
    video_dir = os.path.join(project_root, "assets", "videos")
    demo_dir = os.path.join(project_root, "assets", "demos")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(demo_dir, exist_ok=True)
    mp4_path = os.path.join(video_dir, f"{config_name}.mp4")
    gif_path = os.path.join(video_dir, f"{config_name}.gif")
    demo_npz_path = os.path.join(demo_dir, f"{config_name}.npz")
    demo_json_path = os.path.join(demo_dir, f"{config_name}.json")

    # Write MP4
    if rec_cfg.save_mp4:
        iio.imwrite(mp4_path, frames, fps=rec_cfg.fps, codec="libx264", quality=8)
        print(f"\nSaved MP4 ({len(frames)} frames @ {rec_cfg.fps} fps): {mp4_path}")

    # Resize frames for GIF
    if rec_cfg.save_gif:
        h, w = frames[0].shape[:2]
        gif_height = int(rec_cfg.gif_width * h / w)
        gif_frames = [
            np.array(Image.fromarray(f).resize((rec_cfg.gif_width, gif_height)))
            for f in frames
        ]
        gif_fps = min(rec_cfg.fps, 15)
        iio.imwrite(gif_path, gif_frames, fps=gif_fps, loop=0)
        print(f"Saved GIF  ({len(gif_frames)} frames @ {gif_fps} fps): {gif_path}")

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
                    "num_frames": len(frames),
                    "num_actions": int(action_array.shape[0]),
                    "batch_size": int(action_array.shape[1])
                    if action_array.ndim >= 2
                    else 0,
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
