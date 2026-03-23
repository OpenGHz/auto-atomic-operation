"""Record a demo as MP4 and GIF using the MuJoCo backend.

Uses the same config files as ``run_demo.py``. Switch tasks with ``--config-name``
and override any value via Hydra:

    python examples/record_demo.py --config-name pick_and_place
    python examples/record_demo.py --config-name cup_on_coaster
    python examples/record_demo.py --config-name stack_color_blocks
    python examples/record_demo.py --config-name press_three_buttons

Output files are written to ``assets/videos/<config_name>.mp4`` and
``assets/videos/<config_name>.gif``.

Extra Hydra overrides:

    python examples/record_demo.py recorder.camera=side_cam
    python examples/record_demo.py recorder.fps=15
    python examples/record_demo.py recorder.gif_width=480
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
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


class RecorderConfig(BaseModel):
    camera: str = Field(default="front_cam")
    fps: int = Field(default=25)
    gif_width: int = Field(default=320)


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
    resolved_camera: str | None = None

    def resolve_camera(obs: dict) -> str | None:
        """Return the camera key to use, or None if no suitable camera found."""
        preferred = f"{rec_cfg.camera}/color/image_raw"
        if obs.get(preferred, {}).get("data") is not None:
            return rec_cfg.camera
        # Fall back to any camera whose name contains "front"
        for key in obs:
            cam_name = key.split("/")[0]
            if "front" in cam_name and obs[key].get("data") is not None:
                print(
                    f"Camera '{rec_cfg.camera}' not found, using '{cam_name}' instead."
                )
                return cam_name
        return None

    def capture() -> None:
        backend = runner._context and runner._context.backend
        if not isinstance(backend, MujocoTaskBackend):
            return
        obs = backend.env.capture_observation()
        nonlocal resolved_camera
        if resolved_camera is None:
            resolved_camera = resolve_camera(obs)
            if resolved_camera is None:
                raise RuntimeError(
                    f"No usable front camera found in observation. "
                    f"Available keys: {list(obs.keys())}"
                )
        data = obs.get(f"{resolved_camera}/color/image_raw", {}).get("data")
        if data is not None:
            frames.append(np.asarray(data, dtype=np.uint8))

    try:
        print("Reset task")
        print(runner.reset())
        capture()

        while True:
            update = runner.update()
            capture()
            print(update)
            if update.done:
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
    out_dir = os.path.join(hydra.utils.get_original_cwd(), "assets", "videos")
    os.makedirs(out_dir, exist_ok=True)
    mp4_path = os.path.join(out_dir, f"{config_name}.mp4")
    gif_path = os.path.join(out_dir, f"{config_name}.gif")

    # # Write MP4
    # iio.imwrite(mp4_path, frames, fps=rec_cfg.fps, codec="libx264", quality=8)
    # print(f"\nSaved MP4 ({len(frames)} frames @ {rec_cfg.fps} fps): {mp4_path}")

    # Resize frames for GIF
    h, w = frames[0].shape[:2]
    gif_height = int(rec_cfg.gif_width * h / w)
    gif_frames = [
        np.array(Image.fromarray(f).resize((rec_cfg.gif_width, gif_height)))
        for f in frames
    ]
    gif_fps = min(rec_cfg.fps, 15)
    iio.imwrite(gif_path, gif_frames, fps=gif_fps, loop=0)
    print(f"Saved GIF  ({len(gif_frames)} frames @ {gif_fps} fps): {gif_path}")


if __name__ == "__main__":
    main()
