"""Save per-camera GS mask images for visual inspection.

This script loads a GS-enabled scene config, resets to the initial keyframe,
renders GS RGB plus GS-derived binary masks for every configured camera, and
saves the outputs under ``outputs/gs_mask_<config>_<timestamp>/``.

Saved files per camera:

- ``<camera>_rgb.png``
- ``<camera>_mask.png``
- ``<camera>_overlay.png``
- ``<camera>_heat_<operation>.png`` (when heat-map channels exist)

Usage::

    python examples/save_gs_mask_render.py
    python examples/save_gs_mask_render.py --config-name press_three_buttons_gs
    python examples/save_gs_mask_render.py show=true
    python examples/save_gs_mask_render.py +recorder.enabled=true
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


class MaskRecorderConfig(BaseModel):
    enabled: bool = Field(default=False)
    fps: int = Field(default=25)
    max_steps: int = Field(default=300)
    save_mp4: bool = Field(default=True)
    save_gif: bool = Field(default=False)
    video_stream: str = Field(default="overlay")
    """One of: overlay, rgb, mask."""


def _resolve_single_env(env):
    if hasattr(env, "envs"):
        return env.envs[0]
    return env


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    plt.imsave(path, np.asarray(rgb, dtype=np.uint8))


def _save_mask(path: Path, mask: np.ndarray) -> None:
    plt.imsave(
        path, np.asarray(mask, dtype=np.uint8) * 255, cmap="gray", vmin=0, vmax=255
    )


def _make_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.asarray(rgb, dtype=np.float32).copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.ndim != 2:
        raise TypeError(f"Expected 2D mask, got shape {mask_bool.shape}")
    overlay_color = np.array([255.0, 64.0, 64.0], dtype=np.float32)
    alpha = 0.45
    base[mask_bool] = (1.0 - alpha) * base[mask_bool] + alpha * overlay_color
    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def _select_video_frame(
    stream: str,
    rgb: np.ndarray,
    mask: np.ndarray,
    overlay: np.ndarray,
) -> np.ndarray:
    if stream == "overlay":
        return overlay
    if stream == "rgb":
        return rgb
    if stream == "mask":
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=-1)
    raise ValueError(f"Unsupported recorder.video_stream='{stream}'")


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="press_three_buttons_gs",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    show: bool = bool(cfg.get("show", False))
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")
    rec_cfg = MaskRecorderConfig.model_validate(raw.pop("recorder", {}))

    task_file = prepare_task_file(cfg)
    env = ComponentRegistry.get_env(task_file.task.env_name)
    single_env = _resolve_single_env(env)

    required_attrs = (
        "_render_gs_camera",
        "_render_gs_masks_for_camera",
        "_gs_mask_renderers",
    )
    if not all(hasattr(single_env, attr) for attr in required_attrs):
        raise TypeError(
            "save_gs_mask_render.py requires a GS env with GS mask helpers."
        )

    if not single_env._gs_mask_renderers:
        raise RuntimeError(
            "No GS mask renderers were created. Check config.mask_objects and gaussian_render.body_gaussians."
        )

    runner = TaskRunner().from_config(task_file)
    config_name = HydraConfig.get().job.config_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / f"gs_mask_{config_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_frames: dict[str, list[np.ndarray]] = {
        cam_name: [] for cam_name in single_env._camera_specs
    }

    try:
        runner.reset()

        step_idx = 0
        while True:
            for cam_name, spec in single_env._camera_specs.items():
                cam_id = single_env._camera_ids[cam_name]
                rgb_t, _depth_t = single_env._render_gs_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
                (
                    _fg_rgb,
                    fg_depth,
                    _bg_rgb,
                    bg_depth,
                    _full_rgb,
                    _full_depth,
                ) = single_env._render_gs_camera_batch(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
                scene_depth_t = single_env._compose_mask_scene_depth(
                    fg_depth=fg_depth,
                    bg_depth=bg_depth,
                )
                binary_mask, heat_map = single_env._render_gs_masks_for_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                    scene_depth_t=scene_depth_t,
                )

                rgb = np.clip(rgb_t.detach().cpu().numpy() * 255.0, 0.0, 255.0).astype(
                    np.uint8
                )
                overlay = _make_overlay(rgb, binary_mask)

                if step_idx == 0:
                    _save_rgb(out_dir / f"{cam_name}_rgb.png", rgb)
                    _save_mask(out_dir / f"{cam_name}_mask.png", binary_mask)
                    _save_rgb(out_dir / f"{cam_name}_overlay.png", overlay)

                    if heat_map.ndim == 3 and heat_map.shape[-1] == len(
                        single_env.config.operations
                    ):
                        for channel_idx, operation_name in enumerate(
                            single_env.config.operations
                        ):
                            channel = heat_map[..., channel_idx]
                            if np.any(channel):
                                _save_mask(
                                    out_dir / f"{cam_name}_heat_{operation_name}.png",
                                    channel,
                                )

                    print(
                        f"Saved GS mask inspection images for camera '{cam_name}' to {out_dir}"
                    )

                    if show:
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
                        axes[0, 0].imshow(rgb)
                        axes[0, 0].set_title(f"{cam_name} RGB")
                        axes[0, 0].axis("off")
                        axes[0, 1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
                        axes[0, 1].set_title(f"{cam_name} GS Mask")
                        axes[0, 1].axis("off")
                        axes[0, 2].imshow(overlay)
                        axes[0, 2].set_title(f"{cam_name} Overlay")
                        axes[0, 2].axis("off")
                        plt.tight_layout()
                        plt.show()
                        plt.close(fig)

                if rec_cfg.enabled:
                    video_frames[cam_name].append(
                        _select_video_frame(
                            rec_cfg.video_stream,
                            rgb=rgb,
                            mask=binary_mask,
                            overlay=overlay,
                        )
                    )

            if not rec_cfg.enabled:
                break

            step_idx += 1
            if step_idx >= rec_cfg.max_steps:
                print(f"Reached recorder.max_steps={rec_cfg.max_steps}, stopping.")
                break

            update = runner.update()
            if bool(np.all(update.done)):
                break
    finally:
        runner.close()

    if rec_cfg.enabled:
        for cam_name, frames in video_frames.items():
            if not frames:
                continue
            if rec_cfg.save_mp4:
                mp4_path = out_dir / f"{cam_name}_{rec_cfg.video_stream}.mp4"
                iio.imwrite(
                    mp4_path,
                    frames,
                    fps=rec_cfg.fps,
                    codec="libx264",
                    quality=8,
                )
                print(f"Saved MP4 for '{cam_name}': {mp4_path}")
            if rec_cfg.save_gif:
                gif_path = out_dir / f"{cam_name}_{rec_cfg.video_stream}.gif"
                gif_fps = min(rec_cfg.fps, 15)
                iio.imwrite(gif_path, frames, fps=gif_fps, loop=0)
                print(f"Saved GIF for '{cam_name}': {gif_path}")


if __name__ == "__main__":
    main()
