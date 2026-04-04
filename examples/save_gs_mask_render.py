"""Save per-camera mask images for visual inspection.

This script loads a scene config (GS or native MuJoCo), resets to the initial
keyframe, renders RGB plus binary masks for every configured camera, and saves
the outputs under ``outputs/mask_render_<config>_<timestamp>/``.

Saved files per camera:

- ``<camera>_rgb.png``
- ``<camera>_mask.png``
- ``<camera>_overlay.png``
- ``<camera>_heat_<operation>.png`` (when heat-map channels exist)

Usage::

    python examples/save_gs_mask_render.py
    python examples/save_gs_mask_render.py --config-name press_three_buttons
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


def _is_gs_env(env) -> bool:
    """Return True if the environment uses Gaussian Splatting rendering."""
    if hasattr(env, "_gs_mask_renderers"):
        return bool(env._gs_mask_renderers)
    single = _resolve_single_env(env)
    if hasattr(single, "_gs_mask_renderers"):
        return bool(single._gs_mask_renderers)
    return False


def _resolve_interest_pairs(single_env) -> tuple[list[str], list[str]]:
    mask_objects = list(getattr(single_env.config, "mask_objects", []))
    operations = list(getattr(single_env.config, "operations", []))
    if not mask_objects or not operations:
        return [], []
    if len(mask_objects) == len(operations):
        return mask_objects, operations
    if len(operations) == 1:
        return mask_objects, operations * len(mask_objects)
    raise ValueError(
        "Cannot infer mask_object -> operation mapping from config: "
        f"mask_objects={mask_objects}, operations={operations}. "
        "Expected equal lengths, or a single operation to broadcast."
    )


def _set_export_interest_focus(env, single_env) -> None:
    object_names, operation_names = _resolve_interest_pairs(single_env)
    if not object_names:
        return
    if hasattr(env, "envs"):
        for child_env in env.envs:
            # print(f"{object_names=}, {operation_names=}")
            child_env.set_interest_objects_and_operations(object_names, operation_names)
    else:
        env.set_interest_objects_and_operations(object_names, operation_names)


def _normalize_obs_image_shape(data: np.ndarray, expected_ndim: int) -> np.ndarray:
    if data.ndim == expected_ndim + 1:
        return data[0]
    if data.ndim == expected_ndim:
        return data
    raise TypeError(
        f"Expected observation with ndim {expected_ndim} or {expected_ndim + 1}, "
        f"got shape {data.shape}"
    )


def _find_obs_image(
    obs: dict, cam_name: str, suffix: str, expected_ndim: int
) -> np.ndarray | None:
    candidates = [
        f"{cam_name}/{suffix}",
        f"camera/{cam_name}/" + suffix,
        f"camera/{cam_name.split('_')[0]}/" + suffix,
    ]
    for key in candidates:
        if key not in obs:
            continue
        data = np.asarray(obs[key]["data"])
        return _normalize_obs_image_shape(data, expected_ndim)
    return None


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    plt.imsave(path, np.asarray(rgb, dtype=np.uint8))


def _save_mask(path: Path, mask: np.ndarray) -> None:
    plt.imsave(
        path, np.asarray(mask, dtype=np.uint8) * 255, cmap="gray", vmin=0, vmax=255
    )


def _channel_palette(num_channels: int) -> np.ndarray:
    """Return a (num_channels, 3) uint8 palette with one distinct color per channel."""
    if num_channels <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    cmap_name = "tab20" if num_channels > 10 else "tab10"
    cmap = plt.get_cmap(cmap_name, max(num_channels, 1))
    return np.asarray(
        [np.round(np.asarray(cmap(i)[:3]) * 255.0) for i in range(num_channels)],
        dtype=np.uint8,
    )


def _make_heatmap_rgb(
    heat_map: np.ndarray,
    operation_names: list[str] | None = None,
) -> np.ndarray:
    """Create an RGB image where each one-hot channel gets a unique color.

    For pixels with multiple active channels the colors are averaged.
    """
    heat = np.asarray(heat_map, dtype=np.float32)
    if heat.ndim != 3:
        raise TypeError(f"Expected 3D heat map, got shape {heat.shape}")

    H, W, C = heat.shape
    palette = _channel_palette(C)  # (C, 3) uint8
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    if C == 0:
        return rgb.astype(np.uint8)

    active_count = heat.sum(axis=-1, keepdims=True).clip(1.0)  # (H, W, 1)
    for ch in range(C):
        mask_ch = heat[..., ch : ch + 1]  # (H, W, 1)
        rgb += mask_ch * palette[ch].astype(np.float32)
    rgb /= active_count
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _save_heatmap_with_legend(
    path: Path,
    heatmap_rgb: np.ndarray,
    operation_names: list[str],
    heat_map: np.ndarray,
) -> None:
    """Save heatmap RGB with a color legend bar at the bottom."""
    palette = _channel_palette(len(operation_names))
    active_channels = [
        i
        for i in range(len(operation_names))
        if np.any(heat_map[..., i])
        if i < heat_map.shape[-1]
    ]
    if not active_channels:
        _save_rgb(path, heatmap_rgb)
        return

    fig, ax = plt.subplots(
        figsize=(heatmap_rgb.shape[1] / 100, heatmap_rgb.shape[0] / 100 + 0.6),
    )
    ax.imshow(heatmap_rgb)
    ax.axis("off")

    # Draw legend patches at the bottom
    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=palette[i].astype(np.float32) / 255.0,
            edgecolor="black",
            label=operation_names[i] if i < len(operation_names) else f"ch{i}",
        )
        for i in active_channels
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(handles), 5),
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    config_name="press_three_buttons",
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
    use_gs = _is_gs_env(env)

    print(
        f"[info] Rendering mode: {'Gaussian Splatting' if use_gs else 'native MuJoCo'}"
    )

    runner = TaskRunner().from_config(task_file)
    config_name = HydraConfig.get().job.config_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    render_tag = "gs" if use_gs else "mj"
    out_dir = Path("outputs") / f"mask_render_{render_tag}_{config_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_frames: dict[str, list[np.ndarray]] = {
        cam_name: [] for cam_name in single_env._camera_specs
    }

    try:
        runner.reset()
        _set_export_interest_focus(env, single_env)
        obs = env.capture_observation()

        step_idx = 0
        while True:
            for cam_name, spec in single_env._camera_specs.items():
                rgb = _find_obs_image(obs, cam_name, "color/image_raw", expected_ndim=3)
                binary_mask = _find_obs_image(
                    obs, cam_name, "mask/image_raw", expected_ndim=2
                )
                heat_map = _find_obs_image(
                    obs, cam_name, "mask/heat_map", expected_ndim=3
                )

                if rgb is None:
                    print(f"[warn] No RGB image for camera '{cam_name}', skipping.")
                    continue
                if binary_mask is None:
                    print(f"[warn] No mask image for camera '{cam_name}', skipping.")
                    continue

                rgb = np.asarray(rgb, dtype=np.uint8)
                binary_mask = np.asarray(binary_mask, dtype=np.uint8)
                op_names = list(single_env.config.operations)
                has_heat_map = heat_map is not None
                if has_heat_map:
                    heat_map = np.asarray(heat_map, dtype=np.uint8)
                    heatmap_rgb = _make_heatmap_rgb(heat_map, op_names)
                else:
                    heatmap_rgb = None
                overlay = _make_overlay(rgb, binary_mask)

                if step_idx == 0:
                    _save_rgb(out_dir / f"{cam_name}_rgb.png", rgb)
                    _save_mask(out_dir / f"{cam_name}_mask.png", binary_mask)
                    _save_rgb(out_dir / f"{cam_name}_overlay.png", overlay)
                    if heatmap_rgb is not None:
                        _save_heatmap_with_legend(
                            out_dir / f"{cam_name}_heatmap.png",
                            heatmap_rgb,
                            op_names,
                            heat_map,
                        )
                        if not np.any(heat_map):
                            print(
                                f"[info] Heat map for camera '{cam_name}' is all zeros "
                                "on the saved frame."
                            )
                    else:
                        print(
                            f"[info] No heat map observation for camera '{cam_name}', "
                            "skipping heatmap RGB export."
                        )

                    if (
                        has_heat_map
                        and heat_map.ndim == 3
                        and heat_map.shape[-1] == len(op_names)
                    ):
                        for channel_idx, operation_name in enumerate(op_names):
                            channel = heat_map[..., channel_idx]
                            if np.any(channel):
                                _save_mask(
                                    out_dir / f"{cam_name}_heat_{operation_name}.png",
                                    channel,
                                )

                    print(f"Saved mask images for camera '{cam_name}' to {out_dir}")

                    if show:
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4), squeeze=False)
                        axes[0, 0].imshow(rgb)
                        axes[0, 0].set_title(f"{cam_name} RGB")
                        axes[0, 0].axis("off")
                        axes[0, 1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
                        axes[0, 1].set_title(f"{cam_name} Mask")
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
            obs = env.capture_observation()
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
