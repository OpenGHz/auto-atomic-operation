"""Save per-camera rendering outputs for visual inspection.

This script loads a scene config (GS or native MuJoCo), resets to the initial
keyframe, renders available per-camera outputs, and saves them under
``outputs/rendering_<backend>_<config>_<timestamp>/``.

Saved files per camera:

- ``<camera>_rgb.png``
- ``<camera>_mask.png`` (when mask is available)
- ``<camera>_overlay.png`` (when mask is available)
- ``<camera>_depth.png`` (when depth is available)
- ``<camera>_heat_<operation>.png`` (when heat-map channels exist)

When ``env.batch_size > 1``, files get an additional ``_env<index>`` suffix
before the extension, for example ``<camera>_rgb_env2.png``.

Usage::

    python examples/save_rendering.py
    python examples/save_rendering.py --config-name press_three_buttons
    python examples/save_rendering.py --config-name press_three_buttons_gs
    python examples/save_rendering.py show=true
    python examples/save_rendering.py +recorder.enabled=true
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


class RenderingRecorderConfig(BaseModel):
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


def _split_obs_image_batch(
    data: np.ndarray,
    expected_ndim: int,
    batch_size: int,
) -> list[np.ndarray]:
    if data.ndim == expected_ndim:
        if batch_size != 1:
            raise TypeError(
                "Expected batched observation with leading env dimension for "
                f"batch_size={batch_size}, got shape {data.shape}"
            )
        return [data]
    if data.ndim == expected_ndim + 1:
        if data.shape[0] != batch_size:
            raise TypeError(
                f"Expected leading batch dimension {batch_size}, got shape {data.shape}"
            )
        return [data[env_index] for env_index in range(batch_size)]
    raise TypeError(
        f"Expected observation with ndim {expected_ndim} or {expected_ndim + 1}, "
        f"got shape {data.shape}"
    )


def _find_obs_images(
    obs: dict,
    cam_name: str,
    suffix: str,
    expected_ndim: int,
    batch_size: int,
) -> list[np.ndarray] | None:
    candidates = [
        f"{cam_name}/{suffix}",
        f"camera/{cam_name}/" + suffix,
        f"camera/{cam_name.split('_')[0]}/" + suffix,
    ]
    for key in candidates:
        if key not in obs:
            continue
        data = np.asarray(obs[key]["data"])
        return _split_obs_image_batch(data, expected_ndim, batch_size)
    return None


def _env_suffix(batch_size: int, env_index: int) -> str:
    return "" if batch_size == 1 else f"_env{env_index}"


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    plt.imsave(path, np.asarray(rgb, dtype=np.uint8))


def _save_depth(path: Path, depth: np.ndarray) -> None:
    """Save depth as a colorized PNG using the turbo colormap."""
    valid = depth[depth > 0]
    if valid.size == 0:
        plt.imsave(path, np.zeros_like(depth), cmap="turbo")
        return
    vmin, vmax = float(valid.min()), float(valid.max())
    normed = np.where(depth > 0, (depth - vmin) / max(vmax - vmin, 1e-6), 0.0)
    plt.imsave(path, normed, cmap="turbo", vmin=0.0, vmax=1.0)


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
    mask: np.ndarray | None,
    overlay: np.ndarray | None,
) -> np.ndarray:
    if stream == "overlay":
        if overlay is None:
            return rgb
        return overlay
    if stream == "rgb":
        return rgb
    if stream == "mask":
        if mask is None:
            return rgb
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
    rec_cfg = RenderingRecorderConfig.model_validate(raw.pop("recorder", {}))

    task_file = prepare_task_file(cfg)
    env = ComponentRegistry.get_env(task_file.task.env_name)
    single_env = _resolve_single_env(env)
    use_gs = _is_gs_env(env)

    print(
        f"[info] Rendering mode: {'Gaussian Splatting' if use_gs else 'native MuJoCo'}"
    )

    runner = TaskRunner().from_config(task_file)
    batch_size = int(getattr(env, "batch_size", 1))
    config_name = HydraConfig.get().job.config_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    render_tag = "gs" if use_gs else "mj"
    out_dir = Path("outputs") / f"rendering_{render_tag}_{config_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_frames: dict[str, list[list[np.ndarray]]] = {
        cam_name: [[] for _ in range(batch_size)]
        for cam_name in single_env._camera_specs
    }

    try:
        runner.reset()
        _set_export_interest_focus(env, single_env)
        obs = env.capture_observation()

        step_idx = 0
        while True:
            for cam_name in single_env._camera_specs:
                rgb_batch = _find_obs_images(
                    obs,
                    cam_name,
                    "color/image_raw",
                    expected_ndim=3,
                    batch_size=batch_size,
                )
                binary_mask_batch = _find_obs_images(
                    obs,
                    cam_name,
                    "mask/image_raw",
                    expected_ndim=2,
                    batch_size=batch_size,
                )
                depth_batch = _find_obs_images(
                    obs,
                    cam_name,
                    "depth/image_raw",
                    expected_ndim=2,
                    batch_size=batch_size,
                )
                if depth_batch is None:
                    depth_batch = _find_obs_images(
                        obs,
                        cam_name,
                        "aligned_depth_to_color/image_raw",
                        expected_ndim=2,
                        batch_size=batch_size,
                    )
                heat_map_batch = _find_obs_images(
                    obs,
                    cam_name,
                    "mask/heat_map",
                    expected_ndim=3,
                    batch_size=batch_size,
                )

                if rgb_batch is None:
                    print(f"[warn] No RGB image for camera '{cam_name}', skipping.")
                    continue

                op_names = list(single_env.config.operations)
                for env_index in range(batch_size):
                    suffix = _env_suffix(batch_size, env_index)
                    rgb = np.asarray(rgb_batch[env_index], dtype=np.uint8)

                    binary_mask = None
                    if binary_mask_batch is not None:
                        binary_mask = np.asarray(
                            binary_mask_batch[env_index], dtype=np.uint8
                        )
                    has_mask = binary_mask is not None

                    heat_map = None
                    if heat_map_batch is not None:
                        heat_map = np.asarray(heat_map_batch[env_index], dtype=np.uint8)
                    has_heat_map = heat_map is not None
                    if has_heat_map:
                        heatmap_rgb = _make_heatmap_rgb(heat_map, op_names)
                    else:
                        heatmap_rgb = None

                    overlay = _make_overlay(rgb, binary_mask) if has_mask else None

                    depth = None
                    if depth_batch is not None:
                        depth = np.asarray(depth_batch[env_index], dtype=np.float32)
                    has_depth = depth is not None

                    if step_idx == 0:
                        _save_rgb(out_dir / f"{cam_name}_rgb{suffix}.png", rgb)
                        if has_depth:
                            _save_depth(
                                out_dir / f"{cam_name}_depth{suffix}.png", depth
                            )
                        else:
                            print(
                                f"[info] No depth observation for camera '{cam_name}' "
                                f"(env {env_index}), skipping depth export."
                            )
                        if has_mask:
                            _save_mask(
                                out_dir / f"{cam_name}_mask{suffix}.png", binary_mask
                            )
                            _save_rgb(
                                out_dir / f"{cam_name}_overlay{suffix}.png", overlay
                            )
                        else:
                            print(
                                f"[info] No mask observation for camera '{cam_name}' "
                                f"(env {env_index}), skipping mask/overlay export."
                            )
                        if heatmap_rgb is not None:
                            _save_heatmap_with_legend(
                                out_dir / f"{cam_name}_heatmap{suffix}.png",
                                heatmap_rgb,
                                op_names,
                                heat_map,
                            )
                            if not np.any(heat_map):
                                print(
                                    f"[info] Heat map for camera '{cam_name}' "
                                    f"(env {env_index}) is all zeros on the saved frame."
                                )
                        else:
                            print(
                                f"[info] No heat map observation for camera '{cam_name}' "
                                f"(env {env_index}), skipping heatmap RGB export."
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
                                        out_dir
                                        / f"{cam_name}_heat_{operation_name}{suffix}.png",
                                        channel,
                                    )

                        if batch_size == 1:
                            print(
                                f"Saved rendering outputs for camera '{cam_name}' to "
                                f"{out_dir}"
                            )
                        else:
                            print(
                                f"Saved rendering outputs for camera '{cam_name}' "
                                f"(env {env_index}) to {out_dir}"
                            )

                        if show and has_mask:
                            fig, axes = plt.subplots(
                                1, 3, figsize=(12, 4), squeeze=False
                            )
                            axes[0, 0].imshow(rgb)
                            axes[0, 0].set_title(f"{cam_name}{suffix} RGB")
                            axes[0, 0].axis("off")
                            axes[0, 1].imshow(binary_mask, cmap="gray", vmin=0, vmax=1)
                            axes[0, 1].set_title(f"{cam_name}{suffix} Mask")
                            axes[0, 1].axis("off")
                            axes[0, 2].imshow(overlay)
                            axes[0, 2].set_title(f"{cam_name}{suffix} Overlay")
                            axes[0, 2].axis("off")
                            plt.tight_layout()
                            plt.show()
                            plt.close(fig)
                        elif show:
                            fig, ax = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
                            ax[0, 0].imshow(rgb)
                            ax[0, 0].set_title(f"{cam_name}{suffix} RGB")
                            ax[0, 0].axis("off")
                            plt.tight_layout()
                            plt.show()
                            plt.close(fig)

                    if rec_cfg.enabled:
                        video_frames[cam_name][env_index].append(
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
        for cam_name, frames_by_env in video_frames.items():
            for env_index, frames in enumerate(frames_by_env):
                if not frames:
                    continue
                suffix = _env_suffix(batch_size, env_index)
                if rec_cfg.save_mp4:
                    mp4_path = (
                        out_dir / f"{cam_name}_{rec_cfg.video_stream}{suffix}.mp4"
                    )
                    iio.imwrite(
                        mp4_path,
                        frames,
                        fps=rec_cfg.fps,
                        codec="libx264",
                        quality=8,
                    )
                    if batch_size == 1:
                        print(f"Saved MP4 for '{cam_name}': {mp4_path}")
                    else:
                        print(
                            f"Saved MP4 for '{cam_name}' (env {env_index}): {mp4_path}"
                        )
                if rec_cfg.save_gif:
                    gif_path = (
                        out_dir / f"{cam_name}_{rec_cfg.video_stream}{suffix}.gif"
                    )
                    gif_fps = min(rec_cfg.fps, 15)
                    iio.imwrite(gif_path, frames, fps=gif_fps, loop=0)
                    if batch_size == 1:
                        print(f"Saved GIF for '{cam_name}': {gif_path}")
                    else:
                        print(
                            f"Saved GIF for '{cam_name}' (env {env_index}): {gif_path}"
                        )


if __name__ == "__main__":
    main()
