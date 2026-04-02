"""Compare native MuJoCo depth with GS depth variants on the same frame.

Loads a GS-enabled scene config, resets to the initial keyframe, and renders
the first frame from every configured camera using:

- native MuJoCo depth buffer
- GS foreground accumulated depth
- GS composited depth (press_da_button-style)

Results are saved to ``outputs/compare_depth_<config>_<timestamp>.png``.

Usage::

    # Default config: press_three_buttons_gs
    python examples/compare_depth_render.py

    # Any other GS scene config (Hydra --config-name override)
    python examples/compare_depth_render.py --config-name cup_on_coaster_gs
    python examples/compare_depth_render.py --config-name stack_color_blocks_gs

    # Display the result interactively (pass as Hydra override)
    python examples/compare_depth_render.py show=true
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
import mujoco
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


def _resolve_single_env(env):
    """Return the first concrete env replica from a batched wrapper if needed."""
    if hasattr(env, "envs"):
        return env.envs[0]
    return env


def _resolve_gs_env(env):
    if hasattr(env, "_fg_gs_renderer"):
        return env
    single_env = _resolve_single_env(env)
    if hasattr(single_env, "_fg_gs_renderer"):
        return single_env
    return env


def _native_depth(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    cam_id: int,
    scene_option: mujoco.MjvOption,
) -> np.ndarray:
    """Render one depth frame with the native MuJoCo renderer."""
    renderer.update_scene(data, camera=cam_id, scene_option=scene_option)
    renderer.enable_depth_rendering()
    depth = np.asarray(renderer.render(), dtype=np.float32)
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    return depth


def _depth_limits(
    native_depth: np.ndarray,
    fg_depth: np.ndarray,
    comp_depth: np.ndarray,
) -> tuple[float, float]:
    """Choose a shared display range for one camera row."""
    valid = []
    for depth in (native_depth, fg_depth, comp_depth):
        arr = np.asarray(depth, dtype=np.float32)
        mask = np.isfinite(arr) & (arr > 0.0)
        if np.any(mask):
            valid.append(arr[mask])
    if not valid:
        return 0.0, 1.0
    merged = np.concatenate(valid)
    vmin = float(np.min(merged))
    vmax = float(np.max(merged))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _to_depth_image(depth: np.ndarray) -> np.ndarray:
    """Normalize a rendered depth tensor/array to a 2D ``(H, W)`` image."""
    arr = np.asarray(depth, dtype=np.float32)
    # Squeeze all leading and trailing singleton dimensions
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    while arr.ndim > 2 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[-1] == 1:
        return arr[0, ..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0]
    if arr.ndim != 2:
        raise TypeError(f"Unsupported depth image shape: {arr.shape}")
    return arr


def _save_comparison(
    rows: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    config_name: str,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(
        n,
        3,
        figsize=(16, 4.5 * n),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        f"Depth Comparison  |  {config_name}",
        fontsize=13,
        y=1.02,
    )

    titles = (
        "Native MuJoCo Depth",
        "GS FG Accumulated Depth",
        "GS Composited Depth",
    )
    last_im = None
    for row_idx, (cam_name, native_depth, fg_depth, comp_depth) in enumerate(rows):
        vmin, vmax = _depth_limits(native_depth, fg_depth, comp_depth)
        for col_idx, depth in enumerate((native_depth, fg_depth, comp_depth)):
            ax = axes[row_idx, col_idx]
            last_im = ax.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"{cam_name} — {titles[col_idx]}", fontsize=10)
            ax.axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.72, label="Depth (m)")

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="press_three_buttons_gs",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    show: bool = bool(cfg.get("show", False))

    task_file = prepare_task_file(cfg)
    env = ComponentRegistry.get_env(task_file.task.env_name)
    single_env = _resolve_single_env(env)
    gs_env = _resolve_gs_env(env)

    if not hasattr(gs_env, "_fg_gs_renderer"):
        raise TypeError(
            "compare_depth_render.py requires a GS env "
            "(for example a config using BatchedGSUnifiedMujocoEnv)."
        )

    runner = TaskRunner().from_config(task_file)
    rows: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    try:
        runner.reset()

        for cam_name, renderer in single_env._renderers.items():
            cam_id = single_env._camera_ids[cam_name]
            spec = single_env._camera_specs[cam_name]

            native_depth = _native_depth(
                renderer,
                single_env.data,
                cam_id,
                single_env._renderer_scene_option,
            )
            native_depth = _to_depth_image(native_depth.copy())
            native_depth[native_depth > spec.depth_max] = 0.0

            if hasattr(gs_env, "_render_gs_camera_batch") and hasattr(
                gs_env, "_render_gs_camera"
            ):
                _, _, _, _, _, fg_depth = gs_env._render_gs_camera_batch(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
                _, comp_depth = gs_env._render_gs_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
            else:
                body_pos = np.stack(
                    [np.asarray(e.data.xpos, dtype=np.float32) for e in gs_env.envs]
                )
                body_quat = np.stack(
                    [np.asarray(e.data.xquat, dtype=np.float32) for e in gs_env.envs]
                )
                fg_gsb = gs_env._fg_gs_renderer.batch_update_gaussians(
                    body_pos, body_quat
                )
                cam_pos = np.stack(
                    [
                        np.asarray(e.data.cam_xpos[cam_id], dtype=np.float32)
                        for e in gs_env.envs
                    ]
                ).reshape(gs_env.batch_size, 1, 3)
                cam_xmat = np.stack(
                    [
                        np.asarray(e.data.cam_xmat[cam_id], dtype=np.float32)
                        for e in gs_env.envs
                    ]
                ).reshape(gs_env.batch_size, 1, 9)
                fovy = np.broadcast_to(
                    np.asarray(
                        gs_env.envs[0].model.cam_fovy[cam_id], dtype=np.float32
                    ).reshape(1, 1),
                    (gs_env.batch_size, 1),
                ).copy()
                _, fg_depth, _, _, _, comp_depth = gs_env._render_batched_camera(
                    fg_gsb,
                    cam_pos,
                    cam_xmat,
                    spec.height,
                    spec.width,
                    fovy,
                    body_pos,
                    body_quat,
                    cam_id,
                )
                fg_depth = fg_depth[0, 0]
                comp_depth = comp_depth[0, 0]

            fg_depth_np = _to_depth_image(
                fg_depth.detach().cpu().numpy().astype(np.float32)
            )
            comp_depth_np = _to_depth_image(
                comp_depth.detach().cpu().numpy().astype(np.float32)
            )
            fg_depth_np[fg_depth_np > spec.depth_max] = 0.0
            comp_depth_np[comp_depth_np > spec.depth_max] = 0.0

            rows.append((cam_name, native_depth, fg_depth_np, comp_depth_np))
            print(
                f"  {cam_name}: native {native_depth.shape}  "
                f"fg {fg_depth_np.shape}  comp {comp_depth_np.shape}"
            )
    finally:
        runner.close()

    if not rows:
        print("No camera depths captured. Ensure the scene has configured cameras.")
        return

    config_name = HydraConfig.get().job.config_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"compare_depth_{config_name}_{timestamp}.png"
    _save_comparison(rows, config_name, out_path, show=show)


if __name__ == "__main__":
    main()
