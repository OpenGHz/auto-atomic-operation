"""Side-by-side comparison of Gaussian Splatting vs native MuJoCo rendering.

Loads a GS-enabled scene config, resets to the initial keyframe, and renders the
first frame from every configured camera in both GS and native MuJoCo modes.
Results are saved to ``outputs/compare_<config>_<timestamp>.png``.

Usage::

    # Default config: press_three_buttons_gs
    python examples/compare_gs_render.py

    # Any other GS scene config (Hydra --config-name override)
    python examples/compare_gs_render.py --config-name cup_on_coaster_gs
    python examples/compare_gs_render.py --config-name stack_color_blocks_gs

    # Display the result interactively (pass as Hydra override)
    python examples/compare_gs_render.py show=true

Must be run from the project root (same working directory as `aao_demo`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import hydra
import mujoco
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from auto_atom.runner.common import get_config_dir
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _native_color(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    cam_id: int,
    scene_option: mujoco.MjvOption,
) -> np.ndarray:
    """Render one frame with the native MuJoCo rasteriser, return uint8 RGB (H, W, 3)."""
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=cam_id, scene_option=scene_option)
    return np.asarray(renderer.render(), dtype=np.uint8)


def _find_gs_image(obs: dict, cam_name: str) -> np.ndarray | None:
    """Look up a GS color image in the observation dict (handles structured/flat keys)."""
    candidates = [
        f"{cam_name}/color/image_raw",
        f"camera/{cam_name}/color/image_raw",
        f"camera/{cam_name.split('_')[0]}/color/image_raw",
    ]
    for key in candidates:
        if key in obs:
            return obs[key]["data"]
    return None


def _save_comparison(
    rows: List[Tuple[str, np.ndarray, np.ndarray]],
    config_name: str,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n), squeeze=False)
    fig.suptitle(
        f"GS vs Native MuJoCo  |  {config_name}",
        fontsize=13,
        y=1.002,
    )

    for row_idx, (cam_name, gs_img, native_img) in enumerate(rows):
        axes[row_idx, 0].imshow(gs_img)
        axes[row_idx, 0].set_title(f"{cam_name} — GS", fontsize=10)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(native_img)
        axes[row_idx, 1].set_title(f"{cam_name} — Native MuJoCo", fontsize=10)
        axes[row_idx, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="press_three_buttons_gs",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    show: bool = bool(cfg.get("show", False))

    # ── 1. Instantiate env (same pattern as aao_demo) ───────────────────────
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict)

    ComponentRegistry.clear()
    if "env" in cfg and cfg.env is not None:
        instantiate(cfg.env)

    env_name: str = OmegaConf.select(cfg, "env.name") or "env"
    env = ComponentRegistry.get_env(env_name)

    # ── 2. Reset to initial keyframe ─────────────────────────────────────────
    task_file = TaskFileConfig.model_validate(raw)
    runner = TaskRunner().from_config(task_file)
    runner.reset()

    # ── 3. GS observation ───────────────────────────────────────────────────
    gs_obs: dict = env.capture_observation()

    # ── 4. Native render per camera ──────────────────────────────────────────
    rows: List[Tuple[str, np.ndarray, np.ndarray]] = []

    for cam_name, renderer in env._renderers.items():
        gs_img = _find_gs_image(gs_obs, cam_name)
        if gs_img is None:
            print(f"[warn] No GS color image for camera '{cam_name}', skipping.")
            continue

        cam_id = env._camera_ids[cam_name]
        native_img = _native_color(
            renderer, env.data, cam_id, env._renderer_scene_option
        )

        rows.append((cam_name, gs_img, native_img))
        print(f"  {cam_name}: GS {gs_img.shape}  native {native_img.shape}")

    runner.close()

    if not rows:
        print("No camera images captured. Ensure cameras have enable_color=True.")
        return

    # ── 5. Save figure ───────────────────────────────────────────────────────
    config_name = HydraConfig.get().job.config_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"compare_{config_name}_{timestamp}.png"

    _save_comparison(rows, config_name, out_path, show=show)


if __name__ == "__main__":
    main()
