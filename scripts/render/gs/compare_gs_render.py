"""Side-by-side comparison of Gaussian Splatting vs native MuJoCo rendering.

Loads a GS-enabled scene config, resets to the initial keyframe, and renders the
first frame from every configured camera in both GS and native MuJoCo modes.
Results are saved to ``outputs/compare_<run_name>_<timestamp>.png`` where
``<run_name>`` is ``<task>__<embodiment>__gs``.

Usage::

    # Requires Gaussian Splatting rendering: always pass render=gs
    python scripts/render/gs/compare_gs_render.py task=press_three_buttons render=gs

    # Any other task / embodiment with GS assets
    python scripts/render/gs/compare_gs_render.py task=cup_on_coaster render=gs
    python scripts/render/gs/compare_gs_render.py task=stack_color_blocks render=gs
    python scripts/render/gs/compare_gs_render.py task=press_blue_button embodiment=p7_g2p render=gs

    # Display the result interactively (pass as Hydra override)
    python scripts/render/gs/compare_gs_render.py task=press_three_buttons render=gs +show=true

Must be run from the project root (same working directory as `aao-demo`).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import hydra
import mujoco
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from auto_atom.runner.common import get_config_dir, get_run_name, prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner

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
            data = np.asarray(obs[key]["data"])
            if data.ndim >= 4:
                data = data[0]
            return data
    return None


def _resolve_single_env(env):
    if hasattr(env, "envs"):
        return env.envs[0]
    return env


def _require_gs_render() -> None:
    """Exit with a usage hint unless the run selected ``render=gs``."""
    render = HydraConfig.get().runtime.choices.get("render")
    if render != "gs":
        raise SystemExit(
            f"compare_gs_render.py compares Gaussian Splatting against native "
            f"MuJoCo rendering, but render={render} was selected. Pass render=gs, "
            "e.g.\n    python scripts/render/gs/compare_gs_render.py "
            "task=press_three_buttons render=gs"
        )


def _save_comparison(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    run_name: str,
    out_path: Path,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.5 * n), squeeze=False)
    fig.suptitle(
        f"GS vs Native MuJoCo  |  {run_name}",
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
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _require_gs_render()
    show: bool = bool(cfg.get("show", False))

    # ── 1. Instantiate env with resolved Hydra config ───────────────────────
    task_file = prepare_task_file(cfg)
    env = ComponentRegistry.get_env(task_file.task.env_name)
    single_env = _resolve_single_env(env)

    # ── 2. Reset to initial keyframe ─────────────────────────────────────────
    runner = TaskRunner().from_config(task_file)
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        runner.reset()

        # ── 3. GS observation ───────────────────────────────────────────────
        gs_obs: dict = env.capture_observation()

        # ── 4. Native render per camera ──────────────────────────────────────
        for cam_name, renderer in single_env._renderers.items():
            gs_img = _find_gs_image(gs_obs, cam_name)
            if gs_img is None:
                print(f"[warn] No GS color image for camera '{cam_name}', skipping.")
                continue

            cam_id = single_env._camera_ids[cam_name]
            native_img = _native_color(
                renderer,
                single_env.data,
                cam_id,
                single_env._renderer_scene_option,
            )

            rows.append((cam_name, gs_img, native_img))
            print(f"  {cam_name}: GS {gs_img.shape}  native {native_img.shape}")
    finally:
        runner.close()

    if not rows:
        print("No camera images captured. Ensure cameras have enable_color=True.")
        return

    # ── 5. Save figure ───────────────────────────────────────────────────────
    run_name = get_run_name()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"compare_{run_name}_{timestamp}.png"

    _save_comparison(rows, run_name, out_path, show=show)


if __name__ == "__main__":
    main()
