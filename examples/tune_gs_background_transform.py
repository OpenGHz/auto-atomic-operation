"""Interactively tune GS background xyz offset with live color + mask preview.

Usage:
    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        examples/tune_gs_background_transform.py --config-name press_three_buttons_gs

    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        examples/tune_gs_background_transform.py --config-name wipe_the_table_gs \
        --camera env1_cam --step 0.002

Extra Hydra overrides can be appended after ``--``:
    ... -- --bg3dgs_name=discover-lab2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import ComponentRegistry

HELP_TEXT = [
    "Controls:",
    "  a/d : x -/+",
    "  s/w : y -/+",
    "  q/e : z -/+",
    "  [/]: step /2, *2",
    "  r   : reset to initial offset",
    "  p   : print current offset",
    "  x/esc: exit",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="press_three_buttons_gs")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera name to preview. Repeat to show multiple cameras.",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Preview all configured cameras.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.005,
        help="Initial xyz step size in metres.",
    )
    parser.add_argument(
        "--window-scale",
        type=float,
        default=0.75,
        help="Preview scale factor for the composed OpenCV window.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Also keep the MuJoCo passive viewer open.",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Optional Hydra overrides after '--'.",
    )
    return parser.parse_args()


def _compose_cfg(config_name: str, overrides: list[str]) -> DictConfig:
    config_dir = get_config_dir()
    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        return compose(config_name=config_name, overrides=overrides)


def _resolve_camera_names(
    cfg: DictConfig, requested: list[str] | None, all_cameras: bool
) -> list[str]:
    cameras = list(cfg.env.cameras)
    if not cameras:
        raise ValueError("Config has no cameras.")
    available = [str(cam.name) for cam in cameras]
    if all_cameras:
        return available
    if not requested:
        return [available[0]]

    resolved: list[str] = []
    for item in requested:
        for name in (part.strip() for part in item.split(",")):
            if not name:
                continue
            if name not in available:
                raise ValueError(
                    f"Unknown camera '{name}'. Available cameras: {available}"
                )
            if name not in resolved:
                resolved.append(name)
    if not resolved:
        return [available[0]]
    return resolved


def _prepare_cfg_for_preview(
    cfg: DictConfig,
    camera_names: list[str],
    keep_viewer: bool,
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    cfg.env.batch_size = 1
    if not keep_viewer:
        if "viewer" not in cfg.env or cfg.env.viewer is None:
            cfg.env.viewer = {"disable": True}
        else:
            cfg.env.viewer.disable = True

    has_mask_objects = bool(cfg.env.get("mask_objects"))
    for cam in cfg.env.cameras:
        enabled = str(cam.name) in camera_names
        cam.enable_color = enabled
        cam.enable_depth = False
        cam.enable_heat_map = False
        cam.enable_mask = bool(enabled and has_mask_objects)
    return cfg


def _extract_first_image(payload: dict[str, Any] | None) -> np.ndarray | None:
    if not payload or payload.get("data") is None:
        return None
    data = np.asarray(payload["data"])
    if data.ndim >= 4:
        data = data[0]
    return data


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(
        img,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _mask_to_bgr(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
    if mask_bgr.shape[:2] != target_shape:
        mask_bgr = cv2.resize(
            mask_bgr,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask_bgr


def _draw_hud(canvas: np.ndarray, offset: np.ndarray, step: float) -> np.ndarray:
    hud = canvas.copy()
    lines = [
        f"offset xyz = [{offset[0]:+.4f}, {offset[1]:+.4f}, {offset[2]:+.4f}] m",
        f"step = {step:.4f} m",
        *HELP_TEXT,
    ]
    x, y = 16, 28
    for line in lines:
        cv2.putText(
            hud,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            hud,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return hud


def _compose_camera_preview(
    camera_name: str,
    color_rgb: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    target_shape = color_bgr.shape[:2]

    if mask is None:
        mask_panel = np.zeros_like(color_bgr)
        cv2.putText(
            mask_panel,
            "Mask unavailable",
            (24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        overlay = color_bgr.copy()
    else:
        mask_panel = _mask_to_bgr(mask, target_shape)
        overlay = color_bgr.copy()
        overlay[mask_panel[..., 0] > 0] = (
            0.45 * overlay[mask_panel[..., 0] > 0]
            + 0.55 * np.array([0, 0, 255], dtype=np.float32)
        ).astype(np.uint8)

    for title, panel in (
        ("GS color", color_bgr),
        ("GS mask", mask_panel),
        ("Overlay", overlay),
    ):
        cv2.putText(
            panel,
            f"{camera_name} | {title}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return cv2.hconcat([color_bgr, mask_panel, overlay])


def _pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] >= width:
        return img
    pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=img.dtype)
    return cv2.hconcat([img, pad])


def _stack_rows(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(panels), cols):
        row_panels = panels[start : start + cols]
        max_height = max(panel.shape[0] for panel in row_panels)
        normalized = []
        for panel in row_panels:
            if panel.shape[0] != max_height:
                panel = cv2.resize(
                    panel,
                    (
                        max(
                            1, int(round(panel.shape[1] * max_height / panel.shape[0]))
                        ),
                        max_height,
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            normalized.append(panel)
        row = cv2.hconcat(normalized)
        rows.append(row)
    max_width = max(row.shape[1] for row in rows)
    rows = [_pad_to_width(row, max_width) for row in rows]
    return cv2.vconcat(rows)


def _render_preview(
    env,
    camera_names: list[str],
    offset: np.ndarray,
    step: float,
    scale: float,
):
    env.set_background_transform(offset.tolist())
    obs = env.capture_observation()
    kc = env._key_creator
    panels: list[np.ndarray] = []
    for camera_name in camera_names:
        color = _extract_first_image(obs.get(kc.create_color_key(camera_name)))
        if color is None:
            raise RuntimeError(f"No color image found for camera '{camera_name}'.")
        mask = _extract_first_image(obs.get(kc.create_mask_key(camera_name)))
        panels.append(_compose_camera_preview(camera_name, color, mask))

    cols = 1 if len(panels) <= 2 else 2
    combined = _stack_rows(panels, cols=cols)
    combined = _draw_hud(combined, offset, step)
    return _resize(combined, scale)


def main() -> None:
    args = _parse_args()
    overrides = [ov for ov in args.overrides if ov != "--"]
    cfg = _compose_cfg(args.config_name, overrides)
    camera_names = _resolve_camera_names(cfg, args.camera, args.all_cameras)
    cfg = _prepare_cfg_for_preview(cfg, camera_names, keep_viewer=args.viewer)

    task_file = prepare_task_file(cfg)
    env = ComponentRegistry.get_env(task_file.task.env_name)
    if not hasattr(env, "set_background_transform"):
        raise TypeError(
            f"Env '{type(env).__name__}' does not support live GS background transform tuning."
        )

    initial = np.asarray(
        env.config.gaussian_render.resolved_background_transform()[0],
        dtype=np.float64,
    )
    offset = initial.copy()
    step = float(args.step)
    window_name = (
        f"GS background tuner | {args.config_name} | {', '.join(camera_names)}"
    )

    print(f"Cameras: {camera_names}")
    print(f"Initial offset: {offset.tolist()}")
    print("\n".join(HELP_TEXT))

    env.reset()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    dirty = True
    try:
        while True:
            if dirty:
                preview = _render_preview(
                    env,
                    camera_names=camera_names,
                    offset=offset,
                    step=step,
                    scale=float(args.window_scale),
                )
                cv2.imshow(window_name, preview)
                dirty = False

            key = cv2.waitKey(30) & 0xFF
            if key in (255,):
                continue
            if key in (27, ord("x")):
                break
            if key == ord("a"):
                offset[0] -= step
                dirty = True
            elif key == ord("d"):
                offset[0] += step
                dirty = True
            elif key == ord("s"):
                offset[1] -= step
                dirty = True
            elif key == ord("w"):
                offset[1] += step
                dirty = True
            elif key == ord("q"):
                offset[2] -= step
                dirty = True
            elif key == ord("e"):
                offset[2] += step
                dirty = True
            elif key == ord("["):
                step = max(0.0005, step / 2.0)
                dirty = True
            elif key == ord("]"):
                step = min(0.1, step * 2.0)
                dirty = True
            elif key == ord("r"):
                offset = initial.copy()
                dirty = True
            elif key == ord("p"):
                print(
                    f"{env.config.gaussian_render.background_ply} -> "
                    f"[{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]"
                )
                bg_name = Path(str(env.config.gaussian_render.background_ply)).stem
                print(
                    f"YAML: {bg_name}: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}]"
                )
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
