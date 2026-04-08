"""Test MuJoCo environment construction from task configs.

Uses the same Hydra config infrastructure as ``aao_demo`` / ``record_demo``.
Switch tasks with ``--config-name``:

    python tests/test_mujoco.py --config-name pick_and_place
    python tests/test_mujoco.py --config-name cup_on_coaster
    python tests/test_mujoco.py --config-name press_three_buttons
"""

import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from auto_atom.runner.common import get_config_dir
from auto_atom.runtime import ComponentRegistry


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    ComponentRegistry.clear()
    if "env" not in cfg or cfg.env is None:
        raise RuntimeError("Config must contain an 'env' section.")
    env = instantiate(cfg.env)

    raw = OmegaConf.to_container(cfg, resolve=True)
    env_raw = raw["env"]
    batch_size = env_raw.get("batch_size", 1)
    cameras = env_raw.get("cameras", [])
    mask_objects = env_raw.get("mask_objects", [])
    operations = env_raw.get("operations", [])
    heatmap_operations = env_raw.get("heatmap_operations", [])
    structured = env_raw.get("structured", False)

    print(f"Env type     : {type(env).__name__}")
    print(f"Batch size   : {batch_size}")
    print(f"Cameras      : {[c['name'] for c in cameras]}")
    print(f"Mask objects : {mask_objects}")
    print(f"Operations   : {operations}")

    try:
        env.reset()
        obs = env.capture_observation()
        print(f"\nObservation keys ({len(obs)}):")
        for key in sorted(obs.keys()):
            data = obs[key]["data"]
            if isinstance(data, dict):
                for sub_key, sub_val in data.items():
                    arr = np.asarray(sub_val)
                    print(f"  {key}/{sub_key}: shape={arr.shape} dtype={arr.dtype}")
            else:
                arr = np.asarray(data)
                print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")

        # --- helpers for structured image data ---
        def _decode_image(img_msg, shape, dtype=np.uint8):
            """Decode a structured image message dict to numpy array."""
            return np.frombuffer(img_msg["data"], dtype=dtype).reshape(shape)

        def _get_image_array(obs_entry, shape, dtype=np.uint8):
            """Extract image array from obs entry, handling both raw and structured."""
            data = obs_entry["data"]
            if structured:
                # data is a list of image message dicts (one per env)
                return np.stack(
                    [_decode_image(d, shape[1:], dtype) for d in data], axis=0
                )
            else:
                return np.asarray(data)

        # --- Validate camera observations ---
        for cam_cfg in cameras:
            cam_name = cam_cfg["name"]
            h, w = cam_cfg["height"], cam_cfg["width"]
            obs_cam = "camera/" + cam_name.split("_")[-2] if structured else cam_name
            prefix = f"/robot/{obs_cam}" if structured else obs_cam

            if cam_cfg.get("enable_color", False):
                key = (
                    f"{prefix}/color/image_raw"
                    if not structured
                    else f"{prefix}/video_encoded"
                )
                assert key in obs, f"Missing color obs: {key}"
                color = _get_image_array(obs[key], (batch_size, h, w, 3), np.uint8)
                assert color.shape == (batch_size, h, w, 3), (
                    f"{key}: expected ({batch_size},{h},{w},3), got {color.shape}"
                )
                assert color.dtype == np.uint8
                print(f"\n[PASS] {key}: {color.shape}")

            if cam_cfg.get("enable_depth", False):
                field = "aligned_depth_to_color" if not structured else "depth"
                key = f"{prefix}/{field}/image_raw"
                assert key in obs, f"Missing depth obs: {key}"
                depth = _get_image_array(obs[key], (batch_size, h, w), np.float32)
                assert depth.shape == (batch_size, h, w), (
                    f"{key}: expected ({batch_size},{h},{w}), got {depth.shape}"
                )
                print(f"[PASS] {key}: {depth.shape}")

            if cam_cfg.get("enable_mask", False) and mask_objects:
                key = f"{prefix}/mask/image_raw"
                assert key in obs, f"Missing mask obs: {key}"
                mask = _get_image_array(obs[key], (batch_size, h, w), np.uint8)
                assert mask.shape == (batch_size, h, w), (
                    f"{key}: expected ({batch_size},{h},{w}), got {mask.shape}"
                )
                assert mask.dtype == np.uint8
                print(f"[PASS] {key}: {mask.shape}")

            if cam_cfg.get("enable_heat_map", False) and heatmap_operations:
                n_ops = len(heatmap_operations)
                key = f"{prefix}/mask/heat_map"
                assert key in obs, f"Missing heat_map obs: {key}"
                hm = _get_image_array(obs[key], (batch_size, h, w, n_ops), np.uint8)
                assert hm.shape == (batch_size, h, w, n_ops), (
                    f"{key}: expected ({batch_size},{h},{w},{n_ops}), got {hm.shape}"
                )
                assert hm.dtype == np.uint8
                print(f"[PASS] {key}: {hm.shape}")

        print("\nAll camera assertions passed.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
