from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.basis.mjc.mujoco_basis import DataType, MujocoBasis
from auto_atom.basis.mjc.mujoco_env import (
    KeyCreator,
    UnifiedMujocoEnv,
    create_image_data,
)


class _FakeRenderer:
    def __init__(self, image: np.ndarray):
        self._image = image

    def update_scene(self, data, camera, scene_option) -> None:
        return None

    def disable_depth_rendering(self) -> None:
        return None

    def disable_segmentation_rendering(self) -> None:
        return None

    def enable_depth_rendering(self) -> None:
        return None

    def enable_segmentation_rendering(self) -> None:
        return None

    def render(self) -> np.ndarray:
        return self._image


def test_create_image_data_includes_frame_id() -> None:
    msg = create_image_data(
        np.zeros((2, 3, 3), dtype=np.uint8),
        time_sec=1.25,
        frame_id="camera_color_optical_frame",
    )

    assert msg["header"]["frame_id"] == "camera_color_optical_frame"
    assert msg["header"]["stamp"] == {"sec": 1, "nanosec": 250000000}


def test_create_image_data_depth_float32() -> None:
    depth = np.zeros((4, 6), dtype=np.float32)
    msg = create_image_data(depth, time_sec=0.0)
    assert msg["encoding"] == "32FC1"
    assert msg["step"] == 6 * 1 * 4  # w * c * itemsize
    assert msg["height"] == 4
    assert msg["width"] == 6


def test_create_image_data_mono8() -> None:
    mask = np.zeros((4, 6), dtype=np.uint8)
    msg = create_image_data(mask, time_sec=0.0)
    assert msg["encoding"] == "mono8"
    assert msg["step"] == 6


def test_create_image_data_rejects_unsupported() -> None:
    import pytest

    with pytest.raises(ValueError, match="unsupported dtype/channel"):
        create_image_data(np.zeros((4, 6, 5), dtype=np.uint8), time_sec=0.0)

    with pytest.raises(ValueError, match="expected 2-D or 3-D"):
        create_image_data(np.zeros((4,), dtype=np.uint8), time_sec=0.0)


def test_camera_info_exposes_header_frame_id() -> None:
    env = MujocoBasis.__new__(MujocoBasis)
    env._camera_ids = {"camera_color_optical_frame": 0}
    env._camera_specs = {
        "camera_color_optical_frame": SimpleNamespace(width=640, height=480)
    }
    env.model = SimpleNamespace(cam_fovy=np.asarray([45.0], dtype=np.float64))

    camera_info = env._get_camera_info()["camera_color_optical_frame"]

    assert camera_info["header"]["frame_id"] == "camera_color_optical_frame"
    assert camera_info["width"] == 640
    assert camera_info["height"] == 480


def test_structured_camera_messages_share_frame_id_with_camera_info() -> None:
    cam_name = "camera_color_optical_frame"
    env = UnifiedMujocoEnv.__new__(UnifiedMujocoEnv)
    env.config = SimpleNamespace(
        structured=True,
        stamp_ns=False,
        enabled_sensors={DataType.CAMERA},
    )
    env.data = SimpleNamespace(time=1.25)
    env._key_creator = KeyCreator(True)
    env._operators = {}
    env._tactile_manager = None
    env._renderers = {cam_name: _FakeRenderer(np.zeros((2, 3, 3), dtype=np.uint8))}
    env._camera_ids = {cam_name: 0}
    env._camera_specs = {
        cam_name: SimpleNamespace(
            enable_color=True,
            enable_depth=False,
            enable_mask=False,
            enable_heat_map=False,
            depth_max=10.0,
        )
    }
    env._renderer_scene_option = object()
    env.get_info = lambda: {
        "cameras": {
            cam_name: {
                "camera_info": {
                    "color": {
                        "header": {"frame_id": cam_name},
                        "width": 3,
                        "height": 2,
                        "distortion_model": "plumb_bob",
                        "d": [0.0] * 5,
                        "k": [0.0] * 9,
                        "r": [0.0] * 9,
                        "p": [0.0] * 12,
                    }
                },
                "camera_extrinsics": {
                    "frame": "eef_pose",
                    "translation": np.zeros(3, dtype=np.float64),
                    "rotation_matrix": np.eye(3, dtype=np.float64),
                },
            }
        }
    }

    obs = env._collect_obs(True)

    image_msg = obs[env._key_creator.create_color_key(cam_name)]["data"]
    camera_info_msg = obs[env._key_creator.create_camera_info_key(cam_name)]["data"]
    assert image_msg["header"]["frame_id"] == cam_name
    assert camera_info_msg["header"]["frame_id"] == cam_name
    assert image_msg["header"]["stamp"] == camera_info_msg["header"]["stamp"]
