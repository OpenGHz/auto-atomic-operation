from contextlib import contextmanager
from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np
import mujoco
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.basis.mjc.mujoco_basis import (
    CameraCalibrationConfig,
    CameraExtrinsicsConfig,
    CameraSpec,
    DataType,
    EnvConfig,
    MujocoBasis,
)
from auto_atom.basis.mjc.mujoco_env import (
    KeyCreator,
    UnifiedMujocoEnv,
    create_image_data,
)
from auto_atom.framework import RandomizationConstraintConfig
from auto_atom.scene_composition import SceneConfig
from auto_atom.utils.pose import PoseState


class _FakeRenderer:
    def __init__(self, image: np.ndarray):
        self._image = image
        self.update_count = 0

    def update_scene(self, data, camera, scene_option) -> None:
        self.update_count += 1
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
        create_image_data(np.zeros((4, 6, 2), dtype=np.uint8), time_sec=0.0)

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


@pytest.mark.parametrize("field", ["rgb_clip_range_m", "depth_clip_range_m"])
def test_camera_clip_ranges_require_near_before_far(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        CameraSpec(name="camera", **{field: (1.0, 1.0)})


def test_camera_clip_scope_converts_metric_range_and_restores_model() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <statistic extent="2"/>
          <visual><map znear="0.01" zfar="50"/></visual>
          <worldbody><camera name="camera"/></worldbody>
        </mujoco>
        """
    )
    env = MujocoBasis.__new__(MujocoBasis)
    env.model = model
    original = (float(model.vis.map.znear), float(model.vis.map.zfar))

    with env._camera_clip_scope((0.001, 5.0)):
        assert float(model.vis.map.znear) == pytest.approx(0.0005)
        assert float(model.vis.map.zfar) == pytest.approx(2.5)

    assert (float(model.vis.map.znear), float(model.vis.map.zfar)) == original


def test_camera_model_intersects_enabled_rgb_and_depth_clip_ranges() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <statistic extent="2"/>
          <visual><map znear="0.01" zfar="50"/></visual>
          <worldbody><camera name="camera"/></worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = MujocoBasis.__new__(MujocoBasis)
    env.model = model
    env.data = data
    env._camera_ids = {"camera": 0}
    env._camera_specs = {
        "camera": CameraSpec(
            name="camera",
            rgb_clip_range_m=(0.5, 4.0),
            depth_clip_range_m=(0.2, 5.0),
        )
    }

    camera = env.get_camera_model("camera")

    assert camera.near == pytest.approx(0.5)
    assert camera.far == pytest.approx(4.0)


def test_randomization_visibility_uses_rgb_depth_clip_intersection() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <statistic extent="2"/>
          <visual><map znear="0.01" zfar="50"/></visual>
          <worldbody>
            <camera name="camera"/>
            <body name="subject"><geom type="sphere" size="0.01"/></body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = MujocoBasis.__new__(MujocoBasis)
    env.model = model
    env.data = data
    env._camera_ids = {"camera": 0}
    env._camera_specs = {
        "camera": CameraSpec(
            name="camera",
            rgb_clip_range_m=(0.5, 4.0),
            depth_clip_range_m=(0.2, 5.0),
        )
    }

    report = env.evaluate_randomization_constraints(
        {
            "subject": PoseState(
                position=np.asarray([[0.0, 0.0, -4.5]], dtype=np.float64),
                orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            )
        },
        constraints=RandomizationConstraintConfig(
            visible_in={"cameras": ["camera"], "geometry": "center"}
        ),
    )

    assert not report.valid
    assert report.violations == ("subject:outside_depth:camera",)


def test_native_camera_uses_separate_rgb_and_depth_clip_scopes() -> None:
    cam_name = "camera"
    env = UnifiedMujocoEnv.__new__(UnifiedMujocoEnv)
    env.config = SimpleNamespace(
        structured=False,
        stamp_ns=False,
        enabled_sensors={DataType.CAMERA},
    )
    env.data = SimpleNamespace(time=0.0)
    env._key_creator = KeyCreator(False)
    env._operators = {}
    env._tactile_manager = None
    renderer = _FakeRenderer(np.zeros((2, 3, 3), dtype=np.uint8))
    env._renderers = {cam_name: renderer}
    env._camera_ids = {cam_name: 0}
    env._camera_specs = {
        cam_name: CameraSpec(
            name=cam_name,
            rgb_clip_range_m=(0.001, 50.0),
            depth_clip_range_m=(0.1, 5.0),
        )
    }
    env._camera_hidden_geom_ids = frozenset()
    env._renderer_scene_option = object()
    clip_scopes: list[tuple[float, float] | None] = []

    @contextmanager
    def record_clip_scope(clip_range_m: tuple[float, float] | None):
        clip_scopes.append(clip_range_m)
        yield

    env._camera_clip_scope = record_clip_scope

    env._collect_obs(False)

    assert clip_scopes == [(0.001, 50.0), (0.1, 5.0)]
    assert renderer.update_count == 2


def test_yaml_camera_calibration_overrides_fovy_and_parent_frame_pose(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.xml"
    scene.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="mount" pos="1 2 3">
              <camera name="mounted_cam"/>
            </body>
          </worldbody>
        </mujoco>
        """,
        encoding="utf-8",
    )
    config = EnvConfig(
        scene=SceneConfig(base=scene),
        enabled_sensors={DataType.CAMERA},
        cameras=[
            CameraSpec(
                name="mounted_cam",
                parent_frame="mount",
                calibration=CameraCalibrationConfig(
                    fovy_deg=60.0,
                    extrinsics=CameraExtrinsicsConfig(
                        position=(0.1, 0.2, 0.3),
                        orientation=(0.0, 0.0, 0.0, 1.0),
                    ),
                ),
            )
        ],
    )
    env = MujocoBasis(config)
    try:
        camera_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_CAMERA, "mounted_cam"
        )
        np.testing.assert_allclose(env.data.cam_xpos[camera_id], [1.1, 2.2, 3.3])
        assert float(env.model.cam_fovy[camera_id]) == pytest.approx(60.0)
    finally:
        env.close()


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
            rgb_clip_range_m=None,
            depth_clip_range_m=None,
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
