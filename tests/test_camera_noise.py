import numpy as np

from auto_atom.basis.mjc.camera_noise import CameraNoiseProcessor
from auto_atom.basis.mjc.mujoco_basis import (
    CameraNoiseConfig,
    CameraSpec,
    DepthNoiseConfig,
    RGBNoiseConfig,
)
from auto_atom.basis.mjc.mujoco_env import KeyCreator, create_image_data


CAMERA = "camera_color_optical_frame"


def _spec(**kwargs) -> CameraSpec:
    return CameraSpec(
        name=CAMERA,
        noise=CameraNoiseConfig(**kwargs),
        depth_clip_range_m=(0.5, 2.0),
    )


def test_rgb_noise_does_not_use_depth_validity() -> None:
    spec = _spec(rgb=RGBNoiseConfig(gaussian_std=0.2))
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=5)
    key_creator = KeyCreator(False)
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        key_creator.create_color_key(CAMERA): {"data": rgb, "t": 0.0},
        key_creator.create_depth_key(CAMERA): {
            "data": np.zeros((8, 8), dtype=np.float32),
            "t": 0.0,
        },
    }

    result = processor.process_observation(observation, key_creator, structured=False)

    noisy_rgb = result[key_creator.create_color_key(CAMERA)]["data"]
    assert np.any(noisy_rgb != 0)
    assert np.all(result[key_creator.create_depth_key(CAMERA)]["data"] == 0)


def test_depth_invalid_and_out_of_range_values_are_zero() -> None:
    spec = _spec(depth=DepthNoiseConfig())
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=7)
    key_creator = KeyCreator(False)
    depth = np.array([[1.0, 0.0], [np.nan, 3.0]], dtype=np.float32)
    observation = {key_creator.create_depth_key(CAMERA): {"data": depth, "t": 0.0}}

    result = processor.process_observation(observation, key_creator, structured=False)

    np.testing.assert_array_equal(
        result[key_creator.create_depth_key(CAMERA)]["data"],
        np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
    )


def test_fixed_seed_reproduces_each_capture_sequence() -> None:
    spec = _spec(rgb=RGBNoiseConfig(gaussian_std=0.1))
    key_creator = KeyCreator(False)
    image = np.full((4, 4, 3), 128, dtype=np.uint8)

    def capture(processor: CameraNoiseProcessor) -> np.ndarray:
        observation = {
            key_creator.create_color_key(CAMERA): {"data": image.copy(), "t": 0.0}
        }
        return processor.process_observation(
            observation, key_creator, structured=False
        )[key_creator.create_color_key(CAMERA)]["data"]

    first = CameraNoiseProcessor({CAMERA: spec}, seed=11)
    second = CameraNoiseProcessor({CAMERA: spec}, seed=11)
    np.testing.assert_array_equal(capture(first), capture(second))
    np.testing.assert_array_equal(capture(first), capture(second))


def test_batched_rows_use_independent_noise_streams() -> None:
    spec = _spec(rgb=RGBNoiseConfig(gaussian_std=0.1))
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=13)
    key_creator = KeyCreator(False)
    key = key_creator.create_color_key(CAMERA)
    observation = {
        key: {
            "data": np.full((2, 6, 6, 3), 128, dtype=np.uint8),
            "t": np.zeros(2),
        }
    }

    result = processor.process_batched_observation(
        observation, key_creator, structured=False, batch_size=2
    )

    assert result[key]["data"].shape == (2, 6, 6, 3)
    assert not np.array_equal(result[key]["data"][0], result[key]["data"][1])


def test_structured_image_payload_is_reencoded_with_noise() -> None:
    spec = _spec(rgb=RGBNoiseConfig(gaussian_std=0.1))
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=17)
    key_creator = KeyCreator(True)
    key = key_creator.create_color_key(CAMERA)
    observation = {
        key: {
            "data": create_image_data(
                np.full((3, 4, 3), 128, dtype=np.uint8), 1.25, CAMERA
            ),
            "t": 1.25,
        }
    }

    result = processor.process_observation(observation, key_creator, structured=True)
    message = result[key]["data"]

    assert message["encoding"] == "rgb8"
    assert message["header"]["frame_id"] == CAMERA
    assert message["header"]["stamp"] == {"sec": 1, "nanosec": 250000000}
    assert isinstance(message["data"], bytes)
