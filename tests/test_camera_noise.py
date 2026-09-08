import numpy as np
import pytest

from auto_atom.basis.mjc.camera_noise import CameraNoiseProcessor
from auto_atom.basis.mjc.mujoco_basis import (
    CameraNoiseConfig,
    CameraSpec,
    DepthNoiseConfig,
    NoiseDistributionConfig,
    RGBNoiseConfig,
    TemporalNoiseConfig,
)
from auto_atom.basis.mjc.mujoco_env import (
    BatchedUnifiedMujocoEnv,
    KeyCreator,
    create_image_data,
)


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


def test_depth_bias_scale_and_invalid_probability_alias() -> None:
    config = DepthNoiseConfig(
        scale=1.1,
        bias_m=0.02,
        bias_distance_power=1.0,
        invalid_probability=0.0,
    )
    spec = _spec(depth=config)
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=21)
    key_creator = KeyCreator(False)
    key = key_creator.create_depth_key(CAMERA)
    result = processor.process_observation(
        {key: {"data": np.ones((4, 4), dtype=np.float32), "t": 0.0}},
        key_creator,
        structured=False,
    )
    np.testing.assert_allclose(result[key]["data"], 1.12, atol=1e-6)

    alias = DepthNoiseConfig(dropout_probability=1.0)
    assert alias.invalid_probability == 1.0


def test_depth_resolution_scaling_uses_actual_image_size() -> None:
    spec = _spec(
        depth=DepthNoiseConfig(
            bias_m=0.1,
            resolution_reference=(4, 4),
            resolution_power=1.0,
        )
    )
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=21)
    key_creator = KeyCreator(False)
    key = key_creator.create_depth_key(CAMERA)
    result = processor.process_observation(
        {key: {"data": np.ones((2, 2), dtype=np.float32), "t": 0.0}},
        key_creator,
        structured=False,
    )

    np.testing.assert_allclose(result[key]["data"], 1.4, atol=1e-6)


def test_partial_temporal_reset_only_clears_selected_batch_row() -> None:
    spec = _spec(
        rgb=RGBNoiseConfig(
            temporal=TemporalNoiseConfig(drift_std=0.05, drift_decay=1.0)
        )
    )
    key_creator = KeyCreator(False)
    key = key_creator.create_color_key(CAMERA)

    def capture(processor: CameraNoiseProcessor) -> np.ndarray:
        return processor.process_batched_observation(
            {
                key: {
                    "data": np.full((2, 4, 4, 3), 0.5, dtype=np.float32),
                    "t": np.zeros(2),
                }
            },
            key_creator,
            structured=False,
            batch_size=2,
        )[key]["data"]

    reset_processor = CameraNoiseProcessor({CAMERA: spec}, seed=24)
    unchanged_processor = CameraNoiseProcessor({CAMERA: spec}, seed=24)
    capture(reset_processor)
    capture(unchanged_processor)
    reset_processor.reset([1])

    reset_result = capture(reset_processor)
    unchanged_result = capture(unchanged_processor)

    np.testing.assert_array_equal(reset_result[0], unchanged_result[0])
    assert not np.array_equal(reset_result[1], unchanged_result[1])


def test_full_reset_restarts_capture_sequence() -> None:
    spec = _spec(
        rgb=RGBNoiseConfig(
            gaussian_std=0.05,
            temporal=TemporalNoiseConfig(drift_std=0.05),
        )
    )
    key_creator = KeyCreator(False)
    key = key_creator.create_color_key(CAMERA)

    def capture(processor: CameraNoiseProcessor) -> np.ndarray:
        observation = {key: {"data": np.full((4, 4, 3), 0.5, dtype=np.float32)}}
        return processor.process_observation(
            observation, key_creator, structured=False
        )[key]["data"]

    processor = CameraNoiseProcessor({CAMERA: spec}, seed=25)
    first = capture(processor)
    capture(processor)
    processor.reset()
    np.testing.assert_array_equal(first, capture(processor))


def test_batched_full_reset_restarts_noise_sequence() -> None:
    class ResettableEnv:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset(self) -> None:
            self.reset_count += 1

    class NoiseProcessor:
        def __init__(self) -> None:
            self.reset_calls: list[np.ndarray | None] = []

        def reset(self, logical_env_indices=None) -> None:
            self.reset_calls.append(logical_env_indices)

    env = BatchedUnifiedMujocoEnv.__new__(BatchedUnifiedMujocoEnv)
    env.batch_size = 2
    env.envs = [ResettableEnv(), ResettableEnv()]
    processor = NoiseProcessor()
    env._camera_noise_processor = processor

    env.reset()

    assert [child.reset_count for child in env.envs] == [1, 1]
    assert processor.reset_calls == [None]

    env.reset(np.asarray([False, True]))

    assert [child.reset_count for child in env.envs] == [1, 2]
    np.testing.assert_array_equal(processor.reset_calls[1], np.asarray([1]))


def test_student_t_and_temporal_noise_are_reproducible() -> None:
    config = RGBNoiseConfig(
        gaussian_std=0.05,
        distribution=NoiseDistributionConfig(kind="student_t", degrees_of_freedom=4.0),
        temporal=TemporalNoiseConfig(
            jitter_std=0.02,
            ar1_coefficient=0.8,
            drift_std=0.01,
            drift_decay=0.95,
        ),
    )
    spec = _spec(rgb=config)
    key_creator = KeyCreator(False)
    key = key_creator.create_color_key(CAMERA)

    def captures() -> list[np.ndarray]:
        processor = CameraNoiseProcessor({CAMERA: spec}, seed=22)
        outputs = []
        for _ in range(3):
            observation = {key: {"data": np.full((12, 12, 3), 128, np.uint8), "t": 0.0}}
            outputs.append(
                processor.process_observation(
                    observation, key_creator, structured=False
                )[key]["data"]
            )
        return outputs

    first, second = captures(), captures()
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(first[0], first[1])


def test_torch_path_preserves_device_and_dtype() -> None:
    torch = pytest.importorskip("torch")
    spec = _spec(
        rgb=RGBNoiseConfig(gaussian_std=0.01),
        depth=DepthNoiseConfig(gaussian_std_m=0.001),
    )
    processor = CameraNoiseProcessor({CAMERA: spec}, seed=23)
    key_creator = KeyCreator(False)
    color_key = key_creator.create_color_key(CAMERA)
    depth_key = key_creator.create_depth_key(CAMERA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observation = {
        color_key: {"data": torch.full((4, 4, 3), 0.5, device=device), "t": 0.0},
        depth_key: {
            "data": torch.ones((4, 4), dtype=torch.float32, device=device),
            "t": 0.0,
        },
    }
    result = processor.process_observation(observation, key_creator, structured=False)
    assert result[color_key]["data"].device.type == device.split(":")[0]
    assert result[depth_key]["data"].device.type == device.split(":")[0]
    assert result[color_key]["data"].dtype == torch.float32
    assert result[depth_key]["data"].dtype == torch.float32
