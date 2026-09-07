"""Independent RGB/depth camera sensor-noise processing.

The processor deliberately lives after rendering and before the public
observation encoding.  It therefore applies the same sensor model to native
MuJoCo and Gaussian-Splatting images while leaving segmentation masks and heat
maps untouched.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

_ENCODING_DTYPES: dict[str, tuple[np.dtype, int]] = {
    "mono8": (np.dtype(np.uint8), 1),
    "rgb8": (np.dtype(np.uint8), 3),
    "rgba8": (np.dtype(np.uint8), 4),
    "mono16": (np.dtype(np.uint16), 1),
    "rgb16": (np.dtype(np.uint16), 3),
    "32FC1": (np.dtype(np.float32), 1),
    "64FC1": (np.dtype(np.float64), 1),
}


class CameraNoiseProcessor:
    """Apply deterministic, independent RGB and depth noise streams.

    ``camera_specs`` is a mapping of camera name to a ``CameraSpec``-like
    object.  The processor accepts both unstructured NumPy/Torch arrays and
    structured image messages produced by :func:`create_image_data`.
    """

    def __init__(self, camera_specs: Mapping[str, Any], seed: int | None = None):
        self._camera_specs = dict(camera_specs)
        self._capture_index = 0
        self._seed = self._normalize_seed(seed)

    @staticmethod
    def _normalize_seed(seed: int | None) -> int:
        if seed is None:
            return int(np.random.SeedSequence().entropy)
        return int(seed)

    def set_seed(self, seed: int | None) -> None:
        """Set the root seed and restart the per-capture sequence."""
        self._seed = self._normalize_seed(seed)
        self._capture_index = 0

    @property
    def seed(self) -> int:
        """Root seed used to derive camera/stream/capture RNGs."""
        return self._seed

    def process_observation(
        self,
        observation: dict[str, dict[str, Any]],
        key_creator: Any,
        *,
        structured: bool,
        logical_env_index: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Apply one capture's noise to a single logical observation."""
        capture_index = self._capture_index
        self._capture_index += 1
        for camera_name, spec in self._camera_specs.items():
            noise = getattr(spec, "noise", None)
            if noise is None:
                continue
            self._process_stream(
                observation,
                key_creator.create_color_key(camera_name),
                getattr(noise, "rgb", None),
                stream="rgb",
                camera_name=camera_name,
                capture_index=capture_index,
                logical_env_index=logical_env_index,
                structured=structured,
            )
            self._process_stream(
                observation,
                key_creator.create_depth_key(camera_name),
                getattr(noise, "depth", None),
                stream="depth",
                camera_name=camera_name,
                capture_index=capture_index,
                logical_env_index=logical_env_index,
                structured=structured,
                clip_range=getattr(spec, "depth_clip_range_m", None),
            )
        return observation

    def process_batched_observation(
        self,
        observation: dict[str, dict[str, Any]],
        key_creator: Any,
        *,
        structured: bool,
        batch_size: int,
    ) -> dict[str, dict[str, Any]]:
        """Apply one capture's noise independently to every logical batch row."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        capture_index = self._capture_index
        self._capture_index += 1
        for camera_name, spec in self._camera_specs.items():
            noise = getattr(spec, "noise", None)
            if noise is None:
                continue
            self._process_batched_stream(
                observation,
                key_creator.create_color_key(camera_name),
                getattr(noise, "rgb", None),
                stream="rgb",
                camera_name=camera_name,
                capture_index=capture_index,
                structured=structured,
                batch_size=batch_size,
            )
            self._process_batched_stream(
                observation,
                key_creator.create_depth_key(camera_name),
                getattr(noise, "depth", None),
                stream="depth",
                camera_name=camera_name,
                capture_index=capture_index,
                structured=structured,
                batch_size=batch_size,
                clip_range=getattr(spec, "depth_clip_range_m", None),
            )
        return observation

    def _rng(
        self,
        *,
        camera_name: str,
        stream: str,
        capture_index: int,
        logical_env_index: int,
    ) -> np.random.Generator:
        digest = hashlib.blake2b(camera_name.encode("utf-8"), digest_size=8).digest()
        camera_id = int.from_bytes(digest, byteorder="little", signed=False)
        stream_id = 0 if stream == "rgb" else 1
        sequence = np.random.SeedSequence(
            [
                self._seed,
                int(capture_index),
                int(logical_env_index),
                camera_id & 0xFFFFFFFF,
                camera_id >> 32,
                stream_id,
            ]
        )
        return np.random.default_rng(sequence)

    def _process_stream(
        self,
        observation: dict[str, dict[str, Any]],
        key: str,
        config: Any,
        *,
        stream: str,
        camera_name: str,
        capture_index: int,
        logical_env_index: int,
        structured: bool,
        clip_range: Any = None,
    ) -> None:
        if config is None or key not in observation:
            return
        rng = self._rng(
            camera_name=camera_name,
            stream=stream,
            capture_index=capture_index,
            logical_env_index=logical_env_index,
        )
        value = observation[key]["data"]
        processed = self._process_value(
            value,
            config,
            stream=stream,
            rng=rng,
            structured=structured,
            clip_range=clip_range,
        )
        if structured and not (
            isinstance(processed, Mapping) and "encoding" in processed
        ):
            from auto_atom.basis.mjc.mujoco_env import create_image_data

            timestamp = observation[key].get("t", 0.0)
            timestamp = float(timestamp)
            if timestamp > 1.0e6:
                timestamp /= 1.0e9
            processed = create_image_data(
                self._to_numpy(processed), timestamp, camera_name, tobytes=False
            )
        observation[key]["data"] = processed

    def _process_batched_stream(
        self,
        observation: dict[str, dict[str, Any]],
        key: str,
        config: Any,
        *,
        stream: str,
        camera_name: str,
        capture_index: int,
        structured: bool,
        batch_size: int,
        clip_range: Any = None,
    ) -> None:
        if config is None or key not in observation:
            return
        value = observation[key]["data"]
        if structured and isinstance(value, list):
            if len(value) != batch_size:
                raise ValueError(
                    f"structured observation '{key}' has {len(value)} rows, "
                    f"expected {batch_size}"
                )
            from auto_atom.basis.mjc.mujoco_env import create_image_data

            encoded_rows = []
            for row, item in enumerate(value):
                processed = self._process_value(
                    item,
                    config,
                    stream=stream,
                    rng=self._rng(
                        camera_name=camera_name,
                        stream=stream,
                        capture_index=capture_index,
                        logical_env_index=row,
                    ),
                    structured=True,
                    clip_range=clip_range,
                )
                if not (isinstance(processed, Mapping) and "encoding" in processed):
                    timestamp = observation[key].get("t", 0.0)
                    timestamp = float(np.asarray(timestamp).reshape(-1)[row])
                    if timestamp > 1.0e6:
                        timestamp /= 1.0e9
                    processed = create_image_data(
                        self._to_numpy(processed), timestamp, camera_name, tobytes=False
                    )
                encoded_rows.append(processed)
            observation[key]["data"] = encoded_rows
            return

        array = value
        shape = getattr(array, "shape", None)
        if shape is None or len(shape) < 1 or int(shape[0]) != batch_size:
            raise ValueError(
                f"batched observation '{key}' must have leading dimension "
                f"{batch_size}, got {shape}"
            )
        if structured:
            from auto_atom.basis.mjc.mujoco_env import create_image_data

            timestamp_values = np.asarray(observation[key].get("t", 0.0)).reshape(-1)
            if timestamp_values.size == 1:
                timestamp_values = np.repeat(timestamp_values, batch_size)
            if timestamp_values.size != batch_size:
                raise ValueError(
                    f"structured observation '{key}' has {timestamp_values.size} "
                    f"timestamps, expected {batch_size}"
                )
            encoded_rows = []
            for row in range(batch_size):
                processed = self._process_value(
                    array[row],
                    config,
                    stream=stream,
                    rng=self._rng(
                        camera_name=camera_name,
                        stream=stream,
                        capture_index=capture_index,
                        logical_env_index=row,
                    ),
                    structured=False,
                    clip_range=clip_range,
                )
                timestamp = float(timestamp_values[row])
                if timestamp > 1.0e6:
                    timestamp /= 1.0e9
                encoded_rows.append(
                    create_image_data(
                        self._to_numpy(processed),
                        timestamp,
                        camera_name,
                        tobytes=False,
                    )
                )
            observation[key]["data"] = encoded_rows
            return

        # A copy keeps broadcast views and caller-owned arrays immutable.
        rows = [
            self._process_value(
                array[row],
                config,
                stream=stream,
                rng=self._rng(
                    camera_name=camera_name,
                    stream=stream,
                    capture_index=capture_index,
                    logical_env_index=row,
                ),
                structured=False,
                clip_range=clip_range,
            )
            for row in range(batch_size)
        ]
        try:
            import torch

            if isinstance(array, torch.Tensor):
                observation[key]["data"] = torch.stack(rows, dim=0)
                return
        except ImportError:  # pragma: no cover - torch is an optional backend
            pass
        observation[key]["data"] = np.stack(rows, axis=0)

    def _process_value(
        self,
        value: Any,
        config: Any,
        *,
        stream: str,
        rng: np.random.Generator,
        structured: bool,
        clip_range: Any = None,
    ) -> Any:
        if structured and isinstance(value, Mapping) and "encoding" in value:
            array = self._decode_image_message(value)
            processed = self._process_array(
                array, config, stream=stream, rng=rng, clip_range=clip_range
            )
            from auto_atom.basis.mjc.mujoco_env import create_image_data

            stamp = value.get("header", {}).get("stamp", {})
            time_sec = (
                float(stamp.get("sec", 0.0)) + float(stamp.get("nanosec", 0.0)) / 1.0e9
            )
            frame_id = str(value.get("header", {}).get("frame_id", ""))
            tobytes = isinstance(value.get("data"), (bytes, bytearray, memoryview))
            return create_image_data(processed, time_sec, frame_id, tobytes=tobytes)
        return self._process_array(
            value, config, stream=stream, rng=rng, clip_range=clip_range
        )

    @staticmethod
    def _decode_image_message(message: Mapping[str, Any]) -> np.ndarray:
        encoding = str(message["encoding"])
        try:
            dtype, channels = _ENCODING_DTYPES[encoding]
        except KeyError as exc:
            raise ValueError(
                f"unsupported structured image encoding: {encoding}"
            ) from exc
        height = int(message["height"])
        width = int(message["width"])
        raw = message["data"]
        if isinstance(raw, (bytes, bytearray, memoryview)):
            array = np.frombuffer(raw, dtype=dtype)
        else:
            array = np.asarray(raw, dtype=dtype)
        expected = height * width * channels
        if array.size != expected:
            raise ValueError(
                f"structured image payload has {array.size} values, expected {expected}"
            )
        shape = (height, width) if channels == 1 else (height, width, channels)
        return np.asarray(array).reshape(shape).copy()

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert NumPy/Torch image values to a CPU NumPy array."""
        try:
            import torch

            if isinstance(value, torch.Tensor):
                return value.detach().cpu().numpy()
        except ImportError:  # pragma: no cover - torch is optional
            pass
        return np.asarray(value)

    @staticmethod
    def _process_array(
        value: Any,
        config: Any,
        *,
        stream: str,
        rng: np.random.Generator,
        clip_range: Any = None,
    ) -> Any:
        is_torch = False
        torch_device = None
        torch_dtype = None
        try:
            import torch

            is_torch = isinstance(value, torch.Tensor)
            if is_torch:
                torch_device = value.device
                torch_dtype = value.dtype
                value = value.detach().cpu().numpy()
        except ImportError:  # pragma: no cover - torch is optional
            torch = None

        array = np.asarray(value)
        if stream == "rgb":
            result = CameraNoiseProcessor._process_rgb(array, config, rng)
        else:
            result = CameraNoiseProcessor._process_depth(
                array, config, rng, clip_range=clip_range
            )
        if is_torch:
            return torch.as_tensor(result, device=torch_device, dtype=torch_dtype)
        return result

    @staticmethod
    def _process_rgb(
        array: np.ndarray, config: Any, rng: np.random.Generator
    ) -> np.ndarray:
        original_dtype = array.dtype
        normalized = array.astype(np.float64, copy=False)
        if np.issubdtype(original_dtype, np.integer):
            normalized = normalized / float(np.iinfo(original_dtype).max)
        else:
            normalized = normalized.copy()

        shot_scale = getattr(config, "shot_noise_scale", None)
        if shot_scale is not None:
            scale = float(shot_scale)
            normalized = rng.poisson(np.clip(normalized, 0.0, 1.0) * scale) / scale
        gaussian_std = float(getattr(config, "gaussian_std", 0.0))
        if gaussian_std > 0.0:
            normalized += rng.normal(0.0, gaussian_std, size=normalized.shape)
        dropout_probability = float(getattr(config, "dropout_probability", 0.0))
        if dropout_probability > 0.0:
            pixel_shape = normalized.shape[:2]
            dropped = rng.random(pixel_shape) < dropout_probability
            if normalized.ndim == 2:
                normalized[dropped] = 0.0
            else:
                normalized[dropped, ...] = 0.0
        normalized = np.clip(normalized, 0.0, 1.0)
        if np.issubdtype(original_dtype, np.integer):
            return np.rint(normalized * np.iinfo(original_dtype).max).astype(
                original_dtype
            )
        return normalized.astype(original_dtype, copy=False)

    @staticmethod
    def _process_depth(
        array: np.ndarray,
        config: Any,
        rng: np.random.Generator,
        *,
        clip_range: Any = None,
    ) -> np.ndarray:
        original_dtype = array.dtype
        depth = array.astype(np.float64, copy=True)
        valid = np.isfinite(depth) & (depth > 0.0)
        gaussian_std = float(getattr(config, "gaussian_std_m", 0.0))
        relative_std = float(getattr(config, "relative_std", 0.0))
        sigma = np.sqrt(gaussian_std**2 + (relative_std * depth) ** 2)
        noise = rng.normal(0.0, 1.0, size=depth.shape) * sigma
        depth[valid] += noise[valid]

        quantization_step = getattr(config, "quantization_step_m", None)
        if quantization_step is not None:
            step = float(quantization_step)
            depth[valid] = np.round(depth[valid] / step) * step

        if clip_range is None:
            clip_range = getattr(config, "clip_range_m", None)
        if clip_range is not None:
            near_m, far_m = (float(v) for v in clip_range)
            valid &= depth >= near_m
            valid &= depth <= far_m

        dropout_probability = float(getattr(config, "dropout_probability", 0.0))
        if dropout_probability > 0.0:
            valid &= rng.random(depth.shape) >= dropout_probability

        result = np.zeros_like(depth)
        result[valid] = depth[valid]
        return result.astype(original_dtype, copy=False)


__all__ = ["CameraNoiseProcessor"]
