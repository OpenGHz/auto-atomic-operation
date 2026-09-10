"""Independent RGB/depth camera sensor-noise processing.

The processor deliberately lives after rendering and before the public
observation encoding.  It therefore applies the same sensor model to native
MuJoCo and Gaussian-Splatting images while leaving segmentation masks and heat
maps untouched.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from ...utils.seed import resolve_run_seed

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
        self._seed = resolve_run_seed(seed)
        self._temporal_state: dict[tuple[str, str, int], tuple[float, float]] = {}

    def set_seed(self, seed: int | None) -> None:
        """Set the root seed and restart the per-capture sequence."""
        self._seed = resolve_run_seed(seed)
        self._capture_index = 0
        self._temporal_state.clear()

    def reset(self, logical_env_indices: Iterable[int] | None = None) -> None:
        """Clear temporal state for every or selected logical environment row.

        Resetting temporal state prevents AR(1) and drift values from leaking
        into the next reset.  The capture index deliberately continues so a
        fixed root seed remains a stream of distinct sensor exposures across
        resets; call :meth:`set_seed` when the entire sequence must restart.
        """
        if logical_env_indices is None:
            self._temporal_state.clear()
            self._capture_index = 0
            return
        indices = {int(index) for index in logical_env_indices}
        if not indices:
            return
        self._temporal_state = {
            key: state
            for key, state in self._temporal_state.items()
            if key[2] not in indices
        }

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
        sequence = self._seed_sequence(
            camera_name=camera_name,
            stream=stream,
            capture_index=capture_index,
            logical_env_index=logical_env_index,
        )
        return np.random.default_rng(sequence)

    def _seed_sequence(
        self,
        *,
        camera_name: str,
        stream: str,
        capture_index: int,
        logical_env_index: int,
    ) -> np.random.SeedSequence:
        digest = hashlib.blake2b(camera_name.encode("utf-8"), digest_size=8).digest()
        camera_id = int.from_bytes(digest, byteorder="little", signed=False)
        stream_id = 0 if stream == "rgb" else 1
        return np.random.SeedSequence(
            [
                self._seed,
                int(capture_index),
                int(logical_env_index),
                camera_id & 0xFFFFFFFF,
                camera_id >> 32,
                stream_id,
            ]
        )

    def _torch_generator(
        self,
        *,
        camera_name: str,
        stream: str,
        capture_index: int,
        logical_env_index: int,
        device: Any,
    ) -> Any:
        import torch

        state = self._seed_sequence(
            camera_name=camera_name,
            stream=stream,
            capture_index=capture_index,
            logical_env_index=logical_env_index,
        ).generate_state(1, dtype=np.uint64)
        seed = int(state[0] % np.uint64(2**63 - 1))
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

    def _temporal_offset(
        self,
        config: Any,
        *,
        camera_name: str,
        stream: str,
        logical_env_index: int,
        rng: np.random.Generator,
    ) -> float:
        temporal = getattr(config, "temporal", None)
        if temporal is None:
            return 0.0
        jitter_std = float(getattr(temporal, "jitter_std", 0.0))
        ar1 = float(getattr(temporal, "ar1_coefficient", 0.0))
        drift_std = float(getattr(temporal, "drift_std", 0.0))
        drift_decay = float(getattr(temporal, "drift_decay", 1.0))
        if jitter_std == 0.0 and drift_std == 0.0:
            return 0.0
        state_key = (camera_name, stream, int(logical_env_index))
        previous_jitter, previous_drift = self._temporal_state.get(
            state_key, (0.0, 0.0)
        )
        innovation_std = jitter_std * np.sqrt(max(0.0, 1.0 - ar1 * ar1))
        jitter = ar1 * previous_jitter + float(rng.normal(0.0, innovation_std))
        drift = drift_decay * previous_drift + float(rng.normal(0.0, drift_std))
        self._temporal_state[state_key] = (jitter, drift)
        return jitter + drift

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
        temporal_offset = self._temporal_offset(
            config,
            camera_name=camera_name,
            stream=stream,
            logical_env_index=logical_env_index,
            rng=rng,
        )
        processed = self._process_value(
            value,
            config,
            stream=stream,
            rng=rng,
            structured=structured,
            clip_range=clip_range,
            temporal_offset=temporal_offset,
            camera_name=camera_name,
            capture_index=capture_index,
            logical_env_index=logical_env_index,
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
                row_rng = self._rng(
                    camera_name=camera_name,
                    stream=stream,
                    capture_index=capture_index,
                    logical_env_index=row,
                )
                processed = self._process_value(
                    item,
                    config,
                    stream=stream,
                    rng=row_rng,
                    structured=True,
                    clip_range=clip_range,
                    temporal_offset=self._temporal_offset(
                        config,
                        camera_name=camera_name,
                        stream=stream,
                        logical_env_index=row,
                        rng=row_rng,
                    ),
                    camera_name=camera_name,
                    capture_index=capture_index,
                    logical_env_index=row,
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
                row_rng = self._rng(
                    camera_name=camera_name,
                    stream=stream,
                    capture_index=capture_index,
                    logical_env_index=row,
                )
                processed = self._process_value(
                    array[row],
                    config,
                    stream=stream,
                    rng=row_rng,
                    structured=False,
                    clip_range=clip_range,
                    temporal_offset=self._temporal_offset(
                        config,
                        camera_name=camera_name,
                        stream=stream,
                        logical_env_index=row,
                        rng=row_rng,
                    ),
                    camera_name=camera_name,
                    capture_index=capture_index,
                    logical_env_index=row,
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
        rows = []
        for row in range(batch_size):
            row_rng = self._rng(
                camera_name=camera_name,
                stream=stream,
                capture_index=capture_index,
                logical_env_index=row,
            )
            rows.append(
                self._process_value(
                    array[row],
                    config,
                    stream=stream,
                    rng=row_rng,
                    structured=False,
                    clip_range=clip_range,
                    temporal_offset=self._temporal_offset(
                        config,
                        camera_name=camera_name,
                        stream=stream,
                        logical_env_index=row,
                        rng=row_rng,
                    ),
                    camera_name=camera_name,
                    capture_index=capture_index,
                    logical_env_index=row,
                )
            )
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
        temporal_offset: float = 0.0,
        camera_name: str = "",
        capture_index: int = 0,
        logical_env_index: int = 0,
    ) -> Any:
        if structured and isinstance(value, Mapping) and "encoding" in value:
            array = self._decode_image_message(value)
            processed = self._process_array(
                array,
                config,
                stream=stream,
                rng=rng,
                clip_range=clip_range,
                temporal_offset=temporal_offset,
                camera_name=camera_name,
                capture_index=capture_index,
                logical_env_index=logical_env_index,
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
            value,
            config,
            stream=stream,
            rng=rng,
            clip_range=clip_range,
            temporal_offset=temporal_offset,
            camera_name=camera_name,
            capture_index=capture_index,
            logical_env_index=logical_env_index,
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

    def _process_array(
        self,
        value: Any,
        config: Any,
        *,
        stream: str,
        rng: np.random.Generator,
        clip_range: Any = None,
        temporal_offset: float = 0.0,
        camera_name: str = "",
        capture_index: int = 0,
        logical_env_index: int = 0,
    ) -> Any:
        try:
            import torch
        except ImportError:  # pragma: no cover - torch is optional
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            generator = self._torch_generator(
                camera_name=camera_name,
                stream=stream,
                capture_index=capture_index,
                logical_env_index=logical_env_index,
                device=value.device,
            )
            if stream == "rgb":
                return self._process_rgb_torch(
                    value,
                    config,
                    generator,
                    temporal_offset=temporal_offset,
                )
            return self._process_depth_torch(
                value,
                config,
                generator,
                clip_range=clip_range,
                temporal_offset=temporal_offset,
            )

        array = np.asarray(value)
        if stream == "rgb":
            result = self._process_rgb(
                array, config, rng, temporal_offset=temporal_offset
            )
        else:
            result = self._process_depth(
                array,
                config,
                rng,
                clip_range=clip_range,
                temporal_offset=temporal_offset,
            )
        return result

    def _process_rgb(
        self,
        array: np.ndarray,
        config: Any,
        rng: np.random.Generator,
        *,
        temporal_offset: float = 0.0,
    ) -> np.ndarray:
        original_dtype = array.dtype
        normalized = array.astype(np.float64, copy=False)
        if np.issubdtype(original_dtype, np.integer):
            normalized = normalized / float(np.iinfo(original_dtype).max)
        else:
            normalized = normalized.copy()

        normalized *= float(getattr(config, "exposure", 1.0))
        normalized *= float(getattr(config, "gain", 1.0))
        illumination_std = float(getattr(config, "illumination_std", 0.0))
        gain_std = float(getattr(config, "gain_std", 0.0))
        if illumination_std > 0.0:
            normalized *= max(0.0, 1.0 + float(rng.normal(0.0, illumination_std)))
        if gain_std > 0.0:
            normalized *= max(0.0, 1.0 + float(rng.normal(0.0, gain_std)))
        normalized += temporal_offset

        shot_scale = getattr(config, "shot_noise_scale", None)
        if shot_scale is not None:
            scale = float(shot_scale)
            normalized = rng.poisson(np.clip(normalized, 0.0, 1.0) * scale) / scale
        gaussian_std = float(getattr(config, "gaussian_std", 0.0))
        if gaussian_std > 0.0:
            normalized += self._sample_numpy_noise(
                normalized.shape, gaussian_std, config, rng
            )
        quantization_levels = getattr(config, "quantization_levels", None)
        if quantization_levels is not None:
            levels = float(quantization_levels)
            normalized = np.round(normalized * levels) / levels
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

    def _process_depth(
        self,
        array: np.ndarray,
        config: Any,
        rng: np.random.Generator,
        *,
        clip_range: Any = None,
        temporal_offset: float = 0.0,
    ) -> np.ndarray:
        original_dtype = array.dtype
        depth = array.astype(np.float64, copy=True)
        valid = np.isfinite(depth) & (depth > 0.0)
        original_depth = depth.copy()
        resolution_factor = self._depth_resolution_factor(
            config,
            width=depth.shape[1],
            height=depth.shape[0],
        )
        configured_scale = float(getattr(config, "scale", 1.0))
        scale = 1.0 + (configured_scale - 1.0) * resolution_factor
        bias = float(getattr(config, "bias_m", 0.0)) * resolution_factor
        bias_power = float(getattr(config, "bias_distance_power", 0.0))
        depth[valid] = (
            original_depth[valid] * scale
            + bias * np.power(original_depth[valid], bias_power)
            + temporal_offset
        )
        gaussian_std = float(getattr(config, "gaussian_std_m", 0.0)) * resolution_factor
        relative_std = float(getattr(config, "relative_std", 0.0)) * resolution_factor
        sigma = np.sqrt(gaussian_std**2 + (relative_std * original_depth) ** 2)
        noise = self._sample_numpy_noise(depth.shape, 1.0, config, rng) * sigma
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

        invalid_probability = float(
            getattr(
                config,
                "invalid_probability",
                getattr(config, "dropout_probability", 0.0),
            )
        )
        invalid_probability = min(1.0, invalid_probability * resolution_factor)
        if invalid_probability > 0.0:
            valid &= rng.random(depth.shape) >= invalid_probability

        result = np.zeros_like(depth)
        result[valid] = depth[valid]
        return result.astype(original_dtype, copy=False)

    @staticmethod
    def _depth_resolution_factor(
        config: Any,
        *,
        width: int,
        height: int,
    ) -> float:
        """Return the scalar depth-error multiplier for the rendered resolution."""
        reference = getattr(config, "resolution_reference", None)
        power = float(getattr(config, "resolution_power", 0.0))
        if reference is None or power == 0.0:
            return 1.0
        reference_width, reference_height = (int(value) for value in reference)
        reference_pixels = reference_width * reference_height
        actual_pixels = int(width) * int(height)
        if actual_pixels <= 0:
            raise ValueError(
                f"depth image dimensions must be positive, got ({width}, {height})"
            )
        return (reference_pixels / actual_pixels) ** power

    @staticmethod
    def _sample_numpy_noise(
        shape: tuple[int, ...], scale: float, config: Any, rng: np.random.Generator
    ) -> np.ndarray:
        distribution = getattr(config, "distribution", None)
        kind = getattr(distribution, "kind", "gaussian")
        if kind == "gaussian":
            return rng.normal(0.0, scale, size=shape)
        degrees = float(getattr(distribution, "degrees_of_freedom", 5.0))
        samples = rng.standard_t(degrees, size=shape)
        return samples * scale / np.sqrt(degrees / (degrees - 2.0))

    @staticmethod
    def _sample_torch_noise(
        shape: tuple[int, ...],
        scale: Any,
        config: Any,
        generator: Any,
        *,
        device: Any,
        dtype: Any,
    ) -> Any:
        import torch

        distribution = getattr(config, "distribution", None)
        kind = getattr(distribution, "kind", "gaussian")
        if kind == "gaussian":
            return (
                torch.randn(shape, generator=generator, device=device, dtype=dtype)
                * scale
            )
        degrees = float(getattr(distribution, "degrees_of_freedom", 5.0))
        normal = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        concentration = torch.full(
            shape,
            degrees / 2.0,
            device=device,
            dtype=dtype,
        )
        gamma = torch._standard_gamma(concentration, generator=generator)
        student = normal / torch.sqrt(2.0 * gamma / degrees)
        return student * scale / np.sqrt(degrees / (degrees - 2.0))

    def _process_rgb_torch(
        self, value: Any, config: Any, generator: Any, *, temporal_offset: float
    ) -> Any:
        import torch

        original_dtype = value.dtype
        work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        normalized = value.to(dtype=work_dtype)
        if not value.dtype.is_floating_point:
            normalized = normalized / float(torch.iinfo(value.dtype).max)
        normalized = normalized * float(getattr(config, "exposure", 1.0))
        normalized = normalized * float(getattr(config, "gain", 1.0))
        illumination_std = float(getattr(config, "illumination_std", 0.0))
        gain_std = float(getattr(config, "gain_std", 0.0))
        if illumination_std > 0.0:
            factor = (
                1.0
                + torch.randn((), generator=generator, device=value.device)
                * illumination_std
            ).clamp_min(0.0)
            normalized = normalized * factor.to(dtype=work_dtype)
        if gain_std > 0.0:
            factor = (
                1.0
                + torch.randn((), generator=generator, device=value.device) * gain_std
            ).clamp_min(0.0)
            normalized = normalized * factor.to(dtype=work_dtype)
        normalized = normalized + temporal_offset
        shot_scale = getattr(config, "shot_noise_scale", None)
        if shot_scale is not None:
            scale = float(shot_scale)
            normalized = (
                torch.poisson(
                    torch.clamp(normalized, 0.0, 1.0) * scale, generator=generator
                )
                / scale
            )
        gaussian_std = float(getattr(config, "gaussian_std", 0.0))
        if gaussian_std > 0.0:
            normalized = normalized + self._sample_torch_noise(
                tuple(normalized.shape),
                gaussian_std,
                config,
                generator,
                device=value.device,
                dtype=work_dtype,
            )
        levels = getattr(config, "quantization_levels", None)
        if levels is not None:
            normalized = torch.round(normalized * float(levels)) / float(levels)
        dropout_probability = float(getattr(config, "dropout_probability", 0.0))
        if dropout_probability > 0.0:
            mask_shape = tuple(normalized.shape[:2])
            dropped = (
                torch.rand(mask_shape, generator=generator, device=value.device)
                < dropout_probability
            )
            if normalized.ndim > 2:
                dropped = dropped.unsqueeze(-1)
            normalized = normalized.masked_fill(dropped, 0.0)
        normalized = normalized.clamp(0.0, 1.0)
        if not value.dtype.is_floating_point:
            return torch.round(normalized * float(torch.iinfo(value.dtype).max)).to(
                dtype=original_dtype
            )
        return normalized.to(dtype=original_dtype)

    def _process_depth_torch(
        self,
        value: Any,
        config: Any,
        generator: Any,
        *,
        clip_range: Any,
        temporal_offset: float,
    ) -> Any:
        import torch

        original_dtype = value.dtype
        work_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
        depth = value.to(dtype=work_dtype)
        valid = torch.isfinite(depth) & (depth > 0.0)
        original_depth = depth.clone()
        resolution_factor = self._depth_resolution_factor(
            config,
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
        )
        configured_scale = float(getattr(config, "scale", 1.0))
        scale = 1.0 + (configured_scale - 1.0) * resolution_factor
        bias = float(getattr(config, "bias_m", 0.0)) * resolution_factor
        bias_power = float(getattr(config, "bias_distance_power", 0.0))
        transformed = (
            original_depth * scale
            + bias * torch.pow(original_depth.clamp_min(0.0), bias_power)
            + temporal_offset
        )
        gaussian_std = float(getattr(config, "gaussian_std_m", 0.0)) * resolution_factor
        relative_std = float(getattr(config, "relative_std", 0.0)) * resolution_factor
        sigma = torch.sqrt(gaussian_std**2 + (relative_std * original_depth) ** 2)
        transformed = (
            transformed
            + self._sample_torch_noise(
                tuple(depth.shape),
                1.0,
                config,
                generator,
                device=value.device,
                dtype=work_dtype,
            )
            * sigma
        )
        depth = transformed
        quantization_step = getattr(config, "quantization_step_m", None)
        if quantization_step is not None:
            step = float(quantization_step)
            depth = torch.where(valid, torch.round(depth / step) * step, depth)
        valid = valid & torch.isfinite(depth) & (depth > 0.0)
        if clip_range is not None:
            near_m, far_m = (float(v) for v in clip_range)
            valid = valid & (depth >= near_m) & (depth <= far_m)
        invalid_probability = float(
            getattr(
                config,
                "invalid_probability",
                getattr(config, "dropout_probability", 0.0),
            )
        )
        invalid_probability = min(1.0, invalid_probability * resolution_factor)
        if invalid_probability > 0.0:
            valid = valid & (
                torch.rand(tuple(depth.shape), generator=generator, device=value.device)
                >= invalid_probability
            )
        result = torch.zeros_like(depth)
        result = torch.where(valid, depth, result)
        return result.to(dtype=original_dtype)


__all__ = ["CameraNoiseProcessor"]
