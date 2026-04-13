"""Gaussian Splatting rendering extension for UnifiedMujocoEnv.

Usage
-----
Replace ``EnvConfig`` / ``UnifiedMujocoEnv`` with ``GSEnvConfig`` /
``GSUnifiedMujocoEnv`` in the YAML config and add a ``gaussian_render``
section:

.. code-block:: yaml

    env:
      _target_: auto_atom.basis.mjc.gs_mujoco_env.GSUnifiedMujocoEnv
      config:
        _target_: auto_atom.basis.mjc.gs_mujoco_env.GSEnvConfig
        model_path: assets/xmls/scenes/pick_and_place/demo.xml
        ...
        gaussian_render:
          body_gaussians:
            link1: /path/to/link1.ply
            link2: /path/to/link2.ply
          background_ply: /path/to/background.ply   # optional

When ``gaussian_render`` is set:
  - The ``model_validator`` automatically disables ``enable_color`` on cameras
    listed in ``gs_color_cameras`` (defaults to all cameras with
    ``enable_color=True``) so the native MuJoCo renderer skips their RGB output.
  - ``GSUnifiedMujocoEnv.capture_observation`` injects GS-rendered color
    images under the same observation keys the native renderer would have
    produced.
  - Depth rendering can also be routed through GS via ``gs_depth_cameras``.
    Rendering follows the same ``BatchSplatRenderer.batch_env_render`` flow used
    by ``press_da_button``. Foreground-only renders use accumulated depth
    ``∑wᵢzᵢ``; when ``background_ply`` is configured the foreground render
    receives ``bg_imgs`` from the background renderer, matching the third-party
    pipeline.
  - Mask and heat-map outputs are unaffected.
"""

from __future__ import annotations

import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, ConfigDict, Field, model_validator
from gaussian_renderer import BatchSplatConfig, MjxBatchSplatRenderer, GSRendererMuJoCo
from gaussian_renderer.core.util_gau import load_ply, save_ply
from auto_atom.basis.mjc.mujoco_env import (
    BatchedUnifiedMujocoEnv,
    EnvConfig,
    UnifiedMujocoEnv,
    create_image_data,
)


def create_image_data_batch(
    image_batch, timestamps, frame_id: str = "", tobytes: bool = True
):
    return [
        create_image_data(image, time_sec, frame_id, tobytes)
        for image, time_sec in zip(image_batch, timestamps / 1e9)
    ]


def _normalize_xyz_offset(offset: Any) -> tuple[float, float, float]:
    arr = np.asarray(offset, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(
            f"background offset must be length-3 xyz, got shape {arr.shape}"
        )
    return tuple(float(v) for v in arr.tolist())


def _resolve_background_offset(
    background_ply: str | None,
    background_offset: tuple[float, float, float] | None,
    background_offsets: Dict[str, tuple[float, float, float]],
) -> tuple[float, float, float]:
    if background_offset is not None:
        return _normalize_xyz_offset(background_offset)
    if not background_ply:
        return (0.0, 0.0, 0.0)

    bg_path = Path(background_ply)
    for key in (background_ply, str(bg_path), bg_path.name, bg_path.stem):
        if key in background_offsets:
            return _normalize_xyz_offset(background_offsets[key])
    return (0.0, 0.0, 0.0)


def _materialize_shifted_background_ply(
    background_ply: str | None,
    offset_xyz: tuple[float, float, float],
) -> str | None:
    if background_ply is None:
        return None

    offset_xyz = _normalize_xyz_offset(offset_xyz)
    if np.allclose(offset_xyz, 0.0):
        return background_ply

    src_path = Path(background_ply).expanduser().resolve()
    cache_key = hashlib.sha1(
        f"{src_path}|{offset_xyz[0]:.6f},{offset_xyz[1]:.6f},{offset_xyz[2]:.6f}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    cache_dir = Path(".cache/gs_background_offsets")
    cache_path = cache_dir / f"{src_path.stem}__bg_offset_{cache_key}.ply"
    if cache_path.exists():
        return str(cache_path)

    gaussians = load_ply(str(src_path))
    gaussians.xyz = gaussians.xyz + np.asarray(offset_xyz, dtype=np.float32)
    save_ply(gaussians, cache_path)
    return str(cache_path)


class GaussianRenderConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    body_gaussians: Dict[str, str] = Field(default_factory=dict)
    """Mapping from MuJoCo body name to PLY file path."""
    background_ply: str | None = None
    """Optional background PLY (loaded under the reserved key ``'background'``)."""
    background_offset: tuple[float, float, float] | None = None
    """Optional xyz translation applied to the configured background PLY."""
    background_offsets: Dict[str, tuple[float, float, float]] = Field(
        default_factory=dict
    )
    """Per-background xyz offsets keyed by full path, file name, or file stem."""

    def resolved_background_offset(self) -> tuple[float, float, float]:
        return _resolve_background_offset(
            self.background_ply,
            self.background_offset,
            self.background_offsets,
        )

    def resolved_background_ply(self) -> str | None:
        return _materialize_shifted_background_ply(
            self.background_ply,
            self.resolved_background_offset(),
        )


class GSEnvConfig(EnvConfig):
    """``EnvConfig`` extended with Gaussian Splatting rendering support.

    When ``gaussian_render`` is supplied the validator automatically sets
    ``enable_color=False`` on every camera so native RGB rendering is skipped.
    The names of cameras that originally had ``enable_color=True`` are stored
    in ``gs_color_cameras`` so ``GSUnifiedMujocoEnv`` knows which ones to
    render with GS.
    """

    gaussian_render: GaussianRenderConfig = GaussianRenderConfig()
    """Gaussian Splatting render config."""
    gs_color_cameras: Set[str] = Field(default_factory=set)
    """Names of cameras whose color output uses GS rendering. If empty, all cameras with ``enable_color=True`` are used."""
    gs_depth_cameras: Set[str] = Field(default_factory=set)
    """Names of cameras whose depth output uses GS rendering. If empty, all cameras with ``enable_depth=True`` are used."""
    gs_mask_cameras: Set[str] = Field(default_factory=set)
    """Names of cameras whose binary mask output uses GS rendering.
    Populated automatically from cameras with ``enable_mask=True``
    when ``mask_objects`` is non-empty; native MuJoCo segmentation is then disabled."""
    gs_heat_map_cameras: Set[str] = Field(default_factory=set)
    """Names of cameras whose heat_map output uses GS rendering.
    Populated automatically from cameras with ``enable_heat_map=True``
    when ``mask_objects`` is non-empty."""
    to_numpy: bool = True
    """Whether to convert GS renderer output to numpy arrays.
    Defaults to True for consistency with all other observation data types."""

    @model_validator(mode="after")
    def setup_gs_cameras(self):
        color_cams = {c.name for c in self.cameras if c.enable_color}
        gs_color = color_cams if not self.gs_color_cameras else self.gs_color_cameras
        if gs_color - color_cams:
            raise ValueError(
                f"gs_color_cameras {gs_color} must be a subset of "
                f"cameras with enable_color=True: {color_cams}"
            )
        object.__setattr__(self, "gs_color_cameras", gs_color)

        depth_cams = {c.name for c in self.cameras if c.enable_depth}
        gs_depth = depth_cams if not self.gs_depth_cameras else self.gs_depth_cameras
        if gs_depth - depth_cams:
            raise ValueError(
                f"gs_depth_cameras {gs_depth} must be a subset of "
                f"cameras with enable_depth=True: {depth_cams}"
            )
        object.__setattr__(self, "gs_depth_cameras", gs_depth)

        # Disable native mask/heat_map when GS mask renderers will handle them
        gs_mask = (
            {c.name for c in self.cameras if c.enable_mask}
            if not self.gs_mask_cameras
            else self.gs_mask_cameras
        )
        gs_heat = (
            {c.name for c in self.cameras if c.enable_heat_map}
            if not self.gs_heat_map_cameras
            else self.gs_heat_map_cameras
        )
        if not self.mask_objects:
            gs_mask = set()
            gs_heat = set()
        object.__setattr__(self, "gs_mask_cameras", gs_mask)
        object.__setattr__(self, "gs_heat_map_cameras", gs_heat)

        for cam in self.cameras:
            if cam.name in gs_color:
                object.__setattr__(cam, "enable_color", False)
            if cam.name in gs_depth:
                object.__setattr__(cam, "enable_depth", False)
            if cam.name in gs_mask:
                object.__setattr__(cam, "enable_mask", False)
            if cam.name in gs_heat:
                object.__setattr__(cam, "enable_heat_map", False)
        return self


class GSUnifiedMujocoEnv(UnifiedMujocoEnv):
    """``UnifiedMujocoEnv`` that replaces native RGB with Gaussian Splatting."""

    _GS_MASK_ALPHA_THRESHOLD = 0.5
    _GS_MASK_DEPTH_EPS = 0.01

    def __init__(self, config: GSEnvConfig) -> None:
        super().__init__(config)
        self.config: GSEnvConfig
        gs_cfg = config.gaussian_render
        self._gs_background_source_ply = gs_cfg.background_ply
        self._gs_renderer: GSRendererMuJoCo | None = None
        fg_cfg = BatchSplatConfig(
            body_gaussians=dict(gs_cfg.body_gaussians),
            background_ply=None,
            minibatch=512,
        )
        self._fg_gs_renderer = MjxBatchSplatRenderer(fg_cfg, self.model)
        self._bg_gs_renderer: MjxBatchSplatRenderer | None = None
        self.set_background_offset(gs_cfg.resolved_background_offset())
        background_ply = gs_cfg.resolved_background_ply()
        if background_ply:
            self.get_logger().debug(
                f"GS renderer initialised with {len(gs_cfg.body_gaussians)} body gaussian(s) + background"
            )
        else:
            self.get_logger().debug(
                f"GS renderer initialised with {len(gs_cfg.body_gaussians)} body gaussian(s)"
            )
        self._gs_mask_renderers = self._build_gs_mask_renderers(
            dict(gs_cfg.body_gaussians)
        )

    def set_background_offset(
        self, offset_xyz: tuple[float, float, float] | list[float]
    ) -> tuple[float, float, float]:
        offset_xyz = _normalize_xyz_offset(offset_xyz)
        background_ply = _materialize_shifted_background_ply(
            self._gs_background_source_ply,
            offset_xyz,
        )
        combined_models = dict(self.config.gaussian_render.body_gaussians)
        if background_ply:
            combined_models["background"] = background_ply
        self._gs_renderer = GSRendererMuJoCo(combined_models, self.model)
        self._bg_gs_renderer = (
            MjxBatchSplatRenderer(
                BatchSplatConfig(
                    body_gaussians={},
                    background_ply=background_ply,
                    minibatch=512,
                ),
                self.model,
            )
            if background_ply
            else None
        )
        object.__setattr__(self.config.gaussian_render, "background_offset", offset_xyz)
        return offset_xyz

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        obs = super().capture_observation()
        self._inject_gs_renders(obs)
        return obs

    def _inject_gs_renders(self, obs: dict[str, dict[str, Any]]) -> None:
        """Render GS color and/or depth and insert into *obs* in-place."""
        gs_color_set = self.config.gs_color_cameras
        gs_depth_set = self.config.gs_depth_cameras
        gs_mask_set = self.config.gs_mask_cameras | self.config.gs_heat_map_cameras
        all_gs_cams = [
            c
            for c in self._camera_specs
            if c in gs_color_set | gs_depth_set | gs_mask_set
        ]
        if not all_gs_cams:
            return

        kc = self._key_creator
        t = int(self.data.time * 1e9) if self.config.stamp_ns else float(self.data.time)

        for cam_name in all_gs_cams:
            spec = self._camera_specs[cam_name]
            cam_id = self._camera_ids[cam_name]
            if cam_name in gs_color_set:
                rgb_t = self._render_gs_color_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
                rgb = torch.clamp(rgb_t, 0.0, 1.0).mul(255).to(torch.uint8)
                if self.config.to_numpy:
                    rgb = rgb.cpu().numpy()
                obs[kc.create_color_key(cam_name)] = {"data": rgb, "t": t}
            depth_t: torch.Tensor | None = None
            scene_depth_t: torch.Tensor | None = None
            need_mask = cam_name in gs_mask_set and self._gs_mask_renderers
            if cam_name in gs_depth_set or need_mask:
                (
                    _fg_rgb,
                    fg_depth,
                    _bg_rgb,
                    bg_depth,
                    _full_rgb,
                    full_depth,
                ) = self._render_gs_camera_batch(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
                scene_depth_t = self._compose_mask_scene_depth(
                    fg_depth=fg_depth,
                    bg_depth=bg_depth,
                )
                depth_t = full_depth[0, 0]
            if cam_name in gs_depth_set and depth_t is not None:
                depth = depth_t[..., 0]  # (H, W, 1) -> (H, W)
                if self.config.to_numpy:
                    depth = depth.cpu().numpy()
                    depth[depth > spec.depth_max] = 0.0
                else:
                    depth = torch.where(
                        depth > spec.depth_max, torch.zeros_like(depth), depth
                    )
                obs[kc.create_depth_key(cam_name)] = {
                    "data": depth,
                    "t": t,
                }
            if need_mask and scene_depth_t is not None:
                binary_mask, heat_map = self._render_gs_masks_for_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                    scene_depth_t=scene_depth_t,
                )
                if cam_name in self.config.gs_mask_cameras:
                    obs[kc.create_mask_key(cam_name)] = {
                        "data": binary_mask,
                        "t": t,
                    }
                if cam_name in self.config.gs_heat_map_cameras:
                    obs[kc.create_heat_map_key(cam_name)] = {
                        "data": heat_map,
                        "t": t,
                    }

    def _render_gs_camera(
        self,
        *,
        cam_id: int,
        width: int,
        height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render one camera with legacy GS RGB + batch depth compositing."""
        # Keep RGB on the legacy GSRendererMuJoCo path (same as commit 5c46007),
        # because it has more natural fg/bg appearance than batch depth-driven swaps.
        rgb_t = self._render_gs_color_camera(
            cam_id=cam_id,
            width=width,
            height=height,
        )
        _, _, _, _, _, full_depth = self._render_gs_camera_batch(
            cam_id=cam_id,
            width=width,
            height=height,
        )
        # batch_env_render returns (Nenv, Ncam, H, W, C); here Nenv=Ncam=1.
        return rgb_t, full_depth[0, 0]

    def _render_gs_color_camera(
        self,
        *,
        cam_id: int,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Render RGB strictly through GSRendererMuJoCo to match the old path."""
        self._gs_renderer.update_gaussians(self.data)
        result = self._gs_renderer.render(
            self.model, self.data, [cam_id], width, height
        )
        if cam_id not in result:
            raise RuntimeError(
                f"GS renderer did not return output for camera ID {cam_id}"
            )
        rgb_t, _ = result[cam_id]
        return rgb_t

    def _render_gs_camera_batch(
        self,
        *,
        cam_id: int,
        width: int,
        height: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Render one camera through the same batch_env_render flow as press_da_button."""
        cam_pos, cam_xmat, fovy = self._compute_camera_batch_inputs(cam_id)
        body_pos, body_quat = self._compute_body_batch_inputs()

        fg_gsb = self._fg_gs_renderer.batch_update_gaussians(body_pos, body_quat)

        bg_rgb = None
        bg_depth = None
        bg_imgs = None
        if self._bg_gs_renderer is not None:
            bg_gsb = self._bg_gs_renderer.batch_update_gaussians(body_pos, body_quat)
            bg_rgb, bg_depth = self._bg_gs_renderer.batch_env_render(
                bg_gsb, cam_pos, cam_xmat, height, width, fovy
            )
            bg_imgs = bg_rgb

        fg_rgb, fg_depth = self._fg_gs_renderer.batch_env_render(
            fg_gsb,
            cam_pos,
            cam_xmat,
            height,
            width,
            fovy,
            bg_imgs=bg_imgs,
        )
        alphas = self._fg_gs_renderer.rasterizations[1]
        if bg_depth is not None:
            # `fg_rgb` is already composited against `bg_imgs` inside
            # batch_env_render. Do not blend background color a second time,
            # otherwise edges wash out and produce white halos.
            full_rgb = fg_rgb
            full_depth = fg_depth * alphas + bg_depth * (1 - alphas)
        else:
            full_rgb = fg_rgb
            full_depth = fg_depth
        return fg_rgb, fg_depth, bg_rgb, bg_depth, full_rgb, full_depth

    @staticmethod
    def _compose_mask_scene_depth(
        *,
        fg_depth: torch.Tensor,
        bg_depth: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build a crisp visible-surface depth map for mask occlusion tests."""
        if bg_depth is None:
            return fg_depth

        fg_valid = fg_depth > 0
        bg_valid = bg_depth > 0
        bg_closer = bg_valid & ((~fg_valid) | (bg_depth + 1e-4 < fg_depth))
        return torch.where(bg_closer, bg_depth, fg_depth)

    def _compute_camera_batch_inputs(
        self, cam_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cam_pos = np.asarray(
            self.data.cam_xpos[cam_id : cam_id + 1], dtype=np.float32
        ).reshape(1, 1, 3)
        cam_xmat = np.asarray(
            self.data.cam_xmat[cam_id : cam_id + 1], dtype=np.float32
        ).reshape(1, 1, 9)
        fovy = np.asarray(
            self.model.cam_fovy[cam_id : cam_id + 1], dtype=np.float32
        ).reshape(1, 1)
        return cam_pos, cam_xmat, fovy

    def _compute_body_batch_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        body_pos = np.asarray(self.data.xpos, dtype=np.float32).reshape(
            1, self.model.nbody, 3
        )
        # MuJoCo stores body xquat in wxyz order. Keep that order here so it
        # matches the scalar_first=True path used by MjxBatchSplatRenderer.
        body_quat_wxyz = np.asarray(self.data.xquat, dtype=np.float32).reshape(
            1, self.model.nbody, 4
        )
        return body_pos, body_quat_wxyz

    def _build_gs_mask_renderers(
        self, body_gaussians: Dict[str, str]
    ) -> dict[str, MjxBatchSplatRenderer]:
        """Create one-object GS renderers for configured mask_objects."""
        renderers: dict[str, MjxBatchSplatRenderer] = {}
        for object_name in self.config.mask_objects:
            body_name = self._resolve_gs_mask_body_name(object_name, body_gaussians)
            if body_name is None:
                self.get_logger().warning(
                    "Skipping GS mask renderer for '%s': no matching GS body found in body_gaussians.",
                    object_name,
                )
                continue
            mask_cfg = BatchSplatConfig(
                body_gaussians={body_name: body_gaussians[body_name]},
                background_ply=None,
                minibatch=512,
            )
            renderers[object_name] = MjxBatchSplatRenderer(mask_cfg, self.model)
        return renderers

    @staticmethod
    def _resolve_gs_mask_body_name(
        object_name: str, body_gaussians: Dict[str, str]
    ) -> str | None:
        if object_name in body_gaussians:
            return object_name
        gs_name = f"{object_name}_gs"
        if gs_name in body_gaussians:
            return gs_name
        return None

    def _render_gs_masks_for_camera(
        self,
        *,
        cam_id: int,
        width: int,
        height: int,
        scene_depth_t: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render GS masks with occlusion culling.

        All heavy computation stays on GPU; only the final uint8 masks are
        transferred to CPU.
        """
        cam_pos, cam_xmat, fovy = self._compute_camera_batch_inputs(cam_id)
        body_pos, body_quat = self._compute_body_batch_inputs()
        if scene_depth_t.ndim == 5:
            scene_depth = scene_depth_t[0, 0, :, :, 0]
        elif scene_depth_t.ndim == 3 and scene_depth_t.shape[-1] == 1:
            scene_depth = scene_depth_t[:, :, 0]
        elif scene_depth_t.ndim == 2:
            scene_depth = scene_depth_t
        else:
            raise TypeError(
                "scene_depth_t must be (1,1,H,W,1), (H,W,1), or (H,W), "
                f"got shape {tuple(scene_depth_t.shape)}"
            )
        scene_depth = torch.nan_to_num(
            scene_depth.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )

        dev = scene_depth.device
        n_heatmap_ops = len(self.config.heatmap_operations)
        binary_mask_t = torch.zeros((height, width), dtype=torch.bool, device=dev)
        heat_map_t = torch.zeros(
            (height, width, n_heatmap_ops), dtype=torch.bool, device=dev
        )

        heatmap_ops_index = {
            op: i for i, op in enumerate(self.config.heatmap_operations)
        }

        # Convert shared inputs to GPU tensors once.
        body_pos_t = torch.as_tensor(body_pos, device=dev, dtype=torch.float32)
        body_quat_t = torch.as_tensor(body_quat, device=dev, dtype=torch.float32)
        cam_pos_t = torch.as_tensor(cam_pos, device=dev, dtype=torch.float32)
        cam_xmat_t = torch.as_tensor(cam_xmat, device=dev, dtype=torch.float32)

        for object_name, renderer in self._gs_mask_renderers.items():
            gsb = renderer.batch_update_gaussians(body_pos_t, body_quat_t)
            alpha_t, obj_depth_t = renderer.batch_env_render(
                gsb, cam_pos_t, cam_xmat_t, height, width, fovy
            )

            # Stay on GPU: max over channels, clean NaN
            alpha_max = alpha_t[0, 0].detach().max(dim=-1).values  # (H, W)
            obj_depth = torch.nan_to_num(
                obj_depth_t[0, 0, :, :, 0].detach(), nan=0.0, posinf=0.0, neginf=0.0
            )  # (H, W)

            visible = alpha_max > self._GS_MASK_ALPHA_THRESHOLD
            depth_valid = (scene_depth > 0.0) & (obj_depth > 0.0)
            occluded = depth_valid & (obj_depth > scene_depth + self._GS_MASK_DEPTH_EPS)
            visible = visible & ~occluded
            binary_mask_t |= visible

            operation_name = self._interest_object_operations.get(object_name)
            if operation_name is None:
                continue
            channel_idx = heatmap_ops_index.get(operation_name)
            if channel_idx is None:
                continue
            heat_map_t[..., channel_idx] |= visible

        # Single GPU→CPU transfer
        binary_mask = binary_mask_t.to(torch.uint8).cpu().numpy()
        heat_map = heat_map_t.to(torch.uint8).cpu().numpy()
        return binary_mask, heat_map

    def close(self) -> None:
        self._gs_renderer = None
        self._fg_gs_renderer = None
        self._bg_gs_renderer = None
        self._gs_mask_renderers = {}
        super().close()


class BatchedGSUnifiedMujocoEnv(BatchedUnifiedMujocoEnv):
    """Truly batched GS rendering across multiple ``UnifiedMujocoEnv`` replicas.

    Unlike the single-env ``GSUnifiedMujocoEnv`` which renders one environment
    at a time, this class shares a single set of ``MjxBatchSplatRenderer``
    instances and calls ``batch_env_render`` once with ``(Nenv, 1, ...)``
    tensors, leveraging GPU parallelism across environments.
    """

    _GS_MASK_ALPHA_THRESHOLD = 0.5
    _GS_MASK_DEPTH_EPS = 0.01

    def __init__(self, config: Optional[GSEnvConfig] = None, **kwargs) -> None:
        if config is None:
            config = GSEnvConfig.model_validate(kwargs)
        # Parent creates N plain UnifiedMujocoEnv instances (physics only).
        # GSEnvConfig validator already disabled native color/depth on GS cameras.
        super().__init__(config, **kwargs)

        gs_cfg = config.gaussian_render
        self._gs_background_source_ply = gs_cfg.background_ply

        # Single shared foreground renderer
        self._fg_gs_renderer = MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians=dict(gs_cfg.body_gaussians),
                background_ply=None,
                minibatch=512,
            ),
            self.envs[0].model,
        )

        # Single shared background renderer (optional)
        self._bg_gs_renderer: MjxBatchSplatRenderer | None = None

        # Per-object mask renderers (shared) — build via env[0] which has
        # .model, .get_logger(), and .config.mask_objects needed by the builder.
        self._gs_mask_renderers = self._build_shared_mask_renderers(
            dict(gs_cfg.body_gaussians)
        )

        # Cache camera specs from env[0] (homogeneous)
        self._camera_specs = self.envs[0]._camera_specs
        self._camera_ids = self.envs[0]._camera_ids

        # Background cache: (cam_ids_tuple, w, h) → (bg_rgb, bg_depth) tensors
        self._bg_cache: dict[
            tuple[tuple[int, ...], int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self.set_background_offset(gs_cfg.resolved_background_offset())
        background_ply = gs_cfg.resolved_background_ply()

        n_bodies = len(gs_cfg.body_gaussians)
        bg_str = " + background" if background_ply else ""
        self.envs[0].get_logger().debug(
            f"Batched GS renderer initialised with {n_bodies} body gaussian(s){bg_str}"
        )

    def set_background_offset(
        self, offset_xyz: tuple[float, float, float] | list[float]
    ) -> tuple[float, float, float]:
        offset_xyz = _normalize_xyz_offset(offset_xyz)
        background_ply = _materialize_shifted_background_ply(
            self._gs_background_source_ply,
            offset_xyz,
        )
        self._bg_gs_renderer = (
            MjxBatchSplatRenderer(
                BatchSplatConfig(
                    body_gaussians={},
                    background_ply=background_ply,
                    minibatch=512,
                ),
                self.envs[0].model,
            )
            if background_ply
            else None
        )
        self._bg_cache.clear()
        object.__setattr__(self.config.gaussian_render, "background_offset", offset_xyz)
        return offset_xyz

    # ------------------------------------------------------------------
    # observation capture
    # ------------------------------------------------------------------

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        obs = super().capture_observation()
        self._inject_batched_gs_renders(obs)
        return obs

    def _inject_batched_gs_renders(self, obs: dict[str, dict[str, Any]]) -> None:
        """Batch-render GS color/depth/mask across all envs and cameras.

        Cameras are grouped by resolution so that ``batch_env_render`` is
        called once per resolution group with ``Ncam > 1``, rather than once
        per camera.
        """
        gs_color_set = self.config.gs_color_cameras
        gs_depth_set = self.config.gs_depth_cameras
        gs_mask_set = self.config.gs_mask_cameras | self.config.gs_heat_map_cameras
        all_gs_cams = [
            c
            for c in self._camera_specs
            if c in gs_color_set | gs_depth_set | gs_mask_set
        ]
        if not all_gs_cams:
            return

        structured: bool = self.config.structured
        timestamps = np.array(
            [
                int(env.data.time * 1e9)
                if self.config.stamp_ns
                else float(env.data.time)
                for env in self.envs
            ]
        )

        # Gather body poses from all envs: (Nenv, Nbody, 3/4)
        body_pos = np.stack(
            [np.asarray(env.data.xpos, dtype=np.float32) for env in self.envs]
        )
        body_quat = np.stack(
            [np.asarray(env.data.xquat, dtype=np.float32) for env in self.envs]
        )

        # Update foreground gaussians ONCE for all envs
        fg_gsb = self._fg_gs_renderer.batch_update_gaussians(body_pos, body_quat)

        # ---- Group cameras by resolution for multi-camera batching ----
        # batch_env_render requires uniform (H, W), so we group by resolution
        # and render all cameras in a group with a single call.
        from collections import OrderedDict

        res_groups: OrderedDict[tuple[int, int, bool], list[str]] = OrderedDict()
        for cam_name in all_gs_cams:
            spec = self._camera_specs[cam_name]
            key = (spec.height, spec.width, spec.is_static)
            res_groups.setdefault(key, []).append(cam_name)

        for (H, W, is_static), cam_names in res_groups.items():
            Ncam = len(cam_names)
            cam_ids = [self._camera_ids[c] for c in cam_names]

            # Gather camera params: (Nenv, Ncam, ...)
            cam_pos = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(env.data.cam_xpos[cid], dtype=np.float32)
                            for cid in cam_ids
                        ]
                    )
                    for env in self.envs
                ]
            )  # (Nenv, Ncam, 3)
            cam_xmat = np.stack(
                [
                    np.stack(
                        [
                            np.asarray(env.data.cam_xmat[cid], dtype=np.float32)
                            for cid in cam_ids
                        ]
                    )
                    for env in self.envs
                ]
            )  # (Nenv, Ncam, 9)
            fovy = np.broadcast_to(
                np.asarray(
                    [self.envs[0].model.cam_fovy[cid] for cid in cam_ids],
                    dtype=np.float32,
                ).reshape(1, Ncam),
                (self.batch_size, Ncam),
            ).copy()  # (Nenv, Ncam)

            # Determine what this group needs
            any_color = any(c in gs_color_set for c in cam_names)
            any_depth = any(c in gs_depth_set for c in cam_names)
            any_mask = (
                any(c in gs_mask_set for c in cam_names) and self._gs_mask_renderers
            )
            need_depth_render = any_depth or any_mask

            # ---- Single FG+BG render for all cameras in this group ----
            fg_rgb = fg_depth = bg_depth = full_rgb = full_depth = alphas = None
            if need_depth_render:
                fg_rgb, fg_depth, bg_depth, full_rgb, full_depth, alphas = (
                    self._render_batched_multicam(
                        fg_gsb,
                        cam_pos,
                        cam_xmat,
                        H,
                        W,
                        fovy,
                        body_pos,
                        body_quat,
                        cam_ids,
                        use_cache=is_static,
                    )
                )
            elif any_color:
                # Color-only group — still batch all cameras
                cached = self._get_cached_bg_multicam(
                    cam_ids,
                    W,
                    H,
                    cam_pos,
                    cam_xmat,
                    fovy,
                    body_pos,
                    body_quat,
                    use_cache=is_static,
                )
                bg_imgs = cached[0] if cached is not None else None
                fg_rgb, _ = self._fg_gs_renderer.batch_env_render(
                    fg_gsb, cam_pos, cam_xmat, H, W, fovy, bg_imgs=bg_imgs
                )
                # fg_rgb: (Nenv, Ncam, H, W, 3)

            # ---- Distribute per-camera outputs ----
            kc = self._key_creator
            for cam_idx, cam_name in enumerate(cam_names):
                spec = self._camera_specs[cam_name]
                # color
                has_color = True
                if cam_name in gs_color_set and full_rgb is not None:
                    rgb = full_rgb[:, cam_idx]  # (Nenv, H, W, 3)
                    rgb = torch.clamp(rgb, 0.0, 1.0).mul(255).to(torch.uint8)
                    if self.config.to_numpy:
                        rgb = rgb.cpu().numpy()
                elif cam_name in gs_color_set and fg_rgb is not None:
                    rgb = fg_rgb[:, cam_idx]
                    rgb = torch.clamp(rgb, 0.0, 1.0).mul(255).to(torch.uint8)
                    if self.config.to_numpy:
                        rgb = rgb.cpu().numpy()
                else:
                    has_color = False
                if has_color:
                    obs[kc.create_color_key(cam_name)] = {
                        "data": rgb
                        if not structured
                        else create_image_data_batch(
                            rgb, timestamps, cam_name, tobytes=False
                        ),
                        "t": timestamps,
                    }

                # depth
                if cam_name in gs_depth_set and full_depth is not None:
                    depth = full_depth[:, cam_idx, :, :, 0]  # (Nenv, H, W)
                    if self.config.to_numpy:
                        depth = depth.cpu().numpy()
                        depth[depth > spec.depth_max] = 0.0
                    else:
                        depth = torch.where(
                            depth > spec.depth_max,
                            torch.zeros_like(depth),
                            depth,
                        )
                    data = (
                        create_image_data_batch(depth, timestamps, cam_name)
                        if structured
                        else depth
                    )
                    obs[kc.create_depth_key(cam_name)] = {
                        "data": data,
                        "t": timestamps,
                    }

            # ---- Mask rendering: one call per object, all cameras ----
            if any_mask and fg_depth is not None:
                scene_depth_t = GSUnifiedMujocoEnv._compose_mask_scene_depth(
                    fg_depth=fg_depth,
                    bg_depth=bg_depth,
                )
                # scene_depth_t: (Nenv, Ncam, H, W, 1)
                all_masks, all_heat_maps = self._render_batched_gs_masks_multicam(
                    cam_pos=cam_pos,
                    cam_xmat=cam_xmat,
                    fovy=fovy,
                    height=H,
                    width=W,
                    body_pos=body_pos,
                    body_quat=body_quat,
                    scene_depth_t=scene_depth_t,
                )
                # all_masks: (Nenv, Ncam, H, W), all_heat_maps: (Nenv, Ncam, H, W, Nops)
                for cam_idx, cam_name in enumerate(cam_names):
                    if cam_name not in gs_mask_set:
                        continue

                    if cam_name in self.config.gs_mask_cameras:
                        data = (
                            create_image_data_batch(
                                all_masks[:, cam_idx], timestamps, cam_name
                            )
                            if structured
                            else all_masks[:, cam_idx]
                        )
                        obs[kc.create_mask_key(cam_name)] = {
                            "data": data,
                            "t": timestamps,
                        }
                    if cam_name in self.config.gs_heat_map_cameras:
                        data = (
                            create_image_data_batch(
                                all_heat_maps[:, cam_idx], timestamps, cam_name
                            )
                            if structured
                            else all_heat_maps[:, cam_idx]
                        )
                        obs[kc.create_heat_map_key(cam_name)] = {
                            "data": data,
                            "t": timestamps,
                        }

    # ------------------------------------------------------------------
    # batch GS rendering helpers
    # ------------------------------------------------------------------

    def _build_shared_mask_renderers(
        self, body_gaussians: Dict[str, str]
    ) -> dict[str, MjxBatchSplatRenderer]:
        """Create one-object GS renderers for configured mask_objects."""
        env0 = self.envs[0]
        renderers: dict[str, MjxBatchSplatRenderer] = {}
        for object_name in self.config.mask_objects:
            body_name = GSUnifiedMujocoEnv._resolve_gs_mask_body_name(
                object_name, body_gaussians
            )
            if body_name is None:
                env0.get_logger().warning(
                    "Skipping GS mask renderer for '%s': no matching GS body found.",
                    object_name,
                )
                continue
            mask_cfg = BatchSplatConfig(
                body_gaussians={body_name: body_gaussians[body_name]},
                background_ply=None,
                minibatch=512,
            )
            renderers[object_name] = MjxBatchSplatRenderer(mask_cfg, env0.model)
        return renderers

    def _get_cached_bg_multicam(
        self,
        cam_ids: list[int],
        width: int,
        height: int,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        use_cache: bool = True,
    ) -> torch.Tensor | None:
        """Return background (rgb, depth) for multiple cameras.

        When *use_cache* is ``True`` (static cameras), the result is cached by
        ``(tuple(cam_ids), width, height)`` and reused across frames.  When
        ``False`` (dynamic / moving cameras), the background is re-rendered
        every call.

        Returns ``(bg_rgb, bg_depth)`` each of shape
        ``(Nenv, Ncam, H, W, C)`` or ``None`` when no background renderer is
        configured.
        """
        if self._bg_gs_renderer is None:
            return None
        cache_key = (tuple(cam_ids), width, height)
        if use_cache and cache_key in self._bg_cache:
            return self._bg_cache[cache_key]
        bg_gsb = self._bg_gs_renderer.batch_update_gaussians(body_pos, body_quat)
        bg_rgb, bg_depth = self._bg_gs_renderer.batch_env_render(
            bg_gsb, cam_pos, cam_xmat, height, width, fovy
        )
        if use_cache:
            self._bg_cache[cache_key] = (bg_rgb, bg_depth)
        return (bg_rgb, bg_depth)

    def _render_batched_multicam(
        self,
        fg_gsb,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        height: int,
        width: int,
        fovy: np.ndarray,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        cam_ids: list[int],
        use_cache: bool = True,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Render FG+BG for *multiple* cameras in a single ``batch_env_render``.

        Parameters have shapes ``(Nenv, Ncam, ...)``.

        Returns
        -------
        fg_rgb : (Nenv, Ncam, H, W, 3)
        fg_depth : (Nenv, Ncam, H, W, 1)
        bg_depth : (Nenv, Ncam, H, W, 1) or None
        full_rgb : (Nenv, Ncam, H, W, 3)
        full_depth : (Nenv, Ncam, H, W, 1)
        alphas : (Nenv, Ncam, H, W, 1)
        """
        cached = self._get_cached_bg_multicam(
            cam_ids,
            width,
            height,
            cam_pos,
            cam_xmat,
            fovy,
            body_pos,
            body_quat,
            use_cache=use_cache,
        )
        bg_imgs = cached[0] if cached is not None else None

        fg_rgb, fg_depth = self._fg_gs_renderer.batch_env_render(
            fg_gsb, cam_pos, cam_xmat, height, width, fovy, bg_imgs=bg_imgs
        )
        alphas = self._fg_gs_renderer.rasterizations[1]

        bg_depth = None
        if cached is not None:
            _, bg_depth = cached
            full_rgb = fg_rgb
            full_depth = fg_depth * alphas + bg_depth * (1 - alphas)
        else:
            full_rgb = fg_rgb
            full_depth = fg_depth

        return fg_rgb, fg_depth, bg_depth, full_rgb, full_depth, alphas

    def _render_batched_gs_masks_multicam(
        self,
        *,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy: np.ndarray,
        height: int,
        width: int,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        scene_depth_t: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render GS masks for all envs × all cameras with occlusion culling.

        All heavy computation (max, nan_to_num, comparisons) stays on GPU;
        only the final uint8 masks are transferred to CPU.

        Parameters
        ----------
        scene_depth_t : (Nenv, Ncam, H, W, 1)

        Returns
        -------
        binary_mask : (Nenv, Ncam, H, W) uint8
        heat_map : (Nenv, Ncam, H, W, Nops) uint8
        """
        B = self.batch_size
        # scene_depth_t: (Nenv, Ncam, H, W, 1) → (Nenv, Ncam, H, W)
        if scene_depth_t.ndim == 5:
            scene_depth = scene_depth_t[..., 0]
        elif scene_depth_t.ndim == 4:
            scene_depth = scene_depth_t
        else:
            raise TypeError(
                "scene_depth_t must be (B,Ncam,H,W,1) or (B,Ncam,H,W), "
                f"got shape {tuple(scene_depth_t.shape)}"
            )
        Ncam = scene_depth.shape[1]
        scene_depth = torch.nan_to_num(
            scene_depth.detach(), nan=0.0, posinf=0.0, neginf=0.0
        )

        n_heatmap_ops = len(self.config.heatmap_operations)
        dev = scene_depth.device
        binary_mask_t = torch.zeros(
            (B, Ncam, height, width), dtype=torch.bool, device=dev
        )
        heat_map_t = torch.zeros(
            (B, Ncam, height, width, n_heatmap_ops), dtype=torch.bool, device=dev
        )

        # Pre-build per-object → (env_idx, channel_idx) mapping to avoid
        # per-env Python loops inside the hot path.
        heatmap_ops_list = self.config.heatmap_operations
        heatmap_ops_index = {op: i for i, op in enumerate(heatmap_ops_list)}

        # Convert shared inputs to GPU tensors ONCE so that per-object
        # batch_update_gaussians / batch_env_render skip their internal
        # torch.tensor() calls (isinstance check short-circuits).
        body_pos_t = torch.as_tensor(body_pos, device=dev, dtype=torch.float32)
        body_quat_t = torch.as_tensor(body_quat, device=dev, dtype=torch.float32)
        cam_pos_t = torch.as_tensor(cam_pos, device=dev, dtype=torch.float32)
        cam_xmat_t = torch.as_tensor(cam_xmat, device=dev, dtype=torch.float32)

        for object_name, renderer in self._gs_mask_renderers.items():
            gsb = renderer.batch_update_gaussians(body_pos_t, body_quat_t)
            alpha_t, obj_depth_t = renderer.batch_env_render(
                gsb, cam_pos_t, cam_xmat_t, height, width, fovy
            )
            # alpha_t: (B, Ncam, H, W, 3), obj_depth_t: (B, Ncam, H, W, 1)
            # max over channels, stay on GPU
            alpha_max = alpha_t.detach().max(dim=-1).values  # (B, Ncam, H, W)
            obj_depth = torch.nan_to_num(
                obj_depth_t[..., 0].detach(), nan=0.0, posinf=0.0, neginf=0.0
            )  # (B, Ncam, H, W)

            visible = alpha_max > self._GS_MASK_ALPHA_THRESHOLD
            depth_valid = (scene_depth > 0.0) & (obj_depth > 0.0)
            occluded = depth_valid & (obj_depth > scene_depth + self._GS_MASK_DEPTH_EPS)
            visible = visible & ~occluded
            binary_mask_t |= visible

            # Heat-map: gather which (env_idx, channel_idx) pairs need this object
            for env_idx, env in enumerate(self.envs):
                operation_name = env._interest_object_operations.get(object_name)
                if operation_name is None:
                    continue
                channel_idx = heatmap_ops_index.get(operation_name)
                if channel_idx is None:
                    continue
                heat_map_t[env_idx, ..., channel_idx] |= visible[env_idx]

        # Single GPU→CPU transfer of final compact results
        binary_mask = binary_mask_t.to(torch.uint8).cpu().numpy()
        heat_map = heat_map_t.to(torch.uint8).cpu().numpy()
        return binary_mask, heat_map

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._fg_gs_renderer = None
        self._bg_gs_renderer = None
        self._gs_mask_renderers = {}
        self._bg_cache.clear()
        super().close()
