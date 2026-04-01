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

import numpy as np
import torch
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from gaussian_renderer import BatchSplatConfig, MjxBatchSplatRenderer, GSRendererMuJoCo
from auto_atom.basis.mjc.mujoco_env import (
    BatchedUnifiedMujocoEnv,
    EnvConfig,
    UnifiedMujocoEnv,
)


class GaussianRenderConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    body_gaussians: Dict[str, str] = Field(default_factory=dict)
    """Mapping from MuJoCo body name to PLY file path."""
    background_ply: str | None = None
    """Optional background PLY (loaded under the reserved key ``'background'``)."""


class GSEnvConfig(EnvConfig):
    """``EnvConfig`` extended with Gaussian Splatting rendering support.

    When ``gaussian_render`` is supplied the validator automatically sets
    ``enable_color=False`` on every camera so native RGB rendering is skipped.
    The names of cameras that originally had ``enable_color=True`` are stored
    in ``gs_color_cameras`` so ``GSUnifiedMujocoEnv`` knows which ones to
    render with GS.
    """

    gaussian_render: GaussianRenderConfig
    """Gaussian Splatting render config (required)."""
    gs_color_cameras: List[str] = Field(default_factory=list)
    """Names of cameras whose color output uses GS rendering. If empty, all cameras with ``enable_color=True`` are used."""
    gs_depth_cameras: List[str] = Field(default_factory=list)
    """Names of cameras whose depth output uses GS rendering. If empty, all cameras with ``enable_depth=True`` are used."""
    to_numpy: bool = True
    """Whether to convert GS renderer output to numpy arrays.
    Defaults to True for consistency with all other observation data types."""

    @model_validator(mode="after")
    def setup_gs_cameras(self):
        color_cams = [c.name for c in self.cameras if c.enable_color]
        gs_color_cameras = (
            color_cams if not self.gs_color_cameras else self.gs_color_cameras
        )
        if set(gs_color_cameras) - set(color_cams):
            raise ValueError(
                f"gs_color_cameras {gs_color_cameras} must be a subset of "
                f"cameras with enable_color=True: {color_cams}"
            )
        object.__setattr__(self, "gs_color_cameras", gs_color_cameras)

        depth_cams = [c.name for c in self.cameras if c.enable_depth]
        gs_depth_cameras = (
            depth_cams if not self.gs_depth_cameras else self.gs_depth_cameras
        )
        if set(gs_depth_cameras) - set(depth_cams):
            raise ValueError(
                f"gs_depth_cameras {gs_depth_cameras} must be a subset of "
                f"cameras with enable_depth=True: {depth_cams}"
            )
        object.__setattr__(self, "gs_depth_cameras", gs_depth_cameras)

        gs_color_set = set(gs_color_cameras)
        gs_depth_set = set(gs_depth_cameras)
        for cam in self.cameras:
            if cam.name in gs_color_set:
                object.__setattr__(cam, "enable_color", False)
            if cam.name in gs_depth_set:
                object.__setattr__(cam, "enable_depth", False)
        return self


class GSUnifiedMujocoEnv(UnifiedMujocoEnv):
    """``UnifiedMujocoEnv`` that replaces native RGB with Gaussian Splatting."""

    _GS_MASK_ALPHA_THRESHOLD = 0.5
    _GS_MASK_DEPTH_EPS = 0.01

    def __init__(self, config: GSEnvConfig) -> None:
        self.config: GSEnvConfig
        super().__init__(config)
        gs_cfg = config.gaussian_render
        combined_models = dict(gs_cfg.body_gaussians)
        if gs_cfg.background_ply:
            combined_models["background"] = gs_cfg.background_ply
        self._gs_renderer = GSRendererMuJoCo(combined_models, self.model)
        fg_cfg = BatchSplatConfig(
            body_gaussians=dict(gs_cfg.body_gaussians),
            background_ply=None,
            minibatch=512,
        )
        self._fg_gs_renderer = MjxBatchSplatRenderer(fg_cfg, self.model)
        self._bg_gs_renderer = (
            MjxBatchSplatRenderer(
                BatchSplatConfig(
                    body_gaussians={},
                    background_ply=gs_cfg.background_ply,
                    minibatch=512,
                ),
                self.model,
            )
            if gs_cfg.background_ply
            else None
        )
        if gs_cfg.background_ply:
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

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        obs = super().capture_observation()
        self._inject_gs_renders(obs)
        return obs

    def _inject_gs_renders(self, obs: dict[str, dict[str, Any]]) -> None:
        """Render GS color and/or depth and insert into *obs* in-place."""
        gs_color_set = set(self.config.gs_color_cameras)
        gs_depth_set = set(self.config.gs_depth_cameras)
        gs_mask_set = {
            cam_name
            for cam_name, spec in self._camera_specs.items()
            if spec.enable_mask or spec.enable_heat_map
        }
        all_gs_cams = [
            c
            for c in self._camera_specs
            if c in gs_color_set | gs_depth_set | gs_mask_set
        ]
        if not all_gs_cams:
            return

        structured: bool = self.config.structured
        t = int(self.data.time * 1e9) if self.config.stamp_ns else float(self.data.time)

        for cam_name in all_gs_cams:
            spec = self._camera_specs[cam_name]
            cam_id = self._camera_ids[cam_name]
            obs_cam_name = (
                "camera/" + cam_name.split("_")[0] if structured else cam_name
            )
            if cam_name in gs_color_set:
                rgb_t = self._render_gs_color_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                )
                rgb = torch.clamp(rgb_t, 0.0, 1.0).mul(255).to(torch.uint8)
                if self.config.to_numpy:
                    rgb = rgb.cpu().numpy()
                obs[f"{obs_cam_name}/color/image_raw"] = {"data": rgb, "t": t}
            depth_t: torch.Tensor | None = None
            scene_depth_t: torch.Tensor | None = None
            if cam_name in gs_depth_set or spec.enable_mask or spec.enable_heat_map:
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
                obs[f"{obs_cam_name}/aligned_depth_to_color/image_raw"] = {
                    "data": depth,
                    "t": t,
                }
            if (
                (spec.enable_mask or spec.enable_heat_map)
                and self._gs_mask_renderers
                and scene_depth_t is not None
            ):
                binary_mask, heat_map = self._render_gs_masks_for_camera(
                    cam_id=cam_id,
                    width=spec.width,
                    height=spec.height,
                    scene_depth_t=scene_depth_t,
                )
                if spec.enable_mask:
                    obs[f"{obs_cam_name}/mask/image_raw"] = {
                        "data": binary_mask,
                        "t": t,
                    }
                if spec.enable_heat_map:
                    obs[f"{obs_cam_name}/mask/heat_map"] = {
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

        # render_sig = inspect.signature(self._fg_gs_renderer.batch_env_render)
        # if "bg_depth" in render_sig.parameters:
        #     fg_rgb, fg_depth = self._fg_gs_renderer.batch_env_render(
        #         fg_gsb,
        #         cam_pos,
        #         cam_xmat,
        #         height,
        #         width,
        #         fovy,
        #         bg_imgs=bg_imgs,
        #         bg_depth=bg_depth,
        #     )
        #     full_rgb = fg_rgb
        #     full_depth = fg_depth
        # else:
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
        """Render GS masks with occlusion culling, following press_one_button logic."""
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
        scene_depth_np = scene_depth.detach().cpu().numpy()
        scene_depth_np = np.nan_to_num(scene_depth_np, nan=0.0, posinf=0.0, neginf=0.0)

        binary_mask = np.zeros((height, width), dtype=np.uint8)
        heat_map = np.zeros(
            (height, width, len(self.config.operations)), dtype=np.uint8
        )

        for object_name, renderer in self._gs_mask_renderers.items():
            gsb = renderer.batch_update_gaussians(body_pos, body_quat)
            alpha_t, obj_depth_t = renderer.batch_env_render(
                gsb, cam_pos, cam_xmat, height, width, fovy
            )

            alpha_np = alpha_t[0, 0].detach().cpu().numpy().max(axis=-1)
            obj_depth_np = obj_depth_t[0, 0, :, :, 0].detach().cpu().numpy()
            obj_depth_np = np.nan_to_num(obj_depth_np, nan=0.0, posinf=0.0, neginf=0.0)
            visible_np = alpha_np > self._GS_MASK_ALPHA_THRESHOLD
            depth_valid_np = (scene_depth_np > 0.0) & (obj_depth_np > 0.0)
            occluded_np = depth_valid_np & (
                obj_depth_np > scene_depth_np + self._GS_MASK_DEPTH_EPS
            )
            visible_np[occluded_np] = False
            binary_mask[visible_np] = 1

            operation_name = self._interest_object_operations.get(object_name)
            if operation_name is None:
                continue
            if operation_name not in self.config.operations:
                continue
            channel_idx = self.config.operations.index(operation_name)
            heat_map[visible_np, channel_idx] = 1

        return binary_mask, heat_map

    def close(self) -> None:
        self._gs_renderer = None
        self._fg_gs_renderer = None
        self._bg_gs_renderer = None
        self._gs_mask_renderers = {}
        super().close()


class BatchedGSUnifiedMujocoEnv(BatchedUnifiedMujocoEnv):
    """Aggregate multiple homogeneous ``GSUnifiedMujocoEnv`` replicas."""

    def __init__(self, config: Optional[GSEnvConfig] = None, **kwargs) -> None:
        if config is None:
            config = GSEnvConfig.model_validate(kwargs)
        self.config = config
        self.batch_size = int(config.batch_size)
        self.envs: list[GSUnifiedMujocoEnv] = []
        for env_index in range(self.batch_size):
            viewer = config.viewer if env_index == config.viewer_env_index else None
            env_cfg = config.model_copy(update={"batch_size": 1, "viewer": viewer})
            self.envs.append(GSUnifiedMujocoEnv(env_cfg))
        if config.name:
            from auto_atom.runtime import ComponentRegistry

            ComponentRegistry.register_env(config.name, self)
