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
  - Depth, mask, and heat-map outputs are unaffected.
"""

from __future__ import annotations
import torch
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field, model_validator
from gaussian_renderer import GSRendererMuJoCo
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
        gs_color_set = set(gs_color_cameras)
        for cam in self.cameras:
            if cam.name in gs_color_set:
                object.__setattr__(cam, "enable_color", False)
        return self


class GSUnifiedMujocoEnv(UnifiedMujocoEnv):
    """``UnifiedMujocoEnv`` that replaces native RGB with Gaussian Splatting."""

    def __init__(self, config: GSEnvConfig) -> None:
        self.config: GSEnvConfig
        super().__init__(config)
        gs_cfg = config.gaussian_render
        models_dict = dict(gs_cfg.body_gaussians)
        if gs_cfg.background_ply:
            models_dict["background"] = gs_cfg.background_ply
        self._gs_renderer = GSRendererMuJoCo(models_dict, self.model)
        self.get_logger().debug(
            f"GS renderer initialised with {len(gs_cfg.body_gaussians)} body gaussian(s)"
            + (" + background" if gs_cfg.background_ply else "")
        )

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        obs = super().capture_observation()
        self._inject_gs_color(obs)
        return obs

    def _inject_gs_color(self, obs: dict[str, dict[str, Any]]) -> None:
        """Render GS color and insert into *obs* in-place."""
        gs_color_cameras = self.config.gs_color_cameras
        if not gs_color_cameras:
            return

        self._gs_renderer.update_gaussians(self.data)
        structured: bool = self.config.structured
        t = int(self.data.time * 1e9) if self.config.stamp_ns else float(self.data.time)

        for cam_name in gs_color_cameras:
            spec = self._camera_specs[cam_name]
            cam_id = self._camera_ids[cam_name]
            result = self._gs_renderer.render(
                self.model, self.data, [cam_id], spec.width, spec.height
            )
            if cam_id not in result:
                raise RuntimeError(
                    f"GS renderer did not return output for camera '{cam_name}' (ID {cam_id})"
                )
            rgb_t, _ = result[cam_id]
            rgb = torch.clamp(rgb_t, 0.0, 1.0).mul(255).to(torch.uint8)
            if self.config.to_numpy:
                rgb = rgb.cpu().numpy()
            obs_cam_name = (
                "camera/" + cam_name.split("_")[0] if structured else cam_name
            )
            obs[f"{obs_cam_name}/color/image_raw"] = {"data": rgb, "t": t}

    def close(self) -> None:
        self._gs_renderer = None
        super().close()


class BatchedGSUnifiedMujocoEnv(BatchedUnifiedMujocoEnv):
    """Aggregate multiple homogeneous ``GSUnifiedMujocoEnv`` replicas."""

    def __init__(self, config: GSEnvConfig):
        self.config = config
        self.batch_size = int(config.batch_size)
        self.envs: list[GSUnifiedMujocoEnv] = []
        for env_index in range(self.batch_size):
            viewer = config.viewer if env_index == config.viewer_env_index else None
            env_cfg = config.model_copy(update={"batch_size": 1, "viewer": viewer})
            self.envs.append(GSUnifiedMujocoEnv(env_cfg))
