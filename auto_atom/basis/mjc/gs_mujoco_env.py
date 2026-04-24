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

import glob as _glob
import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import torch
from gaussian_renderer import BatchSplatConfig, GSRendererMuJoCo, MjxBatchSplatRenderer
from gaussian_renderer.core.util_gau import load_ply, save_ply
from natsort import natsorted
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auto_atom.basis.mjc.mujoco_env import (
    BatchedUnifiedMujocoEnv,
    EnvConfig,
    UnifiedMujocoEnv,
    create_image_data,
)

_GLOB_META = ("*", "?", "[")


def _has_glob(pattern: str) -> bool:
    return any(c in pattern for c in _GLOB_META)


def _expand_background_entry(entry: str) -> list[str]:
    if not _has_glob(entry):
        return [entry]
    matches = _glob.glob(entry)
    if not matches:
        raise FileNotFoundError(f"background_ply glob matched no files: {entry}")
    return list(natsorted(matches))


def create_image_data_batch(
    image_batch, timestamps, frame_id: str = "", tobytes: bool = True
):
    return [
        create_image_data(image, time_sec, frame_id, tobytes)
        for image, time_sec in zip(image_batch, timestamps / 1e9)
    ]


BackgroundPose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
"""(position_xyz, orientation_xyzw) pair describing a background transform."""

_IDENTITY_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


def _normalize_background_pose(value: Any) -> BackgroundPose:
    """Normalize a 3-element (xyz) or 7-element (xyz + xyzw) sequence into a
    ``BackgroundPose``.

    Length 3 → pure translation with identity orientation.
    Length 7 → ``[x, y, z, qx, qy, qz, qw]``.
    A ``BackgroundPose`` tuple ``((x, y, z), (qx, qy, qz, qw))`` is passed
    through unchanged.
    """
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], tuple)
        and len(value[0]) == 3
        and isinstance(value[1], tuple)
        and len(value[1]) == 4
    ):
        return value  # already a BackgroundPose
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.shape[0] == 3:
        return tuple(float(v) for v in arr), _IDENTITY_QUAT
    if arr.shape[0] == 7:
        pos = tuple(float(v) for v in arr[:3])
        quat = arr[3:]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-12:
            quat_t = _IDENTITY_QUAT
        else:
            quat_t = tuple(float(v) for v in (quat / norm))
        return pos, quat_t
    raise ValueError(
        f"background transform must be length 3 (xyz) or 7 (xyz+xyzw), "
        f"got {arr.shape[0]}"
    )


def _is_identity_pose(pose: BackgroundPose) -> bool:
    pos, quat = pose
    return np.allclose(pos, 0.0) and np.allclose(quat, _IDENTITY_QUAT)


def _sample_env_background_indices(
    batch_size: int,
    num_backgrounds: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign one background index per env.

    When enough backgrounds are available, sample without replacement so each
    environment receives a distinct background. Otherwise fall back to sampling
    with replacement because duplicates are unavoidable.
    """
    if batch_size <= 0:
        return np.zeros(0, dtype=np.int64)
    if num_backgrounds <= 0:
        return np.zeros(batch_size, dtype=np.int64)
    if batch_size <= num_backgrounds:
        return np.asarray(rng.permutation(num_backgrounds)[:batch_size], dtype=np.int64)
    return rng.integers(0, num_backgrounds, size=batch_size, dtype=np.int64)


def _resolve_background_transform(
    background_ply: str | None,
    background_transform: Any | None,
    background_transforms: Dict[str, Any],
) -> BackgroundPose:
    if background_transform is not None:
        return _normalize_background_pose(background_transform)
    if not background_ply:
        return (0.0, 0.0, 0.0), _IDENTITY_QUAT

    bg_path = Path(background_ply)
    for key in (background_ply, str(bg_path), bg_path.name, bg_path.stem):
        if key in background_transforms:
            return _normalize_background_pose(background_transforms[key])
    return (0.0, 0.0, 0.0), _IDENTITY_QUAT


class BodyMirrorSpec(BaseModel):
    """Reflect a per-body Gaussian PLY across a plane at load time, with
    an optional post-reflection rigid transform.

    The reflection plane's normal is specified either directly in the PLY's
    local (GS) coordinates via ``axis``, or derived from a MuJoCo body
    quaternion plus a body-frame axis.  Useful when the PLY is pre-rotated
    (body quat = identity) and you can reason about left/right directly in
    the PLY frame.

    When ``position`` / ``orientation`` are set, a rigid transform is
    applied *after* the reflection, giving a single entry that can express
    rotoreflections (mirror ∘ rotate — which is not representable as a
    single reflection in general).  The reflection center doubles as the
    rotation pivot unless the body-wide ``center`` is otherwise specified.
    """

    model_config = ConfigDict(extra="forbid")

    axis: list | None = None
    """Unit vector in the PLY's local (GS) coords. Plane normal."""
    body_quat: list | None = None
    """MuJoCo-convention body quaternion (wxyz). Used with ``body_axis``
    when ``axis`` is not given: ``gs_axis = R(body_quat)^T @ body_axis``."""
    body_axis: list = Field(default_factory=lambda: [1.0, 0.0, 0.0])
    """Body-frame direction to mirror along (only read when
    ``body_quat`` is set)."""
    center: list | None = None
    """Explicit mirror-plane center in GS coords. When omitted, the PLY
    centroid is used. Also doubles as the rotation pivot for the
    optional post-reflection transform."""
    share_center_with: str | None = None
    """Name of another entry in ``body_gaussians``; reuse that body's center
    as this body's mirror center.  The target's **explicit** ``center`` is
    used when set; otherwise the target PLY's centroid.  Use this to keep
    paired objects (e.g. door + knob) aligned after mirroring (and, with
    ``position``/``orientation`` set, rotating around a shared pivot).
    Ignored when ``center`` is given on this spec."""
    position: list | None = None
    """Optional post-reflection translation ``[x, y, z]`` in PLY-local
    (GS) coords. Applied after the reflection."""
    orientation: list | None = None
    """Optional post-reflection rotation. Either a quaternion
    ``[x, y, z, w]`` (length 4) or Euler ``[roll, pitch, yaw]`` radians
    (length 3). Applied in PLY-local coords about the mirror's ``center``
    (or PLY centroid when no center is given)."""

    def resolved_post_pose(self) -> BackgroundPose:
        """Resolve the optional post-reflection rigid transform to a
        ``(position, quat_xyzw)`` tuple. Identity when neither
        ``position`` nor ``orientation`` is set."""
        pos = (0.0, 0.0, 0.0)
        quat = _IDENTITY_QUAT
        if self.position is not None:
            arr = np.asarray(self.position, dtype=np.float64).ravel()
            if arr.shape != (3,):
                raise ValueError(
                    f"BodyMirrorSpec.position must be length 3, got {arr.shape}"
                )
            pos = tuple(float(v) for v in arr)
        if self.orientation is not None:
            arr = np.asarray(self.orientation, dtype=np.float64).ravel()
            if arr.shape == (3,):
                from scipy.spatial.transform import Rotation

                quat = tuple(
                    float(v) for v in Rotation.from_euler("xyz", arr).as_quat()
                )
            elif arr.shape == (4,):
                norm = float(np.linalg.norm(arr))
                quat = (
                    _IDENTITY_QUAT
                    if norm < 1e-12
                    else tuple(float(v) for v in (arr / norm))
                )
            else:
                raise ValueError(
                    "BodyMirrorSpec.orientation must be length 3 (Euler) "
                    f"or 4 (quaternion), got {arr.shape}"
                )
        return pos, quat

    def resolved_axis(self) -> np.ndarray:
        if self.axis is not None:
            a = np.asarray(self.axis, dtype=np.float64).ravel()
            if a.shape != (3,):
                raise ValueError(f"BodyMirrorSpec.axis must be length 3, got {a.shape}")
        elif self.body_quat is not None:
            from scipy.spatial.transform import Rotation

            q = np.asarray(self.body_quat, dtype=np.float64).ravel()
            if q.shape != (4,):
                raise ValueError(
                    f"BodyMirrorSpec.body_quat must be length 4 (wxyz), got {q.shape}"
                )
            quat_xyzw = q[[1, 2, 3, 0]]
            R = Rotation.from_quat(quat_xyzw).as_matrix()
            ba = np.asarray(self.body_axis, dtype=np.float64).ravel()
            if ba.shape != (3,):
                raise ValueError(
                    f"BodyMirrorSpec.body_axis must be length 3, got {ba.shape}"
                )
            a = R.T @ ba
        else:
            raise ValueError(
                "BodyMirrorSpec requires either 'axis' or 'body_quat' to be set"
            )
        n = float(np.linalg.norm(a))
        if n < 1e-12:
            raise ValueError("BodyMirrorSpec axis resolves to zero vector")
        return a / n


class BodyTransformSpec(BaseModel):
    """Rigid transform baked into a per-body Gaussian PLY at load time."""

    model_config = ConfigDict(extra="forbid")

    position: list | None = None
    """Translation [x, y, z] applied in the PLY's local GS coords."""
    orientation: list | None = None
    """Quaternion [x, y, z, w] or Euler [roll, pitch, yaw] in radians."""
    center: list | None = None
    """Optional pivot point [x, y, z] in the PLY's local GS coords."""
    share_center_with: str | None = None
    """Reuse another body's pivot point.  The target's **explicit**
    ``center`` is used when set; otherwise the target PLY's centroid."""

    def resolved_pose(self) -> BackgroundPose:
        pos = (0.0, 0.0, 0.0)
        quat = _IDENTITY_QUAT
        if self.position is not None:
            arr = np.asarray(self.position, dtype=np.float64).ravel()
            if arr.shape != (3,):
                raise ValueError(
                    f"BodyTransformSpec.position must be length 3, got {arr.shape}"
                )
            pos = tuple(float(v) for v in arr)
        if self.orientation is not None:
            arr = np.asarray(self.orientation, dtype=np.float64).ravel()
            if arr.shape == (3,):
                from scipy.spatial.transform import Rotation

                quat = tuple(
                    float(v) for v in Rotation.from_euler("xyz", arr).as_quat()
                )
            elif arr.shape == (4,):
                norm = float(np.linalg.norm(arr))
                quat = (
                    _IDENTITY_QUAT
                    if norm < 1e-12
                    else tuple(float(v) for v in (arr / norm))
                )
            else:
                raise ValueError(
                    "BodyTransformSpec.orientation must be length 3 (Euler) "
                    f"or 4 (quaternion), got {arr.shape}"
                )
        return pos, quat


def _mirror_gaussians_inplace(gaussians, axis: np.ndarray, center: np.ndarray) -> None:
    """Reflect positions, rotations, and SH band-1 across the plane through
    ``center`` perpendicular to ``axis`` (unit, GS-local).

    Mirrors ``third_party/mirror_door_plys.py``: positions flipped via
    Householder projection; quaternions via ``M @ R @ M`` where
    ``M = I - 2 a aᵀ``; SH DC invariant; SH band-1 reflected per channel.
    Higher SH bands are left as-is (perturbation dominated by SH noise).
    """
    from scipy.spatial.transform import Rotation

    ax = axis.astype(np.float64)
    ctr = center.astype(np.float64)

    dp = gaussians.xyz - ctr
    proj = (dp @ ax)[:, None] * ax[None, :]
    gaussians.xyz = (ctr + dp - 2.0 * proj).astype(gaussians.xyz.dtype)

    M = np.eye(3) - 2.0 * np.outer(ax, ax)

    rot_wxyz = gaussians.rot
    rot_xyzw = rot_wxyz[:, [1, 2, 3, 0]]
    R_orig = Rotation.from_quat(rot_xyzw).as_matrix()
    R_mirror = np.einsum("ij,njk,kl->nil", M, R_orig, M)
    rot_mirror_xyzw = Rotation.from_matrix(R_mirror).as_quat()
    gaussians.rot = rot_mirror_xyzw[:, [3, 0, 1, 2]].astype(rot_wxyz.dtype)

    sh = gaussians.sh
    if sh.ndim == 3 and sh.shape[1] > 3:
        band1 = sh[:, 1:4, :].copy()
        for ch in range(3):
            xyz = np.stack([band1[:, 2, ch], band1[:, 0, ch], band1[:, 1, ch]], axis=-1)
            xyz_m = (M @ xyz.T).T
            sh[:, 1, ch] = xyz_m[:, 1]
            sh[:, 2, ch] = xyz_m[:, 2]
            sh[:, 3, ch] = xyz_m[:, 0]
        gaussians.sh = sh


def _materialize_mirrored_body_ply(
    src_ply: str,
    axis: np.ndarray,
    center: np.ndarray,
    post_pose: BackgroundPose | None = None,
) -> str:
    """Return a path to a (possibly cached) mirrored PLY, optionally with
    a rigid transform baked in *after* the reflection.

    When ``post_pose`` is non-identity, a rigid-body transform
    ``p' = R @ (p - center) + center + t`` is applied in place once the
    reflection is done, using the mirror's ``center`` as the rotation
    pivot. This expresses rotoreflections (mirror ∘ rotate) in one step.

    Follows the same cache pattern as
    ``_materialize_transformed_background_ply``.
    """
    src_path = Path(src_ply).expanduser().resolve()
    post_identity = post_pose is None or _is_identity_pose(
        _normalize_background_pose(post_pose)
    )
    post_key = "none"
    if not post_identity:
        pos, quat = _normalize_background_pose(post_pose)
        post_key = (
            f"{pos[0]:.9f},{pos[1]:.9f},{pos[2]:.9f}"
            f"|{quat[0]:.9f},{quat[1]:.9f},{quat[2]:.9f},{quat[3]:.9f}"
        )
    cache_key = hashlib.sha1(
        (
            f"{src_path}"
            f"|{axis[0]:.9f},{axis[1]:.9f},{axis[2]:.9f}"
            f"|{center[0]:.9f},{center[1]:.9f},{center[2]:.9f}"
            f"|{post_key}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = Path(".cache/gs_body_mirrors")
    cache_path = cache_dir / f"{src_path.stem}__mirror_{cache_key}.ply"
    if cache_path.exists():
        return str(cache_path)

    gaussians = load_ply(str(src_path))
    _mirror_gaussians_inplace(gaussians, axis, center)
    if not post_identity:
        pos, quat = _normalize_background_pose(post_pose)
        from scipy.spatial.transform import Rotation

        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(pos, dtype=np.float64)
        Tc = np.eye(4, dtype=np.float64)
        Tc[:3, 3] = center.astype(np.float64)
        Tnc = np.eye(4, dtype=np.float64)
        Tnc[:3, 3] = -center.astype(np.float64)
        T = Tc @ T @ Tnc

        from gaussian_renderer.transform_gs_model import transform_gaussian

        transform_gaussian(gaussians, T, silent=True)
    save_ply(gaussians, cache_path)
    return str(cache_path)


def _materialize_transformed_body_ply(
    src_ply: str,
    pose: BackgroundPose,
    center: np.ndarray | None = None,
) -> str:
    """Return a path to a (possibly cached) body PLY with *pose* baked in.

    When *center* is provided, rotation is applied about that pivot:
    ``p' = R @ (p - center) + center + t``.
    """
    pose = _normalize_background_pose(pose)
    if _is_identity_pose(pose):
        return src_ply

    pos, quat = pose
    src_path = Path(src_ply).expanduser().resolve()
    center_key = (
        "origin"
        if center is None
        else f"{center[0]:.6f},{center[1]:.6f},{center[2]:.6f}"
    )
    cache_key = hashlib.sha1(
        (
            f"{src_path}"
            f"|{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}"
            f"|{quat[0]:.6f},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f}"
            f"|{center_key}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = Path(".cache/gs_body_transforms")
    cache_path = cache_dir / f"{src_path.stem}__body_xform_{cache_key}.ply"
    if cache_path.exists():
        return str(cache_path)

    gaussians = load_ply(str(src_path))
    is_identity_rot = np.allclose(quat, _IDENTITY_QUAT)
    if is_identity_rot:
        gaussians.xyz = gaussians.xyz + np.asarray(pos, dtype=np.float32)
    else:
        from scipy.spatial.transform import Rotation

        R = Rotation.from_quat(quat).as_matrix()
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = np.asarray(pos, dtype=np.float64)
        if center is not None:
            Tc = np.eye(4, dtype=np.float64)
            Tc[:3, 3] = np.asarray(center, dtype=np.float64)
            Tnc = np.eye(4, dtype=np.float64)
            Tnc[:3, 3] = -np.asarray(center, dtype=np.float64)
            T = Tc @ T @ Tnc

        from gaussian_renderer.transform_gs_model import transform_gaussian

        transform_gaussian(gaussians, T, silent=True)

    save_ply(gaussians, cache_path)
    return str(cache_path)


def _materialize_transformed_background_ply(
    background_ply: str | None,
    pose: BackgroundPose,
) -> str | None:
    """Return a path to a (possibly cached) PLY with *pose* baked in.

    *pose* is ``((x, y, z), (qx, qy, qz, qw))``.  When the orientation is
    identity only a translation is applied; otherwise the full rigid-body
    transform (position, rotation, SH rotation) is applied via
    ``gaussian_renderer.transform_gs_model.transform_gaussian``.
    """
    if background_ply is None:
        return None

    pose = _normalize_background_pose(pose)
    if _is_identity_pose(pose):
        return background_ply

    pos, quat = pose
    src_path = Path(background_ply).expanduser().resolve()
    cache_key = hashlib.sha1(
        (
            f"{src_path}"
            f"|{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f}"
            f"|{quat[0]:.6f},{quat[1]:.6f},{quat[2]:.6f},{quat[3]:.6f}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_dir = Path(".cache/gs_background_transforms")
    cache_path = cache_dir / f"{src_path.stem}__bg_xform_{cache_key}.ply"
    if cache_path.exists():
        return str(cache_path)

    gaussians = load_ply(str(src_path))

    is_identity_rot = np.allclose(quat, _IDENTITY_QUAT)
    if is_identity_rot:
        # Pure translation — fast path, no rotation needed.
        gaussians.xyz = gaussians.xyz + np.asarray(pos, dtype=np.float32)
    else:
        # Full rigid-body transform via gaussian_renderer.
        from scipy.spatial.transform import Rotation

        R = Rotation.from_quat(quat).as_matrix()  # xyzw convention
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = pos

        from gaussian_renderer.transform_gs_model import transform_gaussian

        transform_gaussian(gaussians, T, silent=True)

    save_ply(gaussians, cache_path)
    return str(cache_path)


class GaussianRenderConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    body_gaussians: Dict[str, str] = Field(default_factory=dict)
    """Mapping from MuJoCo body name to PLY file path."""
    background_ply: str | list[str] | None = None
    """Optional background PLY, or a list of PLYs.
    When a list is given, GS envs randomly assign one background initially and
    optionally reassign backgrounds on each ``reset``."""
    randomize_background_on_reset: bool = False
    """Whether to reassign list-valued backgrounds on every ``reset``.
    When ``False``, multi-background envs keep the initial random assignment
    across resets."""
    background_transform: list | tuple | None = None
    """Default pose transform for the background PLY.
    Length 3 → ``[x, y, z]`` (pure translation).
    Length 7 → ``[x, y, z, qx, qy, qz, qw]`` (full rigid-body transform).
    When ``background_ply`` is a list, this transform is applied to *every*
    entry (useful when all backgrounds share a common capture frame);
    individual entries can still be overridden via ``background_transforms``
    keyed by path/name/stem."""
    background_transforms: Dict[str, list] = Field(default_factory=dict)
    """Per-background pose transforms keyed by full path, file name, or stem.
    Values: ``[x, y, z]`` or ``[x, y, z, qx, qy, qz, qw]``."""
    body_transforms: Dict[str, BodyTransformSpec] = Field(default_factory=dict)
    """Per-body rigid transforms applied to ``body_gaussians`` PLYs at load time.
    Keys must match entries in ``body_gaussians``. Transformed PLYs are cached
    under ``.cache/gs_body_transforms/`` keyed by (path, pose, center)."""
    body_mirrors: Dict[str, BodyMirrorSpec] = Field(default_factory=dict)
    """Per-body reflections applied to ``body_gaussians`` PLYs at load time.
    Keys must match entries in ``body_gaussians``. Mirrored PLYs are cached
    under ``.cache/gs_body_mirrors/`` keyed by (path, axis, center)."""
    minibatch: int = 512
    """Gaussian splat renderer minibatch size. Controls how many gaussians are
    processed per kernel launch; larger values use more VRAM. Passed through to
    every ``BatchSplatConfig`` built by the GS env classes."""
    share_physics: bool = False
    """In batched GS envs, share a single physics replica and foreground GS
    render across the whole batch; only the composited background differs per
    env. Requires ``background_ply`` to be multi-valued (list/glob) and
    ``batch_size > 1``. Validated on ``GSEnvConfig``."""

    def resolved_body_gaussians(self) -> Dict[str, str]:
        """Return ``body_gaussians`` with any configured transforms / mirrors
        substituted by the corresponding cached PLY paths."""
        if not self.body_transforms and not self.body_mirrors:
            return dict(self.body_gaussians)

        unknown = (set(self.body_transforms) | set(self.body_mirrors)) - set(
            self.body_gaussians
        )
        if unknown:
            raise ValueError(
                f"body_transforms/body_mirrors reference unknown body(s): {sorted(unknown)}. "
                f"Keys must appear in body_gaussians: {sorted(self.body_gaussians)}"
            )

        centroids: Dict[tuple[str, str], np.ndarray] = {}

        def _centroid(paths: Dict[str, str], body_name: str) -> np.ndarray:
            key = (body_name, paths[body_name])
            if key not in centroids:
                centroids[key] = (
                    load_ply(str(paths[body_name])).xyz.mean(axis=0).astype(np.float64)
                )
            return centroids[key]

        def _explicit_center(spec, label: str, body_name: str) -> np.ndarray:
            arr = np.asarray(spec.center, dtype=np.float64).ravel()
            if arr.shape != (3,):
                raise ValueError(f"{label}['{body_name}'].center must be length 3")
            return arr

        def _share_center(
            target: str,
            paths: Dict[str, str],
            same_spec_dict: Dict[str, object],
            label: str,
        ) -> np.ndarray:
            """Resolve a ``share_center_with`` reference. Prefer the target's
            explicit ``center`` field (from the same spec dict as the caller),
            falling back to the target PLY's centroid when not set."""
            target_spec = same_spec_dict.get(target)
            if target_spec is not None and target_spec.center is not None:
                return _explicit_center(target_spec, label, target)
            return _centroid(paths, target)

        transformed: Dict[str, str] = {}
        for body_name, src_ply in self.body_gaussians.items():
            spec = self.body_transforms.get(body_name)
            if spec is None:
                transformed[body_name] = src_ply
                continue
            pose = spec.resolved_pose()
            center = None
            if spec.center is not None:
                center = _explicit_center(spec, "body_transforms", body_name)
            elif spec.share_center_with is not None:
                if spec.share_center_with not in self.body_gaussians:
                    raise ValueError(
                        f"body_transforms['{body_name}'].share_center_with="
                        f"'{spec.share_center_with}' is not in body_gaussians"
                    )
                center = _share_center(
                    spec.share_center_with,
                    self.body_gaussians,
                    self.body_transforms,
                    "body_transforms",
                )
            transformed[body_name] = _materialize_transformed_body_ply(
                src_ply, pose, center
            )

        if not self.body_mirrors:
            return transformed

        resolved: Dict[str, str] = {}
        for body_name, src_ply in transformed.items():
            spec = self.body_mirrors.get(body_name)
            if spec is None:
                resolved[body_name] = src_ply
                continue
            axis = spec.resolved_axis()
            if spec.center is not None:
                center = _explicit_center(spec, "body_mirrors", body_name)
            elif spec.share_center_with is not None:
                if spec.share_center_with not in self.body_gaussians:
                    raise ValueError(
                        f"body_mirrors['{body_name}'].share_center_with="
                        f"'{spec.share_center_with}' is not in body_gaussians"
                    )
                center = _share_center(
                    spec.share_center_with,
                    transformed,
                    self.body_mirrors,
                    "body_mirrors",
                )
            else:
                center = _centroid(transformed, body_name)
            post_pose = spec.resolved_post_pose()
            resolved[body_name] = _materialize_mirrored_body_ply(
                src_ply, axis, center, post_pose
            )
        return resolved

    def _background_ply_list(self) -> list[str]:
        """Return ``background_ply`` as a list (empty when unset).

        Entries may contain glob patterns (``*``, ``?``, ``[...]``); matches
        are expanded with natural sort order.
        """
        if self.background_ply is None:
            return []
        entries = (
            [self.background_ply]
            if isinstance(self.background_ply, str)
            else list(self.background_ply)
        )
        out: list[str] = []
        for entry in entries:
            out.extend(_expand_background_entry(entry))
        return out

    def is_multi_background(self) -> bool:
        if isinstance(self.background_ply, (list, tuple)):
            return True
        if isinstance(self.background_ply, str) and _has_glob(self.background_ply):
            return True
        return False

    def resolved_background_transform(self) -> BackgroundPose:
        """Resolve the singular pose transform for a single-path background.

        For list-valued ``background_ply``, use ``resolved_background_plys``
        which applies the singular ``background_transform`` as the default for
        every entry plus any per-entry overrides from ``background_transforms``.
        """
        if self.is_multi_background():
            # Callers for list backgrounds should go through
            # ``resolved_background_plys``. Return identity here so the outer
            # "store pose" step in the single-bg flow is a no-op.
            return (0.0, 0.0, 0.0), _IDENTITY_QUAT
        return _resolve_background_transform(
            self.background_ply,
            self.background_transform,
            self.background_transforms,
        )

    def resolved_background_ply(self) -> str | None:
        """Return a materialized single background path.

        Raises when ``background_ply`` is a list; callers should use
        ``resolved_background_plys`` in that case.
        """
        if self.is_multi_background():
            raise ValueError(
                "background_ply is a list; call resolved_background_plys() instead."
            )
        return _materialize_transformed_background_ply(
            self.background_ply,
            self.resolved_background_transform(),
        )

    def resolved_background_plys(self) -> list[str]:
        """Return materialized bg paths for each entry in ``background_ply``.

        Pose resolution per entry (in order of precedence):

        1. ``background_transforms`` entry keyed by full path / file name /
           stem (per-entry override).
        2. The singular ``background_transform`` when set (applied as the
           default to every entry — e.g. a shared world pose for a pool of
           backgrounds captured in the same frame).
        3. Identity.
        """
        default_pose: BackgroundPose | None = None
        if self.background_transform is not None:
            default_pose = _normalize_background_pose(self.background_transform)

        out: list[str] = []
        for bg_ply in self._background_ply_list():
            bg_path = Path(bg_ply)
            pose: BackgroundPose | None = None
            for key in (bg_ply, str(bg_path), bg_path.name, bg_path.stem):
                if key in self.background_transforms:
                    pose = _normalize_background_pose(self.background_transforms[key])
                    break
            if pose is None:
                pose = (
                    default_pose
                    if default_pose is not None
                    else (
                        (0.0, 0.0, 0.0),
                        _IDENTITY_QUAT,
                    )
                )
            materialized = _materialize_transformed_background_ply(bg_ply, pose)
            if materialized is not None:
                out.append(materialized)
        return out


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

        if self.gaussian_render.share_physics:
            if self.batch_size <= 1:
                raise ValueError(
                    "gaussian_render.share_physics requires batch_size > 1; "
                    f"got batch_size={self.batch_size}"
                )
            if not self.gaussian_render.is_multi_background():
                raise ValueError(
                    "gaussian_render.share_physics requires background_ply to "
                    "be multi-valued (list or glob). A single background would "
                    "produce identical observations across the batch."
                )
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
        self._is_multi_bg = gs_cfg.is_multi_background()
        self._gs_body_gaussians = gs_cfg.resolved_body_gaussians()
        self._gs_renderer: GSRendererMuJoCo | None = None
        fg_cfg = BatchSplatConfig(
            body_gaussians=dict(self._gs_body_gaussians),
            background_ply=None,
            minibatch=self.config.gaussian_render.minibatch,
        )
        self._fg_gs_renderer = MjxBatchSplatRenderer(fg_cfg, self.model)
        self._bg_gs_renderer: MjxBatchSplatRenderer | None = None

        # Multi-background state: parallel lists of per-bg renderers, plus the
        # index of the currently active bg. ``_gs_renderer`` / ``_bg_gs_renderer``
        # are re-pointed into these lists on reset.
        self._gs_renderers_list: list[GSRendererMuJoCo] = []
        self._bg_gs_renderers_list: list[MjxBatchSplatRenderer | None] = []
        self._bg_source_plys: list[str] = []
        self._active_bg_idx: int = 0
        self._bg_rng = np.random.default_rng()

        if self._is_multi_bg:
            self._bg_source_plys = gs_cfg.resolved_background_plys()
            for bg_ply in self._bg_source_plys:
                self._gs_renderers_list.append(self._make_combined_gs_renderer(bg_ply))
                self._bg_gs_renderers_list.append(self._make_bg_renderer(bg_ply))
            self._randomize_active_bg()
            reset_mode = (
                "random pick per reset"
                if gs_cfg.randomize_background_on_reset
                else "fixed after initial pick"
            )
            self.get_logger().debug(
                f"GS renderer initialised with {len(self._gs_body_gaussians)} "
                f"body gaussian(s) + {len(self._bg_source_plys)} backgrounds "
                f"({reset_mode})"
            )
        else:
            self.set_background_transform(gs_cfg.resolved_background_transform())
            background_ply = gs_cfg.resolved_background_ply()
            if background_ply:
                self.get_logger().debug(
                    f"GS renderer initialised with {len(self._gs_body_gaussians)} body gaussian(s) + background"
                )
            else:
                self.get_logger().debug(
                    f"GS renderer initialised with {len(self._gs_body_gaussians)} body gaussian(s)"
                )
        self._gs_mask_renderers = self._build_gs_mask_renderers(
            dict(self._gs_body_gaussians)
        )

    def _make_combined_gs_renderer(
        self, background_ply: str | None
    ) -> GSRendererMuJoCo:
        combined = dict(self._gs_body_gaussians)
        if background_ply:
            combined["background"] = background_ply
        return GSRendererMuJoCo(combined, self.model)

    def _make_bg_renderer(
        self, background_ply: str | None
    ) -> MjxBatchSplatRenderer | None:
        if not background_ply:
            return None
        return MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians={},
                background_ply=background_ply,
                minibatch=self.config.gaussian_render.minibatch,
            ),
            self.model,
        )

    def _randomize_active_bg(self) -> int:
        if not self._gs_renderers_list:
            return self._active_bg_idx
        self._active_bg_idx = int(
            self._bg_rng.integers(0, len(self._gs_renderers_list))
        )
        self._gs_renderer = self._gs_renderers_list[self._active_bg_idx]
        self._bg_gs_renderer = self._bg_gs_renderers_list[self._active_bg_idx]
        return self._active_bg_idx

    def reset(self) -> None:
        super().reset()
        if (
            self._is_multi_bg
            and self.config.gaussian_render.randomize_background_on_reset
        ):
            self._randomize_active_bg()

    def set_background_transform(
        self, pose: BackgroundPose | list[float]
    ) -> BackgroundPose:
        if self._is_multi_bg:
            raise ValueError(
                "set_background_transform is not supported when background_ply "
                "is a list; use background_transforms in the config to set "
                "per-background poses."
            )
        pose = _normalize_background_pose(pose)
        background_ply = _materialize_transformed_background_ply(
            self._gs_background_source_ply,
            pose,
        )
        self._gs_renderer = self._make_combined_gs_renderer(background_ply)
        self._bg_gs_renderer = self._make_bg_renderer(background_ply)
        object.__setattr__(self.config.gaussian_render, "background_transform", pose)
        return pose

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
                minibatch=self.config.gaussian_render.minibatch,
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
        self._gs_renderers_list = []
        self._bg_gs_renderers_list = []
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
    # Class-level defaults so that tests which bypass ``__init__`` via
    # ``object.__new__(BatchedGSUnifiedMujocoEnv)`` still see sensible values
    # for the shared-physics attrs.
    _share_physics: bool = False
    _virtual_batch_size: int = 0

    def __init__(self, config: Optional[GSEnvConfig] = None, **kwargs) -> None:
        if not torch.cuda.is_available():
            import os

            raise RuntimeError(
                f"BatchedGSUnifiedMujocoEnv requires CUDA for Gaussian Splatting rendering, but no CUDA device is available. CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
        if config is None:
            config = GSEnvConfig.model_validate(kwargs)

        gs_cfg = config.gaussian_render
        self._share_physics = bool(gs_cfg.share_physics)
        self._virtual_batch_size = int(config.batch_size)

        # Parent creates N plain UnifiedMujocoEnv instances (physics only).
        # GSEnvConfig validator already disabled native color/depth on GS cameras.
        # In share_physics mode, only 1 physics replica is created; self.batch_size
        # is then restored to the virtual batch size so downstream bg bookkeeping,
        # capture_observation, and user-visible APIs continue to see N.
        if self._share_physics:
            physics_cfg = config.model_copy(update={"batch_size": 1})
            super().__init__(physics_cfg, **kwargs)
            self.batch_size = self._virtual_batch_size
            self.config = config
            # Expose N aliases of the single physics env so external code
            # (backends, eval scripts) can index ``env.envs[i]`` for any i in
            # ``[0, N)``. Parent methods that iterate ``self.envs`` see N, so
            # the step-like methods below override them to avoid N× work.
            self.envs = [self.envs[0]] * self._virtual_batch_size
        else:
            super().__init__(config, **kwargs)

        self._gs_background_source = gs_cfg.background_ply
        self._is_multi_bg = gs_cfg.is_multi_background()
        self._gs_body_gaussians = gs_cfg.resolved_body_gaussians()

        # Single shared foreground renderer
        self._fg_gs_renderer = MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians=dict(self._gs_body_gaussians),
                background_ply=None,
                minibatch=self.config.gaussian_render.minibatch,
            ),
            self.envs[0].model,
        )

        # List of background renderers, one per unique PLY. Empty when no bg.
        self._bg_gs_renderers: list[MjxBatchSplatRenderer] = []
        self._bg_source_plys: list[str] = []
        # Per-env mapping env_idx -> index into ``_bg_gs_renderers``.
        self._env_bg_idx: np.ndarray = np.zeros(self.batch_size, dtype=np.int64)
        self._bg_rng = np.random.default_rng()

        # Per-object mask renderers (shared) — build via env[0] which has
        # .model, .get_logger(), and .config.mask_objects needed by the builder.
        self._gs_mask_renderers = self._build_shared_mask_renderers(
            dict(self._gs_body_gaussians)
        )

        # Cache camera specs from env[0] (homogeneous)
        self._camera_specs = self.envs[0]._camera_specs
        self._camera_ids = self.envs[0]._camera_ids

        # Background cache: (cam_ids_tuple, w, h) → (bg_rgb, bg_depth) tensors
        self._bg_cache: dict[
            tuple[tuple[int, ...], int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        if self._is_multi_bg:
            self._bg_source_plys = gs_cfg.resolved_background_plys()
            self._bg_gs_renderers = [
                self._make_bg_renderer(p) for p in self._bg_source_plys
            ]
            self._randomize_env_bg_assignment()
            reset_mode = (
                "unique per env when available, randomized on reset"
                if gs_cfg.randomize_background_on_reset
                else "unique per env when available, fixed after initial assignment"
            )
            bg_str = f" + {len(self._bg_source_plys)} backgrounds ({reset_mode})"
        else:
            self.set_background_transform(gs_cfg.resolved_background_transform())
            background_ply = gs_cfg.resolved_background_ply()
            bg_str = " + background" if background_ply else ""

        n_bodies = len(self._gs_body_gaussians)
        self.envs[0].get_logger().info(
            f"Batched GS renderer initialised with {n_bodies} body gaussian(s){bg_str}"
        )

    # ------------------------------------------------------------------
    # shared-physics step-like overrides (run the single physics env once)
    # ------------------------------------------------------------------
    #
    # In shared-physics mode ``self.envs`` is N aliases of the same underlying
    # ``UnifiedMujocoEnv``. Parent methods that iterate ``for env in self.envs``
    # therefore call the same env N times; for getters that is cheap (identity
    # rows stacked), but for anything that advances / mutates physics it is a
    # waste at best, and incorrect (last-wins on different inputs) at worst.
    # Override the hot step-like methods to apply element ``[0]`` of any
    # ``(N, …)`` input exactly once.

    @staticmethod
    def _any_in_mask(batch_size: int, env_mask) -> bool:
        if env_mask is None:
            return True
        return bool(np.asarray(env_mask, dtype=bool).reshape(-1).any())

    def step(self, action: np.ndarray, env_mask: np.ndarray | None = None) -> None:
        if not self._share_physics:
            return super().step(action, env_mask)
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            raise ValueError(
                f"Batched step expects shape (B, action_dim), got {action.shape}"
            )
        if action.shape[0] != self.batch_size:
            raise ValueError(
                f"Expected action shape ({self.batch_size}, action_dim), got {action.shape}"
            )
        if self._any_in_mask(self.batch_size, env_mask):
            self.envs[0].step(action[0])

    def update(self, env_mask: np.ndarray | None = None) -> None:
        if not self._share_physics:
            return super().update(env_mask)
        if self._any_in_mask(self.batch_size, env_mask):
            self.envs[0].update()

    def apply_joint_action(
        self,
        operator: str,
        action,
        env_mask: np.ndarray | None = None,
        kinematic: bool = False,
    ) -> None:
        if not self._share_physics:
            return super().apply_joint_action(operator, action, env_mask, kinematic)
        action = np.asarray(action, dtype=np.float64)
        first = action if action.ndim == 1 else action[0]
        if self._any_in_mask(self.batch_size, env_mask):
            self.envs[0].apply_joint_action(operator, first, kinematic=kinematic)

    def apply_pose_action(
        self,
        operator: str,
        position,
        orientation,
        gripper=None,
        env_mask: np.ndarray | None = None,
        kinematic: bool = False,
    ) -> None:
        if not self._share_physics:
            return super().apply_pose_action(
                operator, position, orientation, gripper, env_mask, kinematic
            )
        pos = np.asarray(position)
        ori = np.asarray(orientation)
        p0 = pos if pos.ndim == 1 else pos[0]
        o0 = ori if ori.ndim == 1 else ori[0]
        g0 = None
        if gripper is not None:
            g = np.asarray(gripper, dtype=np.float64)
            g0 = g if g.ndim == 1 else g[0]
        if self._any_in_mask(self.batch_size, env_mask):
            self.envs[0].apply_pose_action(operator, p0, o0, g0, kinematic=kinematic)

    # ------------------------------------------------------------------
    # background lifecycle
    # ------------------------------------------------------------------

    def _make_bg_renderer(self, background_ply: str) -> MjxBatchSplatRenderer:
        return MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians={},
                background_ply=background_ply,
                minibatch=self.config.gaussian_render.minibatch,
            ),
            self.envs[0].model,
        )

    def _randomize_env_bg_assignment(self) -> np.ndarray:
        """Randomly pick one bg per environment.

        When the configured background count covers the whole batch, sampling is
        done without replacement so every env gets a distinct background. Cache
        is cleared because cached bg tensors are keyed by the prior env→bg
        mapping.
        """
        self._env_bg_idx = _sample_env_background_indices(
            batch_size=self.batch_size,
            num_backgrounds=len(self._bg_gs_renderers),
            rng=self._bg_rng,
        )
        self._bg_cache.clear()
        return self._env_bg_idx

    def reset(self, env_mask: np.ndarray | None = None) -> None:
        if self._share_physics:
            # Partial per-env resets aren't meaningful when physics is shared;
            # reset the single replica iff any virtual env asked for it.
            if self._any_in_mask(self.batch_size, env_mask):
                self.envs[0].reset()
        else:
            super().reset(env_mask)
        if (
            self._is_multi_bg
            and self.config.gaussian_render.randomize_background_on_reset
        ):
            self._randomize_env_bg_assignment()

    def refresh_viewer(self) -> None:
        if self._share_physics:
            self.envs[0].refresh_viewer()
        else:
            super().refresh_viewer()

    def is_updated(self) -> np.ndarray:
        if not self._share_physics:
            return super().is_updated()
        # ``UnifiedMujocoEnv.is_updated()`` is stateful: it advances the
        # env's cached ``_last_time`` when reporting ``True``. In shared
        # physics mode ``self.envs`` contains N aliases of the same env, so
        # calling the parent implementation would yield
        # ``[True, False, False, ...]`` for a single physics tick, causing
        # downstream batched samplers to only record env 0. Probe the shared
        # physics replica once and broadcast the result to all virtual envs.
        updated = bool(self.envs[0].is_updated())
        return np.full(self.batch_size, updated, dtype=bool)

    def set_background_transform(
        self, pose: BackgroundPose | list[float]
    ) -> BackgroundPose:
        if self._is_multi_bg:
            raise ValueError(
                "set_background_transform is not supported when background_ply "
                "is a list; use background_transforms in the config to set "
                "per-background poses."
            )
        pose = _normalize_background_pose(pose)
        background_ply = _materialize_transformed_background_ply(
            self._gs_background_source,
            pose,
        )
        self._bg_source_plys = [background_ply] if background_ply else []
        self._bg_gs_renderers = (
            [self._make_bg_renderer(background_ply)] if background_ply else []
        )
        self._env_bg_idx = np.zeros(self.batch_size, dtype=np.int64)
        self._bg_cache.clear()
        object.__setattr__(self.config.gaussian_render, "background_transform", pose)
        return pose

    # ------------------------------------------------------------------
    # observation capture
    # ------------------------------------------------------------------

    def capture_observation(self) -> dict[str, dict[str, Any]]:
        if self._share_physics:
            # Bypass parent's N-fold loop: in shared mode ``self.envs`` is N
            # aliases of the same env, so parent would call
            # ``env.capture_observation()`` N times redundantly.
            obs = self._broadcast_single_obs(self.envs[0].capture_observation())
        else:
            obs = super().capture_observation()
        self._inject_batched_gs_renders(obs)
        return obs

    def _broadcast_single_obs(
        self, obs_one: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Replicate a single-env obs dict into the batched ``(N, ...)`` format."""
        n = self.batch_size
        batched: dict[str, dict[str, Any]] = {}
        for key, entry in obs_one.items():
            data = entry["data"]
            t = entry["t"]
            if isinstance(data, dict):
                # Structured entries: list-of-N where each row is the same dict.
                batched[key] = {
                    "data": [data] * n,
                    "t": np.asarray([t] * n),
                }
            else:
                arr = np.asarray(data)
                batched[key] = {
                    "data": np.broadcast_to(arr[None, ...], (n,) + arr.shape).copy(),
                    "t": np.asarray([t] * n),
                }
        return batched

    def _inject_batched_gs_renders(self, obs: dict[str, dict[str, Any]]) -> None:
        """Batch-render GS color/depth/mask across all envs and cameras.

        Cameras are grouped by resolution so that ``batch_env_render`` is
        called once per resolution group with ``Ncam > 1``, rather than once
        per camera.
        """
        if self._share_physics:
            self._inject_shared_gs_renders(obs)
            return

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
                minibatch=self.config.gaussian_render.minibatch,
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

        When multiple background renderers are configured, each unique
        renderer is invoked only on the subset of envs that use it and the
        per-env results are scattered back into the full ``(Nenv, …)`` tensor.
        """
        if not self._bg_gs_renderers:
            return None
        cache_key = (tuple(cam_ids), width, height)
        if use_cache and cache_key in self._bg_cache:
            return self._bg_cache[cache_key]

        if len(self._bg_gs_renderers) == 1:
            bg_rend = self._bg_gs_renderers[0]
            bg_gsb = bg_rend.batch_update_gaussians(body_pos, body_quat)
            bg_rgb, bg_depth = bg_rend.batch_env_render(
                bg_gsb, cam_pos, cam_xmat, height, width, fovy
            )
        else:
            bg_rgb, bg_depth = self._render_per_env_backgrounds(
                cam_pos, cam_xmat, fovy, height, width, body_pos, body_quat
            )

        if use_cache:
            self._bg_cache[cache_key] = (bg_rgb, bg_depth)
        return (bg_rgb, bg_depth)

    def _render_per_env_backgrounds(
        self,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy: np.ndarray,
        height: int,
        width: int,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render each unique background on its subset of envs and scatter
        the results back into a full ``(Nenv, Ncam, …)`` tensor.
        """
        Nenv = self.batch_size
        bg_rgb_full: torch.Tensor | None = None
        bg_depth_full: torch.Tensor | None = None
        unique_bg_idxs = np.unique(self._env_bg_idx)
        for bg_idx in unique_bg_idxs:
            env_mask = self._env_bg_idx == bg_idx
            env_sel = np.nonzero(env_mask)[0]
            bg_rend = self._bg_gs_renderers[int(bg_idx)]
            sub_body_pos = body_pos[env_sel]
            sub_body_quat = body_quat[env_sel]
            sub_cam_pos = cam_pos[env_sel]
            sub_cam_xmat = cam_xmat[env_sel]
            sub_fovy = fovy[env_sel]
            gsb = bg_rend.batch_update_gaussians(sub_body_pos, sub_body_quat)
            sub_rgb, sub_depth = bg_rend.batch_env_render(
                gsb, sub_cam_pos, sub_cam_xmat, height, width, sub_fovy
            )
            if bg_rgb_full is None:
                shape_rgb = (Nenv,) + tuple(sub_rgb.shape[1:])
                shape_depth = (Nenv,) + tuple(sub_depth.shape[1:])
                bg_rgb_full = torch.empty(
                    shape_rgb, dtype=sub_rgb.dtype, device=sub_rgb.device
                )
                bg_depth_full = torch.empty(
                    shape_depth, dtype=sub_depth.dtype, device=sub_depth.device
                )
            env_idx_t = torch.as_tensor(
                env_sel, dtype=torch.long, device=bg_rgb_full.device
            )
            bg_rgb_full.index_copy_(0, env_idx_t, sub_rgb)
            bg_depth_full.index_copy_(0, env_idx_t, sub_depth)
        assert bg_rgb_full is not None and bg_depth_full is not None
        return bg_rgb_full, bg_depth_full

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
    # shared-physics GS path
    # ------------------------------------------------------------------

    def _inject_shared_gs_renders(self, obs: dict[str, dict[str, Any]]) -> None:
        """Render GS color/depth/mask for the shared-physics mode.

        Physics (and thus body/camera poses) is shared across the virtual
        batch: foreground gaussians are updated and rasterized only once at
        ``nenv=1``. Each unique background is rendered once at ``nenv=1`` and
        scattered to the rows of the virtual batch that use it. The
        compositing against the shared foreground is done in Python with
        PyTorch broadcasting between the ``(1, ...)`` foreground tensors and
        ``(N, ...)`` background tensors.
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
        N = self._virtual_batch_size
        env0 = self.envs[0]
        t0 = (
            int(env0.data.time * 1e9) if self.config.stamp_ns else float(env0.data.time)
        )
        timestamps = np.full((N,), t0)

        # body_pos/body_quat come from the single shared physics env.
        body_pos = np.asarray(env0.data.xpos, dtype=np.float32).reshape(
            1, env0.model.nbody, 3
        )
        body_quat = np.asarray(env0.data.xquat, dtype=np.float32).reshape(
            1, env0.model.nbody, 4
        )

        fg_gsb = self._fg_gs_renderer.batch_update_gaussians(body_pos, body_quat)

        res_groups: OrderedDict[tuple[int, int, bool], list[str]] = OrderedDict()
        for cam_name in all_gs_cams:
            spec = self._camera_specs[cam_name]
            key = (spec.height, spec.width, spec.is_static)
            res_groups.setdefault(key, []).append(cam_name)

        for (H, W, is_static), cam_names in res_groups.items():
            Ncam = len(cam_names)
            cam_ids = [self._camera_ids[c] for c in cam_names]

            cam_pos = np.stack(
                [
                    np.asarray(env0.data.cam_xpos[cid], dtype=np.float32)
                    for cid in cam_ids
                ]
            ).reshape(1, Ncam, 3)
            cam_xmat = np.stack(
                [
                    np.asarray(env0.data.cam_xmat[cid], dtype=np.float32)
                    for cid in cam_ids
                ]
            ).reshape(1, Ncam, 9)
            fovy = np.asarray(
                [env0.model.cam_fovy[cid] for cid in cam_ids], dtype=np.float32
            ).reshape(1, Ncam)

            any_color = any(c in gs_color_set for c in cam_names)
            any_depth = any(c in gs_depth_set for c in cam_names)
            any_mask = (
                any(c in gs_mask_set for c in cam_names) and self._gs_mask_renderers
            )

            # Foreground render at nenv=1, no bg_imgs (we composite manually
            # so background can have leading dim N).
            fg_rgb_1, fg_depth_1 = self._fg_gs_renderer.batch_env_render(
                fg_gsb, cam_pos, cam_xmat, H, W, fovy
            )
            alphas_1 = self._fg_gs_renderer.rasterizations[1]

            # Background render per unique bg idx, scatter to (N, Ncam, ...).
            # Static cameras reuse the cached (N, Ncam, ...) tensor across frames;
            # bg assignment changes (on reset with randomize=True) clear the cache.
            bg_rgb_N: torch.Tensor | None = None
            bg_depth_N: torch.Tensor | None = None
            if self._bg_gs_renderers:
                bg_rgb_N, bg_depth_N = self._render_shared_per_env_backgrounds(
                    cam_pos,
                    cam_xmat,
                    fovy,
                    H,
                    W,
                    body_pos,
                    body_quat,
                    cam_ids=cam_ids,
                    use_cache=is_static,
                )

            # Composite with broadcasting.
            if bg_rgb_N is not None:
                one_minus_alpha = 1 - alphas_1
                full_rgb = fg_rgb_1 * alphas_1 + bg_rgb_N * one_minus_alpha
                full_depth = fg_depth_1 * alphas_1 + bg_depth_N * one_minus_alpha
            else:
                full_rgb = fg_rgb_1.expand(N, *fg_rgb_1.shape[1:])
                full_depth = fg_depth_1.expand(N, *fg_depth_1.shape[1:])

            kc = self._key_creator
            for cam_idx, cam_name in enumerate(cam_names):
                spec = self._camera_specs[cam_name]
                if cam_name in gs_color_set:
                    rgb = (
                        torch.clamp(full_rgb[:, cam_idx], 0.0, 1.0)
                        .mul(255)
                        .to(torch.uint8)
                    )
                    if self.config.to_numpy:
                        rgb = rgb.cpu().numpy()
                    obs[kc.create_color_key(cam_name)] = {
                        "data": rgb
                        if not structured
                        else create_image_data_batch(
                            rgb, timestamps, cam_name, tobytes=False
                        ),
                        "t": timestamps,
                    }
                if cam_name in gs_depth_set:
                    depth = full_depth[:, cam_idx, :, :, 0]
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

            if any_mask:
                scene_depth_t = GSUnifiedMujocoEnv._compose_mask_scene_depth(
                    fg_depth=fg_depth_1,
                    bg_depth=bg_depth_N,
                )
                all_masks, all_heat_maps = self._render_shared_gs_masks_multicam(
                    cam_pos=cam_pos,
                    cam_xmat=cam_xmat,
                    fovy=fovy,
                    height=H,
                    width=W,
                    body_pos=body_pos,
                    body_quat=body_quat,
                    scene_depth_t=scene_depth_t,
                )
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
            # Silence unused flags on pure-color groups.
            _ = any_color, any_depth

    def _render_shared_per_env_backgrounds(
        self,
        cam_pos: np.ndarray,
        cam_xmat: np.ndarray,
        fovy: np.ndarray,
        height: int,
        width: int,
        body_pos: np.ndarray,
        body_quat: np.ndarray,
        cam_ids: list[int],
        use_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Render each unique background once at ``nenv=1``, scatter to
        ``(N, Ncam, H, W, C)`` at rows selected by ``self._env_bg_idx``.

        When ``use_cache`` is True (static cameras + fixed bg assignment), the
        scattered ``(N, Ncam, ...)`` tensor is cached under ``self._bg_cache``
        keyed by ``(tuple(cam_ids), width, height)``; subsequent frames return
        the cached tensor unchanged. ``_randomize_env_bg_assignment`` and
        ``set_background_transform`` both clear the cache so stale mappings
        cannot leak across bg resampling.
        """
        cache_key = (tuple(cam_ids), width, height)
        if use_cache and cache_key in self._bg_cache:
            return self._bg_cache[cache_key]

        N = self._virtual_batch_size
        bg_rgb_full: torch.Tensor | None = None
        bg_depth_full: torch.Tensor | None = None
        unique_bg_idxs = np.unique(self._env_bg_idx)
        for bg_idx in unique_bg_idxs:
            rows = np.nonzero(self._env_bg_idx == bg_idx)[0]
            bg_rend = self._bg_gs_renderers[int(bg_idx)]
            gsb = bg_rend.batch_update_gaussians(body_pos, body_quat)
            sub_rgb, sub_depth = bg_rend.batch_env_render(
                gsb, cam_pos, cam_xmat, height, width, fovy
            )
            # sub_rgb, sub_depth: (1, Ncam, H, W, C)
            if bg_rgb_full is None:
                shape_rgb = (N,) + tuple(sub_rgb.shape[1:])
                shape_depth = (N,) + tuple(sub_depth.shape[1:])
                bg_rgb_full = torch.empty(
                    shape_rgb, dtype=sub_rgb.dtype, device=sub_rgb.device
                )
                bg_depth_full = torch.empty(
                    shape_depth, dtype=sub_depth.dtype, device=sub_depth.device
                )
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=bg_rgb_full.device)
            bg_rgb_full.index_copy_(
                0, rows_t, sub_rgb.expand(len(rows), *sub_rgb.shape[1:])
            )
            bg_depth_full.index_copy_(
                0, rows_t, sub_depth.expand(len(rows), *sub_depth.shape[1:])
            )
        assert bg_rgb_full is not None and bg_depth_full is not None
        if use_cache:
            self._bg_cache[cache_key] = (bg_rgb_full, bg_depth_full)
        return bg_rgb_full, bg_depth_full

    def _render_shared_gs_masks_multicam(
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
        """Shared-physics mask rendering.

        Each mask object is rasterized once at ``nenv=1``; the visibility
        comparison is broadcast against ``(N, Ncam, H, W)`` scene depth.
        Heat-map channel assignment is driven by ``envs[0]`` because all
        virtual envs share physics (and therefore interest-object state).
        """
        N = self._virtual_batch_size
        # scene_depth_t may be (1, Ncam, H, W, 1) or (N, Ncam, H, W, 1).
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
            (N, Ncam, height, width), dtype=torch.bool, device=dev
        )
        heat_map_t = torch.zeros(
            (N, Ncam, height, width, n_heatmap_ops), dtype=torch.bool, device=dev
        )

        heatmap_ops_index = {
            op: i for i, op in enumerate(self.config.heatmap_operations)
        }

        body_pos_t = torch.as_tensor(body_pos, device=dev, dtype=torch.float32)
        body_quat_t = torch.as_tensor(body_quat, device=dev, dtype=torch.float32)
        cam_pos_t = torch.as_tensor(cam_pos, device=dev, dtype=torch.float32)
        cam_xmat_t = torch.as_tensor(cam_xmat, device=dev, dtype=torch.float32)

        interest_ops = self.envs[0]._interest_object_operations

        for object_name, renderer in self._gs_mask_renderers.items():
            gsb = renderer.batch_update_gaussians(body_pos_t, body_quat_t)
            alpha_t, obj_depth_t = renderer.batch_env_render(
                gsb, cam_pos_t, cam_xmat_t, height, width, fovy
            )
            # alpha_t: (1, Ncam, H, W, 3), obj_depth_t: (1, Ncam, H, W, 1)
            alpha_max = alpha_t.detach().max(dim=-1).values  # (1, Ncam, H, W)
            obj_depth = torch.nan_to_num(
                obj_depth_t[..., 0].detach(), nan=0.0, posinf=0.0, neginf=0.0
            )  # (1, Ncam, H, W)

            raw_visible = alpha_max > self._GS_MASK_ALPHA_THRESHOLD  # (1, ...)
            depth_valid = (scene_depth > 0.0) & (
                obj_depth > 0.0
            )  # broadcasts to (N, ...)
            occluded = depth_valid & (obj_depth > scene_depth + self._GS_MASK_DEPTH_EPS)
            visible = raw_visible & ~occluded  # (N, Ncam, H, W)
            binary_mask_t |= visible

            operation_name = interest_ops.get(object_name)
            if operation_name is None:
                continue
            channel_idx = heatmap_ops_index.get(operation_name)
            if channel_idx is None:
                continue
            heat_map_t[..., channel_idx] |= visible

        binary_mask = binary_mask_t.to(torch.uint8).cpu().numpy()
        heat_map = heat_map_t.to(torch.uint8).cpu().numpy()
        return binary_mask, heat_map

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._fg_gs_renderer = None
        self._bg_gs_renderers = []
        self._gs_mask_renderers = {}
        self._bg_cache.clear()
        super().close()
