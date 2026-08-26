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
        scene:
          base: assets/xmls/scenes/pick_and_place/demo.xml
          layers: []
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

import gc
import glob as _glob
import hashlib
import itertools
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Set

import numpy as np
import torch
from gaussian_renderer import BatchSplatConfig, GSRendererMuJoCo, MjxBatchSplatRenderer
from gaussian_renderer.core.gaussiandata import GaussianData
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


def _is_dict_background(value: Any) -> bool:
    """Whether ``background_ply`` is the parts-dict form."""
    return isinstance(value, dict)


def _merge_background_plys(
    plys: Sequence[str], cache_subdir: str = "gs_background_combos"
) -> str:
    """Concatenate several PLYs into one merged PLY (cached on disk).

    Each combination of part PLYs becomes a single ``GaussianData`` whose
    arrays are the concatenation of the sources along axis 0. Cached under
    ``.cache/<cache_subdir>/`` keyed by sha1 of the sorted absolute
    source paths so re-runs reuse the merged file.

    Used for both multi-part backgrounds (default ``gs_background_combos``)
    and list-valued ``body_gaussians`` (caller passes ``gs_body_combos``).
    """
    if not plys:
        raise ValueError("_merge_background_plys requires at least one PLY")
    if len(plys) == 1:
        return plys[0]

    abs_paths = [str(Path(p).expanduser().resolve()) for p in plys]
    key_src = "|".join(sorted(abs_paths))
    cache_key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:12]
    stems = "+".join(Path(p).stem for p in plys)
    cache_dir = Path(".cache") / cache_subdir
    cache_path = cache_dir / f"{stems}__merged_{cache_key}.ply"
    if cache_path.exists():
        return str(cache_path)

    parts = [load_ply(p) for p in plys]
    # Reshape any (N, K, 3) SH arrays into the flat (N, K*3) form before
    # measuring widths; ``save_ply`` writes flat layout, but ``load_ply``
    # variants may return either shape.
    sh_arrays = [
        gd.sh.reshape(gd.sh.shape[0], -1) if gd.sh.ndim > 2 else gd.sh for gd in parts
    ]
    # Pad lower-SH-degree PLYs up to the max width with zeros on the trailing
    # coefficients (DC stays in [:, :3]; missing higher-order bands contribute
    # zero — no view-dependent effect from those points).
    max_sh_dim = max(sh.shape[1] for sh in sh_arrays)
    padded_sh: list[np.ndarray] = []
    for p, sh in zip(plys, sh_arrays):
        if sh.shape[1] == max_sh_dim:
            padded_sh.append(sh)
            continue
        if (max_sh_dim - sh.shape[1]) % 3 != 0:
            raise ValueError(
                f"PLYs to merge have SH widths that aren't aligned to RGB "
                f"triples: max={max_sh_dim}, got={sh.shape[1]} from {p}"
            )
        pad = np.zeros((sh.shape[0], max_sh_dim - sh.shape[1]), dtype=sh.dtype)
        padded_sh.append(np.concatenate([sh, pad], axis=1))
    merged = GaussianData(
        xyz=np.concatenate([gd.xyz for gd in parts], axis=0),
        rot=np.concatenate([gd.rot for gd in parts], axis=0),
        scale=np.concatenate([gd.scale for gd in parts], axis=0),
        opacity=np.concatenate([gd.opacity for gd in parts], axis=0),
        sh=np.concatenate(padded_sh, axis=0),
    )
    save_ply(merged, cache_path)
    return str(cache_path)


def _sample_combinations(
    part_plys: Sequence[Sequence[str]],
    cap: int | None,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """Sample combinations from the cartesian product of per-part PLY lists.

    When ``cap`` is ``None`` or ``cap >= total``, returns the full product in
    deterministic order (``itertools.product``). Otherwise samples ``cap``
    flat indices without replacement and decodes each via ``np.unravel_index``
    to a per-part multi-index.
    """
    if not part_plys:
        return []
    sizes = [len(p) for p in part_plys]
    if any(s == 0 for s in sizes):
        raise ValueError(
            "background_ply dict has an empty part; every part must expand to "
            "at least one PLY"
        )
    total = int(np.prod(sizes))
    if cap is None or cap >= total:
        return [tuple(combo) for combo in itertools.product(*part_plys)]
    flat_idx = rng.choice(total, size=cap, replace=False)
    out: list[tuple[str, ...]] = []
    for fi in flat_idx:
        multi = np.unravel_index(int(fi), sizes)
        out.append(tuple(part_plys[i][multi[i]] for i in range(len(part_plys))))
    return out


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


def _fg_variant_lengths(body_gaussians: Dict[str, str | list[str]]) -> Dict[str, int]:
    """For each body, length of its variant list (1 for str values).

    Returns an empty dict when ``body_gaussians`` is empty.
    """
    out: Dict[str, int] = {}
    for body_name, value in body_gaussians.items():
        if isinstance(value, str):
            out[body_name] = 1
        else:
            n = len(list(value))
            if n == 0:
                raise ValueError(
                    f"body_gaussians['{body_name}'] is an empty list; "
                    f"provide at least one PLY path"
                )
            out[body_name] = n
    return out


def _fg_variant_count(body_gaussians: Dict[str, str | list[str]]) -> int:
    """Cartesian-product count over per-body variant lists.

    Returns ``1`` when all values are single-string (degenerate, single FG).
    Returns ``0`` only when ``body_gaussians`` is empty.
    """
    if not body_gaussians:
        return 0
    total = 1
    for n in _fg_variant_lengths(body_gaussians).values():
        total *= n
    return total


def _fg_variant_at(
    body_gaussians: Dict[str, str | list[str]], variant_idx: int
) -> Dict[str, str]:
    """Return the ``body_gaussians`` dict for variant index ``variant_idx``.

    Variant indices are decoded against the per-body list sizes via
    ``np.unravel_index``; body iteration order matches the source dict's
    insertion order (Python 3.7+ guaranteed) so the mapping is stable.
    Single-string body entries contribute a constant value across all
    variants. The total variant count is ``_fg_variant_count``.
    """
    sizes_dict = _fg_variant_lengths(body_gaussians)
    body_order = list(body_gaussians.keys())
    sizes = [sizes_dict[b] for b in body_order]
    total = 1
    for s in sizes:
        total *= s
    if not 0 <= variant_idx < total:
        raise IndexError(f"variant_idx={variant_idx} out of range [0, {total})")
    multi = np.unravel_index(int(variant_idx), sizes) if sizes else ()
    out: Dict[str, str] = {}
    for i, body_name in enumerate(body_order):
        value = body_gaussians[body_name]
        if isinstance(value, str):
            out[body_name] = value
        else:
            out[body_name] = list(value)[int(multi[i])]
    return out


class _FGBGCombinationCursor:
    """FG-grouped round-robin cursor over the M*N (fg, bg) combination space.

    Each ``next_batch()`` returns ``(fg_idx, [bg_idx_0, ..., bg_idx_{B-1}])``
    where all ``B`` background indices are distinct and the foreground index
    is the same. Within one round, every (fg, bg) combo appears at most
    once. When the current FG's BG pool has fewer than ``B`` indices left,
    the trailing remainder is dropped for the round and the cursor
    advances to the next FG. After all ``M`` foregrounds are exhausted,
    the round restarts with a fresh random permutation of both axes.

    The cursor reuses the env's ``np.random.Generator`` so the entire
    sampling sequence is reproducible under a fixed seed.
    """

    def __init__(
        self,
        num_fg: int,
        num_bg: int,
        batch_size: int,
        rng: np.random.Generator,
    ) -> None:
        if num_fg <= 0:
            raise ValueError(f"num_fg must be > 0; got {num_fg}")
        if num_bg <= 0:
            raise ValueError(f"num_bg must be > 0; got {num_bg}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {batch_size}")
        if num_bg < batch_size:
            raise ValueError(
                f"foreground_variant mode requires num_bg >= batch_size "
                f"(no duplicates allowed per batch); got num_bg={num_bg}, "
                f"batch_size={batch_size}"
            )
        self.M = num_fg
        self.N = num_bg
        self.B = batch_size
        self.rng = rng
        self._start_new_round()

    def _start_new_round(self) -> None:
        self._fg_order: list[int] = [int(i) for i in self.rng.permutation(self.M)]
        # Per-FG BG ordering, freshly shuffled at round start.
        self._bg_orders: Dict[int, list[int]] = {
            fg: [int(i) for i in self.rng.permutation(self.N)] for fg in range(self.M)
        }
        self._fg_cursor = 0
        self._bg_cursor = 0

    def next_batch(self) -> tuple[int, list[int]]:
        """Return one (fg_idx, bg_idxs) pair, advancing the cursor.

        Skips per-FG remainders smaller than ``batch_size`` and reshuffles
        when the round is exhausted.
        """
        while True:
            if self._fg_cursor >= len(self._fg_order):
                self._start_new_round()
            current_fg = self._fg_order[self._fg_cursor]
            bg_order = self._bg_orders[current_fg]
            remaining = len(bg_order) - self._bg_cursor
            if remaining >= self.B:
                bgs = bg_order[self._bg_cursor : self._bg_cursor + self.B]
                self._bg_cursor += self.B
                return current_fg, list(bgs)
            # Drop the trailing remainder for this FG and advance.
            self._fg_cursor += 1
            self._bg_cursor = 0


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

    body_gaussians: Dict[str, str | list[str]] = Field(default_factory=dict)
    """Mapping from MuJoCo body name to PLY file path(s).

    Each value is either:

    - ``str`` — a single PLY path for the body.
    - ``list[str]`` — multiple PLYs that are concatenated into one merged
      PLY (cached under ``.cache/gs_body_combos/`` keyed by sha1 of the
      sorted absolute source paths). All subsequent ``body_transforms`` /
      ``body_mirrors`` operations treat the merged PLY as the body's
      source, so a multi-part body still shows up as one rigid asset.
      The same SH-degree-padding rule as background merging applies."""
    background_ply: str | list[str] | Dict[str, str | list[str]] | None = None
    """Optional background PLY. Three accepted forms:

    - ``str`` — single background PLY (or glob expanding to a pool).
    - ``list[str]`` — pool of full-scene backgrounds; each env is assigned one
      so backgrounds randomize across the batch.
    - ``Dict[str, str | list[str]]`` — *parts* dict. Each value is a path or
      glob expanding to a pool of PLYs for that part (e.g. ``wall``, ``inside``).
      Each env's background is a *combination* — one PLY pulled from each part —
      and the per-env combinations are sampled (without replacement when the
      cartesian product is large enough) up to ``batch_size``. Per-part PLYs
      are merged into a single combined PLY (cached on disk by hash) before
      being handed to the renderer.

    For the list and dict forms, GS envs randomly assign one background per env
    initially and optionally reassign on each ``reset``."""
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
    background_transform_randomization: Dict[str, Dict[str, list]] = Field(
        default_factory=dict
    )
    """Per-background *position-only* randomization ranges, sampled
    independently per env (combination) at ``_setup_gs_render_state`` time
    using the same ``_bg_rng`` used for background pool sampling.

    Outer key matches by the same precedence as ``background_transforms``
    (full path / ``str(Path)`` / file name / stem), plus the part name in
    the dict (parts) form of ``background_ply``. Inner dict maps axis
    ``'x'`` / ``'y'`` / ``'z'`` to a ``[low, high]`` range; a uniform offset
    is drawn per axis and *added* to the deterministic base position
    resolved from ``background_transforms`` / ``background_transform``.
    Omitted axes default to no perturbation. Orientation is **not**
    randomized.

    Example (parts dict)::

        background_ply:
          wall:   ${bg3dgs_dir}/wall*.ply
          inside: ${bg3dgs_dir}/inside*.ply
        background_transforms:
          wall:   [-0.035, 0.491151, -0.136038, ...]   # base 7-vec
          inside: [ 0.300, 0.491151, -0.136038, ...]
        background_transform_randomization:
          wall:
            x: [-0.05, 0.05]      # ± 5 cm along world X
            z: [-0.02, 0.02]
          inside:
            x: [-0.10, 0.10]

    Each env gets one merged background PLY whose wall / inside parts are
    independently shifted by the sampled offsets. Sampled offsets are
    baked into the cache file name via ``_materialize_transformed_background_ply``,
    so distinct samples become distinct cache entries.

    Note: randomization is sampled once per renderer at init time. Setting
    ``randomize_background_on_reset=true`` re-shuffles which env uses which
    pre-built renderer but does not re-sample the transform offsets — the
    pool of distinct visual variants is fixed at init.
    """
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
    env. Requires ``background_ply`` to be multi-valued (list, glob, or parts
    dict) and ``batch_size > 1``. Validated on ``GSEnvConfig``."""
    foreground_variant: bool = False
    """Enable FG-grouped round-robin sampling over the foreground × background
    combination space.

    When ``True``:

    - Any ``list[str]`` value in ``body_gaussians`` is treated as a pool of
      *variants* for that body (not merged). The total foreground variant
      count ``M`` is the cartesian product of per-body list lengths
      (non-list entries contribute a factor of 1, shared across all variants).
    - The background pool has ``N`` entries (existing list / glob / parts
      dict resolution).
    - At init and on every ``reset`` (when
      ``randomize_background_on_reset=true``), a single foreground variant
      is selected for the entire batch and ``batch_size`` distinct
      backgrounds are assigned to the envs without replacement. The cursor
      consumes ``floor(N / batch_size) * batch_size`` backgrounds per FG
      before advancing to the next foreground; the trailing ``N % batch_size``
      backgrounds are skipped within the current round. After all ``M``
      foregrounds are exhausted the combination space reshuffles.
    - Requires ``N >= batch_size`` so each batch can have distinct
      backgrounds without replacement. Validated in
      ``_setup_gs_render_state``.
    - GPU renderers for FG variants, mask renderers, and BG entries are
      built lazily on first use and cached for the lifetime of the env.

    Default ``False`` preserves the existing semantics: ``body_gaussians``
    list values are *merged* and the per-env background assignment uses
    independent sampling from the BG pool."""

    @model_validator(mode="after")
    def _validate_background_transform_randomization(
        self,
    ) -> "GaussianRenderConfig":
        """Range entries must be ``[low, high]`` with ``low <= high``."""
        for key, axis_dict in self.background_transform_randomization.items():
            if not isinstance(axis_dict, dict):
                raise ValueError(
                    f"background_transform_randomization['{key}'] must be a "
                    f"dict mapping axis -> [low, high]; got {type(axis_dict).__name__}"
                )
            for axis, rng in axis_dict.items():
                if axis not in ("x", "y", "z"):
                    raise ValueError(
                        f"background_transform_randomization['{key}']: "
                        f"unknown axis '{axis}' (expected 'x', 'y', or 'z')"
                    )
                rng_seq = list(rng)
                if len(rng_seq) != 2:
                    raise ValueError(
                        f"background_transform_randomization['{key}']['{axis}'] "
                        f"must be [low, high]; got length {len(rng_seq)}"
                    )
                lo, hi = float(rng_seq[0]), float(rng_seq[1])
                if lo > hi:
                    raise ValueError(
                        f"background_transform_randomization['{key}']['{axis}'] "
                        f"requires low <= high; got [{lo}, {hi}]"
                    )
        return self

    def _randomization_for(self, *keys: str) -> Dict[str, list] | None:
        """Look up ``background_transform_randomization`` by candidate keys
        (in order); return the first match, or ``None`` when none configured."""
        for k in keys:
            if k in self.background_transform_randomization:
                return self.background_transform_randomization[k]
        return None

    def _sample_position_offset(
        self,
        key_candidates: tuple[str, ...],
        rng: np.random.Generator,
    ) -> tuple[float, float, float]:
        """Sample a uniform position offset for the first matching key.

        Returns ``(0.0, 0.0, 0.0)`` when no entry matches.
        """
        cfg = self._randomization_for(*key_candidates)
        if cfg is None:
            return (0.0, 0.0, 0.0)
        out = []
        for axis in ("x", "y", "z"):
            rng_pair = cfg.get(axis)
            if rng_pair is None:
                out.append(0.0)
            else:
                lo, hi = float(rng_pair[0]), float(rng_pair[1])
                out.append(float(rng.uniform(lo, hi)))
        return tuple(out)

    @staticmethod
    def _ply_key_candidates(bg_ply: str, part: str | None = None) -> tuple[str, ...]:
        """Return the candidate keys (in precedence order: PLY-level → part)
        used to look up per-PLY config in ``background_transforms`` /
        ``background_transform_randomization``."""
        bg_path = Path(bg_ply)
        keys = (bg_ply, str(bg_path), bg_path.name, bg_path.stem)
        return keys + ((part,) if part is not None else ())

    def has_position_randomization(self) -> bool:
        """Whether any entry of ``background_transform_randomization`` is set."""
        return bool(self.background_transform_randomization)

    def resolved_body_gaussians(self, variant_idx: int | None = None) -> Dict[str, str]:
        """Return ``body_gaussians`` with any configured transforms / mirrors
        substituted by the corresponding cached PLY paths.

        Two ways to choose the per-body PLY source:

        - ``variant_idx is None`` (default) — *merge* mode. List-valued
          entries are merged into a single PLY per body via
          ``_merge_background_plys`` (cached under
          ``.cache/gs_body_combos/``). Used by the standard rendering
          path that treats list values as multi-part bodies.
        - ``variant_idx`` is an ``int`` — *variant* mode used by
          ``foreground_variant``: each list-valued body picks its
          ``np.unravel_index``-decoded variant via ``_fg_variant_at``,
          producing one of ``M = ∏ len(list)`` distinct per-body
          combinations. No merging occurs.

        ``body_transforms`` / ``body_mirrors`` then operate on whichever
        per-body source was selected. The returned dict always maps each
        body name to a single PLY path.
        """
        if variant_idx is None:
            sources: Dict[str, str] = {}
            for body_name, value in self.body_gaussians.items():
                if isinstance(value, str):
                    sources[body_name] = value
                else:
                    paths = list(value)
                    if not paths:
                        raise ValueError(
                            f"body_gaussians['{body_name}'] is an empty list; "
                            f"provide at least one PLY path"
                        )
                    sources[body_name] = _merge_background_plys(
                        paths, cache_subdir="gs_body_combos"
                    )
        else:
            sources = _fg_variant_at(self.body_gaussians, variant_idx)

        if not self.body_transforms and not self.body_mirrors:
            return sources

        unknown = (set(self.body_transforms) | set(self.body_mirrors)) - set(sources)
        if unknown:
            raise ValueError(
                f"body_transforms/body_mirrors reference unknown body(s): {sorted(unknown)}. "
                f"Keys must appear in body_gaussians: {sorted(sources)}"
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
        for body_name, src_ply in sources.items():
            spec = self.body_transforms.get(body_name)
            if spec is None:
                transformed[body_name] = src_ply
                continue
            pose = spec.resolved_pose()
            center = None
            if spec.center is not None:
                center = _explicit_center(spec, "body_transforms", body_name)
            elif spec.share_center_with is not None:
                if spec.share_center_with not in sources:
                    raise ValueError(
                        f"body_transforms['{body_name}'].share_center_with="
                        f"'{spec.share_center_with}' is not in body_gaussians"
                    )
                center = _share_center(
                    spec.share_center_with,
                    sources,
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
                if spec.share_center_with not in sources:
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
        are expanded with natural sort order. For the dict (parts) form this
        flattens every part into a single combined list — callers that need
        per-env *combinations* should use ``resolved_background_plys`` instead.
        """
        if self.background_ply is None:
            return []
        if _is_dict_background(self.background_ply):
            entries: list[str] = []
            for value in self.background_ply.values():
                if isinstance(value, str):
                    entries.append(value)
                else:
                    entries.extend(value)
        else:
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
        if _is_dict_background(self.background_ply):
            return True
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

        Raises when ``background_ply`` is a list or parts dict; callers should
        use ``resolved_background_plys`` in that case.
        """
        if self.is_multi_background():
            raise ValueError(
                "background_ply is a list or dict; call "
                "resolved_background_plys() instead."
            )
        return _materialize_transformed_background_ply(
            self.background_ply,
            self.resolved_background_transform(),
        )

    def _per_ply_pose(
        self, bg_ply: str, default_pose: BackgroundPose | None
    ) -> BackgroundPose:
        """Resolve a per-PLY pose from ``background_transforms``.

        Key precedence (highest first):

        1. Per-PLY exact match — full path / ``str(Path)`` normalised path /
           file name / stem.
        2. ``default_pose`` (typically the part-name default in dict mode, or
           the singular ``background_transform``).
        3. Identity.
        """
        bg_path = Path(bg_ply)
        for key in (bg_ply, str(bg_path), bg_path.name, bg_path.stem):
            if key in self.background_transforms:
                return _normalize_background_pose(self.background_transforms[key])
        if default_pose is not None:
            return default_pose
        return (0.0, 0.0, 0.0), _IDENTITY_QUAT

    def _resolved_part_plys(self) -> Dict[str, list[str]]:
        """For the dict form of ``background_ply``: expand each part to a list
        of PLY paths, baking in per-PLY pose transforms.

        Per-PLY pose precedence inside a part:

        1. ``background_transforms[<full path | str(Path) | file name | stem>]``
           — single PLY override.
        2. ``background_transforms[<part name>]`` — applied to every PLY in
           that part (e.g. ``background_transforms.wall`` covers every
           ``wall*.ply`` matched by the part's glob).
        3. Singular ``background_transform`` — applied to every part PLY.
        4. Identity.

        Raises ``ValueError`` if ``background_ply`` is not a dict, or if any
        part expands to zero PLYs.
        """
        if not _is_dict_background(self.background_ply):
            raise ValueError(
                "_resolved_part_plys is only valid when background_ply is a dict"
            )
        global_default: BackgroundPose | None = None
        if self.background_transform is not None:
            global_default = _normalize_background_pose(self.background_transform)

        out: Dict[str, list[str]] = {}
        for part, value in self.background_ply.items():
            entries = [value] if isinstance(value, str) else list(value)
            part_default: BackgroundPose | None = global_default
            if part in self.background_transforms:
                part_default = _normalize_background_pose(
                    self.background_transforms[part]
                )
            part_plys: list[str] = []
            for entry in entries:
                for bg_ply in _expand_background_entry(entry):
                    pose = self._per_ply_pose(bg_ply, part_default)
                    materialized = _materialize_transformed_background_ply(bg_ply, pose)
                    if materialized is not None:
                        part_plys.append(materialized)
            if not part_plys:
                raise ValueError(f"background_ply part '{part}' expanded to zero PLYs")
            out[part] = part_plys
        return out

    def resolved_background_plys(
        self,
        *,
        max_combinations: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[str]:
        """Return materialized bg paths for ``background_ply``.

        For ``str`` / ``list`` / glob forms: one path per pool entry, with
        per-entry pose resolution (precedence: per-entry override in
        ``background_transforms`` → singular ``background_transform`` →
        identity). When ``background_transform_randomization`` matches an
        entry, a uniform per-axis position offset is sampled and added
        before baking, producing one randomly-perturbed PLY per pool entry.

        For the dict (parts) form: each part is independently expanded and
        per-PLY-transformed; combinations are sampled from the cartesian
        product of part lists (limited to ``max_combinations`` without
        replacement when given) and each combination is merged into a
        single PLY (cached on disk). When ``background_transform_randomization``
        is configured, ``max_combinations`` independent combos are drawn
        regardless of the pool size, and each combo gets its own per-part
        position offset baked in.
        """
        rng_local = rng if rng is not None else np.random.default_rng()

        if _is_dict_background(self.background_ply):
            if self.has_position_randomization():
                return self._resolve_dict_with_position_randomization(
                    max_combinations, rng_local
                )
            parts = self._resolved_part_plys()
            combos = _sample_combinations(
                list(parts.values()), max_combinations, rng_local
            )
            return [_merge_background_plys(list(combo)) for combo in combos]

        default_pose: BackgroundPose | None = None
        if self.background_transform is not None:
            default_pose = _normalize_background_pose(self.background_transform)

        out: list[str] = []
        for bg_ply in self._background_ply_list():
            pose = self._per_ply_pose(bg_ply, default_pose)
            if self.has_position_randomization():
                offset = self._sample_position_offset(
                    self._ply_key_candidates(bg_ply), rng_local
                )
                pose = (
                    (
                        pose[0][0] + offset[0],
                        pose[0][1] + offset[1],
                        pose[0][2] + offset[2],
                    ),
                    pose[1],
                )
            materialized = _materialize_transformed_background_ply(bg_ply, pose)
            if materialized is not None:
                out.append(materialized)
        return out

    def _resolve_dict_with_position_randomization(
        self,
        max_combinations: int | None,
        rng: np.random.Generator,
    ) -> list[str]:
        """Resolve dict-mode combinations preferring distinct file tuples
        before falling back to position randomization.

        Two-stage strategy:

        1. **Distinct file combos (no random offset).** Up to
           ``min(max_combinations, cartesian_product_size)`` PLY tuples
           are drawn from the cartesian product of the per-part pools
           via ``_sample_combinations`` (without replacement when
           batch size < total combos), and each tuple is baked using
           only the deterministic transforms from
           ``background_transforms`` / ``background_transform``.
        2. **Overflow with randomization.** If ``max_combinations`` still
           exceeds the cartesian product size, the remaining slots draw
           PLY tuples *with* replacement and apply a per-part random
           position offset (sampled via
           ``background_transform_randomization``) so the overflow envs
           remain visually distinct.

        Result: when the file pool can cover the batch, no randomization
        is applied at all (the configured ranges are silent). The ranges
        only kick in to differentiate envs beyond what the file pool can
        already supply.
        """
        # Expand each part to its raw PLY pool (no transforms baked yet).
        part_pools: Dict[str, list[str]] = {}
        for part, value in self.background_ply.items():
            entries = [value] if isinstance(value, str) else list(value)
            pool: list[str] = []
            for entry in entries:
                pool.extend(_expand_background_entry(entry))
            if not pool:
                raise ValueError(f"background_ply part '{part}' expanded to zero PLYs")
            part_pools[part] = pool

        global_default: BackgroundPose | None = None
        if self.background_transform is not None:
            global_default = _normalize_background_pose(self.background_transform)

        # Resolve part-name defaults once.
        part_defaults: Dict[str, BackgroundPose | None] = {}
        for part in part_pools:
            if part in self.background_transforms:
                part_defaults[part] = _normalize_background_pose(
                    self.background_transforms[part]
                )
            else:
                part_defaults[part] = global_default

        parts = list(part_pools.keys())
        sizes = [len(part_pools[p]) for p in parts]
        n_files = int(np.prod(sizes))
        # ``max_combinations=None`` means "no cap" — return all distinct file
        # combinations, no overflow needed.  Previously this defaulted to 1
        # which silently capped foreground_variant's BG pool to a single
        # combination.
        batch = max_combinations if max_combinations is not None else n_files
        n_distinct = min(batch, n_files)
        n_overflow = batch - n_distinct

        def _bake_combo(combo: Sequence[str], with_offset: bool) -> str:
            transformed_parts: list[str] = []
            for i, part in enumerate(parts):
                ply = combo[i]
                base_pose = self._per_ply_pose(ply, part_defaults[part])
                if with_offset:
                    offset = self._sample_position_offset(
                        self._ply_key_candidates(ply, part=part), rng
                    )
                    pose = (
                        (
                            base_pose[0][0] + offset[0],
                            base_pose[0][1] + offset[1],
                            base_pose[0][2] + offset[2],
                        ),
                        base_pose[1],
                    )
                else:
                    pose = base_pose
                materialized = _materialize_transformed_background_ply(ply, pose)
                if materialized is not None:
                    transformed_parts.append(materialized)
            return _merge_background_plys(transformed_parts)

        out: list[str] = []
        # Phase 1: distinct file combos, deterministic transforms only.
        if n_distinct > 0:
            distinct_combos = _sample_combinations(
                list(part_pools.values()), n_distinct, rng
            )
            for combo in distinct_combos:
                out.append(_bake_combo(combo, with_offset=False))

        # Phase 2: overflow → file pool exhausted, differentiate via
        # randomized offsets (sampled with replacement from the same pools).
        for _ in range(n_overflow):
            combo = tuple(
                part_pools[parts[i]][int(rng.integers(sizes[i]))]
                for i in range(len(parts))
            )
            out.append(_bake_combo(combo, with_offset=True))

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
    warmup: bool = False
    """Whether to perform a warmup (reset and capture the first observation) on environment initialization. Can help reduce outliers in the first few frames of the first episode, which may be important for some use cases."""

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
                    "be multi-valued (list, glob, or parts dict). A single "
                    "background would produce identical observations across "
                    "the batch."
                )
        return self


class GSUnifiedMujocoEnv(UnifiedMujocoEnv):
    """``UnifiedMujocoEnv`` that replaces native RGB with Gaussian Splatting."""

    _GS_MASK_ALPHA_THRESHOLD = 0.5
    _GS_MASK_DEPTH_EPS = 0.01

    def __init__(self, config: GSEnvConfig) -> None:
        super().__init__(config)
        self.config: GSEnvConfig
        self._pending_gs_config: GaussianRenderConfig | None = None
        self._bg_rng = np.random.default_rng()
        self._setup_gs_render_state()
        if config.warmup:
            self.get_logger().info("Performing GS renderer warmup...")
            self.reset()
            self.capture_observation()
            self.get_logger().info("GS renderer warmup complete.")

    def _setup_gs_render_state(self) -> None:
        """(Re)build all GS renderers from ``self.config.gaussian_render``.

        Old renderers are dereferenced before new ones are constructed so the
        GPU memory they hold is reclaimed by GC before the new allocation,
        avoiding a transient 2× VRAM peak.
        """
        gs_cfg = self.config.gaussian_render
        # Drop old GPU-backed renderers first (no explicit close on these
        # classes — relies on GC). Force a collection so device memory is
        # actually freed before the new allocations below.
        self._gs_renderer: GSRendererMuJoCo | None = None
        self._bg_gs_renderer: MjxBatchSplatRenderer | None = None
        self._gs_renderers_list: list[GSRendererMuJoCo] = []
        self._bg_gs_renderers_list: list[MjxBatchSplatRenderer | None] = []
        self._bg_source_plys: list[str] = []
        self._active_bg_idx: int = 0
        self._fg_gs_renderer = None
        self._gs_mask_renderers = {}
        gc.collect()

        self._gs_background_source_ply = gs_cfg.background_ply
        self._is_multi_bg = gs_cfg.is_multi_background()
        if bool(gs_cfg.foreground_variant):
            raise ValueError(
                "foreground_variant=True is only supported on "
                "BatchedGSUnifiedMujocoEnv (batched class); the single-env "
                "GSUnifiedMujocoEnv has no batch dimension to amortize FG "
                "renderer reuse across."
            )
        self._gs_body_gaussians = gs_cfg.resolved_body_gaussians()
        self._fg_gs_renderer = MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians=dict(self._gs_body_gaussians),
                background_ply=None,
                minibatch=gs_cfg.minibatch,
            ),
            self.model,
        )
        if self._is_multi_bg:
            # Single-env class: build all available combinations so that
            # ``randomize_background_on_reset`` can meaningfully randomize.
            self._bg_source_plys = gs_cfg.resolved_background_plys(
                max_combinations=None,
                rng=self._bg_rng,
            )
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

    def update_gaussian_render(
        self,
        config: GaussianRenderConfig | None = None,
        **kwargs,
    ) -> None:
        """Stage a new Gaussian render config; takes effect on next ``reset()``.

        Pass a full ``GaussianRenderConfig`` via ``config``, kwargs to patch
        the current config's fields, or both (kwargs override).
        """
        base = config if config is not None else self.config.gaussian_render
        if not isinstance(base, GaussianRenderConfig):
            base = GaussianRenderConfig.model_validate(base)
        if kwargs:
            base = base.model_copy(update=kwargs)
        self._pending_gs_config = base

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
        if self._pending_gs_config is not None:
            object.__setattr__(self.config, "gaussian_render", self._pending_gs_config)
            self._pending_gs_config = None
            self._setup_gs_render_state()
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
                "is a list or parts dict; use background_transforms in the "
                "config to set per-background (or per-part) poses."
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
        self._pending_gs_config = None
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
            # the canonical batch adapter routes step-like calls to the sole
            # physical replica to avoid N× work.
            self.envs = [self.envs[0]] * self._virtual_batch_size
        else:
            super().__init__(config, **kwargs)

        self._pending_gs_config: GaussianRenderConfig | None = None
        self._bg_rng = np.random.default_rng()

        # Cache camera specs from env[0] (homogeneous)
        self._camera_specs = self.envs[0]._camera_specs
        self._camera_ids = self.envs[0]._camera_ids

        # Background cache: (cam_ids_tuple, w, h) → (bg_rgb, bg_depth) tensors
        self._bg_cache: dict[
            tuple[tuple[int, ...], int, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        self._setup_gs_render_state()

        if config.warmup:
            self.get_logger().info("Performing GS renderer warmup...")
            self.reset()
            self.capture_observation()
            self.get_logger().info("GS renderer warmup complete.")

    def _setup_gs_render_state(self) -> None:
        """(Re)build all GS renderers from ``self.config.gaussian_render``.

        Old renderers are dereferenced before new ones are constructed so the
        GPU memory they hold is reclaimed by GC before the new allocation,
        avoiding a transient 2× VRAM peak.
        """
        gs_cfg = self.config.gaussian_render
        # Drop old GPU-backed renderers first (no explicit close on these
        # classes — relies on GC). Force a collection so device memory is
        # actually freed before the new allocations below.
        self._fg_gs_renderer = None
        self._bg_gs_renderers: list[MjxBatchSplatRenderer] = []
        self._bg_source_plys: list[str] = []
        self._env_bg_idx: np.ndarray = np.zeros(self.batch_size, dtype=np.int64)
        self._bg_cache.clear()
        self._gs_mask_renderers = {}
        # Per-variant lazy caches used in foreground_variant mode. Always
        # initialised (cleared on each setup) so accessor helpers can read
        # them unconditionally.
        self._fg_renderer_cache: Dict[int, MjxBatchSplatRenderer] = {}
        self._mask_renderer_cache: Dict[int, Dict[str, MjxBatchSplatRenderer]] = {}
        self._bg_renderer_cache: Dict[int, MjxBatchSplatRenderer] = {}
        self._combo_cursor: _FGBGCombinationCursor | None = None
        self._active_fg_idx: int = 0
        gc.collect()

        self._gs_background_source = gs_cfg.background_ply
        self._is_multi_bg = gs_cfg.is_multi_background()
        self._foreground_variant_mode = bool(gs_cfg.foreground_variant)

        if self._foreground_variant_mode:
            self._setup_foreground_variant_mode(gs_cfg)
            return

        self._gs_body_gaussians = gs_cfg.resolved_body_gaussians()

        self._fg_gs_renderer = MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians=dict(self._gs_body_gaussians),
                background_ply=None,
                minibatch=gs_cfg.minibatch,
            ),
            self.envs[0].model,
        )

        self._gs_mask_renderers = self._build_shared_mask_renderers(
            dict(self._gs_body_gaussians)
        )

        if self._is_multi_bg:
            # Batched env: only materialize as many combinations as we have
            # envs (cartesian-product sampling without replacement when the
            # full product is larger than ``batch_size``).
            self._bg_source_plys = gs_cfg.resolved_background_plys(
                max_combinations=self.batch_size,
                rng=self._bg_rng,
            )
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
        self.get_logger().info(
            f"GS renderer initialised with {n_bodies} body gaussian(s){bg_str}"
        )

    def _setup_foreground_variant_mode(self, gs_cfg: GaussianRenderConfig) -> None:
        """Initialise FG-grouped round-robin combination sampling.

        Resolves the FG variant count ``M`` (cartesian product over
        ``body_gaussians`` list values) and the BG pool size ``N``
        (existing ``resolved_background_plys`` resolution, no
        ``max_combinations`` cap since BG renderers are built lazily).
        Validates ``N >= batch_size`` so each batch can use distinct
        backgrounds without replacement, then primes the
        ``_FGBGCombinationCursor`` and selects the first combination
        through ``_advance_combination_cursor``.
        """
        if not self._is_multi_bg:
            raise ValueError(
                "foreground_variant=True requires a multi-valued "
                "background_ply (list, glob, or parts dict)"
            )
        if not gs_cfg.body_gaussians:
            raise ValueError(
                "foreground_variant=True requires non-empty body_gaussians"
            )

        # FG variant count from body_gaussians cartesian product. Single-string
        # entries contribute factor 1 (constant across variants).
        self._fg_variant_count = _fg_variant_count(gs_cfg.body_gaussians)

        # BG pool: enumerate the FULL multi-background space, no batch cap.
        # ``_FGBGCombinationCursor`` cycles through every BG before
        # advancing FG, so we want the full N renderers available.
        self._bg_source_plys = gs_cfg.resolved_background_plys(
            max_combinations=None,
            rng=self._bg_rng,
        )
        n_bg = len(self._bg_source_plys)
        if n_bg < self.batch_size:
            raise ValueError(
                f"foreground_variant=True requires the background pool to "
                f"have at least batch_size entries; got {n_bg} backgrounds "
                f"vs batch_size={self.batch_size}"
            )

        self._combo_cursor = _FGBGCombinationCursor(
            num_fg=self._fg_variant_count,
            num_bg=n_bg,
            batch_size=self.batch_size,
            rng=self._bg_rng,
        )

        # Cache the first variant's resolved body_gaussians so existing
        # accessors (mask-name lookup, log messages) work.
        self._gs_body_gaussians = gs_cfg.resolved_body_gaussians(variant_idx=0)

        # First combination: builds FG/mask renderers for variant 0 and BG
        # renderers for the chosen batch on demand.
        self._advance_combination_cursor()

        reset_mode = (
            "randomized FG×BG combo on reset"
            if gs_cfg.randomize_background_on_reset
            else "fixed FG×BG combo after initial assignment"
        )
        self.get_logger().info(
            f"GS renderer initialised with foreground_variant mode: "
            f"M={self._fg_variant_count} FG variant(s) × N={n_bg} background(s) "
            f"({reset_mode})"
        )

    def _get_fg_renderer(self, fg_idx: int) -> MjxBatchSplatRenderer:
        """Lazy-build the FG splat renderer for ``fg_idx``; cache for later."""
        cached = self._fg_renderer_cache.get(fg_idx)
        if cached is not None:
            return cached
        gs_cfg = self.config.gaussian_render
        body_gaussians = gs_cfg.resolved_body_gaussians(variant_idx=fg_idx)
        renderer = MjxBatchSplatRenderer(
            BatchSplatConfig(
                body_gaussians=dict(body_gaussians),
                background_ply=None,
                minibatch=gs_cfg.minibatch,
            ),
            self.envs[0].model,
        )
        self._fg_renderer_cache[fg_idx] = renderer
        return renderer

    def _get_mask_renderers_for_variant(
        self, fg_idx: int
    ) -> Dict[str, MjxBatchSplatRenderer]:
        """Lazy-build per-object mask renderers for FG variant ``fg_idx``."""
        cached = self._mask_renderer_cache.get(fg_idx)
        if cached is not None:
            return cached
        gs_cfg = self.config.gaussian_render
        body_gaussians = gs_cfg.resolved_body_gaussians(variant_idx=fg_idx)
        renderers = self._build_shared_mask_renderers(dict(body_gaussians))
        self._mask_renderer_cache[fg_idx] = renderers
        return renderers

    def _lookup_bg_renderer(self, bg_idx: int) -> MjxBatchSplatRenderer:
        """Return the BG renderer for ``bg_idx``, building it lazily in
        foreground_variant mode and indexing the eager list otherwise."""
        if self._foreground_variant_mode:
            cached = self._bg_renderer_cache.get(int(bg_idx))
            if cached is not None:
                return cached
            renderer = self._make_bg_renderer(self._bg_source_plys[int(bg_idx)])
            self._bg_renderer_cache[int(bg_idx)] = renderer
            return renderer
        return self._bg_gs_renderers[int(bg_idx)]

    def _advance_combination_cursor(self) -> tuple[int, np.ndarray]:
        """Pick the next ``(fg_idx, bg_idxs)`` from ``_combo_cursor``, swap
        the active FG/mask renderers, update ``_env_bg_idx``, and pre-build
        the BG renderers for the chosen batch so the next ``render`` does
        not pay JIT cost mid-call.

        Returns ``(fg_idx, env_bg_idx)`` for tests / logging.
        """
        assert self._combo_cursor is not None, (
            "_advance_combination_cursor called outside foreground_variant mode"
        )
        fg_idx, bg_idxs = self._combo_cursor.next_batch()
        self._active_fg_idx = fg_idx
        self._env_bg_idx = np.asarray(bg_idxs, dtype=np.int64)
        self._fg_gs_renderer = self._get_fg_renderer(fg_idx)
        self._gs_mask_renderers = self._get_mask_renderers_for_variant(fg_idx)
        # Pre-build the BG renderers used by this batch.
        for bg_idx in bg_idxs:
            self._lookup_bg_renderer(bg_idx)
        # Cached bg tensors are keyed by the prior env→bg mapping.
        self._bg_cache.clear()
        return fg_idx, self._env_bg_idx

    def update_gaussian_render(
        self,
        config: GaussianRenderConfig | None = None,
        **kwargs,
    ) -> None:
        """Stage a new Gaussian render config; takes effect on next ``reset()``.

        Pass a full ``GaussianRenderConfig`` via ``config``, kwargs to patch
        the current config's fields, or both (kwargs override).
        """
        base = config if config is not None else self.config.gaussian_render
        if not isinstance(base, GaussianRenderConfig):
            base = GaussianRenderConfig.model_validate(base)
        if kwargs:
            base = base.model_copy(update=kwargs)
        self._pending_gs_config = base

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
            num_backgrounds=len(self._bg_source_plys),
            rng=self._bg_rng,
        )
        self._bg_cache.clear()
        return self._env_bg_idx

    def reset(self, env_mask: np.ndarray | None = None) -> None:
        if self._pending_gs_config is not None:
            object.__setattr__(self.config, "gaussian_render", self._pending_gs_config)
            self._pending_gs_config = None
            # import time
            # start = time.perf_counter()
            self._setup_gs_render_state()
            # print(f"GS render state setup took {time.perf_counter() - start:.3f} seconds")

        # ``BatchedUnifiedMujocoEnv`` routes this through the canonical batch
        # adapter.  In shared-physics mode the adapter validates ``env_mask``
        # and invokes the sole physical replica once when any logical row is
        # active; in replicated mode it preserves per-row dispatch.
        super().reset(env_mask)
        if (
            self._is_multi_bg
            and self.config.gaussian_render.randomize_background_on_reset
        ):
            if self._foreground_variant_mode:
                self._advance_combination_cursor()
            else:
                self._randomize_env_bg_assignment()

    def refresh_viewer(self) -> None:
        if self._share_physics:
            self.envs[0].refresh_viewer()
        else:
            super().refresh_viewer()

    def is_updated(self) -> np.ndarray:
        # The canonical adapter performs one stateful probe for shared physics
        # and broadcasts its result to all logical rows.
        return super().is_updated()

    def set_background_transform(
        self, pose: BackgroundPose | list[float]
    ) -> BackgroundPose:
        if self._is_multi_bg:
            raise ValueError(
                "set_background_transform is not supported when background_ply "
                "is a list or parts dict; use background_transforms in the "
                "config to set per-background (or per-part) poses."
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
        # The canonical adapter stacks replicated observations or broadcasts a
        # single shared-physics observation exactly once.
        obs = super().capture_observation()
        self._inject_batched_gs_renders(obs)
        return obs

    def _broadcast_single_obs(
        self, obs_one: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Compatibility wrapper for the canonical observation adapter."""
        return self._batch_adapter().broadcast_observation(obs_one)

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
        if not self._bg_source_plys:
            return None
        cache_key = (tuple(cam_ids), width, height)
        if use_cache and cache_key in self._bg_cache:
            return self._bg_cache[cache_key]

        if len(self._bg_source_plys) == 1:
            bg_rend = self._lookup_bg_renderer(0)
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
            bg_rend = self._lookup_bg_renderer(int(bg_idx))
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
            if self._bg_source_plys:
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
            bg_rend = self._lookup_bg_renderer(int(bg_idx))
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
        self._fg_renderer_cache = {}
        self._mask_renderer_cache = {}
        self._bg_renderer_cache = {}
        self._combo_cursor = None
        self._bg_cache.clear()
        self._pending_gs_config = None
        super().close()

    def get_logger(self):
        return logging.getLogger(self.__class__.__name__)
