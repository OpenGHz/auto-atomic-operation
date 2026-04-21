# Per-Body PLY Transforms & Mirrors

`env.gaussian_render` can bake a rigid transform and/or a reflection into any
per-body Gaussian PLY at load time. This is useful when the **physical**
MuJoCo body pose is already correct but the **reconstructed GS asset** needs
an extra local correction — for example:

- a PLY was reconstructed in a frame that doesn't match the MuJoCo body frame
- a scene layout mirrors an existing captured object (e.g. back-side door
  using the front-side asset)

Both operations rewrite the Gaussians in-memory once, then cache the result
under `.cache/gs_body_transforms/` or `.cache/gs_body_mirrors/` so subsequent
runs skip the work.

Relevant implementation:

- `auto_atom/basis/mjc/gs_mujoco_env.py`
  - `BodyTransformSpec`, `BodyMirrorSpec`, `GaussianRenderConfig.resolved_body_gaussians()`
- Tests: `tests/test_gs_body_transforms.py`

## Configuration

Both fields live under `env.gaussian_render` and are keyed by entries in
`body_gaussians`. Unknown keys raise a `ValueError` at config resolution.

```yaml
env:
  gaussian_render:
    body_gaussians:
      door_body: ${assets_dir}/gs/objects/door/door.ply
      handle_gs_frame: ${assets_dir}/gs/objects/door/handle.ply

    body_transforms:
      door_body:
        position: [0.0, 0.0, 0.01]     # optional
        orientation: [0, 0, 0, 1]      # xyzw or [roll, pitch, yaw]
        center: [0.033, 0.007, 1.589]  # optional pivot for the rotation

    body_mirrors:
      door_body:
        axis: [1, 0, 0]                # plane normal in PLY-local coords
      handle_gs_frame:
        axis: [1, 0, 0]
        share_center_with: door_body   # keep paired objects aligned
```

When both fields are set for the same body, **`body_transforms` is applied
first, then `body_mirrors`.** The mirror path can additionally bake a
post-reflection rigid transform (see [Rotoreflections](#rotoreflections))
for the common "mirror then rotate around the mirror plane" case.

## `body_transforms`

Rigid transform applied in the PLY's local GS coordinates.

| Field               | Type            | Default            | Description |
|---------------------|-----------------|--------------------|-------------|
| `position`          | `[x, y, z]`     | `null` (no shift)  | Translation in PLY-local coords (metres). |
| `orientation`       | `[x, y, z, w]` (quat) or `[r, p, y]` (Euler) | `null` (no rotation) | Applied after translation-free pivot logic. |
| `center`            | `[x, y, z]`     | `null` (origin)    | Explicit pivot point for the rotation. |
| `share_center_with` | body key        | `null`             | Reuse another body's pivot — see [Shared centers](#shared-centers). |

Rotation is applied about the pivot when `center` (or `share_center_with`)
is set: `p' = R · (p − c) + c + t`. With no center, translation is added
to every point unchanged and rotation acts about the PLY origin.

## `body_mirrors`

Reflect the PLY across a plane. Mirrors positions, rotations, and the
SH band-1 coefficients per channel; higher SH bands are left untouched
(their contribution is small and dominated by SH noise).

| Field               | Type            | Default                  | Description |
|---------------------|-----------------|--------------------------|-------------|
| `axis`              | `[x, y, z]`     | — (one of axis / body_quat required) | Plane normal in PLY-local (GS) coords. Use this when the PLY is pre-rotated and you can reason about the mirror direction in the PLY frame. |
| `body_quat`         | `[w, x, y, z]`  | `null`                   | MuJoCo-convention body quaternion. When set, the mirror axis is derived as `gs_axis = R(body_quat)ᵀ · body_axis`. |
| `body_axis`         | `[x, y, z]`     | `[1, 0, 0]`              | Body-frame direction to mirror along (only read when `body_quat` is set). |
| `center`            | `[x, y, z]`     | PLY centroid             | Explicit mirror-plane center in GS coords. Also doubles as the rotation pivot for the optional post-reflection transform. |
| `share_center_with` | body key        | `null`                   | Reuse another body's center. See [Shared centers](#shared-centers). |
| `position`          | `[x, y, z]`     | `null`                   | Optional post-reflection translation (PLY-local coords). |
| `orientation`       | `[x, y, z, w]` or `[r, p, y]` | `null`     | Optional post-reflection rotation about the mirror center. |

### Choosing between `axis` and `body_quat` + `body_axis`

- **`axis`** is simplest when the PLY is pre-rotated and the body's MuJoCo
  quaternion is identity (or nearly so). You can reason about "mirror along
  X" directly in the PLY frame.
- **`body_quat` + `body_axis`** lets you say "mirror along the body's local
  X axis" even when the PLY is not pre-rotated. The resolved axis is
  `R(body_quat)ᵀ · body_axis`, evaluated in PLY-local coords at load time.

Exactly one of the two must be set. Zero-length axes raise a `ValueError`.

## Shared centers

When two PLYs represent parts of the same physical object (e.g. door panel
+ handle), they must pivot/mirror around a shared point to stay aligned.
`share_center_with` references another entry by its `body_gaussians` key:

```yaml
body_mirrors:
  door_body:
    axis: [1, 0, 0]
    center: [0.033, 0.007, 1.589]
  handle_gs_frame:
    axis: [1, 0, 0]
    share_center_with: door_body
```

Resolution rules:

1. If the referenced entry has an **explicit** `center` set (same spec
   dict — `body_transforms` sharing reads from `body_transforms`, mirrors
   from `body_mirrors`), that center is reused.
2. Otherwise the referenced PLY's centroid is computed and reused.

Setting both `center` and `share_center_with` on the same entry is allowed
but `center` takes precedence.

Invalid references (pointing to a body that's not in `body_gaussians`)
raise a `ValueError` at config resolution.

## Rotoreflections

A pure reflection cannot represent "mirror, then rotate about the mirror
plane". This combination (a rotoreflection) shows up in practice when
reusing an asset for a back-side scene layout: the door must both be
mirrored and rotated ~180° around the door frame's vertical. The
`position` and `orientation` fields on `BodyMirrorSpec` bake a rigid
transform in after the reflection, using the mirror `center` as the
rotation pivot:

```yaml
body_mirrors:
  door_body:
    body_quat: [0.0235, -0.0148, 0.7161, -0.6974]
    body_axis: [1, 0, 0]
    orientation: [0.0131, 0.9986, 0.0261, 0.0436]   # xyzw post-reflection rotation
  handle_gs_frame:
    body_quat: [0.0235, -0.0148, 0.7161, -0.6974]
    body_axis: [1, 0, 0]
    orientation: [0.0131, 0.9986, 0.0261, 0.0436]
    share_center_with: door_body
```

This is exactly the pattern used by
`aao_configs/open_door_airbot_play_back_gs.yaml` to reuse the front-side
door PLYs on the back side of the scene.

## Caching

Resolved PLYs are cached per (source path, operation, parameters):

- `body_transforms`: `.cache/gs_body_transforms/<stem>__body_xform_<hash>.ply`
  — keyed by `(path, pose, center)`.
- `body_mirrors`: `.cache/gs_body_mirrors/<stem>__mirror_<hash>.ply`
  — keyed by `(path, axis, center, post_pose)`.

Delete the cache directory to force re-materialisation after changing the
source PLY without changing the transform parameters.

## Interaction with other features

- Applied at **load time**, before any `BatchSplatConfig` is constructed,
  so the transformed PLYs are visible to the foreground renderer, the
  background renderer, and per-object mask renderers alike.
- Independent from the runtime body pose set by MuJoCo — `mj_data.xquat`
  continues to rotate the transformed PLY as usual at render time.
- Plays nicely with multi-background setups: `body_transforms` /
  `body_mirrors` affect only the foreground per-body PLYs; the per-env
  background pool is unrelated.
