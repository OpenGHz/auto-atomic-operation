# Keeping Gaussian Splatting and Non-GS Rendering Consistent

This guide explains how to author MuJoCo scene XMLs and Gaussian Splatting (GS) assets so that both the standard MuJoCo renderer and the GS renderer produce geometrically consistent results (objects appear horizontal, correctly placed, and without color/lighting distortion).

## Background: Why Body Euler Rotation Breaks GS Rendering

The GS renderer (`GSRendererMuJoCo`) applies each body's world pose (`mj_data.xpos`, `mj_data.xquat`) to transform PLY Gaussians at render time via `update_gaussian_properties`. This function rotates:

- **Gaussian positions** (xyz) — ✅ rotated correctly
- **Gaussian orientations** (rot) — ✅ rotated correctly
- **Spherical Harmonics (SH) coefficients** — ❌ **NOT rotated**

SH coefficients encode view-dependent color (specular highlights, shading). When the body has a non-identity rotation at render time, the Gaussian positions and orientations are rotated but the SH frame is not, causing visible color distortion (specular highlights appear in wrong directions, surface appears "twisted").

**Conclusion**: GS reference bodies must have **identity orientation** at render time.

## The Coordinate Frame Problem

PLY files captured in a Franka 3DGS world frame use a coordinate convention different from MuJoCo's Z-up world frame. Specifically, objects in the Franka frame are tilted relative to the MuJoCo frame by a rotation:

```
R = Rx(55°) ⊗ Ry(3°)   (approximately, intrinsic XYZ)
```

For `press_three_buttons`, this corresponds to MuJoCo `euler="0.959931089 0.052359878 0"` (radians, intrinsic XYZ sequence). If you place a PLY file at a body with identity orientation, the Gaussians appear tilted in MuJoCo world.

## The Solution: Pre-Rotate PLY Files

Instead of applying body euler at render time (which breaks SH), **pre-bake the rotation into the PLY file** using `transform_gaussian`. This rotates xyz, rot, AND SH coefficients together using `e3nn`, so the result is correct in every respect.

After pre-rotation, the PLY is stored in MuJoCo world frame. The GS reference body uses identity orientation, and no runtime rotation is applied — SH remains undistorted.

### Step 1: Convert MuJoCo Euler to xyzw Quaternion

MuJoCo `euler` uses **intrinsic XYZ** convention: `euler="α β γ"` → `q = qx(α) ⊗ qy(β) ⊗ qz(γ)`.

```python
import numpy as np

def euler_intrinsic_xyz_to_quat_xyzw(ax, ay, az):
    """MuJoCo euler (intrinsic XYZ) → xyzw quaternion."""
    cx, sx = np.cos(ax/2), np.sin(ax/2)
    cy, sy = np.cos(ay/2), np.sin(ay/2)
    cz, sz = np.cos(az/2), np.sin(az/2)
    # Compose qx ⊗ qy ⊗ qz
    qx = np.array([sx, 0, 0, cx])  # xyzw
    qy = np.array([0, sy, 0, cy])
    qz = np.array([0, 0, sz, cz])
    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz,
        ])
    return qmul(qmul(qx, qy), qz)

# For euler="0.959931089 0.052359878 0"
q = euler_intrinsic_xyz_to_quat_xyzw(0.959931089, 0.052359878, 0.0)
# → xyzw ≈ (0.46159, 0.02322, 0.01209, 0.88671)
```

You can verify by cross-checking against the Franka scene keyframe quaternion (wxyz → xyzw swap).

### Step 2: Pre-Rotate the PLY File

Use the provided convenience script `examples/preprocess_gs_ply.py`, which wraps `transform_gaussian` and handles both single-file and batch-directory modes:

```bash
# Rotate all PLYs in a source directory using MuJoCo euler angles (radians)
python examples/preprocess_gs_ply.py \
    third_party/.../3dgs/ \
    -o assets/gs/scenes/my_task/ \
    --euler 0.959931089 0.052359878 0.0

# Or supply the xyzw quaternion directly (skips the euler conversion step)
python examples/preprocess_gs_ply.py \
    third_party/.../button_blue.ply \
    -o assets/gs/scenes/my_task/button_blue.ply \
    -r 0.46159 0.02322 0.01209 0.88671

# Dry run: preview which files would be written without touching disk
python examples/preprocess_gs_ply.py \
    third_party/.../3dgs/ \
    -o assets/gs/scenes/my_task/ \
    --euler 0.959931089 0.052359878 0.0 \
    --dry-run
```

The script accepts either `--euler ax ay az` (MuJoCo intrinsic XYZ, radians) or `-r x y z w` (xyzw quaternion). When `--euler` is used, the converted quaternion is printed so you can verify it. Internally it calls `transform_gaussian`, which rotates positions, orientations, and SH coefficients together.

For `press_three_buttons`, the pre-rotated files are stored in `assets/gs/scenes/press_three_buttons/` and were generated with:

```bash
python examples/preprocess_gs_ply.py \
    third_party/press_da_button/gs_playground/models/tasks/table30/_01_press_three_buttons/3dgs/ \
    -o assets/gs/scenes/press_three_buttons/ \
    --euler 0.959931089 0.052359878 0.0
```

### Step 3: Store Pre-Rotated Files

Place pre-rotated PLY files in `assets/gs/scenes/<task>/` (alongside original scene XMLs and meshes). Keep originals in the third-party source directory (`third_party/`).

## Scene XML Pattern (Two-Layer Body Structure)

For objects that need separate GS-reference and physics bodies:

```xml
<!-- OUTER body: GS reference — identity orientation, world pos from PLY capture frame -->
<body name="button_blue_gs" pos="0.443 0.0197 0.0812">

  <!-- INNER body: physics/YAML target — identity orientation, z-offset only -->
  <body name="button_blue" pos="0 0 0.012">

    <!-- Collision: invisible cylinder, no density override -->
    <geom name="button_blue_col" type="cylinder" size="0.040 0.02"
          mass="0.1" rgba="0 0 0 0" contype="1" conaffinity="1"/>

    <!-- Visual mesh: euler on GEOM (not body) to orient the OBJ mesh in world frame -->
    <geom name="button_blue_vis" type="mesh" mesh="blue_button_mesh" material="blue_button_mat"
          euler="0.959931089 0.052359878 0" density="0" contype="0" conaffinity="0" group="1"/>

    <site name="button_blue_site" pos="0 0 0" size="0.004" rgba="0 0 1 0"/>
  </body>
</body>
```

Key rules:
- **Outer body (`button_*_gs`)**: identity orientation. `body_gaussians` in YAML maps this body to the pre-rotated `.ply`.
- **Inner body (`button_*`)**: identity orientation, position offset only. YAML stages target this body (`object: button_blue`).
- **Visual mesh geom**: `euler` is applied to the OBJ mesh geometry, NOT to the body. This is necessary when OBJ files were exported in the 3DGS coordinate frame and need reorientation for the MuJoCo Z-up world. `density="0"` prevents the mesh volume from contributing mass.
- **Collision geom**: simple primitive (box/cylinder), `rgba="0 0 0 0"` (invisible), explicit `mass` to avoid inertia-from-volume issues.

> **Why euler on geom, not body?**
> OBJ meshes are plain geometry — their `euler` only reorients the mesh surface for rendering. It has no effect on GS rendering (GS uses body pose, not geom pose). Putting euler on the geom keeps the body at identity while still displaying the non-GS mesh correctly.

## YAML Configuration Pattern

```yaml
gs_dir: assets/gs/scenes/press_three_buttons   # pre-rotated PLYs
bg_dir: third_party/.../3dgs                   # background (no pre-rotation needed)

env:
  env:
    config:
      gaussian_render:
        background_ply: ${bg_dir}/background.ply
        body_gaussians:
          button_blue_gs: ${gs_dir}/button_blue.ply    # outer body → pre-rotated PLY
          button_green_gs: ${gs_dir}/button_green.ply
          button_pink_gs: ${gs_dir}/button_pink.ply
```

The `body_gaussians` keys must match the **outer** (`*_gs`) body names in the XML.

## Visual Comparison Tool

Use `examples/compare_gs_render.py` to render the first frame of a GS scene and compare GS vs native MuJoCo images side-by-side per camera.

```bash
# Default config: press_three_buttons_table30_gs
python examples/compare_gs_render.py

# Any other GS scene config (Hydra --config-name override)
python examples/compare_gs_render.py --config-name cup_on_coaster_gs
python examples/compare_gs_render.py --config-name stack_color_blocks_gs

# Display result interactively (in addition to saving)
python examples/compare_gs_render.py show=true
```

Must be run from the project root. Output is saved to `outputs/compare_<config>_<timestamp>.png`.

Each row in the output image corresponds to one camera, with the GS render on the left and the native MuJoCo rasteriser on the right. This is useful for verifying that GS and non-GS renders are geometrically aligned after following the pre-rotation workflow above.

## Summary Checklist

When adding a new GS object to a scene:

1. **Identify the frame mismatch** between the PLY capture frame and MuJoCo world frame.
2. **Convert the corrective rotation** to an xyzw quaternion (using MuJoCo intrinsic XYZ euler convention).
3. **Pre-rotate the PLY file** with `transform_gs_model` using that quaternion.
4. **Store pre-rotated PLY** in `assets/gs/scenes/<task>/`.
5. **Use identity orientation** on the GS reference body in the scene XML.
6. **Put visual mesh `euler` on the geom** (not the body) for non-GS rendering.
7. **Map the outer (`*_gs`) body** in `body_gaussians` YAML to the pre-rotated PLY.

Following this pattern ensures both GS and non-GS renders are geometrically consistent without SH color distortion.

---

## Appendix: Why Third-Party Code Works Without PLY Pre-Processing

The third-party `gs_playground` codebase (e.g., `_01_press_three_buttons.py`) does NOT pre-rotate PLY files, yet produces visually correct results. This section explains the equivalence and the subtle difference.

### Third-Party Architecture

In `table30_01_press_three_buttons.xml`, each button has a **single outer body with the euler rotation and a freejoint**:

```xml
<body name="button_blue" pos="0.443 0.0197 0.0812" euler="0.959931089 0.052359878 0">
  <joint type="free" frictionloss="0.0001"/>
  ...
</body>
```

`task_gaussians()` maps **this outer body** (with R_euler) to the original PLY:

```python
TASK_GAUSSIANS = {
    "button_blue": ASSETS_TASK_DIR / "button_blue.ply",  # outer body, has R_euler
}
```

The PLY xyz positions are stored in the **button's body-local coordinate frame**. At render time, `batch_update_gaussians` applies:

```
xyz_world = R_euler @ xyz_local + body_world_pos   ✓ geometrically correct
rot_world = R_euler ⊗ rot_local                    ✓ geometrically correct
sh         = sh_template  (not rotated)             ⚠ theoretically wrong, but imperceptible
```

The geometry is correct because `R_euler` correctly maps from body-local to world frame. The SH is not rotated, so view-dependent color has a slight error — but buttons are low-reflectance diffuse objects (solid blue/green/pink), and the SH view-dependence is negligible in practice.

### Why the Original `demo_gs.xml` Had Distortion

The original `press_three_buttons_gs.yaml` had:

```yaml
body_gaussians:
  button_blue: ...    # INNER body (identity orientation)
```

The PLY data was designed for the outer body's R_euler frame, but it was being rendered relative to the **inner** body (identity). This caused:

```
xyz_world = I @ xyz_R_euler_local + inner_body_world_pos   → wrong positions/orientation
```

The Gaussians appeared tilted and misaligned ("扭曲"/distorted) because the coordinate frame of the PLY data did not match the identity rotation of the inner body.

### Two Approaches Compared

| | Third-Party Approach | Our Approach |
|---|---|---|
| Body orientation | R_euler (~55°) | identity |
| PLY coordinate frame | body-local (tilted) | pre-rotated to MuJoCo world frame |
| Geometry | ✓ correct | ✓ correct |
| SH rotation | ✗ not rotated (minor error) | ✓ pre-rotated with e3nn |
| Suitable for | diffuse/matte objects | all objects incl. specular |
| Two-layer body structure | not needed | needed (GS body + physics body) |

### When Each Approach Is Appropriate

**Use the third-party approach (body euler + original PLY)** when:
- The scene XML has a single body per object (no physics/GS split needed)
- Objects are predominantly diffuse with weak SH view-dependence
- You want to reuse PLY files directly without a pre-processing step

**Use the pre-rotation approach (identity body + pre-rotated PLY)** when:
- You need a two-layer body structure (separate GS reference body and physics/YAML target body)
- Objects have specular highlights or strong view-dependent appearance
- SH correctness is important for data collection or visual fidelity

### Key Rule: Always Match PLY Frame to Body Orientation

The fundamental invariant is:

> **PLY local-frame positions must be expressed in the same coordinate frame as the body's local frame at render time.**

- If the body has R_euler at render time → PLY must be in R_euler body-local frame (no pre-processing)
- If the body has identity at render time → PLY must be pre-rotated to identity/world-aligned frame
- Mismatching these two (e.g., R_euler PLY + identity body) causes geometric distortion
