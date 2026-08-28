# Scene Initialization & Randomization

auto_atom supports flexible scene initialization via YAML configuration:
per-entity pose overrides, per-joint position overrides, and pose
randomization applied at each `reset()`.  Together these let you set up
initial conditions and evaluate task robustness without editing MuJoCo XML
files.

## Initial Joint Positions

`initial_joint_positions` under `env` lets you override individual joint
positions (qpos) after the keyframe reset.  This is useful for setting a
specific arm configuration or gripper opening at startup.

```yaml
env:
  initial_joint_positions:
    joint1: 0.276
    joint2: -1.651
    joint3: 0.775
    joint4: 1.981
    joint5: 1.110
    joint6: 0.408
    eef_claw_joint: 0.01        # gripper partially closed
```

### Semantics

- Values are written directly to `data.qpos` after keyframe reset, before
  `mj_forward`.
- Joint names must match the names defined in the MuJoCo XML (including any
  prefix added by `<attach … prefix="…"/>`).
- For **parallel-linkage grippers** (e.g. xf9600, robotiq): the driven joint
  and passive linkage joints are connected by equality constraints, which are
  only resolved during `mj_step`.  The framework automatically runs a short
  physics settle after setting initial joint positions when the model has
  equality constraints, so passive joints converge to a constraint-consistent
  state.
- When `eef_mapper` is configured for the operator, `initial_joint_positions`
  expects **raw joint values** (not finger distance), since it writes directly
  to qpos.  For replay from mcap data where values are in finger-distance
  space, the replay pipeline excludes eef joints from `initial_joint_positions`
  and applies them through the mapper instead.
- These overrides are applied **before** `_record_default_poses()`, so they
  become the baseline for randomization in the current rollout.  Every later
  `reset()` restores the composed XML/model baseline first and reapplies the
  override; sampled poses are never fed back as the next episode's baseline.

### Interaction with eef_mapper

If the operator has an [`eef_mapper`](../mujoco-backend/eef_mapper.md) configured,
`capture_observation` reports
finger distance and `apply_joint_action` accepts finger distance.  But
`initial_joint_positions` bypasses the mapper — it sets raw qpos.  Keep this
distinction in mind when mixing the two.

## Initial Pose Override

Before randomization is applied, you can override an object's initial pose (the
baseline that randomization offsets from) using `initial_pose` under `task`.
The same `PoseOverrideConfig` model is used for objects, cameras, and operator
initial states; only the fields relevant to the selected owner are interpreted.
Both **freejoint** and **static** (fixed) bodies are supported.

```yaml
task:
  initial_pose:
    source_block:
      position: [0.1, 0.0, 0.078]
      orientation: [0, 0, 0, 1]        # quaternion xyzw
    cup:
      position: [0.3, 0.1, 0.085]
      # orientation omitted → keeps keyframe default
    button_blue:
      position: [-0.01, -0.04, 0.08]   # static body — also works
```

### Operator baseline via `task_operators`

`task.initial_pose` only applies to scene objects. To change the baseline pose
used for operator randomization, configure the operator under
`task_operators.<name>.initial_state`:

```yaml
task_operators:
  arm:
    initial_state:
      base_pose:
        position: [-0.45, -0.06, 0.0]  # should match the XML `link0` pose
        orientation: [0, 0, 0, 1]

task:
  randomization:
    arm:
      base:
        x: [-0.015, 0.015]
        y: [-0.015, 0.015]
```

An operator base (or EEF home pose) can instead be expressed relative to a
named scene element.  The element may be a site, body, geom, or joint exported
by the composed MuJoCo model:

```yaml
task_operators:
  arm:
    initial_state:
      base_pose:
        reference: door__handle_grasp_center
        # T_world_base = T_world_reference × T_reference_base
        position: [0.2474, -0.4666, -0.1]
        orientation: [0.0, 0.0, 0.70710678, 0.70710678]
```

The named frame is resolved independently for each environment during backend
setup and reset.  It is a setup-time anchor: if the referenced joint later
moves during a Stage, the robot base remains fixed.  A `base_pose` with
`reference: world` keeps the ordinary world-frame behavior.  Built-in
references other than `world` are rejected for a base pose; `reference: base`
is meaningful for an EEF pose, not for the base itself.

In joint mode the resolved base pose also relocates the physical operator root
body. In pure mocap mode it updates the virtual base frame while the registered
mocap home remains the physical pose; this distinction is intentional because
the two control paths use different kinematic seams.

These initial-state overrides are applied before operator randomization
defaults are recorded, so:

- `task.randomization.arm.base` uses `initial_state.base_pose` as its baseline.
- `task.randomization.arm.eef` uses `initial_state.eef_pose` as its home EEF
  baseline.
- `initial_state.eef` only sets the gripper/open-close control value; it does
  not change the pose randomization baseline.
- If an operator `initial_state` field is omitted, the baseline falls back to
  the pose from the loaded scene / XML / operator registration state.

For fixed-base arms such as Franka, keep `initial_state.base_pose` aligned with
the robot root body pose in XML unless you intentionally want to shift the
whole robot base from YAML. Use
[Tune Randomization Extremes](../tools/tune_randomization_extremes.md) to
inspect the YAML-defined default state and the randomization ranges around it.

### Shared `PoseOverrideConfig` fields

| Field          | Format                                  | Default |
|----------------|-----------------------------------------|---------|
| `position`     | `[x, y, z]` in the selected reference frame | `null` (preserve fallback) |
| `orientation`  | 4 floats: quaternion `[x, y, z, w]` **or** 3 floats: Euler `[roll, pitch, yaw]` in radians | `null` (preserve fallback) |
| `reference`    | `world`, `base` (EEF only), or a named scene element | `world` |

Both pose fields are optional.  For a named reference, omitted components keep
the fallback pose after it is transformed into that frame; provided components
replace only that local component.  The six-value operator EEF shorthand
`[x, y, z, yaw, pitch, roll]` is accepted as a complete world-frame pose; use
the structured form for partial values or a non-world reference.

The configuration boundary validates these shapes before a simulator is
created: `position` must contain exactly three finite values, `orientation`
must contain either three finite RPY values or four finite quaternion values,
and a four-value quaternion must be non-zero.  The flat EEF shorthand must
contain exactly six finite values.  Pose overrides are frozen models with tuple
components, so a loaded configuration cannot be mutated in place.

### Interaction with randomization

`initial_pose` and operator initial states are applied **after** the keyframe
reset and **before** `_record_default_poses()` / randomization.  This means:

- The initial pose becomes the **default/baseline** for randomization in that
  reset.  A subsequent reset starts from the composed XML/model baseline and
  reapplies the configured initial pose, so randomization offsets do not
  accumulate across episodes.
- `reference: relative` adds offsets on top of the initial pose (not the
  XML keyframe).
- `reference: absolute_world` replaces sampled axes with absolute values
  as usual; unsampled axes fall back to the initial pose.

Named initial-pose references are resolved after the keyframe reset and before
pose randomization.  When a `reference` exactly matches another key in
`task.initial_pose`, that entry is treated as a dependency and applied first;
the dependency graph is topologically ordered and cycles are rejected before
any scene pose is mutated.  References to scene frames that are not configured
`initial_pose` keys are resolved directly from the current reset baseline.
These are not randomization delta-carry references: use `task.randomization`
when an entity should follow another entity's sampled displacement.

```yaml
task:
  initial_pose:
    source_block:
      position: [0.1, 0.0, 0.078]      # new baseline
  randomization:
    source_block:
      x: [-0.03, 0.03]                  # jitters around x=0.1
      y: [-0.03, 0.03]
```

## Randomization YAML Configuration

Add a `randomization` block under `task` in your YAML config.
Keys are object or operator names.

- Objects take a direct per-axis range.
- Operators must use the **nested form** with explicit `base:` and/or `eef:`
  sub-entries. The direct per-axis shorthand on an operator key is no longer
  supported and raises `TypeError` at sample time.

For a target whose valid workspace is disjoint, use
`PoseRandomizationConfig`'s `regions` list. The legacy direct form remains
valid; `regions` is a non-empty list of complete per-region configurations.
Each region owns its axis ranges, `reference`, and `collision_radius`.
On every sampling attempt, exactly one region is selected with equal
probability for each target and environment. A collision-rejection retry
selects a region again, so a failed sample can move to a different disconnected
area. This is a uniform **proposal** mixture; the final accepted samples are
conditioned on collision rejection, so regions with different collision-free
acceptance rates need not appear equally often.

```yaml
task:
  seed: 42                     # numpy RNG seed for reproducibility
  # randomization_debug: true  # see "Debug Mode" below
  randomization:
    source_block:
      x: [-0.03, 0.03]         # metres, world frame
      y: [-0.03, 0.03]
      # yaw: [-0.524, 0.524]   # radians
      collision_radius: 0.04   # metres, for collision rejection
    arm:
      base:                    # randomize the operator's base
        x: [-0.015, 0.015]
        y: [-0.015, 0.015]
        collision_radius: 0.15
      eef:                     # ...and/or the home end-effector pose
        x: [-0.01, 0.01]
        y: [-0.01, 0.01]
        z: [-0.005, 0.005]
```

### Multiple disjoint regions

`regions` represents an equal-weight **proposal mixture of axis-aligned
boxes**. Position coordinates are sampled independently from each selected
region's `x`, `y`, and `z` intervals (and orientation axes from its `roll`,
`pitch`, and `yaw` intervals). The intervals must describe boxes in the
selected reference frame; arbitrary polygons, masks, or a single bounding box
with holes are not represented.

```yaml
task:
  randomization:
    source_block:
      regions:
        # Left work area: relative to the block's default pose.
        - x: [-0.30, -0.15]
          y: [-0.10, 0.10]
          reference: relative
          collision_radius: 0.04
        # Right work area: absolute world coordinates, with its own radius.
        - x: [0.45, 0.60]
          y: [0.20, 0.35]
          reference: absolute_world
          collision_radius: 0.06

    arm:
      eef:
        regions:
          - x: [0.20, 0.30]
            y: [-0.10, 0.00]
            z: [0.15, 0.20]
            reference: absolute_base
            collision_radius: 0.15
          - x: [0.35, 0.45]
            y: [0.05, 0.15]
            z: [0.15, 0.20]
            reference: absolute_base
            collision_radius: 0.12
```

The same wrapper can be used under an operator's `base:` entry. Region
selection is independent for `base` and `eef`; references declared in every
region participate in dependency ordering. If a selected region references an
entity, the existing delta-carry and reference-connected collision semantics
apply to that sampled region.

`regions: []` is invalid. Cameras and per-waypoint randomization currently
accept only a single `PoseRandomRange`; use `task.randomization` (and an
operator's nested `base`/`eef`) for disjoint entity workspaces.

### Supported axes

| Axis    | Unit    | Description            |
|---------|---------|------------------------|
| `x`     | metres  | X translation          |
| `y`     | metres  | Y translation          |
| `z`     | metres  | Z translation          |
| `roll`  | radians | Rotation about X       |
| `pitch` | radians | Rotation about Y       |
| `yaw`   | radians | Rotation about Z       |

Each axis takes a `[min, max]` tuple. **Omitted axes default to `None`**,
meaning "do not randomize this axis — keep the default pose's value for it".
This is true in every mode, including the absolute modes described below.

### Reference modes

Each randomization entry (and each sub-entry in the nested operator form) can
set a `reference` field that selects how the `[min, max]` ranges are
interpreted:

| `reference`       | Meaning                                                                                 |
|-------------------|-----------------------------------------------------------------------------------------|
| `relative` (default) | Sampled values are **added** to the entity's default pose (the existing behavior). |
| `absolute_world`  | Sampled values are **absolute world-frame** coordinates (metres) / Euler angles (rad). |
| `absolute_base`   | Sampled values are absolute coordinates in the **operator's base frame**, then transformed to world before being applied. **Only valid for the nested operator `eef:` sub-entry.** |
| `<entity_name>`   | **Entity-reference mode.** The referenced entity is randomized first (dependency ordering via topological sort). Then a **delta-carry** is applied: `delta = ref_sampled * ref_default⁻¹` is computed and applied to this entity's default pose, preserving the original spatial relationship. After carrying, the per-axis ranges are applied as additive offsets (like `relative` mode). For an **object** name the referenced pose is the object pose; for an **operator** name (plain, no suffix) the referenced pose is the operator's **base** — equivalent to `<operator>.base` below. |
| `<operator>.base` / `<operator>.eef` | **Operator-attribute reference.** Same delta-carry semantics as an entity name, but anchored to the operator's **base** (`get_base_pose()`) or **home end-effector** (`get_end_effector_pose()`) pose. Only `.base` / `.eef` are recognized, and only for operator names. A plain operator name (e.g. `arm`) is equivalent to `arm.base`. |

Examples:

```yaml
task:
  randomization:
    # Place the cup anywhere in a world-frame rectangle on the table,
    # keeping its default height and orientation
    cup:
      reference: absolute_world
      x: [0.30, 0.50]
      y: [-0.15, 0.15]
      collision_radius: 0.04

    # Small relative jitter of the arm base plus a home-EEF box expressed
    # in that base frame. Operator randomization always uses this nested form.
    arm:
      base:
        x: [-0.01, 0.01]
        y: [-0.01, 0.01]
      eef:
        reference: absolute_base
        x: [0.25, 0.35]
        y: [-0.05, 0.05]
        z: [0.20, 0.30]
```

Entity-reference example (arrange_flowers: flower tracks vase):

```yaml
task:
  randomization:
    vase:
      reference: absolute_world
      x: [0.22, 0.58]
      y: [-0.32, 0.27]
    flower:
      reference: vase            # carry with vase, then jitter ±5mm
      x: [-0.005, 0.005]
      y: [-0.005, 0.005]
    vase2:
      reference: absolute_world
      x: [0.22, 0.58]
      y: [-0.32, 0.27]
```

When `vase` moves from its default to a new position, the flower is "carried"
by the same rigid displacement (preserving the original spatial relationship),
then receives its own small perturbation on top. This ensures the flower always
stays inside the vase's opening regardless of where the vase is placed.

The entries are automatically topologically sorted — `vase` is processed before
`flower`. Circular references (A → B → A) raise a ``ValueError``.

Operator-attribute reference example (objects drift with the arm's base):

```yaml
task:
  randomization:
    arm:
      base:
        x: [-0.05, 0.05]
        y: [-0.05, 0.05]
    vase:
      reference: arm.base        # equivalent to `reference: arm`
      x: [-0.005, 0.005]
      y: [-0.005, 0.005]
    gripper_hover: # rarely used
      reference: arm.eef         # track the operator's home EEF instead
      z: [-0.01, 0.01]
```

The plain `reference: arm` form is equivalent to `arm.base`; write `arm.eef`
explicitly to track the operator's home end-effector pose. If the referenced
attribute is not randomized (for example `arm.base` with no `base:` sub-entry),
the delta is zero and this entry keeps its own default pose before applying the
per-axis offsets. `.base` / `.eef` suffixes only apply to operator names; using
them on an object (e.g. `vase.base`) raises a `ValueError`.

Restrictions on `absolute_base`:

- Object entries reject `absolute_base` (no base frame is defined for an object).
- The nested `base:` sub-entry rejects `absolute_base` (the base IS the frame).
- Per-waypoint `randomization` rejects `absolute_base`; use the waypoint's own
  `reference: base` field together with `relative` or `absolute_world` instead.

### Operator semantics

Operator entries must use the nested form with explicit `base:` and/or `eef:`
sub-entries. Writing per-axis ranges directly under an operator key (the legacy
"direct form") is rejected at sample time with a `TypeError`.

- `base` randomizes `get_base_pose()`
  - for mocap operators this is the virtual base frame
  - for joint-mode operators this is the base reference frame
- `eef` randomizes the operator home end-effector pose; reset updates the
  stored home EEF pose and then homes the operator to it
- `base` and `eef` can be configured together; each sub-entry has its own
  `reference`, `collision_radius`, and per-axis ranges

### collision_radius

Each entity has a `collision_radius` (default 0.05 m). After sampling, pairwise
Euclidean distances are checked: if any two entities are closer than the sum of
their radii, the sample is rejected and redrawn. After 100 failed attempts the
last sample is applied with a warning.

Entities linked by an entity-name reference chain are excluded from rejection
against each other. This allows carried assemblies such as `flower -> vase` to
move together while still rejecting overlap against unrelated randomized
entities (for example `vase2`).

When a referenced child collides with an unrelated randomized entity, the whole
reference-connected component is re-sampled together. In the `flower -> vase`
example, a child collision triggers a fresh sample for both `vase` and
`flower`, rather than retrying only the flower's local jitter.

Practical implications:

- Collision rejection is not strictly "per YAML key". It operates on
  **reference-connected components** formed by entity-name references.
- A component is accepted only when **all of its members** are collision-free
  against already accepted components and earlier accepted entities in the same
  reset.
- This matters most when a child has only a small local jitter. If `flower`
  carries with `vase` and ends up too close to `arm.eef`, retrying only the
  flower's `±5 mm` offset would usually not help; the sampler therefore retries
  the whole `vase + flower` component.
- Direct collisions inside the same reference chain are still ignored, so a
  carried child is allowed to remain inside/on its referenced parent as
  intended.

## Per-Waypoint Randomization

In addition to entity-level randomization under `task.randomization`, individual
waypoints inside a stage's `pre_move` / `post_move` list may carry their own
`randomization` block. At stage execution time this perturbs the waypoint's
nominal `position` (and optionally orientation), independently of entity pose
randomization. The same `reference` modes (`relative` / `absolute_world`) are
supported as for entity randomization.

```yaml
stages:
  - name: grasp_and_open
    operator: arm
    operation: push
    param:
      pre_move:
        - position: [-0.10, 0.0955, -0.020]
          reference: object_world
          orientation: [0.7133, -0.0293, 0.0043, 0.7002]
        - position: [-0.05, 0.0955, -0.020]
          reference: object_world
          randomization:
            x: [-0.02, 0.00]
            y: [-0.00, 0.02]
            z: [-0.01, 0.01]
```

Semantics:

- Supported axes are the same as entity randomization (`x/y/z/roll/pitch/yaw`);
  **omitted axes default to `None` and are not touched** — the waypoint keeps
  its nominal value on that axis.
- Supports `reference: relative` (default) and `reference: absolute_world`.
  In relative mode the sampled values are added to the waypoint's nominal
  position/orientation; in `absolute_world` mode they replace it.
- `reference: absolute_base` is **not supported** for per-waypoint randomization
  because a waypoint already carries its own `reference` field (e.g.
  `object_world`, `world`, `eef_world`, `base`) that selects the frame in which
  the sampled numbers are interpreted by the pose controller. To randomize in
  the base frame, set the waypoint's `reference: base` and use `absolute_world`
  or `relative` inside its `randomization` block.
- The sampled numbers are always expressed in the waypoint's own `reference`
  frame, so the perturbation follows the frame the waypoint is anchored to.
- Sampling happens once per `reset()`. The runner uses the backend's public
  `get_random_generator()` when provided, otherwise a runner-owned generator
  seeded by a nonzero `task.seed`; per-waypoint offsets therefore remain
  reproducible without depending on a private backend field.
- Per-waypoint randomization is independent from entity randomization and does
  not participate in `collision_radius` rejection; keep ranges small enough that
  the resulting motion stays reachable.
- Debug mode (`randomization_debug: true`) also cycles per-waypoint extremes.

## Camera Initial Pose Overrides

Before camera randomization records its baseline, `task.camera_initial_pose`
lets you override the XML-defined camera pose from YAML. This is useful for
moving a camera to a calibrated viewpoint without editing the scene XML, or
for per-task camera overrides that should persist across randomization.

```yaml
task:
  camera_initial_pose:
    env1_cam:
      position: [2.4, 0.6, -0.1]
      orientation: [-0.5, 0.5, 0.5, 0.5]   # xyzw quaternion
    env0_cam:
      position: [2.3, 0.15, 0.1]
      # orientation omitted → keeps XML value
```

Camera overrides use the same `PoseOverrideConfig` as object and operator
initial states.  The pose values are interpreted in the selected reference
frame; `world` is the usual choice, while a named site/body/geom/joint can be
used when the camera should be calibrated from a composed scene element.

| Field          | Format                                                              | Default |
|----------------|---------------------------------------------------------------------|---------|
| `position`     | `[x, y, z]` in the selected reference frame                         | `null` (preserve XML) |
| `orientation`  | 4 floats: quaternion `[x, y, z, w]` **or** 3 floats: Euler `[roll, pitch, yaw]` in radians | `null` (preserve XML) |
| `reference`    | `world` or a named scene element (site/body/geom/joint)              | `world` |

Semantics:

- Overrides are applied at each `reset()` **before** camera randomization
  records its defaults, so `camera_randomization` with `reference: relative`
  jitters around the overridden pose rather than the XML pose.
- Named references are resolved once per environment during setup/reset.  They
  are anchors, not tracking references: if the referenced articulated element
  moves during a Stage, the camera keeps the resolved world pose.
- For cameras marked `is_static: true` in `env.cameras`, the GS background
  is cached at the first render. When overriding such a camera's pose,
  make sure the override is applied before the first render — otherwise
  the cached background reflects the XML pose.

## Camera Pose Randomization

Camera viewpoint randomization is configured under `task.camera_randomization`.
Keys are camera names as defined in the MuJoCo XML model. Each entry is a
`PoseRandomRange` with the same axis fields as entity randomization.

```yaml
task:
  camera_randomization:
    env1_cam:
      x: [-0.05, 0.05]       # metres, jitter around default position
      y: [-0.05, 0.05]
      z: [-0.02, 0.02]
      pitch: [-0.1, 0.1]     # radians
      yaw: [-0.1, 0.1]
    env0_cam:
      reference: absolute_world
      x: [0.8, 1.0]
      y: [-0.1, 0.1]
      z: [0.4, 0.6]
```

### Reference modes

Only `relative` (default) and `absolute_world` are supported for cameras:

| `reference`       | Meaning                                                          |
|-------------------|------------------------------------------------------------------|
| `relative`        | Sampled values are **added** to the camera's effective default pose (the XML pose unless `camera_initial_pose` overrides it). |
| `absolute_world`  | Sampled values are **absolute world-frame** coordinates/angles.  |

`absolute_base` and entity-name references are **rejected** because cameras have
no operator base frame and do not participate in entity dependency ordering.

### Semantics

- Camera randomization is applied at each `reset()` **after** object and operator
  randomization.
- The default camera pose (the baseline for `relative` mode) is the effective
  pose recorded during `setup()`: the XML pose when no
  `camera_initial_pose` entry exists, otherwise that resolved override.
- Cameras do **not** participate in `collision_radius` rejection — they have no
  physical presence.
- Camera randomization uses the same `task.seed` RNG as entity randomization for
  full reproducibility.
- Sampled camera poses are included in the `initial_poses` details returned by
  `TaskRunner.reset()` under a `"_cameras"` key.

### Interaction with GS rendering

If a camera has `is_static: true` in the `env.cameras` config, its GS background
is cached at the first render. Randomizing such a camera will change its
viewpoint each episode while the cached background remains from the original
pose. For cameras that should re-render the GS background after randomization,
set `is_static: false`.

## How It Works

The entity and camera randomization logic is implemented by
`MujocoTaskBackend`, which fulfills the generic `SceneBackend` contract.

### Lifecycle

1. **`setup()`** — starting from the constructed environment's keyframe state,
   homes the operators, then applies `task.initial_pose`,
   `task_operators.*.initial_state` (base first, EEF second), and
   `task.camera_initial_pose`. Named references are resolved at this point
   against the composed scene. The resulting object, operator, and camera
   poses are recorded as the canonical randomization baselines.

2. **`reset()`** — restores the keyframe, model-level XML poses, and
   registered operator home/cache, reapplies all initial overrides (so a named
   reference is resolved for the current environment row), and then calls
   `_apply_randomization()`:
   1. Topologically sorts the randomization keys by entity-reference
      dependency, then groups reference-connected keys into components (for
      example `vase` + `flower`).
   2. Resolves each key to an object handler or operator handler.
   3. For each axis with a `[min, max]` tuple (axes set to `None` are skipped),
      samples a uniform random value.
   4. Combines the sampled values with the default pose according to
      `reference`:
      - `relative` — adds the sampled values to the default pose (translation
        additive; rotation additive in RPY then converted back to quaternion).
      - `absolute_world` — replaces the default pose's value on each sampled
        axis; unsampled axes are left at their default.
      - `absolute_base` (operator EEF only) — transforms the default EEF world
        pose into the operator's base frame, replaces sampled axes there, then
        transforms the result back to world.
   5. Runs collision rejection on the sampled component. If any member of the
      component overlaps an unrelated accepted participant, the **entire
      component** is re-sampled up to 100 times.
   6. Applies object poses, operator base poses, and operator home EEF poses
      through their respective APIs.
   7. Calls `_apply_camera_randomization()` (if configured), then refreshes the
      viewer.

3. **`TaskRunner.reset()`** — after the backend reset, collects the realized
   poses of all task-relevant entities (stage operators/objects, plus any extra
   entities mentioned in `randomization`) and returns them in
   `TaskUpdate.details["initial_poses"]`. This allows the caller to log initial
   conditions without accessing backend internals.
   For operators, the returned value always contains both `base_pose` and
   `eef_pose`, regardless of which sub-entries (`base`, `eef`, or both) were
   configured under the operator key.

### Entity resolution

Each randomization key is resolved in order:
1. `object_handlers[name]` — uses `get_pose()` / `set_pose()`.
2. `operator_handlers[name]` — must use the nested
   `OperatorRandomizationConfig` form:
   - `base:` randomizes `get_base_pose()` and applies via `set_pose()`
   - `eef:` randomizes the home end-effector pose and applies via
     `set_home_end_effector_pose()`
   - for `reference: absolute_base` (only valid on the `eef:` sub-entry) the
     sampler also calls `get_base_pose()` to transform between the base
     frame and world
   - a plain ``PoseRandomRange`` directly under an operator key is rejected
3. If neither matches, a warning is emitted and the key is skipped.

References are resolved with the same fallback:

- `reference: <object_name>` uses the referenced object's `get_pose()`.
- `reference: <operator_name>` uses the operator's **base** pose
  (`get_base_pose()`), equivalent to `<operator_name>.base`.
- `reference: <operator_name>.eef` uses the operator's home end-effector pose (rarely used).
- `reference: <non_operator>.base` / `.eef` raises a `ValueError`.

## Multi-Round Evaluation

Use the `rounds` top-level config key (default 1) to run the task multiple times
with different random seeds:

```bash
aao-demo rounds=10
aao-demo --config-name cup_on_coaster rounds=20
```

Each round resets the scene (applying a fresh random sample) and runs all stages.
A summary is printed at the end:

```
============================================================
SUMMARY
============================================================
Success rate: 8/10

  Round 1: [OK]
    source_block: pos=[0.0123, -0.0201, 0.06]
    arm: pos=[-0.005, 0.012, 0.0]
    stage pick_source: completed
    stage place_source: completed
  Round 2: [FAIL]
    source_block: pos=[-0.028, 0.015, 0.06]
    ...
============================================================
```

## Debug Mode

Set `randomization_debug: true` to cycle through extreme poses before random
sampling. The sequence is:

1. All entities at all-axis **minimum** simultaneously.
2. All entities at all-axis **maximum** simultaneously.
3. For each entity, for each non-trivial axis (where min != max):
   one case at axis min, one at axis max (others at default).

After exhausting all extreme cases, subsequent resets switch to normal random
sampling. This is useful for verifying that configured ranges don't cause
unreachable grasps or collisions.

```bash
aao-demo task.randomization_debug=true rounds=20
```

## Reproducibility

Set `task.seed` to fix the numpy RNG seed:

```bash
aao-demo task.seed=42 rounds=5
```

The same seed produces the same sequence of random poses across runs.
