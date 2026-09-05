# Scene Initialization & Randomization

auto_atom supports flexible scene initialization via YAML configuration:
per-entity pose overrides, per-joint position overrides, and pose
randomization applied at each `reset()`.  Together these let you set up
initial conditions and evaluate task robustness without encoding those choices
in a simulator asset.

This page defines the backend-independent configuration semantics. Logical
object, operator, camera, joint, and frame names are resolved by the selected
backend. How a backend writes simulator state, restores its native reset state,
or handles renderer caches belongs in that backend's documentation. See
[MuJoCo Initialization & Randomization](../mujoco-backend/initialization_randomization.md)
for the built-in MuJoCo mapping.

## Initial Joint Positions

Use `task_operators.<name>.initial_state.joint_positions` to define an
operator's home joint state. The values stay with the logical operator that
owns the joints and are reapplied on every reset.

```yaml
task_operators:
  arm:
    initial_state:
      joint_positions:
        joint1: 0.0
        joint2: -1.5
```

### Semantics

- Values are operator joint-space coordinates, keyed by logical joint name.
  Their units and scalar/vector shape are defined by the operator model; they
  are not end-effector or gripper user-space values. The backend maps these
  logical coordinates to its native state.
- Joint names must belong to the operator's declared arm joints. Unknown or
  unowned names are rejected.
- The joint state is applied after the backend restores its reset state and
  before operator pose randomization records its baseline. Random samples never
  become the next episode's baseline.
- `joint_positions` and `eef_pose` are competing arm-home representations and
  cannot be configured together. `eef` is an independent gripper-control
  override.

Environment-level joint initialization, such as passive scene joints, is part
of the selected `env` schema rather than the shared task schema. For the
built-in environment, see
[MuJoCo Joint Initialization](../mujoco-backend/initialization_randomization.md#joint-initialization).

## Initial Pose Override

Before randomization is applied, you can override an object's initial pose (the
baseline that randomization offsets from) using `initial_pose` under `task`.
The same `PoseOverrideConfig` model is used for objects, cameras, and operator
initial states; only the fields relevant to the selected owner are interpreted.
The target must be a logical object whose pose can be read and set by the
selected backend.

```yaml
task:
  initial_pose:
    source_block:
      position: [0.1, 0.0, 0.078]
      orientation: [0, 0, 0, 1]        # quaternion xyzw
    cup:
      position: [0.3, 0.1, 0.085]
      # orientation omitted → keeps the backend-provided reset orientation
    button_blue:
      position: [-0.01, -0.04, 0.08]
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
        position: [-0.45, -0.06, 0.0]
        orientation: [0, 0, 0, 1]

task:
  randomization:
    arm:
      base:
        x: [-0.015, 0.015]
        y: [-0.015, 0.015]
```

An operator base (or EEF home pose) can instead be expressed relative to a
named scene frame exposed by the selected backend:

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

The named frame is resolved independently for each environment during reset.
It is an initialization-time anchor: if the referenced frame later moves
during a Stage, the robot base remains fixed. A `base_pose` with
`reference: world` keeps the ordinary world-frame behavior.  Built-in
references other than `world` are rejected for a base pose; `reference: base`
is meaningful for an EEF pose, not for the base itself.

These initial-state overrides are applied before operator randomization
defaults are recorded, so:

- `task.randomization.arm.base` uses `initial_state.base_pose` as its baseline.
- `task.randomization.arm.eef` uses `initial_state.eef_pose` as its home EEF
  baseline.
- `initial_state.eef` only sets the gripper/open-close control value; it does
  not change the pose randomization baseline.
- If an operator `initial_state` field is omitted, the baseline falls back to
  the reset pose supplied by the selected backend.

Use
[Tune Randomization Extremes](../tools/tune_randomization_extremes.md) to
inspect the YAML-defined default state and the randomization ranges around it.

### Shared `PoseOverrideConfig` fields

| Field          | Format                                  | Default |
|----------------|-----------------------------------------|---------|
| `position`     | `[x, y, z]`, or `{x, y, z}` where each value is a scalar or `{value, reference}` | `null` (preserve fallback) |
| `orientation`  | 4 floats: quaternion `[x, y, z, w]`, 3 floats: Euler `[roll, pitch, yaw]`, or `{roll, pitch, yaw}` where each value is a scalar or `{value, reference}` | `null` (preserve fallback) |
| `reference`    | `world`, `base` (EEF only), or a named scene frame | `world` |

Both pose fields are optional.  For a named reference, omitted components keep
the fallback pose after it is transformed into that frame; provided components
replace only that local component.  The six-value operator EEF shorthand
`[x, y, z, yaw, pitch, roll]` is accepted as a complete world-frame pose; use
the structured form for partial values or a non-world reference.

Position and RPY orientation can also use per-axis references. A scalar axis
inherits the pose-level `reference`; the expanded `{value, reference}` form has
higher priority. Axis overrides replace the corresponding world coordinate after
the global reference is resolved, so a base anchored to a handle can keep its
local `x/y` while fixing `z` in world coordinates:

```yaml
task_operators:
  arm:
    initial_state:
      base_pose:
        reference: door__handle_grasp_center
        position:
          x: 0.2474
          y: -0.4666
          z:
            value: -0.1
            reference: world
```

The quaternion form remains an atomic four-component orientation. Use the
expanded `roll/pitch/yaw` form when individual orientation axes need different
references.

The configuration boundary validates these shapes before a backend is
constructed: compact `position` must contain exactly three finite values, compact
`orientation` must contain either three finite RPY values or four finite
quaternion values, and a four-value quaternion must be non-zero. Expanded
position/RPY mappings validate each configured component as a finite scalar.
The flat EEF shorthand must contain exactly six finite values. Pose overrides
are frozen models, so a loaded configuration cannot be mutated in place.

### Interaction with randomization

`initial_pose` and operator initial states are applied after the backend
restores its reset state and before randomization records its effective
baseline. This means:

- The initial pose becomes the **default/baseline** for randomization in that
  reset. A subsequent reset starts from the backend-provided reset state and
  reapplies the configured initial pose, so randomization offsets do not
  accumulate across episodes.
- `reference: relative` adds offsets on top of the initial pose (not the
  backend's unmodified reset pose).
- `reference: absolute_world` replaces sampled axes with absolute values
  as usual; unsampled axes fall back to the initial pose.

Named initial-pose references are resolved after the backend reset and before
pose randomization. When a `reference` exactly matches another key in
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
  seed: 42                     # episode randomization seed
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

Each axis may override the entry-level reference without an additional
container. The compact ``[min, max]`` form inherits the entry-level reference;
the expanded form owns a higher-priority reference:

```yaml
task:
  randomization:
    arm:
      base:
        reference: relative       # fallback for x/y/roll/pitch/yaw
        x: [-0.01, 0.01]
        y: [-0.01, 0.01]
        z:
          range: [-0.305, -0.295]
          reference: absolute_world
```

Here `x` and `y` remain offsets from the base's effective initial pose, while
`z` is sampled directly in world coordinates. Omitting `reference` inside the
expanded axis object also inherits the entry-level reference. Named entity or
operator-attribute axis references participate in the same dependency ordering
as entry-level references.

Restrictions on `absolute_base`:

- Object entries reject `absolute_base` (no base frame is defined for an object).
- The nested `base:` sub-entry rejects `absolute_base` (the base IS the frame).
- An EEF entry may use `absolute_base`, but it cannot mix `absolute_base` with
  axis references in other frames within the same pose.
- Per-waypoint `randomization` rejects `absolute_base`; use the waypoint's own
  `reference: base` field together with `relative` or `absolute_world` instead.

### Operator semantics

Operator entries must use the nested form with explicit `base:` and/or `eef:`
sub-entries. Writing per-axis ranges directly under an operator key (the legacy
"direct form") is rejected at sample time with a `TypeError`.

- `base` randomizes the operator's logical base pose.
- `eef` randomizes the operator's home end-effector pose; the next episode
  starts from that sampled home.
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
- Sampling happens once per `reset()` and uses the episode random-number
  stream. A fixed nonzero `task.seed` makes per-waypoint offsets reproducible.
- Per-waypoint randomization is independent from entity randomization and does
  not participate in `collision_radius` rejection; keep ranges small enough that
  the resulting motion stays reachable.
- Debug mode (`randomization_debug: true`) also cycles per-waypoint extremes.

## Camera Initial Pose Overrides

Before camera randomization records its baseline, `task.camera_initial_pose`
lets you override the backend-provided camera pose from YAML. This is useful
for moving a camera to a calibrated viewpoint or defining per-task camera
overrides that should persist across randomization.

```yaml
task:
  camera_initial_pose:
    env1_cam:
      position: [2.4, 0.6, -0.1]
      orientation: [-0.5, 0.5, 0.5, 0.5]   # xyzw quaternion
    env0_cam:
      position: [2.3, 0.15, 0.1]
      # orientation omitted → keeps the backend-provided value
```

Camera overrides use the same `PoseOverrideConfig` as object and operator
initial states.  The pose values are interpreted in the selected reference
frame; `world` is the usual choice, while a named scene frame can be used when
the camera should be calibrated from another element exposed by the backend.

| Field          | Format                                                              | Default |
|----------------|---------------------------------------------------------------------|---------|
| `position`     | `[x, y, z]` in the selected reference frame                         | `null` (preserve reset value) |
| `orientation`  | 4 floats: quaternion `[x, y, z, w]` **or** 3 floats: Euler `[roll, pitch, yaw]` in radians | `null` (preserve reset value) |
| `reference`    | `world` or a named scene frame exposed by the backend                | `world` |

Semantics:

- Overrides are applied at each `reset()` **before** camera randomization
  records its defaults, so `camera_randomization` with `reference: relative`
  jitters around the overridden pose rather than the backend's unmodified
  reset pose.
- Named references are resolved once per environment during reset. They
  are anchors, not tracking references: if the referenced articulated element
  moves during a Stage, the camera keeps the resolved world pose.

## Camera Pose Randomization

Camera viewpoint randomization is configured under `task.camera_randomization`.
Keys are logical camera names exposed by the selected backend. Each entry is a
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
| `relative`        | Sampled values are **added** to the camera's effective reset pose (including any `camera_initial_pose` override). |
| `absolute_world`  | Sampled values are **absolute world-frame** coordinates/angles.  |

`absolute_base` and entity-name references are **rejected** because cameras have
no operator base frame and do not participate in entity dependency ordering.

### Semantics

- Camera randomization is applied at each `reset()` **after** object and operator
  randomization.
- The default camera pose (the baseline for `relative` mode) is the effective
  reset pose: the backend-provided pose when no `camera_initial_pose` entry
  exists, otherwise that resolved override.
- Cameras do **not** participate in `collision_radius` rejection — they have no
  physical presence.
- Camera randomization uses the same `task.seed` RNG as entity randomization for
  full reproducibility.
- Sampled camera poses are included in the `initial_poses` details returned by
  `TaskRunner.reset()` under a `"_cameras"` key.

## Reset Contract and Observability

The configuration describes this observable episode-reset contract:

1. The selected backend restores its own native reset state.
2. Object, operator, and camera initial-state overrides are reapplied. Named
   initialization references are resolved as anchors for this reset.
3. Those effective poses become the baselines for `relative` randomization.
4. Object and operator randomization is sampled with dependency ordering and
   collision rejection; camera and waypoint randomization follow their scopes
   described above.
5. `TaskRunner.reset()` returns realized task-relevant poses in
   `TaskUpdate.details["initial_poses"]`. Operator entries contain both
   `base_pose` and `eef_pose`; camera entries are grouped under `"_cameras"`.

The backend owns the mechanics of realizing this contract. It also defines
which logical names and named scene frames are available. Backend-specific
limitations and state-mapping details must not change the meanings of
`relative`, `absolute_world`, `absolute_base`, omitted axes, or dependency
references described on this page.

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

Set `task.seed` to fix the episode randomization seed:

```bash
aao-demo task.seed=42 rounds=5
```

The same seed produces the same sequence of random poses across runs.
