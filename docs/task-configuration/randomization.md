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
    xfg_claw_joint: 0.01        # gripper partially closed
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
  become the new baseline for any subsequent randomization.

### Interaction with eef_mapper

If the operator has an [`eef_mapper`](../mujoco-backend/eef_mapper.md) configured,
`capture_observation` reports
finger distance and `apply_joint_action` accepts finger distance.  But
`initial_joint_positions` bypasses the mapper — it sets raw qpos.  Keep this
distinction in mind when mixing the two.

## Initial Pose Override

Before randomization is applied, you can override any object's initial pose
(the baseline that randomization offsets from) using `initial_pose` under
`task`.  This lets you set object positions from YAML without editing the
MuJoCo XML or keyframe.  Both **freejoint** and **static** (fixed) bodies
are supported.

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

These initial-state overrides are applied before operator randomization
defaults are recorded, so:

- `task.randomization.arm.base` uses `initial_state.base_pose` as its baseline.
- Direct `task.randomization.arm` and nested `task.randomization.arm.eef`
  use `initial_state.arm` as their home EEF baseline.
- `initial_state.eef` only sets the gripper/open-close control value; it does
  not change the pose randomization baseline.
- If an operator `initial_state` field is omitted, the baseline falls back to
  the pose from the loaded scene / XML / operator registration state.

For fixed-base arms such as Franka, keep `initial_state.base_pose` aligned with
the robot root body pose in XML unless you intentionally want to shift the
whole robot base from YAML. See [Tune Initial State](../tools/tune_initial_state.md) for
a full breakdown of `base_pose`, `arm`, and `eef`.

### Fields

| Field          | Format                                  | Default |
|----------------|-----------------------------------------|---------|
| `position`     | `[x, y, z]` — world-frame metres       | `null` (keep keyframe) |
| `orientation`  | 4 floats: quaternion `[x, y, z, w]` **or** 3 floats: Euler `[roll, pitch, yaw]` in radians | `null` (keep keyframe) |

Both fields are optional.  Omitting a field keeps the value defined in the
XML keyframe for that component.

### Interaction with randomization

`initial_pose` is applied **after** the keyframe reset and **before**
`_record_default_poses()`.  This means:

- The initial pose becomes the new **default/baseline** for randomization.
- `reference: relative` adds offsets on top of the initial pose (not the
  XML keyframe).
- `reference: absolute_world` replaces sampled axes with absolute values
  as usual; unsampled axes fall back to the initial pose.

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
- Operators may use the direct per-axis form (which now means end-effector
  randomization) or a nested form to randomize `base` and `eef` independently.

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
      # Direct form → randomizes the operator's home EEF pose
      x: [-0.015, 0.015]
      y: [-0.015, 0.015]
      collision_radius: 0.15
    # Nested form (same operator name) — randomizes base and eef independently:
    # arm:
    #   base:
    #     x: [-0.015, 0.015]
    #     y: [-0.015, 0.015]
    #   eef:
    #     x: [-0.01, 0.01]
    #     y: [-0.01, 0.01]
    #     z: [-0.005, 0.005]
```

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
| `absolute_base`   | Sampled values are absolute coordinates in the **operator's base frame**, then transformed to world before being applied. **Only valid for operator EEF randomization** (direct operator form or the nested `eef:` sub-entry). |
| `<entity_name>`   | **Entity-reference mode.** The referenced entity is randomized first (dependency ordering via topological sort). Then a **delta-carry** is applied: `delta = ref_sampled * ref_default⁻¹` is computed and applied to this entity's default pose, preserving the original spatial relationship. After carrying, the per-axis ranges are applied as additive offsets (like `relative` mode). |

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

    # Place the arm's home EEF anywhere in a box defined in the arm's base
    # frame — useful when the base itself is randomized or when the "right
    # side of the robot" matters more than "the right side of the room"
    arm:
      reference: absolute_base
      x: [0.20, 0.35]
      y: [-0.10, 0.10]
      z: [0.15, 0.25]

    # Mixed (nested form): small relative jitter of the base plus an
    # absolute-base EEF box — uses the same operator name "arm"
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

Restrictions on `absolute_base`:

- Object entries reject `absolute_base` (no base frame is defined for an object).
- The nested `base:` sub-entry rejects `absolute_base` (the base IS the frame).
- Per-waypoint `randomization` rejects `absolute_base`; use the waypoint's own
  `reference: base` field together with `relative` or `absolute_world` instead.

### Operator semantics

For operator entries:

- **Direct form** `arm: {x: ..., y: ...}` randomizes the operator's **home
  end-effector pose** (previously it randomized the base).
- **Nested form** lets you configure `base` and `eef` independently:
  - `base` randomizes `get_base_pose()`
    - for mocap operators this is the virtual base frame
    - for joint-mode operators this is the base reference frame
  - `eef` randomizes the operator home end-effector pose
  - reset updates the stored home EEF pose and then homes the operator to it
  - `base` and `eef` can be configured together

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
- Sampling happens once per `reset()` using the same RNG as entity
  randomization, so `task.seed` reproduces per-waypoint offsets as well.
- Per-waypoint randomization is independent from entity randomization and does
  not participate in `collision_radius` rejection; keep ranges small enough that
  the resulting motion stays reachable.
- Debug mode (`randomization_debug: true`) also cycles per-waypoint extremes.

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
| `relative`        | Sampled values are **added** to the camera's default XML pose.   |
| `absolute_world`  | Sampled values are **absolute world-frame** coordinates/angles.  |

`absolute_base` and entity-name references are **rejected** because cameras have
no operator base frame and do not participate in entity dependency ordering.

### Semantics

- Camera randomization is applied at each `reset()` **after** object and operator
  randomization.
- The default camera pose (the baseline for `relative` mode) is the pose defined
  in the MuJoCo XML, snapshotted during `setup()`.
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

The randomization logic lives in `SceneBackend` (the mixin used by `MujocoTaskBackend`).

### Lifecycle

1. **`setup()`** — snapshots canonical object poses, operator base poses, and
   operator home EEF poses (after `env.reset()` and operator `home()`).

2. **`reset()`** — after restoring the scene to its canonical state:
   - Calls `_record_default_poses()` if not already recorded.
   - Calls `_apply_randomization()` which:
     1. Topologically sorts the randomization keys by entity-reference
        dependency, then groups reference-connected keys into components
        (for example `vase` + `flower`).
     2. Resolves each key to an object handler or operator handler.
     3. For each axis with a `[min, max]` tuple (axes set to `None` are
        skipped), samples a uniform random value.
     4. Combines the sampled values with the default pose according to
        `reference`:
        - `relative` — adds the sampled values to the default pose
          (translation additive; rotation additive in RPY then converted
          back to quaternion).
        - `absolute_world` — replaces the default pose's value on each
          sampled axis with the sampled value; unsampled axes are left at
          their default.
        - `absolute_base` (operator EEF only) — transforms the default
          EEF world pose into the operator's base frame, replaces sampled
          axes there, then transforms the result back to world.
     5. Runs collision rejection on the sampled component. If any member of the
        component overlaps an unrelated accepted participant, the **entire
        component** is re-sampled up to 100 times.
     6. Applies object poses, operator base poses, and operator home EEF
        poses through their respective APIs.
   - Calls `_apply_camera_randomization()` (if `camera_randomization`
     is configured) which samples camera poses and writes them to
     `model.cam_pos` / `model.cam_quat` per environment.
   - Refreshes the viewer.

3. **`TaskRunner.reset()`** — after the backend reset, collects the realized
   poses of all task-relevant entities (stage operators/objects, plus any extra
   entities mentioned in `randomization`) and returns them in
   `TaskUpdate.details["initial_poses"]`. This allows the caller to log initial
   conditions without accessing backend internals.
   For operators, the returned value always contains both `base_pose` and
   `eef_pose`, regardless of whether the randomization entry used the direct
   shorthand or the nested `base`/`eef` form.

### Entity resolution

Each randomization key is resolved in order:
1. `object_handlers[name]` — uses `get_pose()` / `set_pose()`.
2. `operator_handlers[name]`
   - direct form uses `get_end_effector_pose()` /
     `set_home_end_effector_pose()`
   - nested form can additionally randomize the base via `get_base_pose()` /
     `set_pose()`
   - for `reference: absolute_base` the sampler also calls `get_base_pose()`
     to transform between the base frame and world
3. If neither matches, a warning is emitted and the key is skipped.

## Multi-Round Evaluation

Use the `rounds` top-level config key (default 1) to run the task multiple times
with different random seeds:

```bash
aao_demo rounds=10
aao_demo --config-name cup_on_coaster rounds=20
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
aao_demo task.randomization_debug=true rounds=20
```

## Reproducibility

Set `task.seed` to fix the numpy RNG seed:

```bash
aao_demo task.seed=42 rounds=5
```

The same seed produces the same sequence of random poses across runs.
