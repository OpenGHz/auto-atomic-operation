# Task File Schema

This page is the compact reference for a runnable YAML task file.  It restores
the configuration reference that used to live in the root README, while
following the current split between task configuration, environment
configuration, and runner options.

## Loading and composition

Task files are loaded through [Hydra](https://hydra.cc) and OmegaConf.  A
`defaults` list can compose a robot, scene, gripper, or rendering building
block; later entries override earlier entries.  `_self_` marks where the
current file enters that merge order.  Runnable variants normally put it last
so their local values win; a reusable building block may deliberately put it
earlier so a later mixin overrides its defaults.  `aao-demo` and `aao-eval`
resolve configs from `./aao_configs/` relative to the current working
directory.  For one-off changes, pass a Hydra override instead of cloning a
nearly identical file; see [Reusing & Creating Tasks](reusing_and_creating_tasks.md).

Hydra instantiates the `env` block before the backend factory runs.  The
factory receives the validated `task` and `task_operators` blocks and resolves
the registered environment by `task.env_name`.

## Top-level structure

A composed runnable config normally exposes these top-level keys:

| Key | Purpose |
| --- | --- |
| `env` | Hydra environment definition.  For the built-in backend this is usually a `BatchedUnifiedMujocoEnv` with `scene`, sensors, cameras, operators, and viewer settings. |
| `backend` | Dotted import path to a backend factory, such as `auto_atom.backend.mjc.mujoco_backend.build_mujoco_backend`. |
| `task` | `env_name`, ordered stages, seed, initial poses, and randomization. |
| `task_operators` | Logical operator definitions keyed by name.  Backend-specific control and initial-state settings belong here. |
| `execution` | Optional `TaskRunner` policy, including update boundaries, interval selection, and the `physical` / `object_only` execution mode. |

The CLI also accepts entry-point options such as `rounds`, `max_updates`,
`print_updates`, `perf_count`, `policy`, `recorder`, and `replay`.  These are
consumed by the corresponding runner and are not part of the `AutoAtomConfig`
task model.  See the [CLI Reference](../getting-started/cli_reference.md) and
the tool pages for their complete schemas.

`env.operators` describes how a simulator environment binds logical names to
XML actuators and sensors.  `task_operators` describes the operators exposed
to the task backend.  They commonly use the same names, but they are separate
configuration layers.

## Minimal task file

This example uses the same Hydra registration pattern as the custom-backend
guide.  A concrete MuJoCo task normally composes one of the `basis_*` configs
instead of spelling out every environment field here.

```yaml
env:
  _target_: auto_atom.runtime.ComponentRegistry.register_env
  name: my_env
  env:
    _target_: my_package.basis.my_env.MyEnv
    scene_path: path/to/scene.xml
    batch_size: 1

backend: my_package.backend.my_backend.build_my_backend

task:
  env_name: my_env
  seed: 42
  stages:
    - name: pick_cup
      object: cup
      operation: pick
      operator: arm_a
      param:
        pre_move:
          - position: [0.45, -0.10, 0.08]
            rotation: [0.0, 1.57, 0.0]
            reference: object_world
        eef:
          close: true

    - name: place_on_shelf
      object: shelf
      operation: place
      operator: arm_a
      param:
        pre_move:
          - position: [0.10, 0.25, 0.16]
            orientation: [0.0, 0.0, 0.0, 1.0]
            reference: world
        eef:
          close: false

task_operators:
  arm_a: {}
```

### Operator definitions and initial state

`task_operators` is the task-facing operator contract.  Its mapping key is the
logical name used by `stage.operator`; the key is also the operator name, so a
separate `name` field is normally unnecessary.  The corresponding
`env.operators` entry binds that name to the simulator's root body, actuators,
sites, and sensors.  Keep these two layers distinct: `env.operators` describes
the physical binding, while `task_operators.<name>` supplies task control and
initial-state values.

The optional `initial_state` is applied after the scene keyframe reset and
before operator randomization baselines are recorded. It can set the base
pose, a home EEF pose, or raw qpos values for the operator's declared arm
joints:

```yaml
task_operators:
  arm_a:
    initial_state:
      joint_positions:
        joint1: 0.0
        joint2: -1.5
        joint3: 0.0
        joint4: -0.8
        joint5: 0.0
        joint6: 0.0
        joint7: 0.0
      eef: 0.0                               # gripper control value
      base_pose:
        # A named site/body/geom/joint is resolved in the composed scene.
        reference: door__handle_grasp_center
        position: [0.25, -0.47, -0.10]       # local x/y/z in that frame
        orientation: [0.0, 0.0, 0.7071, 0.7071]  # local xyzw quaternion
```

`joint_positions` is the canonical raw-qpos home representation for an
operator's arm. It is mutually exclusive with `eef_pose`; use `eef` separately
when configuring the gripper control value. Joint names are resolved against
`env.operators.<name>` and may not target passive scene joints. The lower-level
`env.initial_joint_positions` mapping remains available for scene joints and
for basis defaults; a task can set it to `null` to clear inherited entries.
For gripper-only control without raw joint qpos, use `eef` instead of
`joint_positions`.

When an EEF pose is the desired arm-home source instead, omit
`joint_positions` and use the existing structured form:

```yaml
task_operators:
  arm_a:
    initial_state:
      eef_pose:
        reference: base
        position: [0.32, 0.0, 0.18]
        orientation: [0.0, 1.5708, 0.0]
```

`base_pose` and `eef_pose` both use the shared `PoseOverrideConfig` model.  A
pose may provide only `position`, only `orientation`, or both; an omitted
component keeps the current fallback pose after it is expressed in the chosen
reference frame.  Use `orientation: [x, y, z, w]` for an XYZW quaternion or
three values in RPY order (`[roll, pitch, yaw]`) for Euler angles.  Position and
RPY orientation also accept expanded mappings. Their optional component-level
`reference` becomes the default for all contained axes, while
`{value: ..., reference: ...}` overrides one axis. The precedence is
axis-level > component-level > pose-level. The accepted reference forms are:

| Owner | Built-in references | Named references |
| --- | --- | --- |
| object `task.initial_pose` | `world` | MuJoCo site, body, geom, or joint |
| camera `task.camera_initial_pose` | `world` | MuJoCo site, body, geom, or joint |
| operator `base_pose` | `world` | MuJoCo site, body, geom, or joint |
| operator `eef_pose` | `world`, `base` | MuJoCo site, body, geom, or joint; `<operator>.base` / `<operator>.eef` |

Named references are setup/reset anchors, not live tracking references.  For
object `initial_pose` entries, a reference that exactly matches another
configured object key creates a dependency; keys are applied in topological
order and circular references are rejected before any scene mutation.
References to scene frames that are not configured object keys are resolved
from the current reset baseline.  The backend resolves all operator base poses
first, then EEF poses, and finally records the resolved values as the
randomization baselines.  If the referenced articulated body moves later in a
Stage, the robot does not follow it.  For a compact EEF
home pose, the six-value form `[x, y, z, yaw, pitch, roll]` is also accepted and
is interpreted as a complete world-frame pose; use the structured form when a
reference frame or partial override is needed.

`PoseOverrideConfig` validates compact `position` as exactly three finite values
and compact `orientation` as either three finite RPY values or four finite,
non-zero quaternion values. Expanded position/RPY mappings validate each
configured component as a finite scalar. The flat EEF form is exactly six finite
values. These models are frozen, preventing accidental in-place mutation after
task-file loading.

#### Python API migration (intentional breaking change)

`PoseOverrideConfig` is now the single public model for structured setup-time
poses.  The former `InitialPoseConfig` and `ArmPoseConfig` classes were removed
instead of kept as aliases, so Python callers must update imports and type
annotations:

| Before | Now |
| --- | --- |
| `InitialPoseConfig` for `task.initial_pose` | `PoseOverrideConfig` |
| `ArmPoseConfig` for operator `base_pose` / structured `eef_pose` | `PoseOverrideConfig` |

The YAML object keys (`position`, `orientation`, `reference`) are unchanged for
existing structured entries, but the structured operator orientation semantics
are intentionally normalized.  `task.initial_pose` already used RPY
`[roll, pitch, yaw]`; the former `ArmPoseConfig` operator fields used the
historical `[yaw, pitch, roll]` order and their values must therefore be
reordered when migrating to `PoseOverrideConfig`.  The canonical orientation
order is now RPY `[roll, pitch, yaw]` (or XYZW quaternion) everywhere in the
structured model.
The six-value operator EEF shorthand remains a deliberately supported input
format, with its historical `[x, y, z, yaw, pitch, roll]` order; it is converted
at the configuration boundary and is not a second pose model.  Code that
constructs these models directly should import them from `auto_atom.framework`
or the package root (`auto_atom.PoseOverrideConfig`).

For a joint-mode operator, changing `base_pose` relocates the configured root
body so the physical arm and its base-frame IK agree. A pure mocap operator
keeps its registered physical mocap home and changes only the virtual base
frame used for world/base conversion; use the operator's EEF pose or mocap home
configuration when the physical mocap body itself must move.

Example of a mixed-reference base pose:

```yaml
base_pose:
  reference: door__handle_grasp_center
  position:
    x: 0.2474
    y: -0.4666
    z: {value: -0.1, reference: world}
```

The `eef` scalar controls only the gripper command and does not alter either
pose baseline.  If an `initial_state` field is omitted, the corresponding
keyframe/registration pose remains the fallback.

### Object-only execution

Set one global execution override to run a task without loading a physical
operator and without executing EEF approach/grasp/retract motions:

```bash
aao-demo --config-name dishwasher_plate execution.mode=object_only
```

`object_only` removes scene layers and cameras marked `role: operator` (legacy
robot-layer paths and wrist-camera names are recognized as a fallback), clears
operator bindings, and keeps only the scene objects. A `pick` establishes a
logical carried-object identity; a `place` moves that object directly through
its `controlled_frame.kind: held_object` waypoints using linear/quaternion
interpolation, then releases it. EEF-only place waypoints are intentionally
skipped and a place stage without a held-object waypoint fails validation.

The default is `execution.mode: physical`, which preserves the ordinary
operator-controlled behavior. Object-only is a kinematic/ghost transport mode:
it validates task geometry and final placement, but does not claim physical
grasp, contact, reachability, or collision success. Unsupported operations
(`move`, `push`, `pull`, `press`, standalone `grasp`/`release`) fail fast in
this mode. Tune interpolation limits with:

```yaml
execution:
  mode: object_only
  object_motion:
    max_linear_step: 0.02   # metres per update
    max_angular_step: 0.15  # radians per update
```

Layers and cameras can opt into filtering explicitly:

```yaml
env:
  scene:
    layers:
      - kind: mjcf
        path: assets/xmls/robots/my_robot.xml
        role: operator
  cameras:
    - name: wrist_cam
      role: operator
```

For a complete runnable example, inspect `aao_configs/mock.yaml`.  Use
`aao-info` to find current MuJoCo tasks and robot variants.

## Task and stage fields

`task` contains an ordered `stages` list.  The current shared stage executor
runs them sequentially in list order.

| Stage field | Description |
| --- | --- |
| `name` | Optional stable human-readable identifier.  Unnamed stages receive a generated `stage_N` name for diagnostics and interval selection. |
| `object` | Target or reference object name.  For `place`, this is the destination reference; the held object is captured separately when the stage starts.  An empty value is allowed for operations that only need a pose from `param`. |
| `site` | Optional site/body/geom/joint used as the pose-reference origin for `object` and `object_world`; contact and condition checks still use `object`. |
| `operation` | One of `move`, `grasp`, `release`, `pick`, `place`, `push`, `pull`, or `press`. |
| `operator` | Required logical operator name resolved by the backend. |
| `blocking` | Compatibility metadata copied into `ExecutionRecord`; defaults to `true`.  The current shared stage executor remains sequential for either value. |
| `param` | `StageControlConfig`: optional `pre_move`/`post_move` pose lists, an `eef` command, and operation-specific placement/displacement settings. |

The basic `param` shape is:

```yaml
param:
  pre_move:
    - position: [x, y, z]
      rotation: [roll, pitch, yaw]
      # Or replace rotation with: orientation: [x, y, z, w]
      reference: world
  eef:
    close: true
    joint_positions: []
    require_grasp: false
  post_move:
    - position: [x, y, z]
      reference: eef_world
  placed_reference: object
  placed_tolerance:
    position: [0.03, 0.03, null]
    orientation: null
  displacement_threshold: 0.01
```

`require_grasp` is a low-level option for explicit closing EEF primitives.
`pick` and `pull` enforce target-specific grasp completion intrinsically, so
omit this field for those operations; their compiled closing primitive uses
`true` even when an explicit closing EEF supplies other parameters. An
explicit opening EEF is invalid for `pick` and `pull`.

`rotation` is Euler roll/pitch/yaw in radians; `orientation` is an XYZW
quaternion. A waypoint may instead declare an `orientation_goal`, and may
select an EEF or held-object `controlled_frame`. It may also set `relative`,
`static`, interpolation step limits, `arc`, `tolerance`, and per-waypoint
`randomization`. Their detailed semantics are documented in
[Stages & Waypoints](stages_and_waypoints.md) and
[Arc Motion Tuning](../ik-motion-control/arc_motion_tuning.md).

### Controlled frame and orientation goal

A pose waypoint answers two independent questions:

| Question | Field | Default |
| --- | --- | --- |
| Which frame must reach the goal? | `controlled_frame` | `{kind: eef}` |
| In which frame are the waypoint's target values written? | `reference` | `auto` |

Use `controlled_frame.kind: held_object` to specify the desired pose of the
object already held by the stage operator. An optional `frame` selects a named
object-local frame; omit it to control the held object's root. After a verified
grasp, AAO measures the actual rigid EEF-to-object relationship and uses it to
convert the object-frame goal into the concrete EEF command. The task therefore
does not need to predict the EEF pose from an assumed grasp transform.

For example, this waypoint puts a held plate frame at a rack-slot site while
constraining only the plate normal:

```yaml
- controlled_frame:
    kind: held_object
    frame: plate_center_site
  position: [0.0, 0.0, 0.0]
  reference: object
  orientation_goal:
    kind: axis_alignment
    controlled_axis: [0.0, 0.0, 1.0]  # plate normal in the plate frame
    target_axis:
      vector: [0.0, 1.0, 0.0]         # desired direction in the rack-site frame
      reference: object
    direction: same
```

Here, waypoint `reference: object` places the controlled frame relative to the
stage object or its `site`. The separate `target_axis.reference: object`
expresses the target direction in that same object/site orientation. The two
references need not match: the position can be object-relative while the
target axis is expressed in `world` or `base`.

`orientation_goal` is a discriminated choice:

| `kind` | Fields | Meaning |
| --- | --- | --- |
| `fixed` | `quaternion_xyzw: [x, y, z, w]` | Constrain the complete controlled-frame orientation. |
| `axis_alignment` | `controlled_axis`, `target_axis`, `direction` | Constrain one controlled-frame axis and leave twist about that axis free. |

Both axis vectors must be finite unit vectors. `controlled_axis` is expressed
in the controlled frame. `target_axis.vector` accepts an independent
`reference` of `world`, `base`, or `object`; the last mode uses the stage
`site` orientation when one is configured. `direction` has these meanings:

| Value | Axis relationship |
| --- | --- |
| `same` | Controlled axis points in the target direction. |
| `opposite` | Controlled axis points against the target direction. |
| `either` | Both polarities are equivalent; the smaller reorientation is used. |

For `axis_alignment`, AAO applies only the shortest swing needed to align the
axis and retains the current twist. Completion checks the axis-angle error,
not full-quaternion equality, so symmetric objects do not perform unnecessary
rotation about their symmetry axis.

Configuration validation rejects ambiguous or unsupported combinations:

- `orientation_goal` cannot be combined with legacy `orientation` or
  `rotation` fields.
- Rotational waypoint randomization (`roll`, `pitch`, or `yaw`) cannot be
  combined with `orientation_goal`; position randomization remains valid.
- `axis_alignment` currently does not support `relative: true`.
- No `orientation_goal` kind currently supports `arc`.
- `controlled_frame.kind: held_object` does not support `arc`.
- An EEF controlled frame cannot name an object-local `frame`.

A held-object waypoint also fails at execution time if no verified grasp
binding exists, the held-object identity changes, or the waypoint is requested
after release. See
[Stages & Waypoints](stages_and_waypoints.md#held-object-control-and-grasp-binding)
for the lifecycle and a complete Stage example.

### Pose references

The `reference` field determines the frame in which a waypoint's local target
is interpreted; it does not determine whether the EEF or a held object is
controlled. Unless `relative: true` is set, position and explicitly supplied
orientation are absolute within that reference frame.

| Reference | Meaning |
| --- | --- |
| `world` | Fixed world frame. |
| `base` | The current operator base frame. |
| `eef` | The operator's end-effector frame captured when the waypoint starts. |
| `object` | The target object's current position and orientation; it tracks object motion.  A stage `site` replaces this pose when configured. |
| `object_world` | The target object's current position with world-axis orientation.  It tracks object translation; a stage `site` supplies the position anchor. |
| `eef_world` | The EEF position captured when the waypoint starts, with world-axis orientation.  It is a snapshot rather than a tracking reference. |
| `auto` | Uses `object_world` when the stage has an object, otherwise `base`. |

`static: true` snapshots the complete semantic goal at the first tick of the
waypoint; it is most commonly used with `object` or `object_world`. `eef` and
`eef_world` already snapshot the waypoint's position/full-orientation basis,
but an independently object- or base-referenced `target_axis` remains live
unless `static: true` freezes the complete goal. See
[Stages & Waypoints](stages_and_waypoints.md#static-reference-snapshot) for
the interaction with a rigidly grasped object.

### Task-level initialization and randomization

The following optional fields live under `task`:

| Field | Description |
| --- | --- |
| `seed` | NumPy randomization seed; defaults to `0`, which selects an entropy-seeded generator rather than a reproducible fixed seed. |
| `initial_pose` | Per-object `PoseOverrideConfig` values applied after the XML keyframe and before randomization. `reference` may be `world` or a named scene element. |
| `randomization` | Per-object or operator pose ranges sampled at reset. Objects and operator `base` / `eef` entries accept either one range or a non-empty `regions` list for disjoint workspaces; operators use the nested form. |
| `camera_initial_pose` | Per-camera `PoseOverrideConfig` values applied after the keyframe and before camera randomization. `reference` may be `world` or a named scene element. |
| `camera_randomization` | Per-camera relative or absolute-world pose ranges. |
| `randomization_debug` | Cycle through configured extrema before ordinary random sampling when enabled. |

See [Scene Initialization & Randomization](randomization.md) for frame modes,
collision rejection, operator baselines, and camera behavior.

### Execution policy

`execution` is optional.  Its defaults preserve one controller tick per public
`TaskRunner.update()` call:

```yaml
execution:
  update_boundary: control_tick
  render_internal_updates: true
  max_internal_updates_per_update: 10000
```

Use `primitive`, `keypoint`, or `stage` boundaries to aggregate internal
controller updates, and `interval_selection` to run only between two keypoint
boundaries.  See [Stages & Waypoints](stages_and_waypoints.md#task-interval-boundary-selection)
and [Execution Completion Flow](execution_completion_flow.md).

## Related

- [Reusing & Creating Tasks](reusing_and_creating_tasks.md) — compose existing task and robot bases.
- [Scene Composition](scene_composition.md) — host MJCF and asset-assembly layers.
- [Action Space](action_space.md) — joint and pose action conventions.
- [Execution Completion Flow](execution_completion_flow.md) — operation phases and conditions.
- [Implementing a Custom Backend](../mujoco-backend/custom-backend.md) — backend and environment contracts.
