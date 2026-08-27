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
| `execution` | Optional `TaskRunner` update-boundary and interval policy. |

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
| `initial_pose` | Per-object position/orientation overrides applied after the XML keyframe and before randomization. |
| `randomization` | Per-object or operator pose ranges sampled at reset. Objects and operator `base` / `eef` entries accept either one range or a non-empty `regions` list for disjoint workspaces; operators use the nested form. |
| `camera_initial_pose` | Per-camera pose overrides applied before camera randomization. |
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
