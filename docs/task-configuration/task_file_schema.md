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
quaternion.  A waypoint may also set `relative`, `static`, interpolation step
limits, `arc`, `tolerance`, and per-waypoint `randomization`.  Their detailed
semantics are documented in [Stages & Waypoints](stages_and_waypoints.md) and
[Arc Motion Tuning](../ik-motion-control/arc_motion_tuning.md).

### Pose references

The `reference` field determines the frame in which a waypoint's local pose is
interpreted.  Unless `relative: true` is set, position and explicitly supplied
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

`static: true` snapshots the resolved reference at the first tick of the
waypoint; it is most commonly used with `object` or `object_world`.  `eef` and
`eef_world` are already snapshotted.  See
[Stages & Waypoints](stages_and_waypoints.md#static-reference-snapshot) for
the interaction with a rigidly grasped object.

### Task-level initialization and randomization

The following optional fields live under `task`:

| Field | Description |
| --- | --- |
| `seed` | NumPy randomization seed; defaults to `0`, which selects an entropy-seeded generator rather than a reproducible fixed seed. |
| `initial_pose` | Per-object position/orientation overrides applied after the XML keyframe and before randomization. |
| `randomization` | Per-object or operator pose ranges sampled at reset.  Operators use nested `base` and/or `eef` entries. |
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
