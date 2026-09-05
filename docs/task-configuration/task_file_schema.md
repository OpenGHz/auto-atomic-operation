# Task File Schema

This page defines the YAML contract for runnable task files. It covers Hydra
composition, top-level blocks, task and stage fields, pose references, and
runner policy. Detailed waypoint semantics, reset/randomization behavior, and
backend-specific state mapping are documented in the linked guides.

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
| `env` | Environment definition, including scene layers, sensors, cameras, operators, and viewer settings. |
| `backend` | Dotted import path to a backend factory, such as `auto_atom.backend.mjc.mujoco_backend.build_mujoco_backend`. |
| `task` | Environment name, ordered stages, seed, initial poses, and randomization. |
| `task_operators` | Task-facing operator definitions keyed by the names used in `stage.operator`; initial-state fields are documented in [Scene Initialization & Randomization](randomization.md), while control settings are backend-specific. |
| `execution` | Optional `TaskRunner` policy: public update boundaries, interval selection, and execution mode. |

The CLI also accepts entry-point options such as `rounds`, `max_updates`,
`print_updates`, `perf_count`, `policy`, `recorder`, and `replay`.  These are
consumed by the corresponding runner and are not part of the `AutoAtomConfig`
task model.  See the [CLI Reference](../getting-started/cli_reference.md) and
the tool pages for their complete schemas.

`env.operators` binds logical names to the environment's native control, state,
and frame resources. `task_operators` supplies task-facing settings for those
logical names. The names commonly match, but the two layers are independent.

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

## Task and stage fields

`task.stages` is an ordered list. The shared stage executor runs stages in list
order.

| Stage field | Description |
| --- | --- |
| `name` | Optional stable human-readable identifier.  Unnamed stages receive a generated `stage_N` name for diagnostics and interval selection. |
| `object` | Target or reference object name.  For `place`, this is the destination reference; the held object is captured separately when the stage starts.  An empty value is allowed for operations that only need a pose from `param`. |
| `site` | Optional site, body, geom, or joint used as the pose-reference origin for `object` and `object_world`; contact and condition checks still use `object`. |
| `operation` | One of `move`, `grasp`, `release`, `pick`, `place`, `push`, `pull`, or `press`. |
| `operator` | Required logical operator name resolved by the backend. |
| `blocking` | Metadata copied into `ExecutionRecord`; defaults to `true`. The shared stage executor remains sequential for either value. |
| `param` | `StageControlConfig`: optional `pre_move`/`post_move` pose lists, an `eef` command, and operation-specific placement or displacement settings. |

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

`require_grasp` is a low-level option for an explicit closing EEF primitive.
`pick` and `pull` enforce target-specific grasp completion intrinsically; an
explicit opening EEF is invalid for those operations. See
[Execution Completion Flow](execution_completion_flow.md) for primitive
construction and operation conditions.

`rotation` is Euler roll/pitch/yaw in radians; `orientation` is an XYZW
quaternion. A waypoint may also set `controlled_frame`, `orientation_goal`,
`relative`, `static`, interpolation step limits, `arc`, `tolerance`, and
per-waypoint `randomization`. See
[Stages & Waypoints](stages_and_waypoints.md) for waypoint semantics and
[Arc Motion Tuning](../ik-motion-control/arc_motion_tuning.md) for arc targets.

An `arc` waypoint must set exactly one of `angle` (radians) or `arc_length`
(metres). `arc_length` is converted at runtime from the measured pivot-to-EEF
radius and cannot be combined with `absolute: true`.

### Pose references

The `reference` field determines the frame in which a waypoint's target values
are written. It does not determine whether the EEF or a held object is
controlled; that is selected by `controlled_frame`. Unless `relative: true` is
set, position and explicitly supplied orientation are absolute within the
selected reference frame.

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
waypoint. `eef` and `eef_world` already snapshot their position and orientation
basis. See [Stages & Waypoints](stages_and_waypoints.md#static-reference-snapshot)
for tracking references, held-object control, and static snapshots.

### Task-level initialization and randomization

The following optional fields live under `task`:

| Field | Description |
| --- | --- |
| `seed` | Episode randomization seed; defaults to `0`. |
| `initial_pose` | Per-object `PoseOverrideConfig` values applied after backend reset and before randomization. |
| `randomization` | Per-object or operator pose ranges sampled at reset. Operator entries use nested `base` and/or `eef` blocks. |
| `camera_initial_pose` | Per-camera `PoseOverrideConfig` values applied after backend reset and before camera randomization. |
| `camera_randomization` | Per-camera relative or absolute-world pose ranges. |
| `randomization_debug` | Cycles through configured extrema before ordinary random sampling when enabled. |

Operator home joints, base/EEF poses, camera fields, pose references, region
sampling, and reset behavior are documented in
[Scene Initialization & Randomization](randomization.md).

### Execution policy

`execution` is optional. Its defaults preserve one controller tick per public
`TaskRunner.update()` call:

```yaml
execution:
  update_boundary: control_tick
  render_internal_updates: true
  max_internal_updates_per_update: 10000
```

`update_boundary` can be `control_tick`, `primitive`, `keypoint`, or `stage`.
`interval_selection` can restrict execution to the interval between two
keypoint boundaries. See
[Stages & Waypoints](stages_and_waypoints.md#task-interval-boundary-selection)
and [Execution Completion Flow](execution_completion_flow.md) for boundary
semantics.

#### Execution modes

`execution.mode` accepts `physical` (the default) and `object_only`:

```bash
aao-demo --config-name dishwasher_plate execution.mode=object_only
```

`object_only` filters operator layers and cameras, clears operator bindings,
and transports a picked object through `held_object` waypoints without
executing physical EEF motions. It is a kinematic geometry/final-placement
check; it does not claim physical grasp, contact, reachability, or collision
success. Unsupported operations (`move`, `push`, `pull`, `press`, standalone
`grasp`/`release`) fail fast in this mode.

Interpolation limits can be configured under `object_motion`:

```yaml
execution:
  mode: object_only
  object_motion:
    max_linear_step: 0.02   # metres per update
    max_angular_step: 0.15  # radians per update
```

See [Stages & Waypoints](stages_and_waypoints.md) for the held-object waypoint
contract and [Execution Completion Flow](execution_completion_flow.md) for
runtime progression and conditions.

## Related

- [Reusing & Creating Tasks](reusing_and_creating_tasks.md) — compose existing task and robot bases.
- [Scene Composition](scene_composition.md) — environment layers and asset assembly.
- [Stages & Waypoints](stages_and_waypoints.md) — waypoint fields, held-object control, and interval boundaries.
- [Scene Initialization & Randomization](randomization.md) — initial state, pose references, and randomization.
- [Action Space](action_space.md) — joint and pose action conventions.
- [Execution Completion Flow](execution_completion_flow.md) — primitive phases and operation conditions.
- [MuJoCo Initialization & Randomization](../mujoco-backend/initialization_randomization.md) — built-in backend state mapping.
- [Implementing a Custom Backend](../mujoco-backend/custom-backend.md) — backend and environment contracts.
