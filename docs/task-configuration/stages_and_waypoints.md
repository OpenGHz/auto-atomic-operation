# Stages & Waypoints

This page documents four less-obvious fields on task / stage / waypoint
configuration that are easy to miss but frequently needed:

- `TaskFileConfig.execution` — select the public `TaskRunner.update()` boundary,
  choose whether internal ticks are rendered, and optionally run only an
  interval delimited by states before or after two configured keypoints.
- `StageConfig.site` — re-base `object_world` / `object` references onto a
  site or geometry instead of the stage object's body origin.
- `PoseControlConfig.static` — freeze a tracking reference at the first
  control tick so a rigidly-grasped object does not chase itself.
- `StageControlConfig.displacement_threshold` — per-stage override of the
  distance an object must move before the `displaced` post-condition is
  satisfied.

For the base YAML structure, complete stage fields, and pose-reference table,
start with [Task File Schema](task_file_schema.md).

## Task interval boundary selection

`execution.interval_selection` restricts `TaskRunner` / `aao-demo` to an
interval between two explicit keypoint boundaries. It belongs to the task
file's top-level `execution` section, alongside the update-boundary policy:

```yaml
execution:
  update_boundary: control_tick
  render_internal_updates: true
  max_internal_updates_per_update: 10000
  interval_selection:
    start:
      stage: pick_source
      phase: post_move
      waypoint: 0
      side: before
    stop:
      stage: place_source
      phase: post_move
      waypoint: 0
      side: after
    max_fast_forward_updates: 10000

task:
  stages:
    # ...
```

Each endpoint contains:

| Field | Meaning |
| --- | --- |
| `stage` | Exact stage `name`; unnamed stages can use their generated `stage_N` name |
| `phase` | `pre_move`, `eef`, or `post_move` |
| `waypoint` | Zero-based index in that phase; `eef` is a singleton and only accepts `0` |
| `side` | `before` or `after` the referenced keypoint; defaults to `before` for `start` and `after` for `stop` |

Both endpoints use the same `TaskKeypointConfig` schema. Its standalone
`side` default is `None`; `IntervalSelectionConfig` resolves that adaptive
value by endpoint role and exposes the concrete `before` / `after` value when
serialized.

`side` selects one of the two state boundaries around a keypoint:

- `before` is the state immediately before the referenced configured
  keypoint executes. Neither that keypoint's action nor a condition bound to
  its completion has run.
- `after` is the state after the entire configured keypoint completes,
  including any condition attached to its completion boundary.

The role-specific defaults therefore run from immediately before `start`
through immediately after `stop`. Set either field explicitly when a config
should remain self-describing.

Migration note: interval configs created before `side` existed effectively
started at `after`. They still validate when `side` is omitted, but now use
the requested `start.side: before` default. Add `start.side: after` to
preserve the previous reset and first-update behavior.

Top-level `interval_selection`, `update_boundary`, `render_internal_updates`,
`max_internal_updates_per_update`, and `max_fast_forward_updates` are rejected
with their expected `execution...` path so misplaced settings cannot be
silently ignored.

### Public update boundaries

`execution.update_boundary` controls when one public `TaskRunner.update()`
call returns:

| Value | Return boundary |
| --- | --- |
| `control_tick` | One controller update; this is the default and preserves the previous behavior |
| `primitive` | One runtime primitive action, such as one pose action, one gripper action, or one arc sub-action |
| `keypoint` | One complete YAML waypoint; all primitive sub-actions generated from an arc waypoint finish before returning |
| `stage` | One complete stage, including its operation-condition checks |

For `primitive`, `keypoint`, and `stage`, `TaskRunner` performs normal
controller updates internally until each selected environment reaches its own
next boundary. Faster batch environments stop at their first boundary instead
of advancing again while slower environments catch up.

`execution.max_internal_updates_per_update` (default `10000`) is a
per-environment safety limit for those internal controller updates within one
public `update()`. Exhausting it terminates that environment with an
`internal_update_limit_exceeded` failure.

The interval selection's `max_fast_forward_updates` (default `10000`) is a
separate per-environment limit used only while `reset()` advances to `start`.
Increasing one limit does not change the other, and controller-level timeouts
still apply independently.

The endpoints refer to YAML waypoints, not internal controller ticks. If an
arc waypoint expands into several primitive actions, `before` is before its
first sub-action and `after` is only after its final sub-action completes.

### Viewer updates at public boundaries

`execution.render_internal_updates` controls only the interactive viewer:

- `true` (default) refreshes after every controller update and applies
  `env.viewer.step_delay` each time, preserving the previous animation.
- `false` runs all internal physics and controller updates without refreshing
  or sleeping, then refreshes the viewer once when `reset()` or the public
  `update()` returns. The final boundary refresh does not apply `step_delay`.

This setting does not teleport the robot, skip collision/contact handling, or
change camera observations. It only coalesces passive-viewer refreshes. With a
long `stage` boundary, the viewer may appear unresponsive until that public
update reaches its boundary.

### Reset and stop boundary behavior

- With `start.side: before`, `reset()` fast-forwards only to the state
  before `start`; the first public `update()` begins executing that keypoint.
  If `start` is the task's first keypoint, reset performs no prefix updates.
- With `start.side: after`, reset runs the existing control state machine
  through `start`, including a condition attached to its completion. The
  first public `update()` proceeds to the following keypoint.
- With `stop.side: before`, the interval reports successful completion at
  the state before `stop`; that keypoint and its completion condition do not
  run.
- With `stop.side: after`, `stop` completes and any condition attached to
  its completion passes before the interval returns `done=true` and
  `success=true`. The following keypoint does not run.
- Interval stop has priority over `execution.update_boundary`. For example, a
  stop boundary in the middle of a stage returns immediately even when the
  configured public boundary is `stage`.
- Boundary order is `before(K0) < after(K0) < before(K1) < after(K1) ...`.
  For one keypoint, `before -> after` executes exactly that keypoint;
  `before -> before` and `after -> after` are valid empty intervals that
  finish during reset; `after -> before` is invalid. `after(K)` and
  `before(next K)` describe the same physical state, so that adjacent pair is
  also an empty interval completed during reset.
- A stop in the middle of a stage completes the selected interval but does not
  fabricate a successful stage-level `ExecutionRecord`. A stop at the final
  stage action with `side: after` still runs the normal stage
  post-condition.
- A controller failure, timeout, or operation-condition failure during reset
  fast-forward remains a task failure; it is never overwritten as interval
  success.

Fast-forward uses the same physics, IK, contact handling, randomization, pose
references, and timeouts as ordinary updates. It does not teleport state.
Set `execution.render_internal_updates: false` to show only the selected start
boundary state instead of animating the reset prefix.

`TaskUpdate.details[env_index]["interval_selection"]` reports the selected
endpoints, current interval event, configured safety limit, and (on reset)
`fast_forward_updates`. In execution summaries, `updates_used` counts all public
rollout `update()` calls, while `timed_updates` excludes the first warmup call.
`elapsed_time_sec` accumulates only those timed calls, without interactive waits
or console output, and `sim_time_sec` uses the controller updates actually
executed inside all public calls. Reset fast-forward ticks are excluded from
these rollout metrics.

Invalid selections fail during config validation: unknown or ambiguous stage
names, absent phases, out-of-range waypoint indexes, and `start` ordered after
`stop` after accounting for `side` are rejected. Omitting
`execution.interval_selection` preserves
full-task execution; omitting all of `execution` also preserves the default
one-control-tick update behavior.

> `execution.interval_selection`, non-`control_tick` update boundaries, and
> `render_internal_updates: false` apply to `TaskRunner` / `aao-demo`.
> `PolicyEvaluator` / `aao-eval` rejects them:
> reset cannot synthesize external policy actions, and every policy control
> tick requires a newly supplied action.

## Stage reference site

By default, waypoints with `reference: object_world` or `reference: object`
resolve against the pose of the stage's `object` body. When the grasp
point is offset from the body origin (doors, levers, handles, cups) it is
usually more natural to anchor the reference to a site attached to the
object than to the body itself.

Set `stage.site` to any site / body / geom / joint name in the MuJoCo
model:

```yaml
stages:
  - name: grasp_handle
    operator: arm
    object: door
    site: handle_grasp_${door_side}_site   # anchor pose onto the handle site
    operation: pick
    param:
      pre_move:
        - position: [0.0, 0.0, 0.12]
          reference: object_world           # resolved against handle_grasp_* site
        - position: [0.0, 0.0, 0.0]
          reference: object_world
```

Semantics:

- When `site` is set, its world pose replaces the `object` body's pose as
  the reference origin for `object_world` waypoints.
- For `reference: object`, the site's orientation is also used as the
  reference orientation.
- `site` only affects pose reference resolution. The stage's `object`
  field is still used for:
  - contact detection
  - GS rendering mask
  - `set_pose` / randomization
  - arc pivot fallback
- Leave `site: null` (the default) to fall back to the `object` body pose.

## Static reference snapshot

`PoseControlConfig.static: true` snapshots the reference frame at the
first tick of a waypoint, turning a tracking target into a fixed
world-frame target.

When is this needed? By default, `object` / `object_world` references
are re-evaluated on every control tick so the target follows the object
as it moves. That is correct when the object moves independently of the
gripper, but becomes a problem when the gripper is **rigidly gripping**
the object — the reference moves with the gripper, so the residual
between the current pose and the target never closes and the waypoint
never completes.

Typical case: a post-grasp retract or place motion:

```yaml
stages:
  - name: retract_after_grasp
    operator: arm
    object: door
    site: handle_grasp_left_site
    operation: place
    param:
      post_move:
        - position: [0.0, 0.0, 0.15]       # lift 15cm along site +Z
          reference: object
          static: true                      # snapshot at first tick
```

Semantics:

- `static: true` freezes the reference pose at the first tick of this
  waypoint, giving a fixed world-frame target for the rest of the
  motion.
- `EEF` / `EEF_WORLD` references are always snapshotted — `static` is a
  no-op for them.
- `relative` waypoints operate against the current pose at their first
  tick regardless of this flag.

## Stage displacement threshold

Operations whose success constraint is `displaced` (notably `push` and
`pull`) judge success by checking `||current_pos - initial_pos|| >
threshold` on the stage object. The default threshold is `0.01` m
(`MujocoObjectHandler.__init__`), which is fine for picking up a block
from a table but easy to satisfy spuriously when the stage object
articulates instead of translating — for example, a door whose lever
rotates but whose body never opens.

Set `displacement_threshold` under `param` to override the default for a
single stage:

```yaml
stages:
  - name: open_door
    object: handle_body_phys
    operation: push
    operator: arm
    param:
      # Without this the default 1 cm threshold can be satisfied by the
      # lever rotation alone, and the stage would report success even if
      # the door body itself never moved.
      displacement_threshold: 0.10
      pre_move: ...
      post_move: ...
```

Semantics:

- The value (in metres) is forwarded into
  `SceneBackend.is_object_displaced(...)` only for the single stage that
  declares it; other stages keep using the backend default.
- Only meaningful when the operation's success constraint is
  `displaced`. On other operations the field is ignored.
- See [Backend Conditions](../mujoco-backend/mujoco_backend_conditions.md#3-displaced-condition)
  for how `displaced` is computed.

## Related

- [Scene Initialization & Randomization](randomization.md) — per-waypoint
  randomization and the frame semantics of each `reference` mode.
- [IK Control](../ik-motion-control/ik_control.md) — joint-mode control
  chain and the Cartesian step limits that apply per waypoint.
- [Open Door Tuning](../task-tuning/open_door_tuning.md) — a worked example
  combining `site`, `static`, and `transform_resets` for door tasks.
