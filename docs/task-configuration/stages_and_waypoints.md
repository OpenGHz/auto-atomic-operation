# Stages & Waypoints

This page documents six less-obvious fields on task / stage / waypoint
configuration that are easy to miss but frequently needed:

- `AutoAtomConfig.start_after` — use the existing kinematic teleport
  fast-forward and begin rollout after a selected YAML waypoint.
- `AutoAtomConfig.physical_replay` — independently execute the complete task
  physics up to an absolute frame or a waypoint-relative frame.
- `AutoAtomConfig.stop_at` — finish successfully at an absolute task frame or
  a waypoint-relative frame instead of executing the remaining suffix.
- `StageConfig.site` — re-base `object_world` / `object` references onto a
  site or geometry instead of the stage object's body origin.
- `PoseControlConfig.static` — freeze a tracking reference at the first
  control tick so a rigidly-grasped object does not chase itself.
- `StageControlConfig.displacement_threshold` — per-stage override of the
  distance an object must move before the `displaced` post-condition is
  satisfied.

## Reset-time task prefixes

There are two independent ways to turn a complete task definition into a
suffix-only rollout. `task.start_after` keeps the existing teleport mechanism;
`task.physical_replay` is a separate full-physics mechanism. They cannot be
configured together.

### Complete physical replay (`task.physical_replay`)

Use `physical_replay` when the prefix contains contacts or articulated scene
motion. Every controller update, MuJoCo physics substep, pre-step callback,
grasp, release, and stage condition is executed exactly as in a normal
rollout. Only viewer synchronization, viewer `step_delay`, observation capture,
and prefix execution records are skipped.

This example physically executes the complete pick stage during `reset()` and
starts the visible rollout at `place_source`:

```yaml
task:
  physical_replay:
    stage: pick_source
    phase: post_move
    waypoint: 0
```

An optional offset continues physical execution after that waypoint:

```yaml
task:
  physical_replay:
    stage: pick_source
    phase: pre_move
    waypoint: 1
    frame_offset: 20
```

`frame_offset: 0` stops on the control tick where the selected waypoint's last
internal primitive reports `REACHED`. `frame_offset: N` then executes exactly
N additional controller updates; the offset may cross action and stage
boundaries. The task must contain enough subsequent frames for the requested
offset.

Alternatively, stop at an absolute task frame without naming a waypoint:

```yaml
task:
  physical_replay:
    frame: 300
```

Frame numbering is defined at the task controller boundary:

- frame `0` is the randomized backend state immediately after reset, before
  any task action;
- frame `N` is the state after exactly `N` normal controller updates;
- one controller frame contains the backend's configured physics substeps, so
  this is not a raw MuJoCo substep or a downsampled video frame;
- for a complete task recorded without `physical_replay`, the reset
  observation from `record_demo.py` is frame `0`; with `physical_replay`, its
  first observation is instead the requested replay target state.

Stopping in the middle of a primitive preserves its action object, resolved
target, arc snapshot, controller progress, velocities, contacts, and scene
joint state. The first rollout update therefore continues from the exact next
physical tick. No object is manually attached to or moved with the gripper;
drawer and door state emerges entirely from the replayed physical process.

If the task fails, ends before the requested target, or exceeds the global
replay safeguard, `reset()` raises an error instead of returning a silently
clamped or non-physical state. `gaussian_render.share_physics: true` is not
supported for physical reset replay because its virtual batch entries share
one physical world.

### Early successful endpoint (`task.stop_at`)

`stop_at` independently selects where the task should finish. It uses the
same coordinates as `physical_replay`, so the two fields can define a visible
segment of one complete physical task:

```yaml
task:
  # Start the visible rollout after physically executing the complete pick.
  physical_replay:
    stage: pick_source
    phase: post_move
    waypoint: 0

  # End as soon as the second place approach waypoint is reached.
  stop_at:
    stage: place_source
    phase: pre_move
    waypoint: 1
```

An endpoint can include additional controller frames after its waypoint:

```yaml
task:
  stop_at:
    stage: place_source
    phase: pre_move
    waypoint: 0
    frame_offset: 20
```

Or it can use an absolute full-task frame:

```yaml
task:
  physical_replay:
    frame: 300
  stop_at:
    frame: 450
```

Absolute frames do not restart at zero after replay: in the last example,
`reset()` executes frames 1 through 300 and the visible rollout executes only
frames 301 through 450. A waypoint offset follows the same rule and may cross
primitive-action and stage boundaries.

When the endpoint is reached, the environment reports `done: true`,
`success: true`, and a `details.stop_at` entry containing the resolved task
frame. The remaining action/stage suffix is not executed, and an incomplete
stage does not emit a stage-completion record. Controller or task failures
that occur before the endpoint still fail normally.

The endpoint must be reachable from the selected start:

- a target already passed by `physical_replay`, or beyond the natural task
  end, raises an error instead of silently clamping;
- `stop_at.frame` cannot be combined with kinematic `start_after`, because a
  teleport prefix has no physical frame count;
- a waypoint-based `stop_at` may be combined with `start_after`, but it must
  occur after the start waypoint.

Runnable segment example:

```bash
aao-demo --config-name pick_and_place_physical_segment
```

### Existing teleport fast-forward (`task.start_after`)

The existing `start_after` configuration remains unchanged:

```yaml
task:
  start_after:
    stage: pick_source
    phase: post_move
    waypoint: 0
```

Teleport fast-forward is faster for free-space prefixes, but pose waypoints do
not execute the intervening physics. It teleports the EEF and manually preserves
the relative pose of already-grasped objects; gripper close/open commands
still use the normal controller and settle logic. This mode is unsuitable for
reconstructing drawer, door, lever, deformable, or other contact-driven state.

The waypoint selector uses a unique stage name, `pre_move` or `post_move`, and
a zero-based index into that YAML waypoint list. The selected waypoint is
already complete when `reset()` returns. An arc still counts as one YAML
waypoint even when expanded into multiple internal primitives.

Both mechanisms use the normal task semantics where applicable:

- scene, operator, camera, and waypoint randomization are applied first;
- waypoint randomization uses a deterministic stream keyed by task seed,
  environment, reset episode, and stage, so normal execution and reset replay
  resolve the same waypoint regardless of batch execution order;
- waypoint references (`world`, `base`, `object_world`, `eef_world`, etc.)
  are resolved through the same runtime path as a normal rollout;
- `physical_replay` uses the same controller, physics, callbacks, conditions,
  and action state machine as normal rollout;
- `start_after` preserves each already-grasped object's full EEF-relative
  SE(3) pose and physically settles gripper close/open commands;
- skipped stages do not emit execution records and are excluded from the
  current rollout's `total_stages` summary.

Only pose waypoints can be selected by waypoint coordinates. To begin after a
close action, select a following pose waypoint. `start_after` requires
kinematic EEF teleportation and mutable poses for carried objects;
`physical_replay` does not require either capability.

Runnable example:

```bash
aao-demo --config-name pick_and_place_place_only
aao-demo --config-name pick_and_place_place_only_physical_replay
```

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
