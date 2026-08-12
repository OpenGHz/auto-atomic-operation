# Stages & Waypoints

This page documents four less-obvious fields on task / stage / waypoint
configuration that are easy to miss but frequently needed:

- `AutoAtomConfig.start_after` — reconstruct an earlier task prefix during
  reset and begin rollout after a selected YAML waypoint.
- `StageConfig.site` — re-base `object_world` / `object` references onto a
  site or geometry instead of the stage object's body origin.
- `PoseControlConfig.static` — freeze a tracking reference at the first
  control tick so a rigidly-grasped object does not chase itself.
- `StageControlConfig.displacement_threshold` — per-stage override of the
  distance an object must move before the `displaced` post-condition is
  satisfied.

## Reset after a waypoint

`task.start_after` turns a complete task definition into a suffix-only demo
without duplicating the skipped stages or hard-coding an initial grasp pose.
For example, this configuration reconstructs `pick_source` during reset and
starts the visible rollout at `place_source`:

```yaml
task:
  start_after:
    stage: pick_source
    phase: post_move
    waypoint: 0
```

The selector uses a unique stage name, `pre_move` or `post_move`, and a
zero-based index into that YAML waypoint list. The selected waypoint is
already complete when `reset()` returns; rollout continues with the next
primitive action. An arc still counts as one YAML waypoint even when it is
expanded into multiple internal control actions.

Reset replay uses the normal task semantics:

- scene, operator, camera, and waypoint randomization are applied first;
- waypoint randomization uses a deterministic stream keyed by task seed,
  environment, reset episode, and stage, so normal execution and reset replay
  resolve the same waypoint regardless of batch execution order;
- waypoint references (`world`, `base`, `object_world`, `eef_world`, etc.)
  are resolved through the same runtime path as a normal rollout;
- EEF and already-grasped objects are teleported while preserving each
  object's full EEF-relative SE(3) pose;
- gripper close/open commands use the normal controller and settle logic, so
  a skipped pick must establish a real backend grasp rather than a logical
  attachment;
- skipped stages do not emit execution records and are excluded from the
  current rollout's `total_stages` summary.

Only pose waypoints can be selected. To begin after a close action, select a
following pose waypoint. Backends that opt into this feature must implement
kinematic EEF teleportation and mutable object poses; configurations without
`start_after` do not require those capabilities.

Runnable example:

```bash
aao-demo --config-name pick_and_place_place_only
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
