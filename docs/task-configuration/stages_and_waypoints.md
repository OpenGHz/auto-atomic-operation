# Stages & Waypoints

This page documents two less-obvious fields on stage / waypoint
configuration that are easy to miss but frequently needed:

- `StageConfig.site` — re-base `object_world` / `object` references onto a
  site or geometry instead of the stage object's body origin.
- `PoseControlConfig.static` — freeze a tracking reference at the first
  control tick so a rigidly-grasped object does not chase itself.

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

## Related

- [Scene Initialization & Randomization](randomization.md) — per-waypoint
  randomization and the frame semantics of each `reference` mode.
- [IK Control](../ik-motion-control/ik_control.md) — joint-mode control
  chain and the Cartesian step limits that apply per waypoint.
- [Open Door Tuning](../task-tuning/open_door_tuning.md) — a worked example
  combining `site`, `static`, and `transform_resets` for door tasks.
