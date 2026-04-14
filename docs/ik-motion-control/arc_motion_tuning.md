# Arc Motion Tuning Guide

## Overview

Arc motion (`arc` in YAML) rotates the EEF around a pivot point along an axis. This is essential for tasks involving hinged mechanisms (door handles, doors, levers). This document records lessons learned from tuning the open door task.

## Critical: axis is in WORLD frame

The `axis` field in arc config specifies the rotation axis in **world coordinates**, NOT the joint's local frame.

MuJoCo joints define their axis in the **parent body's local frame**. If the parent body is rotated relative to the world, the world axis differs from the local axis.

### How to find the correct world axis

Run this to compute any joint's world axis:

```python
import mujoco, numpy as np

jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "handle_hinge")
body_id = model.jnt_bodyid[jid]
local_axis = model.jnt_axis[jid]           # e.g. [0, 1, 0]
body_R = data.xmat[body_id].reshape(3, 3)  # body rotation in world
world_axis = body_R @ local_axis            # e.g. [-1, 0, 0]
```

### Example: door frame with 90° rotation

The door frame has `quat="0.707 0 0 0.707"` (90° around X), so:

| Joint | Local axis | World axis | Positive angle meaning |
|-------|-----------|------------|----------------------|
| `handle_hinge` | `(0, 1, 0)` | **`(-1, 0, 0)`** | Press handle down (unlock) |
| `door_hinge` | `(0, 0, -1)` | `(0, 0, -1)` | Unchanged (Z is rotation-invariant for this quat) |

**Common mistake**: Using the local axis directly → EEF rotates around the wrong axis, handle doesn't move.

## Arc max_step vs tolerance

Each arc with `absolute: true` is split into sub-waypoints, each rotating by `max_step` radians. Each sub-waypoint must satisfy the position tolerance to advance.

**Rule**: The EEF displacement per sub-step must exceed `tolerance.position`:

```
displacement_per_step = max_step × radius
```

Where `radius` = distance from pivot to EEF.

| Example | max_step | radius | displacement | tolerance | Result |
|---------|----------|--------|-------------|-----------|--------|
| Handle arc | 0.03 rad | 0.095m | 0.003m | 0.008m | **Instant reached** — too small |
| Handle arc | 0.15 rad | 0.095m | 0.014m | 0.008m | OK |
| Door arc | 0.15 rad | 0.80m | 0.120m | 0.008m | OK |

**If displacement < tolerance**: Every sub-waypoint is already "reached" before the arm moves. The arc completes instantly without actually rotating the mechanism.

## Arc absolute mode and handle spring-back

With `absolute: true`, the arc targets a specific joint angle. Once the handle arc completes (handle at 0.45 rad), control moves to the door arc. The handle spring (`stiffness=2.0, springref=0`) immediately pulls the handle back.

**Problem**: If the handle springs back below `unlock_threshold`, the door latch re-engages during the door push.

**Solutions** (pick one):
1. Lower `unlock_threshold` in the latch callback (e.g. 0.12 instead of 0.20)
2. Increase handle arc target angle so spring-back stays above threshold
3. Use a two-stage approach: hold handle down while pushing door (requires two operators or different operation design)

## IK workspace limits

The P7 arm has finite reach. Door arc targets that exceed the workspace cause IK failures — the arm stops moving while pos_err remains constant.

**Symptoms**: `pos_err` stabilizes at exactly `cartesian_max_linear_step` (e.g. 0.025), IK solver returns None every step.

**Diagnosis**: Patch the IK solver to count failures:
```python
orig = solver.solve
fails = [0]
def patched(target, qpos):
    r = orig(target, qpos)
    if r is None: fails[0] += 1
    return r
solver.solve = patched
```

**Fix**: Reduce the arc target angle to stay within reachable workspace. For P7 + door at ~0.8m from base, door_hinge arc is limited to about -0.35 rad (≈20°).

## eef_world reference for post_move

Post-move waypoints should use `eef_world` reference (EEF position snapshot at action start, world orientation). This avoids:

- `object_world`: Target moves with the object — pushing a door makes the target run away
- `world`: Requires knowing absolute coordinates, not portable across scene variations

`eef_world` offsets are relative to where the EEF was when the action started:
```yaml
# Press down 5cm from current position
- position: [0.0, 0.0, -0.05]
  reference: eef_world

# Then push forward 18cm (relative to SAME starting position)
- position: [0.18, 0.0, 0.0]
  reference: eef_world
```

**Note**: Each `eef_world` waypoint is relative to the SAME snapshot (the EEF position when that specific action was created), not cumulative.

## Complete working open door YAML structure

```
pre_move:
  [0] Approach handle (object_world, tracks handle position)
  [1] Grasp position  (object_world)
eef:
  close: true
post_move:
  [2] Rotate handle   (arc, handle_hinge, world axis [-1,0,0])
  [3] Push door        (arc, door_hinge, world axis [0,0,-1])
  [4] Retreat          (eef_world, z+0.10)
```
