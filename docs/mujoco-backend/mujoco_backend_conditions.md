# MuJoCo Backend Condition Constraints

This document explains how the MuJoCo backend (`auto_atom.backend.mjc.mujoco_backend`)
implements the operation conditions defined by the shared [Execution Completion
Flow](../task-configuration/execution_completion_flow.md), and lists the
configurable parameters for threshold tuning.

## Overview

The shared stage state machine asks the MuJoCo backend to evaluate conditions
at the operation-specific checkpoints documented in the execution flow.  The
condition vocabulary is:

- **grasped**: Object is held by the gripper
- **released**: Operator is not holding an object
- **contacted**: Gripper has made contact with the object
- **displaced**: Object has moved from its initial position
- **reached**: End-effector is within the target pose tolerance
- **placed**: Object is at the target location

---

## 1. Grasped Condition

**Implementation**: `MujocoOperatorHandler._is_target_grasped()`

An object is considered "grasped" when the gripper is sufficiently closed and both of the following hold:

1. Physical bilateral finger contact is detected.
2. The optional lateral threshold check passes.

This keeps the predicate contact-aware while still allowing an additional
gripper-centered geometric sanity check in scenes that need it.

### 1.1 Bilateral Contact
- **Left finger contact**: At least one contact pair where the object body touches a geom with name starting with `left_`
- **Right finger contact**: At least one contact pair where the object body touches a geom with name starting with `right_`

**Detection method**: Iterates through `env.data.contact` array, checks `geom_bodyid` to find contacts involving the target body, then checks geom names.

### 1.2 Optional Lateral Check
- **Lateral error**: Distance between object and EEF center on the plane perpendicular to grasp direction
- **Computed in EEF frame**: Object position is transformed to gripper coordinate system
- **Lateral threshold**: Disabled by default with `lateral_threshold: 0.0`; when set above zero, the check is `lateral_error <= lateral_threshold`

**Rationale**: Ensures the object is still laterally inside the grasp volume when
bilateral contact alone is too permissive for a scene.

**Key improvement**: Unlike world-frame horizontal checks, this works correctly for any grasp orientation (vertical, horizontal, or angled).

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `lateral_threshold` | `MujocoGraspConfig` | `0.0` m | Max lateral distance perpendicular to grasp direction; `0.0` disables the lateral check |
| `grasp_axis` | `MujocoGraspConfig` | `2` (Z) | Grasp direction axis in EEF frame: 0=X, 1=Y, 2=Z |

### Debugging

When `eef.close=True` and a target object exists, the `TaskUpdate.details` will include:

```python
"grasp_check": {
    "left_contact": bool,
    "right_contact": bool,
    "lateral_ok": bool,
    "lateral_error": float,  # meters, in EEF frame
    "lateral_threshold": float  # meters
}
```

---

## 2. Contacted Condition

**Implementation**: `MujocoTaskBackend.is_operator_contacting()`

Contact is detected when there exists at least one contact pair between:
- Any geom belonging to the operator's body subtree
- Any geom belonging to the target object's body

**Detection method**:
1. Get operator root body ID and all descendant body IDs
2. Iterate through `env.data.contact` array
3. Check if one geom belongs to operator subtree and the other to target object

### Configurable Parameters

None. Pure contact detection based on MuJoCo's contact solver.

---

## 3. Displaced Condition

**Implementation**: `SceneBackend.is_object_displaced()`

An object is considered "displaced" when:

**Position change**: `||current_pos - initial_pos|| > threshold`

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `displacement_threshold` | `StageControlConfig` (`task.stages[].param.displacement_threshold`) | `0.01` m | Minimum distance moved to count as displaced |
| `displacement_threshold` | `MujocoObjectHandler.__init__` | `0.01` m | Constructor default used when no per-stage value is set |

**Usage**: Set on the stage param to override the backend default for `displaced` checks (typical for `push` / `pull`):

```yaml
stages:
  - name: open_door
    operation: push
    operator: arm
    param:
      # Without this the default 1 cm threshold can be satisfied by the
      # lever rotation alone, even if the door body itself never opens.
      displacement_threshold: 0.10
      pre_move: ...
      post_move: ...
```

The runtime forwards the configured value into `is_object_displaced(...)` in
[`runtime.py`](../../auto_atom/runtime.py); when the field is omitted the
backend default still applies.

---

## 4. Placed Condition

**Constraint**: `OperationConstraint.PLACED`

**Implementation**: Compound check in `_check_stage_condition()` + `MujocoObjectHandler.is_at_target()`

The `place` operation succeeds when **both** conditions are met:

1. **Released**: The operator is no longer grasping any object.
2. **At target**: The **held object** (auto-detected as the object grasped by the operator at stage start) is within tolerance of the **target position**.

### Target Position Resolution

The target position depends on the `placed_reference` config and whether a stage object is set:

| `placed_reference` | `stage.object` set? | Target position |
|---------------------|---------------------|-----------------|
| `"object"` (default) | Yes | `stage.object`'s current pose (the destination reference object) |
| `"object"` | No | Last pre_move waypoint resolved position |
| `"pre_move"` | Yes or No | Last pre_move waypoint resolved position |

**Note**: `stage.object` in a place stage is the **destination reference** (e.g., coaster, box), not the object being placed. The held object is auto-detected via `is_object_grasped()`.

### Tolerance

Position tolerance supports:
- **Scalar** (float): L2-norm threshold (e.g., `0.02` = 2cm sphere)
- **Per-axis** (`[x, y, z]`): Each element is a per-axis threshold. Any element can be `null` to skip that axis (e.g., `[0.03, 0.03, null]` = 3cm XY, ignore Z).

Orientation tolerance:
- **Scalar** (float): Quaternion angular distance in radians.
- **`null`** (default): Orientation is not checked.

### Tolerance Resolution Chain

| Priority | Source | Location |
|----------|--------|----------|
| 1 (highest) | `placed_tolerance` | `StageControlConfig` (per-stage in YAML) |
| 2 | `tolerance.placed` | `MujocoToleranceConfig` (operator-level control config) |
| 3 (fallback) | — | `null` (no constraint — check degrades to released-only) |

A value is considered "configured" only if it is a scalar or a list with at
least one non-null element. An all-null list is treated as unset, so the next
level of the chain is consulted. When nothing is configured at any level, the
dimension is not checked — the PLACED condition then only requires release.

### YAML Example

```yaml
- name: place_cup_on_coaster
  object: coaster            # destination reference object
  operation: place
  param:
    pre_move:
      - position: [0.0, 0.0, 0.15]
        reference: object_world
      - position: [0.0, 0.0, 0.035]
        reference: object_world
    post_move:
      - position: [0.0, 0.0, 0.2]
        reference: object_world
    eef:
      close: false
    placed_tolerance:
      position: [0.03, 0.03, null]  # 3cm XY, no Z check
      orientation: null              # no orientation check
    placed_reference: object         # default
```

### Failure Diagnostics

When the PLACED condition fails, the following details are included:
- `held_object`: Name of the object that was being placed
- `placed_reference`: The reference mode used (`"object"` or `"pre_move"`)
- `target_position`, `current_position`: World positions
- `position_error`: L2 distance between current and target
- `target_orientation`, `current_orientation`: World orientations
- `orientation_error`: Angular distance in radians

---

## 5. Pose Control (Move Actions)

**Implementation**: `MujocoOperatorHandler.move_to_pose()`

A pose target is "reached" when BOTH position and orientation errors are below thresholds:

### 5.1 Position Error
**Metric**: Euclidean distance `||current_pos - target_pos||`

**Threshold**: `position_tolerance` (default 0.01 m)

### 5.2 Orientation Error
**Metric**: Geodesic distance on SO(3), computed as:
```python
quat_diff = target_quat * inverse(current_quat)
angle = 2 * arccos(|quat_diff.w|)
```

**Threshold**: `orientation_tolerance` (default 0.08 radians ≈ 4.6°)

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `position_tolerance` | `MujocoControlConfig.tolerance.position` | `0.01` m | Position error threshold |
| `orientation_tolerance` | `MujocoControlConfig.tolerance.orientation` | `0.08` rad | Orientation error threshold |

**Usage**: Set under `task_operators.<name>.control.tolerance` in YAML.

---

## 6. End-Effector Control

**Implementation**: `MujocoOperatorHandler.control_eef()`

The gripper action is "reached" based on the operation:

### 6.1 Closing with a required target grasp

`pick` and `pull` intrinsically compile their closing EEF with
`require_grasp: true`; task YAML should not repeat it.  An explicit closing EEF
on another operation can opt into the same behavior by setting
`task.stages[].param.eef.require_grasp: true`. Completion then requires
`_is_target_grasped()` to return true for the Stage target (see section 1).
Neither reaching the commanded gripper position nor being blocked by an
arbitrary object is accepted as completion.

**Additional requirement**: Minimum settle steps must elapse before checking grasp.

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `control.grasp.settle_steps` | `MujocoGraspConfig` | `5` | Control updates to wait before checking grasp |

### 6.2 Closing without a required grasp

For a raw closing EEF outside `pick` and `pull`, the default is
`require_grasp: false`. A detected target grasp still completes the primitive.
Otherwise the positional fallback is:

**Condition**: `actual_qpos >= target_ctrl - eef_tolerance`

If an object physically blocks the gripper, AAO waits at least 30 updates and
then accepts measurable motion away from fully open. This fallback is disabled
when `require_grasp` is true.

### 6.3 Opening
**Condition**: `actual_qpos <= eef_open_value + eef_tolerance` after the
configured release-settle updates.

Gripper has opened to within tolerance of fully open, or reached the minimum open threshold (0.05).

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `control.tolerance.eef` | `MujocoToleranceConfig` | `0.03` | Gripper position tolerance |
| `control.grasp.settle_steps` | `MujocoGraspConfig` | `5` | Control updates to wait before grasp check |
| `control.grasp.release_settle_steps` | `MujocoGraspConfig` | `0` | Control updates to wait after opening before completion |

`require_grasp` belongs to the individual `eef` primitive rather than the
operator defaults. `pick` and `pull` supply it automatically; use the explicit
form for an EEF in another operation:

```yaml
operation: push
param:
  pre_move: [...]
  eef:
    close: true
    require_grasp: true
```

---

## 7. Timeout

All control actions have a maximum step limit:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `control.timeout_steps` | `MujocoControlConfig` | `100` | Max simulation steps per action |

At 600 Hz simulation frequency with 30 Hz control, this equals about 3.3 seconds of simulated time.

---

## Summary Table: All Configurable Parameters

Parameters are now structured under `control` in the operator configuration:

```yaml
task_operators:
  arm:
    control:
      tolerance:
        position: 0.01      # meters
        orientation: 0.08   # radians
        eef: 0.03          # gripper tolerance
      grasp:
        lateral_threshold: 0.0      # meters (0 = disabled, >0 to enable check)
        grasp_axis: 2               # 0=X, 1=Y, 2=Z (grasp direction)
        settle_steps: 5             # control updates before grasp check
        release_settle_steps: 0     # control updates after opening
      timeout_steps: 100            # max steps per action
```

| Parameter Path | Default | Unit | Description |
|----------------|---------|------|-------------|
| `control.tolerance.position` | 0.01 | m | Position error threshold for pose control |
| `control.tolerance.orientation` | 0.08 | rad | Orientation error threshold for pose control |
| `control.tolerance.eef` | 0.03 | - | Gripper position tolerance |
| `control.grasp.lateral_threshold` | 0.0 | m | Max lateral distance for valid grasp (0=disabled) |
| `control.grasp.grasp_axis` | 2 | - | Grasp direction axis (0=X, 1=Y, 2=Z) |
| `control.grasp.settle_steps` | 5 | updates | Min control updates before checking grasp |
| `control.grasp.release_settle_steps` | 0 | updates | Min control updates after opening before completion |
| `control.timeout_steps` | 100 | steps | Max steps per action before timeout |
| `control.ik_unreachable_threshold` | 30 | streak | Consecutive IK failures inside `move_to_pose` after which the stage fails fast with `failure_category: ik_unreachable` instead of waiting for `timeout_steps` |

**Note**: `displacement_threshold` is now exposed per-stage as
`task.stages[].param.displacement_threshold`. Object-level
`position_tolerance` remains a constructor argument only.

---

## Future Improvements

1. Add per-stage timeout overrides
2. Support custom post-condition predicates
