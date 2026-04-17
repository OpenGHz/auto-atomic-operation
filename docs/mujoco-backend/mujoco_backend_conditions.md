# MuJoCo Backend Condition Constraints

This document explains how the MuJoCo backend (`auto_atom.backend.mjc.mujoco_backend`) implements the condition constraints described in the main README, and lists the configurable parameters for threshold tuning.

## Overview

The MuJoCo backend evaluates post-conditions after each operation to determine success. The main conditions are:

- **grasped**: Object is held by the gripper
- **contacted**: Gripper has made contact with the object
- **displaced**: Object has moved from its initial position
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
| `displacement_threshold` | `MujocoObjectHandler.__init__` | `0.01` m | Minimum distance moved to count as displaced |

**Usage**: Pass when creating the object handler (not currently exposed in YAML configs).

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

### 6.1 Closing (with target object)
**Condition**: `_is_target_grasped()` returns True (see section 1)

**Additional requirement**: Minimum settle steps must elapse before checking grasp.

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `control.grasp.settle_steps` | `MujocoGraspConfig` | `5` | Simulation steps to wait before checking grasp |

### 6.2 Closing (without target / fallback)
**Condition**: `actual_qpos >= max(target_ctrl - eef_tolerance, 0.45)`

Gripper has closed to within tolerance of the target position, or reached the minimum closed position (0.45).

### 6.3 Opening
**Condition**: `actual_qpos <= max(eef_tolerance, 0.05)`

Gripper has opened to within tolerance of fully open, or reached the minimum open threshold (0.05).

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `control.tolerance.eef` | `MujocoToleranceConfig` | `0.03` | Gripper position tolerance |
| `control.grasp.settle_steps` | `MujocoGraspConfig` | `5` | Steps to wait before grasp check |

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
        settle_steps: 5             # simulation steps
      timeout_steps: 100            # max steps per action
```

| Parameter Path | Default | Unit | Description |
|----------------|---------|------|-------------|
| `control.tolerance.position` | 0.01 | m | Position error threshold for pose control |
| `control.tolerance.orientation` | 0.08 | rad | Orientation error threshold for pose control |
| `control.tolerance.eef` | 0.03 | - | Gripper position tolerance |
| `control.grasp.lateral_threshold` | 0.0 | m | Max lateral distance for valid grasp (0=disabled) |
| `control.grasp.grasp_axis` | 2 | - | Grasp direction axis (0=X, 1=Y, 2=Z) |
| `control.grasp.settle_steps` | 5 | steps | Min steps before checking grasp |
| `control.timeout_steps` | 100 | steps | Max steps per action before timeout |

**Note**: Object-level parameters (`displacement_threshold`, `position_tolerance`) are not yet exposed in YAML.

---

## Future Improvements

1. Expose backend parameters in YAML task configs
2. Add per-stage timeout overrides
3. Support custom post-condition predicates
