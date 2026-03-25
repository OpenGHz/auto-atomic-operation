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

An object is considered "grasped" when ALL of the following are true:

### 1.1 Bilateral Contact
- **Left finger contact**: At least one contact pair where the object body touches a geom with name starting with `left_`
- **Right finger contact**: At least one contact pair where the object body touches a geom with name starting with `right_`

**Detection method**: Iterates through `env.data.contact` array, checks `geom_bodyid` to find contacts involving the target body, then checks geom names.

### 1.2 Horizontal Alignment
- **Horizontal error**: Euclidean distance between object center (x, y) and EEF position (x, y)
- **Threshold**: `horizontal_error <= grasp_horizontal_threshold` (default 0.03 meters)

**Rationale**: Ensures the gripper is centered on the object, not just touching it from the side.

**Important**: This check assumes the grasp point is near the object center (e.g., top-down grasps). For side grasps (e.g., cup handles), the horizontal distance can be large even with a valid grasp. In such cases, increase `grasp_horizontal_threshold` to accommodate the offset.

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `grasp_horizontal_threshold` | `MujocoOperatorHandler` | `0.03` m | Max horizontal distance between object center and EEF for valid grasp |

**Note**: Increase this value for side grasps (e.g., 0.06-0.10 m for cup handles).

### Debugging

When `eef.close=True` and a target object exists, the `TaskUpdate.details` will include:

```python
"grasp_check": {
    "left_contact": bool,
    "right_contact": bool,
    "horizontal_ok": bool,
    "horizontal_error": float,  # meters
    "horizontal_threshold": 0.03
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

**Implementation**: `MujocoObjectHandler.is_displaced()`

An object is considered "displaced" when:

**Position change**: `||current_pos - initial_pos|| > threshold`

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `displacement_threshold` | `MujocoObjectHandler.__init__` | `0.01` m | Minimum distance moved to count as displaced |

**Usage**: Pass when creating the object handler (not currently exposed in YAML configs).

---

## 4. Placed Condition

**Implementation**: `MujocoObjectHandler.is_at_target()`

An object is "placed" at the target when:

**Position error**: `||current_pos - target_pos|| <= threshold`

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `position_tolerance` | `MujocoObjectHandler.__init__` | `0.02` m | Maximum distance from target to count as placed |

**Usage**: Pass when creating the object handler (not currently exposed in YAML configs).

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

**Threshold**: `orientation_tolerance` (default 0.1 radians ≈ 5.7°)

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `position_tolerance` | `MujocoOperatorHandler` | `0.01` m | Position error threshold |
| `orientation_tolerance` | `MujocoOperatorHandler` | `0.1` rad | Orientation error threshold |

**Usage**: Set in backend config (not currently exposed in YAML).

---

## 6. End-Effector Control

**Implementation**: `MujocoOperatorHandler.control_eef()`

The gripper action is "reached" based on the operation:

### 6.1 Closing (with target object)
**Condition**: `_is_target_grasped()` returns True (see section 1)

**Additional requirement**: Minimum settle steps must elapse before checking grasp.

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `eef_grasp_settle_steps` | `MujocoOperatorHandler` | `10` | Simulation steps to wait before checking grasp |

### 6.2 Closing (without target / fallback)
**Condition**: `actual_qpos >= max(target_ctrl - eef_tolerance, 0.45)`

Gripper has closed to within tolerance of the target position, or reached the minimum closed position (0.45).

### 6.3 Opening
**Condition**: `actual_qpos <= max(eef_tolerance, 0.05)`

Gripper has opened to within tolerance of fully open, or reached the minimum open threshold (0.05).

### Configurable Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `eef_tolerance` | `MujocoOperatorHandler` | `0.03` | Gripper position tolerance |
| `eef_grasp_settle_steps` | `MujocoOperatorHandler` | `10` | Steps to wait before grasp check |

---

## 7. Timeout

All control actions have a maximum step limit:

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `command_timeout_steps` | `MujocoOperatorHandler` | `600` | Max simulation steps per action |

At 500 Hz simulation frequency with 20 Hz control, this equals 30 seconds real-time.

---

## Summary Table: All Configurable Parameters

Parameters are now structured under `control` in the operator configuration:

```yaml
operators:
  - name: arm
    control:
      tolerance:
        position: 0.01      # meters
        orientation: 0.08   # radians
        eef: 0.03          # gripper tolerance
      grasp:
        horizontal_threshold: 0.03  # meters (increase for side grasps)
        settle_steps: 5             # simulation steps
      timeout_steps: 600            # max steps per action
```

| Parameter Path | Default | Unit | Description |
|----------------|---------|------|-------------|
| `control.tolerance.position` | 0.01 | m | Position error threshold for pose control |
| `control.tolerance.orientation` | 0.08 | rad | Orientation error threshold for pose control |
| `control.tolerance.eef` | 0.03 | - | Gripper position tolerance |
| `control.grasp.horizontal_threshold` | 0.03 | m | Max horizontal distance for valid grasp |
| `control.grasp.settle_steps` | 5 | steps | Min steps before checking grasp |
| `control.timeout_steps` | 600 | steps | Max steps per action before timeout |

**Note**: Object-level parameters (`displacement_threshold`, `position_tolerance`) are not yet exposed in YAML.

---

## Future Improvements

1. Expose backend parameters in YAML task configs
2. Make grasp `horizontal_threshold` configurable
3. Add per-stage timeout overrides
4. Support custom post-condition predicates
