# EEF Mapper (Finger Distance)

`FingerDistanceMapper` provides bidirectional conversion between raw joint
qpos/ctrl values and finger-pad Euclidean distance (metres).  This lets the
observation and action spaces use a gripper-agnostic distance metric instead
of hardware-specific joint values.

## When to Use

Use an `eef_mapper` for **parallel-linkage grippers** (e.g. xf9600, Robotiq
2F-85) where the relationship between the driven joint and the actual finger
opening is nonlinear due to equality constraints connecting passive linkage
joints.

Without a mapper, observations report raw joint positions and actions must be
specified in raw ctrl values.  With a mapper, both are expressed in metres of
finger-pad distance.

## Configuration

Add an `eef_mapper` block under the operator in your YAML config:

```yaml
env:
  operators:
    arm:
      eef_mapper:
        _target_: auto_atom.mappers.finger_distance.FingerDistanceMapper
        left_pad_geom: xfg_left_finger_pad_upper
        right_pad_geom: xfg_right_finger_pad_upper
        actuator_name: xfg_claw_joint
```

### Parameters

| Parameter        | Type  | Default | Description |
|------------------|-------|---------|-------------|
| `left_pad_geom`  | `str` | required | MuJoCo geom name of the left finger pad |
| `right_pad_geom` | `str` | required | MuJoCo geom name of the right finger pad |
| `actuator_name`  | `str` | required | MuJoCo actuator name for the gripper joint |
| `n_samples`      | `int` | `16`    | Number of sweep points for building the lookup table |

The geom and actuator names must match the names defined in the MuJoCo XML.

## How It Works

### Lookup table construction (`bind`)

At environment setup time, the mapper builds a lookup table (LUT) by sweeping
the actuator's ctrl range:

1. For each of `n_samples` evenly-spaced ctrl values from `ctrl_lo` to
   `ctrl_hi`:
   - Reset to keyframe, set the gripper ctrl, run `mj_step` for 1500 steps
     to let equality constraints settle
   - Record the resulting `(qpos, finger_distance)` pair, where
     `finger_distance` is the Euclidean distance between the left and right
     pad geom world positions
2. Store the pairs as sorted arrays for piecewise-linear interpolation

Gravity is temporarily disabled during the sweep and the simulation state is
fully saved/restored, so `bind()` has no side effects.

### Runtime conversion

- **`obs_map(raw)`** -- forward: raw joint qpos to finger distance (metres).
  Used by `capture_observation` to report gripper state.
- **`ctrl_map(user)`** -- inverse: finger distance (metres) to actuator ctrl
  value.  Used by `apply_joint_action` to accept distance-space commands.

Both use `np.interp` for fast piecewise-linear interpolation.

## Interaction with Other Features

### initial_joint_positions

`initial_joint_positions` writes directly to `qpos` and bypasses the mapper.
Values must be in **raw joint space**, not finger distance.  See
[Scene Initialization & Randomization](../task-configuration/randomization.md#initial-joint-positions).

### MCAP replay

When replaying MCAP data, the replay pipeline relies on the
`eef_mapper` to convert recorded finger-distance gripper values to ctrl.
`apply_joint_action` performs the conversion internally, so:

- No separate gripper-rescaling step is needed (and there is no
  `gripper_range` knob on the replay config).
- EEF joints are **excluded** from `initial_joint_positions` injection
  (mcap values are in finger-distance space, not raw qpos) and applied
  through the reset action instead.

See [Data Replay](../tools/mcap_data_replay.md#3-gripper-finger-distance-handling) for details.

## Related

- [Scene Initialization & Randomization](../task-configuration/randomization.md) -- `initial_joint_positions` interaction
- [Data Replay](../tools/mcap_data_replay.md) -- MCAP replay with eef_mapper
- [Gripper Joint Semantics](gripper_joint_semantics.md) -- gripper joint structure
