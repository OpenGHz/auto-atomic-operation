# Data Replay

`replay_demo.py` replays recorded demonstration data through the simulation,
driven by the `DataReplayRunner` class.  It supports two data sources:

- **NPZ** demos recorded by `record_demo.py` (pose or ctrl mode)
- **ROS2 MCAP** files captured from a real robot (joint mode)

## Quick Start

```bash
# NPZ demos (default: outputs/records/demos/<config_name>.npz)
python examples/replay_demo.py --config-name press_three_buttons
python examples/replay_demo.py --config-name pick_and_place +replay.mode=ctrl
python examples/replay_demo.py --config-name pick_and_place +replay.demo_name=my_demo

# ROS2 mcap replay (auto-selects joint mode)
python examples/replay_demo.py --config-name pick_and_place \
    +replay.mcap_path=data/recording.mcap

# Custom topics
python examples/replay_demo.py --config-name pick_and_place \
    +replay.mcap_path=data/recording.mcap \
    +replay.arm_topic=/robot/right_arm/joint_state \
    +replay.gripper_topic=/robot/right_gripper/distance \
    +replay.base_topic=/robot/base_pose \
    +replay.scene_joint_topic=/scene/door/joint_states
```

## Replay Modes

| Mode   | Data Source | Description |
|--------|-----------|-------------|
| `pose` | NPZ       | Replays recorded EEF position + orientation + gripper |
| `ctrl` | NPZ       | Replays recorded joint ctrl values directly |
| `joint`| MCAP      | Replays joint positions from ROS2 messages (auto-selected when `mcap_path` is set) |

## Configuration Reference

All replay settings live under the `replay` key in Hydra overrides
(prefix with `+replay.`).

| Field                 | Type            | Default                                 | Description |
|-----------------------|-----------------|----------------------------------------|-------------|
| `mode`                | `str`           | `"pose"`                               | Replay mode: `pose`, `ctrl`, or `joint` (auto-set for mcap) |
| `demo_name`           | `str \| None`   | `None` (uses config name)              | NPZ file stem under `demo_dir` |
| `demo_dir`            | `str \| None`   | `outputs/records/demos`                | Directory containing `.npz` demo files |
| `mcap_path`           | `str \| None`   | `None`                                 | Path to a ROS2 `.mcap` file; enables joint-mode replay |
| `arm_topic`           | `str`           | `/robot/right_arm/joint_state`         | ROS2 topic for arm joint states |
| `gripper_topic`       | `str`           | `/robot/right_gripper/joint_state`     | ROS2 topic for gripper joint states |
| `base_topic`          | `str \| None`   | `None`                                 | Optional ROS2 `geometry_msgs/PoseStamped` topic for the operator base pose in world frame |
| `scene_joint_topic`   | `str \| None`   | `None`                                 | Optional ROS2 `sensor_msgs/JointState` topic for passive scene joints such as doors / handles |
| `joint_name_mapping`  | `dict`          | `{"gripper": "xfg_claw_joint"}`        | Maps mcap joint names to YAML actuator names |
| `joint_axis_scale`    | `list[float]`   | `[]`                                   | Per-joint replay multipliers applied to the first `N` actuator columns after reordering; useful for mirroring by negating selected axes |
| `joint_clip`          | `dict[str, {min, max}]` | `{}`                            | Per-actuator `[min, max]` clamp applied to the recorded trajectory once at demo-load time (after column reordering). Values are in the actuator's user-facing units (e.g. finger distance for an `eef_mapper`). Either bound can be omitted. |
| `transform_resets`    | `list[TransformResetConfig]` | `[]`                      | Scene-reset rules driven by recorded `geometry_msgs/TransformStamped` MCAP topics; see "Transform Resets" below |
| `done_on_success`     | `bool`          | `false`                                | `true`: report `done=True` as soon as all stages succeed; `false` (default): keep playing back until all replay data is consumed |
| `reset_from_first_frame` | `bool`       | `true`                                 | Apply the first recorded action as the post-reset initial state |
| `steps_per_action`    | `int`           | `1`                                    | Physics steps per recorded action (set >1 for sub-stepping) |
| `kinematic`           | `bool`          | `false`                                | `true`: write qpos directly (exact positions); `false`: drive actuators through physics |

### Video output (script-level only)

| Field       | Default      | Description |
|-------------|-------------|-------------|
| `camera`    | `env1_cam`  | Camera name for frame capture |
| `save_gif`  | `true`      | Save a GIF of the replay |
| `save_mp4`  | `false`     | Save an MP4 of the replay |
| `fps`       | `25`        | Video frame rate |
| `gif_width` | `320`       | GIF output width in pixels |

Videos are saved to `outputs/records/videos/`.

## MCAP Replay Pipeline

When `mcap_path` is set, the replay pipeline performs several pre-processing
steps before the simulation starts:

### 1. Load and align joints

Joint positions are read from the arm and gripper ROS2 topics.  If the arm
and gripper publish at different rates, gripper samples are aligned to arm
timestamps via nearest-neighbour interpolation.

The combined joint array is reordered to match the YAML actuator declaration
order (`arm_actuators` + `eef_actuators`).  Use `joint_name_mapping` when the
mcap joint names differ from the YAML actuator names.

### 2. Optional base pose replay

If `base_topic` is set, replay reads `geometry_msgs/PoseStamped` messages from
that topic. The message is interpreted as the operator base pose in world frame:

- `pose.position.{x,y,z}` -> `base_position`
- `pose.orientation.{x,y,z,w}` -> `base_orientation` in xyzw order

Base samples are nearest-neighbour aligned to the arm joint timestamps. During
replay, each frame applies the aligned base pose with `set_operator_base_pose`
before applying the arm/gripper joint action.

```yaml
replay:
  mcap_path: data/recording.mcap
  base_topic: /robot/base_pose
```

### 3. Optional scene joint replay

If `scene_joint_topic` is set, replay reads `sensor_msgs/JointState` messages
from that topic and aligns them to the arm joint timestamps. Joint names are
taken from the first message and later messages are reordered to match that
name list. During replay, the aligned positions are written directly into the
MuJoCo scene joints by name before and after the robot action for that frame.
If the topic is configured but missing from a particular MCAP, replay logs a
warning and skips scene-joint playback instead of failing.

This is useful for passive articulated scene elements that are not operator
actuators, for example:

- `door_hinge`
- `handle_hinge`

```yaml
replay:
  mcap_path: data/recording.mcap
  scene_joint_topic: /scene/door/joint_states
```

### 4. Optional joint-axis scaling

If `joint_axis_scale` is provided, the replay multiplies the first `N`
actuator columns by the configured factors after column reordering.  This is
useful for replay mirroring or flipping specific revolute axes, for example:

```yaml
replay:
  joint_axis_scale: [1, 1, -1, 1, 1, 1]
```

Any trailing actuator columns not covered by the list keep a factor of `1.0`.

### 5. Gripper finger-distance handling

Real gripper data is in finger-distance space (metres). The replay pipeline
relies on the operator's [`eef_mapper`](../mujoco-backend/eef_mapper.md) to
convert finger distance to raw ctrl — `apply_joint_action` does the
conversion internally, so no separate rescaling step is performed in
replay. Configure an `eef_mapper` on the EEF operator when replaying mcap
data whose gripper values are finger distances.

### 6. Optional joint clipping

If `joint_clip` is set, the replay clamps the recorded trajectory per
actuator at load time (after column reordering). Keys are actuator names
(post-`joint_name_mapping`); each entry is a `{min, max}` pair, either
side optional. This is primarily useful in kinematic replay where the
recorded command would otherwise drive joints into geometry that
real-world contact had stopped — e.g. a gripper closing further than the
grasped object's thickness:

```yaml
replay:
  joint_clip:
    xfg_claw_joint:
      min: 0.015        # finger distance in metres (with eef_mapper)
```

### 7. Transform resets

`transform_resets` lets the replay reposition scene entities to match a
recorded `geometry_msgs/TransformStamped` topic from the MCAP. Each entry
reads the `message_index`-th transform on `topic` and interprets it as
`T_parent->child`. At reset the runner queries the `move`-side entity's
current pose, the other side's world pose, and repositions the movable
side so the simulated relative pose matches the recording. Entries are
applied after `evaluator.reset()` and before the first-frame action
reset.

```yaml
replay:
  transform_resets:
    - topic: /tf_static
      parent: { kind: site, name: door_frame_site }
      child:  { kind: body, name: door }
      move: child
      use_orientation: true
      offset:
        position: [-0.025, 0, 0]    # local frame of the movable entity
        # orientation accepts xyzw (4 floats) or euler rpy (3 floats)
```

Fields:

| Field             | Type                                        | Default      | Description |
|-------------------|---------------------------------------------|--------------|-------------|
| `topic`           | `str`                                        | required     | MCAP topic carrying `geometry_msgs/TransformStamped` |
| `parent`          | `{kind: site\|body\|operator_base, name: str}` | required | Parent side of the recorded transform |
| `child`           | `{kind: site\|body\|operator_base, name: str}` | required | Child side of the recorded transform |
| `move`            | `"parent" \| "child"`                       | `"parent"`   | Which side to reposition; the other side is treated as fixed |
| `message_index`   | `int`                                        | `0`          | Which message on `topic` to read |
| `use_orientation` | `bool`                                       | `false`      | `false`: keep movable entity's current world orientation, only adjust translation. `true`: compose the full transform |
| `offset`          | `PoseOffset`                                 | identity     | Post-hoc calibration: `position` is in the movable entity's local frame; `orientation` right-multiplies the computed rotation (xyzw or euler rpy) |

`operator_base` as a side refers to the operator's base frame and uses
`override_operator_base_pose` under the hood so mocap and joint-mode
operators behave consistently.

### 8. Initial joint position injection

The first frame's replayed joint positions are injected into
`env.initial_joint_positions` so the simulation resets at the recorded
starting configuration. This includes the robot arm/gripper joints and, when
`scene_joint_topic` is enabled, the passive scene joints from that topic.
When `eef_mapper` is configured, EEF joint values are excluded from
`initial_joint_positions` (since they are in user-space, not raw qpos) and
are applied through the mapper via the reset action instead.

See [Scene Initialization & Randomization](../task-configuration/randomization.md) for details on
`initial_joint_positions`.

### 9. Randomization disabled

Task randomization is automatically disabled (`task.randomization = {}`) to
ensure exact trajectory reproduction.

## DataReplayRunner API

`DataReplayRunner` implements the `RunnerBase` interface, making it
interchangeable with `TaskRunner` for external managers:

```python
from auto_atom.runner.data_replay import DataReplayRunner, DataReplayTaskFileConfig

runner = DataReplayRunner().from_config(task_file_config)
runner.reset()
while runner.remaining_steps > 0:
    update = runner.update()
runner.close()
```

### Dynamic demo switching

Use `set_demo_path()` to change the demo source between episodes without
rebuilding the runner:

```python
runner.set_demo_path(mcap_path="data/recording_002.mcap")
runner.reset()  # loads the new demo
```

### Kinematic vs physics replay

- **`kinematic=False`** (default): actions are applied through actuators and
  the physics engine.  Subject to physics lag and constraint settling.
- **`kinematic=True`**: joint positions are written directly to `qpos`,
  bypassing the physics engine.  Guarantees exact trajectory reproduction
  but skips contact dynamics.

## NPZ Demo Format

NPZ files produced by `record_demo.py` contain:

- `low_dim_keys`: ordered list of observation/action key names
- `low_dim_data__0`, `low_dim_data__1`, ...: arrays for each key
- Pose mode reads `action/arm/pose/position`, `action/arm/pose/orientation`,
  and `action/gripper/joint_state/position` (or `action/eef/joint_state/position`)
- Ctrl mode reads `action/arm/joint_state/position` and
  `action/eef/joint_state/position`
- Both pose and ctrl NPZ replay can also carry optional operator-base commands
  via `action/arm/base_pose/position` and `action/arm/base_pose/orientation`.
  When present, replay applies them with `set_operator_base_pose` before the
  arm/eef action for that frame.

## Related

- [Scene Initialization & Randomization](../task-configuration/randomization.md) — `initial_joint_positions` and `initial_pose`
- [EEF Mapper](../mujoco-backend/eef_mapper.md) — finger-distance mapping for parallel-linkage grippers
- [Data Collection](data_collection.md) — recording demos with `record_demo.py`
