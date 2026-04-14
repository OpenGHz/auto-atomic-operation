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
    +replay.gripper_topic=/robot/right_gripper/distance
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
| `joint_name_mapping`  | `dict`          | `{"gripper": "xfg_claw_joint"}`        | Maps mcap joint names to YAML actuator names |
| `gripper_range`       | `list[float]`   | `[0.0, 0.09]`                         | Real gripper distance `[closed, open]` in metres |
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

### 2. Gripper rescaling

Real gripper data is typically in finger-distance space (metres), while
MuJoCo actuators use raw ctrl ranges.  The pipeline rescales gripper values
from `gripper_range` to the actuator's `ctrlrange`.

**Exception:** When the operator has an [`eef_mapper`](eef_mapper.md)
configured, rescaling is skipped because `apply_joint_action` already
converts finger-distance values to ctrl internally.

### 3. Initial joint position injection

The first frame's joint positions are injected into `env.initial_joint_positions`
so the robot resets at the recorded starting configuration.  When `eef_mapper`
is configured, EEF joint values are excluded from `initial_joint_positions`
(since they are in user-space, not raw qpos) and are applied through the
mapper via the reset action instead.

See [Scene Initialization & Randomization](randomization.md) for details on
`initial_joint_positions`.

### 4. Randomization disabled

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

## Related

- [Scene Initialization & Randomization](randomization.md) — `initial_joint_positions` and `initial_pose`
- [EEF Mapper](eef_mapper.md) — finger-distance mapping for parallel-linkage grippers
- [Data Collection](data_collection.md) — recording demos with `record_demo.py`
