# Tune Initial State

[`examples/tune_initial_state.py`](../examples/tune_initial_state.py) loads a task config via Hydra, opens the MuJoCo viewer, and provides a tkinter editing panel for interactively adjusting operator initial state (base pose, EEF pose, gripper). **No task execution loop is run** — this tool is purely for tuning and visualising the initial configuration. Once satisfied, click **Print YAML** to output a ready-to-paste config snippet.

## Basic usage

```bash
python examples/tune_initial_state.py
python examples/tune_initial_state.py --config-name cup_on_coaster
python examples/tune_initial_state.py --config-name stack_color_blocks
```

The script accepts the same Hydra overrides as `run_demo.py`.

## UI overview

The script opens two windows simultaneously:

1. **MuJoCo Viewer** — live 3D view of the scene and robot. For pure-EEF (mocap) operators such as Robotiq, an RGB coordinate frame is drawn in the viewer to mark the virtual base position.
2. **tkinter Editing Panel** — contains the following editable fields:

| Field | Description |
| ----- | ----------- |
| **Base Pose** | World pose of the virtual arm base. Imagine the EEF is mounted on a robot arm — this sets where the arm's base would be in world coordinates. Changing it only updates the visual marker; the EEF body does **not** move. |
| **EEF Pose** | World pose of the end-effector. Clicking **Apply** teleports the EEF to the specified pose. |
| **EEF ctrl** | Gripper control value (0.0 = open, 0.82 = closed). Clicking **Apply** drives the gripper joints to the target value. |

### Input format

- Each value field is a **single text entry** accepting comma or space-separated floats, e.g. `0.5, 0.0, 0.3` or `0.5 0.0 0.3`.
- Position expects **3 values** (x, y, z).
- Orientation supports two modes, toggled via radio buttons:
  - **Quaternion** — 4 values in xyzw order.
  - **Euler** — 3 values in yaw, pitch, roll order (radians).

## YAML output

Click **Print YAML** to print the current state as an `initial_state` config block to the terminal:

```yaml
initial_state:
  base_pose:
    position: [0.000000, 0.000000, 0.400000]
    orientation: [0.000000, 0.000000, 0.000000, 1.000000]
  arm:
    position: [0.005223, 0.005721, 0.251100]
    orientation: [-0.707107, 0.707107, 0.000000, 0.000000]
  eef: 0.0
```

Copy this snippet into the corresponding operator's `initial_state` field in the task YAML to apply the tuned values.
