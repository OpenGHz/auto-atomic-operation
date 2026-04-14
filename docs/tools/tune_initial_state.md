# Tune Initial State

[`examples/tune_initial_state.py`](../examples/tune_initial_state.py) loads a task config via Hydra, opens the MuJoCo viewer, and provides a tkinter editing panel for interactively adjusting operator initial state (base pose, EEF pose, gripper). **No task execution loop is run** — this tool is purely for tuning and visualising the initial configuration. Once satisfied, click **Print YAML** to output a ready-to-paste config snippet.

## Basic usage

```bash
python examples/tune_initial_state.py
python examples/tune_initial_state.py --config-name cup_on_coaster
python examples/tune_initial_state.py --config-name stack_color_blocks
```

The script accepts the same Hydra overrides as `aao_demo`.

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

## Default sources when YAML omits `initial_state`

If a task YAML does not define `task_operators[].initial_state`, the backend does not apply any extra override during setup. The operator starts from the state already present in the loaded MuJoCo model and the operator registration snapshot.

This is easiest to understand by separating `base_pose`, `arm`, and `eef`.

### 1. `base_pose`

`base_pose` means the operator base pose in world frame, but its default source depends on the operator type.

#### Joint-mode operator

For operators with `arm_actuators`, the default `base_pose` comes from the physical root body pose in the MuJoCo model. In practice, this usually means it is determined by the XML body hierarchy and its initial transform.

Example: Franka in [`aao_configs/pick_and_place_franka.yaml`](../aao_configs/pick_and_place_franka.yaml)

- The config inherits [`aao_configs/basis_franka.yaml`](../aao_configs/basis_franka.yaml), where `arm_actuators` is non-empty.
- This makes the operator a joint-mode arm.
- If `initial_state.base_pose` is omitted, the base pose defaults to the robot root body pose from the loaded XML, such as [`assets/xmls/scenes/pick_and_place/demo_franka.xml`](../assets/xmls/scenes/pick_and_place/demo_franka.xml) and the included robot XML.

Practical implication:

- If the robot base is already placed correctly in XML, you can omit `base_pose`.
- If you want to shift the whole robot base without editing XML, set `initial_state.base_pose` in YAML.

#### Pure mocap / pure EEF operator

For operators without `arm_actuators`, the system uses a virtual base frame instead of a physical arm base. Its default is not parsed from an XML robot base pose. It is initialized to world origin with identity rotation.

Example: [`aao_configs/arrange_flowers.yaml`](../aao_configs/arrange_flowers.yaml)

- The config inherits [`aao_configs/basis_mocap_eef.yaml`](../aao_configs/basis_mocap_eef.yaml), where `arm_actuators: []`.
- This makes the operator a pure EEF mocap operator.
- If `initial_state.base_pose` is omitted, the virtual base defaults to:

```yaml
base_pose:
  position: [0.0, 0.0, 0.0]
  orientation: [0.0, 0.0, 0.0, 1.0]
```

Practical implication:

- For mocap tasks, `base_pose` is mainly a reference frame used by conversions and visualization.
- Changing it does not mean "reading a robot base from XML"; it changes the virtual base definition used by the controller.

### 2. `arm`

`initial_state.arm` represents the operator home end-effector pose in world frame.

If YAML omits it, the default comes from the current EEF site pose after the MuJoCo model has been loaded and the operator has been registered. That pose is therefore determined by the model's initial state, which may include:

- XML body transforms
- joint `qpos`
- mocap body initial pose
- keyframe or reset-applied initial values

This means the default is not a separately stored YAML value. It is the EEF pose currently produced by the loaded simulation state.

#### Joint-mode example

Example: [`aao_configs/pick_and_place_franka.yaml`](../aao_configs/pick_and_place_franka.yaml)

- The Franka EEF site is `gripper`, configured in [`aao_configs/basis_franka.yaml`](../aao_configs/basis_franka.yaml).
- The scene XML defines a `home` keyframe in [`assets/xmls/scenes/pick_and_place/demo_franka.xml`](../assets/xmls/scenes/pick_and_place/demo_franka.xml).
- If `initial_state.arm` is omitted, the default home EEF pose is whatever pose the `gripper` site has under that initial model state.

Practical implication:

- When the arm already starts in a good pose from XML or keyframe, you can omit `initial_state.arm`.
- When you need a different starting EEF pose but do not want to change the XML home state, set `initial_state.arm` in YAML.

#### Pure mocap example

Example: [`aao_configs/cup_on_coaster.yaml`](../aao_configs/cup_on_coaster.yaml)

- The operator is mocap-based, so there is no IK-driven arm joint state.
- If `initial_state.arm` is omitted, the default home EEF pose is the current `eef_pose` site pose from the loaded scene.

Practical implication:

- For mocap tasks, `initial_state.arm` is usually the most direct field to tune, because it directly changes the home EEF pose used by the operator.

### 3. `eef`

`initial_state.eef` is the gripper control value, for example open or closed amount.

If YAML omits it, the default comes from the current actuator control state `data.ctrl` at setup time.

Example:

- In many Robotiq tasks using [`aao_configs/basis_mocap_eef.yaml`](../aao_configs/basis_mocap_eef.yaml), leaving out `initial_state.eef` means the gripper starts from the control value already present in the model reset state.
- In [`aao_configs/press_three_buttons.yaml`](../aao_configs/press_three_buttons.yaml), `initial_state.eef: 0.82` is set explicitly because the task wants a closed gripper for button pressing.

Practical implication:

- If the task depends on a known open/closed state, it is safer to set `initial_state.eef` explicitly.
- If you omit it, the gripper may still behave correctly, but its initial opening is whatever the loaded model/control state already contains.

## Summary by category

### A. Values that usually come from the loaded XML or reset state

- Joint-mode `base_pose`
- Default `arm` home EEF pose
- Default `eef` control value

These are all derived from the initial MuJoCo simulation state after model loading and registration.

### B. Values that are code-defined defaults rather than XML-parsed robot poses

- Pure mocap operator `base_pose`

This defaults to world origin plus identity rotation unless overridden in YAML.

### C. Values that YAML can override directly

- `initial_state.base_pose`
- `initial_state.arm`
- `initial_state.eef`

If any of these are specified, the backend applies them on top of the default registered state.

## Rule of thumb

- If you are using a fixed-base IK arm, think of the default initial state as "coming from the model's initial robot state".
- If you are using a pure mocap EEF operator, think of `arm` and `eef` as coming from the current scene state, but think of `base_pose` as a virtual frame whose default is code-defined.
- If you need repeatable startup behavior across tasks, prefer writing the tuned values back into YAML instead of relying on implicit model defaults.
