# Action Space

The env exposes two high-level action methods. Both accept simple arrays
and handle all internal details (IK, mocap, ctrl clipping) automatically.

## `env.apply_joint_action(operator, action)`

Apply joint angles (arm + gripper) for one operator and step the simulation.

```python
# Joint-mode robot (7 arm + 1 gripper = 8 dims)
env.apply_joint_action("arm", [j1, j2, j3, j4, j5, j6, j7, gripper])

# Mocap robot (0 arm + 1 gripper = 1 dim)
env.apply_joint_action("arm", [gripper])
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | `str` | Operator name (e.g. `"arm"`) |
| `action` | array-like, `(n_arm + n_eef,)` | Target joint positions in radians |

The first `n_arm` elements map to `arm_actuators`, the rest to
`eef_actuators`, as declared in the YAML config.

## `env.apply_pose_action(operator, position, orientation, gripper=None)`

Apply an EEF target pose in the operator's base frame and step.

```python
env.apply_pose_action(
    "arm",
    position=[0.1, -0.04, 0.2],       # (3,) base-frame position
    orientation=[-0.707, 0.707, 0, 0], # (4,) xyzw quaternion
    gripper=[0.82],                    # optional, keeps current if None
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `operator` | `str` | Operator name |
| `position` | array-like, `(3,)` | EEF position in base frame |
| `orientation` | array-like, `(4,)` | EEF quaternion (xyzw) in base frame |
| `gripper` | array-like or `None` | Gripper actuator target(s) |

Internally:
- **Joint-mode robot** — solves IK -> joint angles -> `data.ctrl`
- **Mocap robot** — converts to world frame -> writes `data.mocap_pos/quat`

Both robot types use the same call. Gripper is always written to
`data.ctrl[eef_aidx]`.

## Batched Versions

Both methods accept an extra `env_mask` parameter and broadcast 1-D inputs
to all envs. For per-env actions, pass shape `(B, ...)`:

```python
# Single action broadcast to all envs
env.apply_pose_action("arm", [0.1, 0, 0.2], [-0.707, 0.707, 0, 0])

# Per-env actions, shape (B, 3) and (B, 4)
env.apply_pose_action("arm", positions, orientations, grippers, env_mask=mask)
```

## Actuator Layout Per Robot

### Mocap Robot (basis_mocap_eef / Robotiq 2F-85)

`arm_actuators: []`, `eef_actuators: [fingers_actuator]`

- `apply_joint_action`: action = `[gripper]` (1 dim)
- `apply_pose_action`: arm via mocap, gripper via `gripper` param

### Mocap Robot (basis_mocap_eef_xf9600 / XFG-9600)

`arm_actuators: []`, `eef_actuators: [xfg_claw_joint]`

- Robot XML: `assets/xmls/robots/xf9600_mocap.xml` (mocap-driven floating XF9600 gripper)
- Used by `pick_and_place_xf9600` and any other task that pairs mocap-style EEF control with the XF9600 gripper.
- `apply_joint_action` / `apply_pose_action` behave the same as the Robotiq mocap variant; only the gripper actuator name and ctrl range differ.

### Joint-Mode Robot (basis_p7_xf9600 / Panda + XFG-9600)

`arm_actuators: [joint1..joint7]`, `eef_actuators: [xfg_claw_joint]`

- `apply_joint_action`: action = `[j1..j7, gripper]` (8 dims)
- `apply_pose_action`: arm via IK, gripper via `gripper` param

## Recorded Demo Data

`record_demo.py` captures pose + gripper at every step:

| NPZ key (via `low_dim_keys`) | shape | description |
|------|-------|-------------|
| `action/{op}/pose/position` | `(T, 3)` | EEF target position (base frame) |
| `action/{op}/pose/orientation` | `(T, 4)` | EEF target quaternion xyzw (base frame) |
| `action/{op}/base_pose/position` | `(T, 3)` | Optional operator base position in world frame |
| `action/{op}/base_pose/orientation` | `(T, 4)` | Optional operator base quaternion xyzw in world frame |
| `action/eef/joint_state/position` | `(T, n_eef)` | Gripper target |

## Replay Example

```python
from auto_atom import PolicyEvaluator, load_task_file_hydra

demo = np.load("assets/demos/press_three_buttons.npz")
positions = demo_arrays["action/arm/pose/position"]       # (T, 3)
orientations = demo_arrays["action/arm/pose/orientation"]  # (T, 4)
grippers = demo_arrays["action/eef/joint_state/position"]  # (T, 1)

# In your action_applier:
def action_applier(context, action, env_mask=None):
    if "base_position" in action:
        context.backend.env.set_operator_base_pose(
            "arm", action["base_position"], action["base_orientation"], env_mask=env_mask,
        )
    context.backend.env.apply_pose_action(
        "arm", action["position"], action["orientation"], action["gripper"],
    )
```

See `examples/policy_eval_example.py` for a complete runnable example.

## Low-Level API (advanced)

For direct actuator control without per-operator routing:

- `env.step(action)` — writes raw `data.ctrl` vector
- `env.step_operator_toward_target(op, pos_b, quat_b)` — pose without gripper

These are the building blocks used internally by `apply_joint_action` and
`apply_pose_action`. Most users should not need them directly.
