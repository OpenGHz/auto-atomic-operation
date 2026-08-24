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

`arm_actuators: []`, `eef_actuators: [eef_claw_joint]`

- Robot XML: `assets/xmls/robots/xf9600_mocap.xml` (mocap-driven floating XF9600 gripper)
- Used by `pick_and_place_xf9600` and any other task that pairs mocap-style EEF control with the XF9600 gripper.
- `apply_joint_action` / `apply_pose_action` behave the same as the Robotiq mocap variant; only the gripper actuator name and ctrl range differ.

### Mocap Robot (basis_mocap_eef_umi_v3 / UMI gripper v3)

`arm_actuators: []`, `eef_actuators: [eef_claw_joint]`

- Robot XML: `assets/xmls/robots/umi_gripper_v3_mocap.xml` (mocap-driven floating UMI gripper v3)
- Used by `pick_and_place_umi_v3` and other tasks that need mocap-style
  EEF control with the v3 UMI gripper.
- The driven slide joint range is `0..0.0165` m (lower bound = max open),
  in contrast to the XF9600's reversed range; otherwise the action API
  is identical to the other mocap variants.

### Joint-Mode Robot (basis_p7_xf9600 / P7 + XFG-9600)

`arm_actuators: [joint1..joint7]`, `eef_actuators: [eef_claw_joint]`

- `apply_joint_action`: action = `[j1..j7, gripper]` (8 dims)
- `apply_pose_action`: arm via IK, gripper via `gripper` param

### Airbot Play (basis_airbot_play_xf9600 / basis_airbot_play_g2p)

`arm_actuators: [joint1..joint6]`, `eef_actuators: [eef_claw_joint]`

- Robot XML: `airbot_play_with_xf9600.xml` (XF9600 gripper) or
  `airbot_play_with_g2p.xml` (G2P gripper). Both gripper variants expose the
  same `eef_*` joint / pad geom names, so the operator block and `eef_mapper`
  config are identical.
- `apply_joint_action`: action = `[j1..j6, gripper]` (7 dims)
- `apply_pose_action`: arm via `AirbotKdlIKSolver`, gripper via `gripper` param

### Joint-Mode Robot (basis_p7_g2p / P7 + G2P)

`arm_actuators: [joint1..joint7]`, `eef_actuators: [eef_claw_joint]`

- Robot XML: `p7_arm_with_g2p.xml`. Same actuator layout as
  `basis_p7_xf9600`; only the gripper assembly (and a few mesh / pad geom
  paths) differ.
- `apply_joint_action`: action = `[j1..j7, gripper]` (8 dims)
- `apply_pose_action`: arm via `P7AnalyticalIKSolver`, gripper via `gripper` param

### Joint-Mode Robot (basis_p7_v3_umi_v3 / P7 v3 + UMI v3)

`arm_actuators: [joint1..joint7]`, `eef_actuators: [eef_claw_joint]`

- Robot XML: `p7_arm_v3_with_umi_gripper_v3.xml`. The arm is the v3
  revision of the P7 (different DH parameters from `p7_arm_with_g2p.xml`,
  driven via `P7V3AnalyticalIKSolver`), paired with the v3 UMI gripper
  (`umi_gripper_v3.xml`, `eef_claw_joint` slide range `0..0.0165`).
- IK backend factory: `auto_atom.backend.mjc.ik.p7_v3_analytical_ik_solver.build_p7_v3_umi_v3_backend`.
- `apply_joint_action`: action = `[j1..j7, gripper]` (8 dims)
- `apply_pose_action`: arm via `P7V3AnalyticalIKSolver`, gripper via `gripper` param
- Used by `cup_on_coaster_gs_airbot_p7_umi`, `open_door_p7_v3_umi_v3`.

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
import numpy as np

from auto_atom import PoseActionEnvProtocol, require_env_capability
from auto_atom.runner.data_replay import ReplayBasePoseActionEnvProtocol

demo = np.load("assets/demos/press_three_buttons.npz")
positions = demo_arrays["action/arm/pose/position"]       # (T, 3)
orientations = demo_arrays["action/arm/pose/orientation"]  # (T, 4)
grippers = demo_arrays["action/eef/joint_state/position"]  # (T, 1)

# In your action_applier:
def action_applier(context, action, env_mask=None):
    env = context.backend.get_env()
    if "base_position" in action:
        base_env = require_env_capability(
            env,
            ReplayBasePoseActionEnvProtocol,
            feature="recorded operator-base actions",
        )
        base_env.set_operator_base_pose(
            "arm", action["base_position"], action["base_orientation"], env_mask=env_mask,
        )
    pose_env = require_env_capability(
        env,
        PoseActionEnvProtocol,
        feature="recorded pose actions",
    )
    pose_env.apply_pose_action(
        "arm", action["position"], action["orientation"], action["gripper"],
        env_mask=env_mask,
    )
```

See `examples/policy_eval_example.py` for a complete runnable example.

## Low-Level API (advanced)

For direct actuator control without per-operator routing:

- `env.step(action)` — writes raw `data.ctrl` vector
- `env.step_operator_toward_target(op, pos_b, quat_b)` — pose without gripper

These are the building blocks used internally by `apply_joint_action` and
`apply_pose_action`. Most users should not need them directly.
