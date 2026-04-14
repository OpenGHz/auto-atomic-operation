# Pose Observation

When `DataType.POSE` is enabled, `UnifiedMujocoEnv._collect_obs` publishes
an operator's end-effector pose in two semantic sides: **measurement** (the
current EEF pose read from the simulator) and **action** (the commanded
target pose). The exact shapes depend on `config.structured`.

This is the companion to [Joint State Observation](joint_state_observation.md).

## When Pose Is Emitted

For each operator, pose keys are only produced if the operator was
registered with a valid EEF site (`site_id >= 0`). Operators without a
registered site are **silently skipped** — there is no error. Register an
operator via `register_operator(..., eef_site="<site_name>", ...)` if you
want pose readings for it.

Once per operator, the pose sensor (`framepos`/`framequat` in the XML) is
cross-checked against a direct `get_site_pose` readout to catch XML
misconfiguration. Failures raise at startup, not during training.

## Frame Convention

Pose observations are in the **operator base frame** (set by
`register_operator`, typically the robot root body).

| Quantity | Source |
|---|---|
| Current pose (measurement) | `get_operator_eef_pose_in_base` — site pose transformed to base frame |
| Target pose (action) | `target_pos_in_base` / `target_quat_in_base` — the latest commanded target |

If the env is used standalone (no `register_operator` call, e.g. in
`test_mujoco.py`), both current and target fall back to **world-frame**
site pose. Target equals current in this case since no command has been
issued.

Quaternions are **xyzw** order. The raw MuJoCo framequat sensor is wxyz;
the env converts it via `_quat_wxyz_to_xyzw` before publishing.

## Key Layout — Structured Mode

For operator name `op`:

| Key | Shape of `data` |
|---|---|
| `{op}/pose` | `{ header, pose: { position: {x,y,z}, orientation: {x,y,z,w} } }` — ROS-like nested dict |
| `action/{op}/pose` | Same shape as above, filled from `target_pos_in_base` / `target_quat_in_base` |
| `{op}/pose/rotation` | Euler angles (3,) derived from the current rotation matrix |
| `{op}/pose/rotation_6d` | First 6 entries of the flattened 3x3 rotation matrix (the "6D" rotation representation used by some policies) |

`rotation` and `rotation_6d` are published **in both structured and flat
modes** as separate keys (they don't get nested into the `{op}/pose` ROS
message), and they only describe the **current** orientation — there is no
`action/{op}/pose/rotation` counterpart.

## Key Layout — Flat Mode

| Key | Content |
|---|---|
| `{op}/pose/position` | Current EEF position `(3,)` in base frame |
| `{op}/pose/orientation` | Current EEF quaternion `(4,)` xyzw in base frame |
| `{op}/pose/rotation` | Current orientation as Euler angles `(3,)` |
| `{op}/pose/rotation_6d` | Current orientation as 6D representation `(6,)` |
| `action/{op}/pose/position` | Target EEF position `(3,)` in base frame |
| `action/{op}/pose/orientation` | Target EEF quaternion `(4,)` xyzw in base frame |

Note: flat mode does **not** publish `action/{op}/pose/rotation` or
`action/{op}/pose/rotation_6d`. If a consumer needs the target orientation
in those representations, it must derive them from
`action/{op}/pose/orientation` on its own.

## Example: Mocap Pick-and-Place

Running `test_mujoco.py` against `pick_and_place.yaml` with the flat schema
produces these pose-related keys for the `arm` operator:

```
arm/pose/position          shape=(B, 3)   float32   — current, base frame
arm/pose/orientation       shape=(B, 4)   float32   — current, base frame, xyzw
arm/pose/rotation          shape=(B, 3)   float64   — current, Euler
arm/pose/rotation_6d       shape=(B, 6)   float64   — current, 6D
action/arm/pose/position   shape=(B, 3)   float32   — target, base frame
action/arm/pose/orientation shape=(B, 4)  float32   — target, base frame, xyzw
```

For a mocap-mode robot, `action/arm/pose/*` is the authoritative command
channel — the mocap body is driven by `data.mocap_pos` / `data.mocap_quat`
derived from `target_pos_in_base` after `register_operator` maps it back to
world frame.

For a joint-mode robot, `action/arm/pose/*` still reports the intended EEF
target that the IK solver is tracking toward; the actually applied command
ends up in `action/arm/joint_state/position` (per-joint ctrl).

## Joint-mode vs Mocap Summary

The pose observation shape is the **same** for both operator types — pose
is frame-level, not joint-level, so it doesn't care whether the arm is
driven by per-joint actuators or by a mocap body. The difference is in
where the target comes from:

| Operator type | How `target_pos_in_base` is updated |
|---|---|
| **Joint-mode** | IK solver writes the target each step (`step_operator_toward_target`) |
| **Mocap** | Target is written directly from `apply_pose_action` / planning layer, then propagated to `data.mocap_*` in world frame |

In both cases, the published `action/{op}/pose/*` reflects the latest
committed target in base frame as of the observation step.

## Relevant Code

- Observation assembly: [mujoco_env.py `_collect_obs`](../auto_atom/basis/mjc/mujoco_env.py)
- Current EEF in base frame: `get_operator_eef_pose_in_base` in
  [mujoco_env.py](../auto_atom/basis/mjc/mujoco_env.py)
- Target pose storage: `OperatorState.target_pos_in_base` /
  `target_quat_in_base` in [mujoco_env.py](../auto_atom/basis/mjc/mujoco_env.py)
- Quaternion convention conversion: `_quat_wxyz_to_xyzw`
- Sensor validation: `_validate_pose_sensor_matches_site`
