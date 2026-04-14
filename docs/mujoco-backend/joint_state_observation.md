# Joint State Observation

`UnifiedMujocoEnv.capture_observation()` emits per-operator joint-state keys
in two shapes (structured vs. flat), and two semantic sides (**measurement**
vs. **action**). The behavior below is what `_collect_obs` actually produces
for each operator type.

## Design Rules

1. **Measurement side** reports real sensor-equivalent values read from
   `data.qpos` / `data.qvel` / `data.actuator_force`.
2. **Action side** reports only what is *commanded*. For position-controlled
   actuators that means `data.ctrl`; fields that are not commanded (velocity,
   effort) are left **empty**, never back-filled with a measurement.
3. A component (arm or eef) is **completely omitted** when its actuator index
   array is empty. This is what makes mocap robots report only `eef/*` keys.

## Operator Types

### Joint-mode operator (e.g. Franka, 7 arm + 1 gripper)

Both arm and eef have non-empty `arm_actuators` / `eef_actuators`. Per-joint
values are reported for both components.

### Mocap-mode operator (e.g. Robotiq floating base)

`arm_actuators: []`. The arm component has empty `qidx/vidx/aidx`, so **no
`arm/joint_state/*` keys are emitted at all**. The arm is exposed only
through `arm/pose/*` and `action/arm/pose/*`. Only the eef component
contributes joint-state keys.

## Key Layout — Structured Mode (`config.structured = True`)

For each non-empty `(limb, qidx, vidx, aidx)`:

| Key | `data.position` | `data.velocity` | `data.effort` |
|---|---|---|---|
| `{limb}/joint_state`        | `qpos[qidx]` | `qvel[vidx]` | `actuator_force[aidx]` |
| `action/{limb}/joint_state` | `ctrl[aidx]` | `[]`         | `[]` |

- Measurement `effort` is MuJoCo's `data.actuator_force` — the generalized
  force the actuator actually produced this step. For position actuators it
  is the PD servo output, not an F/T sensor reading.
- Action `velocity` and `effort` are **empty lists**: Robotiq/XF9600/Panda
  position actuators only command position, so nothing meaningful can go in
  those slots. Downstream consumers should treat empty as "not commanded".
- All values are `.tolist()`'d; `t` is shared with the rest of the obs.

## Key Layout — Flat Mode (`config.structured = False`)

Each leaf field becomes its own key. Empty components are skipped by size
checks.

| Key | Source |
|---|---|
| `{limb}/joint_state/position` | `qpos[qidx]` (only if `qidx.size > 0`) |
| `{limb}/joint_state/velocity` | `qvel[vidx]` (only if `vidx.size > 0`, and `JOINT_VELOCITY` enabled) |
| `{limb}/joint_state/effort`   | `actuator_force[aidx]` (only if `aidx.size > 0`, and `JOINT_EFFORT` enabled) |
| `action/{limb}/joint_state/position` | `ctrl[aidx]` (only if `aidx.size > 0`) |

There is no `action/.../velocity` or `action/.../effort` key in flat mode —
position is the only commanded quantity, so nothing else is published.

## Example: Mocap Pick-and-Place (Robotiq)

Running `test_mujoco.py` against `pick_and_place.yaml` (mocap operator) prints:

```
arm/pose/orientation
arm/pose/position
arm/pose/rotation
arm/pose/rotation_6d
action/arm/pose/orientation
action/arm/pose/position
eef/joint_state/position         ← qpos of fingers_actuator joint
eef/joint_state/velocity         ← qvel
eef/joint_state/effort           ← actuator_force (PD servo output)
action/eef/joint_state/position  ← ctrl (gripper target)
```

No `arm/joint_state/*` or `action/arm/joint_state/*` keys appear, because
the mocap arm has no actuators and the whole arm component is skipped.

## Example: Joint-mode Panda + XFG-9600

With `basis_p7_xf9600` (7 arm actuators + 1 gripper), a structured obs
entry looks like:

```python
obs["arm/joint_state"] = {
    "data": {
        "position": [q1, q2, q3, q4, q5, q6, q7],    # rad, from qpos
        "velocity": [v1, v2, v3, v4, v5, v6, v7],    # rad/s, from qvel
        "effort":   [f1, f2, f3, f4, f5, f6, f7],    # N·m, actuator_force
    },
    "t": <stamp>,
}
obs["action/arm/joint_state"] = {
    "data": {
        "position": [c1, c2, c3, c4, c5, c6, c7],    # rad, from ctrl
        "velocity": [],                               # not commanded
        "effort":   [],                               # not commanded
    },
    "t": <stamp>,
}
```

The `eef/joint_state` / `action/eef/joint_state` pair has the same shape,
sized by the gripper's actuator count (typically 1).

## Why `effort` Is Not `ctrl`

A previous version filled `effort` from `data.ctrl`, so measurement and
action duplicated the same command vector. That is semantically wrong for
position actuators, where `ctrl` is a **target angle**, not a force. The
current code uses `data.actuator_force[aidx]`, which is the generalized
force MuJoCo integrated into the dynamics — the closest simulation analog
to a joint torque sensor. Actions keep their commanded `ctrl` under
`action/{limb}/joint_state/position`, which is where it belongs.

## Relevant Code

- Observation assembly: [mujoco_env.py `_collect_obs`](../auto_atom/basis/mjc/mujoco_env.py)
- Per-operator index arrays (`_op_arm_qidx`, `_op_eef_aidx`, …):
  [mujoco_basis.py](../auto_atom/basis/mjc/mujoco_basis.py)
- Operator binding config (`arm_actuators`, `eef_actuators`): `OperatorBinding`
  in the project config module
