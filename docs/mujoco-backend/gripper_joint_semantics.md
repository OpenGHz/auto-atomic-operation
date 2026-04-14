# Gripper Joint Semantics

Different gripper models use different joint types, which determines the physical meaning and unit of `qpos` readings.

## Summary

| Gripper | Joint Name(s) | Joint Type | Unit | Range | 0 = | Max = |
|---------|---------------|------------|------|-------|-----|-------|
| **Robotiq 2F-85** | `left_driver_joint` / `right_driver_joint` | `hinge` (revolute) | **rad** | 0 ~ 0.82 (actuator) / 0 ~ 0.9 (joint limit) | Fully open (~85mm) | Fully closed (~0mm) |
| **XF9600** | `clawj` | `slide` (prismatic) | **m** | 0 ~ 0.02 | Fully open | Fully closed (20mm travel) |
| **Airbot Play** | `endleft` / `endright` | `slide` (prismatic) | **m** | 0 ~ 0.04 | Fully closed (fingers together) | Fully open (40mm apart) |

## Details

### Robotiq 2F-85 (`robotiq.xml`)

```xml
<!-- Joint default (robotiq_assets.xml) -->
<default class="driver">
  <joint range="0 0.9" armature="0.005" damping="0.1" .../>
</default>

<!-- Actuator -->
<position name="fingers_actuator" joint="left_driver_joint"
          ctrlrange="0 0.82" kp="50" kv="5"/>
```

- **Revolute** joint — `qpos` is a rotation angle in **radians**
- Left and right driver joints are coupled via `<equality><joint>` (mirror each other)
- To get actual fingertip distance (m), you need forward kinematics from the driver angle, or read the fingertip body positions directly

### XF9600 (`xf9600_gripper.xml`)

```xml
<default class="claw_joint">
  <joint type="slide" axis="0 0 -1" range=".0 0.02" .../>
  <position kp="5000" kv="140" ctrlrange=".0 0.02" .../>
</default>
```

- **Prismatic** (slide) joint — `qpos` is a linear displacement in **meters**
- Single actuator `clawj` drives both fingers through a linkage mechanism (equality constraints connect l11↔l51 and l21↔l61)
- Range: 0m (open) to 0.02m (closed, 20mm travel)

### Airbot Play (`airbot_play.xml`)

```xml
<default class="finger_left">
  <joint axis="0 1 0" range="0 0.04" type="slide" .../>
</default>
<default class="finger_right">
  <joint axis="0 -1 0" range="0 0.04" type="slide" .../>
</default>
```

- **Prismatic** (slide) joint — `qpos` is a linear displacement in **meters**
- Two independent slide joints (`endleft`, `endright`) coupled via `<equality><joint>` (symmetric motion)
- Left finger slides along +Y, right finger slides along -Y (they move in opposite directions)
- Range: 0m (fingers together, closed) to 0.04m (fingers apart, 40mm each side)
- Actuator controls `endleft` only; `endright` mirrors via equality constraint

## Control Command vs Joint State

All three grippers use **position-based actuators**, so the control command (`ctrl`) and joint state (`qpos`) share the same unit and meaning: `ctrl[gripper] = target_qpos`.

| Gripper | Actuator Type | Actuator Definition | ctrl Unit |
|---------|--------------|---------------------|-----------|
| **Robotiq 2F-85** | `position` | `<position joint="left_driver_joint" ctrlrange="0 0.82" kp="50" kv="5"/>` | **rad** |
| **XF9600** | `position` | `<position joint="clawj" ctrlrange="0 0.02" kp="5000" kv="140"/>` | **m** |
| **Airbot Play** | `general` (PD) | `<general joint="endleft" ctrlrange="0 0.04" gainprm="350 0 0" biasprm="0 -350 -10"/>` | **m** |

- Robotiq and XF9600 use explicit `<position>` actuators (built-in PD control)
- Airbot Play uses a `<general>` actuator with `gainprm`/`biasprm` that form an equivalent PD position servo: `force = gain * (ctrl - qpos) + bias * qvel`
- In all cases: **the value you send as ctrl is the target joint position, and qpos is the current joint position, both in the same unit**

## Reading Gripper State in Code

In `mujoco_env.py`, `qpos[eef_qidx]` returns the raw joint value, and `ctrl[eef_aidx]` returns the commanded target:
- For Robotiq: both in rad (need kinematics conversion to get fingertip distance)
- For XF9600 / Airbot Play: both in m (directly interpretable as displacement)
