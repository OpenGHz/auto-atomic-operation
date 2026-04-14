# Open Door Task Tuning Notes

## Overview

Record of issues and solutions encountered while implementing the P7+XF9600 door opening task (`open_door_p7_ik`).

## 1. EEF Close 永远卡住 (eef_moving timeout)

**现象**: `eef_target=0.82`, `actual_qpos=0.005`, 夹爪完全不响应。

**原因**: `MujocoOperatorHandler._eef_target()` 硬编码返回 `0.82`（Robotiq 2F-85 参数），但 XF9600 夹爪的 ctrlrange 是 `[0, 0.02]`。ctrl 被裁剪到 0.02，而 reached 判定门槛 `max(target - tolerance, 0.45)` 中的 0.45 也是 Robotiq 的值，XF9600 永远不可能达到。

**修复**:
- `MujocoOperatorHandler` 新增 `eef_open_value` / `eef_close_value` 字段
- `_eef_target()` 和 reached 判定改为使用这些值
- `build_p7_xf9600_backend()` 从 model ctrlrange 读取并传入

## 2. 夹爪闭合但 qpos 远低于目标 (物理阻挡)

**现象**: `ctrl=0.02` 已生效，但 `qpos` 稳定在 ~0.005，远低于自由空间稳态值 0.015。

**原因**: XF9600 夹指闭合时被把手杆或门板物理阻挡，qpos 被卡在某个中间值。tolerance-based reached 判定永远不通过。

**修复**: 新增 settle-based reached 条件——当夹爪命令发出 >= 30 步且 qpos 已离开 open 位置，即认为 eef 完成：

```python
elif (eef.close
      and self._eef_steps[env_index] >= max(settle_steps, 30)
      and actual > self.eef_open_value + self.control.tolerance.eef * 0.1):
    reached = True
```

## 3. 夹爪闭合碰到门板导致无法闭合

**现象**: TCP 到达 handle_grasp_site 位置后，闭合夹爪时指尖撞到门板。

**原因**: handle_grasp_site 在 X=1.435，距门板 (X≈1.49) 仅 5.5cm。XF9600 夹指从 TCP 向前延伸约 5cm（TCP local Y 方向），闭合时指尖超出 TCP 位置触碰门面。

**修复**: 将抓取位置从门板方向回退 ~2.5cm，从 X=1.435 改为 X=1.41~1.42。需要在 viewer 中微调确保指垫仍能包住把手杆。

## 4. Stall 检测导致推门极慢 (pos_err 恒定)

**现象**: 推门 waypoint 需要 300+ 步才能 reached，pos_err 长时间停在 ~0.025。

**原因**: `MujocoOperatorHandler.move_to_pose()` 内置 stall 检测——每 8 步位置没改善就把 `_move_step_scale` 砍半（最低 0.1）。推门时接触力导致位置误差反复波动，频繁触发衰减。恢复速率仅 `*1.1`，远慢于衰减 `*0.5`。

`max_linear_step=0.015 * scale=0.1 = 0.0015m/step`，爬行极慢。

**修复**: 在 `MujocoControlConfig` 新增 `adaptive_step_scaling: bool` 开关，默认 False（改了默认值）。接触密集型任务关闭此功能：

```yaml
task_operators:
  - name: arm
    control:
      adaptive_step_scaling: false
```

## 5. Arc waypoint 瞬间 reached (步长小于 tolerance)

**现象**: handle_hinge arc (max_step=0.03rad) 的子 waypoint 全部在 1 步内 reached，把手实际没有转动。

**原因**: 把手杆半径 ~0.095m，arc 每步 EEF 位移 = 0.03 × 0.095 ≈ 0.003m，小于 position tolerance 0.008m。每个子 waypoint 在创建时就已经 "reached"。

**修复**: 增大 `max_step` 使每步位移 > tolerance，或减小 tolerance。例如 `max_step: 0.15` 时位移 ≈ 0.014m > 0.008m。

## 6. Reference 选择原则

| 阶段 | 推荐 reference | 原因 |
|------|---------------|------|
| pre_move (接近物体) | `object_world` | 跟踪物体位移，门位姿变化后仍能对准 |
| post_move (操作物体) | `eef_world` | 以动作开始时 EEF 位置为基准，不随物体运动而变 |
| post_move (回撤) | `world` 或 `eef_world` | 固定目标，不受物体影响 |

**注意**: `object_world` 在 post_move 中会实时跟踪物体——推门时目标随门一起动，永远追不上。必须用 `eef_world`（action 开始时锁定一次）。

## 7. pre_step_callbacks (门锁机制)

通过 YAML `_target_` 配置 per-step 物理回调，无需改 Python 代码：

```yaml
env:
  pre_step_callbacks:
    - _target_: auto_atom.callbacks.door_latch.DoorLatchCallback
      door_joint: door_hinge
      handle_joint: handle_hinge
      kp: 80.0
      kd: 8.0
      unlock_threshold: 0.20
      lock_zone: 0.05
```

回调在每次 `mj_step()` 前执行，通过 `qfrc_applied` 施加条件弹簧力模拟门锁。
