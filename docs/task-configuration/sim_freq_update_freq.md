# sim_freq 与 update_freq 的关系

## 参数定义

- **`sim_freq`**: 物理仿真频率（Hz），决定 `model.opt.timestep = 1/sim_freq`
- **`update_freq`**: 控制更新频率（Hz），必须 <= `sim_freq` 且能整除
- **`n_substeps`**: `sim_freq / update_freq`，每次控制更新内的物理步数

## 控制流（以 `per_step_ik` 模式为例）

每个控制周期执行：

1. **感知**：读取当前关节角 / 末端位姿
2. **规划**：`move_to_pose()` 计算笛卡尔小步目标（受 `max_linear_step` 限制）
3. **IK 求解**：`step_operator_toward_target()` 从当前关节角解出目标关节角
4. **写 ctrl**：`step()` 将目标关节角写入 `data.ctrl`
5. **物理仿真**：`update()` 执行 `n_substeps` 次 `mj_step()`

## 问题：直接降低 update_freq 导致不稳定

MuJoCo 的 position actuator 在每个 `mj_step` 中计算力矩：

```
τ = kp * (ctrl - q) - kv * qdot
```

当 `n_substeps > 1` 且 ctrl 在所有 substep 中保持不变时：

- 前几步：PD 驱动关节向目标加速
- 中间：高 kp（如 350）+ 低 kv（如 7）= 欠阻尼，关节**超调**
- 后几步：关节在目标附近**振荡**，但 ctrl 不更新
- 下一轮：IK 基于振荡后的偏离状态重新求解，误差累积 → **宏观表现为"乱动"**

对比 `n_substeps=1` 时，每 1ms 就更新一次 ctrl，PD 来不及超调就被修正，外层 IK 闭环掩盖了欠阻尼问题。

## 解决方案：ctrl substep 插值

在 `update()` 中，将 `data.ctrl` 从上一次的值线性插值到新值：

```python
def update(self):
    if self._n_substeps > 1:
        new_ctrl = self.data.ctrl.copy()
        if self._prev_ctrl is None:
            self._prev_ctrl = new_ctrl.copy()
        old_ctrl = self._prev_ctrl
        for i in range(self._n_substeps):
            alpha = (i + 1) / self._n_substeps
            self.data.ctrl[:] = old_ctrl + alpha * (new_ctrl - old_ctrl)
            for cb in self._pre_step_callbacks:
                cb(self.model, self.data)
            mujoco.mj_step(self.model, self.data)
        self._prev_ctrl = new_ctrl
    else:
        # n_substeps=1, 无需插值
        ...
```

每个 substep 看到的 ctrl 增量 = `(new - old) / n_substeps`，与 `n_substeps=1` 时一致。

## 各控制模式的兼容性

| 模式 | ctrl 变化 | 插值效果 |
|------|----------|---------|
| `per_step_ik` | 每次写新关节角到 ctrl | 插值平滑了关节角跳变，消除振荡 |
| `solve_once_interpolate` | 每次写插值后的关节角到 ctrl | 同上 |
| mocap 模式 | 不写 ctrl（直接写 mocap pose） | ctrl 不变，插值等于不插值，无影响 |

## 设计原则

- **`sim_freq` 决定控制稳定性**：物理 timestep 和 PD 参数的匹配
- **`update_freq` 只影响控制更新速率**：降低 update_freq = 更少的"感知→规划→执行"周期，但每个 substep 的 PD 行为与 `update_freq=sim_freq` 时一致
