# Stack Color Blocks IK Control

本文档说明 [`third_party/press_da_button/gs_playground/experimental/env/table30/02_stack_color_blocks_data.py`](../third_party/press_da_button/gs_playground/experimental/env/table30/02_stack_color_blocks_data.py) 如何完成 IK 控制，以及末端目标如何一路转换成 Franka 机械臂的关节控制。

## 总览

这条控制链分为四层：

```text
StackColorBlocksCollector._step_logic()
  └─ 生成抓取/搬运阶段的末端目标位置、yaw 和夹爪命令
      └─ 组装 7 维 action = [dx, dy, dz, droll, dpitch, dyaw, gripper]
          └─ env.step(action)
              └─ TaskEnv / BaseRobot 按 action_mode="eef_relative" 解释为相对末端控制
                  └─ BaseRobot.apply_action() 调用 IK 求解器
                      └─ FrankaRobotiq 的 DLS IK 输出关节目标并写入 actuator_ctrls
```

需要注意的一点是，这个数据采集脚本本身不直接实现 IK 求解器。它只负责产生末端目标和夹爪命令。真正的 IK 发生在机器人封装层。

## 1. 高层脚本负责生成末端轨迹

采集脚本的核心逻辑在 `StackColorBlocksCollector._step_logic()`：

- 文件: [`02_stack_color_blocks_data.py`](../third_party/press_da_button/gs_playground/experimental/env/table30/02_stack_color_blocks_data.py)
- 关键位置: `p_app_lift_z`、`p_grasp`、`p_stack` 等关键点计算，以及最终 `action` 组装

它先根据当前 episode 中锁存的起点、目标方块位置和底块位置，构造一组 Manhattan 风格的关键点：

- 抬升到安全高度
- 沿 x 对齐
- 沿 y 对齐
- 单独调整 yaw
- 下压抓取
- 提起并搬运
- 对齐放置位置
- 松开夹爪
- 撤退并回 home

对应代码集中在：

- `p_app_lift_z` 到 `p_home` 的关键点定义
- `set_target()` / `set_target_yaw()` 根据状态机阶段选择当前目标
- `self.exec_pos = smooth_step_pos(...)` 对位置做限幅平滑

最后脚本构造出一个 `(B, 7)` 动作：

```python
action[:, :3] = (self.exec_pos - ref_pos) * 0.5
action[:, 3:6] = rotvec_cmd
action[:, 6] = grip_cmd
self.env.step(action)
```

这 7 维量的语义是：

- 前 3 维: 末端位置增量
- 中间 3 维: 末端姿态增量，这里以旋转向量形式表达
- 最后 1 维: 夹爪开合命令

因此，这里输出的是“末端控制命令”，不是关节角命令。

## 2. 环境配置决定使用 `eef_relative`

在任务环境配置里：

- 文件: [`_02_stack_color_blocks_franka.py`](../third_party/press_da_button/gs_playground/src/manipulation/tasks/table30/_02_stack_color_blocks_franka.py)

定义了：

```python
action_mode: str = "eef_relative"
```

这表示传给环境的 action 会被解释为“相对于参考末端位姿的增量控制”。

环境执行动作时：

- [`TaskEnv.apply_action()`](../third_party/press_da_button/gs_playground/src/manipulation/tasks/task_env.py) 把 action 交给 `self.robot.apply_action(...)`
- [`TaskEnv._before_chunk_step()`](../third_party/press_da_button/gs_playground/src/manipulation/tasks/task_env.py) 在每次 `step()` 开始时调用 `self.robot.update_reference(data)`

而底层执行顺序在 [`mtx_env.py`](../third_party/press_da_button/gs_playground/src/env/motrix_env/mtx_env.py) 中是：

1. `env.step(actions)` 进入
2. `_before_chunk_step()` 更新参考状态
3. `apply_action()` 写入控制
4. `physics_step()` 进行物理仿真
5. `update_state()` 更新观测

这意味着 `eef_relative` 里的参考位姿 `ref_ee_pose`，会在每次 step 开头同步到当下状态或上一次命令状态，从而让相对控制更稳定，不容易因跟踪误差跳变。

## 3. `BaseRobot.apply_action()` 中完成 IK

真正的 IK 核心在：

- 文件: [`base_robot.py`](../third_party/press_da_button/gs_playground/src/manipulation/robots/base_robot.py)
- 方法: `BaseRobot.apply_action()`

当 `action_mode` 是 `eef_relative` 时，代码流程如下。

### 3.1 将 action 解释为相对末端位姿

`BaseRobot.apply_action()` 会先解析动作模式：

```python
parts = action_mode.split('_')
main_mode = parts[0]
sub_mode = parts[1] if len(parts) > 1 else 'absolute'
```

对于 `eef_relative`：

- `main_mode == "eef"`
- `sub_mode == "relative"`

接着取出：

- `act_pose = act[:, :6]`
- `act_gripper = act[:, 6]`

然后基于参考末端位姿 `self.ref_ee_pose` 计算目标末端 6D 位姿：

```python
target_pose_6d = self.ref_ee_pose + act_pose
```

这里的 `target_pose_6d` 格式是：

- `XYZ + RPY`

也就是：

- 位置使用笛卡尔坐标
- 姿态使用 roll/pitch/yaw 欧拉角

### 3.2 将目标姿态从 RPY 转成四元数

后续 IK 使用的是 7D 位姿：

- `XYZ + quat(xyzw)`

所以代码会先把 `RPY` 转成四元数：

```python
t_pos = target_pose_6d[:, :3]
t_rpy = target_pose_6d[:, 3:]
t_euler_zyx = np.flip(t_rpy, axis=-1)
r = Rotation.from_euler('zyx', t_euler_zyx, degrees=False)
t_quat_xyzw = r.as_quat()
desired_pose = np.concatenate([t_pos, t_quat_xyzw], axis=-1)
```

这里之所以 `flip` 一下，是因为内部用 `scipy` 时显式按 `zyx` 构造，而原始末端姿态在该框架里存储为 `[roll, pitch, yaw]`。

### 3.3 调用 IK 求解器

得到 `desired_pose` 后，直接调用：

```python
res = self.solver.solve(self.chain, data, desired_pose)
```

返回结果里包含：

- `res[..., 1]`: IK 残差 `residual`
- `res[..., 2:]`: 关节解 `desired_j`

随后代码只接受残差足够小的解：

```python
good = np.isfinite(residual) & (residual < 2e-2)
if np.any(good):
    ctrl[good, :self.num_dof_arm] = desired_j[good, :self.num_dof_arm]
    self.last_cmd_ee_pose[good] = target_pose_6d[good]
```

也就是说：

- IK 没求好时，不会盲目覆盖机械臂关节控制
- 只有成功求解的环境实例，才会更新 `last_cmd_ee_pose`

这对于 batched 多环境采集尤其重要，因为不同环境的 IK 可达性可能不一致。

### 3.4 夹爪控制独立写入

夹爪命令不经过 IK，直接写到 gripper actuator：

```python
ctrl[:, self.gripper_act_id] = act_gripper
data.actuator_ctrls = ctrl
```

因此整条链路里：

- 机械臂前 7 个自由度由 IK 结果决定
- 夹爪由采集脚本直接给开合值

## 4. Franka 机器人定义了 IK 链和求解器

Franka 的 IK 具体配置在：

- 文件: [`franka_robotiq.py`](../third_party/press_da_button/gs_playground/src/manipulation/robots/franka_emika_panda_robotiq/franka_robotiq.py)

初始化时创建：

```python
self.chain = ik.IkChain(
    self.mx_model,
    start_link="link1",
    end_link="robotiq_base",
    end_effector_offset=[0.0, 0.0, 0.1489, 0.0, 0.0, 0.0, 1.0],
)
self.solver = ik.DlsSolver(
    max_iter=50,
    step_size=0.5,
    tolerance=1e-3,
    damping=1e-3,
)
```

可以据此读出几个关键信息：

- IK 起点链路从 `link1` 开始
- IK 末端链路到 `robotiq_base`
- 通过 `end_effector_offset` 把真实抓取点偏移纳入末端定义
- 求解器是 DLS, 即阻尼最小二乘法

所以这份采集脚本依赖的是 `motrixsim.ik` 这一套 IK，而不是 `examples/ik/mink_franka.py` 中的 `MinkIK`。

## 5. 姿态控制在这个脚本里主要是 yaw 对齐

虽然 `BaseRobot.apply_action()` 支持完整 6D 末端控制，但这个任务脚本实际只显式控制了 yaw。

在 [`02_stack_color_blocks_data.py`](../third_party/press_da_button/gs_playground/experimental/env/table30/02_stack_color_blocks_data.py) 中：

- 读取方块初始四元数，提取 `top_yaw` / `base_yaw`
- 在 `ST_APP_ALIGN_YAW` 和 `ST_TRP_ALIGN_YAW` 两个状态中，单独设置目标 yaw
- 用 `wrap_to_pi()` 和 `pi/2` 对称性约束，考虑立方体每 90 度外观等价
- 通过 `r_err.as_rotvec()` 生成 `rotvec_cmd`

因此这段脚本的姿态控制策略是：

- 平移阶段主要走笛卡尔位置
- 需要抓取或放置前，再做单独 yaw 对齐
- roll/pitch 基本保持起始姿态附近

这也是为什么最终给 `action[:, 3:6]` 的通常只是一个以 z 轴旋转为主的旋转向量。

## 6. Reset 和参考状态同步

为了让 `eef_relative` 模式工作稳定，脚本和环境在 reset 时会主动同步状态。

在 `start_episodes()` 中：

```python
self.env.robot.reset_envs(data, done_mask)
self.env.robot.update_reference(data)
```

这两步的作用是：

- 把 `last_cmd_qpos` / `last_cmd_ee_pose` 同步到当前真实状态
- 把 `ref_qpos` / `ref_ee_pose` 设置为新的控制参考

否则如果环境刚 reset 完还沿用旧 episode 的参考末端位姿，相对控制会瞬间跳变。

## 7. 与 `mink_franka.py` 的关系

[`examples/ik/mink_franka.py`](../third_party/press_da_button/examples/ik/mink_franka.py) 展示的是另一套 IK 控制路径：

- 使用 MuJoCo
- 使用 `MinkIK`
- 通过 mocap target 交互式拖动末端目标

而 `02_stack_color_blocks_data.py`：

- 运行在当前任务环境的 MotrixSim 封装上
- 使用 `FrankaRobotiq -> BaseRobot -> motrixsim.ik.DlsSolver`
- 通过 batched 环境 action 做自动数据采集

两者都在做末端到关节的逆运动学，但不是同一套运行时链路。

## 8. 一句话总结

`02_stack_color_blocks_data.py` 的 IK 控制本质上是：

1. 状态机生成末端目标位置、目标 yaw 和夹爪命令
2. 脚本把这些目标转换成 `eef_relative` 动作
3. `BaseRobot.apply_action()` 将相对末端位姿转成绝对 7D 目标位姿
4. `FrankaRobotiq` 中配置好的 DLS IK 求解器把末端位姿求成关节角
5. 关节角与夹爪命令一起写入 `actuator_ctrls`
6. 仿真执行后再进入下一次状态机迭代

如果要继续深入，建议下一步直接看三处：

- [`02_stack_color_blocks_data.py`](../third_party/press_da_button/gs_playground/experimental/env/table30/02_stack_color_blocks_data.py)
- [`base_robot.py`](../third_party/press_da_button/gs_playground/src/manipulation/robots/base_robot.py)
- [`franka_robotiq.py`](../third_party/press_da_button/gs_playground/src/manipulation/robots/franka_emika_panda_robotiq/franka_robotiq.py)
