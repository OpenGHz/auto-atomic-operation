# Mink 迭代 IK 原理

本文档说明当前 Franka 任务使用的迭代逆运动学求解器原理。实现位于 [`auto_atom/backend/mjc/ik/mink_ik_solver.py`](../auto_atom/backend/mjc/ik/mink_ik_solver.py)。

## 入口

Franka 任务的 backend 工厂是：

- [`build_franka_backend()`](../auto_atom/backend/mjc/ik/mink_ik_solver.py)

它会：

1. 读取环境里的 MuJoCo model
2. 创建 `MinkIKSolver`
3. 把 solver 交给 MuJoCo backend

之后每次 pose 控制都会走到：

- `MujocoOperatorHandler.move_to_pose()`
- `UnifiedMujocoEnv.step_operator_toward_target()`
- `MinkIKSolver.solve()`

## 一句话原理

当前 IK 不是解析解，而是微分迭代解：

- 从当前关节角出发
- 根据当前末端误差求一小步关节速度
- 把这一步积分到关节角上
- 重复很多次
- 最后得到一组近似满足目标位姿的关节角

## 数据流图

```text
目标末端位姿 (base frame)
  │
  ▼
MinkIKSolver.solve(target_pose_in_base, current_qpos)
  │
  ├─ 1. base -> world 坐标变换
  │     pos_b, quat_b
  │       -> pos_w, quat_w
  │
  ├─ 2. 构造目标 SE3
  │     target_se3 = SE3(R_w, pos_w)
  │
  ├─ 3. 设置末端任务
  │     eef_task.set_target(target_se3)
  │
  ├─ 4. 用 current_qpos 初始化 Configuration
  │     configuration.update(q_seed)
  │
  ├─ 5. 设置 posture task
  │     posture_task.set_target_from_configuration(configuration)
  │
  ├─ 6. 迭代 n_iterations 次
  │     vel = mink.solve_ik(configuration, tasks, dt, ...)
  │     configuration.integrate_inplace(vel, dt)
  │
  ├─ 7. 取出 solved qpos
  │     solved = configuration.q[arm_qidx]
  │
  ├─ 8. max_joint_delta 限幅
  │     solved <- clamp(solved - current_qpos)
  │
  ▼
输出目标关节角 solved
```

## 求解步骤

### 1. 目标位姿转换到 world frame

上层传给 solver 的目标位姿是机械臂 base 坐标系下的末端 pose：

- `target_pose_in_base.position`
- `target_pose_in_base.orientation`

但 `mink` 在 world frame 工作，所以代码先用机械臂 base 的世界位姿做一次变换：

- `pos_w = R_base @ pos_b + base_pos`
- `quat_w = quat_base ⊗ quat_b`

这样得到 world frame 下的末端目标位姿。

### 2. 构造末端任务 `FrameTask`

solver 内部有一个 `FrameTask`，对应末端 site：

- 对 Franka 来说是 `gripper`

目标是：

- 让这个 frame 的位置和姿态接近 `target_se3`

这就是 IK 的主任务。

### 3. 用当前关节角作为 seed

每次 `solve()` 时，都会把当前机械臂关节角写进 `mink.Configuration`：

- `configuration.update(q_seed)`

这里的 seed 就是当前 `qpos`。

这意味着当前 IK 是一个“从当前状态出发的局部迭代过程”，不是全局搜索。

### 4. 增加 `PostureTask`

除了末端任务，还有一个 `PostureTask`：

- 它的目标不是让末端靠近目标
- 而是让关节尽量保持接近当前 seed

作用是：

- 防止跳到另一组差别很大的等价关节构型
- 提高连续控制时的平滑性

所以每轮优化不是只看末端误差，还会同时考虑“不要让关节偏离当前构型太多”。

### 5. 迭代求解

每轮都会调用：

```python
vel = mink.solve_ik(
    self._configuration,
    tasks,
    self._dt,
    solver="quadprog",
    limits=self._limits,
)
self._configuration.integrate_inplace(vel, self._dt)
```

这里可以这样理解：

- `mink.solve_ik(...)` 求的是当前最合适的一小步关节速度 `vel`
- `integrate_inplace(...)` 把这一步速度积分成新的关节角

然后下一轮再基于新的关节角继续算。

所以它不是一步直接得到最终关节角，而是：

- 算一点
- 走一点
- 再算一点

### 6. 输出最终关节角

迭代完成后，取出机械臂那几维关节角：

- `solved = configuration.q[arm_qidx]`

这就是 solver 的输出。

### 7. `max_joint_delta` 限幅

最后还会做一次限幅：

- 如果 `solved - current_qpos` 太大
- 就按比例整体缩小这次关节变化

作用是：

- 防止单次 IK 输出跳太远
- 降低奇异附近的大跳变风险

这里要注意，它的限幅方式是：

- 先找出变化量最大的那个关节
- 如果这个最大值超过 `max_joint_delta`
- 就把**整个关节增量向量**按同一个比例缩小

所以可能出现这种现象：

- 大部分关节本来只需要动一点
- 但某一个关节因为多解切换、肘部翻转或奇异附近构型变化，跳得特别大
- 这个单独的大关节触发了 `max_joint_delta`
- 结果所有关节的增量都被一起压小
- 最终末端这一轮只前进了很小一点

因此，某些“明明有解但移动很慢”的现象，本质上可能不是所有关节都应该只动这么小，而是：

- 少数关节跳得过大
- 触发了全局缩放
- 把整组关节更新一起拖慢了

## 它为什么叫“微分 IK”

因为它不是直接求解：

- `f(q) = target_pose`

而是在每一轮只求一个局部小问题：

- 当前如果关节变化一点点，末端会怎么变化
- 那么现在该让关节朝哪个方向动，才能让末端误差变小

这背后依赖的是 Jacobian 线性近似：

```text
dx ≈ J(q) dq
```

其中：

- `q` 是当前关节角
- `dq` 是一小步关节变化
- `dx` 是末端位姿变化

所以当前 solver 本质上是：

- 基于 Jacobian 的局部线性化
- 基于速度的迭代优化

## 为什么它不保证一次就“解对”

因为它是近似迭代法，不是解析解。

结果会受这些因素影响：

- 当前 seed 是什么
- `n_iterations` 是否足够
- `dt` 是否合适
- `position_cost` / `orientation_cost` 权重
- `posture_cost` 多强
- 关节限位约束

因此可能出现：

- solver 返回了关节角
- 但末端并没有足够接近目标位姿

从任务角度看，这就可以理解成：

- 这次迭代 IK 没有真正解成功

## 和解析解的区别

解析解：

- 直接根据几何关系算出满足目标位姿的关节角
- 不依赖迭代收敛

当前 `mink` 迭代解：

- 从当前关节角开始
- 反复修正
- 最终得到一个近似解

所以当前方法的特点是：

- 更通用
- 更容易适配复杂模型和约束
- 但会受初值和参数影响

## 当前实现里的关键参数

配置一般在：

- [`aao_configs/base_franka.yaml`](../aao_configs/base_franka.yaml)
- [`aao_configs/pick_and_place_franka.yaml`](../aao_configs/pick_and_place_franka.yaml)

重要参数有：

- `n_iterations`
  每次 solve 迭代多少轮

- `dt`
  每轮积分步长

- `position_cost`
  末端位置误差权重

- `orientation_cost`
  末端姿态误差权重

- `posture_cost`
  保持当前关节构型的权重

- `max_joint_delta`
  单次 solve 输出允许离当前关节角多远

## 结合当前控制链怎么理解

在当前框架里，IK 不是单独工作的，而是嵌在下面这个闭环里：

```text
笛卡尔子目标
  -> mink 迭代 IK
  -> 输出目标关节角
  -> 关节插值 / PD 执行
  -> 得到新的机器人状态
  -> 下一轮控制
```

所以判断 IK 是否“真的成功”，不能只看：

- 有没有返回关节角

还要看：

- 执行这组关节角后，末端误差是否明显下降

如果 solver 每次都返回关节角，但误差下降很慢，从任务角度看，仍然属于“没真正解好”。
