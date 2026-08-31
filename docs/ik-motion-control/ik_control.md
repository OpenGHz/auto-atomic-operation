# IK Control for Robot Arms

本文档说明 auto_atom 框架中基于逆运动学 (IK) 的机械臂关节空间控制的实现方案和参数配置。

## 概述

框架支持两种 operator 控制模式：

| 模式 | 触发条件 | 控制方式 |
|------|----------|----------|
| **Mocap** | `arm_actuators: []` | 浮动基座 + weld 约束，运动学驱动 |
| **Joint** | `arm_actuators: [a1, a2, ...]` | IK 求解 + PD 关节位控，动力学驱动 |

当 YAML 中 `arm_actuators` 非空时，自动进入 Joint 模式。此时需要提供 `IKSolver` 实例。

## 控制链路

```
TaskRunner.update()
  └─ operator.move_to_pose(target_eef_pose_world)
      ├─ 若配置了 Cartesian step 限幅:
      │    ├─ 当前位置 → 目标位姿做笛卡尔分段
      │    ├─ 位置按直线小步逼近
      │    └─ 姿态按 SLERP 小步逼近
      └─ env.world_to_base(target) → target_eef_pose_base
          └─ env.step_operator_toward_target(target_eef_pose_base)
              ├─ 首次到达该目标时 ik_solver.solve(target_pose_base, current_qpos)
              │    ├─ base→world 坐标变换
              │    ├─ mink 微分IK迭代求解
              │    └─ max_joint_delta clamp（防奇异点跳变）
              ├─ 若 `joint_control_mode=solve_once_interpolate`
              │    └─ 在多个 control step 中对 solved_joint_targets 做线性插值
              ├─ 若 `joint_control_mode=per_step_ik`
              │    └─ 每个 control step 都重新 solve 一次
              ├─ ctrl[arm_aidx] = current_step_joint_targets
              └─ env.step(ctrl)  # PD控制器驱动关节
```

### 关键设计

框架现在支持两种 joint mode 执行策略：

| 策略 | 配置值 | 行为 |
|------|--------|------|
| 每步重求 IK | `per_step_ik` | 每个控制周期都从当前 qpos 出发重新做一次 IK |
| 一次求解 + 关节插值 | `solve_once_interpolate` | 目标改变时只解一次 IK，再按关节位移大小自适应计算插值步数 |

此外，pose 控制还支持笛卡尔分段，但**只在 `per_step_ik` 模式下生效**：

- 位置按 operator 默认的 `control.cartesian_max_linear_step`，或 waypoint 自己的 `max_linear_step` 做直线分段
- 姿态按 operator 默认的 `control.cartesian_max_angular_step`，或 waypoint 自己的 `max_angular_step` 做 SLERP 分段

这层分段发生在 IK 之前，目的是约束末端轨迹形状，而不是只约束关节轨迹形状。

> **`solve_once_interpolate` 模式跳过笛卡尔分段。** 当
> `joint_control_mode=solve_once_interpolate` 时，`MujocoOperatorHandler.move_to_pose`
> 会把整段 waypoint 的最终目标位姿直接交给 env，由 env 在第一次目标变化时一次性求 IK，
> 后续 control step 通过关节插值推进。在这种模式下：
>
> - `cartesian_max_linear_step` / `cartesian_max_angular_step` 不再生效
> - waypoint 自带的 `max_linear_step` / `max_angular_step` 同样被忽略
> - `adaptive_step_scaling`（停滞时自动缩小步长）也被绕过
>
> 想保留每帧重新跟踪笛卡尔子目标的语义，请改用 `per_step_ik` 模式。

#### 一次求解 + 关节插值

这个模式适合单次 IK 已能准确收敛、且希望减少求解次数的任务。若目标存在
不可忽略的残差，它不会继续闭环修正；因此带随机目标与近距离抓取动作的
`aao_configs/pick_and_place_franka.yaml` 使用 `per_step_ik`。

执行过程：

1. 当目标 EEF pose 发生变化时，从**当前 qpos** 出发求一次 IK
2. 求解完成后检查最大关节位移，若超过 `max_joint_delta` 则整体缩放 delta
3. 将“当前关节角 -> IK 解”缓存成一条关节轨迹
4. 根据 `max(abs(q_target - q_current)) / joint_interp_speed` 自适应计算插值步数
5. 在后续这些 control step 中做线性插值
6. 每一步把插值结果写入 actuator ctrl，由 PD 控制器跟踪

这个策略的特点是：

- IK 求解次数更少
- 关节目标变化更平滑，更接近“给一条 joint trajectory”
- 小位移自动用更少插值步，大位移自动分更多步
- 末端轨迹不再是严格意义上的每步笛卡尔重跟踪，而是“先规划一个终点，再在关节空间执行过去”

### 为什么会“卡住”

这里的“卡住”通常不是 IK 直接失败，而是：

- IK 解出了关节角
- 但执行这组关节角之后，末端误差下降得很慢
- 多轮控制后仍然没有明显靠近目标
- 最后被 `timeout_steps` 判成 `move_timeout`

关键点是：**不要只看 IK 有没有解出关节角，要看这组关节角执行后，`position_error` / `orientation_error` 降得快不快。**

正常情况：

- IK 解出来的关节角执行后，末端会稳定朝目标靠近
- 位置和姿态误差会持续明显下降

不正常情况：

- IK 也解出了关节角
- 但执行后误差几乎不变，或者只一点点下降
- 下一轮再解，系统还是在附近反复小修小补

这类问题在笛卡尔步长过大时更容易出现。原因是：

- 每轮给 IK 的子目标离当前状态太远
- 虽然还能求出关节角
- 但这组关节角放到“关节插值 + PD + 下一轮再解”的闭环里，收敛效果很差

所以这里的问题不是“有没有解”，而是：

- **这组解执行后，末端误差是不是能明显下降**

从任务角度看，这本质上可以理解为：

- 迭代 IK 没有真正解成功
- solver 虽然返回了关节角
- 但这组关节角并不真正对应目标位姿，或者对应得不够准确
- 因而放到执行闭环里后，末端不能有效逼近目标

当前框架已经在 `move_to_pose()` 中加入了一个退避机制：

- 若连续若干步几乎没有进展，会自动缩小当前 move 的笛卡尔步长
- 一旦重新出现明显进展，步长会逐步恢复

因此，`max_linear_step` / `max_angular_step` 现在更适合理解为：

- “希望的最大笛卡尔步长上限”
- 而不是“每一帧一定走这么大”

#### 每步重求 IK

这是之前框架中的默认逻辑：

1. 每个控制周期从**当前 qpos** 出发重新求一次 IK
2. 求解完成后做 `max_joint_delta` clamp
3. 将该步的关节目标直接写入 actuator ctrl
4. 下一步再从新的 qpos 继续 solve

这个策略的特点是：

- 更接近连续笛卡尔跟踪
- 目标变化时响应直接
- IK 调用频率更高
- 在某些姿态附近更容易看到“每步都在修正”的控制风格

## IK Solver 实现：MinkIKSolver

位于 [`auto_atom/backend/mjc/ik/mink_ik_solver.py`](../auto_atom/backend/mjc/ik/mink_ik_solver.py)，
基于 [mink](https://github.com/kevinzakka/mink) 微分 IK 库。

### 求解过程

```python
def solve(target_pose_in_base, current_qpos) -> Optional[np.ndarray]:
    # 1. 坐标变换：base frame → world frame（mink 在 world frame 工作）
    pos_w = R_base @ pos_b + base_pos
    quat_w = quat_base ⊗ quat_b

    # 2. 设置 mink 目标 SE3
    eef_task.set_target(SE3(R_w, pos_w))

    # 3. 用 current_qpos 初始化 mink Configuration
    configuration.update(q_seed)
    posture_task.set_target_from_configuration(configuration)  # 动态 posture target

    # 4. 迭代求解
    for _ in range(n_iterations):
        vel = mink.solve_ik(configuration, [eef_task, posture_task], dt, ...)
        configuration.integrate_inplace(vel, dt)

    # 5. Clamp：限制最大关节位移
    delta = solved - current_qpos
    if max(|delta|) > max_joint_delta:
        solved = current_qpos + delta * (max_joint_delta / max(|delta|))
    return solved
```

注意：`MinkIKSolver.solve()` 仍然只负责“求一个目标关节角”。
“一次求解后是否继续做关节插值执行”是在 `UnifiedMujocoEnv.step_operator_toward_target()`
这一层决定的，而不是在 solver 内部完成的。

如果换成解析 IK：

- 通常可以避免“迭代没有真正收敛好”这一类问题
- 但仍然可能遇到解分支选择、关节限位、轨迹连续性和执行层跟踪的问题

所以解析解能减少这类收敛问题，但不能自动解决所有运动控制问题。

### posture_task 的作用

每次 solve 时，posture target 被更新为当前 seed（即 current_qpos）。这意味着：

- IK 在满足末端目标的前提下，倾向于保持关节接近当前配置
- 防止求解器跳到等价但关节差异很大的另一个 IK 分支
- `posture_cost` 控制这个约束的强度（越大越保守，但可能导致末端精度下降）

## YAML 配置

### 基础配置：basis_franka.yaml + 任务 YAML

```yaml
env:
  operators:
    arm:
      arm_actuators: [actuator1, actuator2, ..., actuator7]  # 触发 joint 模式
      eef_actuators: [fingers_actuator]
      pose_site: eef_pose        # EEF 位姿读取 site
  sim_freq: 500
  update_freq: 100              # 每个控制步的物理 substeps = sim_freq / update_freq

backend: auto_atom.backend.mjc.ik.mink_ik_solver.build_franka_backend

task:
  stages:
    - name: pick_source
      object: source_block
      operation: pick
      operator: arm
      param:
        pre_move:
          - position: [0.0, 0.0, 0.12]
            orientation: [-0.7071, 0.7071, 0.0, 0.0]
            reference: object_world
            max_linear_step: 0.02
            max_angular_step: 0.18
          - position: [0.0, 0.0, 0.006]
            orientation: [-0.7071, 0.7071, 0.0, 0.0]
            reference: object_world
            max_linear_step: 0.005
            max_angular_step: 0.08

task_operators:
  arm:
    ik:
      joint_control_mode: solve_once_interpolate
      joint_interp_speed: 0.05
      n_iterations: 300
      dt: 0.1
      position_cost: 1.0
      orientation_cost: 1.0
      posture_cost: 1e-4
      max_joint_delta: 0.8
```

### IK 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `control.cartesian_max_linear_step` | `0.0` | 默认笛卡尔位置分段步长上限（m/tick）。大于 0 时，末端位置按直线小步逼近 |
| `control.cartesian_max_angular_step` | `0.0` | 默认笛卡尔姿态分段步长上限（rad/tick）。大于 0 时，末端姿态按 SLERP 小步逼近 |
| `pose.max_linear_step` | `0.0` | 单个 waypoint 的笛卡尔位置分段步长。若 > 0，则覆盖 operator 默认值 |
| `pose.max_angular_step` | `0.0` | 单个 waypoint 的笛卡尔姿态分段步长。若 > 0，则覆盖 operator 默认值 |
| `joint_control_mode` | `solve_once_interpolate` | Joint 模式执行策略。可选 `solve_once_interpolate` 或 `per_step_ik` |
| `joint_interp_speed` | `0.05` | 当 `joint_control_mode=solve_once_interpolate` 时，每个 control step 允许的最大单关节位移上限（rad/step），系统据此自适应计算插值步数 |
| `control.ik_unreachable_threshold` | `30` | 连续 IK 求解失败次数阈值；超过即把当前 stage 判为 `ik_unreachable` 失败，而不是等到 `timeout_steps` 再报 `move_timeout` |
| `n_iterations` | 300 | 每次 solve 的 mink 迭代步数。越大求解越精确，但越慢 |
| `dt` | 0.1 | 每个 IK 迭代的虚拟时间步（秒）。`n_iterations × dt` = 总积分时长 |
| `position_cost` | 1.0 | EEF 位置跟踪权重 |
| `orientation_cost` | 1.0 | EEF 姿态跟踪权重。增大可提高姿态精度 |
| `posture_cost` | 1e-4 | 关节姿态正则化权重。增大使关节更保守（不易跳变），但降低末端精度 |
| `max_joint_delta` | 0.8 | 单次 solve 允许的最大关节位移（rad）。防止奇异点附近的解跳变 |

### 参数调优指南

### 参数作用分工

这三个参数最容易混淆：

| 参数 | 主要作用 | 调大后的典型效果 | 对直线性的影响 |
|------|----------|------------------|----------------|
| `max_linear_step` | 控制末端位置子目标每次跳多远 | 位置分段更少，移动更快 | 可能变差，太大时更容易绕路或卡住 |
| `max_angular_step` | 控制末端姿态子目标每次转多快 | 姿态分段更少，旋转更快 | 可能变差，姿态变化会更猛 |
| `joint_interp_speed` | 控制已经求出的关节目标在关节空间推进多快 | 关节跟随更快 | 基本不直接改善直线性，只影响关节推进速度 |

一句话理解：

- 想让**末端更直**，优先调 `max_linear_step` / `max_angular_step`
- 想让**已经确定的关节轨迹更快执行**，再调 `joint_interp_speed`
- 不要指望单独增大 `joint_interp_speed` 来修复笛卡尔大步长造成的绕路或卡住

**机械臂运动太慢：**
- 对于远距离接近段，优先增大 `max_linear_step`（如 `0.01 -> 0.02`），减少位置分段数
- 对于姿态变化较大的段，再增大 `max_angular_step`（如 `0.12 -> 0.2`），减少姿态分段数
- 如果末端轨迹已经比较顺但整体还是慢，再增大 `joint_interp_speed`（如 `0.05 -> 0.1`）
- 增大 `max_joint_delta`（如 1.2），允许每步走更远
- 降低 `update_freq`（如 50），增加每步的物理仿真时间，使 PD 控制器有更多时间跟踪

**想提速，同时尽量保持直线性：**
- 不要一次把所有 `max_linear_step` 都调大。先只调远距离 waypoint，近距离下压/插入/放置段保持小步长
- 一般可采用“远距离大步、近距离小步”的分层策略：
  - 远距离接近：`max_linear_step` 可较大，如 `0.02 ~ 0.05`
  - 近距离下压或放置：`max_linear_step` 保持较小，如 `0.003 ~ 0.012`
  - 姿态精细贴合段：`max_angular_step` 保持较小，如 `0.05 ~ 0.12`
- 如果增大 `max_linear_step` 后开始出现明显绕路、摆动或停滞，说明速度已经超过当前 IK 闭环的稳定范围，应回退
- 先保证笛卡尔子目标合理，再用 `joint_interp_speed` 补执行速度，这样通常能在速度和直线性之间取得更好的平衡

**把 `joint_interp_speed` 调大能避免卡住吗：**
- 不一定
- 如果问题只是“关节推进太保守”，增大它会有帮助
- 如果问题是“笛卡尔子目标太远，把 IK/闭环带进了收敛很差的区域”，增大它通常不能从根上解决，甚至可能让摆动更明显
- 这种情况下应优先减小 `max_linear_step` / `max_angular_step`

**经过奇异点时关节跳变 / 末端翻转：**
- 优先使用 `solve_once_interpolate`，减少连续重求解带来的分支抖动
- 减小 `max_joint_delta`（如 0.5），限制关节速度
- 增大 `posture_cost`（如 1e-3），使 IK 更倾向于保持当前关节构型
- 调整 keyframe 中的初始关节角，使 home 配置远离奇异区域

**末端姿态不准确：**
- 增大 `orientation_cost`（如 2.0）
- 增大 `n_iterations`（如 500），给更多迭代时间
- 减小 `posture_cost`（如 1e-5），放松关节约束

**IK 求解太慢（影响实时性）：**
- 使用 `solve_once_interpolate`，降低 IK 调用频率
- 减小 `n_iterations`（如 100），但需确保精度足够
- 增大 `dt`（如 0.2），每步走更远但可能不稳定

### 任务配置中的初始位姿与 IK

`task_operators.<name>.initial_state.base_pose` 描述的是**机械臂底座**的
世界位姿，不是 EEF 初始位姿。`base_pose` 与 `eef_pose` 共用
`PoseOverrideConfig`：`position` 和 `orientation` 可以分别省略，省略的分量
从当前 keyframe/注册位姿继承；姿态三元组统一为 RPY 顺序
`[roll, pitch, yaw]`，四元组为 XYZW quaternion。需要混合坐标系时，
`position` 可展开为 `x/y/z`，RPY 姿态可展开为 `roll/pitch/yaw`；每个轴支持
标量（继承全局 `reference`）或 `{value, reference}`（轴级优先）。例如：

```yaml
base_pose:
  reference: door__handle_grasp_center
  position:
    x: 0.2474
    y: -0.4666
    z: {value: -0.1, reference: world}
```

底座可以直接写在世界坐标系，也可以锚定到组合场景里的 site、body、geom 或
joint。命名 frame 的变换约定为：

```text
T_world_base = T_world_reference × T_reference_base
```

```yaml
task_operators:
  arm:
    initial_state:
      base_pose:
        reference: door__handle_grasp_center
        position: [0.2474, -0.4666, -0.10]  # reference frame
        orientation: [0.0, 0.0, 0.7071, 0.7071]  # XYZW
      eef_pose:
        reference: base
        position: [0.32, 0.0, 0.18]
        orientation: [0.0, 1.5708, 0.0]  # RPY
```

初始化顺序是：场景 keyframe → `env.initial_joint_positions` → operator home →
`task.initial_pose` 对象覆盖 → operator `base_pose` → operator
`eef_pose`/gripper control → randomization。命名引用只在 setup/reset 时解析并
固定为世界位姿；被引用的关节之后运动时，底座不会跟随。
在 joint-mode 下，解析后的 `base_pose` 同时移动物理 root body；纯 mocap
operator 则只更新用于坐标换算的 virtual base，注册的 mocap home 不会因该
字段被改写。
`task.randomization.<name>.base/eef` 的随机化基线是上述覆盖后的实际 home
位姿。

对于 Franka 等固定基座机械臂：

1. **所有 waypoint 都应显式指定 orientation**——如果省略，IK 可能求出不同的腕关节构型；
2. **keyframe 中的 joint7 应接近任务所需的末端朝向**，避免首次移动时大幅旋转；
3. 若不打算移动底座，`base_pose` 应与 XML 中机械臂根 body 的位置一致；若需要按门、桌面等
   场景 frame 放置机械臂，则显式使用命名 `reference`，不要在多个配置层重复推导坐标。

### Home EEF 设置时的 IK 失败处理

`set_operator_home_eef_pose`（由 `MujocoOperatorHandler.set_home_end_effector_pose`、
`build_mujoco_backend` 中的 `initial_state.eef_pose`、以及 `task.randomization`
的 `arm.eef` 采样间接调用）在 joint 模式下会做一次 IK 求解，把目标 EEF 转成
arm `home_arm_qpos`。

如果目标位姿超出工作空间，`ik_solver.solve` 返回 `None`。这种情况下：

- 不再抛 `RuntimeError`，而是记录一条 `WARNING` 日志（logger
  `auto_atom.basis.mjc.mujoco_env`），列出失败的目标 pos / quat。
- `home_arm_qpos` 保持不变；后续 `home(env_mask)` 会回到上一次成功的
  home 关节构型（或 keyframe 默认值）。
- 调用方（包括 `tune_randomization_extremes.py` 等遍历极值的工具）继续执行，
  不会因为单次不可达就退出。

实际后果：YAML 中配置了不可达的 `initial_state.eef_pose` 不会再让程序崩溃，
但日志里会出现该警告——看到这条 warning 时应当回到配置里收紧
`task.randomization.arm.eef` 的范围或修正 `initial_state.eef_pose`，否则那一
帧的 home 位姿与配置不一致。

### 运行期 `move_to_pose` 中的 IK 失败处理

`UnifiedMujocoEnv._solve_ik` 是 env 内所有 IK 调用的唯一入口（包括
`per_step_ik` 的每帧求解、`solve_once_interpolate` 的目标变更求解、以及
`teleport_operator` 等），它统一负责两件事：

1. **失败日志节流**。`ik_solver.solve` 返回 `None` 时增加
   `ik_failure_streak` 计数，并按 1 → 10 → 20 → ... → 100 → 200 → ...
   的节奏打 `WARNING`，列出 base 系下的目标 pos / quat 与当前 seed qpos。
   这样 `per_step_ik` 模式下连续撞同一个不可达目标也不会刷屏。
2. **关节限位接近告警**（见下一节）。

`MujocoOperatorHandler.move_to_pose` 在每个 control step 之后读取 env 暴露的
`get_operator_ik_failure_streak(op_name)`：

- 当连续失败次数 ≥ `control.ik_unreachable_threshold`（默认 30）时，**立刻**
  把这一阶段判为失败，详情中会带上：
  - `event: "ik_unreachable"`
  - `failure_category: "ik_unreachable"`
  - `failure_reason`：连续失败次数 + "target pose is outside the arm's reachable workspace"
  - `ik_failure_streak`：触发时的连续失败计数
- 不再傻等到 `timeout_steps` 才超时——前者明确指向"目标不可达"，后者只能说明"末端没收敛"。

YAML 配置示例：

```yaml
task_operators:
  arm:
    control:
      timeout_steps: 220
      # 连续 30 次（约 0.6s @ update_freq=50）IK 解不出来即判失败
      ik_unreachable_threshold: 30
```

调小 `ik_unreachable_threshold`（如 10）能更早终止跑飞的 stage；调大它对持续
"边缘可达"的目标更宽容。把它设为一个非常大的数，行为会回退到原来的
"等到 `timeout_steps` 再 `move_timeout`"。

## 关节限位接近告警 (joint-limit proximity warning)

每次 IK 求解成功后，env 还能可选地检查解出来的关节角是否贴近硬件限位
（`mjModel.jnt_range`）：当某个关节距离上下限小于 `~0.05 rad`（约 2.9°）时，
打一条 `WARNING`，提示该解可能位于奇异区或即将撞限位。

### 启停

默认**关闭**——demo 跑起来时这是噪声。开启方式：

```python
env.set_joint_limit_warning_enabled(True)
```

`examples/tune_randomization_extremes.py` 在打开时强制 enable，因为它的本职就是
扫极端 pose 找出工作空间边界。

### 行为

- 单关节单侧（lower / upper）只告一次，进入危险带（< 0.05 rad）时触发，
  **退出**危险带需要至少回退 `0.10 rad`（hysteresis），避免在边界处来回刷
  warning。
- 日志里会包含：context（`per_step_ik` / `solve_once_interpolate` / `teleport_operator` 等）、
  operator 名、joint 名、当前角（rad + deg）、距限位的距离、以及限位本身。

### 用途

- 调 `task.randomization.arm.eef` 范围时，看哪些采样让 arm 顶到限位
- 调 `initial_state.base_pose` / `initial_state.eef_pose` 时，看 home 位姿是否
  靠近限位
- 调 keyframe / `initial_joint_positions` 时，提前发现首步 IK 就贴近限位的情况

## 自定义 IK Solver

实现 `IKSolver` 协议即可替换 mink：

```python
from auto_atom.runtime import IKSolver
from auto_atom.utils.pose import PoseState

class MyIKSolver:
    def solve(
        self,
        target_pose_in_base: PoseState,  # 基座系下的末端目标位姿 (xyzw)
        current_qpos: np.ndarray,         # 当前关节角
    ) -> Optional[np.ndarray]:            # 目标关节角，无解返回 None
        ...
```

然后编写自己的 `build_*_backend` 工厂函数，在 YAML 的 `backend` 字段中引用。
