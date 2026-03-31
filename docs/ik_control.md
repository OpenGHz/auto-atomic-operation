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

此外，pose 控制现在还支持独立于 joint mode 的笛卡尔分段：

- 位置按 operator 默认的 `control.cartesian_max_linear_step`，或 waypoint 自己的 `max_linear_step` 做直线分段
- 姿态按 operator 默认的 `control.cartesian_max_angular_step`，或 waypoint 自己的 `max_angular_step` 做 SLERP 分段

这层分段发生在 IK 之前，目的是约束末端轨迹形状，而不是只约束关节轨迹形状。

#### 一次求解 + 关节插值

这是当前 Franka 示例任务推荐的模式，`aao_configs/pick_and_place_franka.yaml`
默认使用它。

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

### 基础配置：base_franka.yaml

```yaml
env:
  env:
    config:
      operators:
        - name: arm
          arm_actuators: [actuator1, actuator2, ..., actuator7]  # 触发 joint 模式
          eef_actuators: [fingers_actuator]
          pose_site: gripper        # EEF 位姿读取 site
      sim_freq: 500
      update_freq: 100              # 每个控制步的物理 substeps = sim_freq / update_freq

backend: auto_atom.backend.mjc.ik.mink_ik_solver.build_franka_backend

stages:
  - name: pick_source
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

operators:
  - name: arm
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

### 任务配置中的注意事项

对于 Franka 等固定基座机械臂：

1. **所有 waypoint 都应显式指定 orientation**——如果省略，IK 可能求出不同的腕关节构型
2. **keyframe 中的 joint7 应接近任务所需的末端朝向**——避免首次移动时大幅旋转
3. **`base_pose` 应匹配 XML 中机械臂底座的实际位置**

```yaml
operators:
  - name: arm
    initial_state:
      base_pose:
        position: [-0.45, -0.06, 0.0]  # 与 XML 中 link0 位置一致
        orientation: [0, 0, 0, 1]
```

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
