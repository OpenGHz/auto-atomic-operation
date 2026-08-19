# TaskRunner Update 粒度与关键点步进方案分析

> `execution.update_boundary`、`execution.interval_selection` 与 viewer 边界刷新
> 均已实现。本文同时保留其他实现方案的对比，以及内部观测 callback 的后续
> 演进建议。

本文分析 `aao-demo --config-name pick_and_place` 为什么需要多次
`TaskRunner.update()` 才能完成一个配置点位，并比较“每次外部 update
直接推进到一个关键点”的可选实现。

## 结论

当前实现保留 `TaskRunner.update()` 的默认单控制周期语义，并通过
`execution.update_boundary` 提供可选的**宏步进**。一次公开调用可以在内部重复
执行现有控制周期，直到到达指定的 primitive、keypoint 或 stage 边界，或任务
失败、超时、结束。

对 `pick_and_place` 而言：

- 当前执行包含 6 个 pose primitive 和 2 个 eef primitive；
- 默认模式实测需要 248 次公开 `update()`；
- `primitive` 或 `keypoint` 模式可表现为 8 次外部调用；
- `stage` 模式可表现为 2 次外部调用；
- 内部仍执行相同的物理控制周期，因此不会自动缩短仿真时间，但可以减少
  外部交互次数；配置 `render_internal_updates: false` 后，viewer 只显示每次
  公开调用的最终 boundary 状态。

如果要求物理状态真正瞬时跳到点位，只能使用 kinematic teleport 或状态
snapshot。这两类方案会绕过正常接触动力学，不适合作为 pick/place
数据采集的默认模式。

## 已实现：统一执行配置

执行粒度和闭区间选择统一位于任务文件顶层的 `execution` 节：

```yaml
execution:
  update_boundary: keypoint
  render_internal_updates: false
  max_internal_updates_per_update: 10000
  interval_selection:
    start:
      stage: pick_source
      phase: post_move
      waypoint: 0
    stop:
      stage: place_source
      phase: post_move
      waypoint: 0
    max_fast_forward_updates: 10000
```

- `update_boundary` 支持 `control_tick`、`primitive`、`keypoint`、`stage`；
- 默认 `control_tick`，保持历史行为；
- `render_internal_updates` 默认 `true`；设为 `false` 时内部物理照常运行，但
  viewer 只在公开边界刷新一次；
- `reset()` 复用正常状态机和物理控制，执行并包含 start，reset observation
  就是 start 状态；
- 后续公开 `update()` 按 `update_boundary` 返回；
- stop 完整到达且相关 condition 通过后，任务成功结束，不再执行下一点；
- stop 优先于更粗的 update boundary，例如 `stage` 不会越过 stage 中间的 stop；
- `start == stop` 时，reset 到达该点后直接返回成功；
- arc 的内部拆分不会暴露为多个可选择 waypoint；`eef` 使用唯一索引 0。

两个安全上限相互独立：

- `max_internal_updates_per_update` 限制一次公开 `update()` 内每个环境的内部
  controller update 数；
- `interval_selection.max_fast_forward_updates` 限制 `reset()` 快进到 start 的
  controller update 数。

二者默认都是 `10000`。前者耗尽会产生
`internal_update_limit_exceeded` 终态失败，后者耗尽会产生 interval
fast-forward timeout；修改其中一个不会影响另一个。

详细配置与校验规则见
[Stages & Waypoints](../task-configuration/stages_and_waypoints.md#inclusive-task-interval-selection)。

## 当前任务与执行粒度

通过 `aao-info pick_and_place` 可确认任务包含两个 stage：

1. `pick_source`
2. `place_source`

`aao_configs/pick_and_place.yaml` 中每个 stage 都会展开成四个 primitive：

```text
pre_move[0] -> pre_move[1] -> eef -> post_move[0]
```

所以完整任务共有：

| 类型 | 数量 |
| --- | ---: |
| pose primitive | 6 |
| eef primitive | 2 |
| primitive 总数 | 8 |
| stage 数量 | 2 |

`TaskFlowBuilder.build_actions()` 负责把 `pre_move`、`eef` 和 `post_move`
配置展开成 primitive action。在默认 `control_tick` 模式下，
`TaskRunner.update()` 对每个选中的环境只执行当前 primitive 的一个控制周期：

- 后端返回 `RUNNING`：保持当前 primitive，并立即返回；
- 后端返回 `REACHED`：推进 action index，但不在同一次 update 中继续执行
  下一个 primitive；
- 后端返回 `FAILED` 或 `TIMED_OUT`：终止当前 stage 和任务环境。

在 `primitive`、`keypoint` 或 `stage` 模式下，公开 `update()` 会在内部重复这
段逻辑，到达对应边界后再返回。完整的完成判定流程见
[Execution Completion Flow](../task-configuration/execution_completion_flow.md)。

## 为什么 `pick_and_place` 会逐步逼近

### 不是显式 Cartesian 小步分段

`pick_and_place` 的 waypoint 没有配置 `max_linear_step` 或
`max_angular_step`，两者默认值都是 `0`。MuJoCo operator 的默认
`cartesian_max_linear_step` 和 `cartesian_max_angular_step` 也都是 `0`。

因此，当前任务已经在每个控制周期向后端传递完整目标位姿；慢速逼近并不是
YAML 将一段运动拆成了大量小 waypoint。

### Mocap body 与物理 body 之间存在软约束

该任务继承 `basis_mocap_eef`，`arm_actuators` 为空，因此 operator 使用
mocap 模式。控制链路是：

```text
TaskRunner.update()
  -> MujocoOperatorHandler.move_to_pose()
  -> 写入 mocap 目标位姿
  -> env.update()
  -> mj_step × n_substeps
  -> 检查实际 EEF 与目标的误差
```

Robotiq 的物理 body 通过 weld 约束跟随 mocap body：

```xml
<weld body1="robotiq_mocap"
      body2="robotiq_interface"
      solref="0.3 1"
      solimp="0.95 0.99 0.001"/>
```

`solref` 的时间常数使物理 body 不会在写入 mocap 目标后立即重合，而是在
后续物理步中逐渐收敛。

### 每次 update 只推进有限仿真时间

默认配置为：

```yaml
env:
  sim_freq: 600
  update_freq: 30
```

所以每次控制 update 执行 20 个 MuJoCo substep，并推进约 `1/30 s` 的
仿真时间。只有实际 EEF 的位置和姿态误差都进入 tolerance，pose primitive
才返回 `REACHED`。夹爪动作还需要满足 actuator、接触和 settle 条件。

`sim_freq`、`update_freq` 和 `n_substeps` 的关系见
[sim_freq 与 update_freq](../task-configuration/sim_freq_update_freq.md)。

### Viewer 还会增加墙钟等待

`pick_and_place.yaml` 配置了：

```yaml
viewer:
  step_delay: 0.03
```

每次环境 update 后 viewer 会额外等待 30 ms。该参数不改变 primitive
完成条件或 update 数量，但 248 次 update 会额外产生约 7.44 秒的观看等待。

## 基准实测

测试条件：

- `pick_and_place`
- `env.batch_size=1`
- viewer disabled
- 默认 seed 和控制参数

各 primitive 使用的控制 update 数如下：

| Stage | Primitive | 控制 update 数 |
| --- | --- | ---: |
| pick | `pre_move[0]` | 49 |
| pick | `pre_move[1]` | 37 |
| pick | close eef | 5 |
| pick | `post_move[0]` | 44 |
| place | `pre_move[0]` | 40 |
| place | `pre_move[1]` | 28 |
| place | open eef | 10 |
| place | `post_move[0]` | 35 |
| 合计 | 8 primitives | 248 |

总仿真时间约为：

```text
248 / 30 Hz = 8.27 s
```

这说明主要成本来自物理收敛与夹爪 settle，而不是配置点位数量过多。

## “每次 update 到一个关键点”的语义

实现区分了四种边界：

| 边界 | `pick_and_place` 外部调用数 | 语义 |
| --- | ---: | --- |
| `control_tick` | 约 248 | 默认行为；一次推进一个控制周期 |
| `primitive` | 8 | 每个 pose 或 eef primitive 完成后返回 |
| `keypoint` | 8 | 每个 YAML waypoint 完成后返回；arc 子段合并为同一 keypoint |
| `stage` | 2 | 一次完成整个 pick 或 place stage |

对当前不含 arc 的 `pick_and_place`，`primitive` 和 `keypoint` 的调用数相同；
差异在 arc waypoint 上。arc 可以展开为多个 runtime primitive：`primitive` 在
每个子段后返回，而 `keypoint` 完成全部子段后才返回。interval endpoint 始终
引用稳定的 YAML keypoint，不引用可能随实现变化的 primitive。

`keypoint` 包含 `eef` keypoint，因此不需要把 close/open 隐式归组到相邻 pose。

## 实现方案对比

### 方案 A：Runner 宏步进到配置 boundary（已采用）

一次公开调用在 `TaskRunner` 内部重复执行现有 control tick，直到配置的
primitive、keypoint 或 stage boundary 完成。默认 `control_tick` 不进入宏循环。

| 维度 | 评价 |
| --- | --- |
| 外部语义 | 一次调用到达配置的 update boundary |
| 物理正确性 | 高；保留当前 IK、weld、接触、grasp/place 条件和 timeout |
| 改动范围 | 中；主要位于 `TaskRunner` 和 CLI 调用层 |
| 调用延迟 | 单次调用会阻塞，最长到 primitive timeout |
| 数据采集 | 宏步调用方只看到边界状态；需要未来的 callback 才能保留内部轨迹 |
| 推荐程度 | 最高 |

优点：

- 最大程度复用当前稳定的 primitive 状态机；
- 不需要修改任务 YAML；
- 不改变 pose reference、waypoint randomization 和 stage condition；
- mock、mocap 和 joint operator 都可以共享相同的高层语义。

实现要点：

- batch 中每个环境独立到达自己的首个边界，完成后从 pending mask 移除；
- `max_internal_updates_per_update` 防止异常 handler 永远返回 `RUNNING`；
- `TaskUpdate.details[env].execution` 报告 boundary、event 和实际 internal update 数；
- interval stop 在检查完绑定条件后优先抢占更粗的 boundary；
- summary 的模拟时间按内部 controller update 数计算，而不是外部调用数。

### 方案 B：Handler 内部阻塞或 action repeat

让 `move_to_pose()` 和 `control_eef()` 自己循环物理步，直到返回终态。

优点是 backend 可直接优化紧密循环，缺点是 primitive 状态、batch mask、stage
条件和统计被下沉到更低层。通用 `OperatorHandler` 接口也会从非阻塞控制器变成
可能长时间阻塞的接口。

该方案与 Runner 宏步效果接近，但层次耦合更重，不建议作为第一版。

### 方案 C：轨迹段或 Action Server

将当前“发送目标、推进物理、轮询反馈”拆分为：

1. `send_goal()`：发送一个 waypoint 或 time-parameterized trajectory；
2. backend/controller 独立执行；
3. `poll()` 或 callback 返回运行状态；
4. 支持 cancel、reset 和 timeout。

该架构适合真实机器人、异步控制和高频独立采样，也可以在执行前加入 joint
limit 与碰撞检查。但它需要重构当前 `move_to_pose()` 中混合的发命令和物理
推进逻辑，实施成本最高。

`solve_once_interpolate` 可以视为这一方向的局部基础，但目前仍需要外部每次
update 推进一步。

### 方案 D：调整 update_freq 或控制参数

降低 `update_freq` 会让一次控制 update 包含更多 MuJoCo substep。实测结果：

| `update_freq` | 完成所需 update | 仿真时间 |
| ---: | ---: | ---: |
| 30 Hz | 248 | 8.27 s |
| 10 Hz | 92 | 9.20 s |
| 5 Hz | 50 | 10.00 s |
| 2 Hz | 25 | 12.50 s |
| 1 Hz | 18 | 18.00 s |

它可以减少公开调用次数，但不能稳定保证一次 update 完成一个 primitive，并且会：

- 改变控制与观测时间间隔；
- 增加每次调用的计算量和仿真时间跨度；
- 对 joint mode 的 PD、IK 和插值稳定性产生影响；
- 改变 eef settle、timeout 等以 control tick 计数的语义。

调整 weld `solref`、扩大 tolerance 或修改 `joint_interp_speed` 也只能加速收敛，
不能建立明确的关键点 API。扩大 tolerance 还可能让 primitive 在实际未到配置点时
提前完成。

### 方案 E：Kinematic teleport

环境已有 `apply_pose_action(..., kinematic=True)` 和 teleport 相关能力，可以直接
写入 qpos、mocap/freejoint，并调用 `mj_forward()`。

优点：

- 位姿跳转速度最快；
- endpoint 确定性高；
- 适合 reset、状态对齐和无接触预览。

问题：

- 跳过正常碰撞与接触动力学，可能穿过物体；
- teleport 夹爪不会自动把已抓物体搬到新位置；
- 当前系统没有通用的 held-object attachment；
- 夹爪关节和被动连杆仍需要额外同步或 settle；
- `mj_forward()` 不推进 MuJoCo `data.time`，AIRDC 的 `is_updated()` 可能把该帧
  判定为未更新并跳过。

若增加临时 weld 或逻辑 attachment 来同步被抓物体，系统就从物理抓取转向符号化
搬运，得到的数据不再等价于正常 pick/place 仿真。

因此该方案应是显式的 debug/replay 模式，不能成为默认任务执行模式。

### 方案 F：预模拟后恢复关键帧 snapshot

reset 后先用正常物理控制跑完整任务，在每个 primitive boundary 保存完整 MuJoCo
状态；正式播放时，每次 update 恢复下一个 snapshot。

相比裸 teleport，它能保留关键点时物体、夹爪和接触系统的完整状态。但它仍有明显
限制：

- 每次随机化后通常都要重新预模拟；
- 不能处理在线扰动、策略分支和失败恢复；
- 状态缓存占用内存，并且高度依赖 MuJoCo；
- 时间戳和内部轨迹需要额外映射。

适合确定性的离线回放或回归测试，不适合通用在线 runner。

## 已采用设计

### 保留 `update()` API，以配置选择返回边界

实现没有新增另一套推进 API，也没有改变 `TaskRunner.update()` 的默认含义。
任务文件通过以下配置选择行为：

```yaml
execution:
  update_boundary: control_tick  # control_tick | primitive | keypoint | stage
  max_internal_updates_per_update: 10000
```

`control_tick` 与旧行为一致；其余值让同一个 `update(env_mask)` 在内部继续调用
原有逐 tick 状态机。调用结果仍是 `TaskUpdate`，每个环境的
`details.execution` 额外报告：

```text
event                            本次返回原因
update_boundary                  配置的边界
internal_updates                 本次公开调用实际执行的内部 update 数
max_internal_updates_per_update  配置的安全上限
```

因此 `RunnerBase` 和现有 `TaskRunner` 调用方仍使用同一方法；不配置
`execution` 的任务无需迁移。当前 IPC service 封装的是 `PolicyEvaluator`，仅支持
`control_tick`，不提供这组 TaskRunner 宏步能力。

### Batch 推进规则

每个选中环境只允许完成一个目标 boundary：

```text
pending = selected envs

while pending is not empty:
    对 pending 环境执行一个 control tick
    对已完成 boundary、done、failed 或 timed out 的环境清除 pending

return
```

不同环境可以消耗不同数量的内部 tick，但不能因为某个环境较快就让它提前跨过
两个目标 boundary。任务结束、失败、timeout、内部上限失败和 interval stop 也会
立即把该环境从 pending 集合移除。

### 已实现：Viewer 快进独立开关

Runner 宏步只改变外部调用粒度。如果内部每个物理 tick 仍然同步 viewer 并执行
`step_delay`，画面仍会逐步运动，只是调用方被阻塞在一次函数调用中。

viewer-only fast-forward 独立于公开调用粒度配置：

- `render_internal_updates=true`：显示完整运动过程；
- `render_internal_updates=false`：内部不 sync/sleep，只在 boundary 刷新一次；

boundary 的最终刷新本身不执行 `step_delay`。两种模式执行相同的物理 tick、IK、
接触、抓取判定与 timeout，也不影响显式 `capture_observation()` 或 camera render。
尚未实现的 dense-recording callback 将继续与 viewer 策略保持独立。

### 数据采集影响

AIRDC 当前按主循环 tick 调用 `runner.update(update_mask)`，采样器也按外部 tick
捕获观测。如果直接替换成 primitive 宏步而不增加内部回调：

- 一条 `pick_and_place` 轨迹可能从约 248 帧变成约 8 帧；
- 中间速度、接触、视觉变化和 action-observation 对齐全部丢失；
- 最终 boundary 是否在保存前被捕获取决于 manager 调度顺序；
- summary 中的 update 数不再代表物理控制周期数。

因此当前使用建议是：

- `aao-demo` 交互和调试可启用宏步模式；
- AIRDC 默认继续使用 `control_tick`；
- 当前宏步只暴露 boundary observation，不暴露内部观测 callback；
- 若 AIRDC 未来需要“外部一次关键点、内部仍保存完整轨迹”，可增加
  `on_internal_update` callback 或 ring buffer 收集内部帧；
- 数据 metadata 同时记录 outer boundary index 和 internal control tick。

数据采集的现有控制/观测顺序见
[Data Collection](../tools/data_collection.md)。

### Policy evaluation 与兼容性

`ConfigDrivenDemoPolicy` 和 `PolicyEvaluator` 有独立的 action 应用与状态推进路径。
外部 policy 每个 control tick 都需要提供新 action，因此当前
`PolicyEvaluator` / `aao-eval` 明确拒绝：

- 任意 `execution.interval_selection`；
- 任意非 `control_tick` 的 `execution.update_boundary`。
- `execution.render_internal_updates=false`。

`aao-demo` 的 `max_updates` 和外部采集循环中的同名限制都表示公开
`TaskRunner.update()` 调用次数，不表示内部 controller tick 数。内部消耗通过
`details.execution.internal_updates` 和模拟时间统计体现。

`StageConfig.blocking` 当前只记录在执行结果中，并未实现异步 stage 调度，不能直接
用它实现关键点步进。

## 配置归属

关键点步进只改变 runner 的执行粒度，不改变机器人、物体、场景或 operation flow。
按照任务复用规则，不应为它复制出新的 `pick_and_place_*` 任务 YAML。

已实现的边界与区间选择都属于任务文件的 runner 执行层，因此集中在与 `task`
同级的 `execution` 节，而不是放入描述 operation flow 的 `task` 内：

```yaml
execution:
  update_boundary: keypoint
  render_internal_updates: false
  max_internal_updates_per_update: 10000
  interval_selection:
    start: {stage: pick_source, phase: post_move, waypoint: 0}
    stop: {stage: place_source, phase: post_move, waypoint: 0}
    max_fast_forward_updates: 10000
```

区间 endpoint 强依赖 stage 定义，但它选择的是“本次如何执行现有任务”，不是任务
本身有哪些阶段；集中在 `execution` 既保留依赖关系，也避免把 rollout policy 混入
可复用的任务语义。误放在顶层的 `interval_selection`、`update_boundary`、
`render_internal_updates` 和两个安全上限字段会被配置校验明确拒绝，并提示对应的
`execution...` 路径，避免静默失效。

## 测试覆盖与后续验证

### 已覆盖的核心行为

- 默认与显式 `control_tick` 行为一致；
- `primitive`、`keypoint`、`stage` 分别在目标边界返回；
- arc 是多个 primitive、一个 keypoint；
- batch 中不同环境以不同速度完成，但每个环境只跨一个边界；
- `max_internal_updates_per_update` 耗尽后显式失败；
- interval stop 抢占更粗的 stage boundary；
- `PolicyEvaluator` 拒绝 interval 和非 `control_tick` boundary；
- viewer 内部刷新可折叠为一次 boundary 刷新，异常退出后仍恢复正常刷新；
- 宏步 summary 使用每个环境的实际内部 update 数计算模拟时间。

### 仍值得持续验证

- 固定 seed 的 `pick_and_place` 成功完成；
- primitive/keypoint boundary 数严格为 8，stage boundary 数严格为 2；
- 宏步与逐 tick 的最终 success、stage records、object state 和模拟时间一致；
- viewer fast-forward 与逐 tick viewer 模式的最终物理结果一致；
- teleport 模式明确验证抓取物体不会被错误假定为自动跟随。

### 数据与兼容性测试

- 未来 dense recording callback 保留内部帧、时间戳和最终帧；
- boundary-only 模式产生预期的关键点帧数；
- AIRDC 不会因 kinematic 模式未推进 `data.time` 而静默丢帧；
- `aao-demo` 与 `aao-eval` parity 保持不变；
- summary 同时正确报告 outer updates、internal updates 和 simulated time。

## 后续演进顺序

1. 根据 AIRDC 是否需要密集轨迹，加入 internal observation callback 或 buffer。
2. 为 boundary-only 与 dense-recording metadata 明确 outer/internal 索引。
3. 真机或异步控制需求明确后，再评估 trajectory/action-server 架构。

## Related

- [Execution Completion Flow](../task-configuration/execution_completion_flow.md)
- [Stages & Waypoints](../task-configuration/stages_and_waypoints.md)
- [sim_freq 与 update_freq](../task-configuration/sim_freq_update_freq.md)
- [IK Control](../ik-motion-control/ik_control.md)
- [Data Collection](../tools/data_collection.md)
