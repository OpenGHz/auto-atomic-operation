# Batch-First Async Runtime Refactor

## Summary

将仓库从“单 backend + 单 env + 标量状态”的执行模型，重构为“同构 batched backend + 每个 env 独立异步推进 + 所有 obs/action 显式带 `env_dim`”的模型。

本期目标不是做一个外层 `for` 循环包装，而是把公共抽象统一升级为 batch-first：
- 单环境不再是主语义，只是 `batch_size=1` 的特例
- runtime / backend / basis / examples / tests / docs 全部切到批语义
- MuJoCo batch 采用“同一 `EnvConfig/XML` 复制 N 份 env 实例”的实现，不支持本期异构 batch

## Key Changes

### 1. 统一公共接口为 batch-first

把 `auto_atom.runtime` 的核心抽象改成按 env 维返回/接收批数据：

- `SceneBackend` 改为 batched 语义，至少显式暴露：
  - `batch_size: int`
  - `reset(env_mask: Optional[np.ndarray] = None) -> None`
  - `get_operator_handler(name) -> BatchedOperatorHandler`
  - `get_object_handler(name) -> Optional[BatchedObjectHandler>`
  - `is_object_grasped(...) -> np.ndarray[bool]`
  - `is_operator_grasping(...) -> np.ndarray[bool]`
  - `is_operator_contacting(...) -> np.ndarray[bool]`
  - `is_object_displaced(...) -> np.ndarray[bool]`
  - `set_interest_objects_and_operations(...)` 支持逐 env 输入或广播输入

- `ObjectHandler` / `OperatorHandler` 升级为 batched 版本：
  - `get_pose()` 返回 batched `PoseState`
  - `set_pose()` 接收 batched `PoseState` 和可选 `env_mask`
  - `move_to_pose()` / `control_eef()` 返回逐 env `ControlResult`
  - `get_end_effector_pose()` / `get_base_pose()` 返回 batched pose

- `ControlResult` 从单个 `signal + details` 改为：
  - `signals: np.ndarray[ControlSignal]`，shape `(B,)`
  - `details: list[dict[str, Any]]` 或 `dict[str, batched_value]`
  - 保证 runtime 能按 env 分别判断 `RUNNING/REACHED/TIMED_OUT/FAILED`

- `PoseState` 扩展支持 batched 表达：
  - `position`: `(B, 3)`
  - `orientation`: `(B, 4)`
  - 单 env 只允许通过 `B=1` 表达，不再保留裸 `(3,)` / `(4,)` 作为主接口

### 2. 重写 TaskRunner 为“逐 env 异步状态机”

把 `TaskRunner` 从单个 `_active_stage` 改成按 env 维护状态：

- `ExecutionContext` 绑定 batched backend
- `ActiveStageState` 改为逐 env 状态数组或 per-env state 列表，至少包含：
  - `stage_index[B]`
  - `action_index[B]`
  - `current_stage_plan[B]`
  - `initial_object_pose[B]`
  - `done_mask[B]`
  - `success_mask[B]`
  - `failed_mask[B]`

- `TaskFlowBuilder` 继续只生成“单份 stage/action 模板”
  - 因为 batch 是同构任务复制，不需要为每个 env 单独编译流程
  - runtime 在执行时为每个 env 维护当前走到模板中的哪一步

- `TaskRunner.reset()` 语义改为：
  - reset 全 batch 或由 `env_mask` 指定子集
  - 清空每个 env 的执行游标
  - 返回 batched `TaskUpdate`

- `TaskRunner.update()` 语义改为：
  - 一次调用推进所有未完成 env
  - 不再假设所有 env 处于同一个 stage / action
  - 对每个 env 独立做前置条件、动作执行、阶段完成、失败判断
  - 返回 batched `TaskUpdate`

- `TaskUpdate` 改为逐 env 状态：
  - `stage_index: np.ndarray[int] | None`
  - `stage_name: list[str]`
  - `status: np.ndarray[StageExecutionStatus]`
  - `done: np.ndarray[bool]`
  - `success: np.ndarray[bool | None]`
  - `phase: list[str | None]`
  - `phase_step: np.ndarray[int | None]`
  - `details: list[dict[str, Any]]`

- `ExecutionRecord` 改为显式带 `env_index`
  - `runner.records` 返回扁平记录列表，每条记录对应一个 env 的一个 stage
  - 便于离线分析、过滤和回放

### 3. 让 MuJoCo basis 变成 BatchedEnv 聚合层

本期不要尝试“单个 MuJoCo 引擎内原生向量化”，而是在 basis 层引入批调度器：

- 新增 batched basis，如 `BatchedUnifiedMujocoEnv`
  - 内部持有 `list[UnifiedMujocoEnv]`
  - 每个子 env 拥有独立 `MjModel/MjData`
  - 共享同一份 `EnvConfig` 模板
  - 通过 `batch_size`、seed 偏移、randomization 实现同构复制

- `create_mujoco_env` 改为注册 batched env
  - 配置新增 `batch_size`
  - 必要时允许 `seed_stride` / `viewer_env_index` 之类批专用参数

- `step(action)` 改为接收 shape `(B, action_dim)`
- `capture_observation()` 改为返回同一套 key，但每个 key 的 `data` 都带首维 `B`
  - 标量/向量低维量：`(B, D)`
  - RGB：`(B, H, W, 3)`
  - depth/mask：`(B, H, W)`
  - heatmap：`(B, H, W, C)`
  - tactile/wrench 等同理首维补 `B`

- `get_info()` 改为返回：
  - 共享静态元信息只保留一份
  - 可变状态信息要么删除，要么改成 batched 字段
  - 不混用单 env / 多 env 结构

### 4. 重写 MuJoCo backend handlers 为逐 env 控制

`auto_atom/backend/mjc/mujoco_backend.py` 需要从“直接读写一个 env”改成“按 env 索引调度 batched env”：

- `MujocoObjectHandler`
  - `get_pose()` 聚合所有子 env 的 body pose 为 `(B, 3)/(B, 4)`
  - `set_pose()` 支持全量或 `env_mask` 部分设置

- `MujocoOperatorHandler`
  - 所有缓存状态改成 batched数组：
    - `_last_move_key[B]`
    - `_move_steps[B]`
    - `_move_best_pos_error[B]`
    - `_move_best_ori_error[B]`
    - `_move_stall_count[B]`
    - `_move_step_scale[B]`
    - `_eef_steps[B]`
  - `move_to_pose()` / `control_eef()` 对每个 env 独立推进并汇总 `signals`
  - IK、mocap、joint control、timeout、stall recovery 都按 env 独立运行

- `MujocoTaskBackend`
  - `reset(env_mask)` 支持局部重置
  - 所有条件判断函数返回 `(B,) bool`
  - randomization 默认逐 env 独立采样
  - `set_interest_objects_and_operations` 至少支持广播到全 batch；最好支持逐 env focus

### 5. 配置、示例、文档一起切到 batch

配置层新增 batch 参数，并统一语义：

- `EnvConfig` 或其 Hydra 包装新增：
  - `batch_size: int`
  - 可选 `seed_offset/seed_stride`
  - viewer 仅允许指定一个观测 env，避免多 viewer 混乱

- README / `docs/custom-backend.md`
  - 所有自定义 backend 示例改成 batch-first
  - 明确单 env 是 `batch_size=1`
  - 明确 observation/action 的首维总是 `env_dim`

- `examples/record_demo.py`
  - 录制低维观测与动作时保存 batched数组
  - metadata 中记录 `batch_size`
  - 回放模式定义清楚是“全 batch 回放”还是“指定 env_index 回放”

- `examples/replay_demo.py`
  - 明确支持：
    - batched action replay
    - 或只回放某个 `env_index`
  - 不再假设 `action.shape == (action_dim,)`

### 6. 明确本期不做的内容

以下内容在方案中明确排除，避免实现时发散：
- 不支持一个 batch 内混合不同 XML / 不同 scene / 不同 operator schema
- 不支持继续保留旧的单环境公共 API
- 不在本期实现 GPU/物理引擎级原生 vectorized MuJoCo
- 不做 batch 内自动动态增减 env 数量，仅支持固定 `batch_size`

## Public API / Type Changes

需要明确变更这些公共接口与类型：

- `SceneBackend`、`OperatorHandler`、`ObjectHandler`：全部从单实例语义改为 batched
- `ControlResult`：单 `signal` 改为 batched `signals`
- `TaskUpdate`：单状态改为逐 env 状态数组/列表
- `ExecutionRecord`：新增 `env_index`
- `ComponentRegistry.register_env/get_env`：
  - 继续保留名字注册机制
  - 但注册对象必须是 batched env
- `UnifiedMujocoEnv.capture_observation()` 的外层 key 不变，value 的 `data` 一律加 `env_dim`

## Test Plan

### Runtime / backend 单元测试
- `batch_size=1` 下，`TaskRunner.reset/update` 行为与旧单环境语义等价
- `batch_size>1` 下，不同 env 可同时处于不同 stage / action / done 状态
- 某个 env 失败不会阻塞其他 env 继续推进
- `env_mask` 局部 reset 只影响指定 env 的状态与 backend 物理状态

### MuJoCo observation / action 测试
- `capture_observation()` 所有 key 的 `data.shape[0] == batch_size`
- color/depth/mask/heatmap/tactile/joint_state/pose/action_pose 都正确补首维
- `step(action)` 要求输入 shape `(B, action_dim)`，错误 shape 明确报错
- `batch_size=1` 时依然返回 `(1, ...)`，不允许偷偷 squeeze

### 条件判断与控制测试
- `is_object_grasped` / `is_operator_grasping` / `is_object_displaced` / `is_operator_contacting` 返回逐 env mask
- 不同 env 上 grasp/contact/displacement 结果可以不同
- IK 失败、超时、stall 恢复都只影响对应 env 的 `signal`

### 集成测试
- 用现有 pick/place demo 改成 `batch_size=2/4` 跑通
- 人为制造一个 env 失败、其他 env 成功，验证 `records` 和 `TaskUpdate`
- `record_demo` 输出的 npz/json 能正确记录 batched 动作和观测
- `replay_demo` 能消费 batched demo 数据

## Assumptions

- batch 内所有 env 使用同一份任务定义、同一套 operators、同一 XML/EnvConfig，仅初始随机化、seed、运行进度不同
- 本期可以接受公共 API 破坏性升级；所有旧调用点同步迁移
- 单环境调用方需要改成显式处理 `env_dim=1`，不保留自动降维兼容
- MuJoCo batch 的实现采用“多个独立 `UnifiedMujocoEnv` 实例聚合”，不是单仿真内原生向量化
