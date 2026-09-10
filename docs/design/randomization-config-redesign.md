# Randomization 配置重构方案（提案，尚未实现）

> 本文是**提案**，描述计划中的配置结构，尚未实现。实现完成后按
> AGENTS.md 约定，将稳定用法迁移回 `docs/task-configuration/randomization.md`
> 并删除本文的"提案"声明。

## 实现状态（2026-09-08 更新）

**已实现并提交**：
- 改名切片（提交 `d836161`）：`distribution.min_distance`→`spacing`、
  `separated.min_distance`→`clearance`、删除未接线的
  `RandomizationGroupDistributionConfig.clearance`。
- 容器化切片（提交 `5626df0`）：`task.randomization` → `RandomizationScopeConfig`
  {`distribution`、`constraints`、`entities`、`cameras`}；`resolve_randomization_scope` 三级回落
  （目标显式 > scope 默认 > 内置）；`failure` 归入 `constraints.failure`；
  `strategy` 归入 `separated.strategy`，删除 `AutoAtomConfig.randomization_strategy`；
  采用**新语义**（默认 `failure=error`，去掉 legacy best_effort 特殊化）。同步后端工厂、
  runtime / execution_config / data_replay、24 个 `aao_configs` 的 `entities` 包装、
  相关测试与 `randomization.md` / `task_file_schema.md` / `ik_control.md` 文档。
- **限制（已实现为单一有效策略）**：后端仍按单策略分发；scope/实体声明的
  `separated.strategy` 若不一致会报错。**逐组件混合策略**（不同组件各用不同策略）
  需另行改造后端分发，未实现。
- **auto collision_radius 扩展到 operator**（提交 `a91b12f`）：base=root body 自身
  footprint、eef=eef site body 子树（详见文末"已记录的后续增强"）。
- **visible_in 确定性判空**（提交 `add77d2`）：范围盒与相机视锥整体不相交时
  `mode=error` 直接以 `attempts=0` 失败并诊断，不再烧 `max_attempts`。
- **相机随机化并入容器**（提交 `61bafb5`）：`task.camera_randomization` →
  `task.randomization.cameras`，删除顶层字段（无兼容入口）。
- **相机继承 `distribution`**：`cameras` 条目与实体一样继承 scope 的
  `distribution`；`constraints` 为实体专属，相机显式声明会被拒绝。
  `resolve_randomization_scope` 改为返回 `ResolvedRandomizationScope`
  （`entities` / `cameras` / `strategy`），三级回落对两类目标共用同一实现。

## 动机与目标

1. **消除 `min_distance` 歧义**：同一个名字在代码里承担两种不同语义，需拆成
   语义明确的字段。
2. **去掉逐实体样板**：`distribution` / `constraints` / `failure` 需要"全局默认 +
   逐实体覆盖"，避免在多个实体下重复写同一段配置。
3. **`visible_in` 从"重试式 rejection"转为"范围约束"**：优先要求用户把范围
   保守缩进可见区内部，而不是靠 `max_attempts` 搜索。
4. **`strategy` / `failure` 支持逐约束、逐实体独立**，而不是全局唯一一个。

## 三种"距离/尺寸"语义（贯穿全文，务必区分）

| 概念 | 字段 | 语义 | 是否随尺寸 |
|------|------|------|-----------|
| 采样间距 | `distribution.spacing`（原 `min_distance`） | 单实体采样流覆盖/间距（Poisson 半径、跨 reset 历史） | 通常逐实体设 |
| 表面净空 | `separated.clearance`（原 `min_distance`） | 实体表面间要求的净空 | 否（尺寸由半径项承担） |
| 尺寸代理 | `collision_radius` | 包围球近似半径 | 是 |

碰撞判定的实际最小中心距：

```text
||p_i - p_j|| >= collision_radius_i + collision_radius_j + separated.clearance
```

`distribution.spacing` **不参与**碰撞判定，也不保证多物体场景不碰撞（采样流是
单实体的，生成阶段不感知其它物体）。

## 核心分层：生成 vs 可行性（正交两轴）

| 层 | 字段 | 是否始终生效 |
|----|------|------------|
| **生成**（怎么提候选） | `distribution` | 可选，默认 iid |
| **可行性**（候选如何成为可行布局） | `constraints` | **始终生效**（内置半径分离承载） |

`constraints` 进一步分两类：

| 子类 | 内容 | 说明 |
|------|------|------|
| 可选硬要求 | `visible_in` / `separated` | 叠加在机制之上 |
| 接受机制 | `separated.strategy`、`constraints.failure` | **始终生效** |

### 为什么 `strategy` / `failure` 不游离在外

RSA / joint_rejection（`strategy`）与 `max_attempts`（`failure`）作用于那条
**始终存在的半径分离**（随机参与者只要 `collision_radius > 0` 就两两避碰），
不依赖用户是否显式写 `separated` / `visible_in`。因此它们必须"始终有个家"——
通过在**全局默认 `constraints`** 中始终携带一份带默认 `strategy` 与 `failure`
的 `separated` 来实现。这样"没写 separated 时 RSA 仍在跑"就有了确定的承载点。

## 最终配置结构

```yaml
task:
  randomization:
    # —— 全局默认（entities 未写对应字段时回落）——
    distribution:               # 生成
      generator: iid
      selector: first_feasible
      candidate_count: 1
      spacing: 0.0              # 原 distribution.min_distance

    constraints:                # 可行性（始终存在，承载 always-on 机制）
      failure:                  # 可行性搜索预算耗尽时怎么办
        mode: error             # error | best_effort
        max_attempts: 100
      separated:                # 分离要求（默认存在 → strategy 有固定家）
        strategy: rsa           # rsa | joint_rejection（原 randomization_strategy）
        scope: randomized       # randomized | scene
        clearance: 0.0          # 原 separated.min_distance
      visible_in: null          # 可选，默认无

    # —— 逐实体（物体 或 operator 的 base/eef）——
    entities:
      plate:
        reference: absolute_world
        x: [0.15, 0.55]
        collision_radius: 0.11
        # 省略 distribution / constraints → 全部回落全局默认

      source_block:
        reference: absolute_world
        x: [0.25, 0.55]
        y: [-0.15, 0.15]
        collision_radius: 0.04
        distribution:           # 实体级覆盖（整块替换 distribution）
          generator: poisson_disk
          spacing: 0.04
        constraints:            # 实体级覆盖（按子字段合并）
          separated:
            strategy: joint_rejection   # 本实体组件用 joint 策略
            clearance: 0.02
          visible_in:
            cameras: all
            geometry: bounding_sphere
            margin_px: 8

      arm:                      # operator 嵌套不变
        base: { ... }
        eef: { ... }
```

## 覆盖规则（三级回落，在 spec 规范化一处实现）

1. **实体级**：实体写了 `distribution` / `constraints` 的哪个子字段，覆盖全局对应
   项；没写的继承全局。
2. **全局默认**：`randomization.distribution` / `randomization.constraints`。
3. **内置默认**：两者都未配时回落代码内置默认（等同现 `RandomizationDistributionConfig()`
   与 `RandomizationSeparationConfig()`）。

`constraints` 的实体覆盖是**按 `separated` / `visible_in` / `failure` 子字段合并**，
避免"覆盖一个丢掉另一个"。`distribution` 与 `constraints.failure` 为整块替换。

## `strategy` 逐组件独立的影响

`separated.strategy`（rsa / joint_rejection）支持逐实体/逐组件独立。当前后端
只支持**单一全局策略**（`randomization_strategy` 默认 RSA，joint 走另一条
`_sample_component_for_env` 路径）。放开为逐组件策略后：

- 组件分发需携带"该组件用哪种策略"；
- 同一 reset 里可能混合 rsa 与 joint_rejection 组件；
- `AutoAtomConfig.randomization_strategy` 旧字段删除，语义迁移到全局默认
  `constraints.separated.strategy`。

## `visible_in` 语义：保守内缩（首选），非重试式 rejection

- `visible_in` 的可满足集**只依赖固定相机 + 实体自身几何**，不依赖其它随机成员，
  因此是**固定几何集合**，可在"选范围"时就前置约束。
- **首选做法（保守内缩）**：用户把 `proposal` / `regions` 缩进可见区内部，使
  采样天然可见，`visible_in` 永不触发拒绝。
- 交集为空 = **确定性不可行**：正确响应是放宽范围 / 调相机，**不该烧
  `max_attempts`**。
- 由于 `proposal` 只支持轴对齐盒、可见区一般非轴对齐，只能保守内缩进一个
  安全盒（配置纪律，非自动计算）。

### 失败/重试语义分层（文档须讲清）

| 约束 | 可重试？ | 失败本质 |
|------|---------|---------|
| collision / `separated` | 是（可行性依赖其它成员落点，随机条件性） | 随机搜索未命中 |
| `visible_in` | 交集非空=搜索；**交集为空=无用** | 范围内确定性不可行 |

## 相机随机化

`camera_randomization` 已并入作用域容器，作为 `randomization` 下与 `entities`
同级的 `cameras` 映射（不再有顶层 `task.camera_randomization`）。

归并的仅是**位置**，不是**语义**：

- `distribution` 描述"位姿流怎么生成"（generator / selector / spacing / 候选池），
  相机也在采样位姿流，因此 `cameras` 条目与实体一样**继承** scope 的
  `distribution`（含三级回落）。
- `constraints`（`visible_in` / `separated` / `failure`）是实体专属语义，相机没有
  碰撞几何、不能作为可见性约束目标、也没有可行性重试循环，因此相机**不继承**
  `constraints`；在相机条目上显式声明 `constraints` 会被**拒绝**（fail-closed），
  而不是静默忽略。

```yaml
task:
  randomization:
    distribution: {...}     # 实体与相机共同继承
    constraints: {...}      # 仅实体继承
    entities: {...}
    cameras: {...}          # 继承 distribution，拒绝 constraints
```

## 已记录的后续增强（进度追踪）

- ~~`visible_in` 交集为空 → 判定不可行并给诊断、不进入 attempt 循环~~（**已完成**，
  提交 `add77d2`）：`MujocoTaskBackend._camera_frustum_disjoint_box` 做保守
  AABB×视锥判空；`_preflight_deterministic_visibility_infeasibility` 在对象采样前、
  `mode=error` 时对可见性范围内置判定为空的 region 直接抛 `RandomizationFailureError`
  （`attempts=0`）并写入诊断。**覆盖边界**：仅支持 `absolute_world` / `relative`
  （相对固定默认位姿）的世界轴对齐位置盒；实体跟踪引用、`segmentation`、
  `scene` 分离等仍走原 attempt 循环。
- ~~`collision_radius: auto`（对象 + operator base/eef）~~（**已完成**：
  - 对象级（提交 `ba6fd80`）：`collision_radius<0` 标记 auto + `collision_margin`；
    后端 `_resolve_collision_radius` 走 `get_support_geometry` 并永久缓存。`0` 仍为
    豁免、`>0` 显式。
  - operator base/eef（提交 `a91b12f`）：语义为 base→operator **root body 自身
    几何**（底座/支架 footprint，非整臂子树）；eef→**eef site 所在 body 的子树**
    （含夹爪/手指，随开合构型变化）。`UnifiedMujocoEnv.get_operator_support_geometry`
    按当前构型测量；radius 缓存 key 含 kind，`_apply_randomization` 每 episode 丢弃
    operator 项（构型相关），对象项保持长期缓存。mocap 夹爪的 base（root 无几何）
    解析为 0（视为豁免）。）

## 实施轮次（每轮独立提交）

- R1：本文档（钉死方案）。
- R2：schema 重构 —— 容器（`distribution` / `constraints` / `entities` / `cameras`）、
  `min_distance`→`spacing`、`separated.min_distance`→`clearance`、
  `separated.strategy`、删 `randomization_strategy`、三级回落。
  （注：原 `framework.py` 已拆分为 `auto_atom/config/` 子包，schema 类按域
  分布于 `config/randomization.py` / `config/motion.py` / `config/task.py` 等。）
- R3：后端读取与逐组件策略分发（`mujoco_backend.py` / `mujoco_basis.py`；
  后端无关契约在 `auto_atom/contracts.py`）。
- R4：重写 `docs/task-configuration/randomization.md`。
- R5：测试迁移与新增（结构解析、改名键、覆盖回落、visible_in 空交集诊断、
  逐组件策略），用受限 runner 跑通。
- R6：`camera_randomization` → `randomization.cameras`（顶层字段删除，无兼容
  入口）。仅搬移容器位置。
- R7：相机条目继承 scope 的 `distribution`（位姿流属性），不继承 `constraints`
  （实体专属），显式声明 `constraints` 的相机条目直接报错。
  `resolve_randomization_scope` 改为返回 `ResolvedRandomizationScope`
  （`entities` / `cameras` / `strategy`）。

## 后续：把执行语义收敛到共享层（进行中）

目标：新增 backend 只需提供接口（位姿读写、命名 frame 解析、相机模型、支撑几何），
不再重写随机化语义。分轮推进，每轮独立提交、独立验证：

- **R-A（已完成，提交 `50126e9`）**：约束评估（视锥投影 + 分离间距 +
  per-episode 缓存）提到 `RandomizationConstraintEvaluator`，
  `MujocoBasis` 只留 `get_camera_model` / `get_support_geometry`。
- **R-B1（已完成，提交 `e287a7d`）**：纯采样/碰撞原语提到共享层
  （`sample_pose_for_env`、`sample_pose_batch`、`select_randomization_region`、
  `find_collision_participant`、`history_clearance`、
  `distribution_uses_space_filling_history`、半径/祖先解析）。
  `CollisionParticipant` 由后端私有提升为共享类型。
- **R-B2a（已完成）**：配置校验与确定性 `visible_in` 预检提到共享层
  （`validate_randomization_configuration`、
  `validate_pose_randomization_spec`、`reference_ancestors`、
  `camera_frustum_disjoint_box`、`object_region_world_box`、
  `find_visibility_infeasibility`）。
- **R-B2b-i（已完成，提交 `308b298`）**：联合拒绝可行性循环 +
  跨 reset 覆盖历史 + 尝试预算 + `maximin` 组选取 + fail-closed/best-effort
  决策搬到 `RandomizationExecutor`。
- **R-B2b-ii（已完成，提交 `fb02505`）**：硬球 RSA 放置循环搬到同一执行器。
- **R-B2b-iii（已完成）**：编排与批量写回 —— `apply_randomization`
  （operator → camera → 可见性预检 → object 顺序策略）与 `sample_component`
  （逐 env 采样并入 batch 形状缓冲）搬到执行器；后端 `_apply_randomization`
  退化为一行委托。

约束（各轮均遵守）：RNG 消费顺序、`sample_index` 公式（`reset_index*1009 +
env_index`、`+ attempt*17 + sum(ord(c))`）、`env_mask` 与 per-component 批量
写回语义逐字保留，否则 reset 复现性会漂。

### 迁移后的职责边界

后端（`RandomizationHost` 协议，16 个成员）只剩：
`randomization_rng` / `randomization_reset_index` /
`batch_size` / `object_names` / `operator_names` / `randomization_plan()` /
`action_dependencies()` / `template_pose(label)` / `sample_target(...)` /
`evaluate_constraints(...)` / `record_randomization_diagnostics(...)` /
`begin_randomization_episode()` / `apply_action(...)` /
`apply_camera_randomization(...)` / `run_visibility_preflight(...)`，
外加 `get_camera_model` / `get_support_geometry` / 命名 frame 解析。

生效的放置策略随 `RandomizationPlan.strategy` 传递（策略是编译结果的一部分：
plan 同时携带它的后果 —— component 分组与生成的 joint-placement groups），
因此执行器不再向 backend 索要策略，backend 也不需要暴露随机化策略。

共享层拥有：计划编译、候选生成（IID/Sobol/Poisson + 覆盖历史）、
区域选择与权重、逐轴 reference 语义、碰撞拒绝判定、约束评估（视锥/分离 +
per-episode 缓存）、配置校验、确定性可见性预检、两条重试循环、排序策略、
批量写回缓冲、失败策略。
