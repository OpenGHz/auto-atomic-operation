# Praxis 与 AAO 开门任务对比分析

## 结论先行

Praxis 和 AAO 都可以驱动 P7 类七轴机械臂完成“抓住把手并开门”，但它们不是同一个层级的实现：

- **Praxis** 是面向 Project 209 数据生产的专用闭环系统。它把自由空间规划、接触后的机构控制、力/碰撞安全、事件顺序验证和带 provenance 的录制放在同一条开门契约里。
- **AAO** 是通用的声明式 Stage/Primitive 执行框架。当前配置用一个 `push` Stage 串起两个接近位姿、夹爪闭合和两个绝对圆弧动作，最终沿通用 `DISPLACED` 条件判定成功。
- 两边的门、夹爪和资产身份并不相同，因此不能把 Praxis 的控制参数直接复制到 AAO。最显著的语义差异是目标角：Praxis 当前标准是 **15° (`0.2617993878 rad`)**，AAO 配置实际写的是 **`0.20 rad`（约 11.46°）**，注释中的“约 15°”不能替代实际配置值。
- 如果目标是让 AAO 的结果可与 Praxis 的开门结果比较，首要迁移项应是**显式的门机构 outcome contract**（解锁、clearance、目标角、保持、回弹、松爪和回撤），而不是先搬运某组 waypoint 或 hybrid controller 数值。

## 1. 范围、入口与证据边界

### 1.1 选取的项目入口

Praxis 仓库同时保留了早期的 `open_room_door_push.yaml` 和若干历史 30°/45° 配置。根据 Praxis 当前 `docs/STATUS.md` 与 `docs/PLAN.md`，本次把下面的入口作为“当前 Praxis 开门项目”：

| 项目 | 本次采用的入口 | 选择理由 |
| --- | --- | --- |
| Praxis | `third_party/praxis/configs/unidoor_lever_right_push_p7_staged.yaml` | 当前唯一活跃的右铰链、P7 staged、15° 单阶段工作包；它通过 `extends`/`compose` 继承 `open_room_door_push_housework_mode_2.yaml` 的基础设施。 |
| AAO | `aao_configs/open_door_p7_v3_umi_v3.yaml` | 用户指定的当前任务配置；通过 `basis_p7_v3_umi_v3` 组合 P7 V3 + UMI gripper V3。 |

Praxis 的 `open_room_door_push.yaml` 在本文中作为基础实现背景引用；已经标记为 retired 的 30°、45°以及 literal 83°/0.20 m 前驱不作为当前动态基线，也不把历史成功数字拼入本次比较。

### 1.2 分析快照

本分析基于以下工作树快照（2026-08-25）：

| 仓库 | commit | 工作树 |
| --- | --- | --- |
| 外层 AAO | `84c447ab0af7c70cab36f4e94bc1af4cd8a817c0` (`test: make pytest collection deterministic`) | clean；`main` 相对 `origin/main` ahead 8 |
| `third_party/praxis` | `2fa0c76246b629a32211e5da193c938f37247b32` (`feat: adopt staged 15-degree door opening`) | clean；与其 `origin/main` 一致 |

`third_party/` 和 `outputs/` 被外层仓库忽略，Praxis 不是外层 Git 子模块。因此文档同时记录两个 commit，不能只用外层 commit 代表两边版本。

### 1.3 AAO 配置解析证据

按项目要求运行：

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/aao-info \
  open_door_p7_v3_umi_v3 --no-progress
```

解析结果为：

```text
Runnable tasks (1):

open_door_p7_v3_umi_v3  (scene: open_door)
  operators:  arm (p7_arm_v3_with_umi_gripper_v3)
  objects:    handle_lever_body
    note: env.mask_objects ['handle_lever_body'] differs from stage objects ['handle_body_phys']
  operations: push
  workflow:
    1. push handle_body_phys [grasp_and_open]
```

这里的 `note` 是配置事实，不足以单独证明任务错误，但它意味着视觉 mask 的对象名和 Stage 的物理对象名需要在比较或数据管线中明确区分。

### 1.4 证据边界

- Praxis 的当前动态证据以其 `docs/STATUS.md` 中的 H15-1/H15-2/H15-3 账本为准；该基线**并非成功基线**。
- 本文没有重新运行 AAO。工作区已有的 `outputs/2026-08-25/10-59-13/summary.json` 和 `11-00-05/summary.json` 各记录一次 `2/2` Stage 成功（206 次 update），但它们只记录 AAO 通用 Stage 成功，没有门角、事件顺序或 settle 字段；因此只能视为 framework-level smoke evidence，不能与 Praxis 的 15° outcome 结果等价。
- 下文明确区分“当前实现事实”和“建议迁移项”；建议不代表已经修改代码。

## 2. 两套方案的总体定位

| 维度 | Praxis 当前 staged 开门 | AAO `open_door_p7_v3_umi_v3` | 直接含义 |
| --- | --- | --- | --- |
| 产品目标 | Project 209 / UniDoorManip Interior 开门数据工厂 | 一个可组合的 `open_door` MuJoCo 场景 | Praxis 面向产品族和数据闭环，AAO 面向任务表达和复用 |
| 任务身份 | `RIGHT-HINGE-PUSH-P7-STAGED-V1`；当前 pilot 为 `D001-H003`、configuration seed 51、runtime seed 209 | 单一 `task_name`；默认 `task.seed=42`，当前 EEF 随机范围为零 | Praxis 把组合、变体、seed 和 provenance 纳入身份；AAO 当前没有同等的产品身份层 |
| 门资产 | UniDoor 源 URDF + 编译后的 `door_push.xml`，可按 55×47 目录替换门/把手/碰撞 | `assets/xmls/scenes/open_door/demo.xml` 中手工定义的盒状门、把手和铰链 | 几何、质量、阻尼和接触不能直接横向套用 |
| 机器人/夹爪 | `kj6310_v3/robot.xml` + V3 自适应夹爪，执行器 `claw_joint` | `p7_arm_v3_with_umi_gripper_v3.xml` + UMI gripper V3，执行器 `eef_claw_joint` 和 finger-distance mapper | 都是七轴 P7 类方案，但 TCP、夹爪行程、控制增益和接触几何不同 |
| 控制闭环 | 状态机 + cuRobo 接触前规划 + Airbot mechanism reference + hybrid 接触控制 | Task/Stage timeline + 每个控制 tick 的 Cartesian/IK primitive | Praxis 的机构反馈和安全闭环更深；AAO 的动作表达更轻量 |
| 终态产物 | 通过 outcome verifier 的 episode、JSONL、physics sidecar、视觉重放和 provenance | `TaskUpdate`/`summary.json` 以及可选相机观测 | 两边的“成功”字段不是同一个数据契约 |

## 3. 执行流程对照

### 3.1 Praxis：显式阶段状态机

当前 P7 staged 运行链为：

```text
RESET(HOME)
  -> PLAN
  -> APPROACH
  -> GRASP
  -> CLOSE_GRIPPER
  -> UNLATCH
  -> SWING
  -> GRIPPER_RELEASE
  -> RETREAT
  -> SETTLE
  -> COMPLETE / FAILED
```

关键转移是由观测反馈触发，而不是固定时长直接跳过：

- `UNLATCH` 命令把手目标 `0.45 rad`，实测把手角达到 `0.45 × 0.80 = 0.36 rad` 才进入 `SWING`。
- `SWING` 只有在实测门角达到当前目标 `0.2617993878 rad` 后才进入 `GRIPPER_RELEASE`。
- `RETREAT` 之后必须回到配置拥有的七轴 home，并在 `SETTLE` 中保持门角；任一数值、关节限位或硬碰撞违规都可立即失败。

证据：`third_party/praxis/src/praxis/state_machine.py:48-187`、`third_party/praxis/docs/STATUS.md:25-37`。

### 3.2 AAO：一个 Stage 内的声明式 primitive 序列

当前配置的有效动作顺序是：

```text
Stage grasp_and_open (operation=push, object=handle_body_phys)
  -> pre_move[0]：object_world，相对把手前方约 14 cm
  -> pre_move[1]：object_world，相对把手前方约 7 cm
  -> eef.close=true（由当前 PUSH 配置显式插入）
  -> post_move[0]：handle_hinge 绝对圆弧，目标 0.45 rad
  -> post_move[1]：door_hinge 绝对圆弧，目标 0.20 rad
  -> Stage postcondition：DISPLACED
```

这里有两个容易混淆的点：

1. AAO 通用 `PUSH` 的默认成功条件仍然是 `DISPLACED`；`eef.close=true` 是此 Stage 的额外配置，不会把 `PUSH` 自动变成 Praxis 那种 `GRASP → UNLATCH → SWING` 语义。
2. 两个圆弧用了 `absolute: true`。它们各自是一个 pose primitive，但每个控制 tick 会读取当前关节角，把剩余角度限制在 `max_step=0.15` 内，再交给 `per_step_ik` 追踪。因此“一个 primitive”不等于“一次物理步”。

证据：`aao_configs/open_door_p7_v3_umi_v3.yaml:49-91`、`auto_atom/runtime.py:896-991,1882-1943`。

## 4. 关键维度详细对照

### 4.1 目标、机构与资产

| 维度 | Praxis | AAO | 风险/解释 |
| --- | --- | --- | --- |
| 当前门角目标 | `0.2617993877991494 rad`，严格为 15° | `0.20 rad`，约 11.46°；配置注释写“约 15°”但数值不是 15° | 不能用两边的成功率直接比较；若要对齐，先决定统一的 outcome identity |
| 把手目标 | `0.45 rad`；以实测 `0.36 rad` 作为释放反馈 | `handle_hinge` 圆弧目标 `0.45 rad` | 命令目标相同不代表反馈阈值相同 |
| 门锁模型 | `door_latch` actuator；stiffness 300、damping 8、max force 40；clearance 0.03 rad | active-by-default 的 door joint equality；`DoorLatchCallback` 按把手行程 `unlock=0.12/relock=0.08 rad` 和 `0.02 rad` 捕获区切换约束 | AAO 现在是不可用有限推力越过的硬锁；Praxis 是有力限幅的机构模型，两者仍不能直接比较破锁力 |
| 门几何 | UniDoor 编译资产；当前产品空间为 55 doors × 47 handles，pilot 组合为 `D001-H003` | 固定手工场景；门板半宽约 0.410 m、半高约 0.805 m，质量约 6 kg | AAO 的 `demo.xml` 是一个具体模板，不是 Praxis 的产品族 |
| 铰链/坐标 | 当前右铰链 push；active profile 将门轴设为 `[0,0,-1]`、把手轴设为 `[0,1,0]`，再应用 wheelchair/world frame 和右侧镜像 variant | `demo.xml` 的 `door_hinge` 轴为 `[0,0,1]`，`handle_hinge` 轴为 `[0,1,0]`，任务圆弧另指定轴向 | 同名 joint 不意味着正方向和安装坐标一致 |

### 4.2 机器人、规划与控制

| 维度 | Praxis | AAO |
| --- | --- | --- |
| 自由空间规划 | cuRobo 负责接触前规划；配置含 64 个候选、goalset 1、3 条 trajectory candidates，并要求验证抓取点 | 没有独立的全局路径规划；两个固定 `pre_move` 由 `MujocoOperatorHandler.move_to_pose` 逐 tick 求 IK |
| 接触后控制 | Airbot mechanism reference + Cartesian servo；`SWING/SETTLE` 使用 hybrid 位置/力偏置、速度目标和接触安全 backoff | Cartesian pose 目标 + P7 V3 analytical IK (`per_step_ik`)；圆弧目标通过当前关节角转换为 EEF pose，没有开门专用 hybrid force loop |
| 抓取反馈 | 左右指、掌部、把手 hub 接触力；最小指力 2 N，抓取丢失有 grace steps | 通用 EEF handler 可检查左右指接触和 lateral 条件；当前配置未声明 Praxis 式最小力、hub owner 或碰撞力门限 |
| 安全停止 | 数值、关节限位、环境碰撞、非法指接触、hub contact policy 均可进入失败路径；push 当前为 `immediate_failure` | 通用 handler 有 IK timeout、接触/夹爪检查，但当前 Stage 没有把门机构安全事件和力阈值纳入成功契约 |
| 时间步 | 配置 `sim_dt=0.001`、`control_decimation=10`、solver iterations=30，控制频率约 100 Hz | `sim_freq=1000`、`update_freq=100`，运行时把 timestep 设为 0.001、每次 update 做 10 个物理子步；源 `demo.xml` 仍声明 `iterations=5` | 名义控制 cadence 接近，但 solver/接触参数和场景 XML 不同，不能据此认为动力学等价 |

证据：Praxis `configs/open_room_door_push.yaml:21-27,87-107,165-217`、`configs/profiles/tasks/unidoor_lever_55x47_right_push_p7_staged.yaml:18-24`、`src/praxis/backends/unilab.py:836-1057,1059-1471`；AAO `aao_configs/open_door_p7_v3_umi_v3.yaml:29-104`、`auto_atom/basis/mjc/mujoco_basis.py:349-369`。

### 4.3 成功判定与失败语义

这是两套实现最重要的差异。

#### Praxis 的 outcome contract

Praxis 的 `EpisodeEventAccumulator` 记录四个事件：

```text
latch release < clearance <= target < settle_end
```

其中：

- `release`：把手达到 `0.36 rad`，且门仍在 `0.03 rad` clearance 以内；
- `clearance`：release 之后门角达到 `0.03 rad`；
- `target`：clearance 之后门角达到 `0.2617993878 rad`；
- `settle_end`：在 `SETTLE` 中保持目标所需的 100 steps。

最终 verifier 还检查：事件完整且有序、门没有回弹到目标的 95% 以下、终态门角达到目标、计划松爪已观察到、P7 七轴回 home、终态 phase 为 `COMPLETE`，并且数值/关节/碰撞失败标志为空。见 `third_party/praxis/src/praxis/verification.py:175-213,232-345,372-385`。

#### AAO 当前的 stage postcondition

`Operation.PUSH` 映射到 `OperationConstraint.DISPLACED`。默认实现比较 Stage 开始和结束时目标物体的位置，位移大于 `0.01 m` 即满足条件：

- Stage object 是 `handle_body_phys`，不是 `door_hinge`；
- 条件在所有 primitive 完成后的 Stage 末尾检查，不是“把手移动 1 cm 就立即结束”；
- 没有显式门角目标、解锁→clearance 顺序、settle hold、rebound gate、计划松爪或回 home 检查。

因此 AAO 的“成功”更准确地说是“配置动作执行完且把手物体发生了足够位移”，不是“门按指定机构事件完整打开”。证据：`auto_atom/framework.py:38-43,94-100`、`auto_atom/stage_execution.py:470-491,802-843`、`auto_atom/runtime.py:624-641`。

### 4.4 传感器、记录与闭环 policy

| 维度 | Praxis | AAO |
| --- | --- | --- |
| 相机契约 | HEAD + right wrist RGB 是产品记录的固定角色，默认 640×352、30 FPS；depth/mask 可显式打开 | 当前有效配置为 `eef_wrist_cam` + `env2_cam`，全局默认 color/depth/heat-map 开启、mask 关闭；尺寸也是 640×352 |
| 观测重点 | 关节、TCP/把手/门角速度、左右指/掌/hub 力、碰撞和非法接触、home error | 通用 pose/tactile/joint/camera 观测；当前任务 YAML 没有专用门角 outcome schema |
| 记录格式 | JSONL episode header/step/result + binary physics sidecar + variant/provenance；失败 episode 也可记录 | `summary.json` 和通用执行记录；相机/热图由 backend 全局开关控制，任务配置不声明 Praxis 式 provenance 完整性 |
| policy 接口 | closed-loop policy 返回 joint/cartesian action chunks，runner 处理 horizon、replan、repeat 和 task state | `PolicyEvaluator`/`aao-eval` 以 AAO Stage/Primitive 为执行表面，默认要求每个 control tick 的外部 action/feedback |

Praxis 的相机默认来自 `configs/open_room_door_push.yaml:219-270`；AAO 的默认媒体开关来自 `aao_configs/common_vars.yaml:7-12` 及本任务配置。两边即使使用相同分辨率，也不是同一套相机标定或数据 schema。

### 4.5 Praxis 门/把手资产拼装与可视化

Praxis 的“拼装”分成**离线资产组合**和**运行时场景编译**两层；AAO 现在也提供了一个轻量的 manifest/OBJ 运行时装配路径，但两者的职责不同：

1. `build-unidoor-lever-catalog` 读取门体和把手源包，分别做尺寸归一化、坐标变换、碰撞几何检查，并写出门/把手的 component manifest、标准化 mesh 和哈希。之后按门 × 把手的笛卡尔积物化组合，例如 `D001-H003`；组合 XML 还会记录门角、把手角、抓取 site 等语义对象。
2. Praxis staged/closed-loop 运行时主要从带机器人和相机的 base scene 创建 `MjSpec`，再由 `UnilabLeverTaskModelFactory._patch_door()` 与 `_patch_handle()` 替换选中变体的 mesh、碰撞代理、安装位、轴和传感器，最后 `spec.compile()` 得到本次仿真的物理模型。
3. AAO 将这类拼装建模为通用 `SceneConfig` 的 `asset_assembly` layer：`package`、`adapter@version`、role selection、placement、namespace 和 integrity policy 是统一字段。当前 `unidoor.lever_door@1` adapter 读取 `product_space.json`、两个 component manifest 和标准化 OBJ，生成 namespaced 门框/门板/把手 MJCF contribution，再由 host composer 与 MJCF 机器人 layer 合并。AAO 运行时**不读取** `combinations/`，因此可以按组件 manifest 直接换门或把手；预生成组合 XML 仅作为回归 oracle。

组合 XML 的核心层级可以简化为：

```text
worldbody
└── unidoor_assembly (`position` / `orientation` 作用于此 root body)
    ├── door_frame
    └── door_panel
        ├── door_hinge
        ├── door_panel_visual / door_panel_collision
        ├── door_handle (门板的子 body，随门板旋转)
        │   ├── handle_hinge (把手下压关节)
        │   ├── door_handle_visual
        │   ├── handle_lever_collision
        │   └── handle_grasp_center
        └── door_lock (可选，仅 visual)
```

其中，门 component manifest 的 `geometry.handle_position_m` 决定 `door_handle` 在门板上的安装位置；把手 component manifest 的 `geometry.grasp_offset_m` 决定 `handle_grasp_center` site 的局部位置。visual mesh 与物理 collision/proxy 分开维护；带锁把手的 `lock.obj` 可以作为静态 visual 加入，但不应被解释成可驱动的锁舌动力学。右铰链变体还会同步镜像 mesh、碰撞补充物和 provenance，不能只翻转一个 joint 轴。canonical package template 显式记录 `door_hinge=[0,0,-1]`、`handle_hinge=[0,1,0]` 及其 axis frame；运行时不从旧 `handedness` 猜测机构轴，离线迁移器负责校验、转换并记录推导依据。配置中的 `xyzw` 环境四元数会转换为 MJCF 的 `wxyz`。

AAO 装配器的最小配置字段如下：

| 字段 | 作用 |
| --- | --- |
| `package` | canonical scene asset package descriptor（可指向外部、可迁移 payload） |
| `adapter` / `selection` | `unidoor.lever_door@1` 与 `door`/`handle` role selection |
| `placement` / `namespace` | 将 canonical 门环境放入 host scene，并生成稳定 namespaced IDs |
| `verify_hashes` | 校验选中 manifest、OBJ 和门碰撞 supplement 的 SHA-256 |

`verify_hashes=true` 时，路径必须留在 catalog root 内；ACD sidecar 只保留为
provenance，首版 AAO 采用门的 box supplement 与把手 AABB，不把 Praxis 的
MotrixSim ACD 凸分解隐式改写成 AAO 碰撞体。

**拼装后的可视化是有的，但有不同层级：**

| 能力 | 是否有 | 产物与边界 |
| --- | --- | --- |
| 组合 XML 的 MuJoCo 加载 | 有 | 可直接检查门/把手层级、joint、site、actuator 和可选 lock；在 AAO 中它是回归 oracle，主要用于结构/加载对照，不是运行时必需输入，也不包含完整机器人交互证据。 |
| AAO manifest/OBJ 运行时装配 | 有 | `SceneConfig` 的 `asset_assembly` layer 通过 `unidoor.lever_door@1` adapter 从组件 manifest/OBJ 生成 namespaced contribution，再与 MJCF layers 统一合并；不读取 `combinations/`。 |
| 全量静态资产审核图 | 有 | `render-unidoor-collision-audit` 生成 55 个门体 × 47 个把手/门锁的图片、`index.html` 和 manifest；用于人工审核碰撞与安装，不改变任务几何。 |
| MuJoCo/EGL 3D 代表性预览 | 有 | `render-unidoor-collision-audit-3d-preview` 查看代表性门体、把手和碰撞代理，仍不是一次动态开门 episode。 |
| 完整 variant visual replay | 有 | `UnilabLeverTaskModelFactory.compile_visual` 与 `MujocoVisualEpisodeRenderer._compile_scene_variant` 会把选中的门、把手、安装位和锁 visual 编译进带相机的完整场景，再重放 episode。需要 header 中的 `scene_variant`/provenance，且 renderer 会校验 mesh hash；缺少这些信息时只能可靠地渲染 base scene。 |
| 动态开门视频/视觉 episode | 有 | `collect` 在录制后可自动调用 `praxis.runtime.render_visual_episodes`，闭环或 replay 也可生成 head/wrist 视觉产物；视频生成通过不等于 outcome verifier 判定开门成功。 |

常用入口例如：

```bash
uv run --no-sync praxis build-unidoor-lever-catalog
uv run --no-sync praxis render-unidoor-collision-audit
MUJOCO_GL=egl uv run --no-sync praxis render-unidoor-collision-audit-3d-preview
uv run --no-sync praxis collect --num-envs 1
```

因此，如果只是想确认“某个门和把手能否拼在一起”，可以先用 AAO 的 manifest/OBJ
装配器做本地编译，再用 catalog 组合 XML 做结构回归，并查看静态/3D audit；如果
要看机器人、相机和该变体在动作中的完整画面，应使用带 `scene_variant` provenance
的 visual replay。任一 MJCF compile/load 通过都不能替代第 4.3 节的门角、事件顺序、
settle 和安全 outcome 验证，也不能自动解决 `third_party/` 被 `.gitignore` 忽略、
源包许可和对外再分发边界。

## 5. 当前动态验证状态

| 方案 | 当前证据 | 能否称为“成功开门” |
| --- | --- | --- |
| Praxis P7 staged 15° | H15-1 preflight PASS；attempt-001 的 H15-2 FAIL，原因 `swing_timeout`。最大门角 `0.237799451 rad = 13.6249°`，终态 `12.7797°`，没有进入 `GRIPPER_RELEASE/RETREAT/SETTLE`。hard safety violation steps 为 0，但 target/settle 缺失。 | 不能；这是当前官方的失败动态基线。 |
| AAO `open_door_p7_v3_umi_v3` | 本次未重新运行。已有两个本地 demo summary 报告 `2/2` Stage success、206 updates；摘要没有门角或 outcome event。 | 只能称为 AAO framework-level Stage smoke success，不能据此证明达到 Praxis 的 15°门角或机构 outcome。 |
| AAO `open_door_unidoor_p7_v3_umi_v3` | 55×47 共 2585 个组合均可编译。当前 `PICK → PULL → PUSH` 配置通过 batch-1 headless 回归：`PULL` 在把手圆弧前确认目标抓取、圆弧后保持握持，`PUSH` 在把手达到解锁行程且门锁解除后才执行门圆弧，终态门角不低于 `0.18 rad`。拆分前 `PICK → PUSH` 配置的 batch-2 `2/2`、405 ticks 结果不作为当前流程的性能基线。 | 已证明当前 D001/H003、0.20-rad 任务目标下的三阶段条件与抓取—解锁—开门顺序成立；仍未覆盖 Praxis 的 15°、release/retreat/home/settle 和安全 outcome contract，不能视为与 Praxis 同口径成功。 |

Praxis 数字证据见 `third_party/praxis/docs/STATUS.md:66-98`。历史 M1/30°成功记录已经被 D-038 标记为 retired，不应当用来掩盖当前 H15-2 失败。

## 6. 可迁移能力分级（建议，不是已实现改动）

| 优先级 | 建议 | 迁移边界 |
| --- | --- | --- |
| P0 | 为 AAO 增加可选的 `DoorOutcomeContract`/door-specific postcondition：门角目标、handle release、clearance、事件顺序、settle hold、rebound、松爪和 home。 | 保留通用 `PUSH → DISPLACED` 作为其他物体的默认语义；只对声明了机构 contract 的 Stage 启用更严格判定。 |
| P0 | 扩展 backend 的观测 seam，至少提供命名 joint angle/velocity、接触力、碰撞/非法接触和 terminal home 状态，并记录事件时间线。 | 先做可测试的观测/验证接口，不要把 Praxis 的某组 kp、力阈值硬编码到所有 AAO 场景。 |
| P1 | 把“接触前规划”和“接触后机构控制”分成可替换的 adapter。 | AAO 仍可使用当前 per-step IK；只有需要闭环机构任务的 backend/task 才挂载 planner + hybrid controller。 |
| P1 | 引入资产/变体 identity（door、handle、hinge side、placement、config/runtime seed、source hash）。 | 只有 AAO 要承担数据工厂或多资产筛选时才值得引入 55×47 catalog；单一 demo 不应复制整套产品管线。 |
| P1 | 对齐对象命名：明确 `handle_body_phys`（Stage/位移判定）与 `handle_lever_body`（mask/视觉对象）是否有意分离。 | 若是有意分离，写入记录 schema；若非有意，应在配置中统一，避免 `aao-info` 的 note 演变成数据标注错误。 |
| P2 | 统一实验指标和录制格式，至少同时保存 command target、measured door/handle angle、phase、failure reason。 | 先统一证据字段，再讨论 RGB-D、压缩格式或 LeRobot 转换；媒体分辨率相同不代表数据可比。 |

### 不建议直接迁移的内容

- 不要把 Praxis 的 `0.261799 rad`、`0.36 rad`、`kp=300` 或 hybrid wrench 参数直接替换进 AAO；两边的门质量、铰链、夹爪 TCP 和坐标系不同。
- 不要把 AAO 的 `DISPLACED` 结果重命名为 Praxis 的“15°成功”；应先通过 outcome contract 重新验证。
- 不要把当前 Praxis H15-2 失败解释成 Praxis 方案已经优于 AAO；它只说明 Praxis 的验证标准更严格，并且当前 staged 动力学仍未通过。

## 7. 结论

从架构定位看，Praxis 的优势是**机构任务的闭环深度和证据完整性**：它知道何时解锁、何时越过 clearance、何时达到目标、何时松爪和何时可以把 episode 交付给数据管线。AAO 的优势是**任务表达简洁、Stage/Keypoint/Primitive 可复用、backend 可替换**。

两边目前最不能直接比较的不是“谁的 waypoint 更好”，而是“成功”所代表的事实不同。将 AAO 用于同类开门研究时，推荐顺序是：先补齐可选的门机构 outcome contract 和观测记录，再评估是否需要 planner/hybrid controller，最后才调目标角、姿态和力参数。这样可以保留 AAO 的通用任务模型，同时让开门结果具备可验证、可复现的语义。

## 8. 主要证据索引

- AAO 任务与基础配置：`aao_configs/open_door_p7_v3_umi_v3.yaml`、`aao_configs/basis_p7_v3_umi_v3.yaml`、`aao_configs/common_vars.yaml`。
- AAO 执行与判定：`auto_atom/runtime.py`、`auto_atom/stage_execution.py`、`auto_atom/framework.py`、`auto_atom/callbacks/door_latch.py`、`auto_atom/backend/mjc/mujoco_backend.py`。
- Praxis 当前入口与状态：`third_party/praxis/configs/unidoor_lever_right_push_p7_staged.yaml`、`third_party/praxis/configs/profiles/tasks/unidoor_lever_55x47_right_push_p7_staged.yaml`、`third_party/praxis/docs/STATUS.md`、`third_party/praxis/docs/PLAN.md`。
- Praxis 执行与验证：`third_party/praxis/src/praxis/state_machine.py`、`third_party/praxis/src/praxis/verification.py`、`third_party/praxis/src/praxis/backends/unilab.py`、`third_party/praxis/src/praxis/recording.py`。
