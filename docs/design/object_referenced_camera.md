# 以物体为参考的相机（Object-Referenced Camera）设计

> **状态：提案，尚未实现。**
>
> 本文只描述方案与接入点，未落地任何代码。实现完成后按 AGENTS.md 约定，把稳定用法迁移到
> `docs/task-configuration/task_file_schema.md`（`env.cameras` 字段语义）与
> `docs/task-configuration/randomization.md`（`camera_initial_pose` / `randomization.cameras`），
> 并同步 `docs/mujoco-backend/initialization_randomization.md`，然后删除本文件的"提案"声明。

## 1. 背景与问题

`execution.mode=object_only` 在 Hydra 构造边界（
`auto_atom/execution_config.py::prepare_task_config_for_instantiation`）就完成裁剪：

- 删除 `role: operator` 的场景层（`_is_operator_layer`）；
- 删除 operator 归属相机（`_is_operator_camera`：显式 `role == "operator"`，或 legacy 命名
  `wrist_cam` / `eef_*` / 名称含 `wrist`）；
- 清空 `env.operators` / `task_operators`，并丢弃 operator 名下的
  `task.randomization.entities` 与已删除相机的 `task.randomization.cameras` 条目。

因此"本体消失"时末端相机必然一起消失，这是设计使然而非缺陷。但"以物体为中心"的数据采集与
评测仍需要**一个随物体运动的相机视角**：物体在 object_only 下被运动学搬运（
`MujocoObjectHandler.set_pose` 写 freejoint qpos，或写 `body_pos` / `body_quat`）时，安装在物体
上的相机应随物体一起运动，且相机与物体的相对位姿在运动过程中保持不变。

现有链路无法表达这件事：

- `RandomizationExecutor.apply_camera_randomization` 采样并写回的是**世界系**位姿，`_default_camera_poses` 记录的也是
  世界位姿；
- 相机随机化发生在 **object 采样之前**（见 §6.1），此时物体最终世界位姿尚未确定，无法围绕
  "物体坐标系"采样。

## 2. 目标与非目标

目标：

1. 相机可以声明为"刚性安装在某个物体上"；
2. 该相机在 `object_only` 下不被删除，并随物体运动（运动过程中相机-物体相对位姿恒定）；
3. 不同 reset 之间该**相对位姿**可按现有随机化配置随机化；
4. 复用现有外参/随机化/渲染链路，不引入每步跟踪或额外物理开销。

非目标：

- 不做"相机看向物体/目标"的自动 look-at 求解，相对位姿由配置显式给出；
- 不改变 `object_only` 移除 operator 及其相机的既有语义；
- 不引入每步（per-step / per-update）重设相机位姿的跟踪循环（见 §4 的原生跟随）；
- 不把相机位姿随机化与传感器噪声（`env.cameras[].noise`）混在一起。

## 3. 配置接口

`CameraSpec`（`auto_atom/config/env_config.py`）扩展：

```yaml
env:
  cameras:
    - name: obj_cam
      role: object       # 新增枚举值：scene | operator | object
      parent_frame: cup  # 既是要挂载到的物体（site 或 body），也是外参参考系
      width: 640
      height: 480
      enable_color: true
      enable_depth: true
      is_static: false   # role=object 时只允许 false
      calibration:
        extrinsics:      # 相对 parent_frame 的安装偏移
          position: [0.0, 0.0, 0.25]
          orientation: [0.0, 0.0, 0.0, 1.0]

task:
  randomization:
    cameras:
      obj_cam:           # 相对基线安装偏移的随机化
        x: {relative: [-0.02, 0.02]}
        yaw: {relative: [-0.1, 0.1]}
```

校验规则（fail-closed，在模型构造期报错）：

| 规则 | 说明 |
| --- | --- |
| `role == "object"` 时必须能确定挂载目标 | 显式 `parent_frame`，否则回落到 `cam_bodyid` 自动探测；结果为 world body 时报错（object 相机必须挂在物体上） |
| 挂载目标必须能解析为模型中的 site 或 body | 解析顺序：同名 site → 同名 body → `<name>_gs` body；失败时报错并列出可用候选名 |
| `role == "object"` 时 `is_static` 必须为 `False` | 运动相机不能走 GS 静态背景缓存 |
| `role == "object"` 时禁止出现 `task.camera_initial_pose[<cam>]` | 安装偏移的唯一来源是 `calibration.extrinsics`，见 §3.2 |
| 相机条目仍受 scope 既有校验约束 | `randomization.cameras[<cam>]` 只接受单条 `PoseRandomRange`（无 `regions`），继承 scope `distribution`，拒绝 `constraints` —— 与其它相机完全一致 |

`role` 语义保持不变（"归属/所有权"，决定 `object_only` 裁剪），新增的 `object` 表示"归某个物体
所有"，因此天然不会命中 `_is_operator_camera`（该函数在显式给出 `role` 时只判定
`== "operator"`）。注意 legacy 名称启发式保留：若省略 `role` 而相机名含 `wrist` / `eef_`，
仍会被当成 operator 相机删除，文档需要提示显式写 `role: object`。

### 3.1 复用 `parent_frame`，不新增 `reference_object`

看到 `parent_frame: eef_pose` 容易以为它就是"挂载点"，但实际是两件独立的事：

- **跟随/挂载** 由 `cam_bodyid` 决定，即 XML 里相机声明在哪个 body 之下；
- **外参坐标系** 由 `parent_frame` 决定，即 `calibration.extrinsics` 的数字与
  `_get_camera_extrinsics` 的报告写在哪个坐标系。

今天 operator 相机之所以跟随手臂，是因为它**在 XML 里就声明在 EEF body 下**（`cam_bodyid` = EEF
body）；`parent_frame: eef_pose` 只负责把外参表达成相对 `eef_pose` site。`parent_frame` 本身不会
改变 `cam_bodyid`，因此单独设置 `parent_frame: cup` **不会**让相机跟随 `cup`。

但结论方向是对的：**不需要新增 `reference_object`**。让 `role: object` 给已有的 `parent_frame` 叠加
"挂载目标"语义即可：

| | scene 相机 | operator 相机 | object 相机 |
| --- | --- | --- | --- |
| 挂载（跟随） | XML 决定 | XML 决定（EEF body） | `parent_frame` 解析出的 body，必要时运行期重定父 |
| 外参参考系 | `parent_frame`（可空） | `parent_frame: eef_pose` | 同一个 `parent_frame` |

于是 `role: object` + `parent_frame: cup` 与 `role: operator` + `parent_frame: eef_pose` 完全对称，
并直接复用已有的 `_resolve_frame`（site 优先、body 回落）与 `model.site_bodyid`。

### 3.2 object 相机不接 `camera_initial_pose`

对既有相机，`calibration.extrinsics` 与 `camera_initial_pose` 的作用域本来不同，因此并存：

| 字段 | 作用域 | 时机 |
| --- | --- | --- |
| `calibration.extrinsics` | 模型级，所有 env 相同 | 构造期一次性写入 model，并成为 `reset()` 恢复的基线 |
| `camera_initial_pose` | episode 级，可 per-env（`env_mask` / 参考系解析） | 每次 `reset()` 覆盖，之后成为随机化基线 |

但对 object 相机，两者表达的是**同一件事**：`parent_frame` 就是挂载物体，initial pose 的参考系解析对
该相机没有第二种有意义的取值（`world` 之类更是无意义），所以 initial pose 能写的值集合与
`extrinsics` 完全重合，唯一差别只是应用时机。保留两个真值来源会带来隐式优先级，违背"一个概念一个
字段"，因此：

- **保留 `calibration.extrinsics`**（缺省回落到 XML 的 `cam_pos` / `cam_quat`）作为安装偏移的唯一来源：
  它是模型级静态量，构造期写入 + baseline 重捕获，正好对应 `reset()` 恢复到安装位姿的语义；
- **`role: object` 相机上出现 `task.camera_initial_pose` 条目 → fail-closed 报错**，提示改用
  `calibration.extrinsics`。

反过来只保留 initial pose 也不合适：它不进入 model 基线，每次 reset 都得在"恢复到旧基线后再覆盖"，
而 object 相机需要的恰是"reset 恢复到安装偏移"这一更简单的时序。

## 4. 挂载与跟随（核心机制）

MuJoCo 中 `model.cam_bodyid[cam]` 决定相机的父 body，`model.cam_pos` / `cam_quat` 是**该父 body
局部坐标系**下的偏移，`mj_kinematics` 每帧由父 body 位姿推导 `data.cam_xpos` / `cam_xmat`。

因此方案为：**把 object 相机重定父到 `parent_frame` 解析出的物体 body，让安装偏移成为该 body 的局部偏移。**

- `MujocoBasis` 相机初始化阶段（收集 `self._camera_ids` 之后）执行挂载：
  `model.cam_bodyid[cam] = body_id`，其中 body 由 `parent_frame` 解析（site → `model.site_bodyid[sid]`，
  body → 自身；未配置时回落到原 `cam_bodyid`）；
- 外参换算**零新增代码**：挂载之后再走现有 `_apply_camera_calibrations`，它本来就把"相对
  `parent_frame` 的 extrinsics"换算成 `cam_bodyid` 局部偏移（经 `_frame_pose_world` 求 world 位姿再回写
  局部），挂载后 `parent_frame` 与 `cam_bodyid` 同属物体，换算结果正好是物体局部安装偏移；
- 之后**不需要任何每步代码**：object_only 通过 freejoint qpos 或 `body_pos` / `body_quat` 搬运物体
  时，相机作为子 body 自动跟随（位置与姿态刚性绑定）；物理模式下物体被机械臂操作时同样跟随；
- 挂载完成后，`_camera_parent_frame` 的自动探测会自然把参考系解析为物体 body，与现有
  "相机外参相对 `parent_frame`" 的语义一致。

已用最小实验验证（临时脚本）：`cam_bodyid` 运行期可写；重定父后物体经 freejoint 平移、绕 z 旋转
90°，`data.cam_xpos` 与 `data.cam_xmat` 均同步变化，即原生跟随成立。

需要同步的细节：

1. `_model_cam_pos_baseline` / `_model_cam_quat_baseline` 是 `MujocoBasis._reset_core` 的恢复源，
   必须在**重定父之后**重新捕获（`_apply_camera_calibrations` 末尾已有同样的重捕获模式可复用），
   否则 `reset()` 会把相机恢复成挂载前的旧值。注意该函数在**没有任何 calibration 时会提前返回**，
   因此重捕获不能只依赖它，需对存在 `role: object` 相机的场景单独保证一次。
2. 无 `calibration` 时，XML 里相机的 `cam_pos` / `cam_quat` 在挂载后即被解释为**物体坐标系偏移**
   （与"XML 直接写在物体 body 下"完全一致）；这一点需在文档里写明，避免被误读为世界坐标。
3. 挂载顺序必须在 `_camera_parent_frame` 自动探测之前，否则参考系仍指向旧 body；显式
   `parent_frame` 不受影响，未显式给出时探测结果即物体 body。
4. 挂载目标的解析只在 `role: object` 时叠加物体规则：同名 site → 同名 body → `<name>_gs` body（GS 任务
   里物体 body 实际名为 `<name>_gs`，逻辑名仍是 `<name>`，与 `MujocoObjectHandler` 的规则一致）；
   不要改动 `_resolve_frame` 的通用语义，避免影响既有相机。
5. `is_static=False` 决定 GS 背景每帧重渲染（`gs_mujoco_env.py` 以 `(H, W, is_static)` 分组并缓存）；
   物体相机是运动相机，必须动态，代价需在性能文档中标注。

## 5. 随机化（物体系语义）

object 相机与其它相机共用同一配置面 —— `task.randomization.cameras` 下的单条 `PoseRandomRange`
（多区域 `regions` 与 `constraints` 已被 scope 校验拒绝），并同样继承 scope 的 `distribution`
（位姿流：generator / selector / spacing / candidate pool）。差别只在**采样的坐标系**与**写回的通道**：

- `_record_default_object_camera_offsets[cam]`：记录挂载并解析 `calibration.extrinsics` 之后的局部基线
  `(cam_pos, cam_quat)`，作为随机化锚点；
- `_apply_object_camera_randomization`：以局部基线为中心，按 `canonical_randomization_spec(entry)` 的
  `distribution` 采样偏移，并直接写回 `cam_pos` / `cam_quat`
  （注意 `PoseState.orientation` 为 xyzw，`cam_quat` 为 wxyz，需要顺序转换）。

补充规则：

- 世界系路径（`RandomizationExecutor.apply_camera_randomization` / `_apply_camera_initial_poses`）对 `role: object` 相机
  **整段跳过**；后者还应在构造期直接拒绝配置（§3.2），而不是静默忽略；
- 参考系约束：object 相机只接受 `relative`（= 相对基线安装偏移）；`absolute_world`、
  `absolute_base`、实体名引用一律报错，并提示"object 相机使用物体坐标系局部偏移语义"；
- 采样时机与世界位姿解耦：局部偏移与物体最终世界位姿无关，因此可在与
  `RandomizationExecutor.apply_camera_randomization` 相同的槽位执行，不受 §6.1 顺序限制。这一点必须在代码注释与文档中
  显式区分两类相机随机化；
- 采样使用同一条 `task.seed` 随机流，`TaskRunner.reset()` 的 `initial_poses["_cameras"]` 仍报告采样后的
  **世界位姿**（由 `cam_xpos` 推导），使 object 相机在观测契约上与其它相机一致。

## 6. 与现有子系统的接入点

### 6.1 reset / 随机化顺序（现状）

`RandomizationExecutor.apply_randomization` 的顺序为：operator 采样 → 相机随机化（`randomization.cameras`）
→ `visible_in` 确定性判空 → object 采样。该顺序是"世界系相机是物体可见性上下文"所要求的；
object 相机改为局部偏移后不再依赖该顺序，但该槽位需要按 `role` 分派两类相机随机化，而不是新增第二个槽位。

### 6.2 `visible_in` 约束边界

`RandomizationExecutor._visibility_camera_names` 在 `cameras: all` 时会枚举模型中的全部相机。object 相机挂在目标物体
上，其视锥随候选物体位姿一起移动，"目标是否在自身所挂相机的视锥内"退化为自指问题，无法用现有
判空/重试语义表达。方案：

- `visible_in` 显式列出 object 相机 → 报错（fail-closed）；
- `cameras: all` → 过滤掉 object 相机。

因此 `_preflight_deterministic_visibility_infeasibility` 的"相机固定、区域确定"前提无需改动。

### 6.3 输出与观测链路

- `get_camera_reset_poses` / `get_camera_model` / `get_info` 外参仍返回**世界位姿**（由
  `cam_xpos` / `cam_xmat` 推导）；object 相机的值变为**动态**，消费方（IPC / 评测 / 数据采集）
  不得在 reset 后缓存外参张量，须每帧读取。
- `mask` / `heat_map` / `mask_objects`、`hide_operators_in_camera`、`enabled_sensors` 语义不变。
- `object_only` 边界无需额外改动：`role: object` 相机既不是 operator 层也不命中
  `_is_operator_camera`，其 `randomization.cameras` 条目也不会被
  `_drop_owned_mapping_entries` 丢弃。

### 6.4 批量与 GS

- replicated batch：逐副本 model 各自重定父并写局部偏移，与现有相机路径一致；
- shared-physics GS batch：相机位姿来自共享物理模型，与 object_only 的逐 env 运动学搬运组合需要
  单独验证；若语义不成立，应在构造期给出明确的不支持错误，而不是静默产生错值。

## 7. 分轮实施计划

| 轮次 | 内容 | 验证 |
| --- | --- | --- |
| 1 | `CameraSpec.role` 与校验（挂载目标解析、`is_static`、`camera_initial_pose` 互斥）；`MujocoBasis` 挂载 + baseline 重捕获 | 挂载后 `cam_bodyid` 与局部偏移正确；`reset()` 后保持；缺失物体报错 |
| 2 | 跟随链路贯通：object_only 配置 → prepare → 构造 → `apply_object_pose` 全程验证相对位姿恒定 | 新增/扩展 `tests/test_object_only_execution.py` 等 |
| 3 | 物体系随机化路径（沿用 `randomization.cameras` 与 scope `distribution` 继承）+ 参考系校验 | 相对位姿随 reset 变化；`absolute_world`/实体引用报错 |
| 4 | `visible_in` 边界 + 文档迁移（`task_file_schema.md`、`randomization.md`、`initialization_randomization.md`） | 文档与测试同步 |
| 5 | GS / 批量 / 性能：`is_static` 校验、GS 缓存键、replicated 与 shared-physics 验证、渲染开销基准 | 批量测试 + 基准 |

相关既有测试：`tests/test_camera_frame_ids.py`、`tests/test_hide_operators_in_camera.py`、
`tests/test_mujoco_reset_baseline.py`、`tests/test_initial_pose_orientation.py`、
`tests/test_randomization_plan.py`、`tests/test_randomization_collision.py`（`randomization.cameras`）、
`tests/test_randomization_diagnostics.py`、`tests/test_object_only_execution.py`。

## 8. 风险与备选

| 风险 | 处理 |
| --- | --- |
| `cam_bodyid` 运行期改写在不同 MuJoCo 版本/渲染路径（native、GS、batch）行为不一致 | 轮 1 加断言并在版本矩阵上验证；备选是在物体所在 layer 的 XML 里直接写 `<camera>`（XML 侧挂载，`role: object` 且不写 `parent_frame` 时自动探测即为物体 body），后端只做校验而不重定父 |
| 相机位于物体网格内部导致自遮挡/近平面裁剪 | 可选校验：安装偏移模长小于物体 `get_support_geometry` 半径时告警 |
| object 相机进入 `visible_in` 造成语义歧义 | §6.2 已 fail-closed |
| 动态相机无法复用 GS 静态背景缓存 | 明确性能代价，`is_static` 强制 `False` |
| object_only 逐 env 运动学 + shared physics 组合语义不明 | 明确不支持时报错，不静默出错值 |

## 9. 兼容性

- 不配置 `role: object` 时行为完全不变（`role` 默认 `scene`，`parent_frame` 语义与解析逻辑不变）；
- 新增的是相机的一种**挂载语义**，不改变 `randomization.cameras` 现有的世界系语义；相机随机化配置面仍
  是 `task.randomization.cameras`（顶层 `task.camera_randomization` 已删除）；
- object 相机的安装偏移必须写在 `calibration.extrinsics`（或缺省 XML `cam_pos`/`cam_quat`）；对既有相机
  的 `calibration` / `camera_initial_pose` / `randomization.cameras` 用法全部保持有效。
