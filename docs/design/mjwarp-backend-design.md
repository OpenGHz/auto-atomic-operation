# MJWarp（GPU MuJoCo）后端设计方案

> **状态：提案，尚未实现。**
>
> 本文描述把 [MJWarp](https://github.com/google-deepmind/mujoco_warp) 接为 AAO
> 后端的设计边界。仓库中没有 `auto_atom/backend/mjwarp` 包，`pyproject.toml`
> 也没有 `mjwarp` extra；文中的包路径与配置字段均为**拟议接口**。
> 轮 1（移除机械臂 margin）已实现于 `393b2fd`，轮 2–4 尚未开始。
>
> 本文结论均来自 `scripts/check_mjwarp_compat.py` 的实测与一次 `object_only`
> 真实运行的方法级 tracing，而非阅读推断。复现方式见文末。

## 1. 结论先行

### 1.0 `object_only` 已经在 MJWarp 上跑通

原始问题——`aao-demo --config-name rack_plate_p7_v4_umi_v3
execution.mode=object_only` 所配置的功能能否跑通——答案是**能**，且已实测：

```
backend = build_mjwarp_object_only_backend
after reset: stage=['pick_plate', 'pick_plate'] done=[False False]
updates=8 done=[True True] success=[True True]
records=4
```

两个 world 全部成功，8 次 `update()`、4 条记录（2 stage x 2 world），与原生路径
一致。**观测采集也已跑通**：同一次运行每个 tick 采到 6 路观测
（2 相机 x color/depth/heat_map），形状与 dtype 与原生一致。

轮 1 与轮 2a–2i 已实现。一处需要知道的边界：**RGB 无法与原生逐像素一致**
（光线追踪 vs 光栅化的着色差异），而 mask / heat map 的 IoU 为 1.0、depth 前景
平均误差 0.9 mm——细节见 2i 那一节。

**接缝已经在正确的位置**，移植工作量比按文件行数估算的小得多。

对 `rack_plate_p7_v4_umi_v3` 做 `object_only` 全流程 tracing（instrument
`MujocoBasis` / `UnifiedMujocoEnv` / `BatchedUnifiedMujocoEnv` /
`MujocoTaskBackend` / `MujocoObjectHandler`，跑完整 pick + place），得到 41 个
公开调用点，其中 **32 个已被现有接缝覆盖**：

| 接缝 | 覆盖内容 |
|---|---|
| `SceneBackend`（`contracts.py`） | 生命周期、object handler、位姿应用、诊断 |
| `RandomizationHost`（`randomization_executor.py`） | 位姿读写、相机模型、support geometry、约束评估 |
| `ObjectHandler` / `OperatorHandler` ABC | `get_pose` / `set_pose` |
| `EnvProtocol` 窄化变体 | `capture_observation`、`batch_size` |

`RandomizationHost` 的 docstring 已经写明了这个意图：*"Nothing here knows what a
randomization is, which is what lets a new backend support the whole
randomization layer by implementing pose access alone."*

剩余 9 项中：`apply_operator_initial_states`、`randomization_executor`、
`get_free_joint_id` 是具体 MuJoCo 后端的内部实现（新后端实现自己的等价物）；
`close` / `get_logger` / `refresh_viewer` / `set_camera_noise_seed` 是平凡生命周期。
**真正没有落在任何接缝上的模拟器读取只有 `get_body_pose` 和 `get_site_pose`。**

因此本方案**不新增"模拟器读取协议"**——那会与 `RandomizationHost` 重复。
新后端的任务是满足已有接缝。

## 2. `object_only` 不步进物理，也不隐式渲染

Tracing 的第二个结果推翻了一个常见假设：这条路径上 `UnifiedMujocoEnv.step`
与 `MujocoBasis.update` **一次都没有被调用**。

`execution.object_motion.mode: direct` 直接写入每个 waypoint 位姿，整个
pick + place 只用 **8 次 `update()`** 就完成（两个 stage 的
pre_move / eef / post_move 相位全部走过，`success=True`）。

渲染同样不在执行路径上：只有显式调用 `capture_observation()` 时才发生。加上
`--observe` 后 `_camera_clip_scope` 被调用 32 次（8 次 update x 2 相机 x 2 流）。

**推论**：`object_only` 后端的最小可用形态不需要物理步进内核，只需要
位姿写入 + 前向运动学 + 按需渲染。这是轮 2 选它作为首个目标的原因。

## 3. 与 MuJoCo 的真实差异

### 3.1 接触遍历：没有 `data.ncon`

MJWarp 用一个跨 world 共享的扁平接触池（`naconmax`），按 `contact.worldid`
打标签，`Data` 上没有 per-world 的 `ncon`。现有按 `range(data.ncon)` 遍历接触
的写法（`mujoco_backend.py:1027`、`:2269`）需要改成按 world 过滤的扫描。

**但这不是硬性前置条件**：`get_data_into(out, mjm, d, world_id)` 返回的宿主
`MjData` 带正确的 `ncon`（实测 physical 37、object_only 2）。可以先用它跑通，
再把热路径改成 world 过滤扫描——那是优化，不是前提。

### 3.2 接触力

`mujoco.mj_contactForce`（`mujoco_backend.py:2209`）没有直接对应物。MJWarp 侧
是 `support.contact_force`，或经 `get_data_into` 回读宿主再算。

**未决**：MJWarp 的接触力数值是否足够接近 `mj_contactForce`，能支撑
`_is_target_grasped` 的抓取判定阈值。需要同场景 CPU/GPU 数值对比，属轮 4。

### 3.3 相机 clip range：唯一的结构性缺口

CPU 路径通过 `_camera_clip_scope` 在 `update_scene` 前临时改写
`vis.map.znear/zfar`，让 RGB 与深度使用**不同**的 clip range
（`mujoco_env.py:1900`、`:1936`）。MuJoCo 把 znear/zfar 烘进投影矩阵，所以这个
scope 必须在 `update_scene` 时生效。

MJWarp 的 `RenderContext` 只在**创建时**固定一个 `znear`，且**没有 zfar 字段**。
对策已在轮 2i 实现：按请求的 clip range 分组，每组建一个 `RenderContext`；
远平面改为对 `get_depth` 返回值做后处理钳制。RGB 与深度 clip range 不同的相机会
出现在两个组里，各只启用相关的流。

`rack_plate` 这个配置的两个相机都用模型默认值，因此只需 **1 个** context；但
physical 配置里的 `eef_wrist_cam` 声明了显式的 per-stream range
（`rgb_clip_range_m: [0.001, 50]`、`depth_clip_range_m: [0.001, 5]`），所以分组
逻辑必须是通用的，不能假设只有一个。

> **更正一处早期判断**：`model.vis` 不在 device model 上（`hasattr(m,'vis')` 为
> False）**不构成问题**。`put_model` 本身要求宿主 `mjm`，后端始终持有它，
> 因此 `vis.map.znear * stat.extent` 照常可读，`mj_name2id` / `mj_id2name`
> 也照常工作。缺口只在 RenderContext 的 znear/zfar 结构上。

### 3.4 `njmax` 必须显式给定

physical 模式在默认容量下第一步就报
`nefc overflow - please increase njmax beyond 128`。设为 512 后告警消失。
该值随 world 数放大，应做成配置项而非硬编码常量。

### 3.5 per-world 批量模型（收益所在）

随机化写入的九个字段全部支持
`put_model(mjm, batch_sizes={field: nworld})`，单独与整体测试均通过：

```
body_pos  body_quat  geom_size  geom_rgba
cam_pos   cam_quat   cam_fovy   jnt_range  qpos0
```

这意味着**一个共享模型 + nworld 个 world** 可以取代现在的 N 份独立
`MjModel`。`BatchedUnifiedMujocoEnv`（`mujoco_env.py:2089`）的 N 副本结构随之
收敛，`BatchExecutionAdapter` 的 REPLICATED / SHARED 区分基本不再必要——
这既是 MJWarp 的收益点，也是一次真正的简化。

（`geom_condim` **不可** batch；`geom_margin` / `geom_gap` / `geom_friction` 可以。）

### 3.6 device 侧是 float32，宿主侧是 float64

`cam_pos` / `cam_quat` / `body_pos` / `body_quat` 在 MuJoCo 宿主模型里是
`float64`，在 MJWarp device 模型里是 `float32`（`np.finfo(float32).eps ≈ 1.19e-7`）。

因此一个用 float64 算出来的写入值回读时会带约 `1e-7` 的相对误差。等价性测试的容差
必须按 float32 取（`rtol=1e-6, atol=1e-7`）：更紧的容差要求的是 device 存储**表达
不出来**的精度，卡住的不是转换 bug。

实测中 `set_camera_pose` 的 world→parent-local 写入在 `atol=1e-9` 下失败，最大绝对
偏差 `8.9e-9`（值本身约 0.18，即约 5e-8 相对误差）——纯量化，不是逻辑错误。
`set_static_body_pose` 的同类断言恰好能在 `1e-9` 下通过，但那是该位姿的值刚好可被
float32 精确表示的**运气**，不是转换的性质，因此也一并放宽。

这条同时限定了 3.2 里那个未决问题的判据：接触力的 CPU/GPU 数值对比也只能在 float32
精度上要求一致。

## 4. 不受影响的部分

- **场景组合**：`MjSpec` 编译在宿主侧完成，`put_model` 只消费编译产物。
  `scene_composition/` 与配置声明相机（`role: object` 挂载等）完全复用。
- **分割语义**：MJWarp `get_segmentation` 返回 `(object_id, object_type)`
  逐像素对，背景 `(-1, -1)`，与 `_build_binary_mask` /
  `_build_operation_mask` 消费的格式一致。
- **传感器类型**：FORCE / TORQUE / TOUCH / TACTILE / CONTACT 均支持；
  仅 PLUGIN 类型不支持，本仓库未使用。
- **积分器**：`implicitfast` 支持（README 中排除的是其 midpoint 变体）。
- **IK**：宿主侧 numpy 计算，与后端无关。

## 5. 实现轮次

| 轮 | 内容 | 状态 |
|---|---|---|
| 1 | 移除 P7 机械臂 default class 的非零 `margin` | 已实现 `393b2fd` |
| 2a | `MjWarpSceneState`：frame 读取、free joint 写入 + 等价性测试 | 已实现 |
| 2b | `MjWarpSceneState`：随机化约束读取（相机位姿/fovy/clip、support geometry） | 已实现 |
| 2c | `MjWarpSceneState`：静态 body 放置（world → parent-local）与统一入口 | 已实现 |
| 2d | `MjWarpSceneState`：批量 frame 读取（`PoseState` 形状，单次 readback） | 已实现 |
| 2e | `MjWarpObjectHandler`：满足 `ObjectHandler` 契约（`apply_object_pose` 的落点） | 已实现 |
| 2f | `MjWarpObjectOnlyEnv`：满足 `EnvProtocol` / `PoseConstraintEnvProtocol` | 已实现 |
| 2g | 相机位姿写入（mount / world 系）+ geom / joint frame 读取 | 已实现 |
| 2h | `MjWarpObjectOnlyBackend`：`SceneBackend` + `RandomizationHost` + builder | 已实现 |
| 2i | 观测采集（渲染）：每个 clip range 一个 `RenderContext` | 已实现 |
| 3a | reset 的 forward pass 合并（6 → 2 次 kernel launch，1.84x） | 已实现 |
| 3b | readback 合并 —— **实测不值得做，已放弃**（见 6.3） | 不做 |
| 4 | physical 模式：执行器、IK、接触、触觉 | 未开始 |

**轮 2a**（`auto_atom/basis/mjwarp/state.py`）是后续各轮的读写底座：它持有
device `Model`/`Data`，并以与原生路径相同的单位、dtype 与约定回答 frame 查询。
`tests/test_mjwarp_state.py` 的每个位姿读取都与**原生 `MujocoBasis` 方法本身**
对比，而不是与"原生做了什么"的复述对比，因此约定不一致会在此处失败，而不是
以后表现为一个略微错误的抓取位姿。

其中一条约定值得单独固定：MJWarp 把 MuJoCo 的 wxyz 存在 `wp.quat` 里，而
`wp.quat` 的 Warp 原生序是 xyzw。直接返回原始数组会**错但看起来合理**，
`test_body_orientation_is_xyzw_not_wxyz` 专门钉住这一点。

**轮 2b** 补上 `RandomizationHost` 需要的读取：相机位姿、fovy、默认 clip range
与 support geometry。两点值得记录：

- **不同的矩阵转四元数辅助函数不能互相替换**。原生 `get_camera_model` 用
  `quaternion_from_matrix_3x3`，而 `get_site_pose` 用 `mju_mat2Quat`。适配器
  逐方法对齐各自对应的辅助函数，而不是统一成一个——否则相机位姿会与原生
  路径产生微小但真实的偏差。
- **原生路径的异常类型不一致，适配器照样复现**。frame 读取抛 `ValueError`，
  随机化读取抛 `KeyError`。这里不做统一，以免调用方现有的 except 分支失效；
  `test_randomization_reads_raise_keyerror_like_native` 同时对原生与适配器
  断言，把这个差异钉成有意行为而非疏漏。

`default_clip_range_m()` 返回**米**而非 `vis.map` 的归一化值：device model 有
`stat` 但没有 `vis`，而 MJWarp `RenderContext` 又只在创建时固定单个 znear、
完全没有 zfar（见 3.3），返回米让调用方不必关心任一种表示。

**轮 2c** 补上无 free joint 的 body 放置——`rack_plate` 里 `rack` 与
`plate_stand` 的随机化走的正是这条路径。`body_pos`/`body_quat` 存的是**父 body
局部系**下的值，而静态场景资产经常嵌套在别的 body 之下，因此直接写入请求的世界
位姿会把它放到别处。转换逐字对齐原生 object handler（含
`mju_negQuat`/`mju_mulQuat`），两个后端因此把同一个 body 放在同一处。

测试场景为此专门加了一个嵌套在非单位位姿父级下的 `nested_static`：**直接挂在
worldbody 下的 body 无法暴露这类 bug**，因为它的父系就是单位系，转换退化为恒等。

这一轮同时把 3.5 预测的收益从断言变成了实测：`body_pos` 形状为
`(nworld, nbody, 3)`，掩码写入只移动被选中的 world，且 world 0 与宿主同样写入
的结果逐位一致。**原生路径要表达同一件事需要每个副本一份 `MjModel`**，因为那里
`body_pos` 是模型状态而非批量字段。

`set_object_pose()` 按 body 的实际机制分派到 free joint 或静态路径，与原生
handler 一致，调用方不需要知道自己拿的是哪一种。

### 5.1 写入 API 必须支持"每个 world 一个不同位姿"

轮 2a / 2c 最初把写入方法设计成"一个位姿 + 一个 world 掩码"，**这个形状是错的**，
在接上 object handler 之前已修正。

依据在 `randomization_executor.py:1870`：executor 对每个环境**独立采样**，把各自
的结果写进同一个批量 `PoseState` 的对应槽位
（`action_buffers[label].pose.position[env_index] = ...`），最后用一次带掩码的
`set_target_pose` 应用。原生 handler 相应地按 `pose.position[env_index]` 逐环境
取值。`PoseState` 本身就是 `(B, 3)` / `(B, 4)`。

因此写入方法现在接受两种形状（`_pose_rows`）：

- `(3,)` / `(4,)` 或 `(1, 3)` / `(1, 4)`：单个位姿，广播到所有被写入的 world；
- `(nworld, 3)` / `(nworld, 4)`：第 `w` 行写入 world `w`。

索引按**绝对 world 下标**，而不是"掩码中的第几个"——与原生
`pose.position[env_index]` 的语义一致。无论哪种形状都只做一次 readback、一次
assign、一次 `forward()`，开销不随 world 数增长。

`test_free_joint_accepts_one_pose_per_world` 等 4 项测试对修正前的实现会失败
（报 `cannot reshape array of size 9 into shape (3,)`），因此它们确实钉住了这个
形状约定，而不是碰巧通过。

**轮 2d** 补上读取侧的对称形状：`get_body_pose_batch` / `get_site_pose_batch`
直接返回 `(nworld, 3)` / `(nworld, 4)`，即 `PoseState` 想要的形状，且**只做一次
device readback**而不是每个 world 一次——运行时每个控制 tick 都要读物体位姿，
所以这条路径的开销不能随 world 数增长。原生批量 env 的 `get_body_pose` 是把各
副本的结果 stack 起来，语义相同。

（`mju_mat2Quat` 是标量接口，因此 site 的矩阵转四元数仍按 world 循环，但 readback
只有一次。）

**轮 2e**（`auto_atom/backend/mjwarp/handlers.py`）是 `apply_object_pose` 的落点，
也就是 `execution.mode: object_only` 的**全部**搬运机制。handler 刻意做得很薄：
名称解析、free joint / 静态路径分派、world → parent-local 转换、批量 readback 都
已经在 `MjWarpSceneState` 里，这一层只负责把运行时的批量 `PoseState` 与
`env_mask` 桥接过去。

与原生 handler 的一处差异值得记录：原生每个副本一份 `MjModel`/`MjData` 并逐个
循环，其 `_stateful_pose_indices` 的存在是为了把带掩码的写入**收敛到单个物理
行**——那是 Gaussian-Splatting 共享物理批次的需求。MJWarp 是一个 device 模型
带真正的 per-world 状态，没有别名可收敛，因此 `env_mask` 直接映射为
`world_mask`，不需要对应的收敛逻辑。

掩码形状的报错信息与原生逐字一致（`env_mask must have shape (2,)`）：
`tests/test_backend_contracts.py` 钉住了原生这条信息，因此
`test_mask_rejection_message_matches_the_native_handler` 用**同一个正则**同时断言
两个后端，让依赖这条信息的调用方跨后端都能继续工作。

**轮 2f**（`auto_atom/basis/mjwarp/env.py`）是 backend 交给运行时的 `get_env()`
对象：结构上满足 `EnvProtocol` 与 `PoseConstraintEnvProtocol`，后者正是随机化层
接受/拒绝候选摆放所需要的能力。

`tests/test_mjwarp_env.py` 不用合成场景，而是直接跑**真实的**
`rack_plate_p7_v4_umi_v3` + `object_only`，因为那才是移植要复现的东西：分层 MJCF、
在 Hydra 边界被剥离的 operator 层、配置声明的物体挂载相机、嵌套静态场景。所有值
都与原生 `UnifiedMujocoEnv` 自己的答案对比，实测全部一致：

- 4 个 support geometry（含配置注释里写的 `object` 半径 0.123857 m）；
- 2 个相机模型的 fovy / near / far / 位姿 / 分辨率；
- 3 个 body 位姿、2 个 site 位姿；
- `visible_in: all` 的见证相机集合——`plate_cam` 骑在待观测物体上，两个后端都把
  它排除在外。

约束算术复用 backend-neutral 的 `RandomizationConstraintEvaluator`，env 只提供
两个 simulator-specific 读取，与原生 basis 相同。**这正是 `visible_in` /
`separated` 语义跨后端一致而非各自重写的原因。**

reset 必须恢复 `body_pos` / `body_quat` / `cam_pos` / `cam_quat` 四个基线：
`reset_data` 只清动态状态，而静态放置与相机随机化写的正是这四个字段，否则上一次
reset 的结构位姿会静默变成下一次 reset 的起点。原生 basis 出于同样原因恢复同样
四个数组。

一处**测试前提被实测推翻**值得记录：最初写了"配置声明了模型里不存在的相机时应当
报错"，但它不会报——`load_composed_scene` 会**创建**配置声明而场景未编写的相机
（见 `37d1cd0`），所以改名后的相机其实存在。该守卫只在 `host_model=` 注入路径上
可达，测试因此改为走那条路径。

渲染在轮 2f 时**尚未实现**，留到轮 2i 单独做：MJWarp `RenderContext` 创建时固定
单个 znear 且没有 zfar，而原生路径按输出流切换 clip range（见 3.3）。这个切分是
有意的——`object_only` 执行路径本身不渲染（采集是显式调用），所以后端可以先跑通
执行，再补采集。

**轮 2g** 补上相机位姿**写入**与两级更深的 frame 读取，两者都是 backend 的前置：

- `set_camera_mount_pose` 直接写 `cam_pos` / `cam_quat`；`set_camera_pose` 走
  world → parent-local 转换，但锚定的是**相机自己的父级**（`cam_bodyid`）而不是
  body 的父级，所以挂在运动 body 上的相机也能正确放置。测试用挂载相机
  （`holder_cam`，父级位姿非单位）验证，否则转换退化为恒等。
- `get_geom_pose` / `get_joint_frame_pose` 是 `get_element_pose` 的第三、四级
  回退。joint 一级读 `xanchor` 而非父 body 原点：测试里 `swing_hinge` 的锚点与其
  body 原点刻意不同，并断言两者确实有差异，否则这个测试不会咬。

这一轮还修正了轮 2f 的一处**遗漏**：原生 `get_element_pose` 解析 site → body →
geom → joint 共四级，而轮 2f 只实现了前两级。`motion_goal.py` 与 `runtime.py` 都
通过它解析命名 frame（`controlled_frame`、门/闩的 arc 支点），因此缺失的两级是真实
缺口，现已补齐并与原生同序。

**轮 2h**（`auto_atom/backend/mjwarp/backend.py`）合上了 `object_only` 后端：
`MjWarpObjectOnlyBackend` 同时满足 `SceneBackend` 与 `RandomizationHost`，
`build_mjwarp_object_only_backend` 是任务文件 `backend:` 字段的构造入口。

实测（`tests/test_mjwarp_backend.py`，走真实 `rack_plate_p7_v4_umi_v3` 任务文件）：

- `dt_per_update = 0.08` = timestep 0.002 x 40 substep（`sim_freq/update_freq` =
  1200/30），与原生推导一致；
- handler 收集到 `object` / `rack` / `rack_target` / `plate_stand` 四个——`rack`
  与 `plate_stand` 从不作为 stage 目标出现，但随机化要移动它们，所以必须有 handler；
- 刚性归属判定给出 place stage 需要的语义：`object_site` **属于** `object`，
  `rack_target_site` **不属于**（它骑在 rack 上）；
- **reset 跑通了带约束的逐 world 随机化**：两个 world 各自采到不同的位姿，且都落在
  配置的 proposal box 内。采样成功本身意味着 `evaluate_pose_constraints` 接受了它，
  因此 `visible_in: rack_camera_front, margin_px: 8` 这条约束路径也被执行到了。
  **这条只有在轮 2a/2c 的写入形状被修正（5.1）之后才可能成立**——否则两个 world 会
  拿到同一个广播位姿。
- 固定 seed 下两次独立构造的采样结果一致，即 run 可复现。

**operator 查询一律抛错，而不是返回空值**。`object_only` 没有实体 operator，运行时
会替换自己的 `ObjectOnlyOperatorHandler`；一个"未抓取"的空状态看起来完全像一个
合法答案，会静默掩盖配置错误的 physical run。唯一例外是 `get_operator_contacts`
返回 `None`——契约本身把 `None` 定义为"该后端不支持接触观测"，而且它是运行时
机会性调用的诊断路径，不参与控制决策。

一处构造期的坑：后端持有的必须是 `ResolvedRandomizationConfig`（由
`from_scope_config` 包装）而非原始 scope config——`.applies` 只在前者上，它把总开关
与"是否为空"合成一个判断。直接用 `config.randomization` 会在 reset 时
`AttributeError`。

`MjWarpObjectOnlyEnv` 现在也会用 `config.name` 注册到 `ComponentRegistry`，与原生
env 同样的接线方式；否则 builder 的 `get_env` 找不到 Hydra 已经实例化的那个 env。

**轮 2i**（`auto_atom/basis/mjwarp/render.py` + `env.capture_observation`）补上观测
采集。至此 `object_only` 的数据采集链路也跑通了：任务跑完 8 次 update，两个 world
全部成功，同时**每个 tick 采到 6 路观测**，形状 `(2, 352, 640[, 3|5])` 与原生一致。

观测契约与原生逐项对齐（`tests/test_mjwarp_render.py`，同配置双后端对比）：
key 集合、shape、dtype 全部相同，`t` 逐 world 取自 `data.time`（它带 world 轴，
所以每个 world 报自己的时钟，而不是广播 world 0）。

### 关于精度：mask 几乎完全一致，RGB 无法一致

这是本轮最重要的结论，**RGB 的差异不是缺陷，而是渲染器本身不同**：

| 流 | 与原生的一致度 |
|---|---|
| `mask/image_raw` | IoU **1.0000**（`plate_cam`）／**0.9966**（`rack_camera_front`，1182 px 中差 4 px） |
| `mask/heat_map` | IoU **1.0000**，逐像素 **100%** |
| depth（前景） | 平均误差 **0.9 mm**，p99 **1.05 cm** |
| depth（背景掩码） | 与原生一致 **99.9991%** |
| `color/image_raw` | 中位差 **1/255**，但均值亮度 44.4 vs 31.7 |

mask 与 heat map 来自 segmentation pass（几何），所以能一致；RGB 来自着色，
MJWarp 光线追踪、MuJoCo 用 OpenGL 光栅化，着色模型不同。实测尝试过 sRGB/linear
两个方向的色彩空间变换，**都让误差变大**（raw 14.3/255 最优，sRGB 82.3，linear
27.7），因此这不是一个可以纠正的编码差异。几何是对的：相关系数 0.72，结构性
不匹配（>96/255）只占 0.07%。

**结论**：训练目标（mask/heat map）和几何量（depth）可以跨后端互换；RGB 不能，
需要 RGB 逐像素一致的场景不要混用两个后端采的数据。测试因此断言契约、depth 与
mask，而把 RGB 差异记录为实测容差而非当作 bug。

### 三处 MJWarp 渲染语义与 MuJoCo 不同

1. **depth 永远是归一化的**。`get_depth` 算的是
   `clamp(value / depth_scale, 0, 1)`，传 `1.0` 会把超过 1 m 的距离全部截断。取米
   必须把远平面作为 scale 传进去再乘回来——这样做把与原生的差距从 **3.16 m 降到
   2.3 mm**（100 m 量程）。
2. **打空的射线返回 0.0，不是远平面**。原生返回远平面。不处理的话，空背景会读成
   "距离 0"，比任何实物都近——正好相反。
3. **图像朝向与 segmentation 编码不需要转换**。实测 segmentation 质心与原生一致到
   小数点后 4 位（无翻转），编码本身已是 MuJoCo 的 `(object_id, object_type)`、
   背景 `(-1, -1)`。

另外修正了轮 2f 的一个真实 bug：相机 id 解析原本无条件执行，但原生（与
`EnvConfig.camera_elements` 的文档）都把它**门控在 `DataType.CAMERA` 上**——没有
camera sensor 时相机声明是惰性的，既不会被materialize 进场景也不该被解析。
无条件解析会拒绝一个完全合法的无相机配置。

**轮 1** 的动机：MJWarp 门禁拒绝 mesh/box CCD pair 上的非零 margin。编译后场景
中 59 个非零 margin geom 里只有 7 个可碰撞（`link1..link7`），其余 52 个是
`contype=conaffinity=0` 的触觉单元，永不产生接触。1200 步 CPU 对比显示被操作
物体落点差异 6e-13 m，故 CPU 行为等价，无需后端特化的模型改写。

**轮 2** 选 `object_only` 作为首个目标：它无需执行器、IK、接触查询与触觉
（`execution.mode: object_only` 在 Hydra 边界剥离 operator 层，`enabled_sensors`
收敛为仅 camera），是验证新后端接缝的最小面。

**轮 4** 最大：后端里 20 处 per-env 串行循环需要向量化成沿 world 轴的操作。

## 6. 未决问题

1. **依赖声明**。`pyproject.toml` 现有
   `mujoco = ["mujoco >=3.12.0,<3.13.0", ...]`；mujoco_warp 3.13.0 要求
   `mujoco>=3.12.0`。实测 mujoco 3.12.0 + mujoco_warp 3.13.0 可共存，因此可新增
   `mjwarp` extra 而不动现有 pin。后端模块必须**惰性导入** `mujoco_warp`，
   保证未安装时项目照常可用（`check_mjwarp_compat.py` 已是这个写法）。
2. **接触力数值一致性**（见 3.2）。轮 4 的前置验证。
3. ~~**单 world 是否值得**~~ —— **已实测，结论与预期相反**，见 6.1。

### 6.1 实测吞吐：渲染即使在单 world 也更快，reset 曾是瓶颈

`rack_plate` + `object_only`，每项取 5 次中位数（秒）：

| batch | 后端 | reset | render | render/world |
|---|---|---|---|---|
| 1 | warp | 0.0126 | **0.0070** | 0.0070 |
| 1 | native | 0.0001 | 0.0258 | 0.0258 |
| 2 | warp | 0.0116 | **0.0137** | 0.0069 |
| 2 | native | 0.0002 | 0.0564 | 0.0282 |
| 4 | warp | 0.0119 | **0.0263** | 0.0066 |
| 4 | native | 0.0002 | 0.1106 | 0.0277 |

**这推翻了本文早先两次给出的判断**（"单 world 很可能比 CPU 慢"）：渲染在
`batch_size=1` 就已经快约 3.7 倍，且 per-world 渲染成本基本恒定（6.6–7.0 ms），
而原生按 world 线性增长（每个约 27–28 ms）。

真正的瓶颈在 reset —— 原本比原生慢约 60 倍。剖析显示 **74% 的 reset 时间花在
6 次 `forward()` kernel launch 上**（5 个实体各写各刷），而不是我先前猜测的
numpy readback（那只占 11 ms）。轮 3a 因此做的是合并 forward pass，不是合并
readback。

> ⚠️ 本机 8 GB 显存。跑吞吐对比时**不要**一次性拉到 8/16 world：曾因此让机器卡死。
> 验证 3a 只需 `batch_size=2`。

### 6.2 轮 3a：`deferred_forward` 把 6 次 pass 降到 2 次

`MjWarpSceneState.deferred_forward()` 是一个引用计数的上下文管理器：块内的
`forward()` 只标记"待刷新"，退出时统一跑一次。backend 的 `reset` 用它包住
`apply_randomization`。

实测 **6 → 2 次 kernel launch，reset 24.0 ms vs 44.2 ms（1.84x）**，且四个物体
与两个相机的位姿**逐位一致**。

为什么是 2 次而不是 1 次：**这正是安全机制在生效**。`set_static_body_pose` 与
`set_camera_pose` 都要读父级的**当前**世界位姿来做 world → parent-local 转换。
`rack_plate` 把 `plate_cam` 挂在被随机化的 `object` 上，所以相机写入前必须先让
`object` 的那次 pass 落地——`_require_fresh_parent` 检测到父级 body 在本块内被写过
就提前 flush。

**这不是"对这个配置刚好安全"，而是通用正确的**：写入时记录 dirty body，读父级
位姿时按需 flush，因此即使某个任务同时随机化一个 body 和它的父级也不会出错。
`test_a_child_write_flushes_its_freshly_moved_parent` 与
`test_camera_write_flushes_a_freshly_moved_mount` 钉住了这一点——**把该守卫改成
no-op 后两者立刻失败**（相机落到 `[0.2, 0.35, 0.95]` 而非 `[0.35, 0.05, 0.55]`，
约 0.5 m 偏差），确认它确实承重而非装饰。

### 6.3 readback 合并：实测不值得做

3a 之后重新剖析一次 reset（`batch_size=2`，逐数组包装 `.numpy()` / `.assign()` 计时）：

| 项 | 时间 | 占比 | 调用次数 |
|---|---|---|---|
| readback（`.numpy()`） | 1.67 ms | **6.6%** | 41 |
| assign（`.assign()`） | 0.33 ms | 1.3% | 14 |
| 其余 | 23.27 ms | 92.1% | — |
| **reset 总计** | 25.28 ms | | |

调用次数看着多（`data.xpos` 9 次、`data.xquat` 9 次），但**每个调用点在自己那次调用
里只读一次**——9 次来自 9 个不同的方法调用，不存在同一方法内的重复读取可以消除。
把它们合并成"一次批量 readback"需要在 state 里引入一层缓存并管理失效，而天花板
只有 6.6%。

**因此 3b 放弃**：这是在 8% 的上限上做有失效风险的重构，不划算。真正的 23 ms 在
randomization 自身的采样与约束评估（纯 CPU、与后端无关），要优化应该去那一层，
而不是 MJWarp 适配层。

> 这也修正了我最初对 3b 的判断——早先说"reset 的 numpy round-trip 值得合并、约
> 11 ms"。那个 11 ms 是 3a 之前把 forward pass 也算进"其余"得到的数字；3a 之后
> 真正的 readback 开销只有 1.67 ms。

## 7. 本机环境注意事项

`nvidia-smi` 报驱动 12.4，warp 1.17.0 编译使用 CUDA Toolkit 12.9。清空 warp
kernel 缓存后，CCD kernel 首次编译会失败：

```
Warp CUDA error 500: named symbol not found
KeyError: 'ccd_kernel_builder__locals__ccd_kernel_..._smem_bytes'
```

恢复缓存后即通过。**与模型内容无关**——不含机械臂的 `object_only` 模型同样触发，
故不是 margin 或场景问题，而是工具链版本偏移。会干扰 CI 判读，排查时优先
确认缓存状态。

### 7.1 受限 runner 的内存上限不够加载 CUDA 模块

`scripts/run_tests_safe.py` 默认 `--memory-max-mb 6144`。在该上限下
`tests/test_mjwarp_state.py` 有 3 项失败，报的正是上面那个 CCD kernel
`KeyError`——**但这不是工具链偏移，而是 cgroup 上限**：同一批测试直接用
`python -m pytest` 跑 12 项全通过（2 s），提高上限后用同一个受限 runner 也
全通过（3.8 s）。

CUDA 模块加载很吃宿主内存，触到 cgroup 硬上限时 warp 报出的症状与符号查找
失败难以区分。**涉及 MJWarp 的测试必须提高上限**：

```bash
python scripts/run_tests_safe.py --test-targets tests/test_mjwarp_state.py \
    --max-concurrency=1 --memory-high-mb 10240 --memory-max-mb 14336
```

排查顺序因此是：先确认内存上限，再确认 warp kernel 缓存，最后才怀疑模型。

## 8. 复现方式

```bash
# 1) 项目环境：走真实 Hydra + 场景组合路径导出各执行模式的模型
python scripts/check_mjwarp_compat.py export --config-name rack_plate_p7_v4_umi_v3

# 2) 装有 mujoco_warp 的环境：加载 .mjb + manifest 逐项检查
/path/to/mjwarp-venv/bin/python scripts/check_mjwarp_compat.py probe \
    outputs/mjwarp-compat --njmax 512

# 或一次跑完（export 在当前环境，probe 交给 --probe-python）
python scripts/check_mjwarp_compat.py --probe-python /path/to/mjwarp-venv/bin/python \
    --njmax 512
```

`.mjb` 与写它的 MuJoCo 版本绑定，manifest 记录该版本，probe 阶段版本不符会
直接报错而非隐性失败。新增任务配置或改动机器人 XML 后可直接重跑做回归。

## 9. 范围外发现

### 9.1 `rack_plate` 连续 reset 到第 4 次必然失败（既存 bug，与本移植无关）

**两个后端表现完全一致**：原生与 MJWarp 都是 3/8 次 reset 成功，第 4 次抛
`RandomizationFailureError: Randomization for 'rack' failed after 100 attempts:
rack:collides:plate_stand; minimum_clearance=-0.0025`。因此这不是移植引入的问题，
而是共享随机化层 + 配置本身的几何问题。

根因是**范围可以吃掉全部余量**：

| 量 | 值 |
|---|---|
| authored 中心距 | 0.40241 m |
| 需要的分离距离（`r_rack + r_stand`，两者都是 `collision_radius: -1` 自动求得） | 0.36936 m |
| authored 余量 | **+0.03305 m** |
| `rack.x` ± / `plate_stand.x` ± | 0.05 / 0.06 → 最坏靠近 **0.11 m** |
| 最坏情况中心距 | 0.29241 m → **无解** |

所以部分采样组合在几何上不可能满足，而 `failure.mode` 默认 `ERROR`（fail-closed）
会直接让这一轮失败，而不是换一组重采。

`rounds` 默认是 1，所以日常 demo 一直没暴露；但数据采集要跑很多轮，必然踩到。
`collision_radius: -1` 目前只有 `rack_plate` 在用，所以影响面限于这一个配置。

**已按用户决定：只记录，暂不改配置。** 可选修法（未实施）：把两者 x 范围收紧到
最坏情况仍可满足（约 ±0.015），保持 fail-closed 语义不变；或改
`failure.mode: best_effort` 让无解时退化为最优候选——但那会静默接受轻微穿模。

### 9.2 触觉传感器的 NaN 风险

`assets/xmls/sensors/tactile_sensor.xml` 声明的触觉单元是**可碰撞的**
（`contype=conaffinity=1`），却带 `friction="0 0 0"` 与 `condim="4"`，MJWarp 报
NaN 风险警告。这是真实问题（CPU 侧同样存在），使用者是
`panda_robotiq.xml` / `robotiq_assets.xml`。这些单元是球体，不触发 CCD margin
门禁，因此**与本移植无关**，应由该传感器接触模型的负责人单独处理。

（对照：`xf9600_tactile_sensor.xml` 的同名警告是伪报——那批单元
`contype=conaffinity=0`，永不产生接触。MJWarp 扫描 geom 参数时未先判可碰撞性。）
