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
拟议对策：按不同 znear 建多个 `RenderContext`；远平面改为对
`get_depth` 返回值做后处理钳制。

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
| 2c | `object_only` MJWarp env/backend，满足现有接缝，不新增协议 | 未开始 |
| 3 | per-world 批量模型取代 N 份 `MjModel` | 未开始 |
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
3. **单 world 是否值得**。本任务 `batch_size: 1`、`sim_freq: 1200`、
   `update_freq: 30`（每次 update 40 substep）。MJWarp 的优势在多 world 并行，
   单 world 很可能比 CPU MuJoCo 慢；首次 `step` 另有约 55 s 的 kernel JIT
   （之后缓存于 `~/.cache/warp/<version>`）。默认后端的选择应在轮 3 之后再评估。

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

`assets/xmls/sensors/tactile_sensor.xml` 声明的触觉单元是**可碰撞的**
（`contype=conaffinity=1`），却带 `friction="0 0 0"` 与 `condim="4"`，MJWarp 报
NaN 风险警告。这是真实问题（CPU 侧同样存在），使用者是
`panda_robotiq.xml` / `robotiq_assets.xml`。这些单元是球体，不触发 CCD margin
门禁，因此**与本移植无关**，应由该传感器接触模型的负责人单独处理。

（对照：`xf9600_tactile_sensor.xml` 的同名警告是伪报——那批单元
`contype=conaffinity=0`，永不产生接触。MJWarp 扫描 geom 参数时未先判可碰撞性。）
