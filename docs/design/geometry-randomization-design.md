# 场景物体尺寸随机化设计

> **状态：提案，尚未实现。**
>
> 本文档描述未来通过 YAML 对场景物体尺寸进行随机化放缩的设计方向。
> 当前版本不能识别 `task.geometry_randomization`，文中的 YAML 仅作为拟议接口，
> 直接加入现有任务配置会因配置模型的额外字段校验而失败。

## 1. 背景与当前边界

当前项目已经有两类容易混淆、但语义不同的能力：

1. `task.randomization` 是 reset 时的物体/操作器位姿随机化，只改变位置和姿态。
2. `env.scene.layers[].scaling` 是 `asset_assembly` 的编译期固定缩放，使用
   `AssetScaleRuleConfig` 根据资产元数据计算一个确定性的统一比例。

现有资产缩放发生在场景片段生成和 `SceneArtifact` 编译阶段，而不是仿真 control
tick 中。普通的 `mjcf` 场景层没有一个可以直接配置的随机缩放入口。

此外，批环境当前先编译一份场景 artifact，再用同一份 XML 创建多个 MuJoCo
replica。因而“每个环境使用不同尺寸”不能通过给现有 pose randomization 增加字段
解决。

相关现状实现：

- [`AssetScaleRuleConfig`](../../auto_atom/scene_composition/config.py) —— 资产装配的
  固定、统一缩放规则。
- [`AssetAssemblyLayerConfig.scaling`](../../auto_atom/scene_composition/config.py) ——
  `asset_assembly` 层的编译期缩放配置。
- [`apply_asset_normalization`](../../auto_atom/scene_composition/normalization.py) ——
  在生成资产片段时应用缩放和 anchor。
- [`BatchedUnifiedMujocoEnv`](../../auto_atom/basis/mjc/mujoco_env.py) —— 当前批环境
  共享编译后的场景 artifact。
- [`MujocoTaskBackend.reset`](../../auto_atom/backend/mjc/mujoco_backend.py) —— 当前
  reset 顺序只处理 simulator reset、初始 pose、操作器初始状态、相机 pose 和 pose
  randomization。

## 2. 目标

未来能力应满足以下目标：

- 通过任务 YAML 配置物体尺寸的随机范围。
- 每个 reset、每个独立物理环境可以得到独立且可复现的尺寸样本。
- 视觉几何、碰撞几何、抓取 site 和相关局部几何保持一致。
- 缩放后仍能正确处理初始支撑、碰撞拒绝、抓取和放置。
- 不修改源 OBJ、资产包或用户维护的 XML 文件。
- 与现有 seed、随机化诊断和 Data Replay 语义一致。
- 保持通用场景资产契约，不为某一个任务或某一种物体增加专用分支。

## 3. 非目标

第一版不应尝试覆盖以下内容：

- 在每个 control tick 中连续改变物体尺寸。
- 通过直接修改已经加载的 MuJoCo mesh 数据来模拟通用缩放。
- 默认缩放机器人、相机或操作器结构。
- 在没有几何 anchor 语义的情况下自动猜测堆叠和放置高度。
- 让外部策略 action 因为尺寸随机化而被隐式改写。

## 4. 拟议 YAML 接口

建议在 `task` 下增加独立的 `geometry_randomization` 字段，而不是把几何变化塞入
现有的位姿 `randomization` 条目：

```yaml
task:
  geometry_randomization:
    cube_blue:
      uniform_scale: [0.90, 1.10]
      anchor: bottom
      physics: preserve_density
```

这段配置是**拟议格式，当前不可用**。字段建议如下：

| 字段 | 拟议含义 |
| --- | --- |
| `cube_blue` | 逻辑物体名；由后端解析到 body、子树和 mesh 依赖。 |
| `uniform_scale` | 相对于原始几何的倍率区间 `[min, max]`，两端均为正数。第一版只支持 XYZ 等比例缩放。 |
| `anchor` | 尺寸变化后的安装参考，如 `center` 或 `bottom`；`bottom` 用于保持物体与支撑面接触。 |
| `physics` | 质量和惯性的处理策略，建议显式选择 `preserve_density` 或 `preserve_mass`。 |

后续如有确切需求，再增加 `scale_xyz` 的独立轴范围。第一版不建议同时开放均匀和
非均匀两套语法，以免把 mesh、碰撞和任务 waypoint 的语义混在一起。

## 5. 随机化语义

### 5.1 采样时机与可复现性

- 尺寸样本在 reset 边界生成，而不是在 control tick 中生成。
- 每个物理环境单独采样；`task.seed` 仍是随机性的根来源。
- 每次 reset 先恢复原始场景，再应用新的尺寸样本，不能让上一个 reset 的尺寸
  继续累乘。
- 实际采用的倍率应进入随机化诊断或 reset metadata，便于复现和排查。
- `randomization_debug` 如果扩展到该能力，应能检查最小倍率、最大倍率和普通样本。

### 5.2 作用范围

逻辑物体名应是公开接口；XML body/mesh 名称只作为后端或场景编译器内部解析结果。
对于 namespaced asset assembly，应通过 semantic export 或等价的稳定映射解析，不能
要求任务作者手写供应商私有的局部 XML 名称。

缩放必须覆盖同一物体的：

- 视觉 mesh；
- 碰撞 geom；
- grasp/pose site；
- 与物体局部坐标有关的几何位置；
- 需要保持物理一致性的质量和惯性参数。

### 5.3 物理策略

如果采用 `preserve_density`，均匀缩放倍率为 `s` 时，建议遵循：

```text
mass' = mass × s³
inertia' = inertia × s⁵
```

如果任务更关心控制行为稳定而不是材料密度，则可以选择 `preserve_mass`，但应明确
说明其惯性处理规则，不能静默采用一个与原 XML 假设不一致的结果。

### 5.4 Anchor 与初始状态

仅缩放尺寸而保持 body 原位置，会产生悬空或插入桌面的物体。`anchor: bottom` 应在
场景实例化时根据缩放后的 bounds 调整 body 的初始位置，使底面继续落在原支撑面上。

如果一个物体由多个 body 组成，应通过声明的 anchor 和 preserve 规则处理子树；不应
依据 body 名称是否包含 `handle`、`button` 等字符串做猜测。

### 5.5 碰撞和可见性

现有 pose randomization 的 `collision_radius` 是近似半径。尺寸变化后，固定半径可能
低估物体的真实占用空间。未来设计应当：

- 优先从缩放后的 support geometry 计算保守半径；或
- 提供 `collision_radius: auto`，并允许用户配置额外 margin；
- 在相机可见性和分离约束中使用缩放后的几何，而不是原始尺寸。

## 6. 推荐运行时架构

### 6.1 不采用直接修改已加载模型数组

不建议把通用实现建立在 reset 时直接修改 `MjModel.geom_size` 或
`MjModel.mesh_scale` 上。MuJoCo 模型还包含 mesh 派生数据、碰撞加速结构、AABB、
惯性和 renderer 相关缓存；只修改一组数组很容易导致视觉尺寸、碰撞尺寸和动态参数
不一致。

### 6.2 场景 variant / model rebuild

推荐引入“场景实例 variant”流程：

```text
原始 SceneConfig
    │
    ├─ 编译 host + layers
    ├─ 采样 geometry_randomization
    ├─ 对最终 XML 应用实例级几何变换
    │    ├─ body 局部几何与 anchor
    │    ├─ 视觉/碰撞 mesh
    │    └─ 质量/惯性
    ├─ 生成该 reset 的 SceneArtifact / MjModel
    └─ 重新绑定 runtime handlers 与 renderer
```

这里的“重新绑定”包括但不限于：

- object/operator handler 的 model 引用和索引；
- camera renderer、camera id 和 mask 对象；
- tactile manager；
- actuator/joint 索引和 IK 相关缓存；
- reset baseline、support geometry 和碰撞检查缓存。

场景 variant 应可缓存。缓存 key 至少应包含基础场景 digest、规范化后的倍率映射和
几何策略，避免相同样本重复编译；缓存不能改变随机数的采样顺序。

### 6.3 共享 mesh 的实例化问题

当前 block 场景中多个 body 可能共享同一个 mesh（例如多个同颜色方块复用一个
`cube_blue_mesh`）。如果不同物体可以得到不同倍率，就不能直接修改共享 mesh 的
全局 scale。编译器需要为需要不同倍率的 body 复制 mesh 定义，并把对应 geom 指向
实例化后的 mesh；同一 body 内仍可复用同一个 clone。

这是 `stack_color_blocks` 和 `place_blocks_on_disk` 这类任务必须覆盖的验收场景。

### 6.4 Batch 与 shared physics

复制 batch 可以让每个物理 replica 使用自己的场景 variant。`share_physics` 模式只有
一份物理模型，不能为不同逻辑行提供不同几何尺寸。建议采取以下明确策略之一：

1. shared physics 下所有逻辑行共享同一个尺寸样本；或
2. 配置校验直接拒绝“每行独立尺寸随机化”。

第一版应优先选择可验证的行为，不能悄悄对不同逻辑行宣称已使用独立模型。

## 7. 与现有 reset、pose randomization 和 replay 的关系

建议 reset 顺序为：

1. 恢复或创建原始场景 variant；
2. 应用 geometry randomization，并建立缩放后的几何/anchor metadata；
3. 执行 simulator reset 和 operator home；
4. 应用 `initial_pose` 与 operator initial state；
5. 记录新的 pose baseline；
6. 执行现有的对象、操作器和相机 pose randomization；
7. 用缩放后的 support geometry 执行约束和碰撞拒绝。

几何随机化必须只发生在 reset 边界。物体已经被抓取或放置后，不应在 reset 中途
改变其尺寸。

Data Replay 必须选择并记录一种明确语义：

- 为精确轨迹复现，自动关闭 geometry randomization，并恢复记录中的场景 variant；
- 或把每个 reset 的实际倍率写入 replay metadata，并按该倍率重建模型。

不能只复现 pose 而忽略尺寸，因为这会改变接触、抓取和放置结果。

## 8. 现有任务的兼容性风险

尺寸随机化不只是一个 XML 变换问题，已有任务可能把几何尺寸写进了 waypoint 或
成功条件。

### 8.1 堆叠方块

[`stack_color_blocks.yaml`](../../aao_configs/stack_color_blocks.yaml) 的注释和 waypoint
使用固定的方块半高/堆叠高度，例如 `0.025` 和 `0.050`。方块尺寸变化后，放置目标
不能继续依赖这些常数，应改为基于目标物体顶部或接触面的动态 reference。

### 8.2 磁盘上的方块

[`place_blocks_on_disk_airbot_play_g2.yaml`](../../aao_configs/place_blocks_on_disk_airbot_play_g2.yaml)
含有固定的接近、放置和 retreat 高度。缩放后需要重新检查：

- 抓取姿态是否仍位于物体有效区域；
- 方块底面是否与磁盘/支撑面一致；
- 磁盘边缘和碰撞拒绝 margin 是否仍然足够；
- 多个方块的布局是否因实际占用半径变化而发生碰撞。

### 8.3 按钮类任务

对按钮或按压目标进行缩放时，还需审查触点 site、按压方向、关节行程和 contacted
条件。不能只把视觉按钮放大，却保留旧的碰撞几何或旧的成功阈值。

因此，第一版应优先覆盖几何结构简单、waypoint 不依赖固定尺寸的独立物体，并把
“几何派生 waypoint/reference”作为后续能力，而不是自动修改所有已有任务的常数。

## 9. 实施阶段建议

### 阶段 0：契约与诊断

- 增加 frozen Pydantic 配置模型和 `AutoAtomConfig` 字段。
- 完成正数范围、目标解析、倍率冲突和 shared-physics 校验。
- 增加只读的 sampled-scale diagnostics；此阶段仍可暂不启用实际模型替换。

### 阶段 1：native MuJoCo 均匀缩放

- 只支持独立物体和 uniform scale。
- 使用场景 variant / model rebuild，不修改源资产。
- 覆盖 primitive geom 和共享 mesh clone。
- 同步碰撞、视觉、site、惯性和 reset baseline。
- 完成复制 batch、seed、reset 不累乘和 replay 测试。

### 阶段 2：几何 anchor 与任务表达

- 提供 bottom/top/contact surface 等稳定几何 reference。
- 让堆叠和放置 waypoint 使用几何派生量，而不是固定尺寸常数。
- 将缩放后的 support geometry 接入 visibility、separation 和 placed 条件。

### 阶段 3：扩展到其他 renderer / asset family

- 让 asset package、普通 MJCF layer 和 GS 资产复用同一几何实例化契约。
- 明确 shared physics、renderer cache 和跨后端能力边界。

## 10. 验收标准

实现完成前，至少应通过以下检查：

- 未配置时现有任务行为和 XML digest 不变。
- 同一 seed、同一 reset 序列得到相同倍率和相同场景 variant。
- 不同复制 replica 可以得到不同倍率；shared physics 的行为符合显式策略。
- 视觉 mesh 和碰撞 geom 的实际尺寸比例一致。
- 共享 mesh 的不同实例可以使用不同倍率。
- `anchor: bottom` 不会产生悬空或明显穿透支撑面的初始状态。
- pose randomization 使用缩放后的 support geometry 和碰撞半径。
- reset 多次不会累积缩放。
- replay 能够精确恢复尺寸语义。
- 抓取、释放、接触和放置条件在最小/最大倍率下均有定向测试。
- 执行资源受限的定向测试，例如：

  ```bash
  /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python scripts/run_tests_safe.py \
    --test-targets tests/test_geometry_randomization.py \
    --max-concurrency=1
  ```

  上述测试路径是拟议的未来测试文件，当前不存在。

## 11. 待决策项

在开始实现前需要固定以下契约：

- 倍率区间采用线性 uniform 还是其他分布；
- `preserve_density` 的惯性定义以及显式质量/惯性 XML 的优先级；
- variant cache 的生命周期和最大容量；
- shared physics 采用共享样本还是直接拒绝；
- geometry metadata 是否写入现有随机化诊断、record 和 replay schema；
- 几何 reference 如何暴露给 stage waypoint 和 placed 条件。
