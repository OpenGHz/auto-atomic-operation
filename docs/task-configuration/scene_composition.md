# Scene composition

场景拼接是一个通用的 `SceneConfig` 契约：一个 host MJCF 加上按声明顺序
编译的 scene layers。layer 可以是已有的 MJCF 文档，也可以是从 scene asset
package 解析出的资产装配。环境、viewer 和 MuJoCo basis 只消费这个契约，
不需要知道资产供应商或具体机构。

## 配置契约

```yaml
env:
  scene:
    base: ${assets_dir}/xmls/scenes/open_door_unidoor/demo.xml
    layers:
      # 机器人也是普通的 MJCF layer；顺序决定声明和资源合并顺序。
      - kind: mjcf
        path: ${assets_dir}/xmls/robots/p7_arm_v3_with_umi_gripper_v3.xml
      # 资产装配使用稳定 namespace，避免多个实例的名字互相覆盖。
      - kind: asset_assembly
        package: assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json
        adapter: unidoor.lever_door@1
        selection: {door: D001, handle: H001}
        namespace: door
        placement:
          # Rotate around the panel centre so the handle faces the robot
          # without moving the panel out of the interaction region.
          position: [1.54, 0.13939759036, -1.0]
          orientation_xyzw: [0.0, 0.0, -0.707106781, 0.707106781]
        verify_hashes: true
        # Optional generic cold-path geometry transforms. Rules select
        # compiled bodies/meshes and derive a uniform factor from component
        # metadata; they are not applied in a control tick.
        scaling:
          - bodies: [door_frame, door_panel]
            preserve_bodies: [door_handle, door_lock]
            meshes: [unidoor_frame_mesh, unidoor_panel_mesh]
            source_bounds: door.geometry.panel_bounds_m
            axis: z
            target_extent_m: 2.0
        anchors:
          - bodies: [door_handle]
            source_bounds: door.geometry.panel_bounds_m
            coordinates:
              x: {edge: min, offset_m: 0.08}
              y: {edge: max, multiplier: -1.0}
              z: {value_m: 1.0}
```

`SceneConfig` 是纯数据模型：不在 YAML 中放 adapter 实例、Python callable 或
临时文件。adapter 只在运行时 registry 中按 `adapter@version` 查找。一个场景
可以声明任意多个 asset assembly；namespace 必须唯一，生成的 MJCF 名称采用
`<namespace>__<local-name>`，adapter 同时返回逻辑 semantic exports，例如：

| logical export | generated name (`namespace=door`) |
| --- | --- |
| `door.door.hinge.joint` | `door__door_hinge` |
| `door.latch.constraint` | `door__door_latch_lock` |
| `door.handle.hinge.joint` | `door__handle_hinge` |
| `door.handle.grasp.site` | `door__handle_grasp_center` |
| `door.handle.object` | `door__door_handle` |

任务配置应引用确定性的 namespaced 名称（或由上层任务编译器消费 exports），
不要依赖某个供应商的全局 `door_hinge`。

## 加载与合并语义

公共入口是 `auto_atom.scene_composition.load_composed_scene(SceneConfig)`；
`EnvConfig.scene` 必须提供这个完整 recipe。compiler 对每个 layer 做以下工作：

1. 解析 host；MJCF layer 递归展开 include，并以其源文件目录为基准绝对化 mesh、
   texture、model 和 compiler 路径。
2. adapter 只返回 `SceneContribution`（fragment、semantic exports、依赖摘要和
   diagnostics），不直接改 host，也不管理临时文件。
3. host composer 合并 `asset`、`worldbody`、`contact`、`equality`、`tendon`、
   `actuator`、`sensor`、`keyframe`、`custom`、`extension` 等 list sections，并
   检查所有 named element/default class 冲突。`option`、`visual`、`compiler` 等
   singleton 只填补 host 未声明的属性，不能被隐藏 layer 覆盖。
4. 只有在需要 layer 时才在 host 同目录 materialize 临时 XML；MuJoCo 加载后
   立即删除。artifact digest 同时覆盖 XML 和依赖路径/内容哈希。

批环境会先编译一次 `SceneArtifact`，再为每个 replica 从同一份 XML artifact
创建独立的 `MjModel/MjData`；因此 batch size 不会把 manifest/OBJ 解析工作重复
放大，物理状态仍彼此隔离。

这种合并规则让第二个 adapter 复用同一 seam，也避免旧 `_merge_fragment` 只认识
少数 sections 而静默丢失物理约束。

## 编译期几何缩放与锚点

这一节的配置可以先用下面的流程来理解：

```text
组件 manifest 的原始 bounds
        │
        ├─ scaling：计算统一缩放倍数，并缩放生成的 MJCF 几何
        │
        ├─ anchors：按缩放后的 bounds 重新确定部件的局部安装位置
        │
        ├─ namespace：将 door_handle 等局部名称改成 door__door_handle
        │
        └─ placement：把整套已经装好的资产放入 host 世界坐标
```

`scaling` 和 `anchors` 都是 `AssetAssemblyLayerConfig` 的通用、编译期（cold
path）变换配置。它们不属于控制器，也不会在仿真每个 control tick 中重新计算。
原始 OBJ、manifest 和资产包不会被改写；变换只作用于 adapter 生成、随后合并进
`SceneArtifact` 的 MJCF fragment。

### `scaling`：把不同来源的资产统一到目标尺寸

例如任务配置中的门规则是：

```yaml
scaling:
  - bodies: [door_frame, door_panel]
    meshes: [unidoor_frame_mesh, unidoor_panel_mesh]
    source_bounds: door.geometry.panel_bounds_m
    axis: z
    target_extent_m: 2.0
```

字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `bodies` | 要缩放其局部几何的 MJCF body 名称；会覆盖 body 拥有的 geom、site、joint、inertial 等尺寸或位置属性。 |
| `preserve_bodies` | 当 `bodies` 包含父 body 时，列出的子 body 及其子树保持原始尺寸、位置和惯量；用于父资产缩放但某个附属部件暂不缩放的情况。 |
| `meshes` | 要设置 `mesh scale` 的 MJCF mesh 名称。 |
| `mesh_prefixes` | 按名称前缀选择一组 mesh，适合数量不固定的凸分解碰撞块。 |
| `source_bounds` | adapter metadata 中 bounds 的点路径，不是文件路径；例如 `door.geometry.panel_bounds_m` 表示 `door → geometry → panel_bounds_m`。 |
| `axis` | 从 bounds 的哪个轴计算原始尺寸。它不是“只缩放这个轴”，当前规则始终做 XYZ 等比例缩放。 |
| `target_extent_m` | 该轴缩放后的目标尺寸，单位为米。 |
| `required` | 默认是 `true`，目标 body/mesh 缺失时编译失败；可选变体（例如没有锁体）才设为 `false`。 |

缩放倍数的公式是：

```text
source_extent = bounds[1][axis] - bounds[0][axis]
scale = target_extent_m / source_extent
```

以 D001 为例，`panel_bounds_m` 的 z 向高度约为 `1.73494 m`，因此门规则得到：

```text
scale = 2.0 / 1.73494 ≈ 1.15278
```

这个比例会同时用于门框和门板的指定 mesh，以及它们局部的碰撞几何、site、
joint 位置等，从而避免“视觉门变大了、碰撞门仍是旧尺寸”的不一致。多个规则如果
引用同一个 `source_bounds`，必须导出相同的比例；冲突会在编译期报错，而不是静默
采用其中一条。

如果被缩放的 body 包含暂时不希望缩放的子 body，可以用 `preserve_bodies` 明确
保留它。例如当前开门配置让门框和门板按 2 m 归一化，同时保留门板子树中的
`door_handle` 和 `door_lock`，避免门的比例变化连带改变把手的碰撞体、抓取 site
和惯量。该字段只影响 body 子树；显式列在 `meshes` 或 `mesh_prefixes` 中的 mesh
仍会按规则缩放。

把手规则的含义相同，只是目标尺寸是 `0.15 m`：

```yaml
- bodies: [door_handle]
  meshes: [unidoor_handle_mesh]
  source_bounds: handle.geometry.handle_bounds_m
  axis: x
  target_extent_m: 0.15
```

H001 的 x 向 bounds 长度约为 `0.15 m`，所以如果启用该规则，它的比例约为 `1.0`；
更长或更短的把手会按各自 manifest 的长度计算不同的比例。当前任务暂时不启用把手
尺寸归一化：配置文件保留了三条原始规则，但全部以 YAML 注释形式保存；把手和锁体
由门规则的 `preserve_bodies` 保持原始尺寸。以后需要统一把手长度时，可恢复这些
注释，并相应移除 `preserve_bodies` 中的部件。锁体规则和凸分解碰撞 mesh 规则复用
同一个把手比例；锁体使用 `required: false`，因为并不是每个把手变体都有锁体。

### `anchors`：缩放之后重新确定安装位置

缩放会改变部件的尺寸，也会改变“门边在哪里”。`anchors` 用来在缩放后重新设置
body 的局部 `pos`：

```yaml
anchors:
  - bodies: [door_handle]
    source_bounds: door.geometry.panel_bounds_m
    coordinates:
      x: {edge: min, offset_m: 0.08}
      y: {edge: max, multiplier: -1.0}
      z: {value_m: 1.0}
```

`coordinates` 只修改列出的轴，未列出的轴保留 adapter 生成的原值。每个轴的坐标
有且只能有两种来源：

| 写法 | 计算方式 |
| --- | --- |
| `value_m: 1.0` | 使用固定的局部坐标 `1.0 m`。 |
| `edge: min` 或 `edge: max` | 从 `source_bounds` 取该轴的最小值或最大值，再应用 `multiplier` 和 `offset_m`。 |

也就是说，投影坐标的公式是：

```text
coordinate = transformed_bounds[edge][axis] * multiplier + offset_m
```

使用 `edge` 时必须提供 `source_bounds`；只使用固定 `value_m` 的 anchor 可以不提供
它。`source_bounds` 会使用对应 `scaling` 规则的比例先变换，再做 `min`/`max`
投影。

对于当前右铰链门：

- `x: {edge: min, offset_m: 0.08}`：从门板自由边（`min.x`）向内偏移 8 cm；
- `y: {edge: max, multiplier: -1.0}`：取门板的另一侧表面并翻到负 y 侧；
- `z: {value_m: 1.0}`：把手中心固定在装配局部高度 1 m。

用 D001 的门板 bounds 和 `1.15278` 比例计算，得到的把手局部位置约为：

```text
[-0.6506024 × 1.15278 + 0.08,
  0.0208359 × 1.15278 × (-1),
  1.0]
≈ [-0.6700, -0.0240, 1.0000]
```

这里的数值仍然是资产装配局部坐标，不是世界坐标。之后还会应用 layer 的
`placement`；当前配置的 `placement.z = -1.0`，所以把手的世界 z 高度约为
`-1.0 + 1.0 = 0`（绕 z 轴的整体旋转不会改变 z）。

### `anchors`、`placement` 与运行时 reference 的区别

- `anchors`：编译时调整“把手相对于门”的局部安装位置；
- `placement`：编译时调整“整套门相对于 host 世界”的平移和旋转；
- Stage waypoint 的 `reference`：执行阶段解析目标位姿的参考坐标系；
- randomization 的 `reference`：reset 时生成随机扰动所依据的参考位姿。

因此 `anchors` 不是运行时跟踪把手的 reference，也不会让机械臂随着门的运动持续
跟随。它只负责把生成资产在初始装配时放到正确位置。

### 为什么这套配置是通用的

通用变换模块只认识两类信息：adapter 传入的 metadata bounds，以及 fragment 中
声明的 body/mesh 名称。它不认识 UniDoor、Praxis 或某一种门的专有分支；门、把手、
工具、夹具等其他资产族都可以复用同样的 `scaling`/`anchors` 字段，只需让自己的
adapter 提供对应的 bounds 和稳定的 MJCF 名称。

UniDoor 的 2 m 门高、1 m 把手安装高度和 0.15 m 把手长度都在
`aao_configs/open_door_unidoor_p7_v3_umi_v3.yaml` 的 asset assembly layer 中声明。
替换其他门或把手时，通常只需修改 `selection`；如果新资产的原始尺寸不同，再按
其 manifest bounds 调整 `target_extent_m` 或 anchor 规则。

## Scene asset package v1

通用 package descriptor 的根 schema 是 `aao.scene-asset-package/v1`，至少包含：

- `package_id` / `revision`、单位和 canonical frame（包括 quaternion 顺序及
  `transform_baked`）；
- component index（opaque id、kind、manifest URI、status、hash）；
- component manifest 中的 `artifacts`、相对 URI、representation、frame、hash，
  可选 `anchors`、`geometry.bounds`、`mechanism.joints`；
- adapter/template catalog、integrity policy、provenance 和 extensions。

artifact URI 必须是 payload-relative POSIX 路径，解析后不能越过声明的
`payload_root`；绝对源包路径只能作为 opaque provenance，不能成为运行时依赖。
关节 axis、pivot、limits、range、stiffness、damping 等动力学属于版本化
assembly template，必须显式记录。运行时 adapter 不从 `hinge_side` 猜测轴；只有
离线迁移器可以推导并把推导依据写入 provenance。

本仓库的 [UniDoor package view](../../assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json)
只增加一个可追踪的小 descriptor，引用 `third_party` 中现有 OBJ/sidecar，不复制
或改写 144 MiB 资产。迁移器会为 102 个 component 生成 176 个 visual 与 102
个 collision artifact 引用（JSON sidecar，不复制 mesh）。当前 UniDoor adapter
将 `product_space.json` 作为该 package 的私有 component index，并校验选中 manifest
和 mesh 的 SHA-256；`combinations/` 只作为结构回归 oracle。

把手 component 可以声明 `motrixsim_acd_convex_parts_v1` collision supplement。
adapter 会校验 sidecar 路径、文件哈希、component/source/handedness identity、固定
slot 顺序、每个凸块的几何哈希和启用数量，然后为每个启用凸块分别生成一个 inline
MJCF mesh geom。不同凸块不会合并，因为 MuJoCo 对单个 mesh 使用凸包碰撞，合并会
丢失凸分解的接触形状；`collision_enabled=false` 的固定拓扑占位槽不会进入模型。

碰撞表示与动力学质量相互独立：所有 ACD geom 都使用零 density，把手 body 使用
与原 AABB 表示相同的 `0.25 kg` 显式质量和 bounds-derived inertia，因此替换把手或
改变凸块数量不会放大质量。未声明 collision supplement 的 component 继续使用
`handle_bounds_m` 的单个 AABB；一旦声明了 supplement，缺文件、未知 representation、
哈希或拓扑错误都会终止编译，不会静默退回 AABB。

## UniDoor 示例

完整任务见 `aao_configs/open_door_unidoor_p7_v3_umi_v3.yaml`。该任务使用
`asset_assembly` layer 默认选择 `D001/H001`，并将门的 semantic names 映射到
`door__door_handle`、`door__handle_grasp_center`、`door__handle_hinge` 和
`door__door_hinge`。替换门或把手只需改 `selection`；替换另一类资产则实现一个
新的 adapter，不需要修改 `EnvConfig`、Basis 或 viewer。

执行流程由三个 Stage 明确分工：`pick_handle` 通过 grasp site 接近，并按 `PICK`
的内建契约验证目标抓取后结束；`pull_handle` 在 EEF 边界再次验证目标仍被
抓住，再把把手圆弧作为 `post_move` 执行，因此旋转前后都会检查同一目标；
`push_open` 则把门铰链圆弧作为 `pre_move` 执行，并以目标位移判定成功。
这里的 `PULL` / `PUSH` 描述条件契约，而不是日常语义中的运动方向。

预生成组合 XML 不参与运行时装配。它们仍可用于对比关节轴、site、尺寸和可选锁
具的结构回归。结构加载通过不代表动态开门成功；waypoint、IK 可达性、碰撞和
任务 postcondition 仍需独立验证。

## Home pose 与 viewer

host 不再需要为每个机器人复制 keyframe。`env.initial_joint_positions` 仍在
`MujocoBasis.reset()` 中应用；scalar、free、ball joint 的写入规则保持不变。
`examples/view_scene.py` 在 reload 时重新读取同一个 `SceneConfig`，因此 viewer
和实际环境共享 compiler、namespace、integrity 检查和 home-pose 逻辑。

## 相关文档

- [ADR-0001：通用场景组合契约](../adr/0001-generic-scene-composition.md)
- [任务复用与创建](reusing_and_creating_tasks.md)
- [View Scene](../tools/view_scene.md)
