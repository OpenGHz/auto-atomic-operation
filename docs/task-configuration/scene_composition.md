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
        selection: {door: D001, handle: H003}
        namespace: door
        placement:
          # Rotate around the panel centre so the handle faces the robot
          # without moving the panel out of the interaction region.
          position: [1.54, 0.13939759036, -1.0]
          orientation_xyzw: [0.0, 0.0, -0.707106781, 0.707106781]
        verify_hashes: true
```

`SceneConfig` 是纯数据模型：不在 YAML 中放 adapter 实例、Python callable 或
临时文件。adapter 只在运行时 registry 中按 `adapter@version` 查找。一个场景
可以声明任意多个 asset assembly；namespace 必须唯一，生成的 MJCF 名称采用
`<namespace>__<local-name>`，adapter 同时返回逻辑 semantic exports，例如：

| logical export | generated name (`namespace=door`) |
| --- | --- |
| `door.door.hinge.joint` | `door__door_hinge` |
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

## UniDoor 示例

完整任务见 `aao_configs/open_door_unidoor_p7_v3_umi_v3.yaml`。该任务使用
`asset_assembly` layer 选择 `D001/H003`，并将门的 semantic names 映射到
`door__door_handle`、`door__handle_grasp_center`、`door__handle_hinge` 和
`door__door_hinge`。替换门或把手只需改 `selection`；替换另一类资产则实现一个
新的 adapter，不需要修改 `EnvConfig`、Basis 或 viewer。

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
