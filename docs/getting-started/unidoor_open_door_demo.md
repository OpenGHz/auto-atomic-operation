# UniDoor 开门 Demo 部署与运行

本文说明其他开发者从零 clone `auto-atomic-operation` 后，如何准备 UniDoor
门/把手资产并运行 P7 开门 Demo。

## 1. 获取代码和 Git LFS 资源

要求 Python 3.10 或更高版本。仓库中的通用 MuJoCo mesh 使用 Git LFS，先
安装 Git LFS，再拉取资源：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/OpenGHz/auto-atomic-operation.git
cd auto-atomic-operation

git lfs install
git lfs pull
```

## 2. 安装 Python 和 MuJoCo 依赖

建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -e '.[mujoco]'
```

只安装 `pip install auto-atomic-operation` 不足以运行仓库内的 Demo：PyPI
发行包不包含当前仓库的 Hydra 配置和演示资产。

## 3. 获取 UniDoor 资产包

UniDoor 资产不在 Git 中。原因是整个 `third_party/` 目录被忽略，而仓库中
跟踪的
`assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json` 只是
一个可重定位的 descriptor，不包含门/把手的 mesh、碰撞和纹理文件。

当前纹理完整的 55×47 右铰链资产包位于 b300-2：

```text
/DATA/disk1/home/zoyo/projects/Praxis-recording-package-worktree/outputs/
unidoor_texture_complete_package_v1/release/archives/
UniDoorManip.praxis_55x47.right_hinge_with_collision.zip
```

包大小为 `106102093` bytes，SHA-256 为：

```text
d32c15add6cf6ae0bd4a918f727bfe3c9d11ca0785a132f9034a5de11541c979
```

有 b300-2 权限时，使用可续传的 `rsync` 下载：

```bash
mkdir -p /tmp/unidoor-download

rsync --partial --append-verify --progress \
  -e ssh \
  b300-2:/DATA/disk1/home/zoyo/projects/Praxis-recording-package-worktree/outputs/unidoor_texture_complete_package_v1/release/archives/UniDoorManip.praxis_55x47.right_hinge_with_collision.zip \
  /tmp/unidoor-download/package.zip
```

下载后先校验，不要跳过这一步：

```bash
sha256sum /tmp/unidoor-download/package.zip
unzip -tq /tmp/unidoor-download/package.zip
```

第一条命令必须输出上面的 SHA-256。确认无误后解压到仓库的
`third_party/`：

```bash
unzip -q /tmp/unidoor-download/package.zip -d third_party
```

检查默认 descriptor 所需的关键文件：

```bash
test -f third_party/unidoor_lever_catalog_pipeline_right_hinge/product_space.json
test -f third_party/unidoor_lever_catalog_pipeline_right_hinge/components/doors/D001/visual/panel/panel.png
```

资产目录必须与 descriptor 的 `payload_root.uri` 对应。只复制
`scene_asset_package.json` 而不提供 payload，或者把环境变量直接指向 raw
catalog 目录，都会在场景装配时失败。

### 使用自定义资产位置

配置默认使用：

```yaml
package: ${oc.env:AAO_UNIDOOR_ASSET_PACKAGE,assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json}
```

因此可以通过环境变量选择另一份 canonical descriptor：

```bash
AAO_UNIDOOR_ASSET_PACKAGE=/data/unidoor/scene_asset_package.json \
aao-demo --config-name open_door_unidoor_p7_v3_umi_v3
```

自定义 descriptor 的 `payload_root.uri` 必须能解析到同一部署中的
`product_space.json`。若资产被放到了新的目录，先为该 payload 生成 sidecar
descriptor，再把环境变量指向生成的 `scene_asset_package.json`：

```bash
python -m auto_atom.scene_composition.migrate \
  /data/unidoor/raw_catalog \
  /data/unidoor/canonical_package \
  --overwrite

AAO_UNIDOOR_ASSET_PACKAGE=/data/unidoor/canonical_package/scene_asset_package.json \
aao-demo --config-name open_door_unidoor_p7_v3_umi_v3
```

运行时会根据 descriptor 读取 `product_space.json`，再加载选中门和把手的
manifest、UV mesh、PNG 材质和碰撞补充；预生成的组合 XML 不是运行时必需输入。

## 4. 运行开门 Demo

CLI 必须从仓库根目录运行，因为 Hydra 从当前目录的 `aao_configs/` 解析配置。

先列出任务，确认配置已被发现：

```bash
aao-info
```

运行默认的 `D001/H001`：

```bash
aao-demo \
  --config-name open_door_unidoor_p7_v3_umi_v3 \
  door_id=D001 \
  handle_id=H001 \
  env.batch_size=1 \
  +env.viewer.disable=true
```

`+env.viewer.disable=true` 会关闭 viewer，适合 SSH 或没有桌面环境的机器。
需要交互式查看场景时，在有可用 OpenGL/桌面的机器上运行：

```bash
python examples/view_scene.py \
  --config-name open_door_unidoor_p7_v3_umi_v3
```

无桌面环境但需要离屏渲染时，可尝试 EGL：

```bash
MUJOCO_GL=egl aao-demo \
  --config-name open_door_unidoor_p7_v3_umi_v3 \
  +env.viewer.disable=true
```

## 5. 运行其他门/把手组合

通过 Hydra 覆盖选择不同资产：

```bash
aao-demo \
  --config-name open_door_unidoor_p7_v3_umi_v3 \
  door_id=D002 \
  handle_id=HL016 \
  +env.viewer.disable=true
```

批量测试使用 `aao-unidoor-sweep`。先用小集合验证环境和任务配置：

```bash
aao-unidoor-sweep \
  --config-name open_door_unidoor_p7_v3_umi_v3 \
  --doors D001,D002 \
  --handles H001,H004,HL001 \
  --max-concurrency=1
```

完整 catalog 是 55×47，即 2,585 个组合，通常需要数小时。默认并发上限为
4；资源有限时显式设置 `--max-concurrency=1`。

Sweep 输出位于 `outputs/unidoor-sweeps/<timestamp>/`，其中：

```text
report.json       # 每个门/把手组合的结构化结果
failures.csv      # 失败组合及可直接复现的命令
sweep_manifest.json
sweep.log
batches/.../summary.json
```

某个组合失败时，可以从 `failures.csv` 的 `reproduce_command` 复现；修复
配置后使用同一目录执行 `aao-unidoor-sweep --resume <目录>`，成功组合不会
重复运行。

## 6. 结果解释与已知限制

资产安装成功只表示场景能够被正确装配和加载，不代表每个门/把手组合都能被
P7 成功操作。当前任务对部分组合仍可能出现 IK 不可达、抓取接触不足或运动
控制失败；这些属于机器人初始位姿、把手几何和任务控制参数问题，不是 clone
或资产下载问题。

若遇到资产错误，优先检查：

1. `product_space.json` 是否存在且可读。
2. descriptor 的 `payload_root.uri` 是否指向实际 payload。
3. 下载包 SHA-256 和 `unzip -tq` 是否通过。
4. 运行命令是否从仓库根目录执行。
5. `python -m pip install -e '.[mujoco]'` 是否完成。

更多场景装配字段说明见
[Scene Composition](../task-configuration/scene_composition.md)，命令行和 sweep
参数见 [CLI Reference](cli_reference.md)。
