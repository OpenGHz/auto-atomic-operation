# GS Multi-Background Assignment

本文档记录当前 Gaussian Splatting 多背景机制的完整现状，包括配置项、batch 环境背景分配、reset 行为、渲染时机和缓存策略。

相关实现：

- `auto_atom/basis/mjc/gs_mujoco_env.py`
- `GaussianRenderConfig`
- `GSUnifiedMujocoEnv`
- `BatchedGSUnifiedMujocoEnv`

## 配置入口

`env.gaussian_render.background_ply` 当前支持四种形式：

| 形式 | 含义 |
| --- | --- |
| `null` | 不使用 GS 背景 |
| `str` | 单个背景路径，所有环境共享 |
| `str` 含 glob 元字符（`*`、`?`、`[...]`） | 单条字符串展开为多背景池 |
| `list[str]` | 多背景池；列表中的任意一项也可以是 glob |
| `Dict[str, str \| list[str]]` | **部件字典**：每个 key 是一个部件类别（如 `wall`、`inside`），每个值是该部件的路径或 glob。每个环境会拿到一个由各部件 PLY 各取一份的 *组合*，而不是从池子里整张挑一张 |

示例：

```yaml
env:
  gaussian_render:
    background_ply:
      - ${assets_dir}/gs/backgrounds/door_bg/bg0.ply
      - ${assets_dir}/gs/backgrounds/door_bg/bg1.ply
      - ${assets_dir}/gs/backgrounds/door_bg/bg2.ply
    background_transform:
      [-0.105, 0.491151190, -0.1360378, -0.51684557, 0.495912190, -0.47649889, 0.509794620]
    randomize_background_on_reset: false
```

### Glob 展开

列表条目或单条字符串如果包含 `*` / `?` / `[...]`，会在配置解析时被 `glob.glob` 展开，
并按 `natsort.natsorted` 排序，保证 `bg2.ply` 在 `bg10.ply` 之前。
没有 glob 元字符的条目原样保留。

```yaml
env:
  gaussian_render:
    background_ply: ${assets_dir}/gs/backgrounds/door_bg/bg*.ply
```

如果 glob 一项都没匹配上，会在解析配置时抛 `FileNotFoundError: background_ply glob matched no files: <pattern>`，而不是静默忽略；这让缺少背景素材可以被尽早发现。

## 部件字典（Dict）模式

`list` / glob 模式假设每张背景 PLY 都是一张完整场景，每个环境从池子里整张挑一张。
部件字典模式则把背景拆成若干 *部件*（例如墙面、室内陈设），各部件分别提供一份候选池，
每个环境的背景是 *从每个部件各取一张 PLY 的组合*：

```yaml
env:
  gaussian_render:
    background_ply:
      wall:   ${bg3dgs_dir}/${wall_name}.ply   # glob，N 张墙面
      inside: ${bg3dgs_dir}/${inside_name}.ply  # 单张室内
```

实现要点：

- 每个部件的值会单独走 `_expand_background_entry` 做 glob 展开。
- 每个部件的 PLY 会先各自烤入 transform（见下文 *Transform 规则*），再做合并。
- 一个 *组合* 是从每个部件各挑一张 PLY 的 tuple；它们会被 `_merge_background_plys`
  拼成一张合并后的背景 PLY，落盘缓存到 `.cache/gs_background_combos/`，
  缓存 key 是组合内源 PLY 绝对路径排序后的 sha1 前 12 位。
- 合并要求所有部件 PLY 的 SH 阶数一致；不一致会抛 `ValueError` 提示具体形状。
- 合并后的 PLY 路径列表喂给现有的多背景渲染管线（参见“Batched 环境的分配规则”），
  也就是说 dict 模式 = list 模式的输入构造前置阶段，渲染层无差别。

### 组合数控制

dict 模式下的组合数 = 各部件候选数的笛卡尔积，单看 `wall` × `inside` 配比可能不大，
但部件多了会爆。当前调用方按使用场景传 `max_combinations`：

| 调用方 | `max_combinations` | 说明 |
| --- | --- | --- |
| `GSUnifiedMujocoEnv`（单环境） | `None`（不限） | 把所有可能组合都生成出来，`randomize_background_on_reset` 才有足够池子可以重抽 |
| `BatchedGSUnifiedMujocoEnv`（批量环境） | `self.batch_size` | 只 materialize batch 实际能用上的组合数 |

采样规则（`_sample_combinations`）：

- 若 `max_combinations is None` 或 `>= 总组合数`：返回完整笛卡尔积，顺序按 `itertools.product` 决定；
- 否则按 `np.random.Generator.choice(total, size=cap, replace=False)` 在扁平化索引上无放回采样，
  再用 `np.unravel_index` 还原成各部件多维索引。

### Transform 在哪一步生效

部件字典模式下，`background_transform` / `background_transforms` 都按 *单 PLY* 烤入；
每张部件 PLY 的 pose 解析按下列优先级（高 → 低）：

1. `background_transforms[<完整路径 | str(Path) | 文件名 | stem>]` — 单 PLY 精确匹配；
2. `background_transforms[<部件名>]` — 整批部件默认（如 `background_transforms.wall` 应用于
   `wall*` 展开后的每一张 PLY）；
3. 全局 `background_transform`；
4. identity。

这样 `wall` 和 `inside` 既可以靠部件名 key 整批指定不同的世界位姿，
又能通过 stem key 给某张 `wall7.ply` 单独再调。

合并出的组合 PLY 不再额外烤 transform——所有刚体变换都已经在部件 PLY 阶段烤完了。

### 与 list 模式的并存关系

`is_multi_background()` 对 dict 模式也返回 `True`，因此：

- `share_physics` 校验、`set_background_transform` 运行时拦截、reset 重分配开关都自动覆盖 dict 模式；
- `set_background_transform(...)` 在 dict 模式下同样禁用，错误信息已扩展为 “list or parts dict”。

## Transform 规则

`background_transform` 和 `background_transforms` 的优先级如下：

| 优先级 | 配置 | 行为 |
| --- | --- | --- |
| 1 | `background_transforms[<完整路径 / str(Path) / 文件名 / stem>]` | 单 PLY 精确匹配，给某张背景单独指定 transform |
| 2 | `background_transforms[<部件名>]` | **仅部件字典模式**：按部件 key 匹配（如 `wall`、`inside`），整批部件 PLY 共享 |
| 3 | `background_transform` | 全局默认 transform；list / 部件字典模式下会应用到每个背景 / 每张部件 PLY |
| 4 | identity | 未配置时使用零平移和单位四元数 |

`background_transform` 支持：

- 长度 3：`[x, y, z]`
- 长度 7：`[x, y, z, qx, qy, qz, qw]`

当 `background_ply` 是 list 或部件字典时，`set_background_transform(...)` 不支持运行时修改整组背景；需要通过配置里的 `background_transform` 或 `background_transforms` 设置。

### `background_transforms` 配置示例

`background_transforms` 是 `Dict[str, list]`，key 按下列顺序匹配（任一命中即生效）：
完整字符串路径 → `str(Path(...))` 规范化路径 → 文件名（含扩展名）→ stem（不含扩展名）。
所以对于 `${bg3dgs_dir}/door_bg.ply`，下面四个 key 等价：
`"${bg3dgs_dir}/door_bg.ply"`、`"assets/gs/backgrounds/door_bg.ply"`、`"door_bg.ply"`、`"door_bg"`。
日常用 stem 写最短，必要时再用全路径区分同名 PLY。

#### 单背景：用 stem 给唯一背景设位姿

```yaml
env:
  gaussian_render:
    background_ply: ${assets_dir}/gs/backgrounds/door_bg.ply
    background_transforms:
      door_bg: [-0.105, 0.491151190, -0.1360378,
                -0.51684557, 0.495912190, -0.47649889, 0.509794620]
```

等效写法（直接用 `background_transform`，仅当只有一张背景时简洁）：

```yaml
env:
  gaussian_render:
    background_ply: ${assets_dir}/gs/backgrounds/door_bg.ply
    background_transform:
      [-0.105, 0.491151190, -0.1360378,
       -0.51684557, 0.495912190, -0.47649889, 0.509794620]
```

#### list / glob 模式：默认 + 个别覆盖

`background_transform` 给整池一个共同位姿，`background_transforms` 给个别条目单独覆盖。

```yaml
env:
  gaussian_render:
    background_ply: ${bg3dgs_dir}/bg*.ply   # 14 张同帧采集的背景
    background_transform:                   # 整池共用的世界位姿
      [-0.105, 0.491151190, -0.1360378,
       -0.51684557, 0.495912190, -0.47649889, 0.509794620]
    background_transforms:
      bg7: [3.185, 0.608849, -0.1360378,    # bg7 单独翻到门的另一侧
            -0.495912, -0.516846, 0.509795, 0.476499]
      bg11: [0.0, 0.0, 0.05]                # bg11 仅在 z 方向上抬 5cm（长度 3 = 纯平移）
```

#### 部件字典模式：按部件名整批指定（最常用）

部件字典模式下，`background_transforms` 的 key 可以直接是部件名（如 `wall`、`inside`），
此时该 key 的 transform 会应用到这一部件展开出来的 *所有* PLY；同部件内某张 PLY 仍然可以用
stem / 文件名 key 继续覆盖（单 PLY 精确匹配优先级更高）。

```yaml
wall_name: wall*
inside_name: inside10
bg3dgs_dir: ${assets_dir}/gs/backgrounds/door_bg/

env:
  gaussian_render:
    background_ply:
      wall:   ${bg3dgs_dir}/${wall_name}.ply
      inside: ${bg3dgs_dir}/${inside_name}.ply
    background_transforms:
      wall:                                   # 应用到 wall* 展开后的每一张
        [-0.105, 0.491151, -0.136038,
         -0.516846, 0.495912, -0.476499, 0.509795]
      inside:                                 # 应用到 inside* 展开后的每一张
        [3.185, 0.608849, -0.186038,
         -0.495912, -0.516846, 0.509795, 0.476499]
```

如需对部件内某张 PLY 单独再调，加一条 stem key 即可（精确匹配优先于部件 key）：

```yaml
env:
  gaussian_render:
    background_ply:
      wall: ${bg3dgs_dir}/wall*.ply
    background_transforms:
      wall:  [-0.105, 0.491151, -0.136038,    # 部件级默认
              -0.516846, 0.495912, -0.476499, 0.509795]
      wall7: [-0.105, 0.491151, -0.086038,    # wall7 单独抬高 5cm
              -0.516846, 0.495912, -0.476499, 0.509795]
```

如果所有部件共用同一位姿，把它提到 `background_transform`：

```yaml
env:
  gaussian_render:
    background_ply:
      wall:   ${bg3dgs_dir}/${wall_name}.ply
      inside: ${bg3dgs_dir}/${inside_name}.ply
    background_transform:                     # 全局默认，所有部件 PLY 共用
      [-0.105, 0.491151, -0.136038,
       -0.516846, 0.495912, -0.476499, 0.509795]
```

匹配优先级：单 PLY 精确匹配 (`stem` / 文件名 / 路径) → `background_transforms[<部件名>]`
→ 全局 `background_transform` → identity。

#### 同名 PLY：用完整路径消歧

stem 不足以区分时退到完整路径。比如两个目录下都有 `bg0.ply`：

```yaml
env:
  gaussian_render:
    background_ply:
      - assets/gs/backgrounds/door_bg/bg0.ply
      - assets/gs/backgrounds/lab_bg/bg0.ply
    background_transforms:
      "assets/gs/backgrounds/door_bg/bg0.ply": [-0.105, 0.491151, -0.136038]
      "assets/gs/backgrounds/lab_bg/bg0.ply":  [0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.7071068, 0.7071068]
```

匹配顺序里完整路径排在 stem 之前，所以这两条 key 不会被同名的 stem `bg0` 错配。

## 位置随机化：`background_transform_randomization`

部件字典 / list / glob 模式下都支持给每条背景或每个部件加一个 *uniform 位置随机偏移*，
让每个 env 拿到位姿略有不同的背景，扩大视觉多样性。orientation **不**参与随机化（保留
`background_transforms` 里给出的固定四元数）。

```yaml
env:
  gaussian_render:
    background_ply:
      wall:   ${bg3dgs_dir}/wall*.ply
      inside: ${bg3dgs_dir}/inside*.ply
    background_transforms:
      wall:   [-0.035, 0.491151, -0.136038,
               -0.516846, 0.495912, -0.476499, 0.509795]
      inside: [ 0.300, 0.491151, -0.136038,
               -0.516846, 0.495912, -0.476499, 0.509795]
    background_transform_randomization:
      wall:
        x: [-0.05, 0.05]   # ± 5 cm 沿世界 X
        z: [-0.02, 0.02]   # ± 2 cm 沿世界 Z；y 省略 = 不扰动
      inside:
        x: [-0.10, 0.10]
```

最终位姿 = `background_transforms[<key>]` 给出的确定基线位置 + 该 key 下每轴
`uniform([low, high])` 采样得到的随机偏移；orientation 直接复用基线。

### Key 匹配优先级

与 `background_transforms` 一致，外层 key 按下列顺序匹配，第一个命中生效：

1. PLY 完整路径 / `str(Path)` / 文件名 / stem — 单 PLY 精确匹配；
2. 部件名（仅部件字典模式）；
3. 未匹配 → 不随机化。

举例：

```yaml
background_transform_randomization:
  wall:  { x: [-1.0, 1.0] }    # 应用到所有 wall* 部件 PLY 的 x 轴
  wall0: { x: [10.0, 10.0] }   # 但 wall0 单独被覆盖为 +10（确定值）
```

### 采样语义（部件字典模式）：先用满文件，再用随机化补差

部件字典模式下采用 **「文件多样性优先」** 策略，分两阶段填满 batch_size 个组合：

1. **Phase 1：文件笛卡尔积，无随机偏移。**
   先取 `min(batch_size, n_files)` 个不重复的部件 PLY 组合（`n_files = 各部件池子大小之积`），
   走 `_sample_combinations`（batch < n_files 时无放回采样，否则枚举完整笛卡尔积），
   每个组合只烤入 `background_transforms` 给出的*确定*位姿，**完全不应用随机偏移**。
2. **Phase 2：溢出时才用随机化补差。**
   如果 `batch_size > n_files`，剩余 `batch_size - n_files` 个 env 才进入随机化分支：
   按部件分别从池子里有放回地各采一张 PLY，然后叠加一份从
   `background_transform_randomization` 采样的随机偏移，把它们材质化为新的 PLY。
   这样溢出的 env 之间靠随机偏移区分，而不会和 phase-1 的确定组合重复。

直观理解：

| `batch_size` vs `n_files` | 实际行为 |
|---|---|
| `batch_size <= n_files` | 全 batch 用不重复的文件组合，**忽略 `background_transform_randomization`** |
| `batch_size > n_files`  | 前 `n_files` 个 env 用不重复文件组合（无偏移）；剩余 env 复用文件并叠加随机偏移 |

也就是说，*配置了随机化范围 ≠ 一定会用上*。只有当文件池不够覆盖整个 batch 时，
随机化才被启用，目的是「在文件多样性已经用尽时再为额外的 env 引入视觉差异」。

**采样器**：所有随机量（phase-1 的 `_sample_combinations` 索引、phase-2 的池采样、
phase-2 的偏移）都共用 `_setup_gs_render_state()` 传进来的 `np.random.Generator`，
即 env 自带的 `_bg_rng`，固定 seed 时整个解析过程可复现。

**缓存**：偏移烤入 PLY 后由 `_materialize_transformed_background_ply` 落盘缓存到
`.cache/gs_background_transforms/`，cache key 包含位姿 hash —— 不同偏移自然落到不同
缓存条目；phase-1 的确定组合的位姿与现有的 `background_transforms` 缓存共享。

**list / glob 模式**：保持原本的"每条池中 PLY 各采一份偏移"语义不变。
`background_transforms`/randomization 在 list 模式里都是按 *单 PLY* 应用的，
没有"组合"概念，因此不存在 phase-1 / phase-2 划分。

### 与 `randomize_background_on_reset` 的关系

随机偏移在初始化时就被烤入 PLY 并被 GPU renderer 持有；之后 `randomize_background_on_reset=true`
仅会重新洗牌 env→renderer 的索引映射，**不会**重新采样偏移。这意味着：

- 一次初始化后，全 batch 的视觉变体池子是固定的（最多 `batch_size` 种）；
- reset 之间各 env 在这个池子里轮换，但池子本身不变。

如果以后需要每次 reset 都换出全新偏移，会涉及销毁并重建 GPU renderer，开销较大；当前
版本未实现，留作后续扩展。

### 配置校验

`GaussianRenderConfig` 的 model_validator 会拒绝下列错误配置：

- 不在 `{x, y, z}` 中的轴名（避免拼写错误悄悄变成 no-op）；
- 长度不是 2 的范围；
- `low > high`。

## Reset 分配开关

新增配置项：

```yaml
env:
  gaussian_render:
    randomize_background_on_reset: false
```

当前默认值是 `false`。

| 值 | 行为 |
| --- | --- |
| `false` | 初始化时随机分配一次背景；后续 `reset()` 保持这次分配 |
| `true` | 初始化时随机分配一次背景；每次 `reset()` 后重新分配 |

这个开关影响所有多背景模式（list、glob、部件字典）。单背景模式下没有背景池可重新分配。

## Batched 环境的分配规则

`BatchedGSUnifiedMujocoEnv` 为每个环境维护一个背景索引：

```python
_env_bg_idx[env_index] -> background_index
```

当前采样规则：

| 条件 | 采样方式 | 结果 |
| --- | --- | --- |
| `batch_size <= 背景数` | 无放回采样 | 同一次分配中，每个环境背景都不一样 |
| `batch_size > 背景数` | 有放回采样 | 允许重复，因为背景数量不够 |
| 背景数为 0 | 全部索引为 0 | 实际不会有背景 renderer |

因此，当背景数多于或等于环境数时，可以保证同一次初始化或同一次 reset 分配后，每个环境拿到不同背景。

需要注意：

- 如果 `randomize_background_on_reset: true`，每次 reset 会重新分配整个 batch。
- 重新分配不保证某个环境一定不同于自己上一次的背景；它只保证本次 batch 内互不重复，前提是背景数足够。
- 当前 `reset(env_mask=...)` 即使只重置部分环境，也会在开启 `randomize_background_on_reset` 时重分配整个 batch 的背景，而不是只重分配 mask 内的环境。

## 单环境 GS 的分配规则

`GSUnifiedMujocoEnv` 在多背景模式下维护一个 active background：

```python
_active_bg_idx
_gs_renderer = _gs_renderers_list[_active_bg_idx]
_bg_gs_renderer = _bg_gs_renderers_list[_active_bg_idx]
```

行为：

- 初始化时随机选一个 active background。
- `randomize_background_on_reset: false` 时，reset 后继续使用这个背景。
- `randomize_background_on_reset: true` 时，每次 reset 后重新随机选一个背景。

单环境没有“batch 内不重复”的问题。

## 渲染生命周期

多背景模式下，初始化会为所有背景创建 renderer，但不会一次性把所有背景都渲染出来。

初始化阶段会做：

- 解析 `background_ply` 列表。
- 对每个背景应用 transform 并生成 materialized PLY。
- 为每个背景创建 background renderer。
- 初始化一次背景分配。

真正的相机渲染发生在 `capture_observation()` 或工具脚本触发观测时。

### Batched 渲染

`BatchedGSUnifiedMujocoEnv` 渲染背景时会先读取当前 `_env_bg_idx`：

```python
unique_bg_idxs = np.unique(self._env_bg_idx)
```

只会对本次 batch 中实际出现的背景调用 renderer。也就是说：

- 如果背景数大于环境数，多余且未分配到任何环境的背景不会在本次渲染。
- 未分配的背景等到未来某次被选中后，才会真正参与相机渲染。
- 同一个背景如果被多个环境使用，会只对这个背景 renderer 调一次 batch 渲染，然后把结果 scatter 回对应环境。

### 单环境渲染

`GSUnifiedMujocoEnv` 只使用当前 active background 的 renderer。没有被选中的背景不会参与当前帧渲染。

## 背景缓存

batched GS 背景有 `_bg_cache`，用于静态相机复用背景渲染结果。

缓存 key：

```python
(tuple(cam_ids), width, height)
```

当前行为：

- 静态相机使用缓存。
- 动态相机或移动相机传入 `use_cache=False`，每次重新渲染背景。
- 背景重新分配时会清空缓存。
- `set_background_transform(...)` 修改单背景 transform 时会清空缓存。

缓存的是当前 env→background 映射下的背景结果；一旦映射变化，旧缓存不能复用。

## Open Door 当前配置状态

`aao_configs/open_door_airbot_play_gs.yaml` 当前已切到部件字典形式，分别给墙面和室内提供候选：

```yaml
wall_name: wall*
inside_name: inside10
bg3dgs_dir: ${assets_dir}/gs/backgrounds/door_bg/

env:
  gaussian_render:
    background_ply:
      wall:   ${bg3dgs_dir}/${wall_name}.ply
      inside: ${bg3dgs_dir}/${inside_name}.ply
```

每个环境的背景 = 一张 `wall*` PLY 与 `inside10.ply` 的组合（合并后落盘缓存到
`.cache/gs_background_combos/`）。

`aao_configs/test/open_the_door.yaml` 当前测试配置里：

```yaml
env:
  batch_size: 3
```

在 batched 模式下，`BatchedGSUnifiedMujocoEnv` 会传 `max_combinations=batch_size=3`，
即只 materialize 3 个 wall×inside 组合给 3 个环境用，无放回采样保证不重复（前提是
笛卡尔积总数 ≥ batch_size）。

如果保持默认 `randomize_background_on_reset: false`，这 3 个环境的背景会在初始化后固定。若希望每次 reset 都重新抽一组背景，需要显式配置：

```yaml
env:
  gaussian_render:
    randomize_background_on_reset: true
```

## 验证覆盖

相关测试位于 `tests/test_gs_background_transform.py`，覆盖：

- 背景数覆盖 batch 时无重复。
- 背景数等于 batch 时用满所有背景。
- 背景数不足时允许重复。
- 单环境 GS 在 reset 时是否根据开关重选背景。
- batched GS 在 reset 时是否根据开关重分配背景。
- 部件字典模式下 `is_multi_background` 返回 `True`。
- `_merge_background_plys` 拼接结果点数 = 各部件点数之和；二次调用命中磁盘缓存（`save_ply` 不再被触发）；单元素直通源 PLY。
- `_sample_combinations` 在不限和无放回采样两种情况下的行为。
- `resolved_background_plys` 对 dict 输入返回笛卡尔积大小的合并 PLY 列表，`max_combinations` 截断时无重复。
- `background_transforms` 按部件 PLY 路径 / 文件名 / stem 匹配后，先烤入再合并。
