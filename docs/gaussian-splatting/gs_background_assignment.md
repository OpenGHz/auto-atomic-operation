# GS Multi-Background Assignment

本文档记录当前 Gaussian Splatting 多背景机制的完整现状，包括配置项、batch 环境背景分配、reset 行为、渲染时机和缓存策略。

相关实现：

- `auto_atom/basis/mjc/gs_mujoco_env.py`
- `GaussianRenderConfig`
- `GSUnifiedMujocoEnv`
- `BatchedGSUnifiedMujocoEnv`

## 配置入口

`env.gaussian_render.background_ply` 现在支持三种形式：

| 形式 | 含义 |
| --- | --- |
| `null` | 不使用 GS 背景 |
| `str` | 单个背景路径，所有环境共享 |
| `str` 含 glob 元字符（`*`、`?`、`[...]`） | 单条字符串展开为多背景池 |
| `list[str]` | 多背景池；列表中的任意一项也可以是 glob |

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

## Transform 规则

`background_transform` 和 `background_transforms` 的优先级如下：

| 优先级 | 配置 | 行为 |
| --- | --- | --- |
| 1 | `background_transforms` | 按完整路径、字符串路径、文件名或 stem 匹配，给单个背景单独指定 transform |
| 2 | `background_transform` | 默认 transform；当 `background_ply` 是 list 时会应用到每个背景 |
| 3 | identity | 未配置时使用零平移和单位四元数 |

`background_transform` 支持：

- 长度 3：`[x, y, z]`
- 长度 7：`[x, y, z, qx, qy, qz, qw]`

当 `background_ply` 是 list 时，`set_background_transform(...)` 不支持运行时修改整组背景；需要通过配置里的 `background_transform` 或 `background_transforms` 设置。

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

这个开关只影响 `background_ply` 为 list 的多背景模式。单背景模式下没有背景池可重新分配。

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

`aao_configs/open_door_airbot_play_gs.yaml` 当前使用 `door_bg` 目录下的多个背景：

```yaml
bg3dgs_dir: ${assets_dir}/gs/backgrounds/door_bg/

env:
  gaussian_render:
    background_ply:
      - ${bg3dgs_dir}/bg0.ply
      - ${bg3dgs_dir}/bg1.ply
      - ${bg3dgs_dir}/bg2.ply
      - ${bg3dgs_dir}/bg3.ply
      - ${bg3dgs_dir}/bg4.ply
      - ${bg3dgs_dir}/bg5.ply
      - ${bg3dgs_dir}/bg6.ply
```

`aao_configs/test/open_the_door.yaml` 当前测试配置里：

```yaml
env:
  batch_size: 3
```

在 7 个背景、3 个环境的情况下，同一次分配会给 3 个环境分到互不重复的背景。

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
