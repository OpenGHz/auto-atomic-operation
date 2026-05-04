# GS Rendering Performance Optimization

`BatchedGSUnifiedMujocoEnv` 在 `+get_obs=true` 时每步触发大量 GS 渲染，本文档记录已完成的优化和后续可做的优化方向。

## Benchmark 工具

```bash
# 无头环境下确保 EGL 可用
export MUJOCO_GL=egl

# 基本测试 (默认关闭 viewer, to_numpy=false)
python examples/bench_env.py open_door_airbot_play_back_gs 30 +test=open_the_door

# 指定 batch_size
python examples/bench_env.py open_door_airbot_play_back_gs 30 +test=open_the_door env.batch_size=4

# 带 cProfile 输出
python examples/bench_env.py open_door_airbot_play_back_gs 10 --profile +test=open_the_door env.batch_size=2
```

> 注：`+test=open_the_door` 会显式设置 `env.to_numpy=true / structured=false`。bench_env.py 默认注入 `+env.to_numpy=false …` 与之冲突时，需要把 bench 默认改为 `++env.to_numpy=false …`（或在命令行用 `++` 显式覆盖），让 to_numpy=false 的"GPU 直留"路径生效。

---

## Benchmark (`open_door_airbot_play_back_gs +test=open_the_door`, 2026-05-04)

> 测试硬件：NVIDIA GeForce RTX 5090（32 GB），驱动 590.48.01 / CUDA 13.1
> 测试命令：`python examples/bench_env.py open_door_airbot_play_back_gs N +test=open_the_door env.batch_size=B`
> bench 默认注入 `++env.viewer.disable=true ++env.to_numpy=false ++env.structured=false`；warmup 不计入统计；测量 `capture_observation` 与 `update` 的纯环境时间。

### 当前配置快照（带 `+test=open_the_door` 覆盖）

| 项 | 值 |
|---|---|
| `BatchedGSUnifiedMujocoEnv` | `share_physics=false`（默认） |
| 相机 | 2 路：`eef_wrist_cam`（动态，挂在末端）+ `env2_cam`（静态） |
| 分辨率 | **640 × 480**（来自 `test/open_the_door.yaml` 的 `cam_width / cam_height`） |
| 单 GS 相机功能 | `enable_color=true`，`enable_depth / enable_mask / enable_heat_map=false` |
| `mask_objects` | `["handle_lever_body"]` — 该物体没有 GS body 对应（启动日志：`Skipping GS mask renderer for 'handle_lever_body'`），`_gs_mask_renderers` 为空，binary mask / heat_map 路径全部跳过 |
| `gaussian_render.background_ply` | `${bg3dgs_dir}/bg*.ply` — 当前资产目录命中 14 张 `bg{0..13}.ply`，每张 ~1.17 M 点；`is_multi_background=True` |
| 前景 GS 资产 | door + handle + lock + airbot_play (7 link) + airbot_g2p (18 link) |
| GPU 峰值显存（batch=2） | **~1.71 GB allocated / ~2.02 GB reserved** |

### 渲染流程图（按当前配置裁剪）

```
capture_observation()
│
├─► super().capture_observation()            （父类 BatchedUnifiedMujocoEnv，串行 N 个 env）
│   for env in self.envs (N):
│     env.capture_observation()              MuJoCo 原生 obs：pose / joint / tactile
│                                            （GS 相机的 color/depth/mask/heat_map 都已被
│                                             setup_gs_cameras 关闭，原生通道不再渲染）
│
└─► _inject_batched_gs_renders(obs)          BatchedGSUnifiedMujocoEnv
    │
    ├─ body_pos, body_quat = np.stack(env.data.x{pos,quat})      （N, Nbody, 3/4）
    ├─ fg_gsb = _fg_gs_renderer.batch_update_gaussians(body_pos, body_quat)
    │                                                            ── 1 次 GPU 上传 + transform
    │
    ├─ 按 (H, W, is_static) 分组相机：
    │   Group A: eef_wrist_cam   (480, 640, dynamic)   — 不缓存背景
    │   Group B: env2_cam        (480, 640, static)    — 背景结果首帧后落 _bg_cache
    │
    └─ 对每个 group：
        ├─ 收集 cam_pos, cam_xmat, fovy                          （N, Ncam, …）
        │
        ├─ _render_batched_multicam(...)
        │   │
        │   ├─ _get_cached_bg_multicam(use_cache=is_static)
        │   │   │
        │   │   ├─ static + 命中缓存 → 返回 (bg_rgb, bg_depth)
        │   │   │
        │   │   └─ 否则（动态相机 / 首帧 static）：
        │   │      多背景路径：每个 unique bg 各跑 1 次 batch_update + batch_env_render
        │   │
        │   ├─ fg_rgb, fg_depth = _fg_gs_renderer.batch_env_render(
        │   │       fg_gsb, cam_pos, cam_xmat, H, W, fovy, bg_imgs=bg_rgb)  ── 1 次光栅化
        │   └─ full = fg * α + bg * (1 − α)                                  GPU elementwise
        │
        └─ 分发：
            color  → torch.clamp(0,1)·255 → uint8（保留在 GPU）
            mask / heat_map：_gs_mask_renderers 为空 → 直接跳过
```

稳态光栅化次数随 batch 中 unique 背景数变化。下面 torch.profiler 在 batch=2 上录到 `_RasterizeToPixels` **40 calls / 10 iter = 4 calls/iter**，对应：1 (Group A FG) + 2 (Group A BG，dynamic，2 个 unique bg) + 1 (Group B FG，BG 已缓存)。

### 端到端 wall-clock 时间

| batch_size | iters | capture_observation (mean / std / min) | update (mean) | total | 频率 |
|---|---|---|---|---|---|
| 1 | 30 | **12.89 / 0.13 / 12.57 ms** | 0.40 ms | 13.29 ms | 75.26 Hz |
| 2 | 30 | **15.81 / 0.17 / 15.35 ms** | 0.79 ms | 16.60 ms | 60.24 Hz |
| 4 | 30 | **20.00 / 0.33 / 19.38 ms** | 1.16 ms | 21.15 ms | 47.27 Hz |
| 8 | 20 | **31.83 / 0.50 / 30.83 ms** | 2.97 ms | 34.79 ms | 28.74 Hz |

`capture_observation` 从 b=1 到 b=8 接近线性放大（~13 → 32 ms，~2.5×），物理 `update` 也线性 N 但量级很小（<3 ms 直到 b=8）。同任务在 RTX 3070 Laptop 上 b=2 ≈ 44 ms，5090 上 ≈ 17 ms，**~2.6× 的端到端加速**主要来自 5090 更高的 SM 数与 rasterization 吞吐。

### torch.profiler（batch_size=2，10 iter 计时窗口）

| Kernel / op | self CUDA | 占 CUDA 总时 | 调用次数 | 单次均值 |
|---|---|---|---|---|
| `gsplat::rasterize_to_pixels_3dgs_fwd_kernel` (`_RasterizeToPixels`) | **118.5 ms** | **81.5 %** | 40 | 2.96 ms |
| `cudaPeekAtLastError` | 9.94 ms | 6.83 % | 880 | 11 µs |
| `cudaLaunchKernel` (CUDA-side) | 9.93 ms | 6.83 % | 3120 | 3 µs |
| `aten::copy_`（H2D 上传 body_pos/body_quat/cam 等） | 8.02 ms | 5.51 % | 880 | 9 µs |
| 元素级 `vectorized_elementwise_kernel`（α-blend 主成分） | 7.03 ms | 4.83 % | 280 | 25 µs |
| `cub::DeviceRadixSortOnesweep`（gaussian sort by depth） | 3.53 ms | 2.42 % | 240 | 15 µs |
| `_SphericalHarmonics` | 2.83 ms | 1.95 % | 40 | 71 µs |
| `gsplat::intersect_tile_kernel`（tile binning） | 1.15 ms | 0.79 % | 80 | 14 µs |
| `_FullyFusedProjection` | 1.02 ms | 0.70 % | 40 | 25 µs |

CPU 侧 self time 第一名是 `cudaStreamSynchronize`（**118.1 ms / 400 calls，62.7 %**），说明 Python/CPU 仍有大量时间在等 GPU 完成。CUDA total 145.5 ms / CPU total 188.3 ms，10 iter 平均 14.5 ms / iter，与 wall-clock 的 15.81 ms / iter 数量级吻合。

完整 trace 落在 `outputs/bench/profiles/open_door_airbot_play_back_gs_b2/`；脚本：`examples/profile_gs_obs.py`。

### 结论

1. **光栅化绝对主导**：`_RasterizeToPixels` 占 CUDA self time **81 %**（5090 上 2.96 ms/call × 4 calls/iter ≈ 11.8 ms），是单一最大瓶颈。要继续大幅降时间必须降点数（背景压缩 / 前景瘦身）或降分辨率，代码层面优化收益有限。
2. **alpha-blend / 数据上传是次要项**：α-blend 类 elementwise kernel 7 ms（5 %），H2D `aten::copy_` 8 ms（5.5 %），合计 ~10 % CUDA。
3. **多背景路径放大渲染次数**：`bg*.ply` 命中 14 张 PLY，`is_multi_background=True`；动态相机每帧需为每个 unique 背景各跑 1 次 BG 渲染，比单一背景配置多 1~N 次光栅化。
4. **mask / heat_map 在该配置下为空**：`handle_lever_body` 没有 GS body，整条 mask 子流程被跳过。
5. **batch 缩放线性、显存非常宽裕**：b=2 峰值 ~1.71 GB allocated，5090 的 32 GB 显存能轻松跑更大 batch；瓶颈出现在 GS 渲染时间而非显存。

### 复现步骤

```bash
# 端到端 wall-clock（4 个 batch_size 串跑）
for b in 1 2 4 8; do
  python examples/bench_env.py open_door_airbot_play_back_gs 30 +test=open_the_door env.batch_size=$b
done

# torch.profiler trace（含 chrome://tracing 用的 JSON）
python examples/profile_gs_obs.py open_door_airbot_play_back_gs 10 +test=open_the_door env.batch_size=2
```

`examples/profile_gs_obs.py` 内置 `wait=1, warmup=1, active=N` 的 schedule（见
[torch.profiler.schedule](https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule)），
排除 gsplat CUDA JIT 与首帧填缓存对均值的污染；trace 落到
`outputs/bench/profiles/<config>_b<batch>/` 下，可 tensorboard 或 chrome://tracing 直接打开。

---

## Phase 1: 早期优化 (已完成)

### 1. 合并 color + depth 为单次 FG 渲染
当相机同时需要 color 和 depth/mask 时，之前会分别调用 `batch_env_render`——一次取 RGB，一次取 depth。现在统一走 `_render_batched_multicam` 一次渲染，从同一结果中分别提取 `full_rgb` 和 `full_depth`。

**节省**: 每步 N_cam 次 FG 渲染（3 相机 -> 省 3 次）

### 2. 缓存 background depth
`_bg_cache` 现在同时缓存 `(bg_rgb, bg_depth)`，`_render_batched_multicam` 直接使用缓存而非重新渲染背景。

**节省**: 每步 ~N_cam 次 BG 渲染

### 3. 禁用 GS 相机的原生 MuJoCo 分割渲染
`GSEnvConfig` validator 现在对 GS 相机自动关闭 `enable_mask` / `enable_heat_map`（分别通过 `gs_mask_cameras` / `gs_heat_map_cameras` 跟踪），避免原生 MuJoCo segmentation 渲染被 GS mask 覆盖的浪费。

**节省**: 每步 N_cam x batch_size 次原生 seg 渲染

### 4. 多相机批量渲染 (multi-camera batching)
`batch_env_render` 支持 `Ncam > 1`，可在单次 GPU kernel 调用中渲染所有相机。将逐相机循环改为一次性传入 `(Nenv, Ncam, ...)` 张量，FG 和每个 mask object 各只需一次 `batch_env_render` 调用。

**节省**: rasterization 调用次数从 N_cam x (1+N_obj) 降为 1+N_obj

---

## Phase 2: GPU 端 mask 计算与传输优化 (已完成)

### 5. Mask 遮挡计算全部留在 GPU

**优化前** (`_render_batched_gs_masks_multicam`):
```
per-object loop:
  alpha_t (GPU) -> .cpu().numpy() -> np.max(axis=-1)         # ~66ms/call
  obj_depth (GPU) -> .cpu().numpy() -> np.nan_to_num()       # ~33ms/call
  scene_depth (GPU) -> .cpu().numpy() -> np.nan_to_num()     # 每帧重复
  numpy boolean ops -> binary_mask (CPU)
```

**优化后**:
```
scene_depth: torch.nan_to_num() on GPU (一次)
per-object loop:
  alpha_max = alpha_t.max(dim=-1) on GPU                     # GPU torch.max
  obj_depth = torch.nan_to_num() on GPU                      # GPU torch op
  visible = GPU boolean ops                                   # GPU 比较
binary_mask = result.to(uint8).cpu().numpy()                  # 一次性传回
```

**关键变化**:
- `np.max` -> `torch.max` (消除 3.98s/60calls 的热点)
- `np.nan_to_num` -> `torch.nan_to_num` (消除 0.66s/80calls)
- 多次 `.cpu()` 合并为一次最终传输
- `visible[occluded] = False` -> `visible & ~occluded` (纯 tensor 操作)
- heat_map 内循环的 `list.index()` 替换为预构建的 dict lookup

**实际提升**: 早期含 mask 渲染场景的端到端 capture_observation 显著下降（数量级 ~5×），mask 路径不再是瓶颈。

### 6. Mask 渲染共享输入预转换

mask 循环中每个 object 都调用 `batch_update_gaussians(body_pos, body_quat)` 和 `batch_env_render(cam_pos, cam_xmat)`，内部每次都执行 `torch.tensor(numpy)` 转换。

**优化**: 在循环前一次性 `torch.as_tensor()` 转 GPU tensor，循环内传 tensor 跳过转换。

**实际效果** (batch_size=10):

| 指标 | 优化前 | 优化后 |
|---|---|---|
| `torch.tensor()` 调用 | 430 次 / 1038ms | **消除** |
| `torch.as_tensor()` | — | 80 次 / 480ms |
| `.cpu()` 调用次数 | 100 次 | **40 次** |

### 7. Benchmark 默认关闭 viewer 和 to_numpy

`bench_env.py` 自动注入以下 Hydra overrides:
- `+env.viewer.disable=true` — 关闭 MuJoCo viewer 窗口
- `+env.to_numpy=false` — color/depth 留在 GPU，省去 `.cpu()` 传输

**实际提升**: 关闭 viewer + `to_numpy=false` 让 color/depth 留在 GPU，省掉若干次 `.cpu()`，bench 比起原始默认快约 **2×**。

---

## Benchmark 结果 (cup_on_coaster_gs, 2026-04-10)

> 测试环境: cup_on_coaster_gs (2 mask objects, 3 cameras 1280x720, color+depth+heat_map)
> 测试工具: `tests/run_bench_suite.py` + `tests/plot_bench_results.py`
> 注意事项: task-level 第一步作为 warmup 不纳入计时，避免 CUDA JIT 编译影响结果

### Task-Level 循环频率

| batch_size | task update | task update + obs | obs 带来的每步开销 | env 吞吐量 |
|---|---|---|---|---|
| 1 | 110.9 Hz (9.0 ms) | 11.9 Hz (84.0 ms) | 75.0 ms | 11.9 env-step/s |
| 2 | 49.2 Hz (20.3 ms) | 8.4 Hz (119.0 ms) | 98.7 ms | 16.8 env-step/s |
| 4 | 24.7 Hz (40.5 ms) | 3.9 Hz (256.4 ms) | 215.9 ms | 15.6 env-step/s |
| 8 | 12.2 Hz (82.0 ms) | 1.9 Hz (526.3 ms) | 444.3 ms | 15.2 env-step/s |

### Env-Level 分解 (capture_observation + update 紧凑循环)

| batch_size | capture | update | total | capture 占比 |
|---|---|---|---|---|
| 1 | 56.7 ms | 1.28 ms | 57.98 ms | 97.8% |
| 2 | 92.95 ms | 2.44 ms | 95.39 ms | 97.4% |
| 4 | 245.66 ms | 4.84 ms | 250.50 ms | 98.1% |
| 8 | 466.61 ms | 10.18 ms | 476.79 ms | 97.9% |

### GPU 显存

所有 batch_size 下峰值显存均为 **514 MB**，显存不随 batch_size 增长（受 GS model 大小主导）。

### 关键结论

1. **observation 获取是性能瓶颈**: capture_observation 占总耗时 >97%，物理 update 不到 3%
2. **batch 吞吐量**: batch_size=2 时 env 吞吐最优 (16.8 env-step/s)，更大 batch 因 GS 渲染线性增长而吞吐趋于饱和 (~15 env-step/s)
3. **task-level vs env-level 一致性**: task-level 的 obs 开销 ≈ env-level capture + ~10-18ms (runner 逻辑 + CUDA 同步开销)，两组数据吻合
4. **warmup 重要性**: 首次 capture_observation 触发 gsplat CUDA JIT 编译（耗时数十秒），必须排除在计时之外。task-level 和 env-level 均已实现 warmup 机制

### 启用 env 内置 warmup

`GSEnvConfig`（以及 `BatchedGSUnifiedMujocoEnv`）支持一个开关式的 warmup，会在初始化结尾自动跑一次 `reset() + capture_observation()`，用于：

- 提前触发 `gsplat` 的 CUDA JIT 编译（第一次渲染耗时数十秒）
- 消除前若干帧 observation 的离群值，降低首帧抖动

```yaml
env:
  gaussian_render:
    warmup: true
```

默认 `false`。开启后初始化日志中会出现：

```
GS renderer initialised with N body gaussian(s)[ + background]
Performing GS renderer warmup...
GS renderer warmup complete.
```

任何对首帧延迟敏感的脚本（数据采集、policy eval、bench）建议显式开启；在 bench 计时时 warmup 仍应排除在计时区间之外。

---

## 场景间对比 (open_door vs cup_on_coaster, 2026-04-19)

> 测试环境:
> - `open_door_airbot_play_back_gs env.batch_size=1 +test=open_the_door`
> - `cup_on_coaster_gs env.batch_size=1`
> - 两组测试都先 `source env.sh`
> - `bench_env.py` 默认注入 `+env.viewer.disable=true +env.to_numpy=false +env.structured=false`

### Benchmark 结果

| 配置 | capture_observation | update | total | total 频率 |
|---|---|---|---|---|
| `open_door_airbot_play_back_gs +test=open_the_door` | **23.06 ms** | 0.35 ms | **23.42 ms** | **42.70 Hz** |
| `cup_on_coaster_gs` | **9.75 ms** | 0.77 ms | **10.52 ms** | **95.03 Hz** |

`open_door` 的总耗时约为 `cup_on_coaster` 的 `2.23x`，差异几乎全部来自 `capture_observation`；物理 `update()` 不是主要瓶颈。

### cProfile 对比

| 热点 | `open_door` | `cup_on_coaster` | 比值 |
|---|---|---|---|
| `_inject_batched_gs_renders` | 21.1 ms / iter | 6.7 ms / iter | 3.15x |
| `batch_update_gaussians` | 19.9 ms / iter | 4.2 ms / iter | 4.74x |
| `torch.tensor` | 18.6 ms / iter | 3.7 ms / iter | 5.03x |

从 profile 看，差异主要在 GS 渲染前的数据准备阶段，而不是光栅化本身。`open_door` 更慢的核心原因是:

1. `batch_update_gaussians()` 更重
   - `open_door` 的时间大量花在 `gaussian_renderer.batch_splat.batch_update_gaussians()`。
   - 该路径内部会反复把 `body_pos/body_quat/cam_pos/cam_xmat/fovy` 以及 gaussian template 从 `numpy` 包装为 `torch.Tensor`。
   - `torch.tensor` 本身就占了 `18.6 ms / iter`，已经接近 `open_door` 单次 `capture_observation` 的大头。

2. GS 资产规模更大
   - `open_door` 前景高斯点数约 `933k`
     - 门 + 把手: `144,834`
     - Airbot Play + G2P 机器人: `788,549`
   - `cup_on_coaster` 前景高斯点数约 `396k`
     - cup + coaster: `379,888`
     - Robotiq: `16,016`
   - 比值约 `2.36x`，与 `capture_observation` 的 `2.36x` 慢速几乎一致。

3. 背景高斯点数也更大
   - `open_door` 背景 `bg0.ply`: `1,177,447` points
   - `cup_on_coaster` 背景 `background_1.ply`: `624,772` points
   - 背景规模约 `1.88x`

### 不是主要原因的项

1. 不是相机分辨率差异
   - 两组测试都使用 `640x352`。

2. 不是相机数量差异
   - `open_door +test=open_the_door` 只保留了 1 个静态相机 `env2_cam`
   - `cup_on_coaster_gs` 默认继承 3 个相机: `wrist_cam`, `env1_cam`, `env0_cam`
   - 即使 `cup` 的相机更多，它仍然明显更快，说明主因不是相机数量，而是每次 GS 更新的数据规模。

3. 不是 mask / heatmap 渲染
   - 两组测试都是 `enable_mask=false`, `enable_heat_map=false`
   - `open_door` 还打印了 `Skipping GS mask renderer for 'handle_lever_body'`，说明这次 benchmark 中 mask 路径没有成为瓶颈。

4. 不是物理 update
   - `open_door` 的 `update()` 反而更快: `0.35 ms` vs `0.77 ms`
   - 这与配置一致: `open_door` 使用 `sim_freq=1000, update_freq=100`，而 `cup_on_coaster` 继承默认 `sim_freq=600, update_freq=30`，单次 `update()` 的物理子步更多。

### 结论

`open_door_airbot_play_back_gs` 比 `cup_on_coaster_gs` 慢，主因不是 camera 配置，而是 GS 场景本身更重，尤其是 Airbot+G2P 机器人和 door background 带来的更大 gaussian 数量，进一步放大了 `batch_update_gaussians()` 中的 `torch.tensor(numpy -> cuda)` 开销。

因此，若要继续优化 `open_door` 场景，优先级应为:

1. 缓存 gaussian template 的 GPU tensor，避免每帧重复 `torch.tensor()`
2. 让 `body_pos/body_quat/cam_pos/cam_xmat/fovy` 在进入 renderer 前就完成 tensor 化
3. 在不影响效果的前提下，优先压缩 Airbot+G2P 和 door background 的 GS 资产规模

---

## 分辨率反直觉现象 (open_door, 2026-04-19)

在 `open_door_airbot_play_back_gs +test=open_the_door` 上观察到一个反直觉现象:

- 将相机分辨率从 `320x240` 提高到 `640x480` 后，端到端帧率不是下降，而是多次测试中整体呈上升趋势。
- 用户在完整 `airdc` 流程中多次复测后确认，该趋势是稳定可复现的，而不是单次抖动。
- 在 `bench_env.py` 的纯环境测试中也能复现同方向结果，说明这不只是视频编码链路或外层 runner 的偶然现象。

### 当前结论

当前 `open_door` 配置下，瓶颈并不主要由像素数量决定，而更接近:

1. 每帧固定开销主导
   - `batch_update_gaussians()`
   - `torch.tensor(numpy -> cuda)`
   - 相机/物体位姿整理与同步

2. GPU 利用率在低分辨率下可能更差
   - `320x240` 时，GS rasterization 可能没有把 GPU 跑满，kernel launch / 同步 / 调度固定成本占比更高。
   - `640x480` 时，虽然像素更多，但并行度更高，反而可能让 GPU 落在更高效的执行区间。

3. 当前场景更像是 fixed-overhead / sync bound，而不是 pixel-fill-rate bound
   - 因此简单降低分辨率，不保证更快。
   - 在该场景中，低分辨率反而可能让固定成本占比进一步放大。

### 实践含义

对于当前 `open_door` 的 GS 观测链路:

- 不应默认认为“降低分辨率一定提升 FPS”。
- 在优化前，应先实测不同分辨率，而不是仅凭像素数做判断。
- 现阶段更值得优先优化的仍然是 `batch_update_gaussians()` 前后的 tensor materialization 和 GPU 同步点，而不是单纯下调分辨率。

### 说明

这一现象已经通过多次端到端运行得到确认，但其底层根因仍应通过 GPU-aware profiling 进一步验证，例如:

- `torch.cuda.Event`
- `torch.profiler`
- `nsys` / Nsight Systems

也就是说，当前可以把“`640x480` 在此场景中整体更快”视为一个稳定经验结论；但若要精确解释是 tile 利用率、kernel 选路还是同步点迁移导致，还需要更细的 GPU 时间线分析。

---

## VRAM / 速度调参：`minibatch`

`env.gaussian_render.minibatch`（默认 `512`）控制每次 gsplat rasterization 的
minibatch 大小，会透传给所有由 GS env 构造的 `BatchSplatConfig`：前景渲染器、
背景渲染器（每张背景一份）、以及每个 object 的 mask 渲染器。

| 取值方向 | 效果 |
|---|---|
| 调大（如 1024 / 2048） | 每次 kernel launch 处理更多 gaussian，减少 launch 次数；**显存占用变高**，适合 batch_size 小但单帧 gaussian 多（例如 `open_door` 这种 ~900k 前景）的场景 |
| 调小（如 256 / 128） | 降低单次显存峰值；吞吐可能下降 |
| 保持默认 `512` | 大多数场景下够用 |

该值完全是一个性能/显存旋钮，不影响渲染数值结果；遇到 CUDA OOM 时首选调小此值，
再考虑 [Phase 3 share_physics](#phase-3-共享物理模式-2026-04-20) 或降低 `batch_size`。

## Phase 3: 共享物理模式 (2026-04-20)

`BatchedGSUnifiedMujocoEnv` 新增 `gaussian_render.share_physics` 开关，专门针对「batch 各 env 仅背景不同、其余物理状态完全一致」的场景随机化用法。开启后：

- 只创建 **1 个** `UnifiedMujocoEnv` 物理副本；
- 每个 step 物理只更新一次；
- 前景 Gaussian 的 `batch_update_gaussians` + `batch_env_render` 也只跑一次（`nenv=1`）；
- 各 unique 背景各渲染一次（`nenv=1`），按 `_env_bg_idx` 映射 scatter 到 `(N, Ncam, H, W, C)`；
- 前景/背景的合成在 Python 侧用 PyTorch 广播完成（`fg * α + bg * (1-α)`），合成结果自然带 `N` 的 batch 维。

### 启用方式

```yaml
env:
  _target_: auto_atom.basis.mjc.gs_mujoco_env.BatchedGSUnifiedMujocoEnv
  config:
    batch_size: 10
    gaussian_render:
      background_ply: ["bg1.ply", "bg2.ply", ...]   # 或 glob
      share_physics: true
```

Config validator 会强制要求：
- `batch_size > 1`（否则退化为单 env，无意义）；
- `background_ply` 为多背景（list / glob / 部件字典；单背景会产生 N 张完全一样的观测，开 share_physics 无收益）。

### 对外 API 兼容性

- `env.envs` 仍提供 N 个别名（`self.envs = [env_0] * N`），外部代码（如 `mujoco_backend.home()`）里的 `self.env.envs[env_index]` 依然可用。
- 所有 getter（`get_body_pose` 等）返回 `(N, …)`：父类 `np.stack` 在别名列表上自然得到 N 行相同数据。
- 热路径的 step-like 方法（`step`、`update`、`apply_joint_action`、`apply_pose_action`、`reset`、`capture_observation`）被重写为「仅在 env_0 上执行一次」，避免父类 `for env in self.envs` 在别名列表上产生 N× 冗余。
- `env_mask` 在 shared 模式下按 `env_mask.any()` 合并——部分子 env reset/step 在共享物理里没有意义。

### Benchmark (open_door_airbot_play_gs, 2026-04-20)

> 同一硬件 / 同一任务，仅切换 `share_physics`；bench 命令：`python examples/bench_env.py env.batch_size=X [+env.gaussian_render.share_physics=true]`
> `open_door_airbot_play_gs` 的背景是 14 张 `bg*.ply`（每张 ~1.17 M 点），相机为静态。
> shared 路径复用 `_bg_cache`，每种 unique 背景首帧渲染后缓存 `(N, Ncam, H, W, C)` 张量，后续帧直接索引。

| batch_size | 模式 | capture_observation (mean / min) | update | total | 频率 |
|---|---|---|---|---|---|
| 4 | baseline | 77.60 / 75.42 ms | 1.66 ms | 79.26 ms | 12.62 Hz |
| 4 | `share_physics=true` | **40.54 / 20.91 ms** | **0.44 ms** | **40.98 ms** | **24.67 Hz** |
| 10 | baseline | **CUDA OOM**（scene + 10 份物理/FG 超出 8 GB 卡容量） | — | — | — |
| 10 | `share_physics=true` | **57.94 / 37.75 ms** | 0.45 ms | 58.39 ms | 17.13 Hz |

关键观察：

1. **物理** `update` 稳定降到 `~0.45 ms`（不随 N 增长，真正跑的只有 1 个 env）。
2. **capture_observation** 在 batch=4 下 ~48%，batch=10 对比"baseline 单 env 57 ms"也只有 ~2% 开销。
3. **N 维只留合成**：batch=10 vs batch=4 的 capture 时间差只有 ~17 ms，全部是 `fg*α + bg*(1-α)` 这类 `(N, Ncam, H, W, C)` 广播 kernel 的线性开销；背景渲染被 cache 掉了。
4. **显存**：开启后显存占用不随 N 增长（fg gaussian buffer 从 `(N, …)` 降到 `(1, …)`），baseline 在 batch=10 OOM 的场景，shared 模式下可轻松跑通。
5. **min 值远低于 mean**：首帧在填 `_bg_cache`，后续帧大部分落在 `min` 附近。意味着稳定态性能比 mean 显示得还要好。

### 什么时候该用

- 训练阶段只想对「背景贴图」做随机化、其余物理一致（视觉域随机化/ sim-to-real）：典型收益。
- 需要在同一 GPU 上跑更大虚拟 batch 以扩展数据多样性：显存不随 N 增长。

### 什么时候不该用

- 各子 env 初始状态/扰动不同（如不同物体初始位置）——shared 模式下它们会全部归一到 env_0 的状态。
- 需要不同的 action/teleport 到不同 env：shared 模式只取 `[0]`，其他会被丢弃。

### 实现要点

- `auto_atom/basis/mjc/gs_mujoco_env.py`:
  - `GaussianRenderConfig.share_physics` 字段 + `GSEnvConfig.setup_gs_cameras` 里的 validator；
  - `BatchedGSUnifiedMujocoEnv.__init__`：`model_copy(update={"batch_size": 1})` 后传给父类，再把 `self.batch_size` 恢复成虚拟 N，并 alias `self.envs`；
  - `_inject_shared_gs_renders` / `_render_shared_per_env_backgrounds` / `_render_shared_gs_masks_multicam`：`nenv=1` 的前景 + 按 unique bg 分组的 BG 渲染 + 广播合成；
  - step/update/apply_*/reset/capture_observation 的 shared 分支只动 `envs[0]`。

---

## 后续优化方向

### A. 合并多 mask object 为单次渲染 (预估 -50~60% mask 时间)

当前每个 mask object 独立渲染（N_obj objects = N_obj 次 `batch_update_gaussians` + N_obj 次 `batch_env_render`）。可以将所有 mask objects 的 gaussian 合并到一个 renderer 中，一次渲染得到所有 object 的 alpha/depth，然后按 point_to_body_idx 拆分回各 object 的 mask。

**挑战**: 需要修改 `gaussian_renderer` 库，增加 per-object alpha 输出通道。

### B. 线程并行原生 obs 采集 (预估 batch_size=10 时 -30ms)

```python
# mujoco_env.py:1667
obs_per_env = [env.capture_observation() for env in self.envs]
```

各 env 串行执行。MuJoCo 渲染期间释放 GIL，可用 `ThreadPoolExecutor` 并行。batch_size=10 时 ~46ms -> ~10ms。

### C. 触觉传感器可选关闭 (预估 batch_size=10 时 -43ms)

触觉传感器 `get_data()` 每个 env 耗时 ~4ms，10 env 串行 = 43ms。如果 policy 不需要触觉数据，可通过配置关闭。

### D. 跳过 GS 接管相机的 `update_scene` (预估 -5~10%)

`super().capture_observation()` 中 `_collect_obs` 对每个相机都调用 `renderer.update_scene()`。如果某相机的 color/depth/mask/heat_map 全部被 GS 接管，这个调用完全浪费。可在相机循环中加 early-continue 判断。

### E. 静态物体 mask 缓存 (取决于场景)

不动的物体每步的 mask 渲染结果不变，可通过比较 body pose 变化来跳过不必要的 mask 重渲染。

### F. 修正 wrist_cam 背景缓存

`_bg_cache` 的 key 是 `(cam_ids, width, height)`，但 `wrist_cam` 挂载在末端执行器上，每步位置变化。当前实现对动态相机传入 `use_cache=False` 每帧重渲。可考虑对 wrist_cam 使用低分辨率背景或近似缓存。
