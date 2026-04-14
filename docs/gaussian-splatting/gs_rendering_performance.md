# GS Rendering Performance Optimization

`BatchedGSUnifiedMujocoEnv` 在 `+get_obs=true` 时每步触发大量 GS 渲染，本文档记录已完成的优化和后续可做的优化方向。

## Benchmark 工具

```bash
# 基本测试 (默认关闭 viewer, to_numpy=false)
python examples/bench_env.py press_three_buttons_gs 10

# 指定 batch_size
python examples/bench_env.py press_three_buttons_gs 10 env.batch_size=4

# 带 cProfile 输出
python examples/bench_env.py press_three_buttons_gs 10 --profile env.batch_size=10
```

## 优化总览

### 总体加速效果

| batch_size | 原始 | Phase 1 后 | Phase 2 后 | 当前最优 | 总加速比 |
|---|---|---|---|---|---|
| **2** | 790 ms | 148 ms | 67 ms | **80 ms** | **9.9x** |
| **10** | — | 679 ms | 406 ms | **395 ms** | — |

> 测试环境: press_three_buttons_gs (3 个 mask object, 2 cameras)

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

**节省**: rasterization 调用次数从 N_cam x (1+N_obj) 降为 1+N_obj（3 相机 2 物体: 9->3）

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

**实际提升**: batch_size=2 时 capture_observation 从 **790ms -> 148ms (5.3x)**

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

**实际提升**: batch_size=2 时 capture_observation 从 148ms -> **67ms (2.2x)**

---

## 瓶颈分析 (当前状态)

### batch_size=2 (total ~80ms)

| 耗时 | 占比 | 来源 |
|---|---|---|
| ~35ms | 44% | mask 渲染 (GPU 计算 + 最终 .cpu()) |
| ~17ms | 21% | GS 光栅化 (batch_env_render + batch_update) |
| ~14ms | 18% | torch.as_tensor (body/cam -> GPU) |
| ~8ms | 10% | MuJoCo 观测 (_collect_obs) |
| ~7ms | 9% | 触觉传感器 |

### batch_size=10 (total ~395ms)

| 耗时 | 占比 | 来源 |
|---|---|---|
| ~302ms | 76% | `_render_batched_gs_masks_multicam` |
| ~48ms | 12% | `torch.as_tensor` (body/cam 预转换) |
| ~46ms | 12% | MuJoCo 观测 (_collect_obs, 10 env 串行) |
| ~43ms | 11% | 触觉传感器 (10 env 串行) |
| ~28ms | 7% | GS 光栅化本身 |

**核心瓶颈**: mask 渲染中每个 object 单独调一轮 `batch_update_gaussians` + `batch_env_render`，3 个 button = 3 次 GPU 往返。光栅化本身(28ms)只占 7%，绝大部分时间在 gaussian 更新和 GPU sync。

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

---

## 后续优化方向

### A. 合并多 mask object 为单次渲染 (预估 -50~60% mask 时间)

当前每个 mask object 独立渲染（3 objects = 3 次 `batch_update_gaussians` + 3 次 `batch_env_render`）。可以将所有 mask objects 的 gaussian 合并到一个 renderer 中，一次渲染得到所有 object 的 alpha/depth，然后按 point_to_body_idx 拆分回各 object 的 mask。

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
