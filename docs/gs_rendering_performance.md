# GS Rendering Performance Optimization

`BatchedGSUnifiedMujocoEnv` 在 `+get_obs=true` 时每步触发大量 GS 渲染，本文档记录已完成的优化和后续可做的优化方向。

## 已完成的优化

### 1. 合并 color + depth 为单次 FG 渲染
当相机同时需要 color 和 depth/mask 时，之前会分别调用 `batch_env_render`——一次取 RGB，一次取 depth。现在统一走 `_render_batched_camera` 一次渲染，从同一结果中分别提取 `full_rgb` 和 `full_depth`。

**节省**: 每步 N_cam 次 FG 渲染（3 相机 → 省 3 次）

### 2. 缓存 background depth
`_bg_cache` 现在同时缓存 `(bg_rgb, bg_depth)`，`_render_batched_camera` 直接使用缓存而非重新渲染背景。

**节省**: 每步 ~N_cam 次 BG 渲染

### 3. 禁用 GS 相机的原生 MuJoCo 分割渲染
`GSEnvConfig` validator 现在对 GS 相机自动关闭 `enable_mask` / `enable_heat_map`（分别通过 `gs_mask_cameras` / `gs_heat_map_cameras` 跟踪），避免原生 MuJoCo segmentation 渲染被 GS mask 覆盖的浪费。

**节省**: 每步 N_cam × batch_size 次原生 seg 渲染

### 4. 多相机批量渲染 (multi-camera batching)
`batch_env_render` 支持 `Ncam > 1`，可在单次 GPU kernel 调用中渲染所有相机。将逐相机循环改为一次性传入 `(Nenv, Ncam, ...)` 张量，FG 和每个 mask object 各只需一次 `batch_env_render` 调用。

**节省**: rasterization 调用次数从 N_cam×(1+N_obj) 降为 1+N_obj（3 相机 2 物体: 9→3）

---

## 后续优化方向

### 5. GPU 端 mask 计算（避免 CPU 往返）
当前 mask 的 occlusion test 和 binary_mask 生成流程:
```
alpha_t (GPU) → .cpu().numpy() → numpy boolean ops → numpy mask
scene_depth (GPU) → .cpu().numpy() → numpy comparison
```
每次 `.cpu()` 都触发 CUDA 同步。如果下游能接受 torch tensor（`to_numpy=False`），整个 occlusion test 可用 torch 在 GPU 完成，省去多次 GPU↔CPU 传输。

**预估提升**: ~15-20%

### 6. 跳过全禁用相机的 `update_scene`
`super().capture_observation()` 中 `mujoco_env.py:894-935` 对每个相机都调用 `renderer.update_scene()`。如果某相机的 color/depth/mask/heat_map 全部被禁用（GS 接管），这个调用完全浪费。

**预估提升**: ~5-10%
**实现**: 在 `_collect_obs` 的相机循环中加 early-continue 判断

### 7. 线程并行原生 obs 采集
```python
# mujoco_env.py:1307
obs_per_env = [env.capture_observation() for env in self.envs]
```
各 env 的 `capture_observation()` 串行执行。MuJoCo 在渲染期间释放 GIL，可用 `ThreadPoolExecutor` 并行。batch_size=2 时效果有限，batch_size 较大时收益明显。

**预估提升**: 随 batch_size 线性增长

### 8. 静态物体 mask 缓存
不动的物体（如 coaster）每步的 mask 渲染结果不变，可通过比较 body pose 变化来跳过不必要的 mask 重渲染。

**预估提升**: 取决于静态物体占比，少量物体时收益有限

### 9. 修正 wrist_cam 背景缓存
`_bg_cache` 的 key 是 `(cam_id, width, height)`，但 `wrist_cam` 挂载在末端执行器上，每步位置变化。当前实现对移动相机返回过期的背景渲染结果。可将 static camera（`env1_cam`、`env0_cam`）和 dynamic camera（`wrist_cam`）分开处理。

**性质**: 正确性修复 + 对 wrist_cam 的性能权衡
