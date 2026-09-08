# 并行整批 step（`parallel_batch_step`）

> MuJoCo 复制式 batch 的 CPU 吞吐优化（2026-09-09 实现，默认关闭）。

## 是什么

`EnvConfig.parallel_batch_step: bool = False`（可配 `parallel_batch_workers`）。

复制式（REPLICATED）batch 下，`BatchedUnifiedMujocoEnv.step(action, env_mask)` 与
`update(env_mask)` 会把每个独立副本的物理推进提交到共享线程池并行执行。
`mj_step` 在 C 层执行期间释放 GIL，因此 B 个独立 `MjModel`/`MjData` 能真正并行。

```yaml
env:
  batch_size: 5
  parallel_batch_step: true        # 默认 false
  # parallel_batch_workers: 4      # 可选，默认 min(batch_size, cpu_count)
```

## 何时生效 / 何时忽略

生效条件（同时满足）：

- `parallel_batch_step=true`
- `batch_size > 1`
- 未挂 viewer（交互模式自动退回顺序执行）
- 本次调用激活行数 > 1（`env_mask` 选单行时仍顺序）
- REPLICATED 模式（共享物理的 GS `share_physics` 场景只有 1 个物理副本，天然无收益）

## 收益实测（mujoco 3.12，20 逻辑核，真实机器人场景，每 update 10 子步）

| batch | 顺序 | 并行 | 加速 |
|---|---|---|---|
| 2 | 51.6 ms | 48.4 ms | 1.06× |
| 5 | 136.2 ms | 65.9 ms | **2.07×** |
| 8 | 230.0 ms | 88.1 ms | **2.61×** |

每 env 单步成本越大（接触越多 / 子步越多），收益越接近早期裸 `mj_step` 基准
（B=4 ~4×、B=8 ~5×）。小负载 + 小 batch（如 B=2 空转场景）不值得开。

适用路径：整批推进物理的调用方——`PolicyEvaluator`、`start_sim_loop`、
`data_replay`。`TaskRunner` 的任务执行是**逐 env**推进（各 env 阶段进度独立），
不在本开关覆盖范围。

## 确定性保证

每个副本是独立的 `MjModel`/`MjData`，worker 线程只推进自己的副本、无共享可变
状态。实测（含单测）**并行与顺序逐位一致**（qpos 最大差 0.0），因此开启后任务
结果与随机种子行为不变。

## 线程安全边界

- 已审计：`pre_step_callbacks` 按 env 独立实例绑定；每 env viewer 独立；观测/渲染
  不在 `step/update` 内。
- 设计约束：**只并行纯物理推进**（`step`/`update`），不并行 IK/位姿规划/观测采集
  （后者涉及跨 env 共享对象，未做线程安全审计）。
- `close()` 会 `shutdown` 线程池；重复 `close` 幂等。
