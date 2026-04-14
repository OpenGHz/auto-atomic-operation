# 基准测试与分析脚本

本文档直接说明如何使用仓库内现成脚本完成基准测试、绘图和分析。

## 脚本位置

- 执行基准：`tests/run_bench_suite.py`
- 绘图与分析：`tests/plot_bench_results.py`

这两个脚本已经覆盖以下三类测试：

- 任务级 `update`
- 任务级 `update + capture_observation`
- 环境级 `capture_observation + update`

并且会自动测试 `batch_size=1,2,4,8`，同时为第一类测试记录显存峰值。

## 使用方式

在仓库根目录执行。当前项目推荐使用 `/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python`；`tests/run_bench_suite.py` 默认也会用这个解释器启动子进程，可通过 `--python` 覆盖。

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python tests/run_bench_suite.py
```

脚本会严格串行执行所有 benchmark，不会并发跑多个命令，以避免资源竞争影响测试稳定性。

基准执行完成后，再生成图表和分析：

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python tests/plot_bench_results.py
```

默认情况下，第二个脚本会读取 `outputs/bench_suite/` 下最新的一次测试结果。

如果要指定某次结果目录，可以执行：

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python tests/plot_bench_results.py outputs/bench_suite/<run_id>
```

## 输出内容

每次运行会在 `outputs/bench_suite/<run_id>/` 下生成结果目录，包含：

- `manifest.json`：本次基准的总清单
- `results/*.json`：标准化后的各项测试结果
- `raw/`：原始 benchmark 输出
- `logs/`：每条命令对应日志
- `analysis.md`：自动生成的文字分析
- `task_level_frequency.png`：任务级性能曲线
- `env_level_breakdown.png`：环境级分解曲线
- `task_level_gpu_memory.png`：任务级显存曲线

## 可调参数

执行脚本支持常见参数：

```bash
/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python tests/run_bench_suite.py \
  --config-name cup_on_coaster_gs \
  --batch-sizes 1 2 4 8 \
  --iterations 10 \
  --max-updates 300
```

可通过 `--help` 查看完整参数说明。

## 建议

- 运行 benchmark 时不要同时启动其他重负载任务。
- 如果重新测量，建议整套流程重新执行一次，避免混用不同时间生成的结果。
- 分析图和文字结论以同一 `run_id` 目录下的文件为准。
