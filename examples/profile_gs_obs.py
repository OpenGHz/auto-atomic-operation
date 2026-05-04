"""torch.profiler around capture_observation for back_gs perf analysis.

Usage:
    python examples/profile_gs_obs.py [config_name] [iterations] [hydra overrides...]

Defaults: config_name=open_door_airbot_play_back_gs, iterations=8 (after warmup).
Output: prints CUDA self-time top-20 and saves Chrome trace to outputs/bench/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)

from auto_atom import load_task_file_hydra


def _parse_args(argv: list[str]) -> tuple[str, int, list[str]]:
    args = list(argv)
    config_name = "open_door_airbot_play_back_gs"
    iterations = 8

    idx = 0
    if (
        idx < len(args)
        and "=" not in args[idx]
        and not args[idx].startswith(("+", "~"))
    ):
        if (Path.cwd() / "aao_configs" / f"{args[idx]}.yaml").exists():
            config_name = args[idx]
            idx += 1
    if idx < len(args) and args[idx].isdigit():
        iterations = int(args[idx])
        idx += 1
    return config_name, iterations, args[idx:]


config_name, iterations, overrides = _parse_args(sys.argv[1:])

bench_defaults = [
    "+env.viewer.disable=true",
    "+env.to_numpy=false",
    "+env.structured=false",
]
overrides = bench_defaults + overrides

print(f"[profile] config={config_name} iters={iterations} overrides={overrides}")
task_file = load_task_file_hydra(config_name, overrides=overrides)
backend = task_file.backend(task_file.task, task_file.task_operators)
backend.setup(task_file.task)
backend.reset()
env = backend.env

print(f"[profile] batch_size={backend.batch_size}")

# Warmup: trigger gsplat JIT, prime caches.
for _ in range(2):
    env.capture_observation()
    env.update()
torch.cuda.synchronize()

trace_dir = Path("outputs/bench/profiles") / f"{config_name}_b{backend.batch_size}"
trace_dir.mkdir(parents=True, exist_ok=True)

prof_schedule = schedule(wait=1, warmup=1, active=iterations, repeat=1)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=prof_schedule,
    on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
    record_shapes=False,
    with_stack=False,
    profile_memory=False,
) as prof:
    for _ in range(iterations + 2):  # +2 for wait + warmup
        with torch.profiler.record_function("capture_observation"):
            env.capture_observation()
        with torch.profiler.record_function("update"):
            env.update()
        prof.step()

torch.cuda.synchronize()

print("\n=== top 20 by CUDA self time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

print("\n=== top 20 by CPU self time ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

print(f"\n[profile] traces written under {trace_dir.resolve()}")

backend.teardown()
