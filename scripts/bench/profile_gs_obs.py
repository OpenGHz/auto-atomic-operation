"""torch.profiler around capture_observation for back_gs perf analysis.

Usage:
    python scripts/bench/profile_gs_obs.py [task] [iterations] [hydra overrides...]

Examples:
    python scripts/bench/profile_gs_obs.py
    python scripts/bench/profile_gs_obs.py open_door_back 16 env.batch_size=4
    python scripts/bench/profile_gs_obs.py press_blue_button 8 embodiment=p7_g2p

Defaults: task=open_door_back, iterations=8 (after warmup). ``render=gs`` is
always applied (a later ``render=...`` override wins).
Output: prints CUDA self-time top-20 and saves Chrome trace to
outputs/bench/profiles/<run_name>_b<batch_size>/.
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

from auto_atom import (
    ObservationEnvProtocol,
    SimulationLoopEnvProtocol,
    require_env_capability,
)
from auto_atom.config_loader import compose_task_run
from auto_atom.runner.common import prepare_task_file


def _parse_args(argv: list[str]) -> tuple[str, int, list[str]]:
    args = list(argv)
    task = "open_door_back"
    iterations = 8

    idx = 0
    if (
        idx < len(args)
        and "=" not in args[idx]
        and not args[idx].startswith(("+", "~"))
        and (Path.cwd() / "aao_configs" / "task" / f"{args[idx]}.yaml").exists()
    ):
        task = args[idx]
        idx += 1
    if idx < len(args) and args[idx].isdigit():
        iterations = int(args[idx])
        idx += 1
    return task, iterations, args[idx:]


task, iterations, overrides = _parse_args(sys.argv[1:])

# `++` sets a key whether or not a config (e.g. `+test=open_the_door`)
# already defines it; user overrides listed later still win.
bench_defaults = [
    "render=gs",
    "++env.viewer.disable=true",
    "++env.to_numpy=false",
    "++env.structured=false",
]
overrides = bench_defaults + overrides

print(f"[profile] task={task} iters={iterations} overrides={overrides}")
cfg, run_name = compose_task_run(task, overrides)
task_file = prepare_task_file(cfg)
backend = task_file.backend(task_file.task, task_file.task_operators)
backend.setup(task_file.task)
backend.reset()
env = backend.get_env()
observation_env = require_env_capability(
    env,
    ObservationEnvProtocol,
    feature="profile_gs_obs observation capture",
    expected_batch_size=backend.batch_size,
)
simulation_env = require_env_capability(
    env,
    SimulationLoopEnvProtocol,
    feature="profile_gs_obs simulation update",
    expected_batch_size=backend.batch_size,
)

print(f"[profile] batch_size={backend.batch_size}")

# Warmup: trigger gsplat JIT, prime caches.
for _ in range(2):
    observation_env.capture_observation()
    simulation_env.update()
torch.cuda.synchronize()

trace_dir = Path("outputs/bench/profiles") / f"{run_name}_b{backend.batch_size}"
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
            observation_env.capture_observation()
        with torch.profiler.record_function("update"):
            simulation_env.update()
        prof.step()

torch.cuda.synchronize()

print("\n=== top 20 by CUDA self time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

print("\n=== top 20 by CPU self time ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

print(f"\n[profile] traces written under {trace_dir.resolve()}")

backend.teardown()
