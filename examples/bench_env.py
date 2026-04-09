"""Minimal benchmark: capture_observation + update timing.

Usage:
    python examples/bench_env.py <config_name> [iterations] [hydra overrides...]

Examples:
    python examples/bench_env.py press_three_buttons_gs
    python examples/bench_env.py press_three_buttons_gs 50 env.batch_size=4
    python examples/bench_env.py cup_on_coaster_gs 20 --profile
"""

import sys
import time

import numpy as np

from auto_atom import load_task_file_hydra

# Parse args: config [N] [--profile] [hydra overrides...]
args = sys.argv[1:]
do_profile = "--profile" in args
if do_profile:
    args.remove("--profile")

CONFIG_NAME = args[0] if args else "press_three_buttons_gs"
N = int(args[1]) if len(args) > 1 and args[1].isdigit() else 10
overrides = [a for a in args[1:] if not a.isdigit()]

# Setup
task_file = load_task_file_hydra(CONFIG_NAME, overrides=overrides)
backend = task_file.backend(task_file.task, task_file.task_operators)
backend.setup(task_file.task)
backend.reset()
env = backend.env

print(f"config={CONFIG_NAME}  batch_size={backend.batch_size}  iterations={N}")

# Warmup (exclude from stats)
env.capture_observation()
env.update()


def bench_loop():
    obs_times = []
    upd_times = []
    for _ in range(N):
        t0 = time.perf_counter()
        env.capture_observation()
        t1 = time.perf_counter()
        env.update()
        t2 = time.perf_counter()
        obs_times.append(t1 - t0)
        upd_times.append(t2 - t1)
    return obs_times, upd_times


if do_profile:
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    obs_times, upd_times = bench_loop()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    print("\n--- cProfile top 20 ---")
    stats.print_stats(20)
else:
    obs_times, upd_times = bench_loop()

obs_arr = np.array(obs_times) * 1000
upd_arr = np.array(upd_times) * 1000
total = obs_arr + upd_arr

fmt = "{:<22s} mean={:>7.2f}ms  std={:>6.2f}ms  min={:>7.2f}ms  max={:>7.2f}ms"
print()
print(
    fmt.format(
        "capture_observation",
        obs_arr.mean(),
        obs_arr.std(),
        obs_arr.min(),
        obs_arr.max(),
    )
)
print(fmt.format("update", upd_arr.mean(), upd_arr.std(), upd_arr.min(), upd_arr.max()))
print(fmt.format("total", total.mean(), total.std(), total.min(), total.max()))

backend.teardown()
