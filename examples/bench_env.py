"""Minimal benchmark: capture_observation + update timing.

Usage:
    python examples/bench_env.py <config_name> [iterations] [hydra overrides...]

Examples:
    python examples/bench_env.py press_three_buttons_gs
    python examples/bench_env.py press_three_buttons_gs 50 env.batch_size=4
    python examples/bench_env.py env.batch_size=10
    python examples/bench_env.py cup_on_coaster_gs 20 --profile
"""

import sys
import time
from pathlib import Path

import numpy as np

from auto_atom import load_task_file_hydra


def _log_progress(message: str) -> None:
    print(f"[bench_env] {message}", flush=True)


def _looks_like_override(arg: str) -> bool:
    return "=" in arg or arg.startswith(("+", "~"))


def _looks_like_config_name(arg: str) -> bool:
    if _looks_like_override(arg) or arg.isdigit():
        return False
    return (Path.cwd() / "aao_configs" / f"{arg}.yaml").exists()


def _parse_args(argv: list[str]) -> tuple[str, int, bool, list[str]]:
    # Parse args: [config] [N] [--profile] [hydra overrides...]
    args = list(argv)
    do_profile = "--profile" in args
    if do_profile:
        args.remove("--profile")

    config_name = "cup_on_coaster_gs"
    iterations = 10
    overrides: list[str] = []

    idx = 0
    if idx < len(args) and _looks_like_config_name(args[idx]):
        config_name = args[idx]
        idx += 1

    if idx < len(args) and args[idx].isdigit():
        iterations = int(args[idx])
        idx += 1

    overrides = args[idx:]
    return config_name, iterations, do_profile, overrides


CONFIG_NAME, N, do_profile, overrides = _parse_args(sys.argv[1:])

# Benchmark defaults: disable viewer, keep data on GPU.
# User overrides can still override these (last wins in Hydra).
bench_defaults = [
    "+env.viewer.disable=true",
    "+env.to_numpy=true",
    "+env.structured=true",
]
overrides = bench_defaults + overrides

# Setup
_log_progress(
    f"loading config={CONFIG_NAME} overrides={overrides or '[]'} iterations={N}"
)
task_file = load_task_file_hydra(CONFIG_NAME, overrides=overrides)
_log_progress("building backend")
backend = task_file.backend(task_file.task, task_file.task_operators)
_log_progress("setting up backend")
backend.setup(task_file.task)
_log_progress("resetting environment")
backend.reset()
env = backend.env

print(f"config={CONFIG_NAME}  batch_size={backend.batch_size}  iterations={N}")

# Warmup (exclude from stats)
_log_progress("running warmup")
env.capture_observation()
env.update()
_log_progress("warmup complete")


def bench_loop():
    obs_times = []
    upd_times = []
    progress_every = max(1, min(10, N // 10 if N > 10 else 1))
    for i in range(N):
        t0 = time.perf_counter()
        env.capture_observation()
        t1 = time.perf_counter()
        env.update()
        t2 = time.perf_counter()
        obs_times.append(t1 - t0)
        upd_times.append(t2 - t1)
        if (i + 1) % progress_every == 0 or i + 1 == N:
            _log_progress(f"benchmark progress: {i + 1}/{N}")
    return obs_times, upd_times


if do_profile:
    import cProfile
    import pstats

    _log_progress("profiling benchmark loop")
    profiler = cProfile.Profile()
    profiler.enable()
    obs_times, upd_times = bench_loop()
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    print("\n--- cProfile top 20 ---")
    stats.print_stats(20)
else:
    _log_progress("running benchmark loop")
    obs_times, upd_times = bench_loop()

obs_arr = np.array(obs_times) * 1000
upd_arr = np.array(upd_times) * 1000
total = obs_arr + upd_arr


def _mean_hz(arr_ms: np.ndarray) -> float:
    mean_ms = float(arr_ms.mean())
    return 1000.0 / mean_ms if mean_ms > 0 else float("inf")


fmt = (
    "{:<22s} mean={:>7.2f}ms  std={:>6.2f}ms  "
    "min={:>7.2f}ms  max={:>7.2f}ms  freq={:>8.2f}Hz"
)
print()
print(
    fmt.format(
        "capture_observation",
        obs_arr.mean(),
        obs_arr.std(),
        obs_arr.min(),
        obs_arr.max(),
        _mean_hz(obs_arr),
    )
)
print(
    fmt.format(
        "update",
        upd_arr.mean(),
        upd_arr.std(),
        upd_arr.min(),
        upd_arr.max(),
        _mean_hz(upd_arr),
    )
)
print(
    fmt.format(
        "total",
        total.mean(),
        total.std(),
        total.min(),
        total.max(),
        _mean_hz(total),
    )
)

_log_progress("tearing down backend")
backend.teardown()
