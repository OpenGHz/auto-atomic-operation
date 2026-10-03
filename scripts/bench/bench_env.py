"""Minimal benchmark: capture_observation + update timing.

Usage:
    python scripts/bench/bench_env.py [task] [iterations] [--profile] [hydra overrides...]

``task`` is an option of the ``task`` config group. Without it the benchmark
runs ``task=cup_on_coaster render=gs``; with it, pass ``render=gs`` /
``embodiment=...`` as overrides when needed.

Examples:
    python scripts/bench/bench_env.py press_three_buttons render=gs
    python scripts/bench/bench_env.py press_three_buttons 50 render=gs env.batch_size=4
    python scripts/bench/bench_env.py env.batch_size=10
    python scripts/bench/bench_env.py cup_on_coaster 20 --profile render=gs

Results are saved to ``outputs/bench/<run_name>.json`` where ``<run_name>`` is
``<task>__<embodiment>[__<render>]``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from auto_atom import (
    ObservationEnvProtocol,
    SimulationLoopEnvProtocol,
    require_env_capability,
)
from auto_atom.config_loader import compose_task_run
from auto_atom.runner.common import prepare_task_file


def _log_progress(message: str) -> None:
    print(f"[bench_env] {message}", flush=True)


def _looks_like_override(arg: str) -> bool:
    return "=" in arg or arg.startswith(("+", "~"))


def _looks_like_task_name(arg: str) -> bool:
    if _looks_like_override(arg) or arg.isdigit():
        return False
    return (Path.cwd() / "aao_configs" / "task" / f"{arg}.yaml").exists()


def _parse_args(argv: list[str]) -> tuple[str, int, bool, list[str]]:
    # Parse args: [task] [N] [--profile] [hydra overrides...]
    args = list(argv)
    do_profile = "--profile" in args
    if do_profile:
        args.remove("--profile")

    task = "cup_on_coaster"
    iterations = 10
    overrides: list[str] = []

    idx = 0
    if idx < len(args) and _looks_like_task_name(args[idx]):
        task = args[idx]
        idx += 1
    else:
        # The default benchmark is the GS variant of the default task.
        overrides.append("render=gs")

    if idx < len(args) and args[idx].isdigit():
        iterations = int(args[idx])
        idx += 1

    overrides += args[idx:]
    return task, iterations, do_profile, overrides


def _bench_loop(
    observation_env: ObservationEnvProtocol,
    simulation_env: SimulationLoopEnvProtocol,
    iterations: int,
) -> tuple[list[float], list[float]]:
    obs_times = []
    upd_times = []
    progress_every = max(1, min(10, iterations // 10 if iterations > 10 else 1))
    for i in range(iterations):
        t0 = time.perf_counter()
        observation_env.capture_observation()
        t1 = time.perf_counter()
        simulation_env.update()
        t2 = time.perf_counter()
        obs_times.append(t1 - t0)
        upd_times.append(t2 - t1)
        if (i + 1) % progress_every == 0 or i + 1 == iterations:
            _log_progress(f"benchmark progress: {i + 1}/{iterations}")
    return obs_times, upd_times


def _mean_hz(arr_ms: np.ndarray) -> float:
    mean_ms = float(arr_ms.mean())
    return 1000.0 / mean_ms if mean_ms > 0 else float("inf")


def _stat_dict(arr_ms: np.ndarray) -> dict:
    return {
        "mean_ms": round(float(arr_ms.mean()), 2),
        "std_ms": round(float(arr_ms.std()), 2),
        "min_ms": round(float(arr_ms.min()), 2),
        "max_ms": round(float(arr_ms.max()), 2),
        "mean_hz": round(_mean_hz(arr_ms), 2),
    }


def main() -> None:
    task, iterations, do_profile, overrides = _parse_args(sys.argv[1:])

    # Benchmark defaults: disable viewer, keep GS data on GPU. `++` sets a key
    # whether or not a config (e.g. `+test=open_the_door`) already defines it;
    # user overrides listed later still win. `to_numpy` only exists on the GS env.
    bench_defaults = [
        "++env.viewer.disable=true",
        "++env.structured=false",
    ]
    if "render=gs" in overrides:
        bench_defaults.append("++env.to_numpy=false")
    overrides = bench_defaults + overrides

    # Setup
    _log_progress(
        f"loading task={task} overrides={overrides or '[]'} iterations={iterations}"
    )
    cfg, run_name = compose_task_run(task, overrides)
    task_file = prepare_task_file(cfg)
    _log_progress("building backend")
    backend = task_file.backend(task_file.task, task_file.task_operators)
    _log_progress("setting up backend")
    backend.setup(task_file.task)
    _log_progress("resetting environment")
    backend.reset()
    env = backend.get_env()
    observation_env = require_env_capability(
        env,
        ObservationEnvProtocol,
        feature="bench_env observation capture",
        expected_batch_size=backend.batch_size,
    )
    simulation_env = require_env_capability(
        env,
        SimulationLoopEnvProtocol,
        feature="bench_env simulation update",
        expected_batch_size=backend.batch_size,
    )

    print(f"run={run_name}  batch_size={backend.batch_size}  iterations={iterations}")

    # Warmup (exclude from stats)
    _log_progress("running warmup")
    observation_env.capture_observation()
    simulation_env.update()
    _log_progress("warmup complete")

    if do_profile:
        import cProfile
        import pstats

        _log_progress("profiling benchmark loop")
        profiler = cProfile.Profile()
        profiler.enable()
        obs_times, upd_times = _bench_loop(observation_env, simulation_env, iterations)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats(pstats.SortKey.CUMULATIVE)
        print("\n--- cProfile top 20 ---")
        stats.print_stats(20)
    else:
        _log_progress("running benchmark loop")
        obs_times, upd_times = _bench_loop(observation_env, simulation_env, iterations)

    obs_arr = np.array(obs_times) * 1000
    upd_arr = np.array(upd_times) * 1000
    total = obs_arr + upd_arr

    fmt = (
        "{:<22s} mean={:>7.2f}ms  std={:>6.2f}ms  "
        "min={:>7.2f}ms  max={:>7.2f}ms  freq={:>8.2f}Hz"
    )
    print()
    for label, arr in (
        ("capture_observation", obs_arr),
        ("update", upd_arr),
        ("total", total),
    ):
        print(
            fmt.format(
                label, arr.mean(), arr.std(), arr.min(), arr.max(), _mean_hz(arr)
            )
        )

    bench_result = {
        "run_name": run_name,
        "task": task,
        "batch_size": backend.batch_size,
        "iterations": iterations,
        "overrides": overrides,
        "capture_observation": _stat_dict(obs_arr),
        "update": _stat_dict(upd_arr),
        "total": _stat_dict(total),
    }

    out_path = Path("outputs") / "bench" / f"{run_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bench_result, indent=2, ensure_ascii=False) + "\n")
    print(f"\nBenchmark saved to {out_path.resolve()}")

    _log_progress("tearing down backend")
    backend.teardown()


if __name__ == "__main__":
    main()
