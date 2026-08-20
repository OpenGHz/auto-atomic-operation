"""Run benchmark scenarios from docs/skills/bench.md sequentially.

This script intentionally runs one command at a time to avoid resource
contention that could skew performance measurements.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_PYTHON = "/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python"
DEFAULT_BATCH_SIZES = [1, 2, 4, 8]
DEFAULT_CONFIG = "cup_on_coaster_gs"
DEFAULT_MAX_UPDATES = 300
DEFAULT_ITERATIONS = 100
DEFAULT_SAMPLE_INTERVAL_SEC = 0.1

TASK_UPDATE_ONLY = "task_update_only"
TASK_UPDATE_WITH_OBS = "task_update_with_obs"
ENV_UPDATE_WITH_OBS = "env_update_with_obs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python executable.")
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help="Task config name used by both benchmarks.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
        help="Batch sizes to benchmark sequentially.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Iterations passed to examples/bench_env.py.",
    )
    parser.add_argument(
        "--max-updates",
        type=int,
        default=DEFAULT_MAX_UPDATES,
        help="Max updates passed to the task-level benchmark.",
    )
    parser.add_argument(
        "--sample-interval-sec",
        type=float,
        default=DEFAULT_SAMPLE_INTERVAL_SEC,
        help="GPU memory sampling interval for the first command.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/bench_suite",
        help="Directory where benchmark artifacts are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    output_root = repo_root / args.output_root
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = output_root / run_id
    suite_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_name": args.config_name,
        "batch_sizes": args.batch_sizes,
        "python": args.python,
        "iterations": args.iterations,
        "max_updates": args.max_updates,
        "sample_interval_sec": args.sample_interval_sec,
        "suite_dir": str(suite_dir.resolve()),
        "results": [],
    }
    _write_json(suite_dir / "manifest.json", manifest)

    for batch_size in args.batch_sizes:
        print(f"\n=== batch_size={batch_size} ===", flush=True)

        task_only_result = run_task_benchmark(
            repo_root=repo_root,
            suite_dir=suite_dir,
            python_exe=args.python,
            config_name=args.config_name,
            batch_size=batch_size,
            max_updates=args.max_updates,
            perf_count=False,
            sample_gpu=True,
            sample_interval_sec=args.sample_interval_sec,
        )
        manifest["results"].append(task_only_result)
        _write_json(suite_dir / "manifest.json", manifest)

        task_obs_result = run_task_benchmark(
            repo_root=repo_root,
            suite_dir=suite_dir,
            python_exe=args.python,
            config_name=args.config_name,
            batch_size=batch_size,
            max_updates=args.max_updates,
            perf_count=True,
            sample_gpu=False,
            sample_interval_sec=args.sample_interval_sec,
        )
        manifest["results"].append(task_obs_result)
        _write_json(suite_dir / "manifest.json", manifest)

        env_result = run_env_benchmark(
            repo_root=repo_root,
            suite_dir=suite_dir,
            python_exe=args.python,
            config_name=args.config_name,
            batch_size=batch_size,
            iterations=args.iterations,
        )
        manifest["results"].append(env_result)
        _write_json(suite_dir / "manifest.json", manifest)

    print(f"\nSuite finished. Results saved to {suite_dir.resolve()}", flush=True)
    return 0


def run_task_benchmark(
    *,
    repo_root: Path,
    suite_dir: Path,
    python_exe: str,
    config_name: str,
    batch_size: int,
    max_updates: int,
    perf_count: bool,
    sample_gpu: bool,
    sample_interval_sec: float,
) -> dict[str, Any]:
    scenario = TASK_UPDATE_WITH_OBS if perf_count else TASK_UPDATE_ONLY
    scenario_dir = suite_dir / "raw" / f"{scenario}_batch_{batch_size}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_exe,
        "-m",
        "auto_atom.runner.demo",
        "--config-name",
        config_name,
        "env.viewer=null",
        f"+perf_count={'true' if perf_count else 'false'}",
        "+print_updates=false",
        f"env.batch_size={batch_size}",
        f"+max_updates={max_updates}",
        f"hydra.run.dir={scenario_dir.as_posix()}",
    ]
    stdout_path = suite_dir / "logs" / f"{scenario}_batch_{batch_size}.log"
    result = run_command(
        command=command,
        cwd=repo_root,
        stdout_path=stdout_path,
        sample_gpu=sample_gpu,
        sample_interval_sec=sample_interval_sec,
    )

    summary_path = scenario_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Expected summary file was not created: {summary_path}"
        )
    summary = json.loads(summary_path.read_text())

    round0 = _first_round(summary)
    if round0["loop_frequency_hz"] is None or round0["timed_updates"] == 0:
        raise RuntimeError(
            "Task benchmark completed before any timed update; increase the "
            "rollout workload to measure loop frequency."
        )
    normalized = {
        "scenario": scenario,
        "batch_size": batch_size,
        "command": command,
        "log_path": str(stdout_path.resolve()),
        "raw_summary_path": str(summary_path.resolve()),
        "loop_frequency_hz": round0["loop_frequency_hz"],
        "elapsed_time_sec": round0["elapsed_time_sec"],
        "updates_used": round0["updates_used"],
        "timed_updates": round0["timed_updates"],
        "success_rate": summary.get("success_rate"),
        "gpu_memory_peak_mb": result["gpu_memory_peak_mb"],
        "gpu_memory_samples": result["gpu_memory_samples"],
    }
    result_path = suite_dir / "results" / f"{scenario}_batch_{batch_size}.json"
    _write_json(result_path, normalized)
    print(
        f"{scenario}: {normalized['loop_frequency_hz']} Hz, "
        f"gpu_peak={normalized['gpu_memory_peak_mb']}",
        flush=True,
    )
    return normalized


def run_env_benchmark(
    *,
    repo_root: Path,
    suite_dir: Path,
    python_exe: str,
    config_name: str,
    batch_size: int,
    iterations: int,
) -> dict[str, Any]:
    scenario = ENV_UPDATE_WITH_OBS
    stdout_path = suite_dir / "logs" / f"{scenario}_batch_{batch_size}.log"
    raw_copy_path = suite_dir / "raw" / f"{scenario}_batch_{batch_size}.json"
    raw_copy_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        python_exe,
        "examples/bench_env.py",
        config_name,
        str(iterations),
        f"env.batch_size={batch_size}",
    ]
    run_command(
        command=command,
        cwd=repo_root,
        stdout_path=stdout_path,
        sample_gpu=False,
        sample_interval_sec=0.0,
    )

    default_output = repo_root / "outputs" / "bench" / f"{config_name}.json"
    if not default_output.exists():
        raise FileNotFoundError(
            f"Expected env benchmark output missing: {default_output}"
        )
    shutil.copy2(default_output, raw_copy_path)
    bench = json.loads(raw_copy_path.read_text())

    normalized = {
        "scenario": scenario,
        "batch_size": batch_size,
        "command": command,
        "log_path": str(stdout_path.resolve()),
        "raw_summary_path": str(raw_copy_path.resolve()),
        "capture_observation_mean_ms": bench["capture_observation"]["mean_ms"],
        "capture_observation_mean_hz": bench["capture_observation"]["mean_hz"],
        "update_mean_ms": bench["update"]["mean_ms"],
        "update_mean_hz": bench["update"]["mean_hz"],
        "total_mean_ms": bench["total"]["mean_ms"],
        "total_mean_hz": bench["total"]["mean_hz"],
    }
    result_path = suite_dir / "results" / f"{scenario}_batch_{batch_size}.json"
    _write_json(result_path, normalized)
    print(
        f"{scenario}: total={normalized['total_mean_ms']} ms "
        f"({normalized['total_mean_hz']} Hz)",
        flush=True,
    )
    return normalized


def run_command(
    *,
    command: list[str],
    cwd: Path,
    stdout_path: Path,
    sample_gpu: bool,
    sample_interval_sec: float,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running:", " ".join(command), flush=True)

    with stdout_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        gpu_samples: list[dict[str, float | int | None]] = []
        if sample_gpu:
            while True:
                return_code = process.poll()
                sample = sample_gpu_memory(process.pid)
                gpu_samples.append(
                    {
                        "elapsed_sec": round(time.monotonic(), 3),
                        "memory_mb": sample,
                    }
                )
                if return_code is not None:
                    break
                time.sleep(sample_interval_sec)
        else:
            return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    peak = None
    values = [s["memory_mb"] for s in gpu_samples if s["memory_mb"] is not None]
    if values:
        peak = int(max(values))
    return {
        "gpu_memory_peak_mb": peak,
        "gpu_memory_samples": gpu_samples,
    }


def sample_gpu_memory(pid: int) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            memory_mb = int(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            return memory_mb
    return None


def _first_round(summary: dict[str, Any]) -> dict[str, Any]:
    rounds = summary.get("rounds", [])
    if not rounds:
        raise ValueError("summary.json did not contain any rounds.")
    return rounds[0]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    sys.exit(main())
