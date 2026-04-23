"""Console entry point for task-runner demos."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from auto_atom.runtime import TaskRunner

from .common import (
    ExampleLoopHooks,
    get_config_dir,
    list_demos,
    prepare_task_file,
    print_final_summary,
    run_example_rounds,
    save_final_summary,
)

if "--list" in sys.argv:
    list_demos(get_config_dir())
    sys.exit(0)


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    task_file = prepare_task_file(cfg)
    t0 = perf_counter()
    runner = TaskRunner().from_config(task_file)
    init_time = perf_counter() - t0

    rounds = int(cfg.get("rounds", 1))
    use_input = bool(cfg.get("use_input", False))
    max_updates = int(cfg.get("max_updates", 600))
    perf_count = bool(cfg.get("perf_count", False))
    _last_obs = [None]  # mutable container to hold observation across steps

    def _step_fn(_step, _update):
        if perf_count:
            _last_obs[0] = runner._context.backend.env.capture_observation()
        return runner.update()

    try:
        round_summaries = run_example_rounds(
            rounds=rounds,
            use_input=use_input,
            hooks=ExampleLoopHooks(
                reset_fn=runner.reset,
                step_fn=_step_fn,
                summarize_fn=lambda update, steps_used, max_updates, elapsed_time_sec: (
                    runner.summarize(
                        update,
                        max_updates=max_updates,
                        updates_used=steps_used,
                        elapsed_time_sec=elapsed_time_sec,
                    )
                ),
                records_fn=lambda: runner.records,
                reset_label="Reset task",
                start_label="Scene reset complete; viewer refreshed. Starting task updates...",
                max_updates=max_updates,
            ),
        )
        print_final_summary(round_summaries, init_time_sec=init_time)
        viewer_cfg = cfg.get("env", {}).get("viewer", None) if "env" in cfg else None
        hydra_cfg = HydraConfig.get()
        save_final_summary(
            round_summaries,
            path=Path(hydra_cfg.runtime.output_dir) / "summary.json",
            init_time_sec=init_time,
            run_config={
                "config_name": hydra_cfg.job.config_name,
                "batch_size": runner._context.backend.batch_size,
                "perf_count": perf_count,
                "viewer": viewer_cfg is not None,
                "rounds": rounds,
                "max_updates": max_updates,
            },
        )
    finally:
        runner.close()


if __name__ == "__main__":
    main()
