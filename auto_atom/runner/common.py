"""Shared CLI loop helpers for runner entry points."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from auto_atom import (
    ComponentRegistry,
    ExecutionRecord,
    ExecutionSummary,
    TaskFileConfig,
    TaskUpdate,
)


@dataclass
class ExampleLoopHooks:
    reset_fn: Callable[[], TaskUpdate]
    step_fn: Callable[[int, TaskUpdate], TaskUpdate]
    summarize_fn: Callable[[TaskUpdate, int, Optional[int], float], ExecutionSummary]
    records_fn: Callable[[], Sequence[ExecutionRecord]]
    before_round_fn: Optional[Callable[[int], None]] = None
    reset_label: str = "Reset"
    start_label: str = "Starting updates..."
    max_updates: Optional[int] = None


def get_config_dir() -> Path:
    return Path.cwd() / "aao_configs"


def list_demos(base_dir: Optional[Path] = None) -> None:
    config_dir = base_dir or get_config_dir()
    names = sorted(p.stem for p in config_dir.glob("*.yaml"))
    print(f"Available demos ({len(names)}):")
    for name in names:
        print(f"  {name}")


def prepare_task_file(cfg: DictConfig) -> TaskFileConfig:
    ComponentRegistry.clear()
    if "env" in cfg and cfg.env is not None:
        instantiate(cfg.env)

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    return TaskFileConfig.model_validate(raw)


def run_example_rounds(
    *,
    rounds: int,
    use_input: bool,
    hooks: ExampleLoopHooks,
) -> List[ExecutionSummary]:
    round_summaries: List[ExecutionSummary] = []

    for r in range(rounds):
        if hooks.before_round_fn is not None:
            hooks.before_round_fn(r)

        if rounds > 1:
            print(f"Round {r + 1}/{rounds}")
            print("=" * 50)

        print(hooks.reset_label)
        update = hooks.reset_fn()
        pprint(update, sort_dicts=False)
        print(hooks.start_label)
        print()

        batch_size = len(update.stage_name)
        env_completion_steps = np.full(batch_size, -1, dtype=np.int64)
        env_completion_time_sec = np.full(batch_size, np.nan, dtype=np.float64)

        # Warmup step 0: may trigger JIT compilation; exclude from timing.
        update = hooks.step_fn(0, update)
        steps_used = 1
        print("Step 0 (warmup):" + "=" * 40)
        pprint(update, sort_dicts=False)
        if bool(np.all(update.done)):
            env_completion_steps[np.asarray(update.done, dtype=bool)] = 1
            env_completion_time_sec[np.asarray(update.done, dtype=bool)] = 0.0

        start_time = perf_counter()
        for step in range(1, hooks.max_updates or 10**18):
            if bool(np.all(update.done)):
                break
            if use_input:
                input("Press Enter to continue...")

            update = hooks.step_fn(step, update)
            steps_used += 1
            elapsed_now = perf_counter() - start_time
            done_mask = np.asarray(update.done, dtype=bool)
            newly_done = done_mask & (env_completion_steps < 0)
            env_completion_steps[newly_done] = steps_used
            env_completion_time_sec[newly_done] = elapsed_now
            print(f"Step {step}:" + "=" * 40)
            pprint(update, sort_dicts=False)
        else:
            if not bool(np.all(update.done)):
                assert hooks.max_updates is not None
                print(f"Reached max_updates={hooks.max_updates}, stopping rollout.")

        elapsed_time_sec = perf_counter() - start_time
        summary = hooks.summarize_fn(
            update,
            steps_used,
            hooks.max_updates,
            elapsed_time_sec,
        )
        summary.env_completion_steps = env_completion_steps
        summary.env_completion_time_sec = env_completion_time_sec
        if summary.sim_time_sec > 0 and summary.updates_used > 0:
            dt = summary.sim_time_sec / summary.updates_used
            sim_times = np.where(
                env_completion_steps >= 0,
                env_completion_steps.astype(np.float64) * dt,
                np.nan,
            )
            summary.env_completion_sim_time_sec = sim_times
        summary.completed_stage_info = _group_completed_stage_info(summary)

        print()
        print("Execution records:")
        for record in hooks.records_fn():
            pprint(record)

        print()
        print("Summary:")
        pprint(summary)

        round_summaries.append(summary)

        if rounds > 1:
            print()

    return round_summaries


def print_final_summary(
    round_summaries: Sequence[ExecutionSummary],
    *,
    init_time_sec: Optional[float] = None,
) -> None:
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if init_time_sec is not None:
        print(f"Sim init time: {init_time_sec:.3f}s")
    if not round_summaries:
        print("Success rate: 0/0")
        print("=" * 60)
        return

    env_successes = sum(_count_env_successes(summary) for summary in round_summaries)
    env_total = sum(len(summary.final_success) for summary in round_summaries)
    print(f"Success rate: {env_successes}/{env_total}")
    print()
    for i, summary in enumerate(round_summaries, start=1):
        round_joint_success = bool(np.all(summary.final_success))
        tag = "OK" if round_joint_success else "FAIL"
        round_env_success = _count_env_successes(summary)
        batch_size = len(summary.final_success)
        failure_lines = _format_failure_lines(summary)
        loop_freq = (
            summary.updates_used / summary.elapsed_time_sec
            if summary.elapsed_time_sec > 0
            else float("inf")
        )
        round_payload = {
            "status": tag,
            "success_rate": f"{round_env_success}/{batch_size}",
            "loop_frequency": f"{loop_freq:.1f} Hz ({summary.updates_used} steps in {summary.elapsed_time_sec:.3f}s)",
            "completed_stage_info": _format_completed_stage_info(
                summary.completed_stage_info
            ),
            "completion_steps": _format_optional_int_list(summary.env_completion_steps),
            "completion_time": _format_optional_time_list(
                summary.env_completion_time_sec
            ),
            "completion_sim_time": _format_optional_time_list(
                summary.env_completion_sim_time_sec
            ),
            "sim_time": _format_sim_time_stats(summary.env_completion_sim_time_sec),
            "completed_stages": summary.completed_stage_count.tolist(),
            "final_stage": summary.final_stage_name,
            "final_success": summary.final_success.tolist(),
        }
        if failure_lines:
            round_payload["failure_reasons"] = failure_lines
        print(f"Round {i}")
        pprint(round_payload, sort_dicts=False)
    print("=" * 60)


def save_final_summary(
    round_summaries: Sequence[ExecutionSummary],
    path: str | Path = "summary.json",
    *,
    init_time_sec: Optional[float] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save summary statistics to a JSON file."""
    data: Dict[str, Any] = {}
    if run_config:
        data["run_config"] = run_config
    if init_time_sec is not None:
        data["init_time_sec"] = round(init_time_sec, 3)

    env_successes = sum(_count_env_successes(s) for s in round_summaries)
    env_total = sum(len(s.final_success) for s in round_summaries)
    data["success_rate"] = f"{env_successes}/{env_total}"

    rounds_data: List[Dict[str, Any]] = []
    for summary in round_summaries:
        loop_freq = (
            summary.updates_used / summary.elapsed_time_sec
            if summary.elapsed_time_sec > 0
            else float("inf")
        )
        entry: Dict[str, Any] = {
            "status": "OK" if bool(np.all(summary.final_success)) else "FAIL",
            "success_rate": f"{_count_env_successes(summary)}/{len(summary.final_success)}",
            "loop_frequency_hz": round(loop_freq, 1),
            "updates_used": summary.updates_used,
            "elapsed_time_sec": round(summary.elapsed_time_sec, 3),
            "completed_stage_info": _format_completed_stage_info(
                summary.completed_stage_info
            ),
            "completion_steps": _format_optional_int_list(summary.env_completion_steps),
            "completion_time": _format_optional_time_list(
                summary.env_completion_time_sec
            ),
            "completion_sim_time": _format_optional_time_list(
                summary.env_completion_sim_time_sec
            ),
            "sim_time": _format_sim_time_stats(summary.env_completion_sim_time_sec),
            "completed_stages": summary.completed_stage_count.tolist(),
            "final_stage": summary.final_stage_name,
            "final_success": summary.final_success.tolist(),
        }
        failure_lines = _format_failure_lines(summary)
        if failure_lines:
            entry["failure_reasons"] = failure_lines
        rounds_data.append(entry)
    data["rounds"] = rounds_data

    out = Path(path)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Summary saved to {out.resolve()}")
    return out


def _format_failure_lines(summary: ExecutionSummary) -> List[str]:
    lines: List[str] = []
    failed_by_env = {
        record.env_index: record
        for record in summary.records
        if record.status == "failed"
        or getattr(record.status, "value", None) == "failed"
    }

    for env_index, final_success in enumerate(summary.final_success.tolist()):
        if final_success is True:
            continue

        failure_record = failed_by_env.get(env_index)
        if failure_record is not None:
            reason = _extract_failure_reason(failure_record.details)
            stage_name = failure_record.stage_name or "<unknown>"
            if reason:
                lines.append(
                    f"failure reason (env {env_index}, stage {stage_name}): {reason}"
                )
            else:
                lines.append(
                    f"failure reason (env {env_index}, stage {stage_name}): unknown"
                )
            continue

        if (
            summary.max_updates is not None
            and summary.final_done.tolist()[env_index] is False
        ):
            stage_name = summary.final_stage_name[env_index] or "<unknown>"
            lines.append(
                f"failure reason (env {env_index}, stage {stage_name}): reached max_updates={summary.max_updates} before completion"
            )
            continue

        stage_name = summary.final_stage_name[env_index] or "<unknown>"
        lines.append(f"failure reason (env {env_index}, stage {stage_name}): unknown")

    return lines


def _group_completed_stage_info(
    summary: ExecutionSummary,
) -> Dict[str, List[Optional[str]]]:
    grouped: Dict[str, List[Optional[str]]] = {}
    batch_size = len(summary.final_success)
    for record in summary.records:
        if record.stage_name not in grouped:
            grouped[record.stage_name] = [None] * batch_size
        grouped[record.stage_name][record.env_index] = record.status.value
    return grouped


def _format_completed_stage_info(
    completed_stage_info: Dict[str, List[Optional[str]]],
) -> Dict[str, List[Optional[str]]]:
    return {
        stage_name: list(statuses)
        for stage_name, statuses in completed_stage_info.items()
    }


def _count_env_successes(summary: ExecutionSummary) -> int:
    return int(np.count_nonzero(np.asarray(summary.final_success, dtype=bool)))


def _format_optional_int_list(values: Optional[np.ndarray]) -> List[Optional[int]]:
    if values is None:
        return []
    result: List[Optional[int]] = []
    for value in np.asarray(values, dtype=np.int64).tolist():
        result.append(None if value < 0 else int(value))
    return result


def _format_optional_time_list(values: Optional[np.ndarray]) -> List[Optional[str]]:
    if values is None:
        return []
    result: List[Optional[str]] = []
    for value in np.asarray(values, dtype=np.float64).tolist():
        if np.isnan(value):
            result.append(None)
        else:
            result.append(f"{float(value):.3f}s")
    return result


def _format_sim_time_stats(values: Optional[np.ndarray]) -> str:
    if values is None:
        return "N/A"
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[~np.isnan(valid)]
    if len(valid) == 0:
        return "N/A"
    if len(valid) == 1:
        return f"{valid[0]:.3f}s"
    return f"min={np.min(valid):.3f}s, max={np.max(valid):.3f}s, mean={np.mean(valid):.3f}s"


def _extract_failure_reason(details: object) -> Optional[str]:
    if not isinstance(details, dict):
        return None
    reason = details.get("failure_reason")
    if isinstance(reason, str) and reason:
        return reason
    event = details.get("event")
    if isinstance(event, str) and event:
        return event
    return None
