"""Shared CLI loop helpers for runner entry points."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Optional, Sequence

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
    summarize_fn: Callable[[TaskUpdate, int, Optional[int]], ExecutionSummary]
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

        steps_used = 0
        for step in range(hooks.max_updates or 10**18):
            if use_input:
                input("Press Enter to continue...")

            update = hooks.step_fn(step, update)
            steps_used += 1
            print(f"Step {step}:" + "=" * 40)
            pprint(update, sort_dicts=False)
            if bool(np.all(update.done)):
                break
        else:
            assert hooks.max_updates is not None
            print(f"Reached max_updates={hooks.max_updates}, stopping rollout.")

        summary = hooks.summarize_fn(update, steps_used, hooks.max_updates)

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


def print_final_summary(round_summaries: Sequence[ExecutionSummary]) -> None:
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_success = sum(1 for s in round_summaries if bool(np.all(s.final_success)))
    print(f"Success rate: {n_success}/{len(round_summaries)}")
    print()
    for i, summary in enumerate(round_summaries, start=1):
        tag = "OK" if bool(np.all(summary.final_success)) else "FAIL"
        print(f"  Round {i}: [{tag}]")
        print(f"    total updates: {summary.updates_used}")
        for record in summary.records:
            print(
                f"    env {record.env_index} stage {record.stage_name}: {record.status.value}"
            )
        print(f"    completed stages: {summary.completed_stage_count.tolist()}")
        print(f"    final stage: {summary.final_stage_name}")
        print(f"    final success: {summary.final_success.tolist()}")
        failure_lines = _format_failure_lines(summary)
        for line in failure_lines:
            print(f"    {line}")
    print("=" * 60)


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
