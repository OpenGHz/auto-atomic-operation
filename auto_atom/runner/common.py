"""Shared CLI loop helpers for runner entry points."""

from __future__ import annotations

import json
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from pprint import pprint
from time import perf_counter
from typing import (
    Any,
    Callable,
    Collection,
    ContextManager,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, TypeAdapter

from auto_atom.config.task import (
    AutoAtomConfig,
    OperatorConfig,
    TaskFileConfig,
    name_task_operators,
)
from auto_atom.config_loader import describe_run
from auto_atom.execution_config import prepare_task_config_for_instantiation
from auto_atom.runtime import (
    ComponentRegistry,
    ExecutionRecord,
    ExecutionSummary,
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
    print_updates: bool = True
    update_limit_fn: Optional[Callable[[int], TaskUpdate]] = None
    quiet_context_fn: Optional[Callable[[], ContextManager[None]]] = None
    """Context entered around each skipped round, e.g. to hold the viewer; it
    must not change what the reset does."""


def parse_round_selection(
    value: Any,
    rounds: Optional[int] = None,
) -> Tuple[int, Optional[FrozenSet[int]]]:
    """Resolve the round count and the selected 1-based rounds.

    ``value`` selects rounds by their printed number: ``N``, an inclusive
    range ``A-B``, an open range ``A-`` up to ``rounds``, or a list or
    comma-separated string of those. ``None`` selects every round. Without
    ``rounds`` the run ends at the last selected round, so an open range then
    needs an explicit ``rounds``.

    Returns ``(rounds, selected)`` with ``selected`` ``None`` for all rounds.
    """
    if rounds is not None and int(rounds) < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    if value is None:
        return (1 if rounds is None else int(rounds)), None
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, str):
        items: List[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if not items:
        raise ValueError("round_selection selects no round")

    ranges: List[Tuple[int, Optional[int]]] = []
    for item in items:
        ranges.append(_parse_round_item(item))
    finite_ends = [end if end is not None else start for start, end in ranges]
    if rounds is None:
        if any(end is None for _, end in ranges):
            raise ValueError(
                "round_selection with an open range ('A-') needs rounds to be set"
            )
        rounds = max(finite_ends)
    rounds = int(rounds)
    selected = set()
    for start, end in ranges:
        stop = rounds if end is None else end
        if stop > rounds:
            raise ValueError(
                f"round_selection selects round {stop}, but the run has only "
                f"{rounds} rounds"
            )
        selected.update(range(start, stop + 1))
    if not selected:
        raise ValueError("round_selection selects no round")
    return rounds, frozenset(selected)


def _parse_round_item(item: Any) -> Tuple[int, Optional[int]]:
    """``(start, end)`` of one selection item; ``end`` is ``None`` when open."""
    if isinstance(item, bool):
        raise ValueError(f"round_selection item {item!r} is not a round number")
    if isinstance(item, int):
        start, end = item, item
    elif isinstance(item, str) and item.strip():
        text = item.strip()
        if "-" in text:
            head, _, tail = text.partition("-")
            if not head.strip().isdigit() or (
                tail.strip() and not tail.strip().isdigit()
            ):
                raise ValueError(f"round_selection item {item!r} must be N, A-B or A-")
            start = int(head)
            end = int(tail) if tail.strip() else None
        elif text.isdigit():
            start = end = int(text)
        else:
            raise ValueError(f"round_selection item {item!r} must be N, A-B or A-")
    else:
        raise ValueError(f"round_selection item {item!r} must be N, A-B or A-")
    if start < 1:
        raise ValueError(f"round_selection uses 1-based round numbers; got {item!r}")
    if end is not None and end < start:
        raise ValueError(f"round_selection range {item!r} ends before it starts")
    return start, end


def get_config_dir() -> Path:
    return Path.cwd() / "aao_configs"


def get_run_name() -> str:
    """:func:`describe_run` for the current ``@hydra.main`` application."""
    from hydra.core.hydra_config import HydraConfig

    return describe_run(HydraConfig.get().runtime.choices)


def prepare_task_file(
    cfg: DictConfig, config_cls: BaseModel = TaskFileConfig
) -> TaskFileConfig:
    ComponentRegistry.clear()
    raw = instantiate(prepare_task_config_for_instantiation(cfg))
    # Strip OmegaConf wrappers so downstream code (framework / runtime /
    # backends) only ever sees plain Python types — no module outside this
    # entry-point layer should need to import omegaconf.
    if isinstance(raw, (DictConfig, ListConfig)):
        raw = OmegaConf.to_container(raw, resolve=True)
    return config_cls.model_validate(raw)


def prepare_task_sections(
    cfg: DictConfig,
) -> Tuple[AutoAtomConfig, Dict[str, OperatorConfig]]:
    """The ``task`` and ``task_operators`` that :func:`prepare_task_file` yields.

    Instantiating the task file is what builds the environment, so this reads
    the two sections without instantiating anything: the config goes through
    the same preparation (``object_only`` stripping) and the sections through
    the same field validation. Use it to re-read a task for a backend that
    already exists (see ``MujocoTaskBackend.reconfigure``).
    """
    raw = OmegaConf.to_container(
        prepare_task_config_for_instantiation(cfg), resolve=True
    )
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")
    task = AutoAtomConfig.model_validate(raw.get("task"))
    operators = name_task_operators(
        TypeAdapter(Dict[str, OperatorConfig]).validate_python(
            raw.get("task_operators") or {}
        )
    )
    return task, operators


def run_example_rounds(
    *,
    rounds: int,
    use_input: bool,
    hooks: ExampleLoopHooks,
    selected_rounds: Optional[Collection[int]] = None,
) -> List[ExecutionSummary]:
    """Run ``rounds`` rollouts and summarize the selected ones.

    ``selected_rounds`` holds 1-based round numbers; ``None`` selects all.

    Every reset draws from the stream of its own reset number
    (:func:`auto_atom.utils.seed.reset_generator`), so a round depends on the
    run seed and its number, not on how earlier episodes went. An unselected
    round before a selected one is therefore only reset, not run:
    ``before_round_fn`` and ``reset_fn`` are called, quietly and inside
    ``hooks.quiet_context_fn``. The reset is kept because a non-IID scene
    generator's coverage history is built from earlier resets. Rounds after
    the last selected one cannot affect it and are not run.
    """
    if hooks.max_updates is not None and hooks.max_updates < 0:
        raise ValueError("max_updates must be non-negative or None")
    selected = None if selected_rounds is None else frozenset(selected_rounds)
    if selected is not None:
        outside = sorted(number for number in selected if not 1 <= number <= rounds)
        if outside:
            raise ValueError(f"selected rounds {outside} lie outside 1..{rounds}")

    round_summaries: List[ExecutionSummary] = []
    skip_time = 0.0
    last_round = rounds if selected is None else max(selected)

    for r in range(last_round):
        number = r + 1
        if selected is not None and number not in selected:
            if number == 1 or number - 1 in selected:
                # First of a run of skipped rounds, which ends before the
                # next selected one.
                block_end = number
                while block_end + 1 not in selected:
                    block_end += 1
                span = (
                    f"round {number}"
                    if block_end == number
                    else f"rounds {number}-{block_end}"
                )
                print(f"Skipping {span} of {rounds} (not selected, reset only)...")
                skip_time = perf_counter()
            context = (
                hooks.quiet_context_fn()
                if hooks.quiet_context_fn is not None
                else nullcontext()
            )
            with context:
                if hooks.before_round_fn is not None:
                    hooks.before_round_fn(r)
                hooks.reset_fn()
            if number + 1 in selected:
                print(f"Skipped in {perf_counter() - skip_time:.1f}s.")
                print()
            continue

        if hooks.before_round_fn is not None:
            hooks.before_round_fn(r)

        if rounds > 1:
            print(f"Round {number}/{rounds}")
            print("=" * 50)

        rollout = _rollout(hooks, use_input=use_input)
        update = rollout.update

        summary = hooks.summarize_fn(
            update,
            rollout.steps_used,
            hooks.max_updates,
            rollout.elapsed_time_sec,
        )
        summary.timed_updates = rollout.timed_updates
        summary.env_completion_steps = rollout.env_completion_steps
        summary.env_completion_time_sec = rollout.env_completion_time_sec
        summary.round_number = number
        if summary.env_completion_sim_time_sec is None:
            if summary.updates_used == 0:
                summary.env_completion_sim_time_sec = np.where(
                    rollout.env_completion_steps == 0,
                    0.0,
                    np.nan,
                )
            elif summary.sim_time_sec > 0:
                dt = summary.sim_time_sec / summary.updates_used
                sim_times = np.where(
                    rollout.env_completion_steps >= 0,
                    rollout.env_completion_steps.astype(np.float64) * dt,
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

    if last_round < rounds:
        span = (
            f"round {rounds}"
            if last_round + 1 == rounds
            else f"rounds {last_round + 1}-{rounds}"
        )
        print(f"Skipped {span} of {rounds}: after the last selected round.")
    return round_summaries


@dataclass
class _Rollout:
    update: TaskUpdate
    steps_used: int
    timed_updates: int
    elapsed_time_sec: float
    env_completion_steps: np.ndarray
    env_completion_time_sec: np.ndarray


def _rollout(hooks: ExampleLoopHooks, *, use_input: bool) -> _Rollout:
    """Reset, then step until every environment is done or the limit is hit."""
    print(hooks.reset_label)
    update = hooks.reset_fn()
    if hooks.print_updates:
        pprint(update, sort_dicts=False)
    print(hooks.start_label)
    print()

    batch_size = len(update.stage_name)
    env_completion_steps = np.full(batch_size, -1, dtype=np.int64)
    env_completion_time_sec = np.full(batch_size, np.nan, dtype=np.float64)

    reset_done_mask = np.asarray(update.done, dtype=bool)
    env_completion_steps[reset_done_mask] = 0
    env_completion_time_sec[reset_done_mask] = 0.0
    steps_used = 0
    timed_updates = 0
    elapsed_time_sec = 0.0
    done_mask = reset_done_mask
    all_done = bool(np.all(done_mask))

    while not all_done and (
        hooks.max_updates is None or steps_used < hooks.max_updates
    ):
        step = steps_used
        is_warmup = step == 0
        if use_input:
            input("Press Enter to continue...")

        if is_warmup:
            # The first real update may trigger JIT compilation.
            update = hooks.step_fn(step, update)
        else:
            step_start = perf_counter()
            update = hooks.step_fn(step, update)
            elapsed_time_sec += perf_counter() - step_start
            timed_updates += 1

        steps_used += 1
        done_mask = np.asarray(update.done, dtype=bool)
        newly_done = done_mask & (env_completion_steps < 0)
        env_completion_steps[newly_done] = steps_used
        env_completion_time_sec[newly_done] = elapsed_time_sec
        all_done = bool(np.all(done_mask))

        if hooks.print_updates:
            label = f"Step {step} (warmup):" if is_warmup else f"Step {step}:"
            print(label + "=" * 40)
            pprint(update, sort_dicts=False)

    if not all_done and hooks.max_updates is not None:
        print(f"Reached max_updates={hooks.max_updates}, stopping rollout.")
        if hooks.update_limit_fn is not None:
            update = hooks.update_limit_fn(hooks.max_updates)
            done_mask = np.asarray(update.done, dtype=bool)
            if not bool(np.all(done_mask)):
                raise RuntimeError(
                    "update_limit_fn must return a terminal TaskUpdate for "
                    "every environment"
                )
            newly_done = done_mask & (env_completion_steps < 0)
            env_completion_steps[newly_done] = steps_used
            env_completion_time_sec[newly_done] = elapsed_time_sec

    return _Rollout(
        update=update,
        steps_used=steps_used,
        timed_updates=timed_updates,
        elapsed_time_sec=elapsed_time_sec,
        env_completion_steps=env_completion_steps,
        env_completion_time_sec=env_completion_time_sec,
    )


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
        round_number = summary.round_number or i
        round_joint_success = bool(np.all(summary.final_success))
        tag = "OK" if round_joint_success else "FAIL"
        round_env_success = _count_env_successes(summary)
        batch_size = len(summary.final_success)
        failure_lines = _format_failure_lines(summary)
        timed_updates = _timed_update_count(summary)
        loop_freq = _loop_frequency_hz(summary)
        loop_frequency = (
            f"{loop_freq:.1f} Hz ({timed_updates} timed steps in "
            f"{summary.elapsed_time_sec:.3f}s, {summary.updates_used} total)"
            if loop_freq is not None
            else f"N/A ({timed_updates} timed steps, {summary.updates_used} total)"
        )
        round_payload = {
            "status": tag,
            "success_rate": f"{round_env_success}/{batch_size}",
            "loop_frequency": loop_frequency,
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
        print(f"Round {round_number}")
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
    for index, summary in enumerate(round_summaries, start=1):
        timed_updates = _timed_update_count(summary)
        loop_freq = _loop_frequency_hz(summary)
        entry: Dict[str, Any] = {
            "round": summary.round_number or index,
            "status": "OK" if bool(np.all(summary.final_success)) else "FAIL",
            "success_rate": f"{_count_env_successes(summary)}/{len(summary.final_success)}",
            "loop_frequency_hz": None if loop_freq is None else round(loop_freq, 1),
            "updates_used": summary.updates_used,
            "timed_updates": timed_updates,
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
        failure_records = _format_failure_records(summary)
        if failure_records:
            entry["failure_records"] = failure_records
        rounds_data.append(entry)
    data["rounds"] = rounds_data

    out = Path(path)
    out.write_text(
        json.dumps(
            _json_native(data),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
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


def _format_failure_records(summary: ExecutionSummary) -> List[Dict[str, Any]]:
    return [
        {
            "env_index": record.env_index,
            "stage_index": record.stage_index,
            "stage_name": record.stage_name,
            "operator": record.operator,
            "operation": record.operation,
            "target_object": record.target_object,
            "status": _json_native(record.status),
            "details": _json_native(record.details),
        }
        for record in summary.records
        if record.status == "failed"
        or getattr(record.status, "value", None) == "failed"
    ]


def _json_native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_native(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


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


def _timed_update_count(summary: ExecutionSummary) -> int:
    if summary.timed_updates is None:
        return summary.updates_used
    return summary.timed_updates


def _loop_frequency_hz(summary: ExecutionSummary) -> Optional[float]:
    timed_updates = _timed_update_count(summary)
    if timed_updates == 0 or summary.elapsed_time_sec <= 0:
        return None
    return timed_updates / summary.elapsed_time_sec


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
