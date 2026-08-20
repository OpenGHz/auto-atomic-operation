from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from auto_atom.runner import common
from auto_atom.runner.common import (
    ExampleLoopHooks,
    print_final_summary,
    run_example_rounds,
    save_final_summary,
)
from auto_atom.runtime import ExecutionSummary, TaskUpdate


def _update(done: Iterable[bool]) -> TaskUpdate:
    done_array = np.asarray(list(done), dtype=bool)
    batch_size = len(done_array)
    return TaskUpdate(
        stage_index=np.zeros(batch_size, dtype=np.int64),
        stage_name=["move"] * batch_size,
        status=np.where(done_array, "succeeded", "running"),
        done=done_array,
        success=done_array.copy(),
        details=[{} for _ in range(batch_size)],
        phase=[None] * batch_size,
        phase_step=np.full(batch_size, -1, dtype=np.int64),
    )


def _summary(
    update: TaskUpdate,
    updates_used: int,
    max_updates: int | None,
    elapsed_time_sec: float,
) -> ExecutionSummary:
    batch_size = len(update.done)
    return ExecutionSummary(
        total_stages=1,
        max_updates=max_updates,
        updates_used=updates_used,
        completed_stage_count=np.asarray(update.done, dtype=np.int64),
        final_stage_index=np.zeros(batch_size, dtype=np.int64),
        final_stage_name=list(update.stage_name),
        final_status=np.asarray(update.status, dtype=object),
        final_done=np.asarray(update.done, dtype=bool),
        final_success=np.asarray(update.success, dtype=bool),
        elapsed_time_sec=elapsed_time_sec,
        records=[],
    )


def _run_scripted(
    *,
    reset_done: Iterable[bool],
    step_done: Iterable[Iterable[bool]],
    max_updates: int | None,
) -> tuple[ExecutionSummary, list[int]]:
    updates = iter(_update(done) for done in step_done)
    step_indices: list[int] = []

    def step_fn(step: int, _previous: TaskUpdate) -> TaskUpdate:
        step_indices.append(step)
        try:
            return next(updates)
        except StopIteration as exc:  # pragma: no cover - clearer failure diagnostic
            raise AssertionError("rollout performed an unexpected update") from exc

    summaries = run_example_rounds(
        rounds=1,
        use_input=False,
        hooks=ExampleLoopHooks(
            reset_fn=lambda: _update(reset_done),
            step_fn=step_fn,
            summarize_fn=_summary,
            records_fn=list,
            max_updates=max_updates,
        ),
    )
    return summaries[0], step_indices


@pytest.mark.parametrize(
    (
        "max_updates",
        "script",
        "expected_step_indices",
        "expected_timed_updates",
        "expected_done",
    ),
    [
        pytest.param(0, [], [], 0, False, id="zero-runs-no-updates"),
        pytest.param(1, [[False]], [0], 0, False, id="one-runs-only-warmup"),
        pytest.param(
            None,
            [[False], [False], [True]],
            [0, 1, 2],
            2,
            True,
            id="none-runs-until-complete",
        ),
    ],
)
def test_max_updates_is_total_step_budget(
    max_updates: int | None,
    script: list[list[bool]],
    expected_step_indices: list[int],
    expected_timed_updates: int,
    expected_done: bool,
) -> None:
    summary, step_indices = _run_scripted(
        reset_done=[False],
        step_done=script,
        max_updates=max_updates,
    )

    assert step_indices == expected_step_indices
    assert summary.updates_used == len(expected_step_indices)
    assert summary.timed_updates == expected_timed_updates
    assert summary.final_done.tolist() == [expected_done]


def test_negative_max_updates_is_rejected() -> None:
    hooks = ExampleLoopHooks(
        reset_fn=lambda: _update([False]),
        step_fn=lambda _step, update: update,
        summarize_fn=_summary,
        records_fn=list,
        max_updates=-1,
    )

    with pytest.raises(ValueError, match=r"max_updates.*(?:non-negative|>= 0)"):
        run_example_rounds(rounds=1, use_input=False, hooks=hooks)


def test_completion_metrics_track_each_environment_first_done_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    updates = iter(
        [
            _update([True, True, False]),
            _update([True, True, True]),
        ]
    )
    step_indices: list[int] = []

    def step_fn(step: int, _previous: TaskUpdate) -> TaskUpdate:
        step_indices.append(step)
        clock.advance(100.0 if step == 0 else 2.5)
        return next(updates)

    monkeypatch.setattr(common, "perf_counter", clock)
    summary = run_example_rounds(
        rounds=1,
        use_input=False,
        hooks=ExampleLoopHooks(
            reset_fn=lambda: _update([True, False, False]),
            step_fn=step_fn,
            summarize_fn=_summary,
            records_fn=list,
            max_updates=None,
        ),
    )[0]

    assert step_indices == [0, 1]
    assert summary.updates_used == 2
    assert summary.timed_updates == 1
    assert summary.env_completion_steps.tolist() == [0, 1, 2]
    assert summary.env_completion_time_sec.tolist() == pytest.approx([0.0, 0.0, 2.5])


@dataclass
class _Clock:
    now: float = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


def test_elapsed_time_only_measures_non_warmup_step_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    updates = iter([_update([False]), _update([True])])
    events: list[tuple[str, int | None]] = []

    def step_fn(_step: int, _previous: TaskUpdate) -> TaskUpdate:
        events.append(("step", _step))
        clock.advance(100.0 if _step == 0 else 2.5)
        return next(updates)

    def slow_input(_prompt: str) -> str:
        events.append(("input", None))
        clock.advance(20.0)
        return ""

    def slow_pprint(*_args, **_kwargs) -> None:
        clock.advance(10.0)

    monkeypatch.setattr(common, "perf_counter", clock)
    monkeypatch.setattr("builtins.input", slow_input)
    monkeypatch.setattr(common, "pprint", slow_pprint)

    summary = run_example_rounds(
        rounds=1,
        use_input=True,
        hooks=ExampleLoopHooks(
            reset_fn=lambda: _update([False]),
            step_fn=step_fn,
            summarize_fn=_summary,
            records_fn=list,
            max_updates=None,
        ),
    )[0]

    assert summary.updates_used == 2
    assert summary.timed_updates == 1
    assert summary.elapsed_time_sec == pytest.approx(2.5)
    assert summary.env_completion_time_sec.tolist() == pytest.approx([2.5])
    assert events == [
        ("input", None),
        ("step", 0),
        ("input", None),
        ("step", 1),
    ]


def test_print_updates_false_suppresses_step_output(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    updates = iter([_update([False]), _update([True])])
    pprint_values: list[object] = []
    monkeypatch.setattr(
        common,
        "pprint",
        lambda value, **_kwargs: pprint_values.append(value),
    )

    summary = run_example_rounds(
        rounds=1,
        use_input=False,
        hooks=ExampleLoopHooks(
            reset_fn=lambda: _update([False]),
            step_fn=lambda _step, _previous: next(updates),
            summarize_fn=_summary,
            records_fn=list,
            max_updates=2,
            print_updates=False,
        ),
    )[0]

    output = capsys.readouterr().out
    assert "Step 0 (warmup):" not in output
    assert "Step 1:" not in output
    assert len(pprint_values) == 1
    assert pprint_values[0] is summary


def test_reported_loop_frequency_excludes_warmup(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    summary = _summary(
        _update([True]),
        updates_used=2,
        max_updates=None,
        elapsed_time_sec=2.5,
    )
    summary.timed_updates = 1

    print_final_summary([summary])
    terminal_output = capsys.readouterr().out
    output_path = save_final_summary([summary], tmp_path / "summary.json")
    saved = json.loads(output_path.read_text())

    assert "0.4 Hz" in terminal_output
    assert saved["rounds"][0]["loop_frequency_hz"] == 0.4
    assert saved["rounds"][0]["timed_updates"] == 1


def test_max_updates_message_is_only_printed_for_an_incomplete_rollout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _run_scripted(
        reset_done=[False],
        step_done=[[False]],
        max_updates=1,
    )
    capped_output = capsys.readouterr().out

    _run_scripted(
        reset_done=[False],
        step_done=[[True]],
        max_updates=1,
    )
    completed_output = capsys.readouterr().out

    assert "Reached max_updates=1, stopping rollout." in capped_output
    assert "Reached max_updates" not in completed_output
