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
from auto_atom.runtime import (
    ExecutionRecord,
    ExecutionSummary,
    StageExecutionStatus,
    TaskUpdate,
)


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


def test_update_limit_callback_terminalizes_incomplete_rollout() -> None:
    callback_limits: list[int] = []

    def terminate(limit: int) -> TaskUpdate:
        callback_limits.append(limit)
        update = _update([True])
        update.status[:] = StageExecutionStatus.FAILED
        update.success[:] = False
        return update

    summary = run_example_rounds(
        rounds=1,
        use_input=False,
        hooks=ExampleLoopHooks(
            reset_fn=lambda: _update([False]),
            step_fn=lambda _step, update: update,
            summarize_fn=_summary,
            records_fn=list,
            max_updates=0,
            update_limit_fn=terminate,
        ),
    )[0]

    assert callback_limits == [0]
    assert summary.final_done.tolist() == [True]
    assert summary.final_success.tolist() == [False]
    assert summary.env_completion_steps.tolist() == [0]


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


def test_saved_summary_preserves_json_native_contact_failure_records(
    tmp_path: Path,
) -> None:
    summary = _summary(
        _update([False]),
        updates_used=3,
        max_updates=10,
        elapsed_time_sec=0.25,
    )
    summary.final_done[:] = True
    summary.final_status[:] = StageExecutionStatus.FAILED
    summary.records = [
        ExecutionRecord(
            env_index=0,
            stage_index=0,
            stage_name="pick_handle",
            operator="arm",
            operation="pick",
            target_object="door__door_handle",
            blocking=True,
            status=StageExecutionStatus.FAILED,
            details={
                "failure_reason": "gripper was blocked before grasping the handle",
                "operator_contact_snapshot": {
                    "status": "observed",
                    "contacts": [
                        {
                            "operator_body": "eef_L53",
                            "operator_geom": "eef_left_finger_collision",
                            "other_body": "door__door_panel",
                            "other_geom": "door__door_panel_collision",
                            "position_world_m": (
                                np.float64(1.508),
                                np.float64(0.574),
                                np.float64(-0.144),
                            ),
                            "signed_distance_m": np.float64(-0.000576),
                            "penetration_depth_m": np.float64(0.000576),
                            "normal_force_n": np.float32(107.31),
                            "tangential_force_n": np.float32(50.13),
                            "nonfinite_probe": np.float64(np.nan),
                            "complex_probe": np.complex64(1 + 2j),
                        }
                    ],
                },
            },
        )
    ]

    output_path = save_final_summary([summary], tmp_path / "summary.json")
    saved = json.loads(output_path.read_text())

    failure_records = saved["rounds"][0]["failure_records"]
    assert failure_records == [
        {
            "env_index": 0,
            "stage_index": 0,
            "stage_name": "pick_handle",
            "operator": "arm",
            "operation": "pick",
            "target_object": "door__door_handle",
            "status": "failed",
            "details": {
                "failure_reason": "gripper was blocked before grasping the handle",
                "operator_contact_snapshot": {
                    "status": "observed",
                    "contacts": [
                        {
                            "operator_body": "eef_L53",
                            "operator_geom": "eef_left_finger_collision",
                            "other_body": "door__door_panel",
                            "other_geom": "door__door_panel_collision",
                            "position_world_m": [1.508, 0.574, -0.144],
                            "signed_distance_m": -0.000576,
                            "penetration_depth_m": 0.000576,
                            "normal_force_n": pytest.approx(107.31),
                            "tangential_force_n": pytest.approx(50.13),
                            "nonfinite_probe": "nan",
                            "complex_probe": "(1+2j)",
                        }
                    ],
                },
            },
        }
    ]


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


# ----------------------------------------------------------------------
# Round selection
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "rounds", "expected"),
    [
        pytest.param(None, None, (1, None), id="default-one-round"),
        pytest.param(None, 5, (5, None), id="all-rounds"),
        pytest.param(13, None, (13, frozenset({13})), id="int-ends-the-run"),
        pytest.param("13", 20, (20, frozenset({13})), id="string-int"),
        pytest.param(
            [3, 7, "10-12"], None, (12, frozenset({3, 7, 10, 11, 12})), id="list"
        ),
        pytest.param("3,7,10-12", None, (12, frozenset({3, 7, 10, 11, 12})), id="csv"),
        pytest.param("8-", 10, (10, frozenset({8, 9, 10})), id="open-range"),
        pytest.param(["2-4", 3], None, (4, frozenset({2, 3, 4})), id="overlap"),
    ],
)
def test_round_selection_is_parsed_into_1_based_rounds(value, rounds, expected) -> None:
    assert common.parse_round_selection(value, rounds) == expected


@pytest.mark.parametrize(
    ("value", "rounds", "message"),
    [
        pytest.param([], None, "selects no round", id="empty"),
        pytest.param(0, None, "1-based", id="zero"),
        pytest.param(-2, None, "1-based", id="negative"),
        pytest.param("5-3", None, "ends before it starts", id="reversed"),
        pytest.param("8-", None, "needs rounds", id="open-without-rounds"),
        pytest.param(12, 10, "only 10 rounds", id="beyond-rounds"),
        pytest.param("a", None, "must be N, A-B or A-", id="text"),
        pytest.param(True, None, "not a round number", id="bool"),
        pytest.param(None, 0, "at least 1", id="zero-rounds"),
    ],
)
def test_invalid_round_selections_are_rejected(value, rounds, message) -> None:
    with pytest.raises(ValueError, match=message):
        common.parse_round_selection(value, rounds)


def _scripted_rounds(
    rounds: int,
    *,
    selected_rounds=None,
    use_input: bool = False,
) -> tuple[list[ExecutionSummary], list[str]]:
    """Each round resets and needs two updates; every call is logged."""
    calls: list[str] = []
    round_counter = {"round": 0}

    def before_round(r: int) -> None:
        round_counter["round"] = r + 1
        calls.append(f"before {r + 1}")

    def reset_fn() -> TaskUpdate:
        calls.append(f"reset {round_counter['round']}")
        return _update([False])

    def step_fn(step: int, _previous: TaskUpdate) -> TaskUpdate:
        calls.append(f"step {round_counter['round']}.{step}")
        return _update([step >= 1])

    class _Quiet:
        def __enter__(self):
            calls.append(f"quiet {round_counter['round'] + 1}")

        def __exit__(self, *exc):
            calls.append("loud")

    summaries = run_example_rounds(
        rounds=rounds,
        use_input=use_input,
        selected_rounds=selected_rounds,
        hooks=ExampleLoopHooks(
            reset_fn=reset_fn,
            step_fn=step_fn,
            summarize_fn=_summary,
            records_fn=list,
            before_round_fn=before_round,
            quiet_context_fn=_Quiet,
        ),
    )
    return summaries, calls


def test_unselected_rounds_are_only_reset(capsys) -> None:
    summaries, calls = _scripted_rounds(4, selected_rounds={2, 4})
    out = capsys.readouterr().out

    # A skipped round prepares and resets, quietly, but takes no update; a
    # selected round makes exactly the calls of a run without a selection.
    assert calls == [
        "quiet 1",
        "before 1",
        "reset 1",
        "loud",
        "before 2",
        "reset 2",
        "step 2.0",
        "step 2.1",
        "quiet 3",
        "before 3",
        "reset 3",
        "loud",
        "before 4",
        "reset 4",
        "step 4.0",
        "step 4.1",
    ]
    assert [summary.round_number for summary in summaries] == [2, 4]

    assert "Skipping round 1 of 4 (not selected, reset only)..." in out
    assert "Skipping round 3 of 4 (not selected, reset only)..." in out
    assert "Round 1/4" not in out and "Round 3/4" not in out
    assert "Round 2/4" in out and "Round 4/4" in out


def test_rounds_after_the_last_selected_one_are_not_run(capsys) -> None:
    summaries, calls = _scripted_rounds(5, selected_rounds={2})

    assert [summary.round_number for summary in summaries] == [2]
    assert not any(call.endswith((" 3", " 4", " 5")) for call in calls)
    assert "Skipped rounds 3-5 of 5: after the last selected round." in (
        capsys.readouterr().out
    )


def test_a_run_of_skipped_rounds_is_announced_once(capsys) -> None:
    _scripted_rounds(5, selected_rounds={5})

    out = capsys.readouterr().out
    assert out.count("Skipping") == 1
    assert "Skipping rounds 1-4 of 5 (not selected, reset only)..." in out


def test_skipped_rounds_do_not_wait_for_input(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": prompts.append(prompt))

    _scripted_rounds(3, selected_rounds={3}, use_input=True)

    # Two updates of round 3 only.
    assert len(prompts) == 2


def test_selected_rounds_outside_the_run_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"outside 1\.\.3"):
        _scripted_rounds(3, selected_rounds={4})


def test_summaries_report_the_selected_round_numbers(capsys, tmp_path: Path) -> None:
    summaries, _ = _scripted_rounds(4, selected_rounds={2, 4})
    capsys.readouterr()

    print_final_summary(summaries)
    out = capsys.readouterr().out
    assert "Round 2\n" in out and "Round 4\n" in out and "Round 1\n" not in out

    path = save_final_summary(summaries, tmp_path / "summary.json")
    rounds = json.loads(path.read_text())["rounds"]
    assert [entry["round"] for entry in rounds] == [2, 4]
