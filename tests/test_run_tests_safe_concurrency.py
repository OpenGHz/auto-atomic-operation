"""Contract tests for bounded batch concurrency in ``run_tests_safe``.

The tests replace the subprocess worker with a small, timing-controlled fake.
This keeps the suite fast and makes the scheduler's concurrency and cancellation
semantics observable without starting MuJoCo or systemd scopes.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from scripts import run_tests_safe as runner


def _batches(count: int) -> list[runner._Batch]:
    return [
        runner._Batch(index=index, targets=(f"tests/test_{index}.py",))
        for index in range(1, count + 1)
    ]


def _manifest(batches: list[runner._Batch]) -> dict[str, object]:
    return {
        "batches": [
            {
                "index": batch.index,
                "targets": list(batch.targets),
                "status": "NOT_STARTED",
            }
            for batch in batches
        ]
    }


def _result(batch: runner._Batch, tmp_path: Path, status: str) -> runner._BatchResult:
    """Build a result whose command identifies its batch for ordering checks."""
    return runner._BatchResult(
        status=status,
        returncode=0 if status == "PASSED" else 1,
        elapsed_seconds=0.0,
        log_path=tmp_path / f"batch-{batch.index}.log",
        junit_path=tmp_path / f"batch-{batch.index}.xml",
        command=(str(batch.index),),
    )


def _run_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: runner.SafeTestConfig,
    batches: list[runner._Batch],
    fake_worker,
):
    """Invoke the scheduler at its public-for-tests batch boundary."""
    monkeypatch.setattr(runner, "_run_batch", fake_worker)
    output = tmp_path / "run"
    (output / "logs").mkdir(parents=True)
    (output / "junit").mkdir()
    manifest = _manifest(batches)
    return runner._run_batches(
        config,
        config.repo_root,
        "prlimit",
        batches,
        output,
        manifest,
    ), manifest


def test_default_max_concurrency_is_four_and_can_be_lowered() -> None:
    assert runner.SafeTestConfig().max_concurrency == 4
    assert runner.SafeTestConfig(max_concurrency=1).max_concurrency == 1
    with pytest.raises(ValueError):
        runner.SafeTestConfig(max_concurrency=0)


def test_concurrency_plan_clamps_to_cpu_and_memory_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.SafeTestConfig(
        max_concurrency=4,
        cpu_set="0-1",
        memory_high_mb=512,
        memory_max_mb=1024,
    )
    monkeypatch.setattr(runner, "_available_memory_mb", lambda: 2048)

    plan = runner._plan_concurrency(config, "systemd", _batches(8))

    # CPU allows two workers; the 75% memory budget allows only one.
    assert plan.requested == 4
    assert plan.effective == 1
    assert any("available memory" in reason for reason in plan.reasons)


def test_prlimit_fallback_clamps_concurrency_even_with_many_cpu_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.SafeTestConfig(max_concurrency=4, cpu_set="0-7")
    monkeypatch.setattr(runner, "_available_memory_mb", lambda: None)

    plan = runner._plan_concurrency(config, "prlimit", _batches(8))

    assert plan.effective == 1
    assert any("prlimit" in reason for reason in plan.reasons)


def test_cuda_visibility_clamps_shared_gpu_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.SafeTestConfig(
        max_concurrency=4,
        cpu_set="0-7",
        cuda_visible_devices="0",
    )
    monkeypatch.setattr(runner, "_available_memory_mb", lambda: 64 * 1024)

    plan = runner._plan_concurrency(config, "systemd", _batches(8))

    assert plan.effective == 1
    assert any("CUDA" in reason for reason in plan.reasons)


def test_scheduler_never_exceeds_configured_concurrency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = runner.SafeTestConfig(repo_root=tmp_path, max_concurrency=2)
    batches = _batches(6)
    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_worker(config, root, launcher, batch, output, manifest, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return _result(batch, tmp_path, "PASSED")

    (results, interrupted), _manifest_value = _run_scheduler(
        monkeypatch, tmp_path, config, batches, fake_worker
    )

    assert interrupted is False
    assert peak <= 2
    assert len(results) == len(batches)


def test_scheduler_returns_results_in_batch_index_order_when_completion_is_out_of_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = runner.SafeTestConfig(repo_root=tmp_path, max_concurrency=3)
    batches = _batches(4)
    completion_order: list[int] = []

    def fake_worker(config, root, launcher, batch, output, manifest, **kwargs):
        # Batch 1 is deliberately slow; later batches complete first.
        time.sleep({1: 0.08, 2: 0.01, 3: 0.02, 4: 0.03}[batch.index])
        completion_order.append(batch.index)
        return _result(batch, tmp_path, "PASSED")

    (results, interrupted), manifest = _run_scheduler(
        monkeypatch, tmp_path, config, batches, fake_worker
    )

    assert interrupted is False
    assert completion_order != [1, 2, 3, 4]
    assert [int(result.command[0]) for result in results] == [1, 2, 3, 4]
    assert manifest["concurrency"]["completion_order"] == completion_order
    assert manifest["concurrency"]["peak_active"] <= 3
    entries = manifest["batches"]
    assert [entry["index"] for entry in entries] == [1, 2, 3, 4]


@pytest.mark.parametrize("failure_status", ["TEST_FAILURE", "OOM"])
def test_fail_fast_does_not_start_pending_batches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_status: str,
) -> None:
    """A known ordinary/resource failure stops queued work, even at concurrency 2."""
    config = runner.SafeTestConfig(
        repo_root=tmp_path,
        max_concurrency=2,
        continue_on_failure=False,
    )
    batches = _batches(5)
    started: list[int] = []
    lock = threading.Lock()

    def fake_worker(config, root, launcher, batch, output, manifest, **kwargs):
        with lock:
            started.append(batch.index)
        if batch.index == 1:
            return _result(batch, tmp_path, failure_status)
        # Keep the second initial slot occupied while the first failure is
        # observed, making cancellation of batches 3+ deterministic.
        if batch.index == 2:
            time.sleep(0.12)
        return _result(batch, tmp_path, "PASSED")

    (results, interrupted), _manifest_value = _run_scheduler(
        monkeypatch, tmp_path, config, batches, fake_worker
    )

    assert interrupted is False
    assert [result.status for result in results] == [failure_status, "PASSED"]
    assert set(started) <= {1, 2}


def test_resource_failure_stops_pending_batches_even_when_continue_is_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = runner.SafeTestConfig(
        repo_root=tmp_path,
        max_concurrency=2,
        continue_on_failure=True,
    )
    batches = _batches(5)
    started: list[int] = []
    lock = threading.Lock()

    def fake_worker(config, root, launcher, batch, output, manifest, **kwargs):
        with lock:
            started.append(batch.index)
        if batch.index == 1:
            return _result(batch, tmp_path, "OOM")
        if batch.index == 2:
            time.sleep(0.12)
        return _result(batch, tmp_path, "PASSED")

    (results, interrupted), _manifest_value = _run_scheduler(
        monkeypatch, tmp_path, config, batches, fake_worker
    )

    assert interrupted is False
    assert [result.status for result in results] == ["OOM", "PASSED"]
    assert set(started) <= {1, 2}


def test_continue_on_failure_runs_all_batches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = runner.SafeTestConfig(
        repo_root=tmp_path,
        max_concurrency=2,
        continue_on_failure=True,
    )
    batches = _batches(5)
    started: list[int] = []
    lock = threading.Lock()

    def fake_worker(config, root, launcher, batch, output, manifest, **kwargs):
        with lock:
            started.append(batch.index)
        return _result(
            batch,
            tmp_path,
            "TEST_FAILURE" if batch.index == 1 else "PASSED",
        )

    (results, interrupted), _manifest_value = _run_scheduler(
        monkeypatch, tmp_path, config, batches, fake_worker
    )

    assert interrupted is False
    assert sorted(started) == [1, 2, 3, 4, 5]
    assert len(results) == len(batches)


def test_keyboard_interrupt_cancels_pending_batches_and_reports_interrupted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = runner.SafeTestConfig(repo_root=tmp_path, max_concurrency=2)
    batches = _batches(5)
    started: list[int] = []
    lock = threading.Lock()

    def fake_worker(config, root, launcher, batch, output, manifest, **kwargs):
        with lock:
            started.append(batch.index)
        if batch.index == 1:
            raise KeyboardInterrupt
        time.sleep(0.08)
        return _result(batch, tmp_path, "PASSED")

    (results, interrupted), manifest = _run_scheduler(
        monkeypatch, tmp_path, config, batches, fake_worker
    )

    assert interrupted is True
    assert set(started) <= {1, 2}
    assert len(results) <= 2
    assert all(
        entry["status"] in {"INTERRUPTED", "NOT_STARTED", "RUNNING"}
        for entry in manifest["batches"][2:]
    )
