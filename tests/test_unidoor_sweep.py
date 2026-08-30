"""Tests for the bounded UniDoor matrix sweep CLI."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

from auto_atom.runner.unidoor_sweep import (
    FAILURES_NAME,
    MANIFEST_NAME,
    UniDoorSweepConfig,
    _append_resume_batches,
    _build_manifest,
    _latest_sweep_dir,
    _report_exit_code,
    _run_batches,
    _summary_result,
    _stream_command,
    _unresolved_jobs,
    _write_json,
    aggregate_sweep,
    load_catalog,
    parse_config,
    run_unidoor_sweep,
)


def _asset_package(
    tmp_path: Path,
    *,
    doors: tuple[str, ...] = ("D001", "D002"),
    handles: tuple[str, ...] = ("H001", "H002"),
) -> Path:
    payload_root = tmp_path / "payload"
    payload_root.mkdir()
    product_space = {
        "schema_version": "test/v1",
        "compiler_version": "test",
        "combination_space": {
            "expected_count": len(doors) * len(handles),
        },
        "components": {
            "doors": [{"asset_id": asset_id, "status": "pass"} for asset_id in doors],
            "handles": [
                {"asset_id": asset_id, "status": "pass"} for asset_id in handles
            ],
        },
    }
    (payload_root / "product_space.json").write_text(
        json.dumps(product_space), encoding="utf-8"
    )
    package_path = tmp_path / "scene_asset_package.json"
    package_path.write_text(
        json.dumps(
            {
                "schema": "aao.scene-asset-package/v1",
                "payload_root": {"uri": "payload"},
                "components": {"index": "product_space.json"},
            }
        ),
        encoding="utf-8",
    )
    return package_path


def _summary(
    *,
    status: str,
    success: bool,
    stage: str = "",
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "rounds": [
            {
                "status": status,
                "final_success": [success],
                "final_stage": [stage],
                "failure_reasons": [reason] if reason else None,
            }
        ]
    }


def _door_panel_contact_failure_record() -> dict[str, object]:
    return {
        "env_index": 0,
        "stage_index": 0,
        "stage_name": "pick_handle",
        "operator": "arm",
        "operation": "pick",
        "target_object": "door__door_handle",
        "status": "failed",
        "details": {
            "event": "eef_timeout",
            "operator_contact_snapshot": {
                "status": "observed",
                "contacts": [
                    {
                        "operator_body": "eef_L53",
                        "operator_geom": "eef_left_finger_collision",
                        "other_body": "door__door_panel",
                        "other_geom": "door__door_panel_collision",
                    }
                ],
            },
        },
    }


def test_parse_config_accepts_kebab_case_and_comma_lists() -> None:
    config = parse_config(
        [
            "--doors",
            "D002,D001",
            "--handles",
            "H001",
            "--exclude-doors",
            "D001",
            "--exclude-handles",
            "H002",
            "--max-updates",
            "0",
            "--rounds",
            "2",
            "--launcher-batch-size",
            "3",
            "--max-concurrency",
            "2",
            "--dry-run",
        ]
    )

    assert config.doors == ["D002", "D001"]
    assert config.handles == ["H001"]
    assert config.exclude_doors == ["D001"]
    assert config.exclude_handles == ["H002"]
    assert config.max_updates == 0
    assert config.rounds == 2
    assert config.launcher_batch_size == 3
    assert config.max_concurrency == 2
    assert config.stop_on_failure is True
    assert config.verbose is False
    assert config.dry_run is True

    assert parse_config(["--no-stop-on-failure"]).stop_on_failure is False
    assert parse_config(["--verbose"]).verbose is True
    assert parse_config(["--resume-latest"]).resume_latest is True

    with pytest.raises(ValueError, match="greater than 0"):
        parse_config(["--rounds", "0"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_config(["--resume", "run", "--resume-latest"])
    with pytest.raises(ValueError, match="cannot be combined with --output-dir"):
        parse_config(["--resume-latest", "--output-dir", "run"])
    with pytest.raises(ValueError, match="only valid when planning"):
        parse_config(["--resume-latest", "--dry-run"])


def test_load_catalog_follows_asset_package_component_index(tmp_path: Path) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D002", "D001"),
        handles=("H004", "HL001"),
    )

    catalog = load_catalog(UniDoorSweepConfig(asset_package=package_path))

    assert catalog.doors == ("D002", "D001")
    assert catalog.handles == ("H004", "HL001")
    assert catalog.product_space == tmp_path / "payload" / "product_space.json"
    assert len(catalog.product_space_sha256) == 64


@pytest.mark.parametrize(
    ("doors", "handles", "exclude_doors", "exclude_handles", "error"),
    [
        (
            ["D001", "D001"],
            ["H001"],
            [],
            [],
            "Duplicate requested door",
        ),
        (["D999"], ["H001"], [], [], "Unknown door"),
        (["D001"], ["H999"], [], [], "Unknown handle"),
        (["D001"], ["H001"], ["D999"], [], "Unknown excluded door"),
        (
            ["D001"],
            ["H001"],
            [],
            ["H002", "H002"],
            "Duplicate excluded handle",
        ),
    ],
)
def test_manifest_rejects_duplicate_and_unknown_selections(
    tmp_path: Path,
    doors: list[str],
    handles: list[str],
    exclude_doors: list[str],
    exclude_handles: list[str],
    error: str,
) -> None:
    package_path = _asset_package(tmp_path)
    config = UniDoorSweepConfig(
        asset_package=package_path,
        doors=doors,
        handles=handles,
        exclude_doors=exclude_doors,
        exclude_handles=exclude_handles,
    )

    with pytest.raises(ValueError, match=error):
        _build_manifest(config, load_catalog(config), tmp_path / "sweep")


def test_manifest_applies_exclusions_after_positive_selection(tmp_path: Path) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001", "D002"),
        handles=("H001", "H002", "H003"),
    )
    config = UniDoorSweepConfig(
        asset_package=package_path,
        doors=["D002", "D001"],
        handles=["H003", "H001", "H002"],
        exclude_doors=["D001"],
        exclude_handles=["H001"],
    )

    manifest = _build_manifest(config, load_catalog(config), tmp_path / "sweep")

    assert manifest["selection"] == {
        "doors": ["D002"],
        "handles": ["H003", "H002"],
        "excluded_doors": ["D001"],
        "excluded_handles": ["H001"],
        "combination_count": 2,
    }
    assert [(job["door_id"], job["handle_id"]) for job in manifest["jobs"]] == [
        ("D002", "H003"),
        ("D002", "H002"),
    ]


@pytest.mark.parametrize(
    ("exclude_doors", "exclude_handles", "error"),
    [
        (["D001", "D002"], [], "Door selection is empty"),
        ([], ["H001", "H002"], "Handle selection is empty"),
    ],
)
def test_manifest_rejects_exclusions_that_empty_a_dimension(
    tmp_path: Path,
    exclude_doors: list[str],
    exclude_handles: list[str],
    error: str,
) -> None:
    package_path = _asset_package(tmp_path)
    config = UniDoorSweepConfig(
        asset_package=package_path,
        exclude_doors=exclude_doors,
        exclude_handles=exclude_handles,
    )

    with pytest.raises(ValueError, match=error):
        _build_manifest(config, load_catalog(config), tmp_path / "sweep")


def test_manifest_uses_bounded_joblib_batches(tmp_path: Path) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001", "D002"),
        handles=("H001", "H002", "H003"),
    )
    config = UniDoorSweepConfig(
        asset_package=package_path,
        launcher_batch_size=2,
        max_concurrency=2,
        stop_on_failure=False,
    )

    manifest = _build_manifest(config, load_catalog(config), tmp_path / "sweep")

    assert [
        (job["job_num"], job["door_id"], job["handle_id"]) for job in manifest["jobs"]
    ] == [
        (0, "D001", "H001"),
        (1, "D001", "H002"),
        (2, "D001", "H003"),
        (3, "D002", "H001"),
        (4, "D002", "H002"),
        (5, "D002", "H003"),
    ]
    assert len(manifest["batches"]) == 4
    first_command = manifest["batches"][0]["argv"]
    assert first_command[:5] == [
        sys.executable,
        "-u",
        "-m",
        "auto_atom.runner.demo",
        "--multirun",
    ]
    assert "handle_id=H001,H002" in first_command
    assert "hydra/launcher=joblib" in first_command
    assert "hydra.launcher.n_jobs=2" in first_command
    assert "hydra.launcher.pre_dispatch=2" in first_command
    assert "hydra.launcher.batch_size=1" in first_command
    assert "env.batch_size=1" in first_command
    assert "env.viewer=null" in first_command
    assert "hydra.sweeper.max_batch_size=null" in first_command
    assert "hydra.sweeper.max_batch_size=1" not in first_command


def test_manifest_uses_bounded_parallel_waves_when_stopping_on_failure(
    tmp_path: Path,
) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001",),
        handles=("H001", "H002", "H003"),
    )
    config = UniDoorSweepConfig(
        asset_package=package_path,
        launcher_batch_size=6,
        max_concurrency=2,
    )

    manifest = _build_manifest(config, load_catalog(config), tmp_path / "sweep")

    assert len(manifest["batches"]) == 2
    assert manifest["execution"] == {
        "requested_launcher_batch_size": 6,
        "effective_launcher_batch_size": 2,
        "max_concurrency": 2,
        "launcher_policy": "joblib_when_parallel",
        "stop_on_failure": True,
    }
    assert [batch["handle_ids"] for batch in manifest["batches"]] == [
        ["H001", "H002"],
        ["H003"],
    ]
    assert "hydra/launcher=joblib" in manifest["batches"][0]["argv"]
    assert manifest["batches"][0]["launcher"] == "joblib"
    assert "hydra/launcher=basic" in manifest["batches"][1]["argv"]
    assert manifest["batches"][1]["launcher"] == "basic"
    assert manifest["progress"]["next_job_num"] == 0
    assert {job["status"] for job in manifest["jobs"]} == {"PENDING"}


def test_manifest_quotes_hydra_output_paths(tmp_path: Path) -> None:
    package_path = _asset_package(tmp_path)
    config = UniDoorSweepConfig(asset_package=package_path)
    sweep_dir = tmp_path / "中文,comma path with spaces"

    manifest = _build_manifest(config, load_catalog(config), sweep_dir)

    command = manifest["batches"][0]["argv"]
    assert f'hydra.sweep.dir="{sweep_dir}/batches/0000__D001"' in command


def test_stream_command_tees_stdout_and_stderr(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "sweep.log"
    returncode = _stream_command(
        [
            sys.executable,
            "-c",
            "import sys; print('sentinel-out'); print('sentinel-err', file=sys.stderr)",
        ],
        workdir=tmp_path,
        environment={},
        log_path=log_path,
    )

    captured = capsys.readouterr().out
    assert returncode == 0
    assert "sentinel-out" in captured
    assert "sentinel-err" in captured
    assert "sentinel-out" in log_path.read_text(encoding="utf-8")
    assert "sentinel-err" in log_path.read_text(encoding="utf-8")


def test_stream_command_can_log_without_echoing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "sweep.log"

    returncode = _stream_command(
        [sys.executable, "-c", "print('log-only-sentinel')"],
        workdir=tmp_path,
        environment={},
        log_path=log_path,
        echo=False,
    )

    assert returncode == 0
    assert "log-only-sentinel" not in capsys.readouterr().out
    assert "log-only-sentinel" in log_path.read_text(encoding="utf-8")


def test_aggregate_classifies_all_result_states(tmp_path: Path) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001",),
        handles=("H001", "H002", "H003", "H004", "H005"),
    )
    sweep_dir = tmp_path / "sweep"
    config = UniDoorSweepConfig(asset_package=package_path)
    manifest = _build_manifest(config, load_catalog(config), sweep_dir)
    manifest["status"] = "FINISHED"
    sweep_dir.mkdir()
    _write_json(sweep_dir / MANIFEST_NAME, manifest)

    jobs = manifest["jobs"]
    success_dir = sweep_dir / jobs[0]["relative_dir"]
    success_dir.mkdir(parents=True)
    _write_json(success_dir / "summary.json", _summary(status="OK", success=True))

    failure_dir = sweep_dir / jobs[1]["relative_dir"]
    failure_dir.mkdir(parents=True)
    _write_json(
        failure_dir / "summary.json",
        _summary(
            status="FAIL",
            success=False,
            stage="pick_handle",
            reason="primitive timeout",
        ),
    )

    no_summary_dir = sweep_dir / jobs[2]["relative_dir"]
    no_summary_dir.mkdir(parents=True)

    invalid_dir = sweep_dir / jobs[4]["relative_dir"]
    invalid_dir.mkdir(parents=True)
    (invalid_dir / "summary.json").write_text("not json", encoding="utf-8")

    report = aggregate_sweep(sweep_dir)

    assert report["counts"] == {
        "SUCCESS": 1,
        "TASK_FAILURE": 1,
        "NO_SUMMARY": 1,
        "NOT_STARTED": 1,
        "INVALID_SUMMARY": 1,
        "LAUNCHER_FAILURE": 0,
    }
    task_failure = report["results"][1]
    assert task_failure["final_stage"] == ["pick_handle"]
    assert task_failure["failure_reasons"] == ["primitive timeout"]
    assert task_failure["reproduce_command"].startswith("env ")
    assert "door_id=D001" in task_failure["reproduce_command"]
    assert "handle_id=H002" in task_failure["reproduce_command"]

    with (sweep_dir / FAILURES_NAME).open(encoding="utf-8", newline="") as stream:
        failure_rows = list(csv.DictReader(stream))
    assert len(failure_rows) == 4
    assert {row["status"] for row in failure_rows} == {
        "TASK_FAILURE",
        "NO_SUMMARY",
        "NOT_STARTED",
        "INVALID_SUMMARY",
    }


def test_summary_result_preserves_failure_records(tmp_path: Path) -> None:
    job = {
        "job_num": 0,
        "batch_num": 0,
        "door_id": "D026",
        "handle_id": "H006",
        "relative_dir": "job",
    }
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    failure_record = _door_panel_contact_failure_record()
    _write_json(
        job_dir / "summary.json",
        {
            "rounds": [
                {
                    "status": "FAIL",
                    "final_success": [False],
                    "final_stage": ["pick_handle"],
                    "failure_reasons": ["primitive timeout"],
                    "failure_records": [failure_record],
                },
                {
                    "status": "FAIL",
                    "final_success": [False],
                    "final_stage": ["pick_handle"],
                    "failure_reasons": ["primitive timeout"],
                    "failure_records": [failure_record],
                },
            ]
        },
    )

    result = _summary_result(job, tmp_path, expected_rounds=2)

    assert result["status"] == "TASK_FAILURE"
    assert result["failure_records"] == [
        {**failure_record, "round_index": 0},
        {**failure_record, "round_index": 1},
    ]


def test_failures_csv_includes_operator_contact_pairs(tmp_path: Path) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D026",),
        handles=("H006",),
    )
    sweep_dir = tmp_path / "sweep"
    config = UniDoorSweepConfig(asset_package=package_path)
    manifest = _build_manifest(config, load_catalog(config), sweep_dir)
    manifest["status"] = "FINISHED"
    sweep_dir.mkdir()
    _write_json(sweep_dir / MANIFEST_NAME, manifest)

    job_dir = sweep_dir / manifest["jobs"][0]["relative_dir"]
    job_dir.mkdir(parents=True)
    _write_json(
        job_dir / "summary.json",
        {
            "rounds": [
                {
                    "status": "FAIL",
                    "final_success": [False],
                    "final_stage": ["pick_handle"],
                    "failure_reasons": ["primitive timeout"],
                    "failure_records": [_door_panel_contact_failure_record()],
                }
            ]
        },
    )

    aggregate_sweep(sweep_dir)

    with (sweep_dir / FAILURES_NAME).open(encoding="utf-8", newline="") as stream:
        failure_rows = list(csv.DictReader(stream))
    assert len(failure_rows) == 1
    assert (
        failure_rows[0]["contact_pairs"]
        == "eef_left_finger_collision -> door__door_panel_collision"
    )


@pytest.mark.parametrize(
    "round_payload",
    [
        {"status": "OK", "final_stage": [""]},
        {"status": "OK", "final_success": [], "final_stage": [""]},
        {"status": "OK", "final_success": [False], "final_stage": [""]},
        {"status": "FAIL", "final_success": [True], "final_stage": [""]},
    ],
)
def test_summary_rejects_inconsistent_final_success(
    tmp_path: Path,
    round_payload: dict[str, object],
) -> None:
    job = {
        "job_num": 0,
        "batch_num": 0,
        "door_id": "D001",
        "handle_id": "H001",
        "relative_dir": "job",
    }
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_json(job_dir / "summary.json", {"rounds": [round_payload]})

    result = _summary_result(job, tmp_path, expected_rounds=1)

    assert result["status"] == "INVALID_SUMMARY"


def test_summary_rejects_wrong_round_count(tmp_path: Path) -> None:
    job = {
        "job_num": 0,
        "batch_num": 0,
        "door_id": "D001",
        "handle_id": "H001",
        "relative_dir": "job",
    }
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    _write_json(
        job_dir / "summary.json",
        {"rounds": [{"status": "OK", "final_success": [True], "final_stage": [""]}]},
    )

    result = _summary_result(job, tmp_path, expected_rounds=2)

    assert result["status"] == "INVALID_SUMMARY"
    assert "expected 2" in result["failure_reasons"][0]


def test_resume_versions_only_unresolved_job_outputs(tmp_path: Path) -> None:
    package_path = _asset_package(tmp_path, doors=("D001",))
    sweep_dir = tmp_path / "sweep"
    config = UniDoorSweepConfig(asset_package=package_path)
    manifest = _build_manifest(config, load_catalog(config), sweep_dir)
    sweep_dir.mkdir()
    completed_job, unresolved_job = manifest["jobs"]
    completed_dir = sweep_dir / completed_job["relative_dir"]
    completed_dir.mkdir(parents=True)
    _write_json(
        completed_dir / "summary.json",
        _summary(status="OK", success=True),
    )
    old_completed_path = completed_job["relative_dir"]
    old_unresolved_path = unresolved_job["relative_dir"]

    pending = _unresolved_jobs(manifest, sweep_dir)
    manifest["resume_count"] = 1
    batch_numbers = _append_resume_batches(manifest, sweep_dir, pending)

    assert [job["handle_id"] for job in pending] == ["H002"]
    assert completed_job["relative_dir"] == old_completed_path
    assert unresolved_job["relative_dir"] != old_unresolved_path
    assert unresolved_job["relative_dir"].startswith("resume/0001/")
    assert len(unresolved_job["attempts"]) == 2
    assert len(batch_numbers) == 1
    new_batch = manifest["batches"][-1]
    assert new_batch["handle_ids"] == ["H002"]
    assert "handle_id=H002" in new_batch["argv"]


def test_run_stops_on_task_failure_and_persists_resume_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001",),
        handles=("H001", "H002", "H003", "H004", "H005"),
    )
    sweep_dir = tmp_path / "sweep"
    config = UniDoorSweepConfig(asset_package=package_path)
    manifest = _build_manifest(config, load_catalog(config), sweep_dir)
    sweep_dir.mkdir()
    _write_json(sweep_dir / MANIFEST_NAME, manifest)
    launched: list[list[str]] = []

    def fake_stream(command: list[str], **_kwargs: object) -> int:
        handle_ids = next(
            argument.removeprefix("handle_id=")
            for argument in command
            if argument.startswith("handle_id=")
        ).split(",")
        launched.append(handle_ids)
        for handle_id in handle_ids:
            job = next(job for job in manifest["jobs"] if job["handle_id"] == handle_id)
            job_dir = sweep_dir / job["relative_dir"]
            job_dir.mkdir(parents=True)
            status = "FAIL" if handle_id == "H002" else "OK"
            _write_json(
                job_dir / "summary.json",
                _summary(
                    status=status,
                    success=status == "OK",
                    reason="grasp failed" if status == "FAIL" else None,
                ),
            )
        return 0

    monkeypatch.setattr("auto_atom.runner.unidoor_sweep._stream_command", fake_stream)

    returncodes, interrupted = _run_batches(manifest, sweep_dir)

    assert returncodes == [0]
    assert interrupted is False
    assert launched == [["H001", "H002", "H003", "H004"]]
    stored = json.loads((sweep_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert stored["status"] == "STOPPED_ON_FAILURE"
    assert [job["status"] for job in stored["jobs"]] == [
        "SUCCESS",
        "TASK_FAILURE",
        "SUCCESS",
        "SUCCESS",
        "PENDING",
    ]
    assert stored["progress"]["next_job_num"] == 1
    assert stored["progress"]["stopped_job_num"] == 1
    assert stored["progress"]["stopped_handle_id"] == "H002"
    assert stored["progress"]["stop_status"] == "TASK_FAILURE"
    assert stored["progress"]["stop_reason"] == "grasp failed"
    assert stored["jobs"][1]["attempts"][-1]["launcher_returncode"] == 0
    report = aggregate_sweep(sweep_dir)
    assert report["run_status"] == "STOPPED_ON_FAILURE"
    assert _report_exit_code(report) == 1


def test_run_records_launcher_failure_when_no_valid_summary_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001",),
        handles=("H001", "H002", "H003", "H004", "H005"),
    )
    sweep_dir = tmp_path / "sweep"
    config = UniDoorSweepConfig(asset_package=package_path)
    manifest = _build_manifest(config, load_catalog(config), sweep_dir)
    sweep_dir.mkdir()
    _write_json(sweep_dir / MANIFEST_NAME, manifest)

    monkeypatch.setattr(
        "auto_atom.runner.unidoor_sweep._stream_command",
        lambda *_args, **_kwargs: 7,
    )

    _run_batches(manifest, sweep_dir)

    stored = json.loads((sweep_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [job["status"] for job in stored["jobs"]] == [
        "LAUNCHER_FAILURE",
        "LAUNCHER_FAILURE",
        "LAUNCHER_FAILURE",
        "LAUNCHER_FAILURE",
        "PENDING",
    ]
    assert stored["progress"]["stop_status"] == "LAUNCHER_FAILURE"
    report = aggregate_sweep(sweep_dir)
    assert report["counts"]["LAUNCHER_FAILURE"] == 4
    assert _report_exit_code(report) == 2


def test_parallel_wave_exit_code_prefers_infrastructure_failure(
    tmp_path: Path,
) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001",),
        handles=("H001", "H002"),
    )
    sweep_dir = tmp_path / "sweep"
    manifest = _build_manifest(
        UniDoorSweepConfig(asset_package=package_path),
        load_catalog(UniDoorSweepConfig(asset_package=package_path)),
        sweep_dir,
    )
    sweep_dir.mkdir()
    task_job, launcher_job = manifest["jobs"]
    task_dir = sweep_dir / task_job["relative_dir"]
    task_dir.mkdir(parents=True)
    _write_json(
        task_dir / "summary.json",
        _summary(status="FAIL", success=False, reason="grasp failed"),
    )
    task_job["status"] = "TASK_FAILURE"
    launcher_job["status"] = "LAUNCHER_FAILURE"
    launcher_job["attempts"][-1]["status"] = "LAUNCHER_FAILURE"
    launcher_job["attempts"][-1]["failure_reasons"] = ["worker crashed"]
    manifest["status"] = "STOPPED_ON_FAILURE"
    manifest["progress"]["stopped_job_num"] = task_job["job_num"]
    manifest["progress"]["stop_status"] = "TASK_FAILURE"
    _write_json(sweep_dir / MANIFEST_NAME, manifest)

    report = aggregate_sweep(sweep_dir)

    assert report["counts"]["TASK_FAILURE"] == 1
    assert report["counts"]["LAUNCHER_FAILURE"] == 1
    assert _report_exit_code(report) == 2


def test_resume_retries_failure_and_suffix_without_repeating_success(
    tmp_path: Path,
) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001",),
        handles=("H001", "H002", "H003"),
    )
    sweep_dir = tmp_path / "sweep"
    config = UniDoorSweepConfig(asset_package=package_path)
    manifest = _build_manifest(config, load_catalog(config), sweep_dir)
    sweep_dir.mkdir()
    success_job, failed_job, pending_job = manifest["jobs"]
    success_dir = sweep_dir / success_job["relative_dir"]
    success_dir.mkdir(parents=True)
    _write_json(success_dir / "summary.json", _summary(status="OK", success=True))
    failure_dir = sweep_dir / failed_job["relative_dir"]
    failure_dir.mkdir(parents=True)
    _write_json(
        failure_dir / "summary.json",
        _summary(status="FAIL", success=False, reason="grasp failed"),
    )
    failed_job["status"] = "TASK_FAILURE"
    pending_job["status"] = "PENDING"
    old_success_path = success_job["relative_dir"]
    old_failure_path = failed_job["relative_dir"]
    manifest["resume_count"] = 1

    unresolved = _unresolved_jobs(manifest, sweep_dir)
    batch_numbers = _append_resume_batches(manifest, sweep_dir, unresolved)

    assert [job["handle_id"] for job in unresolved] == ["H002", "H003"]
    assert success_job["relative_dir"] == old_success_path
    assert len(success_job["attempts"]) == 1
    assert failed_job["relative_dir"] != old_failure_path
    assert len(failed_job["attempts"]) == 2
    assert len(pending_job["attempts"]) == 2
    assert len(batch_numbers) == 1
    new_batches = [
        batch for batch in manifest["batches"] if batch["batch_num"] in batch_numbers
    ]
    assert [batch["handle_ids"] for batch in new_batches] == [["H002", "H003"]]
    assert "hydra/launcher=joblib" in new_batches[0]["argv"]
    assert "hydra.launcher.n_jobs=2" in new_batches[0]["argv"]
    assert all(batch["resume_attempt"] == 1 for batch in new_batches)


def test_report_can_discover_raw_hydra_jobs_without_manifest(tmp_path: Path) -> None:
    job_dir = tmp_path / "D001__H001"
    hydra_dir = job_dir / ".hydra"
    hydra_dir.mkdir(parents=True)
    (hydra_dir / "config.yaml").write_text(
        "door_id: D001\nhandle_id: H001\n", encoding="utf-8"
    )
    _write_json(job_dir / "summary.json", _summary(status="OK", success=True))

    report = aggregate_sweep(tmp_path)

    assert report["run_status"] == "DISCOVERED"
    assert report["counts"]["SUCCESS"] == 1
    assert report["results"][0]["door_id"] == "D001"


def test_dry_run_does_not_create_output_or_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = _asset_package(tmp_path)
    output_dir = tmp_path / "planned"
    config = UniDoorSweepConfig(
        asset_package=package_path,
        doors=["D001"],
        handles=["H001"],
        output_dir=output_dir,
        dry_run=True,
    )

    def fail_if_launched(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run launched Hydra")

    monkeypatch.setattr("auto_atom.runner.unidoor_sweep._run_batches", fail_if_launched)

    assert run_unidoor_sweep(config) == 0
    assert not output_dir.exists()


def test_latest_sweep_dir_uses_manifest_mtime_and_skips_invalid_runs(
    tmp_path: Path,
) -> None:
    older = tmp_path / "20260830-120000"
    newer = tmp_path / "20260831-120000"
    invalid = tmp_path / "20260901-120000"
    for sweep_dir in (older, newer, invalid):
        sweep_dir.mkdir()
    _write_json(older / MANIFEST_NAME, {"schema_version": 1, "jobs": []})
    _write_json(newer / MANIFEST_NAME, {"schema_version": 1, "jobs": []})
    (invalid / MANIFEST_NAME).write_text("not json", encoding="utf-8")
    older_time = 1_700_000_000_000_000_000
    newer_time = older_time + 1_000_000_000
    (older / MANIFEST_NAME).touch()
    (newer / MANIFEST_NAME).touch()
    os.utime(older / MANIFEST_NAME, ns=(older_time, older_time))
    os.utime(newer / MANIFEST_NAME, ns=(newer_time, newer_time))

    assert _latest_sweep_dir(tmp_path) == newer


def test_latest_sweep_dir_rejects_missing_or_empty_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        _latest_sweep_dir(tmp_path / "missing")
    with pytest.raises(ValueError, match="No valid UniDoor sweeps"):
        _latest_sweep_dir(tmp_path)
