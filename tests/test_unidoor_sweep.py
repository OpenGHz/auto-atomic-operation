"""Tests for the bounded serial UniDoor matrix sweep CLI."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from auto_atom.runner.unidoor_sweep import (
    FAILURES_NAME,
    MANIFEST_NAME,
    UniDoorSweepConfig,
    _append_resume_batches,
    _build_manifest,
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


def test_parse_config_accepts_kebab_case_and_comma_lists() -> None:
    config = parse_config(
        [
            "--doors",
            "D002,D001",
            "--handles",
            "H001",
            "--max-updates",
            "0",
            "--rounds",
            "2",
            "--launcher-batch-size",
            "3",
            "--dry-run",
        ]
    )

    assert config.doors == ["D002", "D001"]
    assert config.handles == ["H001"]
    assert config.max_updates == 0
    assert config.rounds == 2
    assert config.launcher_batch_size == 3
    assert config.dry_run is True

    with pytest.raises(ValueError, match="greater than 0"):
        parse_config(["--rounds", "0"])


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
    ("doors", "handles", "error"),
    [
        (["D001", "D001"], ["H001"], "Duplicate requested door"),
        (["D999"], ["H001"], "Unknown door"),
        (["D001"], ["H999"], "Unknown handle"),
    ],
)
def test_manifest_rejects_duplicate_and_unknown_selections(
    tmp_path: Path,
    doors: list[str],
    handles: list[str],
    error: str,
) -> None:
    package_path = _asset_package(tmp_path)
    config = UniDoorSweepConfig(
        asset_package=package_path,
        doors=doors,
        handles=handles,
    )

    with pytest.raises(ValueError, match=error):
        _build_manifest(config, load_catalog(config), tmp_path / "sweep")


def test_manifest_uses_bounded_serial_hydra_batches(tmp_path: Path) -> None:
    package_path = _asset_package(
        tmp_path,
        doors=("D001", "D002"),
        handles=("H001", "H002", "H003"),
    )
    config = UniDoorSweepConfig(
        asset_package=package_path,
        launcher_batch_size=2,
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
    assert "env.batch_size=1" in first_command
    assert "env.viewer=null" in first_command
    assert "hydra.sweeper.max_batch_size=null" in first_command
    assert "hydra.sweeper.max_batch_size=1" not in first_command


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
