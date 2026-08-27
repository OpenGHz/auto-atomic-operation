"""Serial Hydra sweeps for UniDoor door and handle combinations."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)
from pydantic_settings import CliApp
from typing_extensions import Self

DEFAULT_ASSET_PACKAGE = Path(
    "assets/scene_assets/unidoor_lever_right_hinge/scene_asset_package.json"
)
DEFAULT_CONFIG_NAME = "open_door_unidoor_p7_v3_umi_v3"
DEFAULT_OUTPUT_ROOT = Path("outputs/unidoor-sweeps")
MANIFEST_NAME = "sweep_manifest.json"
REPORT_NAME = "report.json"
FAILURES_NAME = "failures.csv"
LOG_NAME = "sweep.log"
_SAFE_ASSET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_INFRASTRUCTURE_STATUSES = {
    "NO_SUMMARY",
    "NOT_STARTED",
    "INVALID_SUMMARY",
}


class _TerminationSignal(BaseException):
    def __init__(self, signum: int):
        self.signum = signum


class UniDoorSweepConfig(BaseModel, frozen=True):
    """Configuration for a serial UniDoor asset-matrix sweep."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        cli_kebab_case=True,
    )

    config_name: str = DEFAULT_CONFIG_NAME
    """Hydra task config passed to ``aao-demo``."""

    asset_package: Path = DEFAULT_ASSET_PACKAGE
    """Scene asset package whose component index supplies door and handle IDs."""

    doors: list[str] = []
    """Door IDs to test; repeat the option or use a comma list; empty means all."""

    handles: list[str] = []
    """Handle IDs to test; repeat the option or use a comma list; empty means all."""

    output_dir: Path | None = None
    """Exact result directory; otherwise a timestamped directory is created."""

    report: Path | None = None
    """Rebuild reports for an existing sweep without launching simulations."""

    resume: Path | None = None
    """Resume unresolved batches from an existing sweep directory."""

    rounds: PositiveInt = 1
    """Demo rounds executed for every door and handle combination."""

    max_updates: NonNegativeInt = 600
    """Maximum public task updates per round; zero is useful for report smoke tests."""

    seed: int = 42
    """Task seed applied identically to every combination for reproducibility."""

    launcher_batch_size: PositiveInt = 6
    """Maximum Hydra jobs per Python process; jobs remain strictly serial."""

    headless: bool = True
    """Disable the interactive viewer during the sweep."""

    dry_run: bool = False
    """Print the complete plan without creating files or launching simulations."""

    @model_validator(mode="after")
    def validate_modes(self) -> Self:
        selected_modes = sum(value is not None for value in (self.report, self.resume))
        if selected_modes > 1:
            raise ValueError("--report and --resume are mutually exclusive.")
        if self.report is not None and self.output_dir is not None:
            raise ValueError("--report cannot be combined with --output-dir.")
        if self.resume is not None and self.output_dir is not None:
            raise ValueError("--resume cannot be combined with --output-dir.")
        if (self.report is not None or self.resume is not None) and self.dry_run:
            raise ValueError("--dry-run is only valid when planning a new sweep.")
        return self


@dataclass(frozen=True)
class _Catalog:
    asset_package: Path
    asset_package_sha256: str
    product_space: Path
    product_space_sha256: str
    schema_version: str
    compiler_version: str
    doors: tuple[str, ...]
    handles: tuple[str, ...]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"File does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_from(path: Path, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (base / expanded).resolve()


def _component_ids(
    payload: dict[str, Any],
    *,
    role: str,
    product_space: Path,
) -> tuple[str, ...]:
    components = payload.get("components")
    records = components.get(role) if isinstance(components, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{product_space} has no components.{role} records.")

    ids: list[str] = []
    rejected: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("asset_id"), str):
            raise ValueError(f"Invalid components.{role} record in {product_space}.")
        asset_id = record["asset_id"]
        if record.get("status") != "pass":
            rejected.append(asset_id)
        if not _SAFE_ASSET_ID.fullmatch(asset_id):
            raise ValueError(
                f"Unsafe {role} asset ID {asset_id!r}; IDs must be Hydra/path safe."
            )
        ids.append(asset_id)

    duplicates = sorted(
        asset_id for asset_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate {role} asset IDs: {', '.join(duplicates)}")
    if rejected:
        raise ValueError(
            f"Non-pass {role} assets are not runnable: {', '.join(rejected)}"
        )
    return tuple(ids)


def load_catalog(config: UniDoorSweepConfig) -> _Catalog:
    """Load the asset package and its canonical product-space index."""
    workdir = Path.cwd().resolve()
    package_path = _resolve_from(config.asset_package, workdir)
    package = _read_json(package_path)
    if package.get("schema") != "aao.scene-asset-package/v1":
        raise ValueError(
            f"Unsupported scene asset package schema in {package_path}: "
            f"{package.get('schema')!r}"
        )

    payload_root = package.get("payload_root")
    components = package.get("components")
    payload_uri = payload_root.get("uri") if isinstance(payload_root, dict) else None
    index_name = components.get("index") if isinstance(components, dict) else None
    if not isinstance(payload_uri, str) or not isinstance(index_name, str):
        raise ValueError(
            f"{package_path} must declare payload_root.uri and components.index."
        )
    payload_path = _resolve_from(Path(payload_uri), package_path.parent)
    product_space = _resolve_from(Path(index_name), payload_path)
    product = _read_json(product_space)

    doors = _component_ids(product, role="doors", product_space=product_space)
    handles = _component_ids(product, role="handles", product_space=product_space)
    combination = product.get("combination_space")
    if not isinstance(combination, dict):
        raise ValueError(f"{product_space} has no combination_space contract.")
    expected = len(doors) * len(handles)
    declared = combination.get("expected_count")
    if declared != expected:
        raise ValueError(
            f"Product-space expected_count is {declared!r}, but components define "
            f"{len(doors)} x {len(handles)} = {expected} combinations."
        )

    return _Catalog(
        asset_package=package_path,
        asset_package_sha256=_sha256(package_path),
        product_space=product_space,
        product_space_sha256=_sha256(product_space),
        schema_version=str(product.get("schema_version", "")),
        compiler_version=str(product.get("compiler_version", "")),
        doors=doors,
        handles=handles,
    )


def _select_ids(
    requested: Sequence[str],
    available: Sequence[str],
    *,
    role: str,
) -> tuple[str, ...]:
    if not requested:
        return tuple(available)
    duplicates = sorted(
        value for value, count in Counter(requested).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate requested {role} IDs: {', '.join(duplicates)}")
    unknown = [value for value in requested if value not in available]
    if unknown:
        raise ValueError(f"Unknown {role} IDs: {', '.join(unknown)}")
    return tuple(requested)


def _hydra_command(
    *,
    config_name: str,
    door_id: str,
    handle_ids: Sequence[str],
    batch_dir: Path,
    rounds: int,
    max_updates: int,
    seed: int,
    headless: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "auto_atom.runner.demo",
        "--multirun",
        "--config-name",
        config_name,
        "hydra/launcher=basic",
        "hydra/sweeper=basic",
        f"door_id={door_id}",
        f"handle_id={','.join(handle_ids)}",
        "env.batch_size=1",
        "task.seed=" + str(seed),
        "++use_input=false",
        "++print_updates=false",
        f"++rounds={rounds}",
        f"++max_updates={max_updates}",
        "hydra.sweeper.max_batch_size=null",
        "hydra.job.chdir=false",
        f"hydra.sweep.dir={json.dumps(str(batch_dir), ensure_ascii=False)}",
        "hydra.sweep.subdir=${hydra.job.num}__${door_id}__${handle_id}",
    ]
    if headless:
        command.append("env.viewer=null")
    return command


def _build_manifest(
    config: UniDoorSweepConfig,
    catalog: _Catalog,
    sweep_dir: Path,
) -> dict[str, Any]:
    doors = _select_ids(config.doors, catalog.doors, role="door")
    handles = _select_ids(config.handles, catalog.handles, role="handle")
    jobs: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    job_num = 0
    batch_num = 0
    for door_id in doors:
        for offset in range(0, len(handles), config.launcher_batch_size):
            batch_handles = handles[offset : offset + config.launcher_batch_size]
            batch_relative_dir = Path("batches") / f"{batch_num:04d}__{door_id}"
            batch_dir = sweep_dir / batch_relative_dir
            command = _hydra_command(
                config_name=config.config_name,
                door_id=door_id,
                handle_ids=batch_handles,
                batch_dir=batch_dir,
                rounds=config.rounds,
                max_updates=config.max_updates,
                seed=config.seed,
                headless=config.headless,
            )
            batch_job_numbers: list[int] = []
            for local_num, handle_id in enumerate(batch_handles):
                relative_dir = batch_relative_dir / (
                    f"{local_num}__{door_id}__{handle_id}"
                )
                jobs.append(
                    {
                        "job_num": job_num,
                        "batch_num": batch_num,
                        "door_id": door_id,
                        "handle_id": handle_id,
                        "relative_dir": relative_dir.as_posix(),
                        "attempts": [
                            {
                                "attempt": 0,
                                "batch_num": batch_num,
                                "relative_dir": relative_dir.as_posix(),
                            }
                        ],
                    }
                )
                batch_job_numbers.append(job_num)
                job_num += 1
            batches.append(
                {
                    "batch_num": batch_num,
                    "door_id": door_id,
                    "handle_ids": list(batch_handles),
                    "relative_dir": batch_relative_dir.as_posix(),
                    "job_numbers": batch_job_numbers,
                    "argv": command,
                }
            )
            batch_num += 1

    git_state = _git_state(Path.cwd())
    return {
        "schema_version": 1,
        "tool": "aao-unidoor-sweep",
        "status": "PLANNED",
        "created_at": _now_iso(),
        "finished_at": None,
        "sweep_dir": str(sweep_dir),
        "config": {
            "config_name": config.config_name,
            "asset_package": str(catalog.asset_package),
            "rounds": config.rounds,
            "max_updates": config.max_updates,
            "seed": config.seed,
            "launcher_batch_size": config.launcher_batch_size,
            "headless": config.headless,
        },
        "catalog": {
            "asset_package": str(catalog.asset_package),
            "asset_package_sha256": catalog.asset_package_sha256,
            "product_space": str(catalog.product_space),
            "product_space_sha256": catalog.product_space_sha256,
            "schema_version": catalog.schema_version,
            "compiler_version": catalog.compiler_version,
        },
        "selection": {
            "doors": list(doors),
            "handles": list(handles),
            "combination_count": len(jobs),
        },
        "git": git_state,
        "environment": {
            "AAO_UNIDOOR_ASSET_PACKAGE": str(catalog.asset_package),
            "HYDRA_FULL_ERROR": "1",
        },
        "batches": batches,
        "jobs": jobs,
        "launcher_returncodes": [],
        "resume_count": 0,
    }


def _git_state(workdir: Path) -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=workdir,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _default_sweep_dir() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return (Path.cwd() / DEFAULT_OUTPUT_ROOT / stamp).resolve()


def _prepare_new_sweep(config: UniDoorSweepConfig) -> tuple[Path, dict[str, Any]]:
    catalog = load_catalog(config)
    sweep_dir = (
        _resolve_from(config.output_dir, Path.cwd())
        if config.output_dir is not None
        else _default_sweep_dir()
    )
    if sweep_dir.exists() and any(sweep_dir.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {sweep_dir}. Use --resume to continue it."
        )
    manifest = _build_manifest(config, catalog, sweep_dir)
    return sweep_dir, manifest


def _load_manifest(sweep_dir: Path) -> dict[str, Any]:
    path = sweep_dir / MANIFEST_NAME
    manifest = _read_json(path)
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("jobs"), list
    ):
        raise ValueError(f"Unsupported or incomplete sweep manifest: {path}")
    return manifest


def _stream_command(
    command: Sequence[str],
    *,
    workdir: Path,
    environment: dict[str, str],
    log_path: Path,
) -> int:
    process_environment = os.environ.copy()
    process_environment.update(environment)
    process_environment["PYTHONUNBUFFERED"] = "1"
    previous_handlers: dict[int, Any] = {}

    def _raise_termination(signum: int, _frame: Any) -> None:
        raise _TerminationSignal(signum)

    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=workdir,
            env=process_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        for signum in (signal.SIGTERM, signal.SIGHUP):
            try:
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, _raise_termination)
            except (ValueError, OSError):
                previous_handlers.pop(signum, None)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        except KeyboardInterrupt:
            _stop_process_group(process, signal.SIGINT)
            raise
        except _TerminationSignal as exc:
            _stop_process_group(process, exc.signum)
            raise KeyboardInterrupt from None
        except BaseException:
            _stop_process_group(process, signal.SIGTERM)
            raise
        finally:
            process.stdout.close()
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _stop_process_group(process: subprocess.Popen[str], first_signal: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _summary_result(
    job: dict[str, Any],
    sweep_dir: Path,
    *,
    expected_rounds: int | None = None,
) -> dict[str, Any]:
    relative_dir = Path(job["relative_dir"])
    job_dir = sweep_dir / relative_dir
    summary_path = job_dir / "summary.json"
    result: dict[str, Any] = {
        "job_num": job.get("job_num"),
        "batch_num": job.get("batch_num"),
        "door_id": job["door_id"],
        "handle_id": job["handle_id"],
        "status": "NOT_STARTED",
        "job_dir": relative_dir.as_posix(),
        "summary": (relative_dir / "summary.json").as_posix(),
        "hydra_config": (relative_dir / ".hydra" / "config.yaml").as_posix(),
        "round_statuses": [],
        "final_stage": [],
        "failure_reasons": [],
    }
    if not job_dir.exists():
        result["failure_reasons"] = ["Hydra job directory was not created."]
        return result
    if not summary_path.exists():
        result["status"] = "NO_SUMMARY"
        result["failure_reasons"] = [
            "Hydra job started but did not produce summary.json; inspect sweep.log."
        ]
        return result

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["status"] = "INVALID_SUMMARY"
        result["failure_reasons"] = [f"Could not parse summary.json: {exc}"]
        return result
    rounds = summary.get("rounds") if isinstance(summary, dict) else None
    if not isinstance(rounds, list) or not rounds:
        result["status"] = "INVALID_SUMMARY"
        result["failure_reasons"] = ["summary.json has no non-empty rounds list."]
        return result

    invalid_reasons: list[str] = []
    if expected_rounds is not None and len(rounds) != expected_rounds:
        invalid_reasons.append(
            f"summary.json has {len(rounds)} rounds; expected {expected_rounds}."
        )
    task_failed = False
    for index, round_result in enumerate(rounds, start=1):
        if not isinstance(round_result, dict):
            invalid_reasons.append(f"Round {index} is not a JSON object.")
            continue
        status = round_result.get("status")
        if status not in {"OK", "FAIL"}:
            invalid_reasons.append(f"Round {index} has invalid status {status!r}.")
            continue
        result["round_statuses"].append(status)
        final_success = round_result.get("final_success")
        if (
            not isinstance(final_success, list)
            or not final_success
            or any(type(value) is not bool for value in final_success)
        ):
            invalid_reasons.append(
                f"Round {index} has no non-empty boolean final_success list."
            )
        else:
            all_succeeded = all(final_success)
            if status == "OK" and not all_succeeded:
                invalid_reasons.append(
                    f"Round {index} reports OK with unsuccessful environments."
                )
            elif status == "FAIL" and all_succeeded:
                invalid_reasons.append(
                    f"Round {index} reports FAIL with every environment successful."
                )
        task_failed = task_failed or status == "FAIL"
        final_stage = round_result.get("final_stage")
        if isinstance(final_stage, list):
            result["final_stage"].extend(str(value) for value in final_stage if value)
        reasons = round_result.get("failure_reasons")
        if isinstance(reasons, list):
            result["failure_reasons"].extend(
                str(reason) for reason in reasons if reason is not None
            )

    if invalid_reasons:
        result["status"] = "INVALID_SUMMARY"
        result["failure_reasons"] = invalid_reasons + result["failure_reasons"]
    elif task_failed:
        result["status"] = "TASK_FAILURE"
        if not result["failure_reasons"]:
            result["failure_reasons"] = [
                "At least one round or environment reported failure."
            ]
    else:
        result["status"] = "SUCCESS"
    return result


def _discover_jobs(sweep_dir: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for config_path in sorted(sweep_dir.rglob(".hydra/config.yaml")):
        try:
            config = OmegaConf.load(config_path)
            door_id = str(config.get("door_id"))
            handle_id = str(config.get("handle_id"))
        except Exception:
            continue
        if door_id == "None" or handle_id == "None":
            continue
        jobs.append(
            {
                "job_num": len(jobs),
                "batch_num": None,
                "door_id": door_id,
                "handle_id": handle_id,
                "relative_dir": config_path.parent.parent.relative_to(
                    sweep_dir
                ).as_posix(),
            }
        )
    if not jobs:
        raise ValueError(
            f"No {MANIFEST_NAME} or Hydra job configs were found under {sweep_dir}."
        )
    return jobs


def _reproduction_command(
    manifest: dict[str, Any] | None,
    result: dict[str, Any],
) -> str:
    config = manifest.get("config", {}) if manifest is not None else {}
    environment = manifest.get("environment", {}) if manifest is not None else {}
    command = [
        sys.executable,
        "-u",
        "-m",
        "auto_atom.runner.demo",
        "--config-name",
        str(config.get("config_name", DEFAULT_CONFIG_NAME)),
        f"door_id={result['door_id']}",
        f"handle_id={result['handle_id']}",
        "env.batch_size=1",
        f"task.seed={config.get('seed', 42)}",
        "++use_input=false",
        "++print_updates=false",
        f"++rounds={config.get('rounds', 1)}",
        f"++max_updates={config.get('max_updates', 600)}",
    ]
    if config.get("headless", True):
        command.append("env.viewer=null")
    prefix = ["env"]
    asset_package = environment.get("AAO_UNIDOOR_ASSET_PACKAGE")
    if asset_package:
        prefix.append(f"AAO_UNIDOOR_ASSET_PACKAGE={asset_package}")
    return shlex.join([*prefix, *command])


def aggregate_sweep(sweep_dir: Path) -> dict[str, Any]:
    """Aggregate one sweep directory into JSON and CSV reports."""
    sweep_dir = sweep_dir.expanduser().resolve()
    manifest_path = sweep_dir / MANIFEST_NAME
    if manifest_path.exists():
        manifest = _load_manifest(sweep_dir)
        jobs = manifest["jobs"]
        run_status = manifest.get("status")
        launcher_returncodes = manifest.get("launcher_returncodes", [])
        expected_rounds = int(manifest.get("config", {}).get("rounds", 1))
    else:
        manifest = None
        jobs = _discover_jobs(sweep_dir)
        run_status = "DISCOVERED"
        launcher_returncodes = []
        expected_rounds = None

    results = [
        _summary_result(job, sweep_dir, expected_rounds=expected_rounds) for job in jobs
    ]
    for result in results:
        result["reproduce_command"] = _reproduction_command(manifest, result)
    counts = Counter(result["status"] for result in results)
    for status in (
        "SUCCESS",
        "TASK_FAILURE",
        "NO_SUMMARY",
        "NOT_STARTED",
        "INVALID_SUMMARY",
    ):
        counts.setdefault(status, 0)
    report = {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "sweep_dir": str(sweep_dir),
        "run_status": run_status,
        "launcher_returncodes": launcher_returncodes,
        "expected_jobs": len(jobs),
        "completed_jobs": counts["SUCCESS"] + counts["TASK_FAILURE"],
        "counts": dict(counts),
        "results": results,
    }
    _write_json(sweep_dir / REPORT_NAME, report)
    _write_failures_csv(sweep_dir / FAILURES_NAME, results)
    return report


def _write_failures_csv(path: Path, results: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "job_num",
        "batch_num",
        "door_id",
        "handle_id",
        "status",
        "job_dir",
        "final_stage",
        "failure_reason",
        "reproduce_command",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if result["status"] == "SUCCESS":
                continue
            writer.writerow(
                {
                    "job_num": result["job_num"],
                    "batch_num": result["batch_num"],
                    "door_id": result["door_id"],
                    "handle_id": result["handle_id"],
                    "status": result["status"],
                    "job_dir": result["job_dir"],
                    "final_stage": "; ".join(result["final_stage"]),
                    "failure_reason": " | ".join(result["failure_reasons"]),
                    "reproduce_command": result["reproduce_command"],
                }
            )
    temporary.replace(path)


def _report_exit_code(report: dict[str, Any]) -> int:
    counts = report["counts"]
    if any(counts.get(status, 0) for status in _INFRASTRUCTURE_STATUSES):
        return 2
    if any(code != 0 for code in report.get("launcher_returncodes", [])):
        return 2
    if counts.get("TASK_FAILURE", 0):
        return 1
    return 0


def _print_report(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print(
        "Sweep report: "
        f"{counts['SUCCESS']} success, "
        f"{counts['TASK_FAILURE']} task failures, "
        f"{counts['NO_SUMMARY']} without summary, "
        f"{counts['NOT_STARTED']} not started, "
        f"{counts['INVALID_SUMMARY']} invalid summaries."
    )
    print(f"Report: {Path(report['sweep_dir']) / REPORT_NAME}")
    print(f"Failures: {Path(report['sweep_dir']) / FAILURES_NAME}")
    for result in report["results"]:
        if result["status"] == "SUCCESS":
            continue
        reason = " | ".join(result["failure_reasons"])
        print(
            f"  {result['status']} {result['door_id']} + {result['handle_id']}: "
            f"{reason}"
        )


def _unresolved_jobs(
    manifest: dict[str, Any],
    sweep_dir: Path,
) -> list[dict[str, Any]]:
    expected_rounds = int(manifest.get("config", {}).get("rounds", 1))
    unresolved: list[dict[str, Any]] = []
    for job in manifest["jobs"]:
        result = _summary_result(
            job,
            sweep_dir,
            expected_rounds=expected_rounds,
        )
        if result["status"] in _INFRASTRUCTURE_STATUSES:
            unresolved.append(job)
    return unresolved


def _append_resume_batches(
    manifest: dict[str, Any],
    sweep_dir: Path,
    unresolved_jobs: Sequence[dict[str, Any]],
) -> set[int]:
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Sweep manifest has no config object for resume.")
    required = {
        "config_name",
        "rounds",
        "max_updates",
        "seed",
        "launcher_batch_size",
        "headless",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(
            "Sweep manifest cannot be resumed; missing config fields: "
            + ", ".join(missing)
        )

    resume_number = int(manifest.get("resume_count", 0))
    next_batch_num = (
        max(
            (int(batch["batch_num"]) for batch in manifest.get("batches", [])),
            default=-1,
        )
        + 1
    )
    jobs_by_door: dict[str, list[dict[str, Any]]] = {}
    for job in unresolved_jobs:
        jobs_by_door.setdefault(str(job["door_id"]), []).append(job)

    new_batches: list[dict[str, Any]] = []
    new_batch_numbers: set[int] = set()
    batch_size = int(config["launcher_batch_size"])
    for door_id, door_jobs in jobs_by_door.items():
        for offset in range(0, len(door_jobs), batch_size):
            batch_jobs = door_jobs[offset : offset + batch_size]
            handle_ids = [str(job["handle_id"]) for job in batch_jobs]
            relative_batch_dir = (
                Path("resume")
                / f"{resume_number:04d}"
                / "batches"
                / f"{next_batch_num:04d}__{door_id}"
            )
            command = _hydra_command(
                config_name=str(config["config_name"]),
                door_id=door_id,
                handle_ids=handle_ids,
                batch_dir=sweep_dir / relative_batch_dir,
                rounds=int(config["rounds"]),
                max_updates=int(config["max_updates"]),
                seed=int(config["seed"]),
                headless=bool(config["headless"]),
            )
            for local_num, job in enumerate(batch_jobs):
                relative_dir = relative_batch_dir / (
                    f"{local_num}__{door_id}__{job['handle_id']}"
                )
                attempts = job.setdefault("attempts", [])
                if not attempts:
                    attempts.append(
                        {
                            "attempt": 0,
                            "batch_num": job.get("batch_num"),
                            "relative_dir": job["relative_dir"],
                        }
                    )
                attempts.append(
                    {
                        "attempt": resume_number,
                        "batch_num": next_batch_num,
                        "relative_dir": relative_dir.as_posix(),
                    }
                )
                job["batch_num"] = next_batch_num
                job["relative_dir"] = relative_dir.as_posix()
            new_batches.append(
                {
                    "batch_num": next_batch_num,
                    "door_id": door_id,
                    "handle_ids": handle_ids,
                    "relative_dir": relative_batch_dir.as_posix(),
                    "job_numbers": [int(job["job_num"]) for job in batch_jobs],
                    "argv": command,
                    "resume_attempt": resume_number,
                }
            )
            new_batch_numbers.add(next_batch_num)
            next_batch_num += 1

    manifest.setdefault("batches", []).extend(new_batches)
    return new_batch_numbers


def _run_batches(
    manifest: dict[str, Any],
    sweep_dir: Path,
    *,
    batch_numbers: set[int] | None = None,
) -> tuple[list[int], bool]:
    batches = manifest["batches"]
    selected = [
        batch
        for batch in batches
        if batch_numbers is None or int(batch["batch_num"]) in batch_numbers
    ]
    environment = {
        key: str(value) for key, value in manifest.get("environment", {}).items()
    }
    returncodes: list[int] = []
    interrupted = False
    log_path = sweep_dir / LOG_NAME
    for position, batch in enumerate(selected, start=1):
        heading = (
            f"\n=== Hydra batch {position}/{len(selected)} "
            f"(catalog batch {batch['batch_num']}, door {batch['door_id']}) ===\n"
        )
        print(heading, end="")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(heading)
            log.write(shlex.join(batch["argv"]) + "\n")
        try:
            returncode = _stream_command(
                batch["argv"],
                workdir=Path.cwd(),
                environment=environment,
                log_path=log_path,
            )
        except KeyboardInterrupt:
            interrupted = True
            break
        except OSError as exc:
            message = f"Could not launch Hydra batch {batch['batch_num']}: {exc}"
            print(message, file=sys.stderr)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(message + "\n")
            returncode = 127
        returncodes.append(returncode)
        if returncode != 0:
            print(
                f"Hydra batch {batch['batch_num']} exited {returncode}; "
                "continuing with the next bounded batch."
            )
    return returncodes, interrupted


def run_unidoor_sweep(config: UniDoorSweepConfig) -> int:
    """Run, resume, or report a UniDoor matrix sweep."""
    if config.report is not None:
        report = aggregate_sweep(_resolve_from(config.report, Path.cwd()))
        _print_report(report)
        return _report_exit_code(report)

    if config.resume is not None:
        sweep_dir = _resolve_from(config.resume, Path.cwd())
        manifest = _load_manifest(sweep_dir)
        unresolved = _unresolved_jobs(manifest, sweep_dir)
        if not unresolved:
            report = aggregate_sweep(sweep_dir)
            _print_report(report)
            return _report_exit_code(report)
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        previous_returncodes = manifest.get("launcher_returncodes", [])
        if previous_returncodes:
            manifest.setdefault("previous_launcher_returncodes", []).append(
                previous_returncodes
            )
        manifest["launcher_returncodes"] = []
        batch_numbers = _append_resume_batches(manifest, sweep_dir, unresolved)
        manifest["status"] = "RUNNING"
        manifest["finished_at"] = None
        _write_json(sweep_dir / MANIFEST_NAME, manifest)
        returncodes, interrupted = _run_batches(
            manifest,
            sweep_dir,
            batch_numbers=batch_numbers,
        )
    else:
        sweep_dir, manifest = _prepare_new_sweep(config)
        print(
            f"Planned {manifest['selection']['combination_count']} combinations in "
            f"{len(manifest['batches'])} bounded serial Hydra batches."
        )
        print(f"Output: {sweep_dir}")
        if config.dry_run:
            for batch in manifest["batches"]:
                print(shlex.join(batch["argv"]))
            return 0
        sweep_dir.mkdir(parents=True, exist_ok=True)
        manifest["status"] = "RUNNING"
        _write_json(sweep_dir / MANIFEST_NAME, manifest)
        returncodes, interrupted = _run_batches(manifest, sweep_dir)

    manifest["launcher_returncodes"].extend(returncodes)
    manifest["status"] = "INTERRUPTED" if interrupted else "FINISHED"
    manifest["finished_at"] = _now_iso()
    _write_json(sweep_dir / MANIFEST_NAME, manifest)
    report = aggregate_sweep(sweep_dir)
    _print_report(report)
    if interrupted:
        return 130
    return _report_exit_code(report)


def parse_config(cli_args: list[str] | None = None) -> UniDoorSweepConfig:
    """Parse command-line arguments into a validated sweep config."""
    return CliApp.run(UniDoorSweepConfig, cli_args=cli_args)


def main(cli_args: list[str] | None = None) -> None:
    """Console entry point for ``aao-unidoor-sweep``."""
    try:
        exit_code = run_unidoor_sweep(parse_config(cli_args))
    except ValueError as exc:
        print(f"aao-unidoor-sweep: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
