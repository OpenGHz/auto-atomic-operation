#!/usr/bin/env python3
"""Run pytest with conservative resource limits and bounded batch isolation.

The default mode runs one test module per bounded subprocess.  Up to
``max_concurrency`` batches may run together, while each subprocess is
placed in a transient user systemd scope when available, so a runaway simulator,
thread pool, or memory leak is contained to that batch.  Results and logs are
written below ``outputs/test-runs`` (which is ignored by git).

Examples::

    python scripts/run_tests_safe.py
    python scripts/run_tests_safe.py --test-targets tests/test_execution_timeline.py
    python scripts/run_tests_safe.py --batch-mode all --pytest-args='-q -k replay'
    python scripts/run_tests_safe.py --exclude "*mujoco*" --dry-run

The script is intentionally Linux-oriented.  ``systemd-run`` is the preferred
launcher because its memory limit is an RSS/cgroup limit.  ``prlimit`` is an
explicitly selectable weaker fallback that limits virtual address space instead.
There is no implicit unlimited fallback.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)
from pydantic_settings import CliApp
from typing_extensions import Self


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_python_executable() -> Path:
    preferred = Path("/home/ghz/.mini_conda3/envs/airbot_play_data/bin/python")
    return preferred if preferred.is_file() else Path(sys.executable)


def _default_cpu_set() -> str:
    """Choose a CPU that is actually available to this process."""
    try:
        available = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return "0"
    return str(available[0]) if available else "0"


class SafeTestConfig(BaseModel):
    """Validated settings for a resource-bounded pytest run."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        cli_kebab_case=True,
        frozen=True,
    )

    repo_root: Path = Field(default_factory=_default_repo_root)
    """Repository root used as the working directory for pytest."""

    python_executable: Path = Field(default_factory=_default_python_executable)
    """Python interpreter used to launch pytest."""

    test_targets: str = "tests"
    """Comma-separated test paths; quote a space-separated list for the shell."""

    exclude: str = ""
    """Comma-separated exclusion globs; quote a space-separated list for the shell."""

    batch_mode: Literal["file", "all"] = "file"
    """Run isolated batches per files, or one batch containing all targets."""

    batch_size: PositiveInt = Field(default=1, le=64)
    """Number of test files per isolated batch in ``file`` mode."""

    max_concurrency: PositiveInt = Field(default=4, le=16)
    """Requested maximum number of isolated batches running at once."""

    pytest_args: str = "-q"
    """Additional pytest arguments, parsed with shell-like quoting."""

    continue_on_failure: bool = True
    """Continue after ordinary test/launch failures; resource failures always stop."""

    output_dir: Path | None = None
    """Exact result directory; otherwise a unique directory is created under outputs/test-runs."""

    dry_run: bool = False
    """Print resolved batches and commands without launching tests or writing output."""

    launcher: Literal["auto", "systemd", "prlimit"] = "auto"
    """Resource launcher; auto prefers systemd and otherwise uses prlimit."""

    cpu_set: str = Field(default_factory=_default_cpu_set)
    """CPU list passed to taskset; the default selects one available core."""

    cpu_quota_percent: PositiveInt = Field(default=100, le=100)
    """Cgroup CPU quota as a percentage of one full CPU."""

    memory_high_mb: PositiveInt = 4096
    """Soft cgroup memory threshold that starts reclaim/throttling."""

    memory_max_mb: PositiveInt = 6144
    """Hard cgroup memory ceiling; exceeding it terminates the batch."""

    memory_swap_max_mb: NonNegativeInt = 0
    """Maximum swap available to a systemd batch; zero prevents swap thrashing."""

    tasks_max: PositiveInt = Field(default=64, le=4096)
    """Maximum number of tasks/threads in a systemd batch."""

    max_file_size_mb: PositiveInt = Field(default=256, le=4096)
    """Maximum size of any regular file written by a batch process or descendant."""

    timeout_minutes: PositiveInt = Field(default=20, le=24 * 60)
    """Wall-clock limit for each batch."""

    kill_after_seconds: PositiveInt = Field(default=30, le=300)
    """Grace period after timeout before the batch is force-killed."""

    thread_count: PositiveInt = Field(default=1, le=8)
    """Thread count exported to BLAS/OpenMP-style libraries."""

    cuda_visible_devices: str | None = None
    """Optional CUDA_VISIBLE_DEVICES value; CPU/RAM limits do not cap GPU VRAM."""

    @field_validator("cpu_set")
    @classmethod
    def validate_cpu_set(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("--cpu-set must not be empty")
        if any(character.isspace() for character in value):
            raise ValueError("--cpu-set must be a taskset CPU list without spaces")
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> Self:
        if self.memory_high_mb > self.memory_max_mb:
            raise ValueError("--memory-high-mb cannot exceed --memory-max-mb")
        if self.launcher == "prlimit" and self.memory_swap_max_mb:
            raise ValueError(
                "--memory-swap-max-mb is only enforceable with the systemd launcher"
            )
        return self


@dataclass(frozen=True)
class _Batch:
    """One deterministic group of pytest targets."""

    index: int
    targets: tuple[str, ...]


@dataclass(frozen=True)
class _BatchResult:
    """Outcome recorded for one bounded subprocess."""

    status: str
    returncode: int | None
    elapsed_seconds: float
    log_path: Path
    junit_path: Path
    command: tuple[str, ...]
    reason: str | None = None


@dataclass(frozen=True)
class _TerminationResult:
    """Bounded cleanup outcome for a wrapper process and its descendants."""

    returncode: int | None
    cleanup_incomplete: bool = False


@dataclass(frozen=True)
class _ConcurrencyPlan:
    """Requested and effective worker count, with transparent clamp reasons."""

    requested: int
    effective: int
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ActiveProcess:
    """A running batch process that can be signalled by the coordinator."""

    process: subprocess.Popen[str]
    launcher: Literal["systemd", "prlimit"]
    unit_name: str


class _ActiveProcessRegistry:
    """Thread-safe registry used to stop all active batches on interruption."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[int, _ActiveProcess] = {}
        self._stop_requested = threading.Event()

    def register(
        self,
        batch_index: int,
        process: subprocess.Popen[str],
        launcher: Literal["systemd", "prlimit"],
        unit_name: str,
    ) -> None:
        with self._lock:
            self._active[batch_index] = _ActiveProcess(process, launcher, unit_name)

    def unregister(self, batch_index: int) -> None:
        with self._lock:
            self._active.pop(batch_index, None)

    def snapshot(self) -> tuple[tuple[int, _ActiveProcess], ...]:
        with self._lock:
            return tuple(self._active.items())

    @property
    def stop_requested(self) -> bool:
        """Whether the coordinator has asked workers to stop."""
        return self._stop_requested.is_set()

    def request_stop(self) -> None:
        """Set the stop latch and terminate every process known so far.

        The latch closes the tiny ``Popen`` → ``register`` race: a worker that
        starts after the coordinator has begun cleanup observes it immediately
        and terminates its newly-created process before waiting on it.
        """
        self._stop_requested.set()
        self.signal_all()

    def signal_all(self) -> None:
        """Request graceful termination for every currently registered batch.

        The process-group signal is deliberately synchronous and cheap.  Each
        worker's normal ``_terminate_process`` path performs the launcher-specific
        scope stop and diagnostics, so the coordinator never serially waits on a
        ``systemctl`` call while handling the first interrupt.
        """
        for _index, active in self.snapshot():
            _send_process_group_signal(active.process, signal.SIGTERM)

    def force_kill_all(self) -> None:
        """Escalate interruption cleanup without waiting on a worker thread."""
        for _index, active in self.snapshot():
            _send_process_group_signal(active.process, signal.SIGKILL)
            if active.launcher == "systemd":
                _kill_unit(active.unit_name)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._active)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _split_values(raw: str) -> list[str]:
    values: list[str] = []
    for token in shlex.split(raw):
        values.extend(part.strip() for part in token.split(",") if part.strip())
    return values


def _resolve_repo_root(config: SafeTestConfig) -> Path:
    root = config.repo_root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository root does not exist: {root}")
    if not (root / "pyproject.toml").is_file():
        raise ValueError(f"Repository root does not contain pyproject.toml: {root}")
    executable = config.python_executable.expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError(f"Python executable is not runnable: {executable}")
    _validate_cpu_set(config.cpu_set)
    return root


def _relative_target(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Test target must be inside repository root: {path}") from exc


def _validate_cpu_set(cpu_set: str) -> None:
    """Fail before collection when ``taskset`` cannot use the requested CPUs."""
    taskset = shutil.which("taskset")
    if taskset is None:
        raise ValueError("Required executable not found on PATH: taskset")
    true_executable = shutil.which("true") or "/bin/true"
    result = subprocess.run(
        [taskset, "-c", cpu_set, true_executable],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or "the requested CPUs are unavailable"
        raise ValueError(f"--cpu-set={cpu_set!r} is not usable: {detail}")


def _expand_targets(config: SafeTestConfig, root: Path) -> list[str]:
    raw_targets = _split_values(config.test_targets)
    if not raw_targets:
        raise ValueError("--test-targets must contain at least one path")
    excluded = _split_values(config.exclude)
    resolved: list[str] = []
    seen: set[str] = set()

    def include(path: Path, suffix: str = "") -> None:
        relative = _relative_target(path, root)
        if any(
            fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)
            for pattern in excluded
        ):
            return
        target = relative + suffix
        if target not in seen:
            seen.add(target)
            resolved.append(target)

    for raw in raw_targets:
        path_text, separator, node_suffix = raw.partition("::")
        candidate = Path(path_text).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        matches: list[Path]
        if any(character in path_text for character in "*?["):
            if separator:
                raise ValueError(
                    "A pytest node selector (::...) must name one file; do not "
                    "combine it with a test glob."
                )
            if Path(path_text).is_absolute():
                raise ValueError(
                    "Absolute test glob patterns are not supported; use a path "
                    "relative to --repo-root."
                )
            matches = sorted(
                item.resolve() for item in root.glob(path_text) if item.is_file()
            )
        elif candidate.is_dir():
            if separator:
                raise ValueError(
                    "A pytest node selector (::...) must name one file; do not "
                    "combine it with a test directory."
                )
            matches = sorted(
                {
                    *candidate.rglob("test_*.py"),
                    *candidate.rglob("*_test.py"),
                }
            )
        elif candidate.is_file():
            matches = [candidate]
        else:
            raise ValueError(f"Test target does not exist: {raw}")
        if not matches:
            raise ValueError(f"Test target pattern matched no files: {raw}")
        for match in matches:
            if not match.is_file():
                continue
            include(match, f"::{node_suffix}" if separator else "")

    if not resolved:
        raise ValueError("All test targets were excluded")
    return sorted(resolved)


def _make_batches(targets: list[str], config: SafeTestConfig) -> list[_Batch]:
    if config.batch_mode == "all":
        return [_Batch(index=1, targets=tuple(targets))]
    size = int(config.batch_size)
    return [
        _Batch(index=index, targets=tuple(targets[start : start + size]))
        for index, start in enumerate(range(0, len(targets), size), start=1)
    ]


@contextmanager
def _manifest_guard(lock: threading.Lock | None) -> Iterator[None]:
    """Serialize in-place manifest updates and atomic metadata writes."""
    if lock is None:
        yield
        return
    with lock:
        yield


def _cpu_set_count(cpu_set: str) -> int | None:
    """Count CPUs in the common taskset list/range syntax."""
    selected: set[int] = set()
    excluded: set[int] = set()
    for raw_part in cpu_set.split(","):
        part = raw_part.strip()
        if not part:
            continue
        target = excluded if part.startswith("^") else selected
        if part.startswith("^"):
            part = part.removeprefix("^")
        # taskset accepts an optional stride (for example ``0-7:2``).
        stride = 1
        if ":" in part:
            part, raw_stride = part.split(":", 1)
            try:
                stride = int(raw_stride)
            except ValueError:
                return None
            if stride <= 0:
                return None
        try:
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text), int(end_text)
                if end < start:
                    return None
                target.update(range(start, end + 1, stride))
            else:
                target.add(int(part))
        except ValueError:
            return None
    if not selected:
        return None
    return max(1, len(selected - excluded))


def _read_cgroup_bytes(path: Path) -> int | None:
    """Read a cgroup byte counter, treating ``max`` as unlimited."""
    try:
        raw_value = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    if not raw_value or raw_value == "max":
        return None
    try:
        value = int(raw_value.split()[0])
    except (ValueError, IndexError):
        return None
    # cgroup v1 uses a very large sentinel for an unlimited limit.
    return None if value >= 1 << 60 else max(0, value)


def _cgroup_memory_available_bytes() -> int | None:
    """Find remaining memory in this cgroup and finite ancestor limits."""
    try:
        cgroup_lines = (
            Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        )
    except (OSError, UnicodeError):
        return None

    candidates: list[tuple[Path, Path]] = []
    for line in cgroup_lines:
        hierarchy, separator, relative = line.partition(":")
        if not separator:
            continue
        controllers, separator, relative = relative.partition(":")
        if not separator:
            continue
        relative_path = relative.lstrip("/")
        if hierarchy == "0":
            candidates.append((Path("/sys/fs/cgroup"), Path(relative_path)))
        elif "memory" in controllers.split(","):
            candidates.append((Path("/sys/fs/cgroup/memory"), Path(relative_path)))

    available_values: list[int] = []
    for mount_root, relative_path in candidates:
        directory = mount_root / relative_path
        if not directory.is_dir():
            continue
        while True:
            if (directory / "memory.max").is_file():
                limit = _read_cgroup_bytes(directory / "memory.max")
                current = _read_cgroup_bytes(directory / "memory.current")
                if limit is not None and current is not None:
                    available_values.append(max(0, limit - current))
            elif (directory / "memory.limit_in_bytes").is_file():
                limit = _read_cgroup_bytes(directory / "memory.limit_in_bytes")
                current = _read_cgroup_bytes(directory / "memory.usage_in_bytes")
                if limit is not None and current is not None:
                    available_values.append(max(0, limit - current))
            if directory == mount_root or mount_root not in directory.parents:
                break
            directory = directory.parent
    return min(available_values) if available_values else None


def _available_memory_mb() -> int | None:
    """Read conservative available memory from host and cgroup state."""
    host_available_mb: int | None = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition(":")
            if key == "MemAvailable" and separator:
                amount_kib = int(value.strip().split()[0])
                host_available_mb = max(1, amount_kib // 1024)
                break
    except (OSError, ValueError, IndexError):
        host_available_mb = None

    cgroup_available_bytes = _cgroup_memory_available_bytes()
    cgroup_available_mb = (
        max(1, cgroup_available_bytes // (1024 * 1024))
        if cgroup_available_bytes is not None
        else None
    )
    if host_available_mb is None:
        return cgroup_available_mb
    if cgroup_available_mb is None:
        return host_available_mb
    return min(host_available_mb, cgroup_available_mb)


def _plan_concurrency(
    config: SafeTestConfig,
    launcher: Literal["systemd", "prlimit"],
    batches: list[_Batch],
) -> _ConcurrencyPlan:
    """Derive a safe effective worker count from the requested upper bound.

    ``max_concurrency`` is intentionally a request rather than a promise.  The
    runner clamps it when the selected CPU set, available memory, GPU visibility,
    or weak fallback launcher cannot safely support all requested batches.  The
    reasons are persisted in metadata so a run remains explainable/reproducible.
    """
    requested = int(config.max_concurrency)
    effective = min(requested, len(batches))
    reasons: list[str] = []
    if config.batch_mode == "all" and effective > 1:
        effective = 1
        reasons.append("batch_mode=all has one shared batch")

    cpu_slots = _cpu_set_count(config.cpu_set)
    if cpu_slots is not None and effective > cpu_slots:
        effective = cpu_slots
        reasons.append(f"cpu_set provides only {cpu_slots} slot(s)")

    available_mb = _available_memory_mb()
    if available_mb is not None:
        # Keep roughly one quarter of currently available host memory for the
        # desktop, kernel, and non-test processes.  At least one slot remains so
        # a constrained machine still produces a useful bounded result.
        memory_budget_mb = max(1, available_mb * 3 // 4)
        memory_slots = max(1, memory_budget_mb // int(config.memory_max_mb))
        if effective > memory_slots:
            effective = memory_slots
            reasons.append(
                f"available memory budget allows {memory_slots} slot(s) "
                f"({available_mb} MiB available)"
            )

    if launcher == "prlimit" and effective > 1:
        # prlimit cannot enforce aggregate RSS, task, or swap limits; serial is
        # the only safe default when the cgroup launcher is unavailable.
        effective = 1
        reasons.append("prlimit fallback has no aggregate cgroup protection")

    if (
        config.cuda_visible_devices
        and config.cuda_visible_devices.strip()
        and effective > 1
    ):
        effective = 1
        reasons.append("CUDA visibility is shared; GPU batches stay serial")

    return _ConcurrencyPlan(requested, max(1, effective), tuple(reasons))


def _contains_parallel_pytest_option(args: list[str]) -> bool:
    parallel_names = {
        "-d",
        "-n",
        "--boxed",
        "--dist",
        "--forked",
        "--looponfail",
        "--max-worker-restart",
        "--numprocesses",
        "--processes",
        "--rsyncdir",
        "--rsyncignore",
        "--tx",
        "--workers",
    }
    return any(
        argument in parallel_names
        or argument.startswith(
            (
                "--dist=",
                "--max-worker-restart=",
                "--numprocesses=",
                "--processes=",
                "--rsyncdir=",
                "--rsyncignore=",
                "--tx=",
                "--workers=",
            )
        )
        or (argument.startswith("-n") and len(argument) > 2)
        for argument in args
    )


def _contains_junit_option(args: list[str]) -> bool:
    return any(
        argument in {"--junitxml", "--junit-xml"}
        or argument.startswith(("--junitxml=", "--junit-xml="))
        for argument in args
    )


def _contains_addopts_override(args: list[str]) -> bool:
    """Detect caller attempts to replace pytest's configured ``addopts``.

    The runner injects its own empty ``addopts`` override to neutralize a
    repository setting that could turn on xdist.  A later caller-supplied override
    could re-enable parallel execution, so the public argument sources may not
    override this particular ini key.  Other ``-o``/``--override-ini`` keys stay
    available.
    """
    expects_value = False
    for argument in args:
        if expects_value:
            expects_value = False
            value = argument
            if value.partition("=")[0].strip().lower() == "addopts":
                return True
            continue
        if argument in {"-o", "--override-ini"}:
            expects_value = True
            continue
        if argument.startswith("-o"):
            value = argument[2:].lstrip("=")
        elif argument.startswith("--override-ini="):
            value = argument.split("=", 1)[1]
        else:
            continue
        if value.partition("=")[0].strip().lower() == "addopts":
            return True
    return False


_PYTEST_VALUE_OPTIONS = {
    "-c",
    "-k",
    "-m",
    "-o",
    "-p",
    "-r",
    "-W",
    "--basetemp",
    "--capture",
    "--color",
    "--code-highlight",
    "--config-file",
    "--confcutdir",
    "--deselect",
    "--doctest-glob",
    "--doctest-report",
    "--ignore",
    "--ignore-glob",
    "--import-mode",
    "--junit-prefix",
    "--durations",
    "--durations-min",
    "--debug",
    "--log-auto-indent",
    "--log-cli-level",
    "--log-cli-format",
    "--log-cli-date-format",
    "--log-disable",
    "--log-level",
    "--log-file",
    "--log-file-date-format",
    "--log-file-mode",
    "--log-file-format",
    "--log-file-level",
    "--log-format",
    "--log-date-format",
    "--maxfail",
    "--override-ini",
    "--pastebin",
    "--pdbcls",
    "--pythonwarnings",
    "--rootdir",
    "--show-capture",
    "--tb",
    "--verbosity",
}

_RESOURCE_FAILURE_STATUSES = {
    "OOM",
    "TIMEOUT",
    "RESOURCE_KILL",
    "FILE_SIZE_LIMIT",
    "CLEANUP_FAILURE",
}

_FILE_SIZE_LIMIT_PATTERN = re.compile(
    r"(?:sigxfsz|errno\s*[:=]?\s*27\b[^\n]{0,120}(?:file too large|"
    r"file size limit))",
    flags=re.IGNORECASE,
)


def _file_size_limit_evidence(path: Path, max_bytes: int = 16_000) -> bool:
    """Look for a bounded tail indicating a handled ``RLIMIT_FSIZE`` error."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            tail = stream.read(max_bytes).decode("utf-8", errors="replace")
    except OSError:
        return False
    return bool(_FILE_SIZE_LIMIT_PATTERN.search(tail))


def _contains_positional_pytest_argument(args: list[str]) -> str | None:
    """Find a bare argument that could inject an extra pytest target.

    Test targets are owned by ``--test-targets``.  Separate-value forms of common
    pytest options remain supported; plugin-specific options should use the
    ``--option=value`` spelling so the ownership rule stays unambiguous.
    """
    expects_value = False
    for index, argument in enumerate(args):
        if expects_value:
            expects_value = False
            continue
        if argument == "--":
            return args[index + 1] if index + 1 < len(args) else None
        if not argument.startswith("-"):
            return argument
        if argument in _PYTEST_VALUE_OPTIONS:
            expects_value = True
    if expects_value:
        raise ValueError(
            "A pytest option is missing its value; use --option=value or provide "
            "the value after a supported option."
        )
    return None


def _pytest_arguments(
    config: SafeTestConfig,
    junit_path: Path,
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Build pytest arguments, folding in safe options from ``PYTEST_ADDOPTS``.

    Environment options are made explicit in the command so the recorded command
    is sufficient to reproduce a batch.  Parallel execution and caller-owned JUnit
    paths are rejected in both sources to keep isolation and artifact ownership
    deterministic.
    """
    source_environment = os.environ if environment is None else environment
    environment_arguments = shlex.split(source_environment.get("PYTEST_ADDOPTS", ""))
    if _contains_addopts_override(environment_arguments):
        raise ValueError(
            "Overriding pytest addopts is disabled by run_tests_safe; pass safe "
            "options directly through --pytest-args instead."
        )
    if _contains_parallel_pytest_option(environment_arguments):
        raise ValueError(
            "Parallel pytest options in PYTEST_ADDOPTS are disabled by "
            "run_tests_safe; use separate serial batches instead of -n/--dist."
        )
    if _contains_junit_option(environment_arguments):
        raise ValueError(
            "Do not set a custom JUnit path in PYTEST_ADDOPTS; run_tests_safe "
            "assigns one artifact per isolated batch."
        )

    caller_arguments = shlex.split(config.pytest_args)
    if _contains_addopts_override(caller_arguments):
        raise ValueError(
            "Overriding pytest addopts is disabled by run_tests_safe; pass safe "
            "options directly through --pytest-args instead."
        )
    # A repository-level ``addopts`` can silently enable xdist.  Neutralize that
    # implicit source; callers can opt into safe serial options explicitly via
    # ``--pytest-args`` or ``PYTEST_ADDOPTS`` (which is copied above).
    arguments = ["-o", "addopts="] + environment_arguments + caller_arguments
    if _contains_parallel_pytest_option(arguments):
        raise ValueError(
            "Parallel pytest options are disabled by run_tests_safe; use separate "
            "serial batches instead of -n/--numprocesses/--dist."
        )
    if _contains_junit_option(arguments):
        raise ValueError(
            "Do not pass a custom JUnit path; run_tests_safe assigns one artifact "
            "per isolated batch."
        )
    positional = _contains_positional_pytest_argument(arguments)
    if positional is not None:
        raise ValueError(
            f"Positional pytest argument {positional!r} is not allowed; pass test "
            "files through --test-targets instead."
        )
    # Keep a caller-provided ``--`` separator in place: generated options must be
    # parsed by pytest rather than accidentally becoming a positional target.
    try:
        separator_index = arguments.index("--")
    except ValueError:
        arguments.append(f"--junitxml={junit_path}")
    else:
        arguments.insert(separator_index, f"--junitxml={junit_path}")
    return arguments


def _base_environment(config: SafeTestConfig) -> dict[str, str]:
    environment = os.environ.copy()
    count = str(config.thread_count)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = count
    # ``_pytest_arguments`` copies safe options into the recorded argv.  Clear the
    # variable here so pytest cannot apply them a second time in the child process.
    environment["PYTEST_ADDOPTS"] = ""
    environment["PYTHONUNBUFFERED"] = "1"
    if config.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    return environment


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ValueError(f"Required executable not found on PATH: {name}")
    return path


def _systemd_user_scope_available(
    systemd_run: str,
    config: SafeTestConfig,
) -> bool:
    """Probe the user manager with the same cgroup properties as a batch."""
    true_executable = shutil.which("true") or "/bin/true"
    unit_name = f"aao-probe-{os.getpid()}-{time.time_ns()}"
    try:
        result = subprocess.run(
            [
                systemd_run,
                "--user",
                "--scope",
                "--quiet",
                "--collect",
                "--unit",
                unit_name,
                "-p",
                f"CPUQuota={config.cpu_quota_percent}%",
                "-p",
                f"MemoryHigh={config.memory_high_mb}M",
                "-p",
                f"MemoryMax={config.memory_max_mb}M",
                "-p",
                f"MemorySwapMax={config.memory_swap_max_mb}M",
                "-p",
                f"TasksMax={config.tasks_max}",
                "-p",
                "RuntimeMaxSec=10s",
                "-p",
                f"TimeoutStopSec={config.kill_after_seconds}s",
                "-p",
                "KillMode=control-group",
                "--",
                true_executable,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _select_launcher(config: SafeTestConfig) -> Literal["systemd", "prlimit"]:
    systemd_run = shutil.which("systemd-run")
    systemctl = shutil.which("systemctl")
    has_prlimit = shutil.which("prlimit") is not None
    if config.launcher == "systemd":
        if systemd_run is None or systemctl is None:
            raise ValueError(
                "--launcher=systemd requested, but systemd-run/systemctl are unavailable"
            )
        if not has_prlimit:
            raise ValueError(
                "--launcher=systemd requested, but prlimit is unavailable for "
                "the per-file limit"
            )
        if not _systemd_user_scope_available(systemd_run, config):
            raise ValueError(
                "--launcher=systemd requested, but the user systemd manager is unavailable"
            )
        return "systemd"
    if config.launcher == "prlimit":
        if not has_prlimit:
            raise ValueError("--launcher=prlimit requested, but prlimit is unavailable")
        return "prlimit"
    if (
        systemd_run is not None
        and systemctl is not None
        and has_prlimit
        and _systemd_user_scope_available(systemd_run, config)
    ):
        return "systemd"
    if has_prlimit:
        if config.memory_swap_max_mb:
            raise ValueError(
                "systemd user scopes are unavailable and the prlimit fallback "
                "cannot enforce --memory-swap-max-mb; set it to 0 or use "
                "--launcher=systemd."
            )
        return "prlimit"
    raise ValueError(
        "No bounded launcher is available; install systemd-run or prlimit. "
        "The script refuses to run pytest without resource limits."
    )


def _payload(
    config: SafeTestConfig,
    pytest_args: list[str],
    targets: tuple[str, ...],
) -> list[str]:
    nice = _tool("nice")
    ionice = _tool("ionice")
    taskset = _tool("taskset")
    return [
        nice,
        "-n",
        "19",
        ionice,
        "-c",
        "3",
        taskset,
        "-c",
        config.cpu_set,
        str(config.python_executable.expanduser().resolve()),
        "-m",
        "pytest",
        *pytest_args,
        *targets,
    ]


def _build_command(
    config: SafeTestConfig,
    launcher: Literal["systemd", "prlimit"],
    batch: _Batch,
    junit_path: Path,
    unit_name: str,
) -> list[str]:
    pytest_args = _pytest_arguments(config, junit_path)
    payload = _payload(config, pytest_args, batch.targets)
    timeout = _tool("timeout")
    timeout_payload = [
        timeout,
        "--foreground",
        "--signal=TERM",
        f"--kill-after={config.kill_after_seconds}s",
        f"{config.timeout_minutes}m",
        *payload,
    ]
    if launcher == "prlimit":
        return [
            _tool("prlimit"),
            f"--as={int(config.memory_max_mb) * 1024 * 1024}",
            f"--cpu={int(config.timeout_minutes) * 60 + int(config.kill_after_seconds)}",
            f"--fsize={int(config.max_file_size_mb) * 1024 * 1024}",
            "--",
            *timeout_payload,
        ]
    # ``LimitFSIZE`` is an exec resource property and is rejected by systemd
    # scope units on supported distributions.  Apply the same RLIMIT_FSIZE in a
    # small inherited wrapper so systemd retains its cgroup-only properties.
    timeout_payload = [
        _tool("prlimit"),
        f"--fsize={int(config.max_file_size_mb) * 1024 * 1024}",
        "--",
        *timeout_payload,
    ]
    runtime_seconds = (
        int(config.timeout_minutes) * 60 + int(config.kill_after_seconds) + 10
    )
    return [
        _tool("systemd-run"),
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        unit_name,
        "--collect",
        "-p",
        f"CPUQuota={config.cpu_quota_percent}%",
        "-p",
        f"MemoryHigh={config.memory_high_mb}M",
        "-p",
        f"MemoryMax={config.memory_max_mb}M",
        "-p",
        f"MemorySwapMax={config.memory_swap_max_mb}M",
        "-p",
        f"TasksMax={config.tasks_max}",
        "-p",
        f"RuntimeMaxSec={runtime_seconds}s",
        "-p",
        f"TimeoutStopSec={config.kill_after_seconds}s",
        "-p",
        "KillMode=control-group",
        "--",
        *timeout_payload,
    ]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _create_output_dir(config: SafeTestConfig, root: Path) -> Path:
    if config.output_dir is not None:
        output = config.output_dir.expanduser()
        output = output if output.is_absolute() else root / output
        output = output.resolve()
        if output.exists():
            raise ValueError(f"Output directory already exists: {output}")
    else:
        base = root / "outputs" / "test-runs"
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output = base / stamp
        suffix = 1
        while output.exists():
            suffix += 1
            output = base / f"{stamp}-{suffix:02d}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    revision = result.stdout.strip()
    return revision or None


def _git_dirty(root: Path) -> bool | None:
    """Return whether uncommitted tracked/untracked changes are present."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode:
        return None
    return bool(result.stdout.strip())


def _unit_properties(unit_name: str) -> dict[str, str]:
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return {}
    try:
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "show",
                f"{unit_name}.scope",
                "-p",
                "Result",
                "-p",
                "OOMKilled",
                "-p",
                "ExecMainCode",
                "-p",
                "ExecMainStatus",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    properties: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            properties[key] = value
    return properties


def _unit_journal(unit_name: str) -> str:
    """Return a bounded diagnostic tail for resource-kill classification."""
    journalctl = shutil.which("journalctl")
    if journalctl is None:
        return ""
    try:
        result = subprocess.run(
            [
                journalctl,
                "--user",
                "-u",
                f"{unit_name}.scope",
                "-n",
                "100",
                "--no-pager",
                "-o",
                "cat",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout[-8_000:]


def _stop_unit(unit_name: str) -> bool:
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return False
    try:
        result = subprocess.run(
            [systemctl, "--user", "stop", f"{unit_name}.scope"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _cleanup_unit(unit_name: str) -> None:
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return
    try:
        subprocess.run(
            [systemctl, "--user", "reset-failed", f"{unit_name}.scope"],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return


def _unit_is_active(unit_name: str) -> bool | None:
    """Return whether a transient scope remains active (or ``None`` if unknown)."""
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return None
    try:
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "show",
                f"{unit_name}.scope",
                "-p",
                "ActiveState",
                "--value",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return False
    return result.stdout.strip() not in {"", "inactive", "failed", "dead"}


def _kill_unit(unit_name: str) -> bool:
    """Force-kill every process still attached to a transient scope."""
    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return False
    try:
        result = subprocess.run(
            [
                systemctl,
                "--user",
                "kill",
                "--kill-who=all",
                "--signal=SIGKILL",
                f"{unit_name}.scope",
            ],
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _send_process_group_signal(process: subprocess.Popen[str], signum: int) -> bool:
    """Signal the wrapper and every descendant in its private session."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            process.send_signal(signum)
        except ProcessLookupError:
            return False
        return True
    return True


def _process_group_exists(process: subprocess.Popen[str]) -> bool:
    """Report whether the private process group still has any member."""
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # A private session should only contain our descendants; permission
        # errors therefore conservatively mean that cleanup is incomplete.
        return True
    return True


def _terminate_process_unmasked(
    process: subprocess.Popen[str],
    launcher: Literal["systemd", "prlimit"],
    unit_name: str,
    grace_seconds: int,
) -> _TerminationResult:
    """Terminate a batch and its descendants, returning its final code."""
    # ``start_new_session=True`` gives the wrapper a private process group.  Try
    # the group first even when the wrapper has exited: a timeout helper can leave
    # grandchildren behind after its own PID disappears.
    group_exists = _process_group_exists(process)
    term_signal_delivered = (
        _send_process_group_signal(process, signal.SIGTERM) if group_exists else False
    )
    stop_acknowledged = True
    if launcher == "systemd":
        stop_acknowledged = _stop_unit(unit_name)
    deadline = time.monotonic() + max(1, grace_seconds)
    while time.monotonic() < deadline:
        if process.poll() is None:
            remaining = max(0.01, deadline - time.monotonic())
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                pass
        if not _process_group_exists(process):
            break
        time.sleep(0.05)

    cleanup_incomplete = _process_group_exists(process)
    if cleanup_incomplete:
        _send_process_group_signal(process, signal.SIGKILL)
        try:
            process.kill()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return _TerminationResult(process.poll(), cleanup_incomplete=True)
        cleanup_incomplete = _process_group_exists(process)
    elif process.poll() is None:
        # The leader can outlive its process group probe in unusual procfs
        # implementations; still reap it without an unbounded wait.
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return _TerminationResult(process.poll(), cleanup_incomplete=True)
    unit_active = _unit_is_active(unit_name) if launcher == "systemd" else False
    if launcher == "systemd" and unit_active is True:
        _kill_unit(unit_name)
        time.sleep(0.1)
        unit_active = _unit_is_active(unit_name)
    if launcher == "systemd" and (
        unit_active is True or (unit_active is None and not stop_acknowledged)
    ):
        cleanup_incomplete = True
    if not term_signal_delivered and _process_group_exists(process):
        cleanup_incomplete = True
    return _TerminationResult(process.returncode, cleanup_incomplete)


def _terminate_process(
    process: subprocess.Popen[str],
    launcher: Literal["systemd", "prlimit"],
    unit_name: str,
    grace_seconds: int,
) -> _TerminationResult:
    """Terminate a batch while ignoring repeated external interrupts."""
    with _suppress_interrupt_signals():
        return _terminate_process_unmasked(
            process,
            launcher,
            unit_name,
            grace_seconds,
        )


def _classify_batch(
    returncode: int | None,
    interrupted: bool,
    systemd_properties: Mapping[str, str],
    journal: str,
    launch_error: str | None = None,
    timeout_expired: bool = False,
    file_size_limit_hit: bool = False,
) -> tuple[str, str | None]:
    """Map process and cgroup evidence to a stable manifest status."""
    if interrupted:
        return "INTERRUPTED", "user interrupt"
    if launch_error is not None:
        return "LAUNCH_FAILURE", launch_error
    result = systemd_properties.get("Result", "")
    oom_marked = (
        result == "oom-kill"
        or systemd_properties.get("OOMKilled", "").lower() == "yes"
        or bool(
            re.search(
                r"\b(?:oom-kill(?:ed)?|oomd|oom killer|memory cgroup out of memory|"
                r"invoked oom-killer|killed process \d+)\b",
                journal,
                flags=re.IGNORECASE,
            )
        )
    )
    if oom_marked:
        return "OOM", "systemd or journal reported a memory kill"
    if timeout_expired:
        return "TIMEOUT", "batch wrapper exceeded the parent wait limit"
    if result in {"timeout", "watchdog"} or returncode == 124:
        return "TIMEOUT", "batch wall-clock limit"
    if file_size_limit_hit or returncode in {128 + signal.SIGXFSZ, -signal.SIGXFSZ}:
        return "FILE_SIZE_LIMIT", "a batch file exceeded --max-file-size-mb"
    if returncode in {137, -9}:
        return (
            "RESOURCE_KILL",
            "batch was SIGKILLed (possible OOM or timeout escalation)",
        )
    if returncode == 0:
        return "PASSED", None
    if returncode is None:
        return "LAUNCH_FAILURE", "batch process could not be started"
    return "TEST_FAILURE", f"pytest return code {returncode}"


@contextmanager
def _interrupt_signals() -> Iterator[None]:
    """Turn external termination into the same cleanup path as Ctrl-C."""
    if threading.current_thread() is not threading.main_thread():
        # Signal handlers are process-global and Python only permits installing
        # them from the main thread.  The CLI always runs there; library callers
        # in a worker thread retain the caller's normal signal behavior.
        yield
        return
    previous: dict[int, Any] = {}

    def raise_keyboard_interrupt(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, raise_keyboard_interrupt)
    except ValueError:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        yield
        return
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _finalize_batch(
    *,
    batch: _Batch,
    output: Path,
    manifest: dict[str, Any],
    command: tuple[str, ...],
    log_path: Path,
    junit_path: Path,
    started: float,
    returncode: int | None,
    interrupted: bool,
    systemd_properties: Mapping[str, str],
    journal: str,
    launch_error: str | None = None,
    cleanup_incomplete: bool = False,
    timeout_expired: bool = False,
    file_size_limit_hit: bool = False,
    manifest_lock: threading.Lock | None = None,
    manifest_open: threading.Event | None = None,
) -> _BatchResult:
    """Persist one batch outcome in one place for normal and launch failures."""
    elapsed = time.monotonic() - started
    status, reason = _classify_batch(
        returncode,
        interrupted,
        systemd_properties,
        journal,
        launch_error,
        timeout_expired,
        file_size_limit_hit,
    )
    if cleanup_incomplete:
        original_status = status
        if status != "INTERRUPTED":
            status = "CLEANUP_FAILURE"
        cleanup_reason = "cleanup did not finish within the hard limit"
        reason = f"{cleanup_reason}; original status={original_status}" + (
            f" ({reason})" if reason else ""
        )
    finished_at = _now()
    with log_path.open("a", encoding="utf-8") as log_stream:
        log_stream.write(
            f"\nfinished_at: {finished_at}\nstatus: {status}\n"
            f"returncode: {returncode}\nelapsed_seconds: {elapsed:.3f}\n"
        )
    result = _BatchResult(
        status=status,
        returncode=returncode,
        elapsed_seconds=elapsed,
        log_path=log_path,
        junit_path=junit_path,
        command=command,
        reason=reason,
    )
    try:
        log_reference = str(result.log_path.relative_to(output))
    except ValueError:
        log_reference = str(result.log_path)
    try:
        junit_reference = str(result.junit_path.relative_to(output))
    except ValueError:
        junit_reference = str(result.junit_path)
    with _manifest_guard(manifest_lock):
        if manifest_open is None or manifest_open.is_set():
            entry = manifest["batches"][batch.index - 1]
            entry.update(
                {
                    "status": result.status,
                    "returncode": result.returncode,
                    "reason": result.reason,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "finished_at": finished_at,
                    "log": log_reference,
                    "junit": junit_reference,
                    "command": list(result.command),
                    "cleanup_incomplete": cleanup_incomplete,
                }
            )
            _write_json(output / "metadata.json", manifest)
    print(f"  {result.status} ({result.elapsed_seconds:.1f}s); log={result.log_path}")
    return result


@contextmanager
def _suppress_interrupt_signals() -> Iterator[None]:
    """Keep diagnostics/cleanup atomic after the first termination request."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, signal.SIG_IGN)
    except ValueError:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        yield
        return
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _batch_unit_name(batch_index: int) -> str:
    """Create a unique, systemd-safe transient unit name."""
    return f"aao-test-{os.getpid()}-{time.time_ns()}-{batch_index:03d}"


def _batch_needs_termination(
    process: subprocess.Popen[str],
    launcher: Literal["systemd", "prlimit"],
    unit_name: str,
) -> bool:
    """Detect descendants that outlived the launcher wrapper."""
    if process.poll() is None:
        return True
    if launcher == "prlimit":
        return _process_group_exists(process)
    # An unknown systemd state is handled conservatively; a completed and
    # collected scope returns ``False`` and avoids an unnecessary stop call.
    return _unit_is_active(unit_name) is not False


def _run_batch(
    config: SafeTestConfig,
    root: Path,
    launcher: Literal["systemd", "prlimit"],
    batch: _Batch,
    output: Path,
    manifest: dict[str, Any],
    *,
    manifest_lock: threading.Lock | None = None,
    active_registry: _ActiveProcessRegistry | None = None,
    manifest_open: threading.Event | None = None,
) -> _BatchResult:
    if manifest_open is not None and not manifest_open.is_set():
        return _synthetic_batch_result(
            batch=batch,
            output=output,
            manifest=manifest,
            reason="batch was queued after the run was closed",
            interrupted=True,
            manifest_lock=manifest_lock,
            manifest_open=manifest_open,
        )
    log_path = output / "logs" / f"batch-{batch.index:03d}.log"
    junit_path = output / "junit" / f"batch-{batch.index:03d}.xml"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_name = _batch_unit_name(batch.index)
    command: tuple[str, ...] = ()
    with _manifest_guard(manifest_lock):
        if manifest_open is None or manifest_open.is_set():
            manifest_entry = manifest["batches"][batch.index - 1]
            manifest_entry["command"] = None
            manifest_entry["junit"] = str(junit_path.relative_to(output))
            manifest_entry["unit"] = unit_name if launcher == "systemd" else None
            _write_json(output / "metadata.json", manifest)
    started = time.monotonic()
    started_at = _now()
    try:
        command = tuple(_build_command(config, launcher, batch, junit_path, unit_name))
    except (OSError, ValueError) as exc:
        log_path.write_text(
            f"started_at: {started_at}\ncommand: <not built>\n\n"
            f"batch launch failed: {exc!r}\n",
            encoding="utf-8",
        )
        return _finalize_batch(
            batch=batch,
            output=output,
            manifest=manifest,
            command=command,
            log_path=log_path,
            junit_path=junit_path,
            started=started,
            returncode=None,
            interrupted=False,
            systemd_properties={},
            journal="",
            launch_error=f"batch launch failed: {exc}",
            manifest_lock=manifest_lock,
            manifest_open=manifest_open,
        )
    with _manifest_guard(manifest_lock):
        if manifest_open is None or manifest_open.is_set():
            manifest_entry = manifest["batches"][batch.index - 1]
            manifest_entry["command"] = list(command)
            _write_json(output / "metadata.json", manifest)
    log_path.write_text(
        f"started_at: {started_at}\ncommand: {shlex.join(command)}\n\n",
        encoding="utf-8",
    )
    print(f"[{batch.index}/{len(manifest['batches'])}] {', '.join(batch.targets)}")
    process: subprocess.Popen[str] | None = None
    interrupted = False
    returncode: int | None = None
    cleanup_incomplete = False
    timeout_expired = False
    file_size_limit_hit = False
    launch_error: str | None = None
    systemd_properties: dict[str, str] = {}
    journal = ""
    registered = False
    with log_path.open("a", encoding="utf-8") as log_stream:
        try:
            # Keep the short fork/exec window atomic.  If an external signal
            # arrives after the child is forked but before ``Popen`` returns,
            # ``process`` must still be assigned so the cleanup path can reap it.
            with _suppress_interrupt_signals():
                process = subprocess.Popen(
                    list(command),
                    cwd=root,
                    env=_base_environment(config),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
            if active_registry is not None:
                # Register immediately after Popen returns.  The registry's
                # stop latch closes the fork/register interruption window.
                active_registry.register(batch.index, process, launcher, unit_name)
                registered = True
                if active_registry.stop_requested and process.poll() is None:
                    interrupted = True
                    termination = _terminate_process(
                        process,
                        launcher,
                        unit_name,
                        int(config.kill_after_seconds),
                    )
                    returncode = termination.returncode
                    cleanup_incomplete = (
                        cleanup_incomplete or termination.cleanup_incomplete
                    )
            try:
                parent_wait_seconds = (
                    int(config.timeout_minutes) * 60
                    + int(config.kill_after_seconds)
                    + 15
                )
                returncode = process.wait(timeout=parent_wait_seconds)
                if (
                    active_registry is not None
                    and active_registry.stop_requested
                    and returncode not in {0, None}
                ):
                    interrupted = True
            except subprocess.TimeoutExpired:
                timeout_expired = True
                termination = _terminate_process(
                    process,
                    launcher,
                    unit_name,
                    int(config.kill_after_seconds),
                )
                returncode = termination.returncode
                cleanup_incomplete = (
                    cleanup_incomplete or termination.cleanup_incomplete
                )
            except KeyboardInterrupt:
                interrupted = True
                termination = _terminate_process(
                    process,
                    launcher,
                    unit_name,
                    int(config.kill_after_seconds),
                )
                returncode = termination.returncode
                cleanup_incomplete = (
                    cleanup_incomplete or termination.cleanup_incomplete
                )
        except KeyboardInterrupt:
            interrupted = True
            if process is not None:
                termination = _terminate_process(
                    process,
                    launcher,
                    unit_name,
                    int(config.kill_after_seconds),
                )
                returncode = termination.returncode
                cleanup_incomplete = (
                    cleanup_incomplete or termination.cleanup_incomplete
                )
        except OSError as exc:
            returncode = None
            launch_error = f"batch launch failed: {exc}"
            log_stream.write(f"batch launch failed: {exc!r}\n")
        except (RuntimeError, TypeError, ValueError) as exc:
            returncode = None
            launch_error = f"batch execution failed: {exc}"
            log_stream.write(f"batch execution failed: {exc!r}\n")
        finally:
            try:
                with _suppress_interrupt_signals():
                    if process is not None and _batch_needs_termination(
                        process, launcher, unit_name
                    ):
                        termination = _terminate_process(
                            process,
                            launcher,
                            unit_name,
                            int(config.kill_after_seconds),
                        )
                        returncode = termination.returncode
                        cleanup_incomplete = (
                            cleanup_incomplete or termination.cleanup_incomplete
                        )
                    if launcher == "systemd" and process is not None:
                        # Capture cgroup/journal evidence before reset-failed removes
                        # the transient unit's result metadata.
                        systemd_properties = _unit_properties(unit_name)
                        journal = _unit_journal(unit_name)
                        _cleanup_unit(unit_name)
                    if journal:
                        log_stream.write(
                            f"\nsystemd_journal_tail:\n{journal.rstrip()}\n"
                        )
            finally:
                if active_registry is not None and registered:
                    active_registry.unregister(batch.index)
    file_size_limit_hit = _file_size_limit_evidence(log_path)
    return _finalize_batch(
        batch=batch,
        output=output,
        manifest=manifest,
        command=command,
        log_path=log_path,
        junit_path=junit_path,
        started=started,
        returncode=returncode,
        interrupted=interrupted,
        systemd_properties=systemd_properties,
        journal=journal,
        launch_error=launch_error,
        cleanup_incomplete=cleanup_incomplete,
        timeout_expired=timeout_expired,
        file_size_limit_hit=file_size_limit_hit,
        manifest_lock=manifest_lock,
        manifest_open=manifest_open,
    )


def _synthetic_batch_result(
    *,
    batch: _Batch,
    output: Path,
    manifest: dict[str, Any],
    reason: str,
    interrupted: bool,
    manifest_lock: threading.Lock,
    manifest_open: threading.Event | None = None,
) -> _BatchResult:
    """Create a persisted result when a scheduler worker cannot return one."""
    log_path = output / "logs" / f"batch-{batch.index:03d}.log"
    junit_path = output / "junit" / f"batch-{batch.index:03d}.xml"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(
            f"started_at: {_now()}\ncommand: <worker did not return>\n\n",
            encoding="utf-8",
        )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"scheduler: {reason}\n")
    return _finalize_batch(
        batch=batch,
        output=output,
        manifest=manifest,
        command=(),
        log_path=log_path,
        junit_path=junit_path,
        started=time.monotonic(),
        returncode=None,
        interrupted=interrupted,
        systemd_properties={},
        journal="",
        launch_error=None if interrupted else reason,
        manifest_lock=manifest_lock,
        manifest_open=manifest_open,
    )


def _run_batches(
    config: SafeTestConfig,
    root: Path,
    launcher: Literal["systemd", "prlimit"],
    batches: list[_Batch],
    output: Path,
    manifest: dict[str, Any],
) -> tuple[list[_BatchResult], bool]:
    """Run a bounded number of isolated batches concurrently.

    The scheduler deliberately owns dispatch and fail-fast policy while
    ``_run_batch`` owns one batch's process/resource lifecycle.  At most
    ``max_concurrency`` futures are submitted; all futures completing in one
    observation window are consumed before another batch is dispatched.  This
    prevents a fast failure from being hidden behind an unbounded queue.
    """
    if not batches:
        return [], False

    worker_count = max(1, min(int(config.max_concurrency), len(batches)))
    manifest_lock = threading.Lock()
    active_registry = _ActiveProcessRegistry()
    manifest_open = threading.Event()
    manifest_open.set()
    completion_times: dict[int, float] = {}
    completion_sequence: dict[int, int] = {}
    completion_times_lock = threading.Lock()
    completion_sequence_counter = [0]
    futures: dict[Future[_BatchResult], _Batch] = {}
    results_by_index: dict[int, _BatchResult] = {}
    completion_order: list[int] = []
    submitted: set[int] = set()
    in_flight = 0
    peak_in_flight = 0
    interrupted = False
    stop_dispatch = False
    next_batch_position = 0

    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    (output / "junit").mkdir(parents=True, exist_ok=True)
    with _manifest_guard(manifest_lock):
        concurrency = manifest.setdefault("concurrency", {})
        requested = int(concurrency.get("requested", int(config.max_concurrency)))
        concurrency.update(
            {
                "requested": requested,
                "effective": worker_count,
                "reasons": list(concurrency.get("reasons", [])),
                "peak_active": int(concurrency.get("peak_active", 0)),
                "completion_order": list(concurrency.get("completion_order", [])),
            }
        )
        _write_json(output / "metadata.json", manifest)

    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="aao-test-batch",
    )

    def mark_running(batch: _Batch) -> None:
        with _manifest_guard(manifest_lock):
            entry = manifest["batches"][batch.index - 1]
            entry.update({"started_at": _now(), "status": "RUNNING"})
            _write_json(output / "metadata.json", manifest)

    def submit_batch(batch: _Batch) -> None:
        nonlocal in_flight, next_batch_position, peak_in_flight
        mark_running(batch)

        def invoke() -> _BatchResult:
            try:
                return _run_batch(
                    config,
                    root,
                    launcher,
                    batch,
                    output,
                    manifest,
                    manifest_lock=manifest_lock,
                    active_registry=active_registry,
                    manifest_open=manifest_open,
                )
            finally:
                _record_batch_completion(
                    batch.index,
                    completion_times,
                    completion_sequence,
                    completion_sequence_counter,
                    completion_times_lock,
                )

        # Keep submission and bookkeeping atomic with respect to the CLI signal
        # handler.  Otherwise a SIGINT could leave a running future absent from
        # ``futures`` (or absent from ``submitted``) and therefore outside the
        # cleanup/accounting path.
        with _suppress_interrupt_signals():
            future = executor.submit(invoke)
            futures[future] = batch
            submitted.add(batch.index)
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
            next_batch_position += 1
        with _manifest_guard(manifest_lock):
            concurrency = manifest.setdefault("concurrency", {})
            concurrency["peak_active"] = max(
                int(concurrency.get("peak_active", 0)), peak_in_flight
            )
            _write_json(output / "metadata.json", manifest)

    def mark_not_started(batch: _Batch, reason: str) -> None:
        with _manifest_guard(manifest_lock):
            entry = manifest["batches"][batch.index - 1]
            if entry.get("status") in {"NOT_STARTED", "RUNNING"}:
                entry.update({"status": "NOT_STARTED", "reason": reason})
            _write_json(output / "metadata.json", manifest)

    def cancel_pending(reason: str) -> None:
        """Cancel executor work that has not begun and mark it explicitly."""
        nonlocal in_flight
        for future, batch in list(futures.items()):
            if future.cancel():
                futures.pop(future, None)
                in_flight = max(0, in_flight - 1)
                mark_not_started(batch, reason)

    def completed_futures() -> set[Future[_BatchResult]]:
        return {future for future in futures if future.done()}

    def ordered_futures(
        done: set[Future[_BatchResult]],
    ) -> list[Future[_BatchResult]]:
        with completion_times_lock:
            return sorted(
                done,
                key=lambda future: (
                    completion_sequence.get(futures[future].index, 10**18),
                    completion_times.get(futures[future].index, float("inf")),
                    futures[future].index,
                ),
            )

    def record_result(batch: _Batch, result: _BatchResult) -> None:
        """Make scheduler metadata complete even for injected/test workers."""
        try:
            log_reference = str(result.log_path.relative_to(output))
        except ValueError:
            log_reference = str(result.log_path)
        try:
            junit_reference = str(result.junit_path.relative_to(output))
        except ValueError:
            junit_reference = str(result.junit_path)
        with _manifest_guard(manifest_lock):
            if not manifest_open.is_set():
                return
            entry = manifest["batches"][batch.index - 1]
            entry.update(
                {
                    "status": result.status,
                    "returncode": result.returncode,
                    "reason": result.reason,
                    "elapsed_seconds": round(result.elapsed_seconds, 3),
                    "finished_at": entry.get("finished_at") or _now(),
                    "log": log_reference,
                    "junit": junit_reference,
                    "command": list(result.command),
                }
            )
            _write_json(output / "metadata.json", manifest)

    def consume_future(future: Future[_BatchResult]) -> None:
        nonlocal in_flight, interrupted, stop_dispatch
        batch = futures.pop(future)
        in_flight = max(0, in_flight - 1)
        try:
            result = future.result()
            if not isinstance(result, _BatchResult):
                raise TypeError(
                    "batch worker returned an unexpected result type: "
                    f"{type(result).__name__}"
                )
        except KeyboardInterrupt:
            interrupted = True
            stop_dispatch = True
            with _suppress_interrupt_signals():
                active_registry.request_stop()
            result = _synthetic_batch_result(
                batch=batch,
                output=output,
                manifest=manifest,
                reason="batch worker raised KeyboardInterrupt",
                interrupted=True,
                manifest_lock=manifest_lock,
                manifest_open=manifest_open,
            )
        except BaseException as exc:  # noqa: BLE001 - isolate worker failures
            result = _synthetic_batch_result(
                batch=batch,
                output=output,
                manifest=manifest,
                reason=f"batch worker raised {type(exc).__name__}: {exc}",
                interrupted=False,
                manifest_lock=manifest_lock,
                manifest_open=manifest_open,
            )
        record_result(batch, result)
        results_by_index[batch.index] = result
        completion_order.append(batch.index)
        with _manifest_guard(manifest_lock):
            concurrency = manifest.setdefault("concurrency", {})
            concurrency["completion_order"] = list(completion_order)
            _write_json(output / "metadata.json", manifest)
        if result.status == "INTERRUPTED":
            interrupted = True
            stop_dispatch = True
            with _suppress_interrupt_signals():
                active_registry.request_stop()
            cancel_pending("stopped after an interruption")
        elif result.status in _RESOURCE_FAILURE_STATUSES or (
            result.status != "PASSED" and not config.continue_on_failure
        ):
            stop_dispatch = True

    def drain_until(deadline: float | None) -> None:
        nonlocal interrupted
        while futures and (deadline is None or time.monotonic() < deadline):
            timeout = None
            if deadline is not None:
                timeout = max(0.01, min(0.2, deadline - time.monotonic()))
            done, _pending = wait(
                tuple(futures), timeout=timeout, return_when=FIRST_COMPLETED
            )
            if not done:
                continue
            ordered_done = ordered_futures(set(done))
            for future in ordered_done:
                if future in futures:
                    consume_future(future)

    try:
        while next_batch_position < len(batches) and len(futures) < worker_count:
            submit_batch(batches[next_batch_position])

        while futures:
            done, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
            # Consume every completion observed in this window before refilling
            # slots, so simultaneous failures cannot be outrun by new work.
            while done or completed_futures():
                done.update(completed_futures())
                for future in ordered_futures(set(done)):
                    if future in futures:
                        consume_future(future)
                done = set()
            if stop_dispatch:
                continue
            while next_batch_position < len(batches) and len(futures) < worker_count:
                submit_batch(batches[next_batch_position])
    except KeyboardInterrupt:
        interrupted = True
        stop_dispatch = True
        with _suppress_interrupt_signals():
            active_registry.request_stop()
            cancel_pending("stopped before batch process started")
            # Give running workers their configured cleanup grace period, then
            # escalate process groups/scopes once more.  This wait is intentionally
            # bounded so Ctrl-C cannot leave the coordinator stuck indefinitely.
            drain_until(time.monotonic() + max(5, int(config.kill_after_seconds) + 5))
            if futures:
                active_registry.force_kill_all()
                drain_until(time.monotonic() + 5)
    finally:
        # Do not let a late worker rewrite a terminal manifest.  Real workers
        # normally have drained by this point; this event also protects the
        # bounded-interruption path from pathological Python workers.
        if interrupted:
            with _suppress_interrupt_signals():
                active_registry.request_stop()
                cancel_pending("stopped before batch process started")
                with _manifest_guard(manifest_lock):
                    manifest_open.clear()
        # Normal completion has no live futures and can join workers.  The
        # interruption path uses cancel_futures and a non-blocking shutdown after
        # its bounded drain; active subprocesses have already been signalled.
        with _suppress_interrupt_signals():
            executor.shutdown(wait=not interrupted, cancel_futures=True)

    if interrupted and futures:
        # A pathological Python worker can outlive the bounded drain even after
        # its child process was killed.  Make the persisted state explicit; the
        # executor is detached above and no new work can be dispatched.
        with _manifest_guard(manifest_lock):
            for batch in futures.values():
                entry = manifest["batches"][batch.index - 1]
                if entry.get("status") == "RUNNING":
                    entry.update(
                        {
                            "status": "INTERRUPTED",
                            "reason": "worker did not finish before interruption cleanup deadline",
                            "finished_at": _now(),
                        }
                    )
            _write_json(output / "metadata.json", manifest)

    for batch in batches:
        if batch.index not in submitted:
            mark_not_started(batch, "stopped before this batch was dispatched")
    if not interrupted:
        with _manifest_guard(manifest_lock):
            manifest_open.clear()
    return [results_by_index[index] for index in sorted(results_by_index)], interrupted


def _record_batch_completion(
    batch_index: int,
    completion_times: dict[int, float],
    completion_sequence: dict[int, int],
    sequence_counter: list[int],
    lock: threading.Lock,
) -> None:
    """Record completion in the worker before its future becomes observable."""
    with lock:
        sequence_counter[0] += 1
        completion_sequence[batch_index] = sequence_counter[0]
        completion_times[batch_index] = time.monotonic()


def _mark_interrupted_batch(
    manifest: dict[str, Any],
    batch: _Batch | None,
    reason: str,
) -> None:
    """Record an interruption that arrived between batch cleanup checkpoints."""
    if batch is None:
        return
    entry = manifest["batches"][batch.index - 1]
    if entry.get("status") in {"NOT_STARTED", "RUNNING"}:
        entry.update(
            {
                "status": "INTERRUPTED",
                "reason": reason,
                "finished_at": _now(),
            }
        )


def _finalize_run(
    *,
    manifest: dict[str, Any],
    batches: list[_Batch],
    results: list[_BatchResult],
    interrupted: bool,
    output: Path,
) -> int:
    """Write the terminal manifest and derive the process exit code."""
    entries = manifest["batches"]
    for entry in entries:
        if entry["status"] == "NOT_STARTED" and not entry.get("reason"):
            entry["reason"] = "stopped before this batch"
    statuses = [entry["status"] for entry in entries]
    manifest["status"] = (
        "INTERRUPTED"
        if interrupted or "INTERRUPTED" in statuses
        else "PASSED"
        if statuses and all(status == "PASSED" for status in statuses)
        else "FAILED"
    )
    manifest["finished_at"] = _now()
    manifest["summary"] = {
        "total_batches": len(batches),
        "started_batches": sum(status != "NOT_STARTED" for status in statuses),
        "passed": statuses.count("PASSED"),
        "test_failures": statuses.count("TEST_FAILURE"),
        "launch_failures": statuses.count("LAUNCH_FAILURE"),
        "cleanup_failures": statuses.count("CLEANUP_FAILURE"),
        "timeouts": statuses.count("TIMEOUT"),
        "oom": statuses.count("OOM"),
        "resource_kills": statuses.count("RESOURCE_KILL"),
        "file_size_limits": statuses.count("FILE_SIZE_LIMIT"),
        "interrupted": statuses.count("INTERRUPTED"),
    }
    _write_json(output / "metadata.json", manifest)
    print(f"Result directory: {output}")
    if manifest["status"] == "INTERRUPTED":
        return 130
    if any(status in {"OOM", "TIMEOUT"} for status in statuses):
        return 2
    if len(results) != len(batches) or any(status != "PASSED" for status in statuses):
        return 1
    return 0


def run_tests(config: SafeTestConfig) -> int:
    """Resolve targets and run resource-bounded pytest batches."""
    root = _resolve_repo_root(config)
    targets = _expand_targets(config, root)
    batches = _make_batches(targets, config)
    launcher = _select_launcher(config)
    concurrency_plan = _plan_concurrency(config, launcher, batches)
    effective_config = config.model_copy(
        update={"max_concurrency": concurrency_plan.effective}
    )
    if config.dry_run:
        print(f"launcher: {launcher}")
        print(f"repo_root: {root}")
        print(f"batches: {len(batches)}")
        print(f"max_concurrency_requested: {concurrency_plan.requested}")
        print(f"max_concurrency_effective: {concurrency_plan.effective}")
        if concurrency_plan.reasons:
            print("max_concurrency_clamp_reasons:")
            for reason in concurrency_plan.reasons:
                print(f"  - {reason}")
        for batch in batches:
            junit = Path("<output>") / "junit" / f"batch-{batch.index:03d}.xml"
            command = _build_command(
                config,
                launcher,
                batch,
                junit,
                f"aao-test-dry-run-{batch.index:03d}",
            )
            print(f"[{batch.index}] {shlex.join(command)}")
        return 0

    output = _create_output_dir(config, root)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_at": _now(),
        "finished_at": None,
        "repo_root": str(root),
        "git_revision": _git_revision(root),
        "git_dirty": _git_dirty(root),
        "launcher": launcher,
        "python_executable": str(config.python_executable.expanduser().resolve()),
        "pytest_args": config.pytest_args,
        "concurrency": {
            "requested": concurrency_plan.requested,
            "effective": concurrency_plan.effective,
            "reasons": list(concurrency_plan.reasons),
            "peak_active": 0,
            "completion_order": [],
        },
        "resource_limits": {
            "cpu_set": config.cpu_set,
            "cpu_quota_percent": config.cpu_quota_percent,
            "memory_high_mb": config.memory_high_mb,
            "memory_max_mb": config.memory_max_mb,
            "memory_swap_max_mb": config.memory_swap_max_mb,
            "tasks_max": config.tasks_max,
            "max_file_size_mb": config.max_file_size_mb,
            "timeout_minutes": config.timeout_minutes,
            "kill_after_seconds": config.kill_after_seconds,
            "thread_count": config.thread_count,
            "cuda_visible_devices": config.cuda_visible_devices,
        },
        "limit_notes": (
            [
                (
                    "prlimit enforces virtual address space and CPU time; it does not "
                    "enforce cgroup RSS, swap, or task-count limits."
                ),
            ]
            if launcher == "prlimit"
            else []
        ),
        "targets": targets,
        "batches": [
            {
                "index": batch.index,
                "targets": list(batch.targets),
                "status": "NOT_STARTED",
                "started_at": None,
                "command": None,
            }
            for batch in batches
        ],
    }
    _write_json(output / "metadata.json", manifest)

    results: list[_BatchResult] = []
    interrupted = False
    try:
        with _interrupt_signals():
            results, interrupted = _run_batches(
                effective_config,
                root,
                launcher,
                batches,
                output,
                manifest,
            )
        with _suppress_interrupt_signals():
            return _finalize_run(
                manifest=manifest,
                batches=batches,
                results=results,
                interrupted=interrupted,
                output=output,
            )
    except KeyboardInterrupt:
        interrupted = True
        with _suppress_interrupt_signals():
            return _finalize_run(
                manifest=manifest,
                batches=batches,
                results=results,
                interrupted=interrupted,
                output=output,
            )


def parse_config(cli_args: list[str] | None = None) -> SafeTestConfig:
    """Parse command-line arguments into a validated test-run configuration."""
    return CliApp.run(SafeTestConfig, cli_args=cli_args)


def main(cli_args: list[str] | None = None) -> None:
    """Run the resource-bounded test command and exit with its status."""
    try:
        exit_code = run_tests(parse_config(cli_args))
    except ValueError as exc:
        print(f"run_tests_safe: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
