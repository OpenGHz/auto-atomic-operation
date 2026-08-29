"""Focused tests for the resource-bounded pytest launcher."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from scripts.run_tests_safe import (
    SafeTestConfig,
    _Batch,
    _build_command,
    _classify_batch,
    _contains_addopts_override,
    _contains_parallel_pytest_option,
    _contains_positional_pytest_argument,
    _expand_targets,
    _file_size_limit_evidence,
    _finalize_batch,
    _make_batches,
    _pytest_arguments,
)


def _test_tree(tmp_path: Path) -> Path:
    """Create a small pytest-like tree without importing any test modules."""
    root = tmp_path / "repo"
    tests = root / "tests"
    tests.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n", encoding="utf-8"
    )
    for name in ("test_alpha.py", "beta_test.py", "test_skip.py", "helper.py"):
        (tests / name).write_text("# fixture\n", encoding="utf-8")
    nested = tests / "nested"
    nested.mkdir()
    (nested / "test_nested.py").write_text("# fixture\n", encoding="utf-8")
    return root


def test_expand_targets_discovers_pytest_file_patterns_and_excludes(
    tmp_path: Path,
) -> None:
    root = _test_tree(tmp_path)
    config = SafeTestConfig(
        repo_root=root,
        test_targets="tests",
        exclude="*skip*",
    )

    assert _expand_targets(config, root) == [
        "tests/beta_test.py",
        "tests/nested/test_nested.py",
        "tests/test_alpha.py",
    ]


def test_expand_targets_supports_relative_globs_and_deduplicates(
    tmp_path: Path,
) -> None:
    root = _test_tree(tmp_path)
    config = SafeTestConfig(
        repo_root=root,
        test_targets="tests/test_*.py, tests/test_alpha.py",
    )

    assert _expand_targets(config, root) == [
        "tests/test_alpha.py",
        "tests/test_skip.py",
    ]


@pytest.mark.parametrize(
    "target", ["tests::test_anything", "tests/test_*.py::test_anything"]
)
def test_expand_targets_rejects_node_selector_on_multiple_files(
    tmp_path: Path, target: str
) -> None:
    root = _test_tree(tmp_path)
    with pytest.raises(ValueError, match="must name one file"):
        _expand_targets(SafeTestConfig(repo_root=root, test_targets=target), root)


def test_make_batches_is_deterministic_and_supports_all_mode() -> None:
    targets = ["tests/a.py", "tests/b.py", "tests/c.py", "tests/d.py", "tests/e.py"]
    config = SafeTestConfig(batch_size=2)

    batches = _make_batches(targets, config)

    assert batches == [
        _Batch(index=1, targets=("tests/a.py", "tests/b.py")),
        _Batch(index=2, targets=("tests/c.py", "tests/d.py")),
        _Batch(index=3, targets=("tests/e.py",)),
    ]
    assert _make_batches(
        targets,
        config.model_copy(update={"batch_mode": "all"}),
    ) == [_Batch(index=1, targets=tuple(targets))]


@pytest.mark.parametrize(
    "argument",
    [
        "-n",
        "-n2",
        "--numprocesses",
        "--numprocesses=2",
        "--dist=loadfile",
        "--workers=2",
        "-d",
        "--tx=popen//python=python3",
        "--looponfail",
    ],
)
def test_parallel_pytest_options_are_detected(argument: str) -> None:
    assert _contains_parallel_pytest_option([argument]) is True


def test_pytest_arguments_reject_parallel_execution() -> None:
    config = SafeTestConfig(pytest_args="-q -n 2")

    with pytest.raises(ValueError, match="Parallel pytest options"):
        _pytest_arguments(config, Path("/tmp/batch.xml"))


@pytest.mark.parametrize(
    "override",
    [
        "-o addopts=-n2",
        "-o 'addopts=-n 2'",
        "-oaddopts=-n2",
        "-o=addopts=-n2",
        "--override-ini=addopts=-n2",
        "--override-ini addopts=-n2",
    ],
)
def test_addopts_override_is_detected_before_internal_reset(override: str) -> None:
    """A later -o addopts=... would otherwise override our serial reset."""
    config = SafeTestConfig(pytest_args=override)

    assert _contains_addopts_override(shlex.split(override)) is True
    with pytest.raises(ValueError, match="addopts"):
        _pytest_arguments(config, Path("batch.xml"))


def test_empty_addopts_override_is_also_rejected() -> None:
    assert _contains_addopts_override(["-o", "addopts="]) is True


@pytest.mark.parametrize(
    "environment_value",
    [
        "-o addopts=-n2",
        "-o=addopts=-n2",
        "--override-ini=addopts=-n2",
        "--override-ini addopts=-n2",
    ],
)
def test_pytest_arguments_reject_addopts_override_from_environment(
    environment_value: str,
) -> None:
    with pytest.raises(ValueError, match="addopts"):
        _pytest_arguments(
            SafeTestConfig(pytest_args="-q"),
            Path("batch.xml"),
            environment={"PYTEST_ADDOPTS": environment_value},
        )


def test_pytest_arguments_resets_repository_addopts_before_custom_config() -> None:
    """The generated override must precede a caller-selected ini file."""
    arguments = _pytest_arguments(
        SafeTestConfig(pytest_args="--config-file=custom.ini -q"),
        Path("batch.xml"),
    )

    assert arguments[:2] == ["-o", "addopts="]
    assert arguments.index("--config-file=custom.ini") > arguments.index("addopts=")


@pytest.mark.parametrize("args", [["tests"], ["--", "tests"]])
def test_positional_pytest_targets_are_detected(args: list[str]) -> None:
    assert _contains_positional_pytest_argument(args) == "tests"


def test_trailing_pytest_separator_is_allowed() -> None:
    arguments = _pytest_arguments(
        SafeTestConfig(pytest_args="-q --"),
        Path("batch.xml"),
    )

    assert arguments == ["-o", "addopts=", "-q", "--junitxml=batch.xml", "--"]


def test_pytest_arguments_reject_positional_target_injection() -> None:
    with pytest.raises(ValueError, match="Positional pytest argument"):
        _pytest_arguments(SafeTestConfig(pytest_args="-q tests"), Path("batch.xml"))


def test_pytest_arguments_allow_common_separate_option_values() -> None:
    arguments = _pytest_arguments(
        SafeTestConfig(pytest_args="-k replay --maxfail 1"),
        Path("batch.xml"),
    )

    assert arguments == [
        "-o",
        "addopts=",
        "-k",
        "replay",
        "--maxfail",
        "1",
        "--junitxml=batch.xml",
    ]


def test_pytest_arguments_inject_batch_junit_path() -> None:
    junit_path = Path("/tmp/aao-test-run/batch.xml")
    arguments = _pytest_arguments(
        SafeTestConfig(pytest_args="-q -k replay"), junit_path
    )

    assert arguments == [
        "-o",
        "addopts=",
        "-q",
        "-k",
        "replay",
        f"--junitxml={junit_path}",
    ]


@pytest.mark.parametrize("custom", ["--junitxml=other.xml", "--junitxml other.xml"])
def test_pytest_arguments_reject_custom_junit_path(custom: str) -> None:
    with pytest.raises(ValueError, match="custom JUnit"):
        _pytest_arguments(SafeTestConfig(pytest_args=custom), Path("batch.xml"))


def test_pytest_arguments_preserve_safe_environment_options() -> None:
    arguments = _pytest_arguments(
        SafeTestConfig(pytest_args="-q"),
        Path("batch.xml"),
        environment={"PYTEST_ADDOPTS": "--maxfail=1"},
    )

    assert arguments == [
        "-o",
        "addopts=",
        "--maxfail=1",
        "-q",
        "--junitxml=batch.xml",
    ]


def test_pytest_arguments_reject_parallel_environment_options() -> None:
    with pytest.raises(ValueError, match="PYTEST_ADDOPTS"):
        _pytest_arguments(
            SafeTestConfig(pytest_args="-q"),
            Path("batch.xml"),
            environment={"PYTEST_ADDOPTS": "-n 2"},
        )


def test_pytest_arguments_reject_short_attached_addopts_override() -> None:
    with pytest.raises(ValueError, match="addopts"):
        _pytest_arguments(
            SafeTestConfig(pytest_args="-o=addopts=-n2"),
            Path("batch.xml"),
        )


def test_build_systemd_command_contains_limits_and_is_argument_vector() -> None:
    config = SafeTestConfig(
        python_executable=Path("/usr/bin/python3"),
        pytest_args="-q",
        cpu_set="2-3",
        cpu_quota_percent=75,
        memory_high_mb=128,
        memory_max_mb=256,
        memory_swap_max_mb=0,
        tasks_max=8,
        timeout_minutes=2,
        kill_after_seconds=7,
    )
    batch = _Batch(index=1, targets=("tests/test_alpha.py",))

    command = _build_command(
        config,
        "systemd",
        batch,
        Path("/tmp/aao-test-run/junit.xml"),
        "aao-test-example-001",
    )

    assert Path(command[0]).name == "systemd-run"
    assert command[1:5] == ["--user", "--scope", "--quiet", "--unit"]
    assert "CPUQuota=75%" in command
    assert "MemoryHigh=128M" in command
    assert "MemoryMax=256M" in command
    assert "TasksMax=8" in command
    assert "LimitFSIZE=256M" not in command
    assert "--fsize=268435456" in command
    assert "RuntimeMaxSec=137s" in command
    assert "KillMode=control-group" in command
    assert "--foreground" in command
    assert "-c" in command
    assert "2-3" in command
    assert "tests/test_alpha.py" in command
    assert all(isinstance(argument, str) for argument in command)


def test_build_command_uses_explicit_regular_file_limit() -> None:
    config = SafeTestConfig(
        python_executable=Path("/usr/bin/python3"),
        max_file_size_mb=17,
        pytest_args="-q",
        memory_high_mb=128,
        memory_max_mb=256,
    )
    batch = _Batch(index=1, targets=("tests/test_alpha.py",))

    systemd = _build_command(
        config,
        "systemd",
        batch,
        Path("/tmp/aao-test-run/junit.xml"),
        "aao-test-example-001",
    )
    prlimit = _build_command(
        config,
        "prlimit",
        batch,
        Path("/tmp/aao-test-run/junit.xml"),
        "aao-test-example-001",
    )

    assert "LimitFSIZE=17M" not in systemd
    assert "--fsize=17825792" in systemd
    assert "--fsize=17825792" in prlimit


def test_build_prlimit_command_keeps_virtual_memory_fallback_explicit() -> None:
    config = SafeTestConfig(
        python_executable=Path("/usr/bin/python3"),
        memory_high_mb=128,
        pytest_args="-q",
        memory_max_mb=256,
        timeout_minutes=1,
        kill_after_seconds=5,
    )
    command = _build_command(
        config,
        "prlimit",
        _Batch(index=1, targets=("tests/test_alpha.py",)),
        Path("/tmp/batch.xml"),
        "aao-test-example-001",
    )

    assert Path(command[0]).name == "prlimit"
    assert command[1] == "--as=268435456"
    assert "--fsize=268435456" in command
    assert "--cpu=65" in command
    assert "systemd-run" not in command


@pytest.mark.parametrize(
    ("returncode", "properties", "journal", "expected_status"),
    [
        (0, {}, "", "PASSED"),
        (1, {}, "", "TEST_FAILURE"),
        (124, {}, "", "TIMEOUT"),
        (143, {}, "", "TEST_FAILURE"),
        (137, {}, "", "RESOURCE_KILL"),
        (153, {}, "", "FILE_SIZE_LIMIT"),
        (1, {}, "scope: killed by the OOM killer", "OOM"),
        (1, {"Result": "timeout"}, "", "TIMEOUT"),
    ],
)
def test_classify_batch_uses_resource_evidence(
    returncode: int,
    properties: dict[str, str],
    journal: str,
    expected_status: str,
) -> None:
    status, _reason = _classify_batch(returncode, False, properties, journal)

    assert status == expected_status


def test_classify_batch_prioritizes_user_interrupt() -> None:
    assert _classify_batch(137, True, {"Result": "oom-kill"}, "") == (
        "INTERRUPTED",
        "user interrupt",
    )


def test_classify_batch_reports_parent_wait_timeout() -> None:
    assert _classify_batch(143, False, {}, "", timeout_expired=True) == (
        "TIMEOUT",
        "batch wrapper exceeded the parent wait limit",
    )


def test_file_size_limit_evidence_reads_only_diagnostic_tail(tmp_path: Path) -> None:
    log_path = tmp_path / "batch.log"
    log_path.write_text(
        "normal output\nOSError: [Errno 27] File too large\n", encoding="utf-8"
    )

    assert _file_size_limit_evidence(log_path) is True

    log_path.write_text("test output: file too large\n", encoding="utf-8")
    assert _file_size_limit_evidence(log_path) is False


def test_classify_batch_uses_file_size_diagnostic() -> None:
    assert _classify_batch(1, False, {}, "", file_size_limit_hit=True) == (
        "FILE_SIZE_LIMIT",
        "a batch file exceeded --max-file-size-mb",
    )


def test_cleanup_failure_is_terminal_even_after_test_failure(tmp_path: Path) -> None:
    output = tmp_path / "run"
    (output / "logs").mkdir(parents=True)
    (output / "junit").mkdir()
    log_path = output / "logs" / "batch-001.log"
    junit_path = output / "junit" / "batch-001.xml"
    manifest = {
        "batches": [{"index": 1, "status": "RUNNING"}],
    }

    result = _finalize_batch(
        batch=_Batch(index=1, targets=("tests/test_alpha.py",)),
        output=output,
        manifest=manifest,
        command=("pytest",),
        log_path=log_path,
        junit_path=junit_path,
        started=0.0,
        returncode=1,
        interrupted=False,
        systemd_properties={},
        journal="",
        cleanup_incomplete=True,
    )

    assert result.status == "CLEANUP_FAILURE"
    assert manifest["batches"][0]["status"] == "CLEANUP_FAILURE"
    assert "original status=TEST_FAILURE" in result.reason
