from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import auto_atom.runtime as runtime
from auto_atom.framework import TaskFileConfig
from auto_atom.ipc.service import _default_action_applier as ipc_action_applier
from auto_atom.mock import MockEnv, build_mock_backend
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runner.policy_eval import _default_action_applier as cli_action_applier
from auto_atom.runtime import (
    ComponentRegistry,
    EnvProtocol,
    InfoEnvProtocol,
    JointActionEnvProtocol,
    KinematicPoseActionEnvProtocol,
    ObservationEnvProtocol,
    PoseActionEnvProtocol,
    SimulationLoopEnvProtocol,
    StepEnvProtocol,
    TaskRunner,
    require_env_capability,
)


class _BareEnv:
    def __init__(self, batch_size: int = 2) -> None:
        self.batch_size = batch_size


class _Backend:
    def __init__(self, env: object, batch_size: int = 2) -> None:
        self._env = env
        self.batch_size = batch_size

    def get_env(self) -> object:
        return self._env


class _PoseEnv(_BareEnv):
    def __init__(self, batch_size: int = 2) -> None:
        super().__init__(batch_size)
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def apply_pose_action(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


class _BasicPoseEnv(_BareEnv):
    def apply_pose_action(
        self,
        operator: str,
        position: Any,
        orientation: Any,
        gripper: Any = None,
        *,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pass


class _InfoEnv(_BareEnv):
    def get_info(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size, "kind": "test"}


class _LegacyStepEnv(_BareEnv):
    def step(self, action: np.ndarray) -> None:
        pass


class _KeywordMaskStepEnv(_BareEnv):
    def step(
        self,
        action: np.ndarray,
        *,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pass


class _NoMaskUpdateEnv(_BareEnv):
    def update(self) -> None:
        pass


class _NonCallableStepEnv(_BareEnv):
    step = 42


class _OpaqueStepEnv(_BareEnv):
    step = np.add


class _AsyncStepEnv(_BareEnv):
    async def step(
        self,
        action: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pass


class _AsyncStepCallable:
    async def __call__(
        self,
        action: np.ndarray,
        env_mask: np.ndarray | None = None,
    ) -> None:
        pass


class _AsyncCallableStepEnv(_BareEnv):
    def __init__(self) -> None:
        super().__init__()
        self.step = _AsyncStepCallable()


class _ExplodingBatchSizeEnv:
    @property
    def batch_size(self) -> int:
        raise TypeError("backend is not ready")


def _context(env: object, batch_size: int = 2) -> Any:
    return SimpleNamespace(backend=_Backend(env, batch_size))


def _mismatched_backend(task: Any, operators: Any) -> Any:
    backend = build_mock_backend(task, operators)
    backend.get_env().batch_size = backend.batch_size + 1
    return backend


def _mock_config(
    env_name: str,
    *,
    backend: Any = build_mock_backend,
) -> TaskFileConfig:
    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": 2},
    )
    return TaskFileConfig.model_validate(
        {
            "backend": backend,
            "task": {"env_name": env_name, "stages": []},
            "task_operators": {},
        }
    )


def test_mock_env_exposes_only_its_supported_capabilities() -> None:
    env = MockEnv(batch_size=2)

    assert isinstance(env, EnvProtocol)
    assert isinstance(env, StepEnvProtocol)
    assert isinstance(env, ObservationEnvProtocol)
    assert isinstance(env, JointActionEnvProtocol)
    assert isinstance(env, PoseActionEnvProtocol)
    assert isinstance(env, KinematicPoseActionEnvProtocol)
    assert not isinstance(env, SimulationLoopEnvProtocol)
    assert not isinstance(env, InfoEnvProtocol)


def test_basic_pose_capability_does_not_require_kinematic_application() -> None:
    env = _BasicPoseEnv()

    assert (
        require_env_capability(
            env,
            PoseActionEnvProtocol,
            feature="basic pose action",
        )
        is env
    )
    with pytest.raises(RuntimeError, match=r"KinematicPoseActionEnvProtocol"):
        require_env_capability(
            env,
            KinematicPoseActionEnvProtocol,
            feature="kinematic pose action",
        )


def test_mock_env_direct_actions_accept_capability_keywords() -> None:
    env = MockEnv(batch_size=2)
    mask = np.asarray([True, False], dtype=bool)

    env.apply_joint_action("arm", np.zeros((2, 7)), env_mask=mask, kinematic=True)
    env.apply_pose_action(
        "arm",
        np.zeros((2, 3)),
        np.zeros((2, 4)),
        env_mask=mask,
        kinematic=True,
    )


def test_capability_error_names_protocol_and_missing_member() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"StepEnvProtocol.*Missing attributes: step",
    ):
        require_env_capability(
            _BareEnv(),
            StepEnvProtocol,
            feature="test policy stepping",
        )


def test_capability_error_rejects_incompatible_method_signature() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"StepEnvProtocol.step.*got _LegacyStepEnv.step\(action",
    ):
        require_env_capability(
            _LegacyStepEnv(),
            StepEnvProtocol,
            feature="test policy stepping",
        )


def test_capability_signature_accepts_the_framework_call_shape() -> None:
    assert (
        require_env_capability(
            _KeywordMaskStepEnv(),
            StepEnvProtocol,
            feature="test policy stepping",
        ).batch_size
        == 2
    )


def test_successful_capability_signature_validation_is_cached(monkeypatch) -> None:
    env = _KeywordMaskStepEnv()
    calls = 0
    original = runtime._validate_environment_protocol_signatures

    def count_validation(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        original(*args, **kwargs)

    monkeypatch.setattr(
        runtime,
        "_validate_environment_protocol_signatures",
        count_validation,
    )

    require_env_capability(env, StepEnvProtocol, feature="first use")
    require_env_capability(env, StepEnvProtocol, feature="hot-path reuse")

    assert calls == 1
    assert (
        require_env_capability(
            _NoMaskUpdateEnv(),
            SimulationLoopEnvProtocol,
            feature="test simulation loop",
        ).batch_size
        == 2
    )


def test_capability_error_rejects_non_callable_method() -> None:
    with pytest.raises(RuntimeError, match=r"requires callable StepEnvProtocol.step"):
        require_env_capability(
            _NonCallableStepEnv(),
            StepEnvProtocol,
            feature="test policy stepping",
        )


def test_capability_error_rejects_non_introspectable_method() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"requires introspectable StepEnvProtocol.step",
    ):
        require_env_capability(
            _OpaqueStepEnv(),
            StepEnvProtocol,
            feature="test policy stepping",
        )


@pytest.mark.parametrize("env", [_AsyncStepEnv(), _AsyncCallableStepEnv()])
def test_capability_error_rejects_async_method(env: object) -> None:
    with pytest.raises(
        RuntimeError, match=r"requires synchronous StepEnvProtocol.step"
    ):
        require_env_capability(
            env,
            StepEnvProtocol,
            feature="test policy stepping",
        )


@pytest.mark.parametrize("batch_size", [True, 1.0, 0, -1])
def test_capability_error_rejects_invalid_environment_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(RuntimeError, match=r"environment batch_size"):
        require_env_capability(
            _BareEnv(batch_size),
            EnvProtocol,
            feature="test core environment",
        )


@pytest.mark.parametrize("batch_size", [True, 1.0, 0, -1])
def test_capability_error_rejects_invalid_backend_batch_size(
    batch_size: object,
) -> None:
    with pytest.raises(RuntimeError, match=r"backend batch_size"):
        require_env_capability(
            _BareEnv(1),
            EnvProtocol,
            feature="test core environment",
            expected_batch_size=batch_size,
        )


def test_capability_error_preserves_batch_size_property_failure() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"readable environment batch_size.*TypeError: backend is not ready",
    ):
        require_env_capability(
            _ExplodingBatchSizeEnv(),
            EnvProtocol,
            feature="test core environment",
        )


@pytest.mark.parametrize(
    "subject_factory",
    [
        pytest.param(TaskRunner, id="task-runner"),
        pytest.param(
            lambda: PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None),
            id="policy-evaluator",
        ),
    ],
)
def test_runner_initialization_rejects_backend_env_batch_mismatch(
    subject_factory: Any,
) -> None:
    ComponentRegistry.clear()
    subject = subject_factory()
    try:
        with pytest.raises(
            RuntimeError,
            match=r"environment batch_size to match backend.batch_size; got 3 and 2",
        ):
            subject.from_config(
                _mock_config(
                    "capability_batch_mismatch",
                    backend=_mismatched_backend,
                )
            )
    finally:
        subject.close()
        ComponentRegistry.clear()


@pytest.mark.parametrize(
    "subject_factory",
    [
        pytest.param(TaskRunner, id="task-runner"),
        pytest.param(
            lambda: PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None),
            id="policy-evaluator",
        ),
    ],
)
def test_initialization_failure_tears_down_constructed_backend(
    subject_factory: Any,
) -> None:
    ComponentRegistry.clear()
    created = []

    def mismatched_backend(task: Any, operators: Any) -> Any:
        backend = build_mock_backend(task, operators)
        backend.get_env().batch_size = backend.batch_size + 1
        created.append(backend)
        return backend

    subject = subject_factory()
    try:
        with pytest.raises(RuntimeError, match="batch_size to match"):
            subject.from_config(
                _mock_config(
                    "capability_teardown_on_failure",
                    backend=mismatched_backend,
                )
            )
        assert created[0].lifecycle_events == ["teardown()"]
    finally:
        subject.close()
        ComponentRegistry.clear()


def test_policy_evaluator_observation_requires_observation_capability() -> None:
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)
    evaluator._context = _context(_BareEnv())

    with pytest.raises(
        RuntimeError,
        match=r"ObservationEnvProtocol.*capture_observation",
    ):
        evaluator.get_observation()


def test_policy_evaluator_info_requires_info_capability() -> None:
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)
    evaluator._context = _context(_BareEnv())

    with pytest.raises(RuntimeError, match=r"InfoEnvProtocol.*get_info"):
        evaluator.get_info()

    evaluator._context = _context(_InfoEnv())
    assert evaluator.get_info() == {"batch_size": 2, "kind": "test"}


def test_policy_evaluator_sim_loop_requires_update_capability() -> None:
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)
    evaluator._context = _context(_BareEnv())

    with pytest.raises(RuntimeError, match=r"SimulationLoopEnvProtocol.*update"):
        evaluator.start_sim_loop()
    assert not evaluator.sim_loop_running


def test_policy_evaluator_sim_loop_runs_and_stops() -> None:
    class UpdatingEnv(_BareEnv):
        def __init__(self) -> None:
            super().__init__()
            self.updated = threading.Event()

        def update(self) -> None:
            self.updated.set()

    env = UpdatingEnv()
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)
    evaluator._context = _context(env)

    evaluator.start_sim_loop(1000.0)
    assert env.updated.wait(1.0)
    evaluator.stop_sim_loop()

    assert not evaluator.sim_loop_running


def test_policy_evaluator_surfaces_sim_loop_failure_and_can_restart() -> None:
    class FailingEnv(_BareEnv):
        def __init__(self) -> None:
            super().__init__()
            self.updated = threading.Event()
            self.fail = True

        def update(self) -> None:
            self.updated.set()
            if self.fail:
                raise ValueError("physics diverged")

    env = FailingEnv()
    evaluator = PolicyEvaluator(
        action_applier=lambda *_args, **_kwargs: None,
        observation_getter=lambda _context: {},
    )
    evaluator._context = _context(env)

    evaluator.start_sim_loop(1000.0)
    assert env.updated.wait(1.0)
    while evaluator.sim_loop_running:
        threading.Event().wait(0.001)

    with pytest.raises(RuntimeError, match="Background simulation loop failed") as exc:
        evaluator.get_observation()
    assert isinstance(exc.value.__cause__, ValueError)
    assert "physics diverged" in str(exc.value.__cause__)

    env.fail = False
    env.updated.clear()
    evaluator.start_sim_loop(1000.0)
    assert env.updated.wait(1.0)
    evaluator.stop_sim_loop()

    env.fail = True
    env.updated.clear()
    evaluator.start_sim_loop(1000.0)
    assert env.updated.wait(1.0)
    while evaluator.sim_loop_running:
        threading.Event().wait(0.001)
    with pytest.raises(RuntimeError, match="Background simulation loop failed"):
        evaluator.summarize()


def test_requested_sim_loop_capability_is_checked_during_initialization() -> None:
    ComponentRegistry.clear()
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)
    try:
        with pytest.raises(
            RuntimeError,
            match=r"PolicyEvaluator background simulation.*SimulationLoopEnvProtocol",
        ):
            evaluator.from_config(
                _mock_config("capability_missing_sim_loop"),
                sim_loop_frequency=60,
            )
    finally:
        evaluator.close()
        ComponentRegistry.clear()


@pytest.mark.parametrize("frequency", [-1.0, np.inf, np.nan])
def test_policy_evaluator_rejects_invalid_pending_sim_loop_frequency(
    frequency: float,
) -> None:
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="sim_loop_frequency must be non-negative"):
        evaluator.from_config(
            TaskFileConfig.model_validate(
                {
                    "backend": build_mock_backend,
                    "task": {"env_name": "unused", "stages": []},
                }
            ),
            sim_loop_frequency=frequency,
        )


@pytest.mark.parametrize("frequency", [0.0, -1.0, np.inf, np.nan])
def test_policy_evaluator_rejects_invalid_sim_loop_frequency(
    frequency: float,
) -> None:
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)
    evaluator._context = _context(_BareEnv())

    with pytest.raises(ValueError, match="frequency must be positive and finite"):
        evaluator.start_sim_loop(frequency)


def test_cli_default_action_applier_requires_step_capability() -> None:
    with pytest.raises(RuntimeError, match=r"StepEnvProtocol.*step"):
        cli_action_applier(_context(_BareEnv()), np.zeros((2, 4)))


def test_ipc_default_action_applier_requires_pose_capability() -> None:
    with pytest.raises(RuntimeError, match=r"PoseActionEnvProtocol.*apply_pose_action"):
        ipc_action_applier(
            _context(_BareEnv()),
            {"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "gripper": 0},
        )


def test_ipc_default_action_applier_forwards_env_mask() -> None:
    env = _PoseEnv()
    mask = np.asarray([True, False], dtype=bool)

    ipc_action_applier(
        _context(env),
        {"position": [0, 0, 0], "orientation": [0, 0, 0, 1], "gripper": 0},
        mask,
    )

    assert len(env.calls) == 1
    args, kwargs = env.calls[0]
    assert args == ("arm", [0, 0, 0], [0, 0, 0, 1], 0)
    assert kwargs.keys() == {"env_mask"}
    np.testing.assert_array_equal(kwargs["env_mask"], mask)
