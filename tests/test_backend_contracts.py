from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np

from auto_atom.framework import (
    AutoAtomConfig,
    EefControlConfig,
    PoseControlConfig,
    TaskFileConfig,
)
from auto_atom.backend.mjc.mujoco_backend import MujocoObjectHandler
from auto_atom.runtime import (
    ControlResult,
    ControlSignal,
    EnvProtocol,
    ObjectHandler,
    OperatorHandler,
    PoseState,
    SceneBackend,
    TaskFlowBuilder,
    TaskRunner,
)


class _ExternalEnv:
    batch_size = 1


class _ExternalObject(ObjectHandler):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pose = PoseState()

    def get_pose(self) -> PoseState:
        return self.pose

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        _ = env_mask
        self.pose = pose


class _ExternalOperator(OperatorHandler):
    @property
    def name(self) -> str:
        return "arm"

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        _ = pose, target, env_mask
        return ControlResult.filled(1, ControlSignal.REACHED)

    def control_eef(
        self,
        eef: EefControlConfig,
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        _ = eef, env_mask
        return ControlResult.filled(1, ControlSignal.REACHED)

    def get_end_effector_pose(self) -> PoseState:
        return PoseState()

    def get_base_pose(self) -> PoseState:
        return PoseState()


class _ExternalBackend(SceneBackend):
    def __init__(self) -> None:
        self._env = _ExternalEnv()
        self._operator = _ExternalOperator()
        self._objects = {"block": _ExternalObject("block")}

    def get_env(self) -> EnvProtocol:
        return self._env

    @property
    def batch_size(self) -> int:
        return self._env.batch_size

    def setup(self, config: AutoAtomConfig) -> None:
        _ = config

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        _ = env_mask

    def teardown(self) -> None:
        pass

    def get_operator_handler(self, name: str) -> OperatorHandler:
        if name != "arm":
            raise KeyError(name)
        return self._operator

    def get_object_handler(self, name: str) -> Optional[ObjectHandler]:
        if not name:
            return None
        return self._objects[name]

    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        _ = operator_name, object_name
        return np.asarray([False], dtype=bool)

    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        _ = operator_name
        return np.asarray([False], dtype=bool)

    def get_grasped_object_name(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[str]:
        _ = operator_name, env_index
        return None

    def is_operator_contacting(
        self,
        operator_name: str,
        object_name: str,
    ) -> np.ndarray:
        _ = operator_name, object_name
        return np.asarray([False], dtype=bool)


def _build_external_backend(task, operators) -> _ExternalBackend:
    _ = task, operators
    return _ExternalBackend()


def test_minimal_external_backend_satisfies_runner_contract() -> None:
    config = TaskFileConfig.model_validate(
        {
            "backend": _build_external_backend,
            "task": {"env_name": "external", "stages": []},
            "task_operators": {},
        }
    )
    runner = TaskRunner().from_config(config)
    try:
        assert runner.batch_size == 1
        assert isinstance(runner.get_env(), _ExternalEnv)
        update = runner.reset()
        assert update.done.tolist() == [False]
        update = runner.update()
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        runner.close()


def test_mujoco_object_handler_rejects_wrong_mask_shape() -> None:
    handler = MujocoObjectHandler(
        name="block",
        env=SimpleNamespace(batch_size=2, envs=[]),
        body_name="block",
    )

    with np.testing.assert_raises_regex(ValueError, r"env_mask must have shape \(2,\)"):
        handler.set_pose(PoseState(), env_mask=np.asarray([True]))


def test_object_handler_rejects_empty_identity() -> None:
    with np.testing.assert_raises_regex(ValueError, "non-empty string"):
        _ExternalObject("")


def test_task_flow_rejects_mismatched_operator_identity() -> None:
    backend = _ExternalBackend()
    backend._operator = SimpleNamespace(name="other")

    with np.testing.assert_raises_regex(ValueError, "mismatched identity"):
        TaskFlowBuilder._select_operator(SimpleNamespace(operator="arm"), backend)
