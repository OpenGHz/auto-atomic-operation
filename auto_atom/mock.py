"""Mock runtime components used for development and examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .framework import (
    AutoAtomConfig,
    EefControlConfig,
    OperatorConfig,
    PoseControlConfig,
)
from .runtime import (
    ComponentRegistry,
    ControlResult,
    ControlSignal,
    ObjectHandler,
    OperatorHandler,
    PoseState,
    SceneBackend,
)


@dataclass
class MockObjectHandler(ObjectHandler):
    kind: str = "mock_object"
    pose: PoseState = field(default_factory=PoseState)

    def get_pose(self) -> PoseState:
        return self.pose


@dataclass
class MockOperatorHandler(OperatorHandler):
    operator_name: str
    batch_size: int = 1
    role: str = "generic"
    _command_key: List[str] = field(default_factory=list)
    _progress: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.int64))
    base_pose: PoseState = field(default_factory=PoseState)
    end_effector_pose: PoseState = field(default_factory=PoseState)

    def __post_init__(self) -> None:
        self._command_key = [""] * self.batch_size
        self._progress = np.zeros(self.batch_size, dtype=np.int64)
        self.base_pose = self.base_pose.broadcast_to(self.batch_size)
        self.end_effector_pose = self.end_effector_pose.broadcast_to(self.batch_size)

    @property
    def name(self) -> str:
        return self.operator_name

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        mask = self._normalize_mask(env_mask)
        details = [{} for _ in range(self.batch_size)]
        signals = np.asarray([ControlSignal.RUNNING] * self.batch_size, dtype=object)
        for env_index, enabled in enumerate(mask):
            if not enabled:
                continue
            command_key = (
                f"pose:{_serialize_param(pose)}:{target.name if target else ''}"
            )
            self._prepare_command(env_index, command_key)
            self._progress[env_index] += 1
            if self._progress[env_index] == 1:
                details[env_index] = {
                    "event": "moving",
                    "operator": self.name,
                    "role": self.role,
                }
                continue
            self.end_effector_pose.position[env_index] = np.asarray(
                pose.position or self.end_effector_pose.position[env_index],
                dtype=np.float64,
            )
            self.end_effector_pose.orientation[env_index] = np.asarray(
                pose.orientation or self.end_effector_pose.orientation[env_index],
                dtype=np.float64,
            )
            signals[env_index] = ControlSignal.REACHED
            details[env_index] = {
                "event": "pose_reached",
                "operator": self.name,
            }
        return ControlResult(signals=signals, details=details)

    def control_eef(
        self,
        eef: EefControlConfig,
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        mask = self._normalize_mask(env_mask)
        details = [{} for _ in range(self.batch_size)]
        signals = np.asarray([ControlSignal.RUNNING] * self.batch_size, dtype=object)
        for env_index, enabled in enumerate(mask):
            if not enabled:
                continue
            command_key = f"eef:{eef.close}:{eef.joint_positions}"
            self._prepare_command(env_index, command_key)
            self._progress[env_index] += 1
            if self._progress[env_index] == 1:
                details[env_index] = {"event": "eef_moving", "operator": self.name}
                continue
            signals[env_index] = ControlSignal.REACHED
            details[env_index] = {"event": "eef_reached", "operator": self.name}
        return ControlResult(signals=signals, details=details)

    def get_end_effector_pose(self) -> PoseState:
        return self.end_effector_pose

    def get_base_pose(self) -> PoseState:
        return self.base_pose

    def _prepare_command(self, env_index: int, command_key: str) -> None:
        if self._command_key[env_index] != command_key:
            self._command_key[env_index] = command_key
            self._progress[env_index] = 0

    def _normalize_mask(self, env_mask: Optional[np.ndarray]) -> np.ndarray:
        if env_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        return np.asarray(env_mask, dtype=bool).reshape(-1)


@dataclass
class MockEnv:
    """Minimal env stub that satisfies the ``SceneBackend.env`` contract."""

    batch_size: int = 1

    def step(self, action: np.ndarray, env_mask: np.ndarray | None = None) -> None:
        pass

    def capture_observation(self) -> Dict[str, Dict[str, Any]]:
        return {}

    def apply_joint_action(
        self, operator: str, action: Any = None, env_mask: Any = None
    ) -> None:
        pass

    def apply_pose_action(
        self,
        operator: str,
        position: Any = None,
        orientation: Any = None,
        gripper: Any = None,
        env_mask: Any = None,
    ) -> None:
        pass


@dataclass
class MockSceneBackend(SceneBackend):
    env_name: str
    batch_size: int = 1
    operators: Dict[str, MockOperatorHandler] = field(default_factory=dict)
    objects: Dict[str, MockObjectHandler] = field(default_factory=dict)
    lifecycle_events: List[str] = field(default_factory=list)
    interest_updates: List[Dict[str, List[str]]] = field(default_factory=list)
    env: MockEnv = field(init=False)

    def __post_init__(self) -> None:
        self.env = MockEnv(batch_size=self.batch_size)

    def setup(self, config: AutoAtomConfig) -> None:
        self.lifecycle_events.append(
            f"setup(env={config.env_name}, seed={config.seed})"
        )

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        self.lifecycle_events.append("reset()")
        mask = (
            np.ones(self.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        for operator in self.operators.values():
            operator._progress[mask] = 0
            for env_index, enabled in enumerate(mask):
                if enabled:
                    operator._command_key[env_index] = ""

    def teardown(self) -> None:
        self.lifecycle_events.append("teardown()")

    def get_operator_handler(self, name: str) -> MockOperatorHandler:
        try:
            return self.operators[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.operators)) or "<empty>"
            raise KeyError(
                f"Unknown operator '{name}'. Known operators: {known}"
            ) from exc

    def get_object_handler(self, name: str) -> Optional[MockObjectHandler]:
        if not name:
            return None
        try:
            return self.objects[name]
        except KeyError as exc:
            known = ", ".join(sorted(self.objects)) or "<empty>"
            raise KeyError(f"Unknown object '{name}'. Known objects: {known}") from exc

    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        _ = self.get_operator_handler(operator_name)
        _ = self.get_object_handler(object_name)
        return np.zeros(self.batch_size, dtype=bool)

    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        _ = self.get_operator_handler(operator_name)
        return np.zeros(self.batch_size, dtype=bool)

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        self.interest_updates.append(
            {
                "objects": list(object_names),
                "operations": list(operation_names),
            }
        )


def create_mock_env(
    kind: str = "mock_env",
    batch_size: int = 1,
    name: str = "",
) -> Dict[str, Any]:
    env = {"kind": kind, "batch_size": batch_size}
    if name:
        ComponentRegistry.register_env(name, env)
    return env


def build_mock_backend(
    task: AutoAtomConfig | Dict[str, Any],
    operators: Dict[str, OperatorConfig],
) -> MockSceneBackend:
    config = (
        task
        if isinstance(task, AutoAtomConfig)
        else AutoAtomConfig.model_validate(task)
    )
    operator_configs = list(operators.values())
    env_payload = ComponentRegistry.get_env(config.env_name)
    batch_size = (
        int(env_payload.get("batch_size", 1)) if isinstance(env_payload, dict) else 1
    )
    operators_map = {
        operator.name: MockOperatorHandler(
            operator_name=operator.name,
            batch_size=batch_size,
            role=operator.model_extra.get("role", "generic")
            if operator.model_extra
            else "generic",
            base_pose=PoseState(
                position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0, 1.0)
            ),
            end_effector_pose=PoseState(
                position=(0.2, 0.0, 0.3),
                orientation=(0.0, 0.0, 0.0, 1.0),
            ),
        )
        for operator in operator_configs
    }
    object_names = sorted({stage.object for stage in config.stages if stage.object})
    objects = {
        object_name: MockObjectHandler(
            name=object_name,
            pose=PoseState(
                position=np.repeat(
                    np.asarray(
                        [[0.4 + 0.1 * index, -0.1 + 0.05 * index, 0.05 * (index + 1)]],
                        dtype=np.float64,
                    ),
                    batch_size,
                    axis=0,
                ),
                orientation=np.repeat(
                    np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
                    batch_size,
                    axis=0,
                ),
            ),
        )
        for index, object_name in enumerate(object_names)
    }
    return MockSceneBackend(
        env_name=config.env_name,
        batch_size=batch_size,
        operators=operators_map,
        objects=objects,
    )


def _serialize_param(param: Any) -> Any:
    if hasattr(param, "model_dump"):
        return param.model_dump(mode="json")
    return param
