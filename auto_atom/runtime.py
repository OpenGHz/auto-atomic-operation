"""YAML-driven batch-first task runner built from primitive controls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol, runtime_checkable

import math
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .framework import (
    ArcControlConfig,
    AutoAtomConfig,
    EefControlConfig,
    OPERATION_CONDITIONS,
    Operation,
    OperationConditionType,
    OperationConstraint,
    Orientation,
    Position,
    PoseControlConfig,
    PoseReference,
    StageConfig,
    StageControlConfig,
    TaskFileConfig,
)
from .utils.pose import (
    PoseState,
    compose_pose,
    euler_to_quaternion,
    inverse_pose,
    pose_config_to_pose_state,
    quaternion_angular_distance,
    rotate_pose_around_axis,
)


class StageExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ControlSignal(str, Enum):
    RUNNING = "running"
    REACHED = "reached"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass
class ObjectHandler:
    name: str

    def get_pose(self) -> PoseState:
        raise NotImplementedError

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError


@dataclass
class ControlResult:
    signals: np.ndarray
    details: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.signals = np.asarray(self.signals, dtype=object).reshape(-1)
        if not self.details:
            self.details = [{} for _ in range(len(self.signals))]
        if len(self.details) != len(self.signals):
            raise ValueError("details length must match signals length")

    @classmethod
    def filled(
        cls,
        batch_size: int,
        signal: ControlSignal,
        details: Optional[List[Dict[str, Any]]] = None,
    ) -> "ControlResult":
        return cls(
            signals=np.asarray([signal] * batch_size, dtype=object),
            details=details or [{} for _ in range(batch_size)],
        )


@runtime_checkable
class IKSolver(Protocol):
    def solve(
        self,
        target_pose_in_base: PoseState,
        current_qpos: np.ndarray,
    ) -> Optional[np.ndarray]: ...


class OperatorHandler(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name used by stage configs."""

    @abstractmethod
    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance motion toward the desired pose for selected envs."""

    @abstractmethod
    def control_eef(
        self,
        eef: EefControlConfig,
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the end-effector toward the desired state for selected envs."""

    @abstractmethod
    def get_end_effector_pose(self) -> PoseState:
        """Return batched world poses for the operator end-effector."""

    @abstractmethod
    def get_base_pose(self) -> PoseState:
        """Return batched world poses for the operator base."""

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError


class SceneBackend(ABC):
    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of envs in the backend batch."""

    @abstractmethod
    def setup(self, config: AutoAtomConfig) -> None:
        """Prepare backend resources for this task."""

    @abstractmethod
    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        """Reset selected envs for a new run."""

    @abstractmethod
    def teardown(self) -> None:
        """Release backend resources after execution."""

    @abstractmethod
    def get_operator_handler(self, name: str) -> OperatorHandler:
        """Resolve an operator handler by name."""

    @abstractmethod
    def get_object_handler(self, name: str) -> Optional[ObjectHandler]:
        """Resolve an object handler by name. Empty names may return None."""

    @abstractmethod
    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        """Return whether the operator is currently grasping the given object."""

    @abstractmethod
    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        """Return whether the operator is currently grasping any object."""

    def is_object_displaced(
        self,
        object_name: str,
        original_pose: PoseState,
        threshold: float = 0.01,
    ) -> np.ndarray:
        handler = self.get_object_handler(object_name)
        if handler is None:
            return np.zeros(self.batch_size, dtype=bool)
        current = handler.get_pose()
        if original_pose.batch_size != self.batch_size:
            original_pose = original_pose.broadcast_to(self.batch_size)
        delta = np.linalg.norm(
            np.asarray(current.position, dtype=np.float64)
            - np.asarray(original_pose.position, dtype=np.float64),
            axis=1,
        )
        return delta > threshold

    def is_operator_contacting(
        self,
        operator_name: str,  # noqa: ARG002
        object_name: str,  # noqa: ARG002
    ) -> np.ndarray:
        return np.zeros(self.batch_size, dtype=bool)

    def get_element_pose(self, name: str, env_index: int = 0) -> PoseState:  # noqa: ARG002
        raise NotImplementedError(
            f"Backend does not support named element lookup (requested '{name}')."
        )

    def get_joint_angle(self, name: str, env_index: int = 0) -> float:  # noqa: ARG002
        raise NotImplementedError(
            f"Backend does not support joint angle lookup (requested '{name}')."
        )

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        """Notify the backend about the current task-focus objects and operations."""


@dataclass
class ArcExecutionSnapshot:
    start_eef_pose: Optional[PoseState] = None
    pivot_world_pos: Optional[Position] = None


@dataclass
class PrimitiveAction:
    kind: str
    pose: Optional[PoseControlConfig] = None
    eef: Optional[EefControlConfig] = None
    resolved_pose: Optional[PoseControlConfig] = None
    arc_snapshot: Optional[ArcExecutionSnapshot] = None
    arc_cumulative_angle: Optional[float] = None


@dataclass
class ExecutionRecord:
    env_index: int
    stage_index: int
    stage_name: str
    operator: str
    operation: str
    target_object: str
    blocking: bool
    status: StageExecutionStatus
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionContext:
    config: AutoAtomConfig
    backend: SceneBackend
    task_file: TaskFileConfig
    plan: List["StageExecutionPlan"] = field(default_factory=list)


@dataclass
class StageExecutionPlan:
    stage_index: int
    stage: StageConfig
    operator_name: str
    last_orientation_before: Optional[Orientation] = None

    @property
    def stage_name(self) -> str:
        return self.stage.name or f"stage_{self.stage_index}"


@dataclass
class TaskUpdate:
    stage_index: Optional[np.ndarray]
    stage_name: List[str]
    status: np.ndarray
    done: np.ndarray
    success: np.ndarray
    details: List[Dict[str, Any]] = field(default_factory=list)
    phase: List[Optional[str]] = field(default_factory=list)
    phase_step: Optional[np.ndarray] = None


@dataclass
class ExecutionSummary:
    total_stages: int
    max_updates: Optional[int]
    updates_used: int
    completed_stage_count: np.ndarray
    final_stage_index: np.ndarray
    final_stage_name: List[str]
    final_status: np.ndarray
    final_done: np.ndarray
    final_success: np.ndarray
    records: List[ExecutionRecord] = field(default_factory=list)


@dataclass
class ActiveStageState:
    plan: StageExecutionPlan
    operator: OperatorHandler
    target: Optional[ObjectHandler]
    actions: List[PrimitiveAction]
    action_index: int = 0
    initial_object_pose: Optional[PoseState] = None


@dataclass
class _EnvRuntimeState:
    stage_cursor: int = 0
    active: Optional[ActiveStageState] = None
    done: bool = False
    success: Optional[bool] = None
    phase: Optional[str] = None
    phase_step: Optional[int] = None
    latest_status: StageExecutionStatus = StageExecutionStatus.PENDING
    latest_details: Dict[str, Any] = field(default_factory=dict)


class ComponentRegistry:
    _env_instances: ClassVar[Dict[str, Any]] = {}

    @classmethod
    def register_env(cls, name: str, env: Any) -> Any:
        cls._env_instances[name] = env
        return env

    @classmethod
    def get_env(cls, name: str) -> Any:
        try:
            return cls._env_instances[name]
        except KeyError as exc:
            known = ", ".join(sorted(cls._env_instances)) or "<empty>"
            raise KeyError(
                f"Environment '{name}' is not registered. Available environments: {known}"
            ) from exc

    @classmethod
    def has_env(cls, name: str) -> bool:
        return name in cls._env_instances

    @classmethod
    def clear(cls) -> None:
        cls._env_instances.clear()


class TaskFlowBuilder:
    """Build stage plans and primitive action lists from validated config."""

    def build(self, context: ExecutionContext) -> List[StageExecutionPlan]:
        plans: List[StageExecutionPlan] = []
        last_orientation: Optional[Orientation] = None
        for index, stage in enumerate(context.config.stages):
            operator_name = self._select_operator(stage, context.backend)
            plans.append(
                StageExecutionPlan(
                    stage_index=index,
                    stage=stage,
                    operator_name=operator_name,
                    last_orientation_before=last_orientation,
                )
            )
            _, last_orientation = self.build_actions(stage, last_orientation)
        return plans

    @staticmethod
    def _select_operator(stage: StageConfig, backend: SceneBackend) -> str:
        if not stage.operator:
            raise ValueError("Stage did not specify an operator.")
        backend.get_operator_handler(stage.operator)
        return stage.operator

    @staticmethod
    def build_actions(
        stage: StageConfig,
        last_orientation: Optional[Orientation] = None,
    ) -> tuple[List[PrimitiveAction], Optional[Orientation]]:
        control = TaskFlowBuilder._normalize_control(stage)

        if stage.operation in {Operation.MOVE, Operation.PUSH, Operation.PRESS}:
            actions, last_orientation = TaskFlowBuilder._build_pose_actions(
                TaskFlowBuilder._require_moves(stage, control, "pre_move"),
                last_orientation,
            )
        else:
            actions, last_orientation = TaskFlowBuilder._build_pose_actions(
                control.pre_move, last_orientation
            )

        if stage.operation == Operation.GRASP:
            actions.append(
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._grasp_eef(control))
            )
        elif stage.operation == Operation.RELEASE:
            actions.append(
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._release_eef(control))
            )
        elif stage.operation in {Operation.PICK, Operation.PULL}:
            actions.append(
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._grasp_eef(control))
            )
            for i, pm in enumerate(control.post_move):
                if pm.reference == PoseReference.OBJECT_WORLD or (
                    pm.reference == PoseReference.AUTO and stage.object
                ):
                    raise ValueError(
                        f"Stage '{stage.name or stage.operation.value}': post_move[{i}] uses "
                        f"reference '{pm.reference.value}' which tracks the target object. "
                        f"After a pick/pull, the grasped object moves with the EEF, causing "
                        f"a runaway feedback loop. Use 'eef_world' instead."
                    )
        elif stage.operation == Operation.PLACE:
            actions.append(
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._release_eef(control))
            )
        elif stage.operation == Operation.PRESS:
            actions.append(
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._grasp_eef(control))
            )
        elif stage.operation == Operation.PUSH:
            if control.eef is not None:
                actions.append(PrimitiveAction(kind="eef", eef=control.eef))
        elif stage.operation != Operation.MOVE:
            raise NotImplementedError(
                f"Unsupported operation '{stage.operation.value}'."
            )

        post_actions, last_orientation = TaskFlowBuilder._build_pose_actions(
            control.post_move, last_orientation
        )
        actions.extend(post_actions)
        return actions, last_orientation

    @staticmethod
    def _normalize_control(stage: StageConfig) -> StageControlConfig:
        if isinstance(stage.param, StageControlConfig):
            return stage.param
        if isinstance(stage.param, PoseControlConfig):
            return StageControlConfig(pre_move=[stage.param])
        if isinstance(stage.param, EefControlConfig):
            return StageControlConfig(eef=stage.param)
        raise TypeError(
            f"Stage '{stage.name or stage.operation.value}' has unsupported param type "
            f"'{type(stage.param).__name__}'."
        )

    @staticmethod
    def _build_pose_actions(
        poses: List[PoseControlConfig],
        last_orientation: Optional[Orientation] = None,
    ) -> tuple[List[PrimitiveAction], Optional[Orientation]]:
        actions: List[PrimitiveAction] = []
        for pose in poses:
            effective_pose = pose
            if pose.orientation:
                last_orientation = pose.orientation
            elif pose.rotation:
                last_orientation = euler_to_quaternion(
                    tuple(float(v) for v in pose.rotation)
                )
            elif last_orientation is not None:
                effective_pose = pose.model_copy(
                    update={"orientation": last_orientation}
                )

            if effective_pose.arc is not None:
                sub_poses = TaskFlowBuilder._split_arc(effective_pose)
                if effective_pose.arc.absolute:
                    for sp in sub_poses:
                        actions.append(PrimitiveAction(kind="pose", pose=sp))
                else:
                    arc_snapshot = ArcExecutionSnapshot()
                    cumulative_angle = 0.0
                    for sp in sub_poses:
                        assert sp.arc is not None
                        cumulative_angle += sp.arc.angle
                        actions.append(
                            PrimitiveAction(
                                kind="pose",
                                pose=sp,
                                arc_snapshot=arc_snapshot,
                                arc_cumulative_angle=cumulative_angle,
                            )
                        )
            else:
                actions.append(PrimitiveAction(kind="pose", pose=effective_pose))
        return actions, last_orientation

    @staticmethod
    def _split_arc(pose: PoseControlConfig) -> List[PoseControlConfig]:
        arc = pose.arc
        assert arc is not None
        if arc.absolute:
            return [pose]
        total = abs(arc.angle)
        n_steps = max(1, math.ceil(total / arc.max_step))
        step_angle = arc.angle / n_steps
        return [
            PoseControlConfig(
                arc=ArcControlConfig(
                    pivot=arc.pivot,
                    axis=arc.axis,
                    angle=step_angle,
                ),
                reference=pose.reference,
            )
            for _ in range(n_steps)
        ]

    @staticmethod
    def _require_moves(
        stage: StageConfig,
        control: StageControlConfig,
        field_name: str,
    ) -> List[PoseControlConfig]:
        poses = getattr(control, field_name)
        if not poses:
            raise ValueError(
                f"Stage '{stage.name or stage.operation.value}' requires at least one pose target in '{field_name}'."
            )
        return poses

    @staticmethod
    def _grasp_eef(control: StageControlConfig) -> EefControlConfig:
        return control.eef or EefControlConfig(close=True)

    @staticmethod
    def _release_eef(control: StageControlConfig) -> EefControlConfig:
        return control.eef or EefControlConfig(close=False)


class TaskRunner:
    """Stateful batch task executor controlled by ``reset`` and repeated ``update`` calls."""

    def __init__(self, builder: Optional[TaskFlowBuilder] = None) -> None:
        self.builder = builder or TaskFlowBuilder()
        self._context: Optional[ExecutionContext] = None
        self._plan: List[StageExecutionPlan] = []
        self._records: List[ExecutionRecord] = []
        self._env_states: List[_EnvRuntimeState] = []
        self._has_reset: np.ndarray = np.zeros(0, dtype=bool)

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
    ) -> ExecutionSummary:
        return _build_execution_summary(
            update=update or self._build_task_update(),
            records=self._records,
            total_stages=len(self._plan),
            max_updates=max_updates,
            updates_used=updates_used,
        )

    def from_yaml(self, path: str | Path) -> "TaskRunner":
        return self.from_config(load_task_file(path))

    def from_config(self, config: TaskFileConfig) -> "TaskRunner":
        backend = config.backend(config.task, config.operators)
        if not isinstance(backend, SceneBackend):
            raise TypeError(
                "Task file backend must be an instantiated SceneBackend. "
                f"Got {type(backend).__name__}."
            )
        self._context = ExecutionContext(
            config=config.task,
            backend=backend,
            task_file=config,
        )
        self._plan = self.builder.build(self._context)
        self._context.plan = self._plan
        self._context.backend.setup(self._context.config)
        self._env_states = [_EnvRuntimeState() for _ in range(backend.batch_size)]
        self._has_reset = np.zeros(backend.batch_size, dtype=bool)
        self._records = []
        return self

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        context.backend.reset(mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._env_states[env_index] = _EnvRuntimeState()
                self._env_states[
                    env_index
                ].latest_details = self._collect_reset_details(env_index, context)
        self._has_reset[mask] = True
        self._set_interest_focus()
        return self._build_task_update()

    def update(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        self._validate_update_mask(mask)
        for env_index, state in enumerate(self._env_states):
            if not mask[env_index] or state.done:
                continue
            self._update_env(env_index, state, context)
        self._set_interest_focus()
        return self._build_task_update()

    def close(self) -> None:
        if self._context is None:
            return
        self._context.backend.teardown()
        self._context = None
        self._plan = []
        self._records = []
        self._env_states = []
        self._has_reset = np.zeros(0, dtype=bool)

    def _update_env(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
    ) -> None:
        if state.stage_cursor >= len(self._plan):
            state.done = True
            state.success = True
            state.latest_status = StageExecutionStatus.SUCCEEDED
            state.phase = None
            state.phase_step = None
            return

        if state.active is None:
            plan = self._plan[state.stage_cursor]
            failure = (
                None
                if plan.stage.operation == Operation.PULL
                else self._check_stage_condition(
                    env_index=env_index,
                    context=context,
                    plan=plan,
                    condition_type=OperationConditionType.PERFORM,
                )
            )
            if failure is not None:
                self._record_failure(env_index, plan, failure)
                state.done = True
                state.success = False
                state.latest_status = StageExecutionStatus.FAILED
                state.latest_details = failure
                return
            state.active = self._start_stage(env_index, context, plan)
            state.latest_status = StageExecutionStatus.RUNNING

        assert state.active is not None
        active = state.active
        action = active.actions[active.action_index]
        mask = self._mask_for_env(env_index)
        result = self._run_action(
            env_index=env_index,
            operator=active.operator,
            action=action,
            target=active.target,
            backend=context.backend,
            env_mask=mask,
        )
        signal = result.signals[env_index]
        details = {
            "env_index": env_index,
            "action": action.kind,
            "action_index": active.action_index,
            **result.details[env_index],
        }

        if signal == ControlSignal.RUNNING:
            phase, phase_step = self._action_phase(active.actions, active.action_index)
            state.latest_status = StageExecutionStatus.RUNNING
            state.latest_details = details
            state.phase = phase
            state.phase_step = phase_step
            return

        if signal == ControlSignal.REACHED:
            completed_action = action
            active.action_index += 1
            op = active.plan.stage.operation
            mid_failure: Optional[Dict[str, Any]] = None
            if completed_action.kind == "eef":
                if op == Operation.PULL:
                    mid_failure = self._check_stage_condition(
                        env_index=env_index,
                        context=context,
                        plan=active.plan,
                        condition_type=OperationConditionType.PERFORM,
                        initial_pose=active.initial_object_pose,
                    )
                elif op == Operation.PICK and not bool(
                    context.backend.is_operator_grasping(active.operator.name)[
                        env_index
                    ]
                ):
                    mid_failure = self._check_stage_condition(
                        env_index=env_index,
                        context=context,
                        plan=active.plan,
                        condition_type=OperationConditionType.SUCCESS,
                        initial_pose=active.initial_object_pose,
                    )
                elif op == Operation.PRESS:
                    mid_failure = self._check_stage_condition(
                        env_index=env_index,
                        context=context,
                        plan=active.plan,
                        condition_type=OperationConditionType.SUCCESS,
                        initial_pose=active.initial_object_pose,
                    )
            if mid_failure is not None:
                self._record_failure(env_index, active.plan, mid_failure)
                state.active = None
                state.done = True
                state.success = False
                state.latest_status = StageExecutionStatus.FAILED
                state.latest_details = mid_failure
                state.phase = None
                state.phase_step = None
                return

            if active.action_index < len(active.actions):
                phase, phase_step = self._action_phase(
                    active.actions, active.action_index
                )
                state.latest_status = StageExecutionStatus.RUNNING
                state.latest_details = details
                state.phase = phase
                state.phase_step = phase_step
                return

            success_failure = (
                None
                if op == Operation.PRESS
                else self._check_stage_condition(
                    env_index=env_index,
                    context=context,
                    plan=active.plan,
                    condition_type=OperationConditionType.SUCCESS,
                    initial_pose=active.initial_object_pose,
                    completion_pose=self._completion_pose_from_active(active),
                )
            )
            if success_failure is not None:
                self._record_failure(env_index, active.plan, success_failure)
                state.active = None
                state.done = True
                state.success = False
                state.latest_status = StageExecutionStatus.FAILED
                state.latest_details = success_failure
                state.phase = None
                state.phase_step = None
                return

            self._records.append(
                ExecutionRecord(
                    env_index=env_index,
                    stage_index=active.plan.stage_index,
                    stage_name=active.plan.stage_name,
                    operator=active.operator.name,
                    operation=active.plan.stage.operation.value,
                    target_object=active.plan.stage.object,
                    blocking=active.plan.stage.blocking,
                    status=StageExecutionStatus.SUCCEEDED,
                    details=details,
                )
            )
            state.stage_cursor += 1
            state.active = None
            state.latest_status = StageExecutionStatus.SUCCEEDED
            state.latest_details = details
            state.phase = None
            state.phase_step = None
            if state.stage_cursor >= len(self._plan):
                state.done = True
                state.success = True
            else:
                state.success = None
            return

        failure = self._build_action_failure_details(active.plan, details, signal)
        self._record_failure(env_index, active.plan, failure)
        state.active = None
        state.done = True
        state.success = False
        state.latest_status = StageExecutionStatus.FAILED
        state.latest_details = failure
        state.phase = None
        state.phase_step = None

    def _record_failure(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        details: Dict[str, Any],
    ) -> None:
        self._records.append(
            ExecutionRecord(
                env_index=env_index,
                stage_index=plan.stage_index,
                stage_name=plan.stage_name,
                operator=plan.operator_name,
                operation=plan.stage.operation.value,
                target_object=plan.stage.object,
                blocking=plan.stage.blocking,
                status=StageExecutionStatus.FAILED,
                details=details,
            )
        )

    def _start_stage(
        self,
        env_index: int,
        context: ExecutionContext,
        plan: StageExecutionPlan,
    ) -> ActiveStageState:
        operator = context.backend.get_operator_handler(plan.operator_name)
        target = context.backend.get_object_handler(plan.stage.object)
        initial_object_pose: Optional[PoseState] = None
        if target is not None:
            initial_object_pose = target.get_pose().select(env_index)
        return ActiveStageState(
            plan=plan,
            operator=operator,
            target=target,
            actions=deepcopy(
                self.builder.build_actions(plan.stage, plan.last_orientation_before)[0]
            ),
            initial_object_pose=initial_object_pose,
        )

    @staticmethod
    def _check_stage_condition(
        env_index: int,
        context: ExecutionContext,
        plan: StageExecutionPlan,
        condition_type: OperationConditionType,
        initial_pose: Optional[PoseState] = None,
        completion_pose: Optional[PoseControlConfig] = None,
    ) -> Optional[Dict[str, Any]]:
        return _check_stage_condition(
            env_index=env_index,
            context=context,
            plan=plan,
            condition_type=condition_type,
            initial_pose=initial_pose,
            completion_pose=completion_pose,
        )

    @staticmethod
    def _build_action_failure_details(
        plan: StageExecutionPlan,
        details: Dict[str, Any],
        signal: ControlSignal,
    ) -> Dict[str, Any]:
        enriched = dict(details)
        enriched.setdefault("failure_stage", "execution")
        enriched.setdefault("operator", plan.operator_name)
        enriched.setdefault("operation", plan.stage.operation.value)
        enriched.setdefault("target_object", plan.stage.object)

        if signal == ControlSignal.TIMED_OUT:
            enriched.setdefault("failure_category", "controller_timeout")
            enriched.setdefault(
                "failure_reason", "primitive action did not finish before timeout"
            )
        elif signal == ControlSignal.FAILED:
            enriched.setdefault("failure_category", "controller_failure")
            enriched.setdefault("failure_reason", "primitive action reported failure")
        else:
            enriched.setdefault("failure_category", "execution_failure")
            enriched.setdefault(
                "failure_reason", "primitive action failed during execution"
            )
        return enriched

    @staticmethod
    def _completion_pose_from_active(
        active: ActiveStageState,
    ) -> Optional[PoseControlConfig]:
        for action in reversed(active.actions):
            if action.kind == "pose" and action.resolved_pose is not None:
                return action.resolved_pose
        return None

    @staticmethod
    def _run_action(
        env_index: int,
        operator: OperatorHandler,
        action: PrimitiveAction,
        target: Optional[ObjectHandler],
        backend: SceneBackend,
        env_mask: np.ndarray,
    ) -> ControlResult:
        if action.kind == "pose" and action.pose is not None:
            is_arc = action.pose.arc is not None
            is_snapshot = action.pose.reference in {
                PoseReference.EEF_WORLD,
                PoseReference.EEF,
            } or (is_arc and not action.pose.arc.absolute)
            if is_snapshot and action.resolved_pose is not None:
                resolved_pose = action.resolved_pose
            else:
                resolved_pose = TaskRunner._resolve_pose_command(
                    env_index=env_index,
                    operator=operator,
                    pose=action.pose,
                    target=target,
                    backend=backend,
                    action=action,
                )
                action.resolved_pose = resolved_pose
            return operator.move_to_pose(resolved_pose, target, env_mask=env_mask)
        if action.kind == "eef" and action.eef is not None:
            return operator.control_eef(action.eef, env_mask=env_mask)
        raise RuntimeError(f"Invalid primitive action '{action.kind}'.")

    @staticmethod
    def _resolve_arc_command(
        env_index: int,
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        backend: SceneBackend,
        action: Optional[PrimitiveAction] = None,
    ) -> PoseControlConfig:
        arc = pose.arc
        assert arc is not None

        angle = arc.angle
        if arc.absolute:
            if not isinstance(arc.pivot, str):
                raise ValueError("Arc absolute mode requires pivot to be a joint name.")
            current_joint = backend.get_joint_angle(arc.pivot, env_index)
            delta = arc.angle - current_joint
            sign = 1.0 if delta >= 0 else -1.0
            angle = sign * min(abs(delta), arc.max_step)
            pivot_world_pos = backend.get_element_pose(arc.pivot, env_index).position[0]
            current_eef = operator.get_end_effector_pose().select(env_index)
        elif action is not None and action.arc_snapshot is not None:
            snapshot = action.arc_snapshot
            if snapshot.pivot_world_pos is None:
                snapshot.pivot_world_pos = TaskRunner._resolve_arc_pivot_world_pos(
                    env_index=env_index,
                    operator=operator,
                    pose=pose,
                    target=target,
                    backend=backend,
                )
            if snapshot.start_eef_pose is None:
                snapshot.start_eef_pose = operator.get_end_effector_pose().select(
                    env_index
                )
            pivot_world_pos = snapshot.pivot_world_pos
            current_eef = snapshot.start_eef_pose
            if action.arc_cumulative_angle is not None:
                angle = action.arc_cumulative_angle
        else:
            pivot_world_pos = TaskRunner._resolve_arc_pivot_world_pos(
                env_index=env_index,
                operator=operator,
                pose=pose,
                target=target,
                backend=backend,
            )
            current_eef = operator.get_end_effector_pose().select(env_index)
        rotated = rotate_pose_around_axis(current_eef, pivot_world_pos, arc.axis, angle)
        return PoseControlConfig(
            position=tuple(float(v) for v in rotated.position[0]),
            orientation=tuple(float(v) for v in rotated.orientation[0]),
            reference=PoseReference.WORLD,
            relative=False,
            use_slerp=pose.use_slerp,
            max_linear_step=pose.max_linear_step,
            max_angular_step=pose.max_angular_step,
        )

    @staticmethod
    def _resolve_pose_command(
        env_index: int,
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        backend: SceneBackend,
        action: Optional[PrimitiveAction] = None,
    ) -> PoseControlConfig:
        if pose.arc is not None:
            return TaskRunner._resolve_arc_command(
                env_index, operator, pose, target, backend, action
            )
        reference_pose = TaskRunner._resolve_reference_pose(
            env_index=env_index,
            operator=operator,
            pose=pose,
            target=target,
        )
        current_pose = operator.get_end_effector_pose().select(env_index)
        local_pose = TaskRunner._pose_config_to_local_pose(pose)
        inherit_orientation = not pose.orientation and not pose.rotation
        current_local = compose_pose(inverse_pose(reference_pose), current_pose)

        if pose.relative:
            target_pose = compose_pose(current_local, local_pose)
        else:
            target_pose = (
                PoseState(
                    position=local_pose.position[0],
                    orientation=current_local.orientation[0],
                )
                if inherit_orientation
                else local_pose
            )

        world_pose = compose_pose(reference_pose, target_pose)
        return PoseControlConfig(
            position=tuple(float(v) for v in world_pose.position[0]),
            orientation=tuple(float(v) for v in world_pose.orientation[0]),
            reference=PoseReference.WORLD,
            relative=False,
            use_slerp=pose.use_slerp,
            max_linear_step=pose.max_linear_step,
            max_angular_step=pose.max_angular_step,
        )

    @staticmethod
    def _resolve_reference_pose(
        env_index: int,
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
    ) -> PoseState:
        reference = pose.reference
        if reference == PoseReference.AUTO:
            reference = (
                PoseReference.OBJECT_WORLD if target is not None else PoseReference.BASE
            )
        if reference == PoseReference.WORLD:
            return PoseState()
        if reference == PoseReference.BASE:
            return operator.get_base_pose().select(env_index)
        if reference == PoseReference.EEF:
            return operator.get_end_effector_pose().select(env_index)
        if reference == PoseReference.OBJECT_WORLD:
            if target is None:
                raise ValueError(
                    "Pose reference OBJECT_WORLD requires a target object."
                )
            object_pose = target.get_pose().select(env_index)
            return PoseState(position=object_pose.position[0])
        if reference == PoseReference.EEF_WORLD:
            eef_pose = operator.get_end_effector_pose().select(env_index)
            return PoseState(position=eef_pose.position[0])
        if reference == PoseReference.OBJECT:
            if target is None:
                raise ValueError("Pose reference OBJECT requires a target object.")
            return target.get_pose().select(env_index)
        raise NotImplementedError(f"Unsupported pose reference '{reference.value}'.")

    @staticmethod
    def _resolve_arc_pivot_world_pos(
        env_index: int,
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        backend: SceneBackend,
    ) -> Position:
        arc = pose.arc
        assert arc is not None
        if isinstance(arc.pivot, str):
            return tuple(
                float(v)
                for v in backend.get_element_pose(arc.pivot, env_index).position[0]
            )
        reference_pose = TaskRunner._resolve_reference_pose(
            env_index=env_index,
            operator=operator,
            pose=pose,
            target=target,
        )
        pivot_local = PoseState(position=arc.pivot)
        composed = compose_pose(reference_pose, pivot_local)
        return tuple(float(v) for v in composed.position[0])

    @staticmethod
    def _pose_config_to_local_pose(pose: PoseControlConfig) -> PoseState:
        return pose_config_to_pose_state(pose)

    @staticmethod
    def _action_phase(
        actions: List[PrimitiveAction], action_index: int
    ) -> tuple[str, Optional[int]]:
        eef_idx: Optional[int] = None
        for idx, action in enumerate(actions):
            if action.kind == "eef":
                eef_idx = idx
                break
        if eef_idx is not None and action_index == eef_idx:
            return "eef", None
        if eef_idx is None or action_index < eef_idx:
            return "pre_move", action_index
        return "post_move", action_index - (eef_idx + 1)

    def _set_interest_focus(self) -> None:
        context = self._require_context()
        object_names: List[str] = []
        operation_names: List[str] = []
        for state in self._env_states:
            if state.active is None:
                object_names.append("")
                operation_names.append("")
            else:
                object_names.append(state.active.plan.stage.object)
                operation_names.append(state.active.plan.stage.operation.value)
        context.backend.set_interest_objects_and_operations(
            object_names, operation_names
        )

    def _build_task_update(self) -> TaskUpdate:
        stage_index: List[int] = []
        stage_name: List[str] = []
        status: List[StageExecutionStatus] = []
        done: List[bool] = []
        success: List[Optional[bool]] = []
        details: List[Dict[str, Any]] = []
        phase: List[Optional[str]] = []
        phase_step: List[int] = []
        for state in self._env_states:
            if state.active is not None:
                stage_index.append(state.active.plan.stage_index)
                stage_name.append(state.active.plan.stage_name)
            elif state.stage_cursor < len(self._plan):
                stage_index.append(self._plan[state.stage_cursor].stage_index)
                stage_name.append(self._plan[state.stage_cursor].stage_name)
            else:
                stage_index.append(-1)
                stage_name.append("")
            status.append(state.latest_status)
            done.append(state.done)
            success.append(state.success)
            details.append(dict(state.latest_details))
            phase.append(state.phase)
            phase_step.append(-1 if state.phase_step is None else state.phase_step)
        return TaskUpdate(
            stage_index=np.asarray(stage_index, dtype=np.int64),
            stage_name=stage_name,
            status=np.asarray(status, dtype=object),
            done=np.asarray(done, dtype=bool),
            success=np.asarray(success, dtype=object),
            details=details,
            phase=phase,
            phase_step=np.asarray(phase_step, dtype=np.int64),
        )

    def _collect_reset_details(
        self,
        env_index: int,
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        initial_poses: Dict[str, Any] = {}
        names_in_order: List[str] = []
        seen_names: set[str] = set()

        for stage in context.config.stages:
            if stage.operator and stage.operator not in seen_names:
                names_in_order.append(stage.operator)
                seen_names.add(stage.operator)
            if stage.object and stage.object not in seen_names:
                names_in_order.append(stage.object)
                seen_names.add(stage.object)

        for name in context.config.randomization:
            if name not in seen_names:
                names_in_order.append(name)
                seen_names.add(name)

        for name in names_in_order:
            object_handler: Optional[ObjectHandler]
            try:
                object_handler = context.backend.get_object_handler(name)
            except KeyError:
                object_handler = None
            if object_handler is not None:
                pose = object_handler.get_pose().select(env_index)
                initial_poses[name] = self._serialize_pose(pose)
                continue

            try:
                operator = context.backend.get_operator_handler(name)
            except KeyError:
                continue
            entry_details = {
                "base_pose": self._serialize_pose(
                    operator.get_base_pose().select(env_index)
                ),
                "eef_pose": self._serialize_pose(
                    operator.get_end_effector_pose().select(env_index)
                ),
            }
            initial_poses[name] = entry_details
        if not initial_poses:
            return {}
        return {"initial_poses": initial_poses}

    @staticmethod
    def _serialize_pose(pose: PoseState) -> Dict[str, List[float]]:
        return {
            "position": [round(float(v), 4) for v in pose.position[0]],
            "orientation": [round(float(v), 4) for v in pose.orientation[0]],
        }

    def _normalize_mask(self, env_mask: Optional[np.ndarray]) -> np.ndarray:
        batch_size = self._require_context().backend.batch_size
        if env_mask is None:
            return np.ones(batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if len(mask) != batch_size:
            raise ValueError(
                f"env_mask must have shape ({batch_size},), got {mask.shape}"
            )
        return mask

    def _mask_for_env(self, env_index: int) -> np.ndarray:
        mask = np.zeros(self._require_context().backend.batch_size, dtype=bool)
        mask[env_index] = True
        return mask

    def _validate_update_mask(self, env_mask: np.ndarray) -> None:
        missing = np.flatnonzero(env_mask & ~self._has_reset)
        if missing.size == 0:
            return
        missing_str = ", ".join(str(int(i)) for i in missing.tolist())
        raise RuntimeError(
            "TaskRunner.update() was called for envs that have not been reset: "
            f"[{missing_str}]. Call reset(env_mask=...) for those envs first."
        )

    def _require_context(self) -> ExecutionContext:
        if self._context is None:
            raise RuntimeError("TaskRunner is not initialized. Call from_yaml() first.")
        return self._context


def _check_stage_condition(
    env_index: int,
    context: ExecutionContext,
    plan: StageExecutionPlan,
    condition_type: OperationConditionType,
    initial_pose: Optional[PoseState] = None,
    completion_pose: Optional[PoseControlConfig] = None,
) -> Optional[Dict[str, Any]]:
    constraints = OPERATION_CONDITIONS.get(plan.stage.operation)
    if not constraints:
        return None

    constraint = constraints.get(condition_type)
    if constraint is None or constraint == OperationConstraint.NONE:
        return None

    operator_name = plan.operator_name
    object_name = plan.stage.object
    backend = context.backend
    is_grasping = bool(backend.is_operator_grasping(operator_name)[env_index])

    if constraint == OperationConstraint.GRASPED:
        satisfied = is_grasping
    elif constraint == OperationConstraint.RELEASED:
        satisfied = not is_grasping
    elif constraint == OperationConstraint.CONTACTED:
        satisfied = bool(
            backend.is_operator_contacting(operator_name, object_name)[env_index]
        )
    elif constraint == OperationConstraint.DISPLACED:
        satisfied = (
            bool(backend.is_object_displaced(object_name, initial_pose)[env_index])
            if initial_pose is not None and object_name
            else True
        )
    elif constraint == OperationConstraint.REACHED:
        operator = backend.get_operator_handler(operator_name)
        tolerance = getattr(
            getattr(getattr(operator, "control", None), "tolerance", None),
            "position",
            0.01,
        )
        orientation_tolerance = getattr(
            getattr(getattr(operator, "control", None), "tolerance", None),
            "orientation",
            0.08,
        )
        if completion_pose is None:
            satisfied = False
        else:
            current_pose = operator.get_end_effector_pose().select(env_index)
            position_error = float(
                np.linalg.norm(
                    np.asarray(current_pose.position[0], dtype=np.float64)
                    - np.asarray(completion_pose.position, dtype=np.float64)
                )
            )
            orientation_error = float(
                quaternion_angular_distance(
                    current_pose.orientation[0],
                    np.asarray(completion_pose.orientation, dtype=np.float64),
                )
            )
            satisfied = position_error <= float(
                tolerance
            ) and orientation_error <= float(orientation_tolerance)
    else:
        satisfied = True

    if satisfied:
        return None

    phase = (
        "precondition"
        if condition_type == OperationConditionType.PERFORM
        else "postcondition"
    )
    failure_category, failure_reason = {
        OperationConstraint.GRASPED: (
            "missing_grasp",
            "operator is not grasping the required object",
        ),
        OperationConstraint.RELEASED: (
            "unexpected_grasp",
            "operator is still grasping when it should be empty-handed",
        ),
        OperationConstraint.CONTACTED: (
            "no_contact",
            f"operator end-effector is not in contact with '{object_name}'",
        ),
        OperationConstraint.DISPLACED: (
            "no_displacement",
            f"object '{object_name}' has not been displaced beyond the threshold",
        ),
        OperationConstraint.REACHED: (
            "target_not_reached",
            "operator end-effector is not within tolerance of the target pose",
        ),
    }.get(constraint, ("condition_mismatch", "stage condition is not satisfied"))
    details = {
        "event": f"stage_{phase}_failed",
        "failure_stage": phase,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "condition_type": condition_type.value,
        "required_constraint": constraint.value,
        "operator": operator_name,
        "operation": plan.stage.operation.value,
        "target_object": object_name,
        "is_operator_grasping": is_grasping,
        "env_index": env_index,
    }
    if constraint == OperationConstraint.REACHED:
        details["completion_pose_available"] = completion_pose is not None
        if completion_pose is not None:
            operator = backend.get_operator_handler(operator_name)
            current_pose = operator.get_end_effector_pose().select(env_index)
            details["target_pose"] = completion_pose.model_dump(mode="json")
            details["current_pose"] = {
                "position": [float(v) for v in current_pose.position[0]],
                "orientation": [float(v) for v in current_pose.orientation[0]],
            }
            details["position_error"] = float(
                np.linalg.norm(
                    np.asarray(current_pose.position[0], dtype=np.float64)
                    - np.asarray(completion_pose.position, dtype=np.float64)
                )
            )
            details["orientation_error"] = float(
                quaternion_angular_distance(
                    current_pose.orientation[0],
                    np.asarray(completion_pose.orientation, dtype=np.float64),
                )
            )
    return details


def _collect_reset_details(
    env_index: int,
    context: ExecutionContext,
) -> Dict[str, Any]:
    initial_poses: Dict[str, Any] = {}
    for name in context.config.randomization:
        object_handler: Optional[ObjectHandler]
        try:
            object_handler = context.backend.get_object_handler(name)
        except KeyError:
            object_handler = None
        if object_handler is not None:
            pose = object_handler.get_pose().select(env_index)
            initial_poses[name] = _serialize_pose(pose)
            continue

        try:
            operator = context.backend.get_operator_handler(name)
        except KeyError:
            continue
        pose = operator.get_base_pose().select(env_index)
        initial_poses[name] = _serialize_pose(pose)
    if not initial_poses:
        return {}
    return {"initial_poses": initial_poses}


def _serialize_pose(pose: PoseState) -> Dict[str, List[float]]:
    return {
        "position": [round(float(v), 4) for v in pose.position[0]],
        "orientation": [round(float(v), 4) for v in pose.orientation[0]],
    }


def _build_execution_summary(
    *,
    update: TaskUpdate,
    records: List[ExecutionRecord],
    total_stages: int,
    max_updates: Optional[int],
    updates_used: int,
) -> ExecutionSummary:
    batch_size = len(update.stage_name)
    completed_stage_count = np.zeros(batch_size, dtype=np.int64)
    for record in records:
        if record.status == StageExecutionStatus.SUCCEEDED:
            completed_stage_count[record.env_index] += 1
    return ExecutionSummary(
        total_stages=total_stages,
        max_updates=max_updates,
        updates_used=updates_used,
        completed_stage_count=completed_stage_count,
        final_stage_index=np.asarray(update.stage_index, dtype=np.int64),
        final_stage_name=list(update.stage_name),
        final_status=np.asarray(update.status, dtype=object),
        final_done=np.asarray(update.done, dtype=bool),
        final_success=np.asarray(update.success, dtype=object),
        records=list(records),
    )


def _resolve_policy_completion_pose(
    *,
    env_index: int,
    operator: OperatorHandler,
    target: Optional[ObjectHandler],
    backend: SceneBackend,
    action: PrimitiveAction,
) -> Optional[PoseControlConfig]:
    if action.pose is None:
        return None
    completion_action = deepcopy(action)
    return TaskRunner._resolve_pose_command(
        env_index=env_index,
        operator=operator,
        pose=completion_action.pose,
        target=target,
        backend=backend,
        action=completion_action,
    )


def load_yaml(path: str | Path) -> Dict[str, Any]:
    config = OmegaConf.load(Path(path))
    data = OmegaConf.to_container(config, resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {path}")
    return data


def load_config(path: str | Path) -> AutoAtomConfig:
    return load_task_file(path).task


def load_task_file(path: str | Path) -> TaskFileConfig:
    config_path = Path(path)
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"YAML root must be a mapping: {config_path}")

    if "env" in config and config.env is not None:
        instantiate(config.env)
    raw = OmegaConf.to_container(config, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")
    return TaskFileConfig.model_validate(raw)
