"""YAML-driven task runner built from primitive pose and end-effector controls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import yaml

from .framework import (
    AutoAtomConfig,
    EefControlConfig,
    Operation,
    OPERATION_CONDITIONS,
    OperationConditionType,
    OperationConstraint,
    PoseControlConfig,
    PoseReference,
    StageConfig,
    StageControlConfig,
    TaskFileConfig,
)
from .utils.pose import PoseState, compose_pose, inverse_pose, pose_config_to_pose_state


class BackendFactory(Protocol):
    """Callable used to build a simulator backend from validated task file."""

    def __call__(self, task_file: TaskFileConfig) -> "SimulatorBackend":
        ...


class StageExecutionStatus(str, Enum):
    """High-level stage status returned to users."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ControlSignal(str, Enum):
    """Low-level controller signal returned by primitive operator commands."""

    RUNNING = "running"
    REACHED = "reached"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass
class ObjectHandler:
    """Opaque object handle resolved by the simulator backend."""

    name: str

    def get_pose(self) -> "PoseState":
        """Return the current world pose of the object."""
        raise NotImplementedError


@dataclass
class ControlResult:
    """Incremental low-level control result."""

    signal: ControlSignal
    details: Dict[str, Any] = field(default_factory=dict)


class OperatorHandler(ABC):
    """Operator exposes only primitive pose and end-effector controls."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name used by stage configs."""

    @abstractmethod
    def move_to_pose(
        self,
        pose: PoseControlConfig,
        simulator: "SimulatorBackend",
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        """Advance motion toward the desired pose."""

    @abstractmethod
    def control_eef(
        self,
        eef: EefControlConfig,
        simulator: "SimulatorBackend",
    ) -> ControlResult:
        """Advance the end-effector toward the desired state."""

    @abstractmethod
    def get_end_effector_pose(self, simulator: "SimulatorBackend") -> PoseState:
        """Return the current world pose of the operator end-effector."""

    @abstractmethod
    def get_base_pose(self, simulator: "SimulatorBackend") -> PoseState:
        """Return the current world pose of the operator base."""


class SimulatorBackend(ABC):
    """Abstract simulator backend used by the task runner."""

    @abstractmethod
    def setup(self, config: AutoAtomConfig) -> None:
        """Prepare backend resources for this task."""

    @abstractmethod
    def reset(self) -> None:
        """Reset simulator state for a new run."""

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
    def is_object_grasped(self, operator_name: str, object_name: str) -> bool:
        """Return whether the operator is currently grasping the given object."""

    @abstractmethod
    def is_operator_grasping(self, operator_name: str) -> bool:
        """Return whether the operator is currently grasping any object."""


@dataclass
class PrimitiveAction:
    """Single primitive control action derived from a stage."""

    kind: str
    pose: Optional[PoseControlConfig] = None
    eef: Optional[EefControlConfig] = None
    resolved_pose: Optional[PoseControlConfig] = None


@dataclass
class ExecutionRecord:
    """Final record for one completed or failed stage."""

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
    """Mutable runtime context shared across the task lifecycle."""

    config: AutoAtomConfig
    backend: SimulatorBackend
    task_file: TaskFileConfig


@dataclass
class StageExecutionPlan:
    """Validated executable stage plan."""

    stage_index: int
    stage: StageConfig
    operator_name: str

    @property
    def stage_name(self) -> str:
        return self.stage.name or f"stage_{self.stage_index}"


@dataclass
class TaskUpdate:
    """User-facing task progress returned by ``TaskRunner.update``."""

    stage_index: Optional[int]
    stage_name: str
    status: StageExecutionStatus
    done: bool
    success: Optional[bool]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveStageState:
    """Internal state for the currently active stage."""

    plan: StageExecutionPlan
    operator: OperatorHandler
    target: Optional[ObjectHandler]
    actions: List[PrimitiveAction]
    action_index: int = 0


class ComponentRegistry:
    """Registry mapping simulator names to backend factories."""

    def __init__(self) -> None:
        self._backend_factories: Dict[str, BackendFactory] = {}

    def register_backend(self, name: str, factory: BackendFactory) -> None:
        self._backend_factories[name] = factory

    def create_backend(self, task_file: TaskFileConfig) -> SimulatorBackend:
        try:
            factory = self._backend_factories[task_file.task.simulator]
        except KeyError as exc:
            known = ", ".join(sorted(self._backend_factories)) or "<empty>"
            raise KeyError(
                f"Simulator backend '{task_file.task.simulator}' is not registered. "
                f"Available backends: {known}"
            ) from exc
        return factory(task_file)


class TaskFlowBuilder:
    """Build stage plans and primitive action lists from validated config."""

    def build(self, context: ExecutionContext) -> List[StageExecutionPlan]:
        plans: List[StageExecutionPlan] = []
        for index, stage in enumerate(context.config.stages):
            operator_name = self._select_operator(stage, context.backend)
            self.build_actions(stage)
            plans.append(
                StageExecutionPlan(
                    stage_index=index,
                    stage=stage,
                    operator_name=operator_name,
                )
            )
        return plans

    @staticmethod
    def _select_operator(stage: StageConfig, backend: SimulatorBackend) -> str:
        if not stage.operator:
            raise ValueError("Stage did not specify an operator.")
        backend.get_operator_handler(stage.operator)
        return stage.operator

    @staticmethod
    def build_actions(stage: StageConfig) -> List[PrimitiveAction]:
        control = TaskFlowBuilder._normalize_control(stage)
        if stage.operation in {Operation.MOVE, Operation.PUSH}:
            return [PrimitiveAction(kind="pose", pose=TaskFlowBuilder._require_pose(stage, control))]
        if stage.operation == Operation.GRASP:
            return [PrimitiveAction(kind="eef", eef=TaskFlowBuilder._grasp_eef(control))]
        if stage.operation == Operation.RELEASE:
            return [PrimitiveAction(kind="eef", eef=TaskFlowBuilder._release_eef(control))]
        if stage.operation in {Operation.PICK, Operation.PULL}:
            return [
                PrimitiveAction(kind="pose", pose=TaskFlowBuilder._require_pose(stage, control)),
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._grasp_eef(control)),
            ]
        if stage.operation == Operation.PLACE:
            return [
                PrimitiveAction(kind="pose", pose=TaskFlowBuilder._require_pose(stage, control)),
                PrimitiveAction(kind="eef", eef=TaskFlowBuilder._release_eef(control)),
            ]
        raise NotImplementedError(f"Unsupported operation '{stage.operation.value}'.")

    @staticmethod
    def _normalize_control(stage: StageConfig) -> StageControlConfig:
        if isinstance(stage.param, StageControlConfig):
            return stage.param
        if isinstance(stage.param, PoseControlConfig):
            return StageControlConfig(pose=stage.param)
        if isinstance(stage.param, EefControlConfig):
            return StageControlConfig(eef=stage.param)
        raise TypeError(
            f"Stage '{stage.name or stage.operation.value}' has unsupported param type "
            f"'{type(stage.param).__name__}'."
        )

    @staticmethod
    def _require_pose(stage: StageConfig, control: StageControlConfig) -> PoseControlConfig:
        if control.pose is None:
            raise ValueError(f"Stage '{stage.name or stage.operation.value}' requires a pose target.")
        return control.pose

    @staticmethod
    def _grasp_eef(control: StageControlConfig) -> EefControlConfig:
        return control.eef or EefControlConfig(close=True)

    @staticmethod
    def _release_eef(control: StageControlConfig) -> EefControlConfig:
        return control.eef or EefControlConfig(close=False)


class TaskRunner:
    """Stateful task executor controlled by ``reset`` and repeated ``update`` calls."""

    def __init__(
        self,
        registry: ComponentRegistry,
        builder: Optional[TaskFlowBuilder] = None,
    ) -> None:
        self.registry = registry
        self.builder = builder or TaskFlowBuilder()
        self._context: Optional[ExecutionContext] = None
        self._plan: List[StageExecutionPlan] = []
        self._active_stage: Optional[ActiveStageState] = None
        self._stage_index = 0
        self._records: List[ExecutionRecord] = []

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)

    def from_yaml(self, path: str | Path) -> "TaskRunner":
        task_file = load_task_file(path)
        backend = self.registry.create_backend(task_file)
        self._context = ExecutionContext(
            config=task_file.task,
            backend=backend,
            task_file=task_file,
        )
        self._plan = self.builder.build(self._context)
        self._context.backend.setup(self._context.config)
        self._stage_index = 0
        self._active_stage = None
        self._records = []
        return self

    def reset(self) -> TaskUpdate:
        context = self._require_context()
        context.backend.reset()
        self._stage_index = 0
        self._active_stage = None
        self._records = []
        return self._build_pending_update()

    def update(self) -> TaskUpdate:
        context = self._require_context()
        if self._stage_index >= len(self._plan):
            return TaskUpdate(
                stage_index=None,
                stage_name="",
                status=StageExecutionStatus.SUCCEEDED,
                done=True,
                success=True,
            )

        if self._active_stage is None:
            plan = self._plan[self._stage_index]
            precondition_failure = self._check_stage_condition(
                context=context,
                plan=plan,
                condition_type=OperationConditionType.PERFORM,
            )
            if precondition_failure is not None:
                self._records.append(
                    ExecutionRecord(
                        stage_index=plan.stage_index,
                        stage_name=plan.stage_name,
                        operator=plan.operator_name,
                        operation=plan.stage.operation.value,
                        target_object=plan.stage.object,
                        blocking=plan.stage.blocking,
                        status=StageExecutionStatus.FAILED,
                        details=precondition_failure,
                    )
                )
                return self._build_update(
                    plan=plan,
                    status=StageExecutionStatus.FAILED,
                    details=precondition_failure,
                    done=True,
                    success=False,
                )
            self._active_stage = self._start_stage(context, plan)

        active = self._active_stage
        action = active.actions[active.action_index]
        result = self._run_action(active.operator, action, context.backend, active.target)
        details = {
            "action": action.kind,
            "action_index": active.action_index,
            **result.details,
        }

        if result.signal == ControlSignal.RUNNING:
            return self._build_update(
                plan=active.plan,
                status=StageExecutionStatus.RUNNING,
                details=details,
                done=False,
                success=None,
            )

        if result.signal == ControlSignal.REACHED:
            active.action_index += 1
            if active.action_index < len(active.actions):
                return self._build_update(
                    plan=active.plan,
                    status=StageExecutionStatus.RUNNING,
                    details=details,
                    done=False,
                    success=None,
                )
            success_failure = self._check_stage_condition(
                context=context,
                plan=active.plan,
                condition_type=OperationConditionType.SUCCESS,
            )
            if success_failure is not None:
                self._records.append(
                    ExecutionRecord(
                        stage_index=active.plan.stage_index,
                        stage_name=active.plan.stage_name,
                        operator=active.operator.name,
                        operation=active.plan.stage.operation.value,
                        target_object=active.plan.stage.object,
                        blocking=active.plan.stage.blocking,
                        status=StageExecutionStatus.FAILED,
                        details=success_failure,
                    )
                )
                self._active_stage = None
                return self._build_update(
                    plan=active.plan,
                    status=StageExecutionStatus.FAILED,
                    details=success_failure,
                    done=True,
                    success=False,
                )
            self._records.append(
                ExecutionRecord(
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
            self._stage_index += 1
            self._active_stage = None
            return self._build_update(
                plan=active.plan,
                status=StageExecutionStatus.SUCCEEDED,
                details=details,
                done=self._stage_index >= len(self._plan),
                success=True if self._stage_index >= len(self._plan) else None,
            )

        self._records.append(
            ExecutionRecord(
                stage_index=active.plan.stage_index,
                stage_name=active.plan.stage_name,
                operator=active.operator.name,
                operation=active.plan.stage.operation.value,
                target_object=active.plan.stage.object,
                blocking=active.plan.stage.blocking,
                status=StageExecutionStatus.FAILED,
                details=self._build_action_failure_details(
                    plan=active.plan,
                    details=details,
                    signal=result.signal,
                ),
            )
        )
        self._active_stage = None
        return self._build_update(
            plan=active.plan,
            status=StageExecutionStatus.FAILED,
            details=self._build_action_failure_details(
                plan=active.plan,
                details=details,
                signal=result.signal,
            ),
            done=True,
            success=False,
        )

    def close(self) -> None:
        if self._context is None:
            return
        self._context.backend.teardown()
        self._context = None
        self._plan = []
        self._active_stage = None

    def _start_stage(
        self,
        context: ExecutionContext,
        plan: StageExecutionPlan,
    ) -> ActiveStageState:
        operator = context.backend.get_operator_handler(plan.operator_name)
        target = context.backend.get_object_handler(plan.stage.object)
        return ActiveStageState(
            plan=plan,
            operator=operator,
            target=target,
            actions=self.builder.build_actions(plan.stage),
        )

    @staticmethod
    def _check_stage_condition(
        context: ExecutionContext,
        plan: StageExecutionPlan,
        condition_type: OperationConditionType,
    ) -> Optional[Dict[str, Any]]:
        constraints = OPERATION_CONDITIONS.get(plan.stage.operation)
        if not constraints:
            return None

        constraint = constraints.get(condition_type)
        if constraint is None:
            return None

        operator_name = plan.operator_name
        backend = context.backend
        is_grasping = backend.is_operator_grasping(operator_name)

        satisfied = True
        if constraint == OperationConstraint.GRASPED:
            satisfied = is_grasping
        elif constraint == OperationConstraint.RELEASED:
            satisfied = not is_grasping

        if satisfied:
            return None

        phase = "precondition" if condition_type == OperationConditionType.PERFORM else "postcondition"
        if constraint == OperationConstraint.GRASPED:
            failure_category = "missing_grasp"
            failure_reason = "operator is not grasping a required object"
        elif constraint == OperationConstraint.RELEASED:
            failure_category = "unexpected_grasp"
            failure_reason = "operator is still grasping an object when it should be empty-handed"
        else:
            failure_category = "condition_mismatch"
            failure_reason = "stage condition is not satisfied"

        return {
            "event": f"stage_{phase}_failed",
            "failure_stage": phase,
            "failure_category": failure_category,
            "failure_reason": failure_reason,
            "condition_type": condition_type.value,
            "required_constraint": constraint.value,
            "operator": operator_name,
            "operation": plan.stage.operation.value,
            "target_object": plan.stage.object,
            "is_operator_grasping": is_grasping,
        }

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
            enriched.setdefault("failure_reason", "primitive action did not finish before timeout")
        elif signal == ControlSignal.FAILED:
            enriched.setdefault("failure_category", "controller_failure")
            enriched.setdefault("failure_reason", "primitive action reported failure")
        else:
            enriched.setdefault("failure_category", "execution_failure")
            enriched.setdefault("failure_reason", "primitive action failed during execution")
        return enriched

    @staticmethod
    def _run_action(
        operator: OperatorHandler,
        action: PrimitiveAction,
        backend: SimulatorBackend,
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        if action.kind == "pose" and action.pose is not None:
            resolved_pose = TaskRunner._resolve_pose_command(
                operator=operator,
                pose=action.pose,
                backend=backend,
                target=target,
            )
            action.resolved_pose = resolved_pose
            return operator.move_to_pose(resolved_pose, backend, target)
        if action.kind == "eef" and action.eef is not None:
            return operator.control_eef(action.eef, backend)
        raise RuntimeError(f"Invalid primitive action '{action.kind}'.")

    @staticmethod
    def _resolve_pose_command(
        operator: OperatorHandler,
        pose: PoseControlConfig,
        backend: SimulatorBackend,
        target: Optional[ObjectHandler],
    ) -> PoseControlConfig:
        reference_pose = TaskRunner._resolve_reference_pose(
            operator=operator,
            pose=pose,
            backend=backend,
            target=target,
        )
        current_pose = operator.get_end_effector_pose(backend)
        local_pose = TaskRunner._pose_config_to_local_pose(pose)

        if pose.relative:
            current_local = compose_pose(
                inverse_pose(reference_pose),
                current_pose,
            )
            target_pose = compose_pose(current_local, local_pose)
        else:
            target_pose = local_pose

        world_pose = compose_pose(reference_pose, target_pose)
        return PoseControlConfig(
            position=world_pose.position,
            orientation=world_pose.orientation,
            reference=PoseReference.WORLD,
            relative=False,
        )

    @staticmethod
    def _resolve_reference_pose(
        operator: OperatorHandler,
        pose: PoseControlConfig,
        backend: SimulatorBackend,
        target: Optional[ObjectHandler],
    ) -> PoseState:
        reference = pose.reference
        if reference == PoseReference.AUTO:
            reference = PoseReference.OBJECT_WORLD if target is not None else PoseReference.BASE
        if reference == PoseReference.WORLD:
            return PoseState()
        if reference == PoseReference.BASE:
            return operator.get_base_pose(backend)
        if reference == PoseReference.END_EFFECTOR:
            return operator.get_end_effector_pose(backend)
        if reference == PoseReference.OBJECT_WORLD:
            if target is None:
                raise ValueError("Pose reference OBJECT_WORLD requires a target object.")
            object_pose = target.get_pose()
            return PoseState(position=object_pose.position)
        if reference == PoseReference.OBJECT:
            if target is None:
                raise ValueError("Pose reference OBJECT requires a target object.")
            return target.get_pose()
        raise NotImplementedError(f"Unsupported pose reference '{reference.value}'.")

    @staticmethod
    def _pose_config_to_local_pose(pose: PoseControlConfig) -> PoseState:
        return pose_config_to_pose_state(pose)

    def _build_pending_update(self) -> TaskUpdate:
        if not self._plan:
            return TaskUpdate(
                stage_index=None,
                stage_name="",
                status=StageExecutionStatus.SUCCEEDED,
                done=True,
                success=True,
            )
        return self._build_update(
            plan=self._plan[self._stage_index],
            status=StageExecutionStatus.PENDING,
            details={},
            done=False,
            success=None,
        )

    @staticmethod
    def _build_update(
        plan: StageExecutionPlan,
        status: StageExecutionStatus,
        details: Dict[str, Any],
        done: bool,
        success: Optional[bool],
    ) -> TaskUpdate:
        return TaskUpdate(
            stage_index=plan.stage_index,
            stage_name=plan.stage_name,
            status=status,
            done=done,
            success=success,
            details=details,
        )

    def _require_context(self) -> ExecutionContext:
        if self._context is None:
            raise RuntimeError("TaskRunner is not initialized. Call from_yaml() first.")
        return self._context


def load_yaml(path: str | Path) -> Dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a mapping: {yaml_path}")
    return data


def load_config(path: str | Path) -> AutoAtomConfig:
    return load_task_file(path).task


def load_task_file(path: str | Path) -> TaskFileConfig:
    raw = load_yaml(path)
    return TaskFileConfig.model_validate(raw)
