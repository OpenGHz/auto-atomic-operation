"""YAML-driven batch-first task runner built from primitive controls."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    ContextManager,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

import numpy as np

from .framework import (
    OPERATION_CONDITIONS,
    ArcControlConfig,
    AutoAtomConfig,
    EefControlConfig,
    IntervalSelectionConfig,
    Operation,
    OperationConditionType,
    OperationConstraint,
    Orientation,
    PlacedToleranceConfig,
    PoseControlConfig,
    PoseReference,
    Position,
    RandomizationReference,
    StageConfig,
    StageControlConfig,
    TaskFileConfig,
    TaskKeypointConfig,
    TaskPhase,
    UpdateBoundary,
    _phase_waypoint_count,
)
from .utils.pose import (
    PoseState,
    compose_pose,
    euler_to_quaternion,
    inverse_pose,
    orientation_within_tolerance_nullable,
    pose_config_to_pose_state,
    position_within_tolerance,
    position_within_tolerance_nullable,
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


@runtime_checkable
class EnvProtocol(Protocol):
    """Minimal environment interface expected by ``SceneBackend.env``."""

    def step(
        self, action: np.ndarray, env_mask: Optional[np.ndarray] = None
    ) -> None: ...

    def capture_observation(self) -> Dict[str, Dict[str, Any]]: ...

    def apply_joint_action(
        self,
        operator: str,
        action: Any,
        env_mask: Optional[np.ndarray] = None,
        kinematic: bool = False,
    ) -> None: ...

    def apply_pose_action(
        self,
        operator: str,
        position: Any,
        orientation: Any,
        gripper: Any = None,
        env_mask: Optional[np.ndarray] = None,
        kinematic: bool = False,
    ) -> None: ...


class SceneBackend(ABC):
    env: EnvProtocol
    """The underlying environment object managed by this backend.
    Concrete backends expose the actual env instance here
    (e.g. ``BatchedUnifiedMujocoEnv`` for MuJoCo)."""

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

    @property
    def dt_per_update(self) -> float:
        """Simulation time advanced per update() call, in seconds.

        Backends that track physics time should override this.
        Returns 0.0 by default (unknown).
        """
        return 0.0

    @contextmanager
    def defer_viewer_updates(self) -> Iterator[None]:
        """Defer viewer refreshes until a compound runner update completes.

        Backends without an interactive viewer keep the default no-op
        implementation. Viewer-backed implementations should coalesce all
        refresh requests inside this context into one final refresh.
        """
        yield

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
    phase: Optional[TaskPhase] = None
    waypoint: Optional[int] = None
    completes_keypoint: bool = True


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


@dataclass(frozen=True)
class _ResolvedTaskKeypoint:
    stage_index: int
    stage_name: str
    phase: TaskPhase
    waypoint: int

    def matches(self, configured: TaskKeypointConfig) -> bool:
        return (
            self.stage_name == configured.stage
            and self.phase == configured.phase
            and self.waypoint == configured.waypoint
        )


@dataclass(frozen=True)
class _EnvUpdateEvent:
    """Boundaries crossed by one internal controller update."""

    control_tick: bool = False
    primitive_reached: bool = False
    keypoint_reached: bool = False
    stage_succeeded: bool = False
    failed: bool = False
    completed_position: Optional[_ResolvedTaskKeypoint] = None
    completed_keypoint: Optional[_ResolvedTaskKeypoint] = None


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
    elapsed_time_sec: float = 0.0
    sim_time_sec: float = 0.0
    env_completion_steps: Optional[np.ndarray] = None
    env_completion_time_sec: Optional[np.ndarray] = None
    env_completion_sim_time_sec: Optional[np.ndarray] = None
    completed_stage_info: Dict[str, List[Optional[str]]] = field(default_factory=dict)
    records: List[ExecutionRecord] = field(default_factory=list)


@dataclass
class ActiveStageState:
    plan: StageExecutionPlan
    operator: OperatorHandler
    target: Optional[ObjectHandler]
    actions: List[PrimitiveAction]
    action_index: int = 0
    initial_object_pose: Optional[PoseState] = None
    held_object_name: Optional[str] = None


@dataclass
class _EnvRuntimeState:
    stage_cursor: int = 0
    active: Optional[ActiveStageState] = None
    done: bool = False
    success: bool = False
    phase: Optional[str] = None
    phase_step: Optional[int] = None
    latest_status: StageExecutionStatus = StageExecutionStatus.PENDING
    latest_details: Dict[str, Any] = field(default_factory=dict)
    reported_keypoint: Optional[_ResolvedTaskKeypoint] = None


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
                phase=TaskPhase.PRE_MOVE,
            )
        else:
            actions, last_orientation = TaskFlowBuilder._build_pose_actions(
                control.pre_move,
                last_orientation,
                phase=TaskPhase.PRE_MOVE,
            )

        if stage.operation == Operation.GRASP:
            actions.append(
                PrimitiveAction(
                    kind="eef",
                    eef=TaskFlowBuilder._grasp_eef(control),
                    phase=TaskPhase.EEF,
                    waypoint=0,
                )
            )
        elif stage.operation == Operation.RELEASE:
            actions.append(
                PrimitiveAction(
                    kind="eef",
                    eef=TaskFlowBuilder._release_eef(control),
                    phase=TaskPhase.EEF,
                    waypoint=0,
                )
            )
        elif stage.operation in {Operation.PICK, Operation.PULL}:
            actions.append(
                PrimitiveAction(
                    kind="eef",
                    eef=TaskFlowBuilder._grasp_eef(control),
                    phase=TaskPhase.EEF,
                    waypoint=0,
                )
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
                PrimitiveAction(
                    kind="eef",
                    eef=TaskFlowBuilder._release_eef(control),
                    phase=TaskPhase.EEF,
                    waypoint=0,
                )
            )
        elif stage.operation == Operation.PRESS:
            actions.append(
                PrimitiveAction(
                    kind="eef",
                    eef=TaskFlowBuilder._grasp_eef(control),
                    phase=TaskPhase.EEF,
                    waypoint=0,
                )
            )
        elif stage.operation == Operation.PUSH:
            if control.eef is not None:
                actions.append(
                    PrimitiveAction(
                        kind="eef",
                        eef=control.eef,
                        phase=TaskPhase.EEF,
                        waypoint=0,
                    )
                )
        elif stage.operation != Operation.MOVE:
            raise NotImplementedError(
                f"Unsupported operation '{stage.operation.value}'."
            )

        post_actions, last_orientation = TaskFlowBuilder._build_pose_actions(
            control.post_move,
            last_orientation,
            phase=TaskPhase.POST_MOVE,
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
        *,
        phase: TaskPhase = TaskPhase.PRE_MOVE,
    ) -> tuple[List[PrimitiveAction], Optional[Orientation]]:
        actions: List[PrimitiveAction] = []
        for waypoint, pose in enumerate(poses):
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
                    for sub_index, sp in enumerate(sub_poses):
                        actions.append(
                            PrimitiveAction(
                                kind="pose",
                                pose=sp,
                                phase=phase,
                                waypoint=waypoint,
                                completes_keypoint=sub_index == len(sub_poses) - 1,
                            )
                        )
                else:
                    arc_snapshot = ArcExecutionSnapshot()
                    cumulative_angle = 0.0
                    for sub_index, sp in enumerate(sub_poses):
                        assert sp.arc is not None
                        cumulative_angle += sp.arc.angle
                        actions.append(
                            PrimitiveAction(
                                kind="pose",
                                pose=sp,
                                phase=phase,
                                waypoint=waypoint,
                                completes_keypoint=sub_index == len(sub_poses) - 1,
                                arc_snapshot=arc_snapshot,
                                arc_cumulative_angle=cumulative_angle,
                            )
                        )
            else:
                actions.append(
                    PrimitiveAction(
                        kind="pose",
                        pose=effective_pose,
                        phase=phase,
                        waypoint=waypoint,
                    )
                )
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
        self._public_internal_updates: np.ndarray = np.zeros(0, dtype=np.int64)
        self._last_execution_details: List[Dict[str, Any]] = []

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
        elapsed_time_sec: float = 0.0,
    ) -> ExecutionSummary:
        dt = self._context.backend.dt_per_update if self._context else 0.0
        task_update = update or self._build_task_update()
        public_internal_updates = (
            int(np.max(self._public_internal_updates))
            if self._public_internal_updates.size
            else 0
        )
        summary = _build_execution_summary(
            update=task_update,
            records=self._records,
            total_stages=len(self._plan),
            max_updates=max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed_time_sec,
            sim_time_sec=public_internal_updates * dt,
        )
        if dt > 0:
            summary.env_completion_sim_time_sec = np.where(
                np.asarray(task_update.done, dtype=bool),
                self._public_internal_updates.astype(np.float64) * dt,
                np.nan,
            )
        return summary

    def from_yaml(self, path: str | Path) -> "TaskRunner":
        from .config_loader import load_task_file

        return self.from_config(load_task_file(path))

    def from_config(self, config: TaskFileConfig) -> "TaskRunner":
        backend = config.backend(config.task, config.task_operators)
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
        selection = config.execution.interval_selection
        if (
            config.execution.update_boundary == UpdateBoundary.KEYPOINT
            or selection is not None
        ):
            self._validate_keypoint_boundary_actions()
        if selection is not None:
            self._validate_interval_actions(selection)
        self._context.plan = self._plan
        self._context.backend.setup(self._context.config)
        self._env_states = [_EnvRuntimeState() for _ in range(backend.batch_size)]
        self._has_reset = np.zeros(backend.batch_size, dtype=bool)
        self._public_internal_updates = np.zeros(backend.batch_size, dtype=np.int64)
        self._last_execution_details = [{} for _ in range(backend.batch_size)]
        self._records = []
        return self

    def _validate_keypoint_boundary_actions(self) -> None:
        """Validate the configured keypoint identity emitted by the active builder."""
        for plan in self._plan:
            actions, _ = self.builder.build_actions(
                plan.stage, plan.last_orientation_before
            )
            groups: List[tuple[tuple[TaskPhase, int], List[int]]] = []
            for action_index, action in enumerate(actions):
                if not isinstance(action.phase, TaskPhase) or not isinstance(
                    action.waypoint, int
                ):
                    raise ValueError(
                        "Keypoint-aware execution requires every action emitted by "
                        f"{type(self.builder).__name__} to define a TaskPhase phase "
                        f"and integer waypoint; {plan.stage_name} action "
                        f"{action_index} does not"
                    )
                waypoint = action.waypoint
                count = _phase_waypoint_count(plan.stage, action.phase)
                if waypoint < 0 or waypoint >= count:
                    raise ValueError(
                        f"{type(self.builder).__name__} emitted invalid keypoint "
                        f"{plan.stage_name}.{action.phase.value}[{waypoint}] for "
                        f"action {action_index}"
                    )
                if not isinstance(action.completes_keypoint, bool):
                    raise ValueError(
                        f"{type(self.builder).__name__} must emit a boolean "
                        f"completes_keypoint for {plan.stage_name} action "
                        f"{action_index}"
                    )

                identity = (action.phase, waypoint)
                if not groups or groups[-1][0] != identity:
                    if any(group_identity == identity for group_identity, _ in groups):
                        raise ValueError(
                            f"{type(self.builder).__name__} emitted non-contiguous "
                            f"primitives for keypoint "
                            f"{plan.stage_name}.{action.phase.value}[{waypoint}]"
                        )
                    groups.append((identity, []))
                groups[-1][1].append(action_index)

            for (phase, waypoint), action_indices in groups:
                completion_indices = [
                    action_index
                    for action_index in action_indices
                    if actions[action_index].completes_keypoint
                ]
                if completion_indices != [action_indices[-1]]:
                    raise ValueError(
                        f"{type(self.builder).__name__} must mark only the final "
                        f"primitive of keypoint "
                        f"{plan.stage_name}.{phase.value}[{waypoint}] with "
                        "completes_keypoint=True"
                    )

    def _validate_interval_actions(
        self,
        selection: IntervalSelectionConfig,
    ) -> None:
        """Verify that the active builder emits both configured boundaries."""
        keypoints: List[_ResolvedTaskKeypoint] = []
        for plan in self._plan:
            actions, _ = self.builder.build_actions(
                plan.stage, plan.last_orientation_before
            )
            for action in actions:
                if (
                    not action.completes_keypoint
                    or action.phase is None
                    or action.waypoint is None
                ):
                    continue
                keypoints.append(
                    _ResolvedTaskKeypoint(
                        stage_index=plan.stage_index,
                        stage_name=plan.stage_name,
                        phase=action.phase,
                        waypoint=action.waypoint,
                    )
                )

        def resolve(field_name: str, configured: TaskKeypointConfig) -> int:
            matches = [
                index
                for index, keypoint in enumerate(keypoints)
                if keypoint.matches(configured)
            ]
            if not matches:
                raise ValueError(
                    f"execution.interval_selection.{field_name} is not emitted by "
                    f"{type(self.builder).__name__}: "
                    f"{configured.stage}.{configured.phase.value}"
                    f"[{configured.waypoint}]"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"execution.interval_selection.{field_name} is emitted more than "
                    "once by "
                    f"{type(self.builder).__name__}: "
                    f"{configured.stage}.{configured.phase.value}"
                    f"[{configured.waypoint}]"
                )
            return matches[0]

        if resolve("start", selection.start) > resolve("stop", selection.stop):
            raise ValueError(
                "execution.interval_selection.start must not come after "
                "execution.interval_selection.stop in the active TaskFlowBuilder"
            )

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        with self._viewer_update_context(context):
            return self._reset_impl(mask, context)

    def _reset_impl(
        self,
        mask: np.ndarray,
        context: ExecutionContext,
    ) -> TaskUpdate:
        context.backend.reset(mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._env_states[env_index] = _EnvRuntimeState()
                self._env_states[
                    env_index
                ].latest_details = self._collect_reset_details(env_index, context)
        self._has_reset[mask] = True
        self._public_internal_updates[mask] = 0
        self._last_execution_details = [
            self._execution_details(
                context,
                event="reset",
                internal_updates=0,
            )
            if mask[env_index]
            else (
                dict(self._last_execution_details[env_index])
                if self._last_execution_details[env_index]
                else self._execution_details(
                    context,
                    event="not_selected",
                    internal_updates=0,
                )
            )
            for env_index in range(len(self._env_states))
        ]
        selection = context.task_file.execution.interval_selection
        if selection is not None:
            self._fast_forward_to_interval_start(mask, context, selection)
        # self._set_interest_focus()
        return self._build_task_update()

    def update(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        self._validate_update_mask(mask)
        with self._viewer_update_context(context):
            return self._update_impl(mask, context)

    @staticmethod
    def _viewer_update_context(
        context: ExecutionContext,
    ) -> ContextManager[None]:
        if context.task_file.execution.render_internal_updates:
            return nullcontext()
        return context.backend.defer_viewer_updates()

    def _update_impl(
        self,
        mask: np.ndarray,
        context: ExecutionContext,
    ) -> TaskUpdate:
        execution = context.task_file.execution
        selection = execution.interval_selection
        boundary = execution.update_boundary
        max_updates = int(execution.max_internal_updates_per_update)
        pending = mask.copy()
        internal_updates = np.zeros(len(self._env_states), dtype=np.int64)

        self._last_execution_details = [
            self._execution_details(
                context,
                event=(
                    "not_selected"
                    if not mask[env_index]
                    else "already_done"
                    if self._env_states[env_index].done
                    else "control_tick"
                ),
                internal_updates=0,
            )
            for env_index in range(len(self._env_states))
        ]

        for env_index, state in enumerate(self._env_states):
            if not mask[env_index] or state.done:
                pending[env_index] = False
                continue
            state.reported_keypoint = None

        while bool(np.any(pending)):
            for env_index_value in np.flatnonzero(pending):
                env_index = int(env_index_value)
                state = self._env_states[env_index]
                if internal_updates[env_index] >= max_updates:
                    self._fail_internal_update_limit(
                        env_index,
                        state,
                        context,
                        int(internal_updates[env_index]),
                    )
                    self._last_execution_details[env_index] = self._execution_details(
                        context,
                        event="internal_update_limit_exceeded",
                        internal_updates=int(internal_updates[env_index]),
                    )
                    pending[env_index] = False
                    continue

                event = self._update_env(env_index, state, context)
                if event.control_tick:
                    internal_updates[env_index] += 1
                    self._public_internal_updates[env_index] += 1

                if event.failed:
                    self._last_execution_details[env_index] = self._execution_details(
                        context,
                        event="execution_failed",
                        internal_updates=int(internal_updates[env_index]),
                    )
                    pending[env_index] = False
                    continue

                completed_keypoint = event.completed_keypoint
                if (
                    selection is not None
                    and completed_keypoint is not None
                    and completed_keypoint.matches(selection.stop)
                ):
                    self._finish_interval(state, completed_keypoint, selection)
                    self._last_execution_details[env_index] = self._execution_details(
                        context,
                        event="interval_stop_reached",
                        internal_updates=int(internal_updates[env_index]),
                    )
                    pending[env_index] = False
                    continue

                boundary_event = self._reached_update_boundary(boundary, event)
                if boundary_event is None:
                    if state.done:
                        self._last_execution_details[env_index] = (
                            self._execution_details(
                                context,
                                event="task_succeeded",
                                internal_updates=int(internal_updates[env_index]),
                            )
                        )
                        pending[env_index] = False
                    continue
                if boundary != UpdateBoundary.CONTROL_TICK:
                    state.reported_keypoint = event.completed_position
                self._last_execution_details[env_index] = self._execution_details(
                    context,
                    event=boundary_event,
                    internal_updates=int(internal_updates[env_index]),
                )
                pending[env_index] = False
        # self._set_interest_focus()
        return self._build_task_update()

    @staticmethod
    def _reached_update_boundary(
        boundary: UpdateBoundary,
        event: _EnvUpdateEvent,
    ) -> Optional[str]:
        if boundary == UpdateBoundary.CONTROL_TICK and event.control_tick:
            return "control_tick"
        if boundary == UpdateBoundary.PRIMITIVE and event.primitive_reached:
            return "primitive_reached"
        if boundary == UpdateBoundary.KEYPOINT and event.keypoint_reached:
            return "keypoint_reached"
        if boundary == UpdateBoundary.STAGE and event.stage_succeeded:
            return "stage_succeeded"
        return None

    @staticmethod
    def _execution_details(
        context: ExecutionContext,
        *,
        event: str,
        internal_updates: int,
    ) -> Dict[str, Any]:
        execution = context.task_file.execution
        return {
            "event": event,
            "update_boundary": execution.update_boundary.value,
            "render_internal_updates": bool(execution.render_internal_updates),
            "internal_updates": internal_updates,
            "max_internal_updates_per_update": int(
                execution.max_internal_updates_per_update
            ),
        }

    def _fail_internal_update_limit(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
        internal_updates: int,
    ) -> None:
        max_updates = int(context.task_file.execution.max_internal_updates_per_update)
        details = {
            "event": "internal_update_limit_exceeded",
            "failure_stage": "execution",
            "failure_category": "internal_update_limit_exceeded",
            "failure_reason": (
                "update() did not reach the configured execution.update_boundary "
                f"within {max_updates} internal updates"
            ),
            "internal_updates": internal_updates,
        }
        if state.active is not None:
            self._record_failure(env_index, state.active.plan, details)
        elif state.stage_cursor < len(self._plan):
            self._record_failure(env_index, self._plan[state.stage_cursor], details)
        state.active = None
        state.done = True
        state.success = False
        state.latest_status = StageExecutionStatus.FAILED
        state.latest_details = details
        state.phase = None
        state.phase_step = None
        state.reported_keypoint = None

    def _fast_forward_to_interval_start(
        self,
        mask: np.ndarray,
        context: ExecutionContext,
        selection: IntervalSelectionConfig,
    ) -> None:
        """Advance selected environments through the inclusive start keypoint."""
        pending = np.asarray(mask, dtype=bool).copy()
        ticks = np.zeros(len(self._env_states), dtype=np.int64)
        reached = np.zeros(len(self._env_states), dtype=bool)
        reached_keypoints: List[Optional[_ResolvedTaskKeypoint]] = [
            None for _ in self._env_states
        ]
        max_updates = int(selection.max_fast_forward_updates)
        reset_details = [dict(state.latest_details) for state in self._env_states]
        record_start = len(self._records)

        while bool(np.any(pending)):
            for env_index in np.flatnonzero(pending):
                index = int(env_index)
                state = self._env_states[index]
                if ticks[index] >= max_updates:
                    self._fail_interval_fast_forward(
                        index,
                        state,
                        context,
                        int(ticks[index]),
                        max_updates=max_updates,
                    )
                    pending[index] = False
                    continue

                event = self._update_env(index, state, context)
                if event.control_tick:
                    ticks[index] += 1
                if event.failed:
                    pending[index] = False
                    continue

                completed = event.completed_keypoint
                if completed is not None and completed.matches(selection.start):
                    reached[index] = True
                    reached_keypoints[index] = completed
                    pending[index] = False
                    continue
                if state.done:
                    if state.success:
                        self._fail_interval_fast_forward(
                            index,
                            state,
                            context,
                            int(ticks[index]),
                            max_updates=max_updates,
                            failure_category="interval_start_not_reached",
                            failure_reason=(
                                "task completed before interval_selection.start "
                                "was reached"
                            ),
                        )
                    pending[index] = False

        # Prefix execution prepares physical state but is outside the selected
        # rollout. Never expose its successful stage records, including when a
        # later prefix action fails; preserve only the actual failure record.
        prefix_records = self._records[record_start:]
        self._records = self._records[:record_start] + [
            record
            for record in prefix_records
            if record.status == StageExecutionStatus.FAILED
        ]

        for env_index in np.flatnonzero(mask):
            index = int(env_index)
            state = self._env_states[index]
            interval_details = {
                "event": "interval_start_reached"
                if reached[index]
                else "interval_fast_forward_failed",
                "start": selection.start.model_dump(mode="json"),
                "stop": selection.stop.model_dump(mode="json"),
                "fast_forward_updates": int(ticks[index]),
                "max_fast_forward_updates": max_updates,
            }
            self._last_execution_details[index] = self._execution_details(
                context,
                event=(
                    "interval_start_reached"
                    if reached[index]
                    else "interval_fast_forward_failed"
                ),
                internal_updates=0,
            )
            if not reached[index]:
                state.latest_details = {
                    **reset_details[index],
                    **state.latest_details,
                    "interval_selection": interval_details,
                }
                continue

            keypoint = reached_keypoints[index]
            assert keypoint is not None
            state.done = False
            state.success = False
            state.latest_status = StageExecutionStatus.PENDING
            state.latest_details = {
                **reset_details[index],
                **state.latest_details,
                "interval_selection": interval_details,
            }
            state.reported_keypoint = keypoint
            state.phase = keypoint.phase.value
            state.phase_step = keypoint.waypoint
            if keypoint.matches(selection.stop):
                self._finish_interval(state, keypoint, selection)
                self._last_execution_details[index] = self._execution_details(
                    context,
                    event="interval_stop_reached",
                    internal_updates=0,
                )

    def _fail_interval_fast_forward(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
        ticks: int,
        *,
        max_updates: int,
        failure_category: str = "interval_fast_forward_timeout",
        failure_reason: Optional[str] = None,
    ) -> None:
        details = {
            "event": failure_category,
            "failure_stage": "reset",
            "failure_category": failure_category,
            "failure_reason": failure_reason
            or (
                "reset() did not reach interval_selection.start within "
                f"{max_updates} internal updates"
            ),
            "fast_forward_updates": ticks,
        }
        if state.active is not None:
            self._record_failure(env_index, state.active.plan, details)
        elif state.stage_cursor < len(self._plan):
            self._record_failure(env_index, self._plan[state.stage_cursor], details)
        state.active = None
        state.done = True
        state.success = False
        state.latest_status = StageExecutionStatus.FAILED
        state.latest_details = details
        state.phase = None
        state.phase_step = None
        state.reported_keypoint = None

    @staticmethod
    def _finish_interval(
        state: _EnvRuntimeState,
        keypoint: _ResolvedTaskKeypoint,
        selection: IntervalSelectionConfig,
    ) -> None:
        previous_interval_details = state.latest_details.get("interval_selection", {})
        state.done = True
        state.success = True
        state.latest_status = StageExecutionStatus.SUCCEEDED
        state.latest_details = {
            **state.latest_details,
            "interval_selection": {
                **previous_interval_details,
                "event": "interval_stop_reached",
                "start": selection.start.model_dump(mode="json"),
                "stop": selection.stop.model_dump(mode="json"),
            },
        }
        state.phase = keypoint.phase.value
        state.phase_step = keypoint.waypoint
        state.reported_keypoint = keypoint

    def get_env(self) -> EnvProtocol:
        """Return the underlying environment object managed by this runner."""
        return self._require_context().backend.env

    def close(self) -> None:
        if self._context is None:
            return
        self._context.backend.teardown()
        self._context = None
        self._plan = []
        self._records = []
        self._env_states = []
        self._has_reset = np.zeros(0, dtype=bool)
        self._public_internal_updates = np.zeros(0, dtype=np.int64)
        self._last_execution_details = []

    def _update_env(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
    ) -> _EnvUpdateEvent:
        if state.stage_cursor >= len(self._plan):
            state.done = True
            state.success = True
            state.latest_status = StageExecutionStatus.SUCCEEDED
            state.phase = None
            state.phase_step = None
            return _EnvUpdateEvent()

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
                return _EnvUpdateEvent(failed=True)
            state.active = self._start_stage(env_index, context, plan)
            state.latest_status = StageExecutionStatus.RUNNING

        assert state.active is not None
        active = state.active
        action = active.actions[active.action_index]
        mask = self._mask_for_env(env_index)
        result = TaskRunner._run_stage_action(
            env_index=env_index,
            plan=active.plan,
            action=action,
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
        if (
            action.kind == "pose"
            and action.pose is not None
            and action.pose.arc is not None
        ):
            arc_cfg = action.pose.arc
            arc_info: Dict[str, Any] = {
                "pivot": arc_cfg.pivot
                if isinstance(arc_cfg.pivot, str)
                else [float(v) for v in arc_cfg.pivot],
                "axis": [float(v) for v in arc_cfg.axis],
                "angle": float(arc_cfg.angle),
                "absolute": bool(arc_cfg.absolute),
            }
            if arc_cfg.absolute and isinstance(arc_cfg.pivot, str):
                try:
                    current_joint = float(
                        context.backend.get_joint_angle(arc_cfg.pivot, env_index)
                    )
                    arc_info["current_joint_angle"] = current_joint
                    arc_info["target_joint_angle"] = float(arc_cfg.angle)
                    arc_info["delta_joint_angle"] = float(arc_cfg.angle) - current_joint
                except (KeyError, NotImplementedError):
                    pass
            elif action.arc_cumulative_angle is not None:
                arc_info["cumulative_angle"] = float(action.arc_cumulative_angle)
            details["action"] = "arc"
            details["arc"] = arc_info

        if signal == ControlSignal.RUNNING:
            phase, phase_step = self._action_phase(
                active.actions,
                active.action_index,
                use_configured_identity=(
                    context.task_file.execution.interval_selection is not None
                    or context.task_file.execution.update_boundary
                    != UpdateBoundary.CONTROL_TICK
                ),
            )
            state.latest_status = StageExecutionStatus.RUNNING
            state.latest_details = details
            state.phase = phase
            state.phase_step = phase_step
            return _EnvUpdateEvent(control_tick=True)

        if signal == ControlSignal.REACHED:
            completed_action = action
            completed_position = (
                _ResolvedTaskKeypoint(
                    stage_index=active.plan.stage_index,
                    stage_name=active.plan.stage_name,
                    phase=completed_action.phase,
                    waypoint=completed_action.waypoint,
                )
                if isinstance(completed_action.phase, TaskPhase)
                and isinstance(completed_action.waypoint, int)
                else None
            )
            completed_keypoint = (
                completed_position if completed_action.completes_keypoint else None
            )
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
                return _EnvUpdateEvent(
                    control_tick=True,
                    primitive_reached=True,
                    keypoint_reached=completed_keypoint is not None,
                    failed=True,
                    completed_position=completed_position,
                    completed_keypoint=completed_keypoint,
                )

            if active.action_index < len(active.actions):
                phase, phase_step = self._action_phase(
                    active.actions,
                    active.action_index,
                    use_configured_identity=(
                        context.task_file.execution.interval_selection is not None
                        or context.task_file.execution.update_boundary
                        != UpdateBoundary.CONTROL_TICK
                    ),
                )
                state.latest_status = StageExecutionStatus.RUNNING
                state.latest_details = details
                state.phase = phase
                state.phase_step = phase_step
                return _EnvUpdateEvent(
                    control_tick=True,
                    primitive_reached=True,
                    keypoint_reached=completed_keypoint is not None,
                    completed_position=completed_position,
                    completed_keypoint=completed_keypoint,
                )

            # Resolve target object pose for PLACED condition
            target_object_pose: Optional[PoseState] = None
            held_name: Optional[str] = active.held_object_name
            if op == Operation.PLACE:
                control = active.plan.stage.param
                ref = getattr(control, "placed_reference", "object")
                target_obj_name = active.plan.stage.object
                if ref == "object" and target_obj_name:
                    target_handler = context.backend.get_object_handler(target_obj_name)
                    if target_handler is not None:
                        target_object_pose = target_handler.get_pose().select(env_index)
                else:
                    target_object_pose = self._pre_move_end_pose(active)

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
                    target_object_pose=target_object_pose,
                    held_object_name=held_name,
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
                return _EnvUpdateEvent(
                    control_tick=True,
                    primitive_reached=True,
                    keypoint_reached=completed_keypoint is not None,
                    failed=True,
                    completed_position=completed_position,
                    completed_keypoint=completed_keypoint,
                )

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
                state.success = False
            return _EnvUpdateEvent(
                control_tick=True,
                primitive_reached=True,
                keypoint_reached=completed_keypoint is not None,
                stage_succeeded=True,
                completed_position=completed_position,
                completed_keypoint=completed_keypoint,
            )

        failure = self._build_action_failure_details(active.plan, details, signal)
        self._record_failure(env_index, active.plan, failure)
        state.active = None
        state.done = True
        state.success = False
        state.latest_status = StageExecutionStatus.FAILED
        state.latest_details = failure
        state.phase = None
        state.phase_step = None
        return _EnvUpdateEvent(control_tick=True, failed=True)

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
        held_object_name: Optional[str] = None
        if plan.stage.operation == Operation.PLACE:
            held_object_name = self._find_grasped_object(
                context.backend, plan.operator_name, env_index
            )
        actions = TaskRunner._build_stage_actions(plan, self.builder, context)
        return ActiveStageState(
            plan=plan,
            operator=operator,
            target=target,
            actions=actions,
            initial_object_pose=initial_object_pose,
            held_object_name=held_object_name,
        )

    @staticmethod
    def _find_grasped_object(
        backend: SceneBackend, operator_name: str, env_index: int
    ) -> Optional[str]:
        """Return the name of the object currently grasped by the operator."""
        handlers = getattr(backend, "object_handlers", {})
        for name in handlers:
            if backend.is_object_grasped(operator_name, name)[env_index]:
                return name
        return None

    @staticmethod
    def _apply_waypoint_randomization(
        actions: List[PrimitiveAction],
        context: ExecutionContext,
    ) -> None:
        """Apply per-waypoint randomization to pose actions in-place.

        Supports ``relative`` mode (sampled values are added to the
        waypoint's existing position/orientation — the default) and
        ``absolute_world`` mode (sampled values replace the waypoint's
        position/orientation entirely). A ``None`` axis is skipped and
        keeps the waypoint's original value in either mode.

        ``absolute_base`` is **not** supported for per-waypoint randomization
        because the waypoint already carries its own ``reference`` field
        (e.g. ``BASE``, ``OBJECT_WORLD``) which selects the frame in which
        the sampled numbers are interpreted by the pose controller. To
        randomize in the base frame, set ``PoseControlConfig.reference =
        BASE`` and use ``absolute_world`` (or ``relative``) in the
        waypoint's ``randomization``.
        """
        rng = getattr(context.backend, "_rng", None)
        if rng is None:
            rng = np.random.default_rng()
        for action in actions:
            if action.kind != "pose" or action.pose is None:
                continue
            rand = action.pose.randomization
            if rand is None:
                continue
            if rand.reference == RandomizationReference.ABSOLUTE_BASE:
                raise ValueError(
                    "Per-waypoint randomization does not support "
                    "'absolute_base'. Set the waypoint's own `reference` "
                    "field (e.g. BASE) and use 'absolute_world' or "
                    "'relative' instead."
                )
            if not isinstance(rand.reference, RandomizationReference):
                raise ValueError(
                    f"Per-waypoint randomization does not support "
                    f"entity-name references (got "
                    f"reference={rand.reference!r}). Use 'relative' or "
                    f"'absolute_world' instead."
                )
            is_absolute = rand.reference == RandomizationReference.ABSOLUTE_WORLD
            pos_ranges = (rand.x, rand.y, rand.z)
            rot_ranges = (rand.roll, rand.pitch, rand.yaw)
            pos = list(action.pose.position)
            for axis, rng_pair in enumerate(pos_ranges):
                if rng_pair is None:
                    continue
                sampled = float(rng.uniform(*rng_pair))
                if is_absolute:
                    pos[axis] = sampled
                else:
                    pos[axis] += sampled
            action.pose = action.pose.model_copy(
                update={"position": tuple(pos), "randomization": None}
            )
            if any(r is not None for r in rot_ranges):
                ori = action.pose.orientation
                if ori and len(ori) == 4:
                    from .utils.pose import quaternion_to_rpy

                    r0, p0, y0 = quaternion_to_rpy(np.asarray(ori))
                    if is_absolute:
                        r_val = (
                            r0
                            if rot_ranges[0] is None
                            else float(rng.uniform(*rot_ranges[0]))
                        )
                        p_val = (
                            p0
                            if rot_ranges[1] is None
                            else float(rng.uniform(*rot_ranges[1]))
                        )
                        y_val = (
                            y0
                            if rot_ranges[2] is None
                            else float(rng.uniform(*rot_ranges[2]))
                        )
                    else:
                        r_val = r0 + (
                            0.0
                            if rot_ranges[0] is None
                            else float(rng.uniform(*rot_ranges[0]))
                        )
                        p_val = p0 + (
                            0.0
                            if rot_ranges[1] is None
                            else float(rng.uniform(*rot_ranges[1]))
                        )
                        y_val = y0 + (
                            0.0
                            if rot_ranges[2] is None
                            else float(rng.uniform(*rot_ranges[2]))
                        )
                    new_ori = euler_to_quaternion((r_val, p_val, y_val))
                    action.pose = action.pose.model_copy(
                        update={"orientation": tuple(float(v) for v in new_ori)}
                    )

    @staticmethod
    def _check_stage_condition(
        env_index: int,
        context: ExecutionContext,
        plan: StageExecutionPlan,
        condition_type: OperationConditionType,
        initial_pose: Optional[PoseState] = None,
        completion_pose: Optional[PoseControlConfig] = None,
        target_object_pose: Optional[PoseState] = None,
        held_object_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        return _check_stage_condition(
            env_index=env_index,
            context=context,
            plan=plan,
            condition_type=condition_type,
            initial_pose=initial_pose,
            completion_pose=completion_pose,
            target_object_pose=target_object_pose,
            held_object_name=held_object_name,
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
    def _pre_move_end_pose(active: ActiveStageState) -> Optional[PoseState]:
        """Return the resolved pose of the last pre_move waypoint (before the
        first eef action) as a single-batch PoseState."""
        last_pose = None
        for action in active.actions:
            if action.kind == "eef":
                break
            if action.kind == "pose" and action.resolved_pose is not None:
                last_pose = action.resolved_pose
        if last_pose is None:
            return None
        return PoseState(
            position=np.asarray(last_pose.position, dtype=np.float64).reshape(1, 3),
            orientation=np.asarray(last_pose.orientation, dtype=np.float64).reshape(
                1, 4
            ),
        )

    @staticmethod
    def _build_stage_actions(
        plan: StageExecutionPlan,
        builder: "TaskFlowBuilder",
        context: ExecutionContext,
    ) -> List[PrimitiveAction]:
        """Build the primitive-action list for a stage.

        Single source of truth shared by ``TaskRunner._start_stage`` and
        ``ConfigDrivenDemoPolicy._get_stage_actions`` so the two execution
        paths cannot drift on what gets handed to the controller (deepcopy +
        per-waypoint randomization).
        """
        actions = deepcopy(
            builder.build_actions(plan.stage, plan.last_orientation_before)[0]
        )
        TaskRunner._apply_waypoint_randomization(actions, context)
        return actions

    @staticmethod
    def _run_stage_action(
        env_index: int,
        plan: StageExecutionPlan,
        action: PrimitiveAction,
        backend: SceneBackend,
        env_mask: np.ndarray,
    ) -> ControlResult:
        """Run one primitive action with operator/target/site resolved from the plan.

        Single source of truth for invoking ``_run_action``. Used by
        ``TaskRunner._update_env`` and ``ConfigDrivenDemoPolicy.action_applier``
        so callers cannot forget to forward fields like ``reference_site``.
        """
        operator = backend.get_operator_handler(plan.operator_name)
        target = backend.get_object_handler(plan.stage.object)
        return TaskRunner._run_action(
            env_index=env_index,
            operator=operator,
            action=action,
            target=target,
            backend=backend,
            env_mask=env_mask,
            reference_site=plan.stage.site,
        )

    @staticmethod
    def _run_action(
        env_index: int,
        operator: OperatorHandler,
        action: PrimitiveAction,
        target: Optional[ObjectHandler],
        backend: SceneBackend,
        env_mask: np.ndarray,
        reference_site: Optional[str] = None,
    ) -> ControlResult:
        if action.kind == "pose" and action.pose is not None:
            is_arc = action.pose.arc is not None
            is_snapshot = (
                action.pose.reference
                in {
                    PoseReference.EEF_WORLD,
                    PoseReference.EEF,
                }
                or (is_arc and not action.pose.arc.absolute)
                or action.pose.static
            )
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
                    reference_site=reference_site,
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
        reference_site: Optional[str] = None,
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
                    reference_site=reference_site,
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
                reference_site=reference_site,
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
        reference_site: Optional[str] = None,
    ) -> PoseControlConfig:
        if pose.arc is not None:
            return TaskRunner._resolve_arc_command(
                env_index, operator, pose, target, backend, action, reference_site
            )
        reference_pose = TaskRunner._resolve_reference_pose(
            env_index=env_index,
            operator=operator,
            pose=pose,
            target=target,
            reference_site=reference_site,
            backend=backend,
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
            tolerance=pose.tolerance,
        )

    @staticmethod
    def _resolve_reference_pose(
        env_index: int,
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        reference_site: Optional[str] = None,
        backend: Optional[SceneBackend] = None,
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
            if reference_site is not None and backend is not None:
                site_pose = backend.get_element_pose(reference_site, env_index)
                return PoseState(position=site_pose.position[0])
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
            if reference_site is not None and backend is not None:
                return backend.get_element_pose(reference_site, env_index)
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
        reference_site: Optional[str] = None,
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
            reference_site=reference_site,
            backend=backend,
        )
        pivot_local = PoseState(position=arc.pivot)
        composed = compose_pose(reference_pose, pivot_local)
        return tuple(float(v) for v in composed.position[0])

    @staticmethod
    def _pose_config_to_local_pose(pose: PoseControlConfig) -> PoseState:
        return pose_config_to_pose_state(pose)

    @staticmethod
    def _action_phase(
        actions: List[PrimitiveAction],
        action_index: int,
        *,
        use_configured_identity: bool = False,
    ) -> tuple[str, Optional[int]]:
        action = actions[action_index]
        if (
            use_configured_identity
            and isinstance(action.phase, TaskPhase)
            and isinstance(action.waypoint, int)
        ):
            return action.phase.value, action.waypoint

        # Compatibility fallback for PrimitiveAction instances constructed
        # outside TaskFlowBuilder without configured keypoint metadata.
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
        selection = (
            self._context.task_file.execution.interval_selection
            if self._context is not None
            else None
        )
        stage_index: List[int] = []
        stage_name: List[str] = []
        status: List[StageExecutionStatus] = []
        done: List[bool] = []
        success: List[bool] = []
        details: List[Dict[str, Any]] = []
        phase: List[Optional[str]] = []
        phase_step: List[int] = []
        for env_index, state in enumerate(self._env_states):
            if state.reported_keypoint is not None:
                stage_index.append(state.reported_keypoint.stage_index)
                stage_name.append(state.reported_keypoint.stage_name)
            elif state.active is not None:
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
            state_details = dict(state.latest_details)
            if selection is not None:
                interval_details = dict(state_details.get("interval_selection", {}))
                interval_details.setdefault(
                    "event",
                    (
                        "interval_pending"
                        if not self._has_reset[env_index]
                        else (
                            "interval_failed"
                            if state.done and not state.success
                            else "interval_running"
                        )
                    ),
                )
                interval_details.setdefault(
                    "start", selection.start.model_dump(mode="json")
                )
                interval_details.setdefault(
                    "stop", selection.stop.model_dump(mode="json")
                )
                interval_details.setdefault(
                    "max_fast_forward_updates",
                    int(selection.max_fast_forward_updates),
                )
                state_details["interval_selection"] = interval_details
            if env_index < len(self._last_execution_details):
                state_details["execution"] = dict(
                    self._last_execution_details[env_index]
                )
            details.append(state_details)
            if state.reported_keypoint is not None:
                phase.append(state.reported_keypoint.phase.value)
                phase_step.append(state.reported_keypoint.waypoint)
            else:
                phase.append(state.phase)
                phase_step.append(-1 if state.phase_step is None else state.phase_step)
        return TaskUpdate(
            stage_index=np.asarray(stage_index, dtype=np.int64),
            stage_name=stage_name,
            status=np.asarray(status, dtype=object),
            done=np.asarray(done, dtype=bool),
            success=np.asarray(success, dtype=bool),
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

        # Collect camera poses if camera randomization is configured.
        cam_rand = getattr(context.backend, "camera_randomization", {})
        if cam_rand:
            camera_poses: Dict[str, Any] = {}
            get_cam_pose = getattr(context.backend, "_get_camera_pose", None)
            if get_cam_pose is not None:
                for cam_name in cam_rand:
                    try:
                        pose = get_cam_pose(cam_name).select(env_index)
                        camera_poses[cam_name] = self._serialize_pose(pose)
                    except (KeyError, AttributeError):
                        continue
            if camera_poses:
                initial_poses["_cameras"] = camera_poses

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
    target_object_pose: Optional[PoseState] = None,
    held_object_name: Optional[str] = None,
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
        threshold = getattr(plan.stage.param, "displacement_threshold", None)
        kwargs = {"threshold": float(threshold)} if threshold is not None else {}
        satisfied = (
            bool(
                backend.is_object_displaced(object_name, initial_pose, **kwargs)[
                    env_index
                ]
            )
            if initial_pose is not None and object_name
            else True
        )
    elif constraint == OperationConstraint.REACHED:
        operator = backend.get_operator_handler(operator_name)
        # Resolve effective tolerance: per-waypoint override > operator default
        wp_tol = (
            getattr(completion_pose, "tolerance", None) if completion_pose else None
        )
        op_tol = getattr(getattr(operator, "control", None), "tolerance", None)
        eff_pos_tol = (
            wp_tol.position
            if wp_tol is not None and wp_tol.position is not None
            else getattr(op_tol, "position", 0.01)
        )
        eff_ori_tol = (
            wp_tol.orientation
            if wp_tol is not None and wp_tol.orientation is not None
            else getattr(op_tol, "orientation", 0.08)
        )
        if completion_pose is None:
            satisfied = False
        else:
            current_pose = operator.get_end_effector_pose().select(env_index)
            pos_diff = np.asarray(
                current_pose.position[0], dtype=np.float64
            ) - np.asarray(completion_pose.position, dtype=np.float64)
            orientation_error = float(
                quaternion_angular_distance(
                    current_pose.orientation[0],
                    np.asarray(completion_pose.orientation, dtype=np.float64),
                )
            )
            satisfied = position_within_tolerance(
                pos_diff, eff_pos_tol
            ) and orientation_error <= float(eff_ori_tol)
    elif constraint == OperationConstraint.PLACED:
        if is_grasping:
            satisfied = False
        elif target_object_pose is None or not held_object_name:
            # No target info or no held object recorded — fall back to released
            satisfied = True
        else:
            handler = backend.get_object_handler(held_object_name)
            if handler is None:
                satisfied = True
            else:
                current = handler.get_pose()
                pos_diff = np.asarray(
                    current.position[env_index], dtype=np.float64
                ) - np.asarray(target_object_pose.position[0], dtype=np.float64)
                # Resolve tolerance: stage > operator > default
                control = plan.stage.param
                stage_pt: Optional[PlacedToleranceConfig] = getattr(
                    control, "placed_tolerance", None
                )
                op_tol = getattr(
                    getattr(
                        backend.get_operator_handler(operator_name), "control", None
                    ),
                    "tolerance",
                    None,
                )
                op_placed: Optional[PlacedToleranceConfig] = getattr(
                    op_tol, "placed", None
                )

                def _is_configured(val):
                    """True when val is an explicit tolerance, not all-None."""
                    if val is None:
                        return False
                    if isinstance(val, (list, np.ndarray)):
                        return any(v is not None for v in val)
                    return True

                stage_pos = stage_pt.position if stage_pt is not None else None
                stage_ori = stage_pt.orientation if stage_pt is not None else None
                op_pos = op_placed.position if op_placed is not None else None
                op_ori = op_placed.orientation if op_placed is not None else None
                # Resolution chain: stage > operator > None (no constraint).
                # All-None (or unset) means "don't check that dimension".
                eff_pos = (
                    stage_pos
                    if _is_configured(stage_pos)
                    else op_pos
                    if _is_configured(op_pos)
                    else None
                )
                eff_ori = (
                    stage_ori
                    if _is_configured(stage_ori)
                    else op_ori
                    if _is_configured(op_ori)
                    else None
                )
                pos_ok = position_within_tolerance_nullable(pos_diff, eff_pos)
                ori_ok = orientation_within_tolerance_nullable(
                    current.orientation[env_index],
                    target_object_pose.orientation[0],
                    eff_ori,
                )
                satisfied = pos_ok and ori_ok
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
        OperationConstraint.PLACED: (
            "placement_failed",
            "held object is not within tolerance of the target position",
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
    elif constraint == OperationConstraint.PLACED:
        details["held_object"] = held_object_name or ""
        details["placed_reference"] = getattr(
            plan.stage.param, "placed_reference", "object"
        )
        if held_object_name and target_object_pose is not None:
            handler = backend.get_object_handler(held_object_name)
            if handler is not None:
                current = handler.get_pose()
                details["target_position"] = [
                    float(v) for v in target_object_pose.position[0]
                ]
                details["current_position"] = [
                    float(v) for v in current.position[env_index]
                ]
                details["position_error"] = float(
                    np.linalg.norm(
                        np.asarray(current.position[env_index], dtype=np.float64)
                        - np.asarray(target_object_pose.position[0], dtype=np.float64)
                    )
                )
                details["target_orientation"] = [
                    float(v) for v in target_object_pose.orientation[0]
                ]
                details["current_orientation"] = [
                    float(v) for v in current.orientation[env_index]
                ]
                details["orientation_error"] = float(
                    quaternion_angular_distance(
                        current.orientation[env_index],
                        target_object_pose.orientation[0],
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
    elapsed_time_sec: float = 0.0,
    sim_time_sec: float = 0.0,
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
        elapsed_time_sec=elapsed_time_sec,
        sim_time_sec=sim_time_sec,
        completed_stage_count=completed_stage_count,
        final_stage_index=np.asarray(update.stage_index, dtype=np.int64),
        final_stage_name=list(update.stage_name),
        final_status=np.asarray(update.status, dtype=object),
        final_done=np.asarray(update.done, dtype=bool),
        final_success=np.asarray(update.success, dtype=bool),
        records=list(records),
    )


def _resolve_policy_completion_pose(
    *,
    env_index: int,
    operator: OperatorHandler,
    target: Optional[ObjectHandler],
    backend: SceneBackend,
    action: PrimitiveAction,
    reference_site: Optional[str] = None,
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
        reference_site=reference_site,
    )
