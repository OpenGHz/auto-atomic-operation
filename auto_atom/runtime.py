"""YAML-driven batch-first task runner built from primitive controls."""

from __future__ import annotations

import inspect
import logging
import math
import operator
import weakref
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    ContextManager,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np

from .framework import (
    ArcControlConfig,
    AutoAtomConfig,
    EefControlConfig,
    IntervalSelectionConfig,
    KeypointSide,
    Operation,
    Orientation,
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
)
from .utils.pose import (
    PoseState,
    compose_pose,
    euler_to_quaternion,
    inverse_pose,
    pose_config_to_pose_state,
    rotate_pose_around_axis,
)

if TYPE_CHECKING:
    from .execution_timeline import ExecutionTimeline
    from .stage_execution import StageExecution

logger = logging.getLogger(__name__)


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
class ObjectHandler(ABC):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"ObjectHandler.name must be a non-empty string; got {self.name!r}."
            )

    @abstractmethod
    def get_pose(self) -> PoseState:
        """Return the object's batched world pose."""

    @abstractmethod
    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> None:
        """Set the object's batched world pose for selected environments."""


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
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the end-effector toward the desired state for selected envs."""

    @abstractmethod
    def get_end_effector_pose(self) -> PoseState:
        """Return batched world poses for the operator end-effector."""

    @abstractmethod
    def get_base_pose(self) -> PoseState:
        """Return batched world poses for the operator base."""

    def get_reached_tolerances(self) -> tuple[Any, Any]:
        """Return default position and orientation tolerances for REACHED.

        Backends whose operator configuration exposes different tolerances
        should override this method.  Keeping this on the public handler
        contract prevents Stage execution from depending on a backend's
        private configuration object.
        """
        return 0.01, 0.08

    def get_placed_tolerances(self) -> tuple[Any, Any]:
        """Return default position and orientation tolerances for PLACED.

        ``None`` means that the corresponding placement component is not
        constrained unless the Stage supplies an explicit tolerance.
        """
        return None, None

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError


@runtime_checkable
class EnvProtocol(Protocol):
    """Core batched environment interface returned by ``SceneBackend.get_env()``.

    Environment features such as stepping, observations, and simulation-loop
    updates are optional capabilities represented by the narrower protocols
    below. Callers should request only the capability they actually use.
    """

    @property
    def batch_size(self) -> int:
        """Number of environments represented by this object."""
        ...


@runtime_checkable
class StepEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for applying batched policy actions."""

    def step(
        self,
        action: np.ndarray,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
    ) -> None: ...


@runtime_checkable
class ObservationEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for capturing policy observations."""

    def capture_observation(self) -> Dict[str, Dict[str, Any]]: ...


@runtime_checkable
class JointActionEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for directly applying operator joint actions."""

    def apply_joint_action(
        self,
        operator: str,
        action: Any,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
        kinematic: bool = False,
    ) -> None: ...


@runtime_checkable
class PoseActionEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for directly applying operator pose actions."""

    def apply_pose_action(
        self,
        operator: str,
        position: Any,
        orientation: Any,
        gripper: Any = None,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
    ) -> None: ...


@runtime_checkable
class KinematicPoseActionEnvProtocol(PoseActionEnvProtocol, Protocol):
    """Pose-action capability that also supports kinematic application."""

    def apply_pose_action(
        self,
        operator: str,
        position: Any,
        orientation: Any,
        gripper: Any = None,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
        kinematic: bool = False,
    ) -> None: ...


@runtime_checkable
class SimulationLoopEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for independently advancing simulation state."""

    def update(self) -> None: ...


@runtime_checkable
class InfoEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for returning serializable metadata."""

    def get_info(self) -> Dict[str, Any]: ...


_EnvCapabilityT = TypeVar("_EnvCapabilityT")
_ENV_CAPABILITY_CACHE: Dict[
    tuple[int, type[Any]],
    weakref.ReferenceType[object],
] = {}


def require_env_capability(
    env: object,
    capability: type[_EnvCapabilityT],
    *,
    feature: str,
    expected_batch_size: Optional[int] = None,
) -> _EnvCapabilityT:
    """Return *env* narrowed to *capability* or raise a clear runtime error.

    A successful structural/signature check is cached for the lifetime of a
    weak-referenceable environment. Environment capability methods are
    therefore expected to remain stable after construction; batch-size values
    are still checked on every call.
    """
    cached = _is_environment_capability_cached(env, capability)
    if not cached:
        if not (
            getattr(capability, "_is_protocol", False)
            and getattr(capability, "_is_runtime_protocol", False)
        ):
            raise TypeError(
                f"{capability!r} is not a runtime-checkable environment protocol."
            )
        required_members = _environment_protocol_members(capability)
        missing_members = []
        for member in required_members:
            try:
                inspect.getattr_static(env, member)
            except AttributeError:
                missing_members.append(member)
        if missing_members:
            missing = (
                f" Missing attributes: {', '.join(missing_members)}."
                if missing_members
                else ""
            )
            raise RuntimeError(
                f"{feature} requires environment capability {capability.__name__}; "
                f"got {type(env).__name__}.{missing}"
            )

    narrowed = cast(_EnvCapabilityT, env)
    try:
        environment_batch_size = getattr(narrowed, "batch_size")
    except Exception as exc:
        raise RuntimeError(
            f"{feature} requires a readable environment batch_size; "
            f"{type(env).__name__}.batch_size raised "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    actual_batch_size = _require_positive_batch_size(
        environment_batch_size,
        owner="environment",
        feature=feature,
    )
    normalized_expected_batch_size = (
        None
        if expected_batch_size is None
        else _require_positive_batch_size(
            expected_batch_size,
            owner="backend",
            feature=feature,
        )
    )
    if (
        normalized_expected_batch_size is not None
        and actual_batch_size != normalized_expected_batch_size
    ):
        raise RuntimeError(
            f"{feature} requires environment batch_size to match "
            f"backend.batch_size; got {actual_batch_size} and "
            f"{normalized_expected_batch_size}."
        )
    if not cached:
        _validate_environment_protocol_signatures(
            narrowed,
            capability,
            feature=feature,
        )
        _cache_environment_capability(env, capability)
    return narrowed


def _is_environment_capability_cached(
    env: object,
    capability: type[Any],
) -> bool:
    reference = _ENV_CAPABILITY_CACHE.get((id(env), capability))
    return reference is not None and reference() is env


def _cache_environment_capability(
    env: object,
    capability: type[Any],
) -> None:
    key = (id(env), capability)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        if _ENV_CAPABILITY_CACHE.get(key) is reference:
            _ENV_CAPABILITY_CACHE.pop(key, None)

    try:
        reference = weakref.ref(env, discard)
    except TypeError:
        return
    _ENV_CAPABILITY_CACHE[key] = reference


def _require_positive_batch_size(
    value: object,
    *,
    owner: str,
    feature: str,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(
            f"{feature} requires {owner} batch_size to be an integer; got {value!r}."
        )
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise RuntimeError(
            f"{feature} requires {owner} batch_size to be an integer; got {value!r}."
        ) from exc
    if normalized <= 0:
        raise RuntimeError(
            f"{feature} requires {owner} batch_size to be positive; got {normalized}."
        )
    return normalized


def _environment_protocol_members(capability: type[Any]) -> List[str]:
    members: set[str] = set()
    for base in capability.__mro__:
        members.update(
            name
            for name in getattr(base, "__annotations__", {})
            if not name.startswith("_")
        )
        members.update(
            name
            for name, value in vars(base).items()
            if not name.startswith("_")
            and (callable(value) or isinstance(value, property))
        )
    return sorted(members)


def _validate_environment_protocol_signatures(
    env: object,
    capability: type[Any],
    *,
    feature: str,
) -> None:
    checked: set[str] = set()
    for base in capability.__mro__:
        for member_name, expected_member in vars(base).items():
            if (
                member_name.startswith("_")
                or member_name in checked
                or not callable(expected_member)
            ):
                continue
            checked.add(member_name)
            try:
                actual_member = getattr(env, member_name)
            except Exception as exc:
                raise RuntimeError(
                    f"{feature} requires readable "
                    f"{capability.__name__}.{member_name}; "
                    f"{type(env).__name__}.{member_name} raised "
                    f"{type(exc).__name__}: {exc}."
                ) from exc
            if not callable(actual_member):
                raise RuntimeError(
                    f"{feature} requires callable "
                    f"{capability.__name__}.{member_name}; got "
                    f"{type(env).__name__}.{member_name}="
                    f"{actual_member!r}."
                )
            callable_implementation = getattr(actual_member, "__call__", None)
            if any(
                inspect.iscoroutinefunction(candidate)
                or inspect.isasyncgenfunction(candidate)
                or inspect.isgeneratorfunction(candidate)
                for candidate in (actual_member, callable_implementation)
                if candidate is not None
            ):
                raise RuntimeError(
                    f"{feature} requires synchronous "
                    f"{capability.__name__}.{member_name}; got "
                    f"{type(env).__name__}.{member_name}."
                )
            expected_signature = inspect.signature(expected_member)
            try:
                actual_signature = inspect.signature(actual_member)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{feature} requires introspectable "
                    f"{capability.__name__}.{member_name}; wrap "
                    f"{type(env).__name__}.{member_name} in a Python method "
                    "with an explicit signature."
                ) from exc

            for args, kwargs in _protocol_signature_calls(expected_signature):
                try:
                    actual_signature.bind(*args, **kwargs)
                except TypeError as exc:
                    raise RuntimeError(
                        f"{feature} requires {capability.__name__}.{member_name} "
                        f"with a signature compatible with {expected_signature}; "
                        f"got {type(env).__name__}.{member_name}{actual_signature}."
                    ) from exc


def _protocol_signature_calls(
    signature: inspect.Signature,
) -> List[tuple[List[object], Dict[str, object]]]:
    sentinel = object()
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]

    minimal_args: List[object] = []
    minimal_kwargs: Dict[str, object] = {}
    full_args: List[object] = []
    full_kwargs: Dict[str, object] = {}
    for parameter in parameters:
        required = parameter.default is inspect.Parameter.empty
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if required:
                minimal_args.append(sentinel)
            full_args.append(sentinel)
        elif parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if required:
                minimal_args.append(sentinel)
                full_args.append(sentinel)
            else:
                full_kwargs[parameter.name] = sentinel
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            if required:
                minimal_kwargs[parameter.name] = sentinel
            full_kwargs[parameter.name] = sentinel

    return [
        (minimal_args, minimal_kwargs),
        (full_args, full_kwargs),
    ]


class SceneBackend(ABC):
    @abstractmethod
    def get_env(self) -> EnvProtocol:
        """Return the stable basis environment exposed to runners and policies.

        This method is called after backend construction and before ``setup``;
        the environment object and its capability methods must already exist.
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of envs in the backend batch."""

    @abstractmethod
    def setup(self, config: AutoAtomConfig) -> None:
        """Prepare task state after environment and handler construction."""

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

    @abstractmethod
    def get_grasped_object_name(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[str]:
        """Return the object grasped by an operator in one environment.

        Return ``None`` when the operator is empty-handed.  This is the
        stable public lookup used by PLACE validation; implementations may
        keep their object registries private.
        """

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

    @abstractmethod
    def is_operator_contacting(
        self,
        operator_name: str,
        object_name: str,
    ) -> np.ndarray:
        """Return whether the operator contacts the object in each environment."""

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

    def get_random_generator(self) -> Optional[np.random.Generator]:
        """Return the backend-owned RNG, when it has one.

        Runner-owned seeded randomness is used when a backend returns ``None``.
        This keeps waypoint randomization deterministic without reaching into a
        backend's private state.
        """
        return None

    def get_camera_reset_poses(self, env_index: int) -> Dict[str, PoseState]:
        """Return randomized camera poses included in reset diagnostics.

        Backends without camera randomization return an empty mapping.
        """
        return {}


def _teardown_backend_after_initialization_failure(backend: SceneBackend) -> None:
    try:
        backend.teardown()
    except Exception:
        logger.exception("Backend teardown failed after initialization error.")


@dataclass
class ArcExecutionSnapshot:
    start_eef_pose: Optional[PoseState] = None
    pivot_world_pos: Optional[Position] = None
    control_ticks: int = 0


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
    random_generator: np.random.Generator = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        seed = self.config.seed if self.config.seed != 0 else None
        self.random_generator = np.random.default_rng(seed)


@dataclass
class StageExecutionPlan:
    stage_index: int
    stage: StageConfig
    operator_name: str

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
    timed_updates: Optional[int] = None


@dataclass
class ActiveStageState:
    plan: StageExecutionPlan
    operator: OperatorHandler
    target: Optional[ObjectHandler]
    actions: List[PrimitiveAction]
    action_index: int = 0
    initial_object_pose: Optional[PoseState] = None
    held_object_name: Optional[str] = None
    completion_pose: Optional[PoseControlConfig] = None


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

    def compile(
        self,
        context: ExecutionContext,
        *,
        validate_boundaries: bool = True,
    ) -> "ExecutionTimeline":
        """Compile the static execution timeline for this builder."""

        from .execution_timeline import ExecutionTimeline

        return ExecutionTimeline.compile(
            self,
            context,
            validate_boundaries=validate_boundaries,
        )

    @staticmethod
    def _select_operator(stage: StageConfig, backend: SceneBackend) -> str:
        if not stage.operator:
            raise ValueError("Stage did not specify an operator.")
        handler = backend.get_operator_handler(stage.operator)
        if handler.name != stage.operator:
            raise ValueError(
                "Backend returned an operator handler with mismatched identity: "
                f"requested {stage.operator!r}, got {handler.name!r}."
            )
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
                    eef=TaskFlowBuilder._grasp_eef(
                        control,
                        require_target_grasp=True,
                    ),
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
    def _grasp_eef(
        control: StageControlConfig,
        *,
        require_target_grasp: bool = False,
    ) -> EefControlConfig:
        eef = control.eef or EefControlConfig(close=True)
        if require_target_grasp:
            if not eef.close:
                raise ValueError(
                    "pick and pull operations require a closing EEF command"
                )
            if not eef.require_grasp:
                return eef.model_copy(update={"require_grasp": True})
        return eef

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
        self._timeline: Optional["ExecutionTimeline"] = None
        self._stage_execution: Optional["StageExecution"] = None

    @property
    def batch_size(self) -> int:
        return self._require_context().backend.batch_size

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
        from .stage_execution import StageExecution

        backend = config.backend(config.task, config.task_operators)
        if not isinstance(backend, SceneBackend):
            raise TypeError(
                "Task file backend must be an instantiated SceneBackend. "
                f"Got {type(backend).__name__}."
            )
        try:
            require_env_capability(
                backend.get_env(),
                EnvProtocol,
                feature="TaskRunner initialization",
                expected_batch_size=backend.batch_size,
            )
            context = ExecutionContext(
                config=config.task,
                backend=backend,
                task_file=config,
            )
            timeline = self.builder.compile(
                context,
                validate_boundaries=True,
            )
            plan = list(timeline.stage_plans)
            context.plan = plan
            backend.setup(context.config)
            stage_execution = StageExecution(
                context,
                plan,
                actions_factory=lambda stage_plan: self._materialize_stage_actions(
                    stage_plan
                ),
                timeline=timeline,
                action_runner=lambda env_index, stage_plan, action, env_mask: (
                    self._run_stage_action(
                        env_index=env_index,
                        plan=stage_plan,
                        action=action,
                        backend=self._require_context().backend,
                        env_mask=env_mask,
                    )
                ),
            )
        except BaseException:
            _teardown_backend_after_initialization_failure(backend)
            raise

        self._context = context
        self._timeline = timeline
        self._plan = plan
        self._stage_execution = stage_execution
        self._env_states = stage_execution.states
        self._has_reset = np.zeros(backend.batch_size, dtype=bool)
        self._public_internal_updates = np.zeros(backend.batch_size, dtype=np.int64)
        self._last_execution_details = [{} for _ in range(backend.batch_size)]
        self._records = self._stage_execution.records
        return self

    def _materialize_stage_actions(
        self,
        plan: StageExecutionPlan,
    ) -> List[PrimitiveAction]:
        timeline = self._require_timeline()
        actions = timeline.clone_stage_actions(plan.stage_index)
        TaskRunner._apply_waypoint_randomization(actions, self._require_context())
        return actions

    def _interval_boundary_state_index(
        self,
        boundary: TaskKeypointConfig,
    ) -> int:
        """Return the physical state boundary reached after N keypoints."""
        return self._require_timeline().boundary_state_index(boundary)

    def _interval_boundary_keypoint(
        self,
        boundary: TaskKeypointConfig,
    ) -> _ResolvedTaskKeypoint:
        return self._require_timeline().boundary_keypoint(boundary)

    def _completed_interval_state_index(
        self,
        completed: _ResolvedTaskKeypoint,
    ) -> int:
        return self._require_timeline().completed_interval_state_index(completed)

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
        self._require_stage_execution().reset(
            mask,
            lambda env_index: self._collect_reset_details(env_index, context),
        )
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
        selection = self._require_timeline().interval_selection
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
        timeline = self._require_timeline()
        selection = timeline.interval_selection
        boundary = timeline.update_boundary
        max_updates = timeline.max_internal_updates_per_update
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
                    and self._completed_interval_state_index(completed_keypoint)
                    == self._interval_boundary_state_index(selection.stop)
                ):
                    self._finish_interval(
                        state,
                        self._interval_boundary_keypoint(selection.stop),
                        selection,
                    )
                    self._last_execution_details[env_index] = self._execution_details(
                        context,
                        event="interval_stop_reached",
                        internal_updates=int(internal_updates[env_index]),
                    )
                    pending[env_index] = False
                    continue

                boundary_event = self._require_timeline().reached_update_boundary(event)
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

    def _execution_details(
        self,
        context: ExecutionContext,
        *,
        event: str,
        internal_updates: int,
    ) -> Dict[str, Any]:
        execution = context.task_file.execution
        timeline = self._require_timeline()
        return {
            "event": event,
            "update_boundary": timeline.update_boundary.value,
            "render_internal_updates": bool(execution.render_internal_updates),
            "internal_updates": internal_updates,
            "max_internal_updates_per_update": timeline.max_internal_updates_per_update,
        }

    def _fail_internal_update_limit(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        internal_updates: int,
    ) -> None:
        max_updates = self._require_timeline().max_internal_updates_per_update
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
        """Advance selected environments to the configured start boundary."""
        pending = np.asarray(mask, dtype=bool).copy()
        ticks = np.zeros(len(self._env_states), dtype=np.int64)
        reached = np.zeros(len(self._env_states), dtype=bool)
        start_state_index = self._interval_boundary_state_index(selection.start)
        stop_state_index = self._interval_boundary_state_index(selection.stop)
        start_keypoint = self._interval_boundary_keypoint(selection.start)
        stop_keypoint = self._interval_boundary_keypoint(selection.stop)
        if start_state_index == 0:
            reached[pending] = True
            pending[:] = False
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
                if (
                    completed is not None
                    and self._completed_interval_state_index(completed)
                    == start_state_index
                ):
                    reached[index] = True
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
        self._records[record_start:] = [
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

            state.done = False
            state.success = False
            state.latest_status = StageExecutionStatus.PENDING
            reached_details = (
                reset_details[index]
                if selection.start.side == KeypointSide.BEFORE
                else {**reset_details[index], **state.latest_details}
            )
            state.latest_details = {
                **reached_details,
                "interval_selection": interval_details,
            }
            state.reported_keypoint = start_keypoint
            state.phase = start_keypoint.phase.value
            state.phase_step = start_keypoint.waypoint
            if start_state_index == stop_state_index:
                self._finish_interval(state, stop_keypoint, selection)
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
        if selection.stop.side == KeypointSide.BEFORE:
            latest_details = {
                field_name: state.latest_details[field_name]
                for field_name in ("initial_poses",)
                if field_name in state.latest_details
            }
            latest_details["event"] = "interval_stop_reached"
        else:
            latest_details = dict(state.latest_details)
        state.done = True
        state.success = True
        state.latest_status = StageExecutionStatus.SUCCEEDED
        state.latest_details = {
            **latest_details,
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
        backend = self._require_context().backend
        return require_env_capability(
            backend.get_env(),
            EnvProtocol,
            feature="TaskRunner.get_env()",
            expected_batch_size=backend.batch_size,
        )

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
        self._timeline = None
        self._stage_execution = None

    def _update_env(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
    ) -> _EnvUpdateEvent:
        if state is not self._env_states[env_index] or context is not self._context:
            raise RuntimeError("Stage execution received state from another runner.")
        return self._require_stage_execution().advance_control(
            env_index,
            use_configured_identity=(
                self._require_timeline().interval_selection is not None
                or self._require_timeline().update_boundary
                != UpdateBoundary.CONTROL_TICK
            ),
        )

    def _record_failure(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        details: Dict[str, Any],
    ) -> None:
        self._require_stage_execution().record_failure(env_index, plan, details)

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
        backend_rng = context.backend.get_random_generator()
        rng = backend_rng if backend_rng is not None else context.random_generator
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
        result = TaskRunner._run_action(
            env_index=env_index,
            operator=operator,
            action=action,
            target=target,
            backend=backend,
            env_mask=env_mask,
            reference_site=plan.stage.site,
        )
        return TaskRunner._refine_absolute_arc_result(
            env_index=env_index,
            action=action,
            backend=backend,
            result=result,
        )

    @staticmethod
    def _refine_absolute_arc_result(
        env_index: int,
        action: PrimitiveAction,
        backend: SceneBackend,
        result: ControlResult,
    ) -> ControlResult:
        """Convert local EEF completion into absolute-joint arc completion."""
        pose = action.pose
        arc = None if pose is None else pose.arc
        if (
            action.kind != "pose"
            or arc is None
            or not arc.absolute
            or not isinstance(arc.pivot, str)
        ):
            return result

        if action.arc_snapshot is None:
            action.arc_snapshot = ArcExecutionSnapshot()
        action.arc_snapshot.control_ticks += 1

        raw_signal = result.signals[env_index]
        if raw_signal in {ControlSignal.FAILED, ControlSignal.TIMED_OUT}:
            return result

        current_joint = float(backend.get_joint_angle(arc.pivot, env_index))
        joint_error = float(arc.angle) - current_joint
        within_tolerance = abs(joint_error) <= float(arc.joint_tolerance)

        signals = result.signals.copy()
        details = [dict(item) for item in result.details]
        env_details = details[env_index]
        env_details.update(
            {
                "absolute_arc_control_ticks": action.arc_snapshot.control_ticks,
                "current_joint_angle": current_joint,
                "target_joint_angle": float(arc.angle),
                "joint_angle_error": joint_error,
                "joint_tolerance": float(arc.joint_tolerance),
            }
        )

        if raw_signal == ControlSignal.REACHED and not within_tolerance:
            signals[env_index] = ControlSignal.RUNNING
            env_details["event"] = "absolute_arc_segment_reached"

        primitive_reached = signals[env_index] == ControlSignal.REACHED
        timeout_exhausted = action.arc_snapshot.control_ticks >= int(arc.timeout_steps)
        if not primitive_reached and timeout_exhausted:
            signals[env_index] = ControlSignal.TIMED_OUT
            env_details.update(
                {
                    "event": "absolute_arc_timeout",
                    "failure_category": "controller_timeout",
                    "failure_reason": (
                        "absolute arc did not reach its target joint angle within "
                        f"{arc.timeout_steps} aggregate control updates"
                    ),
                }
            )

        return ControlResult(signals=signals, details=details)

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
            return operator.control_eef(action.eef, target, env_mask=env_mask)
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
            self._timeline.interval_selection if self._timeline is not None else None
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

        camera_poses = {
            name: self._serialize_pose(pose)
            for name, pose in context.backend.get_camera_reset_poses(env_index).items()
        }
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

    def _require_timeline(self) -> "ExecutionTimeline":
        if self._timeline is None:
            raise RuntimeError("TaskRunner is not initialized. Call from_yaml() first.")
        return self._timeline

    def _require_stage_execution(self) -> "StageExecution":
        if self._stage_execution is None:
            raise RuntimeError("TaskRunner is not initialized. Call from_yaml() first.")
        return self._stage_execution


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
