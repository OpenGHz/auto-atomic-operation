"""YAML-driven task runner built from primitive pose and end-effector controls."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Protocol, runtime_checkable
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from .framework import (
    ArcControlConfig,
    AutoAtomConfig,
    EefControlConfig,
    Operation,
    OPERATION_CONDITIONS,
    OperationConditionType,
    OperationConstraint,
    Orientation,
    PoseControlConfig,
    PoseRandomRange,
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
    quaternion_to_rpy,
    rotate_pose_around_axis,
)
import warnings
import math
import numpy as np


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
    """Opaque object handle resolved by the scene backend."""

    name: str
    """The unique object name used by stage configs and backend lookups."""

    def get_pose(self) -> "PoseState":
        """Return the current world pose of the object."""
        raise NotImplementedError

    def set_pose(self, pose: "PoseState") -> None:  # noqa: ARG002
        """Force-set the object world pose in one step (no physics integration)."""
        raise NotImplementedError


@dataclass
class ControlResult:
    """Incremental low-level control result."""

    signal: ControlSignal
    """The coarse controller state returned after advancing one primitive action step."""
    details: Dict[str, Any] = field(default_factory=dict)
    """Backend-specific diagnostic details associated with the returned control signal."""


@runtime_checkable
class IKSolver(Protocol):
    """Inverse kinematics solver protocol.

    Implementations receive the desired end-effector pose **in the operator's
    base frame** and the current arm joint positions, and return target joint
    positions for the arm actuators.  Return ``None`` when no solution exists.
    """

    def solve(
        self,
        target_pose_in_base: PoseState,
        current_qpos: np.ndarray,
    ) -> Optional[np.ndarray]: ...


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
        target: Optional[ObjectHandler],
    ) -> ControlResult:
        """Advance motion toward the desired pose."""

    @abstractmethod
    def control_eef(
        self,
        eef: EefControlConfig,
    ) -> ControlResult:
        """Advance the end-effector toward the desired state."""

    @abstractmethod
    def get_end_effector_pose(self) -> PoseState:
        """Return the current world pose of the operator end-effector."""

    @abstractmethod
    def get_base_pose(self) -> PoseState:
        """Return the current world pose of the operator base."""

    def set_pose(self, pose: PoseState) -> None:  # noqa: ARG002
        """Force-set the operator base world pose in one step (no physics integration)."""
        raise NotImplementedError


class SceneBackend(ABC):
    """Abstract scene backend used by the task runner."""

    @abstractmethod
    def setup(self, config: AutoAtomConfig) -> None:
        """Prepare backend resources for this task."""

    @abstractmethod
    def reset(self) -> None:
        """Reset scene state for a new run."""

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

    def is_object_displaced(
        self,
        object_name: str,
        original_pose: "PoseState",
        threshold: float = 0.01,
    ) -> bool:
        """Return True if *object_name* has moved more than *threshold* metres from *original_pose*.

        The default implementation uses Euclidean distance between positions.
        Override for orientation-aware or physics-specific displacement metrics.
        """
        handler = self.get_object_handler(object_name)
        if handler is None:
            return False
        current = handler.get_pose()
        delta = float(
            np.linalg.norm(
                np.asarray(current.position, dtype=np.float64)
                - np.asarray(original_pose.position, dtype=np.float64)
            )
        )
        return delta > threshold

    def is_operator_contacting(self, operator_name: str, object_name: str) -> bool:
        """Return True if the operator end-effector is currently in contact with *object_name*.

        The default implementation always returns False.  Override in backends that
        support contact sensing (e.g. MuJoCo contact pair queries or tactile sensors).
        """
        return False

    def get_element_pose(self, name: str) -> "PoseState":
        """Return the world pose of a named scene element (site, body, or joint).

        Backends should override this to resolve the name against the physics
        engine.  The default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Backend does not support named element lookup (requested '{name}')."
        )

    def get_joint_angle(self, name: str) -> float:
        """Return the current angle (radians) of a named joint.

        Backends should override this.  The default raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Backend does not support joint angle lookup (requested '{name}')."
        )

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        """Notify the backend about the current task-focus objects and operations."""

    # ------------------------------------------------------------------
    # Randomization helpers  (shared by all concrete backend subclasses)
    # ------------------------------------------------------------------
    #
    # Concrete backends that support randomization must expose these
    # instance attributes (e.g. as dataclass fields):
    #
    #   randomization  : Dict[str, PoseRandomRange]
    #   _rng           : np.random.Generator
    #   _default_poses : Dict[str, PoseState]
    #
    # and call ``_record_default_poses()`` from ``setup()`` / ``reset()``
    # and ``_apply_randomization()`` at the end of ``reset()``.
    # ------------------------------------------------------------------

    def _record_default_poses(self) -> None:
        """Snapshot the current pose of every entity listed in ``self.randomization``.

        Call this once after the simulation has been reset to its canonical
        initial state (i.e. after ``env.reset()`` and operator ``home()``).
        """
        randomization: Dict[str, "PoseRandomRange"] = getattr(self, "randomization", {})
        default_poses: Dict[str, PoseState] = getattr(self, "_default_poses", {})
        for name in randomization:
            kind, handler = self._resolve_randomization_handler(name)
            if handler is None:
                continue
            if kind == "object":
                default_poses[name] = handler.get_pose()
            else:
                default_poses[name] = handler.get_base_pose()
        # Ensure the attribute is updated in case it was a local default.
        try:
            self._default_poses = default_poses  # type: ignore[attr-defined]
        except AttributeError:
            pass

    def _apply_randomization(self) -> None:
        """Sample random pose offsets for all configured entities and apply them.

        Pairwise collision rejection is performed: if any two entities' sampled
        centres are closer than the sum of their ``collision_radius`` values the
        sample is discarded and redrawn.  After ``_MAX_RANDOMIZATION_RETRIES``
        failed attempts the last sample is applied with a warning.
        """
        randomization: Dict[str, "PoseRandomRange"] = getattr(self, "randomization", {})
        rng: np.random.Generator = getattr(self, "_rng", np.random.default_rng())
        default_poses: Dict[str, PoseState] = getattr(self, "_default_poses", {})

        entries = []
        for name, rand_range in randomization.items():
            kind, handler = self._resolve_randomization_handler(name)
            if handler is None:
                warnings.warn(
                    f"{type(self).__name__}: randomization key '{name}' does not match "
                    "any known object or operator — skipping.",
                    stacklevel=3,
                )
                continue
            entries.append((name, kind, handler, rand_range))

        if not entries:
            return

        # Debug mode: consume the pre-built extreme queue before going random.
        if getattr(self, "randomization_debug", False):
            queue = getattr(self, "_debug_extreme_queue", None)
            if queue is None:
                queue = self._build_debug_extreme_queue()
                try:
                    self._debug_extreme_queue = queue  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            idx: int = getattr(self, "_debug_extreme_index", 0)
            if idx < len(queue):
                poses, label = queue[idx]
                try:
                    self._debug_extreme_index = idx + 1  # type: ignore[attr-defined]
                except AttributeError:
                    pass
                print(f"[DEBUG randomization] extreme {idx + 1}/{len(queue)}: {label}")
                name_to_handler = {n: h for n, _, h, _ in entries}
                for name, pose in poses.items():
                    if name in name_to_handler:
                        name_to_handler[name].set_pose(pose)
                return
            if idx == len(queue):
                print(
                    f"[DEBUG randomization] all {len(queue)} extreme cases exhausted"
                    " — switching to random sampling"
                )
                try:
                    self._debug_extreme_index = idx + 1  # type: ignore[attr-defined]
                except AttributeError:
                    pass

        last_poses: Dict[str, PoseState] = {}
        for _ in range(_MAX_RANDOMIZATION_RETRIES):
            sampled: Dict[str, PoseState] = {}
            for name, _kind, _handler, rand_range in entries:
                default = default_poses.get(name, PoseState())
                sampled[name] = self._sample_random_pose(rng, default, rand_range)

            # Pairwise collision check based on Euclidean distance.
            collision = False
            names = list(sampled)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    ni, nj = names[i], names[j]
                    ri = randomization[ni].collision_radius
                    rj = randomization[nj].collision_radius
                    pi = np.asarray(sampled[ni].position, dtype=np.float64)
                    pj = np.asarray(sampled[nj].position, dtype=np.float64)
                    if float(np.linalg.norm(pi - pj)) < ri + rj:
                        collision = True
                        break
                if collision:
                    break

            last_poses = sampled
            if not collision:
                break
        else:
            warnings.warn(
                f"{type(self).__name__}: could not find a collision-free randomization "
                f"after {_MAX_RANDOMIZATION_RETRIES} attempts — applying last sample.",
                stacklevel=3,
            )

        for name, _kind, handler, _rand_range in entries:
            handler.set_pose(last_poses[name])

    @staticmethod
    def _delta_pose(
        default: PoseState,
        dx: float = 0.0,
        dy: float = 0.0,
        dz: float = 0.0,
        d_roll: float = 0.0,
        d_pitch: float = 0.0,
        d_yaw: float = 0.0,
    ) -> PoseState:
        """Return *default* displaced by the given world-frame deltas."""
        new_pos = (
            default.position[0] + dx,
            default.position[1] + dy,
            default.position[2] + dz,
        )
        r, p, y = quaternion_to_rpy(default.orientation)
        return PoseState(
            position=new_pos,
            orientation=euler_to_quaternion((r + d_roll, p + d_pitch, y + d_yaw)),
        )

    @staticmethod
    def _sample_random_pose(
        rng: np.random.Generator,
        default: PoseState,
        rand_range: "PoseRandomRange",
    ) -> PoseState:
        """Return a new pose sampled uniformly within *rand_range* around *default*."""
        return SceneBackend._delta_pose(
            default,
            dx=float(rng.uniform(*rand_range.x)),
            dy=float(rng.uniform(*rand_range.y)),
            dz=float(rng.uniform(*rand_range.z)),
            d_roll=float(rng.uniform(*rand_range.roll)),
            d_pitch=float(rng.uniform(*rand_range.pitch)),
            d_yaw=float(rng.uniform(*rand_range.yaw)),
        )

    def _build_debug_extreme_queue(
        self,
    ) -> "List[tuple[Dict[str, PoseState], str]]":
        """Build an ordered list of extreme-case pose configurations.

        The sequence is:
        1. All entities at all-axis minimum simultaneously.
        2. All entities at all-axis maximum simultaneously.
        3. For each entity, for each non-trivial axis (lo != hi):
           - one case at the axis minimum (all other axes at default for that entity,
             all other entities at their defaults)
           - one case at the axis maximum (same)
        """
        _AXES = ("x", "y", "z", "roll", "pitch", "yaw")
        _KWARG = {
            "x": "dx",
            "y": "dy",
            "z": "dz",
            "roll": "d_roll",
            "pitch": "d_pitch",
            "yaw": "d_yaw",
        }
        _UNIT = {
            "x": "m",
            "y": "m",
            "z": "m",
            "roll": "rad",
            "pitch": "rad",
            "yaw": "rad",
        }

        randomization: Dict[str, "PoseRandomRange"] = getattr(self, "randomization", {})
        default_poses: Dict[str, PoseState] = getattr(self, "_default_poses", {})

        def _defaults() -> Dict[str, PoseState]:
            return {n: default_poses.get(n, PoseState()) for n in randomization}

        cases: "List[tuple[Dict[str, PoseState], str]]" = []

        # All entities at all-min simultaneously, then all-max.
        for tag, eidx in (("all_min", 0), ("all_max", 1)):
            c: Dict[str, PoseState] = {}
            parts = []
            for name, rand_range in randomization.items():
                default = default_poses.get(name, PoseState())
                kwargs = {_KWARG[ax]: getattr(rand_range, ax)[eidx] for ax in _AXES}
                c[name] = self._delta_pose(default, **kwargs)
                parts.append(name)
            cases.append((c, f"{tag}: " + ", ".join(parts)))

        # Per-entity, per-axis min / max (all other entities stay at default).
        for name, rand_range in randomization.items():
            default = default_poses.get(name, PoseState())
            for axis in _AXES:
                lo, hi = getattr(rand_range, axis)
                if lo == hi:
                    continue
                for val, tag in ((lo, "min"), (hi, "max")):
                    c = _defaults()
                    c[name] = self._delta_pose(default, **{_KWARG[axis]: val})
                    label = f"{name}.{axis}={val:.4g} {_UNIT[axis]} ({tag})"
                    cases.append((c, label))

        return cases

    def _resolve_randomization_handler(
        self, name: str
    ) -> "tuple[str, Optional[ObjectHandler | OperatorHandler]]":
        """Return ``(kind, handler)`` for *name*, trying objects then operators.

        Returns ``(kind, None)`` when the name is not found in either registry.
        ``kind`` is ``'object'`` or ``'operator'``.
        """
        try:
            obj = self.get_object_handler(name)
            if obj is not None:
                return "object", obj
        except (KeyError, NotImplementedError):
            pass
        try:
            return "operator", self.get_operator_handler(name)
        except (KeyError, NotImplementedError):
            return "operator", None


_MAX_RANDOMIZATION_RETRIES = 100


@dataclass
class PrimitiveAction:
    """Single primitive control action derived from a stage."""

    kind: str
    """The primitive action kind, typically ``pose`` or ``eef``."""
    pose: Optional[PoseControlConfig] = None
    """The pose target for pose actions, or ``None`` for non-pose actions."""
    eef: Optional[EefControlConfig] = None
    """The end-effector target for eef actions, or ``None`` for non-eef actions."""
    resolved_pose: Optional[PoseControlConfig] = None
    """The runtime-resolved pose after applying reference-frame conversion, when available."""


@dataclass
class ExecutionRecord:
    """Final record for one completed or failed stage."""

    stage_index: int
    """The zero-based index of the executed stage in the task definition."""
    stage_name: str
    """The human-readable stage name reported to users and logs."""
    operator: str
    """The operator name chosen to execute the stage."""
    operation: str
    """The high-level operation name executed by the stage."""
    target_object: str
    """The target object name associated with the stage, if any."""
    blocking: bool
    """Whether the stage was configured to block task progression until completion."""
    status: StageExecutionStatus
    """The final execution status reached by the stage."""
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional execution metadata collected while running the stage."""


@dataclass
class ExecutionContext:
    """Mutable runtime context shared across the task lifecycle."""

    config: AutoAtomConfig
    """The validated task configuration currently being executed."""
    backend: SceneBackend
    """The scene backend instance bound to this task run."""
    task_file: TaskFileConfig
    """The full validated task file, including operator declarations."""


@dataclass
class StageExecutionPlan:
    """Validated executable stage plan."""

    stage_index: int
    """The zero-based index of the stage in the original task definition."""
    stage: StageConfig
    """The validated stage configuration to execute."""
    operator_name: str
    """The resolved operator name that will execute this stage."""
    last_orientation_before: Optional[Orientation] = None
    """The last configured orientation from preceding stages, for inheritance."""

    @property
    def stage_name(self) -> str:
        return self.stage.name or f"stage_{self.stage_index}"


@dataclass
class TaskUpdate:
    """User-facing task progress returned by ``TaskRunner.update``."""

    stage_index: Optional[int]
    """The active stage index, or ``None`` when the task is idle or finished."""
    stage_name: str
    """The current stage name exposed to the caller."""
    status: StageExecutionStatus
    """The latest high-level status for the active stage or overall task."""
    done: bool
    """Whether the task has fully finished and will produce no further work."""
    success: Optional[bool]
    """The final task success flag when done, or ``None`` while still running."""
    details: Dict[str, Any] = field(default_factory=dict)
    """Additional progress metadata describing the latest update."""
    phase: Optional[str] = None
    """Current execution phase: 'pre_move', 'eef', or 'post_move'.
    None when the task is idle or finished."""
    phase_step: Optional[int] = None
    """Zero-based index of the current waypoint within the active move phase
    (pre_move or post_move). None when the phase is eef or the task is idle/finished."""


@dataclass
class ActiveStageState:
    """Internal state for the currently active stage."""

    plan: StageExecutionPlan
    """The resolved execution plan for the active stage."""
    operator: OperatorHandler
    """The operator handler currently executing primitive actions for this stage."""
    target: Optional[ObjectHandler]
    """The resolved target object handler for the stage, if one is required."""
    actions: List[PrimitiveAction]
    """The ordered primitive actions still being executed for this stage."""
    action_index: int = 0
    """The index of the primitive action currently in progress."""
    initial_object_pose: Optional[PoseState] = None
    """The world pose of the target object captured at stage start; used for displacement checks."""


class ComponentRegistry:
    """Process-global registry storing named component instances shared within the current process."""

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
            if pose.orientation:
                last_orientation = pose.orientation
            # Split arc poses into small sub-steps so the gripper traces the arc
            # instead of cutting through the chord.
            if pose.arc is not None:
                sub_poses = TaskFlowBuilder._split_arc(pose)
                for sp in sub_poses:
                    actions.append(PrimitiveAction(kind="pose", pose=sp))
            else:
                actions.append(PrimitiveAction(kind="pose", pose=pose))
        return actions, last_orientation

    @staticmethod
    def _split_arc(pose: PoseControlConfig) -> List[PoseControlConfig]:
        """Split a large arc into smaller sub-arcs of at most ``arc.max_step`` each.

        Absolute arcs are NOT split — they re-resolve each control step at runtime
        to track the remaining delta, so a single action suffices.
        """
        arc = pose.arc
        assert arc is not None
        if arc.absolute:
            return [pose]
        total = abs(arc.angle)
        n_steps = max(1, math.ceil(total / arc.max_step))
        step_angle = arc.angle / n_steps
        sub_poses: List[PoseControlConfig] = []
        for _ in range(n_steps):
            sub_poses.append(
                PoseControlConfig(
                    arc=ArcControlConfig(
                        pivot=arc.pivot,
                        axis=arc.axis,
                        angle=step_angle,
                    ),
                    reference=pose.reference,
                )
            )
        return sub_poses

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
    """Stateful task executor controlled by ``reset`` and repeated ``update`` calls."""

    def __init__(
        self,
        builder: Optional[TaskFlowBuilder] = None,
    ) -> None:
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
        return self.from_config(task_file)

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
        self._context.backend.setup(self._context.config)
        self._stage_index = 0
        self._active_stage = None
        self._records = []
        return self

    def reset(self) -> TaskUpdate:
        context = self._require_context()
        context.backend.reset()
        context.backend.set_interest_objects_and_operations([], [])
        self._stage_index = 0
        self._active_stage = None
        self._records = []

        # Get initial poses for all operators
        initial_poses = {}
        for plan in self._plan:
            operator_name = plan.operator_name
            try:
                operator = context.backend.get_operator_handler(operator_name)
                current_pose = operator.get_end_effector_pose()
                initial_poses[operator_name] = {
                    "position": list(current_pose.position),
                    "orientation": list(current_pose.orientation),
                }
            except Exception:
                pass  # Skip if operator not found or pose unavailable

        update = self._build_pending_update()
        if initial_poses:
            update.details["initial_poses"] = initial_poses
        return update

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
            precondition_failure = (
                # PULL defers its pre-condition check to after the eef phase.
                None
                if plan.stage.operation == Operation.PULL
                else self._check_stage_condition(
                    context=context,
                    plan=plan,
                    condition_type=OperationConditionType.PERFORM,
                )
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
            context.backend.set_interest_objects_and_operations(
                [plan.stage.object] if plan.stage.object else [],
                [plan.stage.operation.value] if plan.stage.object else [],
            )
            self._active_stage = self._start_stage(context, plan)

        active = self._active_stage
        action = active.actions[active.action_index]
        result = self._run_action(
            active.operator, action, active.target, context.backend
        )
        details = {
            "action": action.kind,
            "action_index": active.action_index,
            **result.details,
        }

        if result.signal == ControlSignal.RUNNING:
            phase, phase_step = self._action_phase(active.actions, active.action_index)
            return self._build_update(
                plan=active.plan,
                status=StageExecutionStatus.RUNNING,
                details=details,
                done=False,
                success=None,
                phase=phase,
                phase_step=phase_step,
            )

        if result.signal == ControlSignal.REACHED:
            completed_action = action
            active.action_index += 1
            op = active.plan.stage.operation

            # --- Phase-boundary condition checks ---
            mid_failure: Optional[Dict[str, Any]] = None
            if completed_action.kind == "eef":
                if op == Operation.PULL:
                    # PULL pre-condition: grasped must hold after eef before post_move.
                    mid_failure = self._check_stage_condition(
                        context=context,
                        plan=active.plan,
                        condition_type=OperationConditionType.PERFORM,
                        initial_pose=active.initial_object_pose,
                    )
                elif op == Operation.PICK and not context.backend.is_operator_grasping(
                    active.operator.name
                ):
                    # PICK: early exit if grasp already failed (avoids unnecessary post_move).
                    mid_failure = self._check_stage_condition(
                        context=context,
                        plan=active.plan,
                        condition_type=OperationConditionType.SUCCESS,
                        initial_pose=active.initial_object_pose,
                    )
                elif op == Operation.PRESS:
                    # PRESS post-condition: contacted must hold at the press point (after eef).
                    mid_failure = self._check_stage_condition(
                        context=context,
                        plan=active.plan,
                        condition_type=OperationConditionType.SUCCESS,
                        initial_pose=active.initial_object_pose,
                    )
            if mid_failure is not None:
                self._records.append(
                    ExecutionRecord(
                        stage_index=active.plan.stage_index,
                        stage_name=active.plan.stage_name,
                        operator=active.operator.name,
                        operation=active.plan.stage.operation.value,
                        target_object=active.plan.stage.object,
                        blocking=active.plan.stage.blocking,
                        status=StageExecutionStatus.FAILED,
                        details=mid_failure,
                    )
                )
                self._active_stage = None
                context.backend.set_interest_objects_and_operations([], [])
                return self._build_update(
                    plan=active.plan,
                    status=StageExecutionStatus.FAILED,
                    details=mid_failure,
                    done=True,
                    success=False,
                )

            if active.action_index < len(active.actions):
                phase, phase_step = self._action_phase(
                    active.actions, active.action_index
                )
                return self._build_update(
                    plan=active.plan,
                    status=StageExecutionStatus.RUNNING,
                    details=details,
                    done=False,
                    success=None,
                    phase=phase,
                    phase_step=phase_step,
                )

            # --- Final post-condition check (after all actions complete) ---
            # PRESS post-condition was already checked after the eef phase; skip here.
            success_failure = (
                None
                if op == Operation.PRESS
                else self._check_stage_condition(
                    context=context,
                    plan=active.plan,
                    condition_type=OperationConditionType.SUCCESS,
                    initial_pose=active.initial_object_pose,
                )
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
                context.backend.set_interest_objects_and_operations([], [])
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
            if self._stage_index >= len(self._plan):
                context.backend.set_interest_objects_and_operations([], [])
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
        context.backend.set_interest_objects_and_operations([], [])
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
        initial_object_pose: Optional[PoseState] = None
        if target is not None:
            try:
                initial_object_pose = target.get_pose()
            except NotImplementedError:
                pass
        return ActiveStageState(
            plan=plan,
            operator=operator,
            target=target,
            actions=self.builder.build_actions(
                plan.stage, plan.last_orientation_before
            )[0],
            initial_object_pose=initial_object_pose,
        )

    @staticmethod
    def _check_stage_condition(
        context: ExecutionContext,
        plan: StageExecutionPlan,
        condition_type: OperationConditionType,
        initial_pose: Optional[PoseState] = None,
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
        is_grasping = backend.is_operator_grasping(operator_name)

        if constraint == OperationConstraint.GRASPED:
            satisfied = is_grasping
        elif constraint == OperationConstraint.RELEASED:
            satisfied = not is_grasping
        elif constraint == OperationConstraint.CONTACTED:
            satisfied = backend.is_operator_contacting(operator_name, object_name)
        elif constraint == OperationConstraint.DISPLACED:
            satisfied = (
                backend.is_object_displaced(object_name, initial_pose)
                if initial_pose is not None and object_name
                else True
            )
        else:
            satisfied = True

        if satisfied:
            return None

        phase = (
            "precondition"
            if condition_type == OperationConditionType.PERFORM
            else "postcondition"
        )
        _failure_map = {
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
        }
        failure_category, failure_reason = _failure_map.get(
            constraint, ("condition_mismatch", "stage condition is not satisfied")
        )

        return {
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
    def _run_action(
        operator: OperatorHandler,
        action: PrimitiveAction,
        target: Optional[ObjectHandler],
        backend: SceneBackend = None,
    ) -> ControlResult:
        if action.kind == "pose" and action.pose is not None:
            # Snapshot-based references: resolve once and cache.
            # Relative arcs are snapshot-based; absolute arcs re-resolve each tick.
            is_arc = action.pose.arc is not None
            is_snapshot = action.pose.reference in {
                PoseReference.EEF_WORLD,
                PoseReference.EEF,
            } or (is_arc and not action.pose.arc.absolute)
            if is_snapshot and action.resolved_pose is not None:
                resolved_pose = action.resolved_pose
            else:
                resolved_pose = TaskRunner._resolve_pose_command(
                    operator=operator,
                    pose=action.pose,
                    target=target,
                    backend=backend,
                )
                action.resolved_pose = resolved_pose
            return operator.move_to_pose(resolved_pose, target)
        if action.kind == "eef" and action.eef is not None:
            return operator.control_eef(action.eef)
        raise RuntimeError(f"Invalid primitive action '{action.kind}'.")

    @staticmethod
    def _resolve_arc_command(
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        backend: Optional[SceneBackend] = None,
    ) -> PoseControlConfig:
        """Resolve an arc pose command: rotate the current EEF pose around the pivot."""
        arc = pose.arc
        assert arc is not None

        # Resolve pivot: string name → world position via backend lookup
        if isinstance(arc.pivot, str):
            if backend is None:
                raise RuntimeError(
                    f"Arc pivot '{arc.pivot}' is a name but no backend is available for lookup."
                )
            pivot_world_pos = backend.get_element_pose(arc.pivot).position
        else:
            reference_pose = TaskRunner._resolve_reference_pose(
                operator=operator,
                pose=pose,
                target=target,
            )
            pivot_local = PoseState(position=arc.pivot)
            pivot_world_pos = compose_pose(reference_pose, pivot_local).position

        # Resolve angle: absolute mode computes delta from current joint angle,
        # clamped to max_step so it converges over multiple control ticks.
        angle = arc.angle
        if arc.absolute:
            if not isinstance(arc.pivot, str):
                raise ValueError(
                    "Arc absolute mode requires pivot to be a joint name (str)."
                )
            if backend is None:
                raise RuntimeError(
                    "Arc absolute mode requires a backend for joint angle lookup."
                )
            current_joint = backend.get_joint_angle(arc.pivot)
            delta = arc.angle - current_joint
            # Clamp to max_step so the gripper traces the arc incrementally
            sign = 1.0 if delta >= 0 else -1.0
            angle = sign * min(abs(delta), arc.max_step)

        current_eef = operator.get_end_effector_pose()
        rotated = rotate_pose_around_axis(
            current_eef,
            pivot_world_pos,
            arc.axis,
            angle,
        )
        return PoseControlConfig(
            position=rotated.position,
            orientation=rotated.orientation,
            reference=PoseReference.WORLD,
            relative=False,
        )

    @staticmethod
    def _resolve_pose_command(
        operator: OperatorHandler,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        backend: Optional[SceneBackend] = None,
    ) -> PoseControlConfig:
        if pose.arc is not None:
            return TaskRunner._resolve_arc_command(operator, pose, target, backend)
        reference_pose = TaskRunner._resolve_reference_pose(
            operator=operator,
            pose=pose,
            target=target,
        )
        current_pose = operator.get_end_effector_pose()
        local_pose = TaskRunner._pose_config_to_local_pose(pose)
        # When orientation/rotation is omitted, preserve the current EEF world orientation.
        # This keeps orientation stable even when switching reference frames between waypoints.
        inherit_orientation = not pose.orientation and not pose.rotation
        current_local = compose_pose(
            inverse_pose(reference_pose),
            current_pose,
        )

        if pose.relative:
            target_pose = compose_pose(current_local, local_pose)
        else:
            target_pose = (
                PoseState(
                    position=local_pose.position,
                    orientation=current_local.orientation,
                )
                if inherit_orientation
                else local_pose
            )

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
            return operator.get_base_pose()
        if reference == PoseReference.EEF:
            return operator.get_end_effector_pose()
        if reference == PoseReference.OBJECT_WORLD:
            if target is None:
                raise ValueError(
                    "Pose reference OBJECT_WORLD requires a target object."
                )
            object_pose = target.get_pose()
            return PoseState(position=object_pose.position)
        if reference == PoseReference.EEF_WORLD:
            eef_pose = operator.get_end_effector_pose()
            return PoseState(position=eef_pose.position)
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
    def _action_phase(actions: List[PrimitiveAction], action_index: int) -> tuple:
        """Return (phase, phase_step) for the given action index."""
        eef_idx: Optional[int] = None
        for idx, a in enumerate(actions):
            if a.kind == "eef":
                eef_idx = idx
                break
        if eef_idx is not None and action_index == eef_idx:
            return "eef", None
        if eef_idx is None or action_index < eef_idx:
            return "pre_move", action_index
        # post_move
        return "post_move", action_index - (eef_idx + 1)

    @staticmethod
    def _build_update(
        plan: StageExecutionPlan,
        status: StageExecutionStatus,
        details: Dict[str, Any],
        done: bool,
        success: Optional[bool],
        phase: Optional[str] = None,
        phase_step: Optional[int] = None,
    ) -> TaskUpdate:
        return TaskUpdate(
            stage_index=plan.stage_index,
            stage_name=plan.stage_name,
            status=status,
            done=done,
            success=success,
            details=details,
            phase=phase,
            phase_step=phase_step,
        )

    def _require_context(self) -> ExecutionContext:
        if self._context is None:
            raise RuntimeError("TaskRunner is not initialized. Call from_yaml() first.")
        return self._context


def load_yaml(path: str | Path) -> Dict[str, Any]:
    config = OmegaConf.load(Path(path))
    data = OmegaConf.to_container(config, resolve=False)
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

    raw = OmegaConf.to_container(config, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping: {config_path}")

    if "env" in config and config.env is not None:
        instantiate(config.env)
    return TaskFileConfig.model_validate(raw)
