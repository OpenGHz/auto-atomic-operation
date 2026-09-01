"""Backend-independent values shared by execution planning and runtime.

The task runner owns the mutable execution context, while the execution
timeline needs a small set of value objects to describe nominal actions and
crossed boundaries.  Keeping those objects in this dependency-free module
lets both modules depend on the model without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

from .framework import (
    EefControlConfig,
    IntervalSelectionConfig,
    PoseControlConfig,
    Position,
    StageConfig,
    TaskKeypointConfig,
    TaskPhase,
    UpdateBoundary,
)
from .utils.pose import PoseState


@dataclass
class ArcExecutionSnapshot:
    """Mutable per-action state captured when an arc primitive starts.

    Length-targeted arcs additionally retain their measured radius and angular
    progress so a dynamic target remains stable across controller updates.
    """

    start_eef_pose: Optional[PoseState] = None
    pivot_world_pos: Optional[Position] = None
    control_ticks: int = 0
    arc_length_radius: Optional[float] = None
    arc_length_total_angle: Optional[float] = None
    arc_length_completed_angle: float = 0.0
    arc_length_segment_angle: Optional[float] = None


class StageExecutionStatus(str, Enum):
    """Terminal/progress status for one Stage in one environment."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ControlSignal(str, Enum):
    """Result of one primitive control update."""

    RUNNING = "running"
    REACHED = "reached"
    TIMED_OUT = "timed_out"
    FAILED = "failed"


@dataclass
class ControlResult:
    """Batched result signals and optional per-row diagnostics."""

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


@dataclass
class ExecutionRecord:
    """One terminal or intermediate Stage execution record."""

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
class ResolvedMotionGoal:
    """One semantic controlled-frame goal and its concrete EEF command."""

    configured_pose: PoseControlConfig
    controlled_world_pose: PoseState
    command_pose: PoseControlConfig
    controlled_object_name: Optional[str] = None
    target_axis_world: Optional[np.ndarray] = None


@dataclass
class ResolvedObjectMotionGoal:
    """One held-object waypoint resolved to controlled-frame and root poses."""

    configured_pose: PoseControlConfig
    controlled_world_pose: PoseState
    object_world_pose: PoseState
    controlled_object_name: str
    target_axis_world: Optional[np.ndarray] = None


@dataclass
class ActiveStageState:
    """Mutable state for the currently running Stage in one environment."""

    plan: StageExecutionPlan
    operator: Any
    target: Any
    actions: List[PrimitiveAction]
    action_index: int = 0
    initial_object_pose: Optional[PoseState] = None
    held_object_name: Optional[str] = None
    completion_pose: Optional[PoseControlConfig] = None
    completion_motion_goal: Optional[ResolvedMotionGoal] = None


@dataclass
class _EnvRuntimeState:
    """Mutable per-environment Stage cursor and status."""

    stage_cursor: int = 0
    active: Optional[ActiveStageState] = None
    done: bool = False
    success: bool = False
    phase: Optional[str] = None
    phase_step: Optional[int] = None
    latest_status: StageExecutionStatus = StageExecutionStatus.PENDING
    latest_details: Dict[str, Any] = field(default_factory=dict)
    reported_keypoint: Optional[_ResolvedTaskKeypoint] = None


@dataclass
class PrimitiveAction:
    """One runtime primitive emitted by a :class:`TaskFlowBuilder`."""

    kind: str
    pose: Optional[PoseControlConfig] = None
    eef: Optional[EefControlConfig] = None
    resolved_pose: Optional[PoseControlConfig] = None
    arc_snapshot: Optional[ArcExecutionSnapshot] = None
    arc_cumulative_angle: Optional[float] = None
    phase: Optional[TaskPhase] = None
    waypoint: Optional[int] = None
    completes_keypoint: bool = True
    resolved_motion_goal: Optional[ResolvedMotionGoal] = None
    resolved_object_motion_goal: Optional[ResolvedObjectMotionGoal] = None
    reference_pose_snapshot: Optional[PoseState] = None


@dataclass
class StageExecutionPlan:
    """Static operator selection and stage identity used by execution."""

    stage_index: int
    stage: StageConfig
    operator_name: str

    @property
    def stage_name(self) -> str:
        return self.stage.name or f"stage_{self.stage_index}"


@dataclass(frozen=True)
class _ResolvedTaskKeypoint:
    """One emitted keypoint identity in the compiled execution timeline."""

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


class ExecutionTimelineProtocol(Protocol):
    """Runtime-facing contract exposed by the compiled execution timeline."""

    stage_plans: tuple[StageExecutionPlan, ...]
    interval_selection: Optional[IntervalSelectionConfig]
    update_boundary: UpdateBoundary
    max_internal_updates_per_update: int

    def clone_stage_actions(self, stage_index: int) -> list[PrimitiveAction]: ...

    def keypoint_for_action(
        self,
        stage_index: int,
        action_index: int,
    ) -> Optional[_ResolvedTaskKeypoint]: ...

    def boundary_state_index(self, boundary: TaskKeypointConfig) -> int: ...

    def boundary_keypoint(
        self,
        boundary: TaskKeypointConfig,
    ) -> _ResolvedTaskKeypoint: ...

    def completed_interval_state_index(
        self,
        completed: _ResolvedTaskKeypoint,
    ) -> int: ...

    def reached_update_boundary(self, event: _EnvUpdateEvent) -> Optional[str]: ...


__all__ = [
    "ArcExecutionSnapshot",
    "ActiveStageState",
    "ControlResult",
    "ControlSignal",
    "ExecutionRecord",
    "PrimitiveAction",
    "ResolvedMotionGoal",
    "ResolvedObjectMotionGoal",
    "StageExecutionPlan",
    "StageExecutionStatus",
    "_EnvRuntimeState",
    "ExecutionTimelineProtocol",
    "_EnvUpdateEvent",
    "_ResolvedTaskKeypoint",
]
