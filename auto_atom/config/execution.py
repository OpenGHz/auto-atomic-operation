"""execution configuration models (split from the former auto_atom.framework monolith)."""

from collections.abc import Mapping
from enum import Enum
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class TaskPhase(str, Enum):
    """A configured phase within a task stage."""

    PRE_MOVE = "pre_move"
    """A pose waypoint executed before the stage's end-effector action."""
    EEF = "eef"
    """The stage's single end-effector action."""
    POST_MOVE = "post_move"
    """A pose waypoint executed after the stage's end-effector action."""


class KeypointSide(str, Enum):
    """A boundary side relative to a configured task keypoint."""

    BEFORE = "before"
    """The state immediately before the keypoint executes."""
    AFTER = "after"
    """The state immediately after the keypoint fully executes."""


class UpdateBoundary(str, Enum):
    """Boundary at which one public runner update returns."""

    CONTROL_TICK = "control_tick"
    """Return after one controller update, preserving the legacy behavior."""
    PRIMITIVE = "primitive"
    """Return after the active primitive action completes."""
    KEYPOINT = "keypoint"
    """Return after the active configured keypoint completes."""
    STAGE = "stage"
    """Return after the active task stage completes."""


class TaskKeypointConfig(BaseModel, frozen=True):
    """A stable reference to one configured task keypoint."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    stage: str = Field(min_length=1)
    """The stage name. Unnamed stages use their generated ``stage_N`` name."""
    phase: TaskPhase
    """The phase containing the keypoint: ``pre_move``, ``eef``, or ``post_move``."""
    waypoint: NonNegativeInt
    """Zero-based YAML waypoint index within the phase; ``eef`` only accepts 0."""
    side: Optional[KeypointSide] = None
    """Boundary side relative to the keypoint. Within ``interval_selection``,
    an omitted value resolves to ``before`` for ``start`` and ``after`` for
    ``stop``."""


class IntervalSelectionConfig(BaseModel, frozen=True):
    """Start and stop boundaries for a TaskRunner rollout."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    start: TaskKeypointConfig
    """The boundary reached by ``reset()`` and exposed as the initial state."""
    stop: TaskKeypointConfig
    """The boundary at which the selected interval succeeds."""
    max_fast_forward_updates: PositiveInt = 10_000
    """Maximum controller updates per environment while ``reset()`` advances
    to ``start``."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_endpoint_sides(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        default_sides = {
            "start": KeypointSide.BEFORE,
            "stop": KeypointSide.AFTER,
        }
        for field_name, default_side in default_sides.items():
            endpoint = normalized.get(field_name)
            if isinstance(endpoint, TaskKeypointConfig):
                endpoint = endpoint.model_dump()
            if isinstance(endpoint, Mapping):
                endpoint = dict(endpoint)
                if endpoint.get("side") is None:
                    endpoint["side"] = default_side
                normalized[field_name] = endpoint
        return normalized


class ExecutionMode(str, Enum):
    """How configured task stages are executed."""

    PHYSICAL = "physical"
    """Execute every waypoint and end-effector command through an operator."""

    OBJECT_ONLY = "object_only"
    """Hide operators and kinematically transport picked objects."""


class ObjectMotionMode(str, Enum):
    """How held-object waypoints are applied in ``object_only`` mode."""

    DIRECT = "direct"
    """Apply each resolved held-object waypoint in one kinematic update."""

    INTERPOLATED = "interpolated"
    """Advance toward each waypoint using the configured per-update limits."""


class ObjectMotionExecutionConfig(BaseModel, frozen=True):
    """Kinematic object-transport settings for ``object_only`` execution."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    mode: ObjectMotionMode = ObjectMotionMode.DIRECT
    """Object-motion strategy; ``direct`` is the efficient default."""

    max_linear_step: PositiveFloat = 0.02
    """Maximum translation per update when ``mode=interpolated``, in metres."""

    max_angular_step: PositiveFloat = 0.15
    """Maximum rotation per update when ``mode=interpolated``, in radians."""


class ExecutionConfig(BaseModel, frozen=True):
    """TaskRunner execution policy for one runnable task file."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    mode: ExecutionMode = ExecutionMode.PHYSICAL
    """Execution strategy. ``object_only`` removes configured operators and
    directly moves the logically picked object along held-object waypoints.
    """
    object_motion: ObjectMotionExecutionConfig = ObjectMotionExecutionConfig()
    """Kinematic object-motion policy used only by ``object_only`` mode."""
    interval_selection: Optional[IntervalSelectionConfig] = None
    """Optional task interval. ``reset()`` advances to the configured start
    boundary, and execution succeeds at the configured stop boundary."""
    update_boundary: UpdateBoundary = UpdateBoundary.CONTROL_TICK
    """Boundary at which each public runner update returns."""
    render_internal_updates: bool = True
    """Whether the viewer renders every controller update inside a public
    runner update. When false, physics still advances normally and the viewer
    refreshes once at the public boundary."""
    max_internal_updates_per_update: PositiveInt = 10_000
    """Maximum controller updates performed internally by one public runner
    update. Interval reset fast-forward has its own independent limit."""
