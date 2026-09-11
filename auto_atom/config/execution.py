"""execution configuration models (split from the former auto_atom.framework monolith)."""

from collections.abc import Mapping
from enum import Enum
from typing import List, Optional, Union

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


class KeypointSelectorConfig(BaseModel, frozen=True):
    """One scoped entry of ``execution.keypoint_selection``.

    A scoped entry addresses a stage, optionally narrowed to one phase, and
    optionally narrowed to keypoint indexes inside that scope. Keypoints
    outside every entry are skipped instead of executed.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    stage: str = Field(min_length=1)
    """The stage name. Unnamed stages use their generated ``stage_N`` name."""
    phase: Optional[TaskPhase] = None
    """Optional phase. Omit to address every keypoint of the stage."""
    waypoint: Optional[int] = None
    """Optional keypoint index inside the addressed scope; ``-1`` is its last
    keypoint, and omitting it selects every keypoint of the scope."""

    @model_validator(mode="before")
    @classmethod
    def _reject_side(cls, value: object) -> object:
        if isinstance(value, Mapping) and value.get("side") is not None:
            raise ValueError(
                "a keypoint_selection entry selects keypoints to execute, so it "
                "has no side; use phase and waypoint to trim the selected range"
            )
        return value


KeypointSelector = Union[int, KeypointSelectorConfig]
"""One ``execution.keypoint_selection`` entry.

An ``int`` addresses a keypoint by its zero-based ordinal in the task's
keypoint sequence, counting from the end when negative. A mapping scopes the
same addressing to one stage, or to one phase of a stage.
"""


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
    keypoint_selection: Optional[List[KeypointSelector]] = None
    """Ordered keypoints to execute instead of the full task. Each entry is
    either a task-wide keypoint ordinal or a ``stage`` / ``phase`` / ``waypoint``
    mapping; negative indexes count from the end of the addressed sequence.
    Selected keypoints must follow task execution order; every keypoint outside
    the selection is skipped. Mutually exclusive with ``interval_selection``."""
    update_boundary: UpdateBoundary = UpdateBoundary.CONTROL_TICK
    """Boundary at which each public runner update returns."""
    render_internal_updates: bool = True
    """Whether the viewer renders every controller update inside a public
    runner update. When false, physics still advances normally and the viewer
    refreshes once at the public boundary."""
    max_internal_updates_per_update: PositiveInt = 10_000
    """Maximum controller updates performed internally by one public runner
    update. Interval reset fast-forward has its own independent limit."""

    @model_validator(mode="after")
    def _validate_task_selection(self) -> "ExecutionConfig":
        selection = self.keypoint_selection
        if self.interval_selection is not None and selection is not None:
            raise ValueError(
                "execution.interval_selection and execution.keypoint_selection "
                "are mutually exclusive; configure one task selection"
            )
        if selection is not None and not selection:
            raise ValueError(
                "execution.keypoint_selection must list at least one keypoint"
            )
        return self
