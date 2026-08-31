import math
from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

Position = Tuple[float, float, float]
"""A 3D position represented as a tuple of three floats (x, y, z)."""
Orientation = Tuple[float, float, float, float]
"""A quaternion orientation represented as a tuple of four floats (x, y, z, w)."""
Rotation = Tuple[float, float, float]
"""A rotation represented as Euler angles in radians, as a tuple of three floats (roll, pitch, yaw)."""


class Operation(str, Enum):
    """Enumeration of possible operations that the AutoAtom operator can perform.
    `MOVE`, `GRASP`, `RELEASE` are three fundamental operations that can be used to construct more complex operations like `PICK`, `PLACE`, `PUSH`, `PULL`, and `PRESS`."""

    MOVE = "move"
    """Execute pre_move waypoints to reach the target pose without interacting with any object. No pre-condition is checked; post-condition `reached` is checked after the final pose action. Failure occurs when the operator fails to reach the target pose within the position tolerance within the time limit."""
    GRASP = "grasp"
    """Execute the eef phase (close gripper) at the current position. Pre-condition `released` is checked before the eef phase; post-condition `grasped` is checked after the eef phase. Failure occurs when the post-condition `grasped` is not satisfied (the gripper closes but no object is effectively grasped)."""
    RELEASE = "release"
    """Execute the eef phase (open gripper) at the current position. Pre-condition `grasped` is checked before the eef phase; post-condition `released` is checked after the eef phase. Failure occurs when the post-condition `released` is not satisfied (the gripper opens but the object is still effectively grasped)."""
    PICK = "pick"
    """Execute pre_move → eef (close gripper) → post_move to approach the Stage target and grasp it. Pre-condition `released` is checked before the pre_move phase; the target-specific post-condition `grasped` is checked after the post_move phase. Failure occurs when the Stage target is not grasped."""
    PLACE = "place"
    """Execute pre_move → eef (open gripper) → post_move to approach a target pose and release the held object. Pre-condition `grasped` is checked before the pre_move phase; post-condition `placed` is checked after the post_move phase. Failure occurs when the held object is still grasped or, when a placement target is available, outside placement tolerance."""
    PUSH = "push"
    """Execute pre_move → post_move to approach and push an object to a target pose. No pre-condition is checked; post-condition `displaced` is checked after the post_move phase. Failure occurs when the post-condition `displaced` is not satisfied (the object has not moved beyond the displacement threshold)."""
    PULL = "pull"
    """Execute pre_move → eef (close gripper) → post_move to approach the Stage target, grasp it, and apply an effect trajectory. Target-specific `grasped` conditions are checked after the eef phase and after the post_move phase. Failure occurs when the Stage target is not grasped at either boundary."""
    PRESS = "press"
    """Execute pre_move → eef → post_move to approach and press an object at the target pose. No pre-condition is checked; post-condition `contacted` is checked after the eef phase (at the moment of contact, before retreat). Failure occurs when the post-condition `contacted` is not satisfied (the operator end-effector is not in contact with the target object after the eef phase)."""


class OperationConstraint(str, Enum):
    """Enumeration of possible constraints for the operations."""

    GRASPED = "grasped"
    """Whether the operator is currently grasping an object."""
    RELEASED = "released"
    """Whether the operator is not currently grasping any object."""
    CONTACTED = "contacted"
    """Whether the operator is in contact with the target object."""
    DISPLACED = "displaced"
    """Whether the target object has been displaced from its original pose (e.g., the distance between the current pose of the object and its original pose is greater than a certain threshold) after the operation."""
    REACHED = "reached"
    """Whether the final waypoint's controlled frame is within its pose tolerance."""
    PLACED = "placed"
    """Whether the operator has released the held object AND the held object
    is within tolerance of the target position/orientation."""
    NONE = "none"
    """No constraint."""


class OperationConditionType(str, Enum):
    PERFORM = "perform"
    """The condition for performing the operation. The operator will only perform the operation when the condition is satisfied."""
    SUCCESS = "success"
    """The condition for the success of the operation. The operation is considered successful when the condition is satisfied after performing the operation."""


_Condition = OperationConditionType
OPERATION_CONDITIONS = {
    Operation.MOVE: {
        _Condition.SUCCESS: OperationConstraint.REACHED,
    },
    Operation.GRASP: {
        _Condition.PERFORM: OperationConstraint.RELEASED,
        _Condition.SUCCESS: OperationConstraint.GRASPED,
    },
    Operation.RELEASE: {
        _Condition.PERFORM: OperationConstraint.GRASPED,
        _Condition.SUCCESS: OperationConstraint.RELEASED,
    },
    Operation.PICK: {
        _Condition.PERFORM: OperationConstraint.RELEASED,
        _Condition.SUCCESS: OperationConstraint.GRASPED,
    },
    Operation.PLACE: {
        _Condition.PERFORM: OperationConstraint.GRASPED,
        _Condition.SUCCESS: OperationConstraint.PLACED,
    },
    Operation.PUSH: {
        _Condition.SUCCESS: OperationConstraint.DISPLACED,
    },
    Operation.PULL: {
        _Condition.PERFORM: OperationConstraint.GRASPED,
        _Condition.SUCCESS: OperationConstraint.GRASPED,
    },
    Operation.PRESS: {
        _Condition.SUCCESS: OperationConstraint.CONTACTED,
    },
}


class RandomizationReference(str, Enum):
    """Reference mode for a :class:`PoseRandomRange`.

    Controls how the per-axis ``[min, max]`` ranges are interpreted when
    sampling a randomized pose.
    """

    RELATIVE = "relative"
    """Ranges are additive offsets from the entity's default/initial pose
    (current default behavior)."""
    ABSOLUTE_WORLD = "absolute_world"
    """Ranges are absolute world-frame values — metres for position axes,
    radians for Euler orientation axes. The entity's default pose is ignored
    for any axis that has an explicit range."""
    ABSOLUTE_BASE = "absolute_base"
    """Ranges are absolute values expressed in the operator's base frame.
    The sampled pose is transformed back into world frame before being
    applied. Only valid for operator end-effector randomization."""


class PoseReference(str, Enum):
    """Enumeration of possible pose references for the pose control."""

    WORLD = "world"
    """The pose is defined in the world coordinate system."""
    BASE = "base"
    """The pose is defined in the robot base coordinate system."""
    EEF = "eef"
    """The pose is defined in the current operator eef coordinate system."""
    OBJECT = "object"
    """The pose is defined in the object coordinate system."""
    OBJECT_WORLD = "object_world"
    """The reference is equivalent to moving the origin of the world system to the origin of the object while keeping the coordinate system direction unchanged. The pose is defined in this new coordinate system. The target pose will track the movement of the object after action start, meaning that the target pose will change accordingly as the object moves."""
    EEF_WORLD = "eef_world"
    """The reference is equivalent to moving the origin of the world system to the operator's end-effector position at the moment the action starts, while keeping the coordinate system direction unchanged. The target pose is snapshotted once at action start and does not track subsequent EEF movement."""
    AUTO = "auto"
    """The pose reference is automatically determined based on the context of the operation."""


def _coerce_pose_reference(value: object) -> object:
    """Keep built-in pose references typed while allowing named scene frames."""
    if isinstance(value, str) and not isinstance(value, PoseReference):
        try:
            return PoseReference(value)
        except ValueError:
            return value
    return value


def _validate_pose_reference(
    value: Optional[Union[PoseReference, str]],
) -> Optional[Union[PoseReference, str]]:
    """Reject empty names while allowing the optional component reference."""
    if (
        isinstance(value, str)
        and not isinstance(value, PoseReference)
        and not value.strip()
    ):
        raise ValueError("reference must be a non-empty frame name")
    return value


class ControlledFrameKind(str, Enum):
    """The kind of frame whose pose a waypoint controls."""

    EEF = "eef"
    """Control the operator end-effector pose directly."""
    HELD_OBJECT = "held_object"
    """Control the pose of the object currently held by the operator."""


class ControlledFrameConfig(BaseModel, frozen=True):
    """Frame whose pose is controlled by a waypoint."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: ControlledFrameKind = ControlledFrameKind.EEF
    """Whether the waypoint controls the end effector or the held object."""

    frame: Optional[str] = Field(default=None, min_length=1)
    """Optional object-local frame; omitted means the held object's root frame."""

    @model_validator(mode="after")
    def validate_frame(self) -> Self:
        """Only held-object control can select an object-local frame."""
        if self.kind == ControlledFrameKind.EEF and self.frame is not None:
            raise ValueError("controlled_frame.frame requires kind='held_object'")
        return self


class OrientationGoalKind(str, Enum):
    """Supported orientation-goal semantics."""

    FIXED = "fixed"
    """Constrain the complete orientation to a quaternion."""
    AXIS_ALIGNMENT = "axis_alignment"
    """Constrain only one controlled-frame axis."""


class AxisAlignmentDirection(str, Enum):
    """Allowed direction relationship between aligned axes."""

    SAME = "same"
    """Require the controlled axis to point in the target-axis direction."""
    OPPOSITE = "opposite"
    """Require the controlled axis to point opposite the target-axis direction."""
    EITHER = "either"
    """Treat equal and opposite target-axis directions as equivalent."""


class AxisReference(str, Enum):
    """Reference frame in which a target axis vector is expressed."""

    WORLD = "world"
    """Express the target axis in the world frame."""
    BASE = "base"
    """Express the target axis in the operator base frame."""
    OBJECT = "object"
    """Express the target axis in the stage object or site frame."""


def _validate_unit_vector(value: Position, field_name: str) -> Position:
    """Reject non-finite or non-unit direction vectors."""
    norm_squared = math.fsum(component * component for component in value)
    if not math.isfinite(norm_squared) or not math.isclose(
        norm_squared,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{field_name} must be a finite unit vector")
    return value


class TargetAxisConfig(BaseModel, frozen=True):
    """Target direction for an axis-alignment orientation goal."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    vector: Position
    """Unit target direction expressed in ``reference``."""

    reference: AxisReference
    """Coordinate frame in which ``vector`` is expressed."""

    @field_validator("vector", mode="after")
    @classmethod
    def validate_vector(cls, value: Position) -> Position:
        """Require a finite unit target direction."""
        return _validate_unit_vector(value, "target_axis.vector")


class FixedOrientationGoalConfig(BaseModel, frozen=True):
    """A full-orientation waypoint goal."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal[OrientationGoalKind.FIXED] = OrientationGoalKind.FIXED
    """Discriminator for a fixed-orientation goal."""

    quaternion_xyzw: Orientation
    """Required controlled-frame orientation as an ``xyzw`` quaternion."""

    @field_validator("quaternion_xyzw", mode="after")
    @classmethod
    def validate_quaternion(cls, value: Orientation) -> Orientation:
        """Require and normalize a finite, non-zero quaternion."""
        norm_squared = math.fsum(component * component for component in value)
        if not math.isfinite(norm_squared) or norm_squared <= 1.0e-24:
            raise ValueError("quaternion_xyzw must be finite and non-zero")
        norm = math.sqrt(norm_squared)
        return tuple(float(component / norm) for component in value)


class AxisAlignmentOrientationGoalConfig(BaseModel, frozen=True):
    """A partial orientation goal that aligns one controlled-frame axis."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal[OrientationGoalKind.AXIS_ALIGNMENT] = (
        OrientationGoalKind.AXIS_ALIGNMENT
    )
    """Discriminator for an axis-alignment goal."""

    controlled_axis: Position
    """Unit axis expressed in the controlled frame."""

    target_axis: TargetAxisConfig
    """Target direction and the frame in which that direction is expressed."""

    direction: AxisAlignmentDirection = AxisAlignmentDirection.SAME
    """Whether the controlled axis must be equal, opposite, or either direction."""

    @field_validator("controlled_axis", mode="after")
    @classmethod
    def validate_controlled_axis(cls, value: Position) -> Position:
        """Require a finite unit controlled-frame axis."""
        return _validate_unit_vector(value, "controlled_axis")


OrientationGoalConfig = Annotated[
    Union[FixedOrientationGoalConfig, AxisAlignmentOrientationGoalConfig],
    Field(discriminator="kind"),
]
"""A discriminated full-or-partial orientation goal."""


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


class ArcControlConfig(BaseModel):
    """Configuration for arc (revolute) movement around a pivot axis.

    When attached to a ``PoseControlConfig``, the end-effector traces an arc
    around ``pivot`` instead of moving in a straight line.  The ``position``,
    ``orientation``, and ``rotation`` fields of the parent config are ignored."""

    model_config = ConfigDict(extra="forbid")

    pivot: Union[Position, str]
    """Pivot point for the arc.  Either explicit ``(x, y, z)`` coordinates in the
    coordinate frame given by the parent's ``reference``, or a **string name** of a
    site, body, or joint in the scene XML whose world position is used automatically."""
    axis: Position
    """Unit-direction of the rotation axis (x, y, z)."""
    angle: float
    """Rotation angle in radians.  Positive follows the right-hand rule around ``axis``.
    When ``absolute`` is False (default), this is a relative rotation from the current
    EEF position.  When ``absolute`` is True and ``pivot`` is a joint name, this is
    the target joint angle and the runtime computes the relative rotation automatically."""
    absolute: bool = False
    """When True, ``angle`` is treated as an absolute target joint angle (radians)
    instead of a relative rotation.  Requires ``pivot`` to be a joint name so the
    runtime can read the current joint angle and compute the delta."""
    max_step: float = 0.2
    """Maximum arc sub-step in radians (~11.5 deg).  Smaller values produce smoother
    arcs at the cost of more waypoints."""
    joint_tolerance: PositiveFloat = 0.01
    """Joint-angle tolerance in radians for completing an absolute arc.  Reaching
    one local end-effector target is not sufficient until the named pivot joint is
    also within this tolerance of ``angle``.  Relative arcs ignore this field."""
    timeout_steps: PositiveInt = 1000
    """Maximum aggregate control updates for one absolute arc.  This task-level
    limit remains effective when successive local end-effector targets reset a
    backend controller's per-pose timeout.  Relative arcs ignore this field."""
    reverse: bool = False
    """When True, the arc is traced in the opposite direction around the axis.

    Implemented as ``axis → -axis`` (rather than ``angle → -angle``) so the
    behaviour is correct in both relative and absolute modes:

    - Relative: negating the axis is mathematically equivalent to negating
      the angle, so the rotation direction flips as expected.
    - Absolute: ``angle`` is the target joint value, not a rotation amount.
      Negating ``angle`` would change the goal (e.g. +0.45 → -0.45) and the
      runtime would chase an unreachable target. Flipping the axis preserves
      the goal while reversing the world-frame rotation direction."""

    @model_validator(mode="after")
    def validate_reverse(self):
        """If reverse is True, negate the axis to reverse the rotation direction."""
        if self.reverse:
            self.axis = tuple(-v for v in self.axis)
        return self


class WaypointToleranceConfig(BaseModel):
    """Per-waypoint tolerance override. When set on a waypoint, these values
    take precedence over the operator-level tolerance for that waypoint only.

    Position tolerance can be a single float (L2 norm) or a list of three
    floats ``[x, y, z]`` for per-axis tolerance checking."""

    model_config = ConfigDict(extra="forbid")

    position: Optional[Union[float, List[float]]] = None
    """Position tolerance. A scalar applies as an L2-norm threshold;
    a 3-element list ``[x, y, z]`` checks each axis independently."""
    orientation: Optional[float] = None
    """Orientation tolerance in radians.

    This is quaternion angular distance for a complete orientation and axis
    angular error for an axis-alignment goal.
    """


class PlacedToleranceConfig(BaseModel):
    """Tolerance for the PLACED post-condition. Each dimension can be null
    to skip checking that dimension."""

    model_config = ConfigDict(extra="forbid")

    position: Optional[Union[float, List[Optional[float]]]] = [None, None, None]
    """Position tolerance. Scalar = L2-norm threshold. List ``[x, y, z]`` =
    per-axis thresholds where ``null`` means no constraint on that axis."""

    orientation: Optional[Union[float, List[Optional[float]]]] = [None, None, None]
    """Orientation tolerance in radians. Scalar = quaternion angular distance
    threshold for complete orientations, or axis angular error for an
    axis-alignment goal. List ``[roll, pitch, yaw]`` = per-axis Euler
    thresholds where ``null`` means no constraint on that axis; lists are not
    valid for axis-alignment goals."""


class RandomizationAxisConfig(BaseModel, frozen=True):
    """Randomization bounds and an optional reference for one pose axis."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    range: Tuple[float, float]
    """Inclusive ``[min, max]`` sampling range for this axis."""

    reference: Optional[Union[RandomizationReference, str]] = None
    """Axis-specific reference. ``None`` inherits the pose-level reference."""

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, v: object) -> object:
        if isinstance(v, str) and not isinstance(v, RandomizationReference):
            try:
                return RandomizationReference(v)
            except ValueError:
                return v
        return v


RandomizationAxisSpec = Union[Tuple[float, float], RandomizationAxisConfig]
"""Compact range or expanded per-axis randomization configuration."""


class PoseRandomRange(BaseModel, frozen=True):
    """Per-entity pose randomization bounds.

    Each axis accepts either a compact ``[min, max]`` range or an expanded
    ``{range: [min, max], reference: ...}`` object. An expanded axis reference
    takes precedence over the pose-level ``reference``; an omitted axis
    reference inherits the pose-level value. The pose-level ``reference``
    selects one of three modes, **or** names another entity to track:

    - ``"relative"`` (default): each per-axis ``[min, max]`` range is an
      additive offset applied to the entity's default/initial pose.
    - ``"absolute_world"``: ranges are absolute world-frame values —
      metres for position, radians for Euler orientation. The default
      pose is ignored for any axis that has an explicit range.
    - ``"absolute_base"``: ranges are absolute values expressed in the
      operator's base frame. The sampled pose is then transformed into
      world frame before being applied. Only valid for operator
      end-effector randomization.
    - **Entity name** (e.g. ``"vase1"``): the referenced entity is
      randomized first; its displacement from its default pose is
      computed (``delta = sampled * default⁻¹``) and applied to this
      entity's default pose so they move together. Then the per-axis
      ranges are applied as additive offsets on top, just like
      ``relative`` mode. For an **operator** name, the plain form
      tracks the operator's **base** pose (equivalent to the
      ``"<operator>.base"`` form below).
    - **Operator attribute** (e.g. ``"arm.base"`` or ``"arm.eef"``):
      same delta-carry semantics as the entity-name form, but
      explicitly anchored to the operator's **base** or
      **end-effector** pose. Only ``.base`` / ``.eef`` suffixes are
      recognized, and only for operator names.

    A ``None`` value on an axis (the default) means "do not randomize
    this axis" — it keeps its value from the default pose (in the
    relevant frame) in all modes. Axes are independent, so absolute-mode
    ``x``/``y`` with ``z``/``roll``/``pitch``/``yaw`` left as ``None``
    produces the natural "place anywhere on this rectangle, keep default
    height and orientation" behavior.

    Example YAML entries::

        # Relative (default): sampled as default_pose + offset
        randomization:
          source_block:
            x: [-0.03, 0.03]
            y:
              range: [-0.20, 0.20]
              reference: absolute_world
            collision_radius: 0.04

        # Absolute world-frame: sampled as world-frame coordinates
        randomization:
          vase1:
            reference: absolute_world
            x: [0.10, 0.45]
            y: [-0.15, 0.15]

        # Entity reference: carry flower with vase1, then jitter ±5mm
        randomization:
          vase1:
            reference: absolute_world
            x: [0.22, 0.58]
            y: [-0.32, 0.27]
          flower:
            reference: vase1
            x: [-0.005, 0.005]
            y: [-0.005, 0.005]

        # Operator-base reference: carry vase with the arm's base
        randomization:
          arm:
            base:
              x: [-0.05, 0.05]
              y: [-0.05, 0.05]
          vase:
            reference: arm.base
            x: [-0.005, 0.005]
            y: [-0.005, 0.005]
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    x: Optional[RandomizationAxisSpec] = None
    """X range in metres, optionally with its own reference."""
    y: Optional[RandomizationAxisSpec] = None
    """Y range in metres, optionally with its own reference."""
    z: Optional[RandomizationAxisSpec] = None
    """Z range in metres, optionally with its own reference."""
    roll: Optional[RandomizationAxisSpec] = None
    """Roll range in radians, optionally with its own reference."""
    pitch: Optional[RandomizationAxisSpec] = None
    """Pitch range in radians, optionally with its own reference."""
    yaw: Optional[RandomizationAxisSpec] = None
    """Yaw range in radians, optionally with its own reference."""
    reference: Union[RandomizationReference, str] = RandomizationReference.RELATIVE
    """One of the :class:`RandomizationReference` modes (``"relative"``,
    ``"absolute_world"``, ``"absolute_base"``), the **name of another
    entity**, or an **operator attribute** (``"<operator>.base"`` /
    ``"<operator>.eef"``). An entity/attribute reference causes this
    entry to track the referenced pose's displacement (delta-carry) and
    then apply the per-axis ranges as relative offsets on top. A plain
    operator name is equivalent to ``"<operator>.base"``."""
    collision_radius: float = 0.05
    """Approximate bounding radius used for pairwise collision rejection (metres)."""

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, v: object) -> object:
        if isinstance(v, str) and not isinstance(v, RandomizationReference):
            try:
                return RandomizationReference(v)
            except ValueError:
                return v  # entity name — validated at sample time
        return v

    def axis_range(self, axis: str) -> Optional[Tuple[float, float]]:
        """Return one axis's concrete sampling range."""
        value = getattr(self, axis)
        if isinstance(value, RandomizationAxisConfig):
            return value.range
        return value

    def axis_reference(
        self,
        axis: str,
    ) -> Union[RandomizationReference, str]:
        """Return one axis's effective reference after fallback resolution."""
        value = getattr(self, axis)
        if isinstance(value, RandomizationAxisConfig) and value.reference is not None:
            return value.reference
        return self.reference

    def references(self) -> Tuple[Union[RandomizationReference, str], ...]:
        """Return every effective reference declared by this pose range."""
        references = [
            self.axis_reference(axis)
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        ]
        return tuple(dict.fromkeys(references))


class PoseRandomizationConfig(BaseModel, frozen=True):
    """A choice among one or more independently configured pose regions.

    Each region is a complete :class:`PoseRandomRange`, so it owns its axis
    ranges, reference mode, and collision radius.  The legacy direct
    ``PoseRandomRange`` form remains valid wherever this wrapper is accepted.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    regions: List[PoseRandomRange] = Field(min_length=1)
    """Non-empty collection of candidate regions sampled at reset."""


PoseRandomizationSpec = Union[PoseRandomRange, PoseRandomizationConfig]
"""Canonical single- or multi-region pose randomization specification."""


def pose_randomization_regions(
    spec: PoseRandomizationSpec,
) -> Tuple[PoseRandomRange, ...]:
    """Return the concrete regions represented by a randomization spec."""
    if isinstance(spec, PoseRandomizationConfig):
        return tuple(spec.regions)
    return (spec,)


class PoseControlConfig(BaseModel):
    """Configuration for the pose control"""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    position: Optional[Position] = None
    """Target controlled-frame position as three floats ``(x, y, z)``."""
    orientation: Optional[Orientation] = None
    """Legacy full controlled-frame orientation as an ``xyzw`` quaternion."""
    rotation: Optional[Rotation] = None
    """Legacy full controlled-frame rotation as Euler angles in ``rpy`` order."""
    controlled_frame: ControlledFrameConfig = ControlledFrameConfig()
    """The frame whose pose this waypoint controls; defaults to the end effector."""
    orientation_goal: Optional[OrientationGoalConfig] = None
    """Optional full or partial orientation goal for the controlled frame."""
    reference: PoseReference = PoseReference.AUTO
    """The reference frame for the pose control."""
    static: bool = False
    """Whether the reference frame should be snapshotted at action start.

    By default, ``OBJECT`` / ``OBJECT_WORLD`` references are re-evaluated
    on every control tick, so the target tracks the object as it moves.
    That is the correct behavior when the object moves independently of
    the gripper. However, when the gripper is *rigidly gripping* the
    object, a tracking target is unreachable — the reference frame moves
    with the gripper, so the residual never closes.

    Set ``static: true`` to freeze the reference pose at the first tick
    of this waypoint, giving a fixed world-frame target. ``EEF`` /
    ``EEF_WORLD`` are always snapshotted and ignore this flag."""
    relative: bool = False
    """Whether the pose control is relative to the current pose. The current pose is determined by the reference frame. """
    use_slerp: bool = False
    """Whether to use SLERP interpolation for smooth orientation transitions."""
    max_linear_step: float = 0.0
    """Maximum Cartesian translation step (metres) applied per control tick.
    When > 0, the runtime moves toward the target position incrementally instead
    of commanding the full translation at once."""
    max_angular_step: float = 0.0
    """Maximum orientation step (radians) applied per control tick.
    When > 0, the runtime SLERPs toward the target orientation incrementally
    instead of commanding the full rotation at once."""
    arc: Optional[ArcControlConfig] = None
    """Optional arc movement configuration. When set, the end-effector traces an arc
    around the specified pivot instead of moving in a straight line to the target position."""
    tolerance: Optional[WaypointToleranceConfig] = None
    """Optional per-waypoint tolerance override. When set, these values take
    precedence over the operator-level tolerance for this waypoint only."""
    randomization: Optional[PoseRandomRange] = None
    """Optional per-waypoint pose randomization. When set, a random offset is
    sampled from these ranges and added to the waypoint position/orientation
    at the start of each episode."""

    @model_validator(mode="after")
    def validate_orientation_goal(self) -> Self:
        """Reject ambiguous or unsupported orientation-goal combinations."""
        if self.orientation_goal is not None and self.arc is not None:
            raise ValueError("orientation_goal does not support arc movement")
        if (
            self.controlled_frame.kind == ControlledFrameKind.HELD_OBJECT
            and self.arc is not None
        ):
            raise ValueError(
                "held_object controlled_frame does not support arc movement"
            )
        if self.orientation_goal is None:
            return self
        if self.orientation is not None or self.rotation is not None:
            raise ValueError(
                "orientation_goal cannot be combined with orientation or rotation"
            )
        if self.randomization is not None and any(
            getattr(self.randomization, axis) is not None
            for axis in ("roll", "pitch", "yaw")
        ):
            raise ValueError(
                "orientation_goal cannot be combined with rotational randomization"
            )
        if isinstance(
            self.orientation_goal,
            AxisAlignmentOrientationGoalConfig,
        ):
            if self.relative:
                raise ValueError(
                    "axis_alignment orientation_goal does not support relative=true"
                )
        return self


class EefControlConfig(BaseModel):
    """Configuration for the end-effector control"""

    model_config = ConfigDict(extra="forbid")

    close: bool
    """Whether to close the end-effector. True for closing the end-effector, False for opening the end-effector. This will set the end-effector joint positions to the lower limit or upper limit defined in the environment model."""
    joint_positions: List[float] = []
    """The target joint positions for the end-effector control. The order and meaning of the joint positions depend on the specific end-effector used in the environment."""
    require_grasp: bool = False
    """When closing on a Stage target, require the backend to verify that target is
    physically grasped before reporting the end-effector primitive as reached."""

    @model_validator(mode="after")
    def validate_require_grasp(self):
        """A grasp completion requirement is only meaningful for closing."""
        if self.require_grasp and not self.close:
            raise ValueError("require_grasp=true requires close=true")
        return self


class StageControlConfig(BaseModel):
    """Configuration for the control of each stage of the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    pre_move: List[PoseControlConfig] = Field(default_factory=list)
    """Optional pose controls to execute before the main stage action."""
    post_move: List[PoseControlConfig] = Field(default_factory=list)
    """Optional pose controls to execute after the main stage action."""
    eef: Optional[EefControlConfig] = None
    """The configuration for the end-effector control in this stage. If not specified, no end-effector control will be performed in this stage."""
    placed_reference: str = "object"
    """Target reference for the PLACED post-condition. ``'object'`` uses the
    stage object's current pose (the destination); ``'pre_move'`` uses the
    last pre_move waypoint resolved position. When the stage has no object,
    ``'pre_move'`` is always used regardless of this setting. A resolved
    held-object pre-move goal is authoritative regardless of this legacy
    selector."""
    placed_tolerance: Optional[PlacedToleranceConfig] = PlacedToleranceConfig()
    """Per-stage tolerance override for the PLACED post-condition. Falls back
    to the operator-level placed tolerance. If neither level configures a
    non-null position or orientation tolerance, placement degrades to
    released-only."""
    displacement_threshold: Optional[float] = None
    """Per-stage threshold (meters) for the DISPLACED post-condition. When set,
    overrides the backend default of 0.01 m used by ``is_object_displaced``.
    Only meaningful for operations whose success constraint is DISPLACED
    (e.g., ``push``)."""


class StageConfig(BaseModel):
    """Configuration for each stage of the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    """The optional human-readable name of this stage."""
    object: str
    """The name of the object to be manipulated in this stage. The object should be defined in the environment and should have a unique name. An empty name means that the corresponding operation does not involve the target object; the target pose is obtained from the corresponding param."""
    site: Optional[str] = None
    """Optional site/body/geom/joint name used as the reference frame for
    ``reference: object_world`` / ``reference: object`` waypoints in this
    stage. When set, its world pose replaces ``object``'s pose as the
    reference origin (and, for ``reference: object``, also as the
    reference orientation). When ``None``, the ``object`` body's pose is
    used as before. This field only affects pose reference resolution —
    ``object`` is still used for contact detection, GS rendering mask,
    ``set_pose``/randomization, and arc pivot fallback."""
    operation: Operation
    """The operation that the AutoAtom operator performs in this stage."""
    param: StageControlConfig
    """The parameter for the operation."""
    operator: str = ""
    """The name of the operator that performs the operation in this stage. The operator should be defined in the environment and should have a unique name. If there is only one operator in the environment, this field can be left empty, and the operator will automatically select that operator to perform the operation."""
    blocking: bool = True
    """Whether the operator should wait for the completion of the operation before proceeding to the next stage. If set to False, the operator will proceed to the next stage immediately after initiating the operation. However, if the operator in the next stage is the same as the current stage, the operator will still wait for the completion of the operation to avoid conflicts."""


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


class ObjectMotionExecutionConfig(BaseModel, frozen=True):
    """Kinematic object-transport settings for ``object_only`` execution."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    max_linear_step: PositiveFloat = 0.02
    """Default maximum object translation per controller update, in metres."""

    max_angular_step: PositiveFloat = 0.15
    """Default maximum object rotation per controller update, in radians."""


class ExecutionConfig(BaseModel, frozen=True):
    """TaskRunner execution policy for one runnable task file."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    mode: ExecutionMode = ExecutionMode.PHYSICAL
    """Execution strategy. ``object_only`` removes configured operators and
    directly moves the logically picked object along held-object waypoints."""
    object_motion: ObjectMotionExecutionConfig = ObjectMotionExecutionConfig()
    """Kinematic object-motion limits used only by ``object_only`` mode."""
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


class OperatorRandomizationConfig(BaseModel):
    """Randomization options for an operator.

    ``base`` controls the operator base pose returned by ``get_base_pose()``.
    For mocap operators this is the virtual base frame; for joint-mode
    operators this is the robot base reference frame.

    ``eef`` controls the operator home end-effector pose in world frame. After
    sampling, reset re-homes the operator to the sampled EEF pose.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    base: Optional[PoseRandomizationSpec] = None
    """Optional single- or multi-region randomization for the operator base."""
    eef: Optional[PoseRandomizationSpec] = None
    """Optional single- or multi-region randomization for the end effector."""


class PoseAxisConfig(BaseModel, frozen=True):
    """One absolute pose component with an optional reference override."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    value: float
    """Component value in the selected reference frame."""
    reference: Optional[Union[PoseReference, str]] = None
    """Axis-specific reference; ``None`` inherits the component or pose reference."""

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, value: object) -> object:
        return _coerce_pose_reference(value)

    @field_validator("value", mode="after")
    @classmethod
    def _validate_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        return value


class PosePositionConfig(BaseModel, frozen=True):
    """Optional x/y/z initial-pose components with a component-level reference."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    reference: Optional[Union[PoseReference, str]] = None
    """Reference for all position components; axis references take precedence."""

    _coerce_reference = field_validator("reference", mode="before")(
        _coerce_pose_reference
    )

    _validate_reference = field_validator("reference", mode="after")(
        _validate_pose_reference
    )

    x: Optional[Union[float, PoseAxisConfig]] = None
    """X component, optionally with an axis-specific reference."""
    y: Optional[Union[float, PoseAxisConfig]] = None
    """Y component, optionally with an axis-specific reference."""
    z: Optional[Union[float, PoseAxisConfig]] = None
    """Z component, optionally with an axis-specific reference."""


class PoseOrientationConfig(BaseModel, frozen=True):
    """Optional roll/pitch/yaw components with a component-level reference."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    reference: Optional[Union[PoseReference, str]] = None
    """Reference for all orientation components; axis references take precedence."""

    _coerce_reference = field_validator("reference", mode="before")(
        _coerce_pose_reference
    )

    _validate_reference = field_validator("reference", mode="after")(
        _validate_pose_reference
    )

    roll: Optional[Union[float, PoseAxisConfig]] = None
    """Roll angle in radians, optionally with an axis-specific reference."""
    pitch: Optional[Union[float, PoseAxisConfig]] = None
    """Pitch angle in radians, optionally with an axis-specific reference."""
    yaw: Optional[Union[float, PoseAxisConfig]] = None
    """Yaw angle in radians, optionally with an axis-specific reference."""


class PoseOverrideConfig(BaseModel, frozen=True):
    """A partial pose override expressed in a named reference frame.

    This is the one configuration model used for initial object, camera, and
    operator poses.  It intentionally contains only pose data; motion-specific
    fields belong to :class:`PoseControlConfig`.

    ``position`` and ``orientation`` are optional.  An omitted component keeps
    the current pose component after transforming the fallback pose into the
    selected reference frame.  ``orientation`` accepts either an ``xyzw``
    quaternion (four values), roll/pitch/yaw Euler angles (three values), or an
    expanded ``{roll, pitch, yaw}`` mapping whose components may override the
    orientation-level or pose-level reference.  Structured ``position`` and
    ``orientation`` mappings may set their own ``reference``; the precedence is
    axis-level, component-level, then pose-level.

    ``reference`` accepts the built-in :class:`PoseReference` values and a
    named MuJoCo site/body/geom/joint.  Which references are legal is checked
    by the owner-specific backend seam (objects/cameras accept scene frames;
    operator EEF poses may additionally use ``base`` and operator frame
    aliases).
    Named-frame poses are resolved once during backend setup/reset; they do not
    continue to follow an articulated frame during execution.

    Example YAML::

        initial_pose:
          source_block:
            position: [0.1, 0.0, 0.078]
            orientation: [0, 0, 0, 1]

        task_operators:
          arm:
            initial_state:
              base_pose:
                reference: door__handle_grasp_center
                position: [0.0, 0.45, 0.30]
                orientation: [0, 0, 0, 1]

        # Component-level reference; axis-level reference overrides it.
        task_operators:
          arm:
            initial_state:
              base_pose:
                reference: door__handle_grasp_center
                position:
                  x: 0.2474
                  y: -0.4666
                  z: {value: -0.10, reference: world}
                orientation:
                  reference: world
                  roll: 0.0
                  pitch: 0.2
                  yaw: 0.0
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    position: Optional[Union[Position, PosePositionConfig]] = None
    """Position tuple or expanded x/y/z components in the selected frames."""
    orientation: Optional[Union[Rotation, Orientation, PoseOrientationConfig]] = None
    """Quaternion, RPY tuple, or expanded roll/pitch/yaw components."""
    reference: Union[PoseReference, str] = PoseReference.WORLD
    """Built-in pose reference or a named scene element."""

    @field_validator("position", mode="before")
    @classmethod
    def _validate_position_shape(cls, value: object) -> object:
        """Reject malformed positions while the configuration is loaded.

        The tuple annotation also enforces this shape, but doing the check at
        the input boundary gives a stable, domain-specific error instead of a
        union-branch error from Pydantic.  Non-sequence values are left to the
        type validator so callers still receive the normal type diagnostic.
        """
        if value is None or isinstance(value, (str, bytes, Mapping)):
            return value
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            return value
        if length != 3:
            raise ValueError("position must contain exactly three values")
        return value

    @field_validator("position", mode="after")
    @classmethod
    def _validate_position_values(
        cls,
        value: Optional[Union[Position, PosePositionConfig]],
    ) -> Optional[Union[Position, PosePositionConfig]]:
        """Reject non-finite position coordinates."""
        if isinstance(value, tuple) and not all(
            math.isfinite(component) for component in value
        ):
            raise ValueError("position must contain only finite values")
        return value

    @field_validator("orientation", mode="before")
    @classmethod
    def _validate_orientation_shape(cls, value: object) -> object:
        """Reject orientation vectors other than RPY or quaternion forms."""
        if value is None or isinstance(
            value,
            (str, bytes, Mapping, PoseOrientationConfig),
        ):
            return value
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            return value
        if length not in (3, 4):
            raise ValueError(
                "orientation must contain exactly three RPY values or "
                "four quaternion values"
            )
        return value

    @field_validator("orientation", mode="after")
    @classmethod
    def _validate_orientation_values(
        cls,
        value: Optional[Union[Rotation, Orientation, PoseOrientationConfig]],
    ) -> Optional[Union[Rotation, Orientation, PoseOrientationConfig]]:
        """Reject non-finite angles and zero quaternions at the config seam."""
        if value is None or isinstance(value, PoseOrientationConfig):
            return value
        if not all(math.isfinite(component) for component in value):
            raise ValueError("orientation must contain only finite values")
        if len(value) == 4:
            norm_squared = math.fsum(component * component for component in value)
            if norm_squared <= 1.0e-24:
                raise ValueError("orientation quaternion must be finite and non-zero")
        return value

    @field_validator("reference", mode="before")
    @classmethod
    def _coerce_reference(cls, value: object) -> object:
        """Keep built-in references typed while allowing named scene frames."""
        return _coerce_pose_reference(value)

    @field_validator("reference", mode="after")
    @classmethod
    def _validate_reference(
        cls, value: Union[PoseReference, str]
    ) -> Union[PoseReference, str]:
        """Reject an empty named frame before backend resolution."""
        return _validate_pose_reference(value)  # type: ignore[return-value]

    def axis_references(self) -> Tuple[Union[PoseReference, str], ...]:
        """Return global, component-level, and axis-level references."""
        references: list[Union[PoseReference, str]] = [self.reference]
        for component in (self.position, self.orientation):
            if not isinstance(component, BaseModel):
                continue
            component_reference = getattr(component, "reference", None)
            if component_reference is not None:
                references.append(component_reference)
            for axis in type(component).model_fields:
                value = getattr(component, axis)
                if isinstance(value, PoseAxisConfig) and value.reference is not None:
                    references.append(value.reference)
        return tuple(dict.fromkeys(references))


class AutoAtomConfig(BaseModel):
    """Configuration for the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    stages: List[StageConfig]
    """A list of StageConfig objects, each representing a stage of the AutoAtom operator. The stages are executed in the order they are defined in the list."""
    env_name: str
    """The registered environment name used to resolve the basis environment instance for the selected scene."""
    seed: int = 0
    """The random seed for the AutoAtom operator. This is used to ensure reproducibility of the operator's behavior."""
    initial_pose: Dict[str, PoseOverrideConfig] = Field(default_factory=dict)
    """Per-object initial pose overrides applied after keyframe reset, before
    randomization.  Keys are object names matching the MuJoCo body (or stage
    ``object`` field).  Supports both freejoint and static bodies."""
    randomization: Dict[
        str,
        Union[PoseRandomizationSpec, OperatorRandomizationConfig],
    ] = {}
    """Per-entity pose randomization applied at each reset.

    Objects accept either a direct ``PoseRandomRange`` or a
    ``PoseRandomizationConfig`` containing one or more disjoint regions.

    Operators must use ``OperatorRandomizationConfig`` with explicit
    ``base`` and/or ``eef`` sub-entries. Each sub-entry accepts either form.
    The direct ``PoseRandomRange`` shorthand is rejected at sample time for
    operator entries.
    """
    camera_initial_pose: Dict[str, PoseOverrideConfig] = Field(default_factory=dict)
    """Per-camera initial pose overrides applied at each reset, before
    camera randomization records its defaults.

    Keys are camera names as defined in the MuJoCo XML. Each entry may
    set ``position`` and/or ``orientation`` (4-float quaternion xyzw or
    3-float Euler roll/pitch/yaw in radians). Components omitted fall
    back to the XML value.

    Example YAML::

        camera_initial_pose:
          env1_cam:
            position: [2.4, 0.6, -0.1]
            orientation: [-0.5, 0.5, 0.5, 0.5]   # xyzw
    """
    camera_randomization: Dict[str, PoseRandomRange] = Field(default_factory=dict)
    """Per-camera pose randomization applied at each reset.

    Keys are camera names as defined in the MuJoCo XML model.  Each entry
    is a ``PoseRandomRange`` controlling which axes are randomized and how.

    Only ``relative`` (default) and ``absolute_world`` reference modes are
    supported.  ``absolute_base`` and entity-name references are rejected
    because cameras have no operator base frame and do not participate in
    entity dependency ordering.

    Example YAML::

        camera_randomization:
          env1_cam:
            x: [-0.05, 0.05]
            y: [-0.05, 0.05]
            pitch: [-0.1, 0.1]
          env0_cam:
            reference: absolute_world
            x: [0.8, 1.0]
            y: [-0.1, 0.1]
            z: [0.4, 0.6]
    """
    randomization_debug: bool = False

    @field_validator(
        "initial_pose",
        "randomization",
        "camera_initial_pose",
        "camera_randomization",
        mode="before",
    )
    @classmethod
    def _strip_none_keys(cls, v: object) -> object:
        """Remove ``None``-valued keys from nested mapping entries.

        Hydra/OmegaConf merges override ``key: null`` as ``key: None``
        rather than deleting the key.  Stripping them here lets child
        configs cleanly switch between forms without triggering Pydantic
        ``extra="forbid"`` errors.
        """
        if not isinstance(v, dict):
            return v

        def _strip(value: object) -> object:
            if isinstance(value, dict):
                return {
                    key: _strip(nested)
                    for key, nested in value.items()
                    if nested is not None
                }
            if isinstance(value, list):
                return [_strip(item) for item in value]
            return value

        return _strip(v)

    """When True the first N resets cycle through extreme poses (each axis at its min/max, then all-min and all-max) before switching to random sampling.  Use this to verify that configured ranges are not too large."""


class OperatorInitialState(BaseModel, frozen=True):
    """Optional override for an operator's home control state applied at reset.

    ``joint_positions`` is the raw-qpos representation for operator-owned arm
    joints. The gripper remains controlled by the separate ``eef`` field.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    joint_positions: Dict[str, Union[float, List[float]]] = Field(default_factory=dict)
    """Raw qpos values for the operator's arm joints, keyed by joint name.

    Values are applied through the operator home seam after the low-level
    environment reset. Operator actuator bindings currently expose one-DOF
    arm joints, so use scalars (a one-value list is also accepted). These
    values bypass any EEF user-space mapper.
    """

    @field_validator("joint_positions", mode="after")
    @classmethod
    def _validate_joint_positions(
        cls, value: Dict[str, Union[float, List[float]]]
    ) -> Dict[str, Union[float, List[float]]]:
        for name, position in value.items():
            if not str(name).strip():
                raise ValueError("joint_positions keys must be non-empty names")
            values = position if isinstance(position, list) else [position]
            if not values:
                raise ValueError(
                    f"joint_positions['{name}'] must contain at least one value"
                )
            if not all(math.isfinite(float(component)) for component in values):
                raise ValueError(
                    f"joint_positions['{name}'] must contain only finite values"
                )
        return value

    eef_pose: Optional[
        Union[Tuple[float, float, float, float, float, float], PoseOverrideConfig]
    ] = None
    """Override for the operator's home end-effector pose.

    Supports two input forms:
    1. Compact six-value form: [x, y, z, yaw, pitch, roll]
    2. Structured dict: {position: [x,y,z], orientation: [roll,pitch,yaw] or [x,y,z,w]}
       - Both position and orientation are optional in structured format
       - orientation can be Euler angles (3 floats) or quaternion (4 floats)

    When omitted the keyframe value is kept."""

    @field_validator("eef_pose", mode="before")
    @classmethod
    def _validate_legacy_eef_shape(cls, value: object) -> object:
        """Require the flat legacy EEF form to contain exactly six values."""
        if value is None or isinstance(
            value, (PoseOverrideConfig, Mapping, str, bytes)
        ):
            return value
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            return value
        if length != 6:
            raise ValueError(
                "eef_pose legacy form must contain exactly six values: "
                "[x, y, z, yaw, pitch, roll]"
            )
        return value

    @field_validator("eef_pose", mode="after")
    @classmethod
    def _validate_eef_values(
        cls,
        value: Optional[
            Union[Tuple[float, float, float, float, float, float], PoseOverrideConfig]
        ],
    ) -> Optional[
        Union[Tuple[float, float, float, float, float, float], PoseOverrideConfig]
    ]:
        """Reject non-finite values in the flat EEF form."""
        if value is None or isinstance(value, PoseOverrideConfig):
            return value
        if not all(math.isfinite(component) for component in value):
            raise ValueError("eef_pose must contain only finite values")
        return value

    eef: Optional[float] = None
    """Override value for the end-effector/gripper control.
    When omitted the keyframe value is kept."""

    @field_validator("eef", mode="after")
    @classmethod
    def _validate_eef_control(cls, value: Optional[float]) -> Optional[float]:
        """Reject non-finite gripper controls before they reach a backend."""
        if value is not None and not math.isfinite(value):
            raise ValueError("eef must be finite")
        return value

    base_pose: Optional[PoseOverrideConfig] = None
    """Override for the operator's base pose.

    ``reference: world`` keeps the historical world-frame behavior.  A named
    site/body/geom/joint expresses the base pose relative to that scene frame;
    the resolved world pose is sampled at setup/reset and then held fixed.
    """

    @model_validator(mode="after")
    def _validate_home_pose_sources(self) -> Self:
        if self.joint_positions and self.eef_pose is not None:
            raise ValueError(
                "joint_positions cannot be combined with eef_pose; configure "
                "one canonical arm home representation"
            )
        return self


class OperatorConfig(BaseModel):
    """Configuration for constructing an operator instance from YAML."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    """The unique operator name referenced by task stages.
    Defaults to empty; populated from the dict key in ``TaskFileConfig.task_operators``
    during validation, so YAML entries do not need to repeat the name."""

    initial_state: Optional[OperatorInitialState] = None
    """Optional initial control state applied to this operator on every reset.
    Overrides the keyframe-defined values for the specified fields."""


def _phase_waypoint_count(stage: StageConfig, phase: TaskPhase) -> int:
    """Return the number of selectable configured points in a stage phase."""
    if phase == TaskPhase.PRE_MOVE:
        return len(stage.param.pre_move)
    if phase == TaskPhase.POST_MOVE:
        return len(stage.param.post_move)
    if stage.operation in {
        Operation.GRASP,
        Operation.RELEASE,
        Operation.PICK,
        Operation.PLACE,
        Operation.PULL,
        Operation.PRESS,
    }:
        return 1
    if stage.operation == Operation.PUSH and stage.param.eef is not None:
        return 1
    return 0


class TaskFileConfig(BaseModel):
    """Top-level YAML schema for a runnable task file."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="allow",
        populate_by_name=True,
    )

    backend: ImportString
    """The backend to execute this task file. The backend should be registered in the ComponentRegistry and should be compatible with the selected scene."""
    task: AutoAtomConfig
    """The task-level configuration describing stages, scene, and environment selection."""
    execution: ExecutionConfig = ExecutionConfig()
    """Runner execution policy. Defaults preserve one-control-tick updates over
    the complete task."""
    task_operators: Dict[str, OperatorConfig] = {}
    """The operator definitions available to the selected backend for this task file,
    keyed by operator name. Using a mapping (rather than a list) lets Hydra overrides
    target individual operators by key, e.g. ``task_operators.arm.control.tolerance.position=0.01``.
    Use ``task_operators`` in YAML; ``env.operators`` is reserved for environment-level operator bindings."""

    @model_validator(mode="before")
    @classmethod
    def _reject_top_level_execution_fields(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        for field_name in (
            "interval_selection",
            "update_boundary",
            "render_internal_updates",
            "max_internal_updates_per_update",
            "max_fast_forward_updates",
        ):
            if field_name in value:
                target = (
                    "execution.interval_selection.max_fast_forward_updates"
                    if field_name == "max_fast_forward_updates"
                    else f"execution.{field_name}"
                )
                raise ValueError(
                    f"Top-level {field_name} is not supported; use {target} instead"
                )
        return value

    @field_validator("task_operators", mode="after")
    @classmethod
    def _populate_operator_names(
        cls, value: Dict[str, OperatorConfig]
    ) -> Dict[str, OperatorConfig]:
        for key, op in value.items():
            if not op.name:
                op.name = key
            elif op.name != key:
                raise ValueError(
                    f"task_operators key '{key}' does not match operator name '{op.name}'. "
                    "Either omit the name field or make it match the key."
                )
        return value

    @model_validator(mode="after")
    def _validate_interval_selection(self) -> "TaskFileConfig":
        selection = self.execution.interval_selection
        if selection is None:
            return self

        stages_by_name: Dict[str, List[Tuple[int, StageConfig]]] = {}
        for index, stage in enumerate(self.task.stages):
            effective_name = stage.name or f"stage_{index}"
            stages_by_name.setdefault(effective_name, []).append((index, stage))

        phase_order = {
            TaskPhase.PRE_MOVE: 0,
            TaskPhase.EEF: 1,
            TaskPhase.POST_MOVE: 2,
        }
        side_order = {
            KeypointSide.BEFORE: 0,
            KeypointSide.AFTER: 1,
        }

        def resolve(
            field_name: str,
            keypoint: TaskKeypointConfig,
        ) -> Tuple[int, int, int, int]:
            matches = stages_by_name.get(keypoint.stage, [])
            if not matches:
                available = ", ".join(stages_by_name) or "<none>"
                raise ValueError(
                    f"execution.interval_selection.{field_name}.stage "
                    f"{keypoint.stage!r} "
                    f"does not match a task stage; available stages: {available}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"execution.interval_selection.{field_name}.stage "
                    f"{keypoint.stage!r} "
                    "is ambiguous because multiple stages use that name"
                )

            stage_index, stage = matches[0]
            count = _phase_waypoint_count(stage, keypoint.phase)
            if count == 0:
                raise ValueError(
                    f"execution.interval_selection.{field_name} references phase "
                    f"{keypoint.phase.value!r}, but stage {keypoint.stage!r} "
                    "does not execute that phase"
                )
            if keypoint.waypoint >= count:
                raise ValueError(
                    f"execution.interval_selection.{field_name}.waypoint "
                    f"{keypoint.waypoint} is out of range for "
                    f"{keypoint.stage}.{keypoint.phase.value}; expected 0..{count - 1}"
                )
            if keypoint.side is None:
                raise ValueError(
                    f"execution.interval_selection.{field_name}.side was not resolved"
                )
            return (
                stage_index,
                phase_order[keypoint.phase],
                int(keypoint.waypoint),
                side_order[keypoint.side],
            )

        start_order = resolve("start", selection.start)
        stop_order = resolve("stop", selection.stop)
        if start_order > stop_order:
            raise ValueError(
                "execution.interval_selection.start must not come after "
                "execution.interval_selection.stop in task execution order"
            )
        return self
