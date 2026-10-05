"""orientation configuration models (split from the former auto_atom.framework monolith)."""

import math
from enum import Enum
from typing import Annotated, List, Literal, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from auto_atom.config.primitives import Orientation, Position


class OrientationGoalKind(str, Enum):
    """Supported orientation-goal semantics."""

    FIXED = "fixed"
    """Constrain the complete orientation to a quaternion."""
    AXIS_ALIGNMENT = "axis_alignment"
    """Constrain only one controlled-frame axis."""
    NEAREST_FEASIBLE = "nearest_feasible"
    """Pick the feasible orientation nearest to the current one."""


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


AngleInterval = Tuple[float, float]
"""Inclusive ``[low, high]`` angle interval in radians."""


def _validate_intervals(
    intervals: Optional[Tuple[AngleInterval, ...]],
    field_name: str,
    bound: float,
) -> Optional[Tuple[AngleInterval, ...]]:
    """Require ordered, finite intervals inside ``±bound``.

    Rounded spellings such as ``1.5708`` for ``pi/2`` overshoot the bound by
    a few microradians; overshoot up to 1e-3 rad is clamped to the bound.
    """
    if intervals is None:
        return None
    if not intervals:
        raise ValueError(f"{field_name} must list at least one interval")
    clamped: List[AngleInterval] = []
    for low, high in intervals:
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"{field_name} intervals must be finite")
        if low > high:
            raise ValueError(f"{field_name} interval [{low}, {high}] is not ordered")
        if low < -bound - 1e-3 or high > bound + 1e-3:
            raise ValueError(
                f"{field_name} interval [{low}, {high}] lies outside "
                f"[-{bound:.6f}, {bound:.6f}]"
            )
        clamped.append((max(low, -bound), min(high, bound)))
    return tuple(clamped)


class AxisRangeConstraintFrame(str, Enum):
    """Frame in which a constrained axis is expressed."""

    CONTROLLED = "controlled"
    """The waypoint's controlled frame (the EEF or the held object's frame)."""
    EEF = "eef"
    """The end effector, through the grasp captured when the object was taken."""


class AxisRangeConfig(BaseModel, frozen=True):
    """Allowed directions of one axis, as elevation and azimuth intervals.

    Angles are measured in the goal's ``reference`` frame: elevation is the
    angle between the axis and that frame's xy plane, in ``[-pi/2, pi/2]``;
    azimuth is the angle of the axis's xy projection from +x, in
    ``[-pi, pi]``. Each field lists alternative intervals, and the axis must
    fall inside one of them; ``None`` leaves that angle free. An interval that
    wraps through ``±pi`` is written as two intervals.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    axis: Position
    """Unit axis, expressed in the frame selected by ``frame``."""

    elevation: Optional[Tuple[AngleInterval, ...]] = None
    """Alternative allowed elevation intervals in radians, or free."""

    azimuth: Optional[Tuple[AngleInterval, ...]] = None
    """Alternative allowed azimuth intervals in radians, or free."""

    @field_validator("axis", mode="after")
    @classmethod
    def validate_axis(cls, value: Position) -> Position:
        """Require a finite unit axis."""
        return _validate_unit_vector(value, "axis")

    @field_validator("elevation", mode="after")
    @classmethod
    def validate_elevation(
        cls, value: Optional[Tuple[AngleInterval, ...]]
    ) -> Optional[Tuple[AngleInterval, ...]]:
        """Require ordered elevation intervals inside ``±pi/2``."""
        return _validate_intervals(value, "elevation", math.pi / 2.0)

    @field_validator("azimuth", mode="after")
    @classmethod
    def validate_azimuth(
        cls, value: Optional[Tuple[AngleInterval, ...]]
    ) -> Optional[Tuple[AngleInterval, ...]]:
        """Require ordered azimuth intervals inside ``±pi``."""
        return _validate_intervals(value, "azimuth", math.pi)


class AxisRangeConstraintConfig(AxisRangeConfig, frozen=True):
    """An additional axis constraint of a ``nearest_feasible`` goal."""

    frame: AxisRangeConstraintFrame = AxisRangeConstraintFrame.CONTROLLED
    """Frame in which ``axis`` is expressed."""

    @model_validator(mode="after")
    def validate_constrains_something(self) -> Self:
        """An axis without any interval would accept every orientation."""
        if self.elevation is None and self.azimuth is None:
            raise ValueError("an axis constraint needs elevation or azimuth intervals")
        return self


class OrientationClearanceConfig(BaseModel, frozen=True):
    """Collision clearance a ``nearest_feasible`` candidate must keep."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    bodies: Tuple[str, ...]
    """Scene bodies whose subtrees' colliding geoms are obstacles."""

    margin: float = Field(default=0.003, ge=0.0)
    """Minimum distance in metres between the probed geometry and obstacles."""

    step: PositiveFloat = 0.01
    """Sampling distance in metres along the path between waypoints."""

    release_open: bool = True
    """Also check the gripper at the stage's release opening, at the release
    pose and along the motion after it.

    Applies to the group whose last waypoint is the last pose before the
    stage's opening EEF action; the opening is that action's command, or the
    chosen posture's ``release_joint_positions``.
    """

    @field_validator("bodies", mode="after")
    @classmethod
    def validate_bodies(cls, value: Tuple[str, ...]) -> Tuple[str, ...]:
        """Require at least one named obstacle body."""
        if not value or any(not str(name).strip() for name in value):
            raise ValueError("clearance.bodies must list at least one body name")
        return tuple(str(name) for name in value)


class OrientationPostureConfig(BaseModel, frozen=True):
    """One alternative posture of a ``nearest_feasible`` goal."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    name: str
    """Label reported in diagnostics."""

    constraints: Tuple[AxisRangeConstraintConfig, ...] = ()
    """Axis constraints of this posture, on top of the goal's own."""

    offset: Position = (0.0, 0.0, 0.0)
    """Translation in metres added to every waypoint of the group when this
    posture is chosen, expressed along the axes of the goal's ``reference``."""

    release_joint_positions: Optional[Tuple[float, ...]] = None
    """Gripper command of the stage's opening EEF action when this posture is
    chosen, as for ``eef.joint_positions``; ``None`` keeps the stage's own.
    The clearance check measures the gripper at this opening."""

    @field_validator("name", mode="after")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Require a non-empty label."""
        if not value.strip():
            raise ValueError("posture name must not be empty")
        return value


class NearestFeasibleOrientationGoalConfig(BaseModel, frozen=True):
    """The feasible orientation nearest to the current one.

    The controlled frame's ``primary_axis`` is parameterized by its azimuth
    and elevation in ``reference`` together with the spin about it. Candidates
    on a grid of those three angles are filtered by the primary-axis
    intervals, every entry of ``constraints`` and, when ``postures`` are
    given, the constraints of one posture; they are then taken in order of
    rotation from the current orientation, and with ``clearance`` the first
    candidate whose path stays clear wins. The chosen posture's ``offset``
    moves every waypoint of the group. All waypoints of one stage phase that
    share an identical goal reuse the orientation and posture solved at the
    first.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal[OrientationGoalKind.NEAREST_FEASIBLE] = (
        OrientationGoalKind.NEAREST_FEASIBLE
    )
    """Discriminator for a nearest-feasible goal."""

    reference: AxisReference = AxisReference.OBJECT
    """Frame in which elevation and azimuth are measured (``world`` or ``object``)."""

    primary_axis: AxisRangeConfig
    """Controlled-frame axis whose direction parameterizes the candidates."""

    constraints: Tuple[AxisRangeConstraintConfig, ...] = ()
    """Additional axis constraints; a candidate must satisfy all of them."""

    postures: Tuple[OrientationPostureConfig, ...] = ()
    """Alternative postures; a candidate must satisfy one of them. Empty means
    a single posture without extra constraints or offset."""

    resolution: float = Field(default=math.radians(5.0), gt=0.0, le=math.pi / 4.0)
    """Grid step in radians for the azimuth, elevation and spin samples."""

    clearance: Optional[OrientationClearanceConfig] = None
    """Optional collision clearance along the group's waypoint path."""

    @field_validator("reference", mode="after")
    @classmethod
    def validate_reference(cls, value: AxisReference) -> AxisReference:
        """Angles are measured in the world or the stage object frame."""
        if value == AxisReference.BASE:
            raise ValueError("nearest_feasible reference must be world or object")
        return value

    @field_validator("postures", mode="after")
    @classmethod
    def validate_postures(
        cls, value: Tuple[OrientationPostureConfig, ...]
    ) -> Tuple[OrientationPostureConfig, ...]:
        """Posture names identify the choice in diagnostics, so keep them unique."""
        names = [posture.name for posture in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"posture names must be unique; repeated: {duplicates}")
        return value


OrientationGoalConfig = Annotated[
    Union[
        FixedOrientationGoalConfig,
        AxisAlignmentOrientationGoalConfig,
        NearestFeasibleOrientationGoalConfig,
    ],
    Field(discriminator="kind"),
]
