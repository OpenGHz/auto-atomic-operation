"""orientation configuration models (split from the former auto_atom.framework monolith)."""

import math
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from auto_atom.config.primitives import Orientation, Position


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
