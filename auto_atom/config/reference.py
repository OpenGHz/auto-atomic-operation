"""reference configuration models (split from the former auto_atom.framework monolith)."""

from enum import Enum
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self


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
