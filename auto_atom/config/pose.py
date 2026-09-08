"""pose configuration models (split from the former auto_atom.framework monolith)."""

import math
from collections.abc import Mapping
from typing import Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, field_validator

from auto_atom.config.primitives import Orientation, Position, Rotation
from auto_atom.config.reference import (
    PoseReference,
    _coerce_pose_reference,
    _validate_pose_reference,
)


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
    named scene frame exposed by the selected backend. Which references are
    legal is checked by the owner-specific backend seam (objects/cameras accept
    scene frames; operator EEF poses may additionally use ``base`` and operator
    frame aliases).
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
    """Built-in pose reference or a named scene frame."""

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
