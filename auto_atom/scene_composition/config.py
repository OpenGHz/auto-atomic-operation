"""Pure-data scene composition recipes.

These models are the public configuration contract.  They intentionally do
not contain adapter instances or callables: Hydra can instantiate the outer
application, while this nested data remains deterministic, serialisable and
safe to validate before any asset code executes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    field_validator,
    model_validator,
)
from typing_extensions import Self


def _finite_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != size
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{label} must contain exactly {size} numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


class TransformConfig(BaseModel, frozen=True):
    """Placement of a scene layer in the host scene frame."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation in metres in the host scene frame."""

    orientation_xyzw: tuple[float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )
    """Quaternion in canonical ``xyzw`` order."""

    @field_validator("position", mode="before")
    @classmethod
    def _position(cls, value: Any) -> tuple[float, float, float]:
        result = _finite_vector(value, 3, "position")
        return result  # type: ignore[return-value]

    @field_validator("orientation_xyzw", mode="before")
    @classmethod
    def _orientation(cls, value: Any) -> tuple[float, float, float, float]:
        result = _finite_vector(value, 4, "orientation_xyzw")
        if math.isclose(sum(item * item for item in result), 0.0):
            raise ValueError("orientation_xyzw cannot be the zero quaternion")
        return result  # type: ignore[return-value]


class AssetScaleRuleConfig(BaseModel, frozen=True):
    """One declarative, uniform scale applied during asset compilation."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    bodies: tuple[str, ...] = ()
    """MJCF body names whose owned geometry receives this scale."""

    preserve_bodies: tuple[str, ...] = ()
    """Child body names whose subtrees stay unchanged during parent scaling."""

    meshes: tuple[str, ...] = ()
    """MJCF mesh names that receive this scale."""

    mesh_prefixes: tuple[str, ...] = ()
    """Prefixes selecting generated mesh names, such as collision-part meshes."""

    source_bounds: str
    """Dotted path to the source bounds in the selected component metadata."""

    axis: Literal["x", "y", "z"]
    """Extent axis used to derive the uniform scale factor."""

    target_extent_m: PositiveFloat
    """Desired extent in metres for the selected source bounds."""

    required: bool = True
    """Whether every named target must be present in the compiled contribution."""

    @field_validator(
        "bodies", "preserve_bodies", "meshes", "mesh_prefixes", mode="before"
    )
    @classmethod
    def _names(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("asset scale targets must be a sequence of names")
        result = tuple(str(item).strip() for item in value)
        if any(not item for item in result):
            raise ValueError("asset scale target names cannot be empty")
        return result

    @field_validator("source_bounds")
    @classmethod
    def _source_bounds(cls, value: str) -> str:
        value = str(value).strip()
        if not value or any(not part for part in value.split(".")):
            raise ValueError("source_bounds must be a non-empty dotted path")
        return value

    @model_validator(mode="after")
    def _has_target(self) -> Self:
        if not self.bodies and not self.meshes and not self.mesh_prefixes:
            raise ValueError("asset scale rule must select a body or mesh")
        if set(self.bodies) & set(self.preserve_bodies):
            raise ValueError("asset scale body cannot also be preserved")
        return self


class AssetAnchorCoordinateConfig(BaseModel, frozen=True):
    """One coordinate projection used to reposition an asset anchor."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    value_m: float | None = None
    """Fixed coordinate value in metres, when no bounds projection is used."""

    edge: Literal["min", "max"] | None = None
    """Bounds edge to project when deriving a coordinate from geometry."""

    multiplier: float = 1.0
    """Multiplier applied to the selected bounds edge."""

    offset_m: float = 0.0
    """Additive offset applied after the bounds projection."""

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.value_m is None) == (self.edge is None):
            raise ValueError(
                "anchor coordinate requires exactly one of value_m or edge"
            )
        for value, label in (
            (self.value_m, "value_m"),
            (self.multiplier, "multiplier"),
            (self.offset_m, "offset_m"),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"anchor coordinate {label} must be finite")
        return self


class AssetAnchorConfig(BaseModel, frozen=True):
    """Declarative placement of one or more compiled asset bodies."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    bodies: tuple[str, ...]
    """MJCF body names sharing the resolved anchor position."""

    source_bounds: str | None = None
    """Dotted path to bounds used by projected coordinates, when needed."""

    coordinates: Mapping[str, AssetAnchorCoordinateConfig]
    """Coordinates to set; omitted axes retain the compiled position."""

    required: bool = True
    """Whether every named body must be present in the compiled contribution."""

    @field_validator("bodies", mode="before")
    @classmethod
    def _bodies(cls, value: Any) -> tuple[str, ...]:
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("asset anchor bodies must be a sequence")
        result = tuple(str(item).strip() for item in value)
        if not result or any(not item for item in result):
            raise ValueError("asset anchor bodies cannot be empty")
        return result

    @field_validator("source_bounds")
    @classmethod
    def _source_bounds(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        if not value or any(not part for part in value.split(".")):
            raise ValueError("anchor source_bounds must be a dotted path")
        return value

    @field_validator("coordinates", mode="before")
    @classmethod
    def _coordinates(cls, value: Any) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("asset anchor coordinates must be a non-empty mapping")
        if any(str(key) not in {"x", "y", "z"} for key in value):
            raise ValueError("asset anchor coordinates may only contain x, y and z")
        return value

    @model_validator(mode="after")
    def _projected_coordinates_need_bounds(self) -> Self:
        if self.source_bounds is None and any(
            coordinate.edge is not None for coordinate in self.coordinates.values()
        ):
            raise ValueError(
                "asset anchor source_bounds is required for projected coordinates"
            )
        return self


class MjcfLayerConfig(BaseModel, frozen=True):
    """A pre-authored MJCF document included as an ordered scene layer."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal["mjcf"] = "mjcf"
    """Layer discriminator."""

    path: Path
    """Path to the MJCF document; includes are resolved relative to it."""

    namespace: str = ""
    """Optional namespace for metadata; existing XML names are not rewritten."""

    role: Literal["scene", "operator"] = "scene"
    """Semantic ownership used by execution policies to retain scene layers
    while removing physical operator layers."""

    @field_validator("namespace")
    @classmethod
    def _namespace(cls, value: str) -> str:
        value = str(value).strip()
        if value and (not value.replace("_", "").replace("-", "").isalnum()):
            raise ValueError("namespace must contain only letters, digits, '_' or '-'")
        return value


class AssetAssemblyLayerConfig(BaseModel, frozen=True):
    """A declarative asset-package assembly recipe.

    ``adapter`` identifies implementation behavior (for example
    ``unidoor.lever_door@1``), while the shape of this object remains generic:
    package, role selections, placement, namespace and integrity policy.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal["asset_assembly"] = "asset_assembly"
    """Layer discriminator."""

    role: Literal["scene", "operator"] = "scene"
    """Semantic ownership used by execution policies to remove operator
    assembly layers together with operator MJCF layers."""

    package: Path
    """Package root or package descriptor path."""

    adapter: str
    """Registered assembly adapter id and version."""

    selection: Mapping[str, str]
    """Role-to-component identifiers selected from the package."""

    placement: TransformConfig = Field(default_factory=TransformConfig)
    """Placement of the assembled contribution in the host frame."""

    namespace: str
    """Required stable namespace for generated names and semantic exports."""

    verify_hashes: bool = True
    """Whether selected manifests and artifacts must pass declared SHA-256 checks."""

    options: Mapping[str, Any] = Field(default_factory=dict)
    """Adapter-specific options kept as data and validated by the adapter."""

    scaling: tuple[AssetScaleRuleConfig, ...] = ()
    """Generic cold-path geometry scaling rules for the assembled contribution."""

    anchors: tuple[AssetAnchorConfig, ...] = ()
    """Generic cold-path anchor projections applied after geometry scaling."""

    @field_validator("adapter")
    @classmethod
    def _adapter(cls, value: str) -> str:
        value = str(value).strip()
        if not value or "@" not in value:
            raise ValueError("adapter must be a non-empty id with an explicit @version")
        return value

    @field_validator("namespace")
    @classmethod
    def _namespace(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("asset assembly namespace is required")
        if not value.replace("_", "").replace("-", "").isalnum():
            raise ValueError("namespace must contain only letters, digits, '_' or '-'")
        return value

    @field_validator("selection", mode="before")
    @classmethod
    def _selection(cls, value: Any) -> dict[str, str]:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("selection must be a non-empty role-to-id mapping")
        result = {str(key): str(item) for key, item in value.items()}
        if any(not key or not item for key, item in result.items()):
            raise ValueError("selection role and component ids cannot be empty")
        return result


SceneLayerConfig = Annotated[
    MjcfLayerConfig | AssetAssemblyLayerConfig,
    Field(discriminator="kind"),
]


class SceneConfig(BaseModel, frozen=True):
    """Host scene plus an ordered list of composable scene layers."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    base: Path
    """Host MJCF scene document."""

    layers: tuple[SceneLayerConfig, ...] = ()
    """Layers compiled and merged in declaration order."""

    @model_validator(mode="after")
    def _validate_namespaces(self) -> SceneConfig:
        namespaces = [
            layer.namespace
            for layer in self.layers
            if isinstance(layer, AssetAssemblyLayerConfig)
        ]
        if len(namespaces) != len(set(namespaces)):
            raise ValueError("asset assembly namespaces must be unique within a scene")
        return self


__all__ = [
    "AssetAnchorConfig",
    "AssetAnchorCoordinateConfig",
    "AssetAssemblyLayerConfig",
    "AssetScaleRuleConfig",
    "MjcfLayerConfig",
    "SceneConfig",
    "SceneLayerConfig",
    "TransformConfig",
]
