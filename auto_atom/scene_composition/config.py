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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class MjcfLayerConfig(BaseModel, frozen=True):
    """A pre-authored MJCF document included as an ordered scene layer."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    kind: Literal["mjcf"] = "mjcf"
    """Layer discriminator."""

    path: Path
    """Path to the MJCF document; includes are resolved relative to it."""

    namespace: str = ""
    """Optional namespace for metadata; existing XML names are not rewritten."""

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
    "AssetAssemblyLayerConfig",
    "MjcfLayerConfig",
    "SceneConfig",
    "SceneLayerConfig",
    "TransformConfig",
]
