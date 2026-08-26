"""Validation models for relocatable scene asset package descriptors.

The descriptor follows the vendor-neutral ``aao.scene-asset-package/v1``
shape. This module performs read-only checks; it never copies or rewrites
payload bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PackageFrameConfig(BaseModel, frozen=True):
    """Canonical coordinate-frame metadata."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="allow")

    name: str
    """Frame name."""
    up: str = "+z"
    """Up-axis declaration."""
    handedness: str = "right"
    """Coordinate handedness."""
    quaternion: Literal["xyzw", "wxyz"] = "xyzw"
    """Quaternion component order."""
    transform_baked: bool = False
    """Whether normalization/reflection is already baked into artifacts."""


class PackagePayloadConfig(BaseModel, frozen=True):
    """Location of an independently deployable package payload."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    uri: str
    """Descriptor-relative POSIX path to the payload root."""

    @field_validator("uri")
    @classmethod
    def _relative_uri(cls, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or "\\" in value:
            raise ValueError("payload_root.uri must be a relative POSIX path")
        return value


class SceneAssetPackageDescriptor(BaseModel, frozen=True):
    """Validated root descriptor for a scene asset package."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="forbid",
        populate_by_name=True,
    )

    schema_id: Literal["aao.scene-asset-package/v1"] = Field(
        default="aao.scene-asset-package/v1", alias="schema"
    )
    """Package schema discriminator."""
    package_id: str
    """Stable package identity."""
    revision: str
    """Immutable package revision."""
    units: Mapping[str, str]
    """Unit declarations for length, angle and mass."""
    canonical_frame: PackageFrameConfig
    """Canonical frame and transform convention."""
    payload_root: PackagePayloadConfig
    """Location of the immutable artifact payload."""
    components: Mapping[str, Any]
    """Component index interpreted by the selected assembly adapter."""
    assembly_templates: tuple[Mapping[str, Any], ...]
    """Versioned assembly behavior, mechanism data and semantic exports."""
    integrity: Mapping[str, Any] = Field(default_factory=dict)
    """Integrity algorithm, policy and optional lock path."""
    provenance: Mapping[str, Any] = Field(default_factory=dict)
    """Opaque source and transformation provenance."""
    extensions: Mapping[str, Any] = Field(default_factory=dict)
    """Adapter-specific metadata outside the generic contract."""


class SceneAssetArtifactDescriptor(BaseModel, frozen=True):
    """One visual/collision/material artifact in a component manifest."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="allow")

    role: str
    """Semantic artifact role."""
    uri: str
    """Package/payload-relative POSIX URI."""
    format: str
    """On-disk representation format."""
    sha256: str
    """Content digest."""
    frame: str = "package_local"
    """Coordinate frame for the artifact."""

    @field_validator("uri")
    @classmethod
    def _relative_uri(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or "\\" in value:
            raise ValueError("artifact uri must be a relative POSIX path")
        return value

    @field_validator("sha256")
    @classmethod
    def _digest(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
            raise ValueError(
                "artifact sha256 must be a 64-character hexadecimal digest"
            )
        return value.lower()


class SceneAssetComponentManifest(BaseModel, frozen=True):
    """Common component-manifest shape shared by all asset families."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="allow",
        populate_by_name=True,
    )

    schema_id: Literal["aao.scene-asset-component/v1"] = Field(
        default="aao.scene-asset-component/v1", alias="schema"
    )
    """Component schema discriminator."""
    id: str
    """Opaque component identity."""
    kind: str
    """Descriptive component kind; adapters define role requirements."""
    revision: str
    """Component revision."""
    status: str
    """Validation status."""
    frame: Mapping[str, Any]
    """Frame and transform metadata."""
    artifacts: Mapping[str, tuple[SceneAssetArtifactDescriptor, ...]]
    """Artifacts grouped by visual/collision/material role."""
    anchors: Mapping[str, Any] = Field(default_factory=dict)
    """Named ports/anchors exposed to assembly templates."""
    geometry: Mapping[str, Any] = Field(default_factory=dict)
    """Optional normalized bounds and measurements."""
    mechanism: Mapping[str, Any] = Field(default_factory=dict)
    """Optional explicit joints and dynamics."""


class PackageValidationReport(BaseModel, frozen=True):
    """Read-only package validation result suitable for diagnostics."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    package_path: Path
    """Descriptor path."""
    payload_root: Path | None = None
    """Resolved legacy payload root, if present."""
    selected_artifacts: int = 0
    """Number of artifacts checked by a caller."""
    warnings: tuple[str, ...] = ()
    """Non-fatal provenance warnings."""


def load_package_descriptor(path: str | Path) -> SceneAssetPackageDescriptor:
    """Load and validate a package descriptor without touching artifacts."""

    descriptor_path = Path(path).expanduser().resolve()
    if not descriptor_path.is_file():
        raise FileNotFoundError(
            f"scene asset package descriptor not found: {descriptor_path}"
        )
    try:
        value = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"invalid scene asset package descriptor: {descriptor_path}"
        ) from exc
    return SceneAssetPackageDescriptor.model_validate(value)


def load_component_manifest(path: str | Path) -> SceneAssetComponentManifest:
    """Load and validate one canonical component sidecar."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"scene asset component manifest not found: {manifest_path}"
        )
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid component manifest: {manifest_path}") from exc
    return SceneAssetComponentManifest.model_validate(value)


def validate_package_payload(path: str | Path) -> PackageValidationReport:
    """Validate the descriptor's payload root and optional integrity lock.

    Relative ``..`` segments are allowed for a separately deployed payload.
    Adapter-specific structure and selected component hashes are checked by
    the adapter; this generic validator owns only package and lock containment.
    """

    descriptor_path = Path(path).expanduser().resolve()
    descriptor = load_package_descriptor(descriptor_path)
    warnings: list[str] = []
    selected_artifacts = 0
    payload_root = (descriptor_path.parent / descriptor.payload_root.uri).resolve()
    if not payload_root.is_dir():
        raise ValueError(f"payload_root is not a directory: {payload_root}")
    integrity = descriptor.integrity
    lock_uri = integrity.get("lock") if isinstance(integrity, Mapping) else None
    if lock_uri is not None:
        if not isinstance(lock_uri, str) or Path(lock_uri).is_absolute():
            raise ValueError("integrity.lock must be a relative path")
        lock_path = (descriptor_path.parent / lock_uri).resolve()
        if (
            not lock_path.is_relative_to(descriptor_path.parent)
            or not lock_path.is_file()
        ):
            raise ValueError("integrity.lock is missing or escapes package root")
        lock = _read_json(lock_path)
        entries = lock.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("integrity lock entries must be a list")
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            uri = entry.get("uri")
            expected = entry.get("sha256")
            if not isinstance(uri, str) or Path(uri).is_absolute():
                raise ValueError("integrity lock entry URI must be relative")
            artifact = (payload_root / uri).resolve()
            if not artifact.is_relative_to(payload_root) or not artifact.is_file():
                raise ValueError(f"integrity lock artifact is missing: {uri}")
            actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(
                    f"integrity lock hash mismatch for {uri}: expected {expected}, got {actual}"
                )
            selected_artifacts += 1
    return PackageValidationReport(
        package_path=descriptor_path,
        payload_root=payload_root,
        selected_artifacts=selected_artifacts,
        warnings=tuple(warnings),
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


__all__ = [
    "PackageFrameConfig",
    "PackagePayloadConfig",
    "PackageValidationReport",
    "SceneAssetArtifactDescriptor",
    "SceneAssetComponentManifest",
    "SceneAssetPackageDescriptor",
    "load_component_manifest",
    "load_package_descriptor",
    "validate_package_payload",
]
