"""Deterministic, non-destructive migration of legacy component catalogs.

The migrator writes only small JSON sidecars.  It never copies OBJ/collision
bytes and never edits the supplied catalog, which makes it suitable for the
ignored ``third_party`` payload as well as external deployments.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict


class MigrationReport(BaseModel, frozen=True):
    """Summary emitted by a catalog migration."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    source_root: Path
    """Legacy catalog root."""
    output_root: Path
    """Canonical sidecar output root."""
    component_count: int
    """Number of component manifests emitted."""
    artifact_count: int
    """Total number of visual and collision artifact references emitted."""
    visual_artifact_count: int
    """Number of visual artifact references emitted."""
    collision_artifact_count: int
    """Number of collision artifact references emitted."""
    warnings: tuple[str, ...] = ()
    """Non-fatal provenance warnings."""


def migrate_legacy_catalog(
    source_root: str | Path,
    output_root: str | Path,
    *,
    overwrite: bool = False,
) -> MigrationReport:
    """Write a canonical package view for a legacy ``product_space.json``.

    Artifact URIs are relative to the *payload root* (the source catalog), not
    copied into the sidecar directory.  The generated descriptor records that
    split explicitly so a deployment can mount the payload independently.
    """

    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    product_path = source / "product_space.json"
    if not product_path.is_file():
        raise FileNotFoundError(f"legacy product_space.json not found: {product_path}")
    product = _read_json(product_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest_dir = output / "components"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    count = 0
    visual_artifacts = 0
    collision_artifacts = 0
    index: dict[str, dict[str, Any]] = {}

    for dimension, kind in (("doors", "door"), ("handles", "handle")):
        refs = product.get("components", {}).get(dimension, [])
        if not isinstance(refs, list):
            raise ValueError(f"product_space.json is missing components.{dimension}")
        for reference in refs:
            if not isinstance(reference, Mapping):
                continue
            asset_id = str(reference.get("asset_id", ""))
            relative_manifest = str(reference.get("manifest", ""))
            manifest_path = _safe_path(
                source, relative_manifest, f"{asset_id} manifest"
            )
            expected = reference.get("manifest_sha256")
            if isinstance(expected, str) and _sha256(manifest_path) != expected:
                raise ValueError(f"{asset_id} manifest sha256 mismatch")
            legacy = _read_json(manifest_path)
            canonical, visual_count, collision_count = _canonical_component(
                source, legacy, kind, manifest_path
            )
            visual_artifacts += visual_count
            collision_artifacts += collision_count
            destination = manifest_dir / f"{asset_id}.json"
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"canonical manifest already exists: {destination}"
                )
            destination.write_text(
                json.dumps(canonical, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            index[asset_id] = {
                "id": asset_id,
                "kind": kind,
                "manifest": f"components/{asset_id}.json",
                "legacy_manifest": relative_manifest,
                "legacy_manifest_sha256": _sha256(manifest_path),
                "status": legacy.get("status", reference.get("status", "unknown")),
            }
            count += 1

    descriptor = {
        "schema": "aao.scene-asset-package/v1",
        "package_id": "migrated_" + source.name,
        "revision": str(product.get("compiler_version", "legacy")),
        "units": {"length": "m", "angle": "rad", "mass": "kg"},
        "canonical_frame": {
            "name": "package_local",
            "up": "+z",
            "handedness": str(
                product.get("combination_space", {}).get("handedness", "unknown")
            ),
            "quaternion": "xyzw",
            "transform_baked": True,
        },
        "payload_root": {"uri": os.path.relpath(source, output).replace("\\", "/")},
        "components": index,
        "assembly_templates": [
            {
                "id": "migrated_lever_door_v1",
                "adapter": "unidoor.lever_door@1",
                "joint_axes": {
                    "door": [
                        0.0,
                        0.0,
                        -1.0
                        if product.get("combination_space", {}).get("handedness")
                        == "right"
                        else 1.0,
                    ],
                    "handle": [
                        0.0,
                        1.0
                        if product.get("combination_space", {}).get("handedness")
                        == "right"
                        else -1.0,
                        0.0,
                    ],
                },
                "joint_specs": {
                    "door": {
                        "range": [0.0, 1.5079644737231006],
                        "springref": 0.0,
                        "stiffness": 1.0,
                        "damping": 1.2,
                        "frictionloss": 0.2,
                        "armature": 0.02,
                        "limited": True,
                    },
                    "handle": {
                        "range": [0.0, 0.65],
                        "springref": 0.0,
                        "stiffness": 3.0,
                        "damping": 0.08,
                        "armature": 0.002,
                        "limited": True,
                    },
                },
                "axis_frame": "package_local",
            }
        ],
        "integrity": {"algorithm": "sha256", "policy": "selected-artifacts"},
        "provenance": {"legacy_product_space": str(product_path)},
    }
    descriptor_path = output / "scene_asset_package.json"
    if descriptor_path.exists() and not overwrite:
        raise FileExistsError(f"canonical package already exists: {descriptor_path}")
    descriptor_path.write_text(
        json.dumps(descriptor, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return MigrationReport(
        source_root=source,
        output_root=output,
        component_count=count,
        artifact_count=visual_artifacts + collision_artifacts,
        visual_artifact_count=visual_artifacts,
        collision_artifact_count=collision_artifacts,
        warnings=tuple(warnings),
    )


def _canonical_component(
    source_root: Path,
    legacy: Mapping[str, Any],
    kind: str,
    manifest_path: Path,
) -> tuple[dict[str, Any], int, int]:
    asset_id = str(legacy.get("asset_id", ""))
    outputs = legacy.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{asset_id} manifest has no outputs")
    visual: list[dict[str, Any]] = []
    for role, value in outputs.items():
        if not isinstance(value, Mapping):
            continue
        path = _safe_path(source_root, str(value.get("path", "")), f"{asset_id}.{role}")
        visual.append(
            {
                "role": str(role),
                "uri": path.relative_to(source_root).as_posix(),
                "format": path.suffix.removeprefix(".") or "binary",
                "frame": "package_local",
                "sha256": _sha256(path),
                "transform_baked": True,
            }
        )
    geometry = (
        legacy.get("geometry") if isinstance(legacy.get("geometry"), Mapping) else {}
    )
    handedness = (
        legacy.get("handedness")
        if isinstance(legacy.get("handedness"), Mapping)
        else {}
    )
    anchors: dict[str, Any] = {}
    if kind == "door" and geometry.get("handle_position_m") is not None:
        anchors["handle_mount"] = {
            "kind": "point",
            "position": geometry["handle_position_m"],
            "frame": "package_local",
        }
    if kind == "handle" and geometry.get("grasp_offset_m") is not None:
        anchors["grasp"] = {
            "kind": "point",
            "position": geometry["grasp_offset_m"],
            "frame": "package_local",
        }
    collision: list[dict[str, Any]] = []
    collision_ref = legacy.get("collision_supplement")
    if isinstance(collision_ref, Mapping):
        collision_path = _safe_path(
            source_root,
            str(collision_ref.get("path", "")),
            f"{asset_id}.collision",
        )
        collision.append(
            {
                "role": "collision",
                "uri": collision_path.relative_to(source_root).as_posix(),
                "format": collision_path.suffix.removeprefix(".") or "json",
                "representation": collision_ref.get("representation", "unknown"),
                "frame": "package_local",
                "static": False,
                "dynamic": True,
                "sha256": _sha256(collision_path),
            }
        )
    result = {
        "schema": "aao.scene-asset-component/v1",
        "id": asset_id,
        "kind": kind,
        "revision": str(legacy.get("compiler_version", "legacy")),
        "status": legacy.get("status", "unknown"),
        "frame": {
            "name": "package_local",
            "units": "m",
            "transform_baked": True,
            "handedness_transform_id": handedness.get("transform_id"),
        },
        "artifacts": {"visual": visual, "collision": collision},
        "anchors": anchors,
        "geometry": geometry,
        "mechanism": _mechanism(kind, handedness),
        "provenance": {
            "legacy_manifest": manifest_path.relative_to(source_root).as_posix(),
            "legacy_manifest_sha256": _sha256(manifest_path),
        },
        "extensions": {"legacy": dict(legacy)},
    }
    return result, len(visual), len(collision)


def _mechanism(kind: str, handedness: Mapping[str, Any]) -> dict[str, Any]:
    side = str(handedness.get("hinge_side", "unknown"))
    if kind != "door":
        return {}
    axis = [0.0, 0.0, -1.0 if side == "right" else 1.0]
    return {
        "joints": [
            {
                "id": "hinge",
                "axis": axis,
                "axis_frame": "package_local",
                "inferred_from": "legacy.handedness",
                "inferred": True,
            }
        ]
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _safe_path(root: Path, relative: str, label: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"{label} must be a relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} escapes or is missing: {relative}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["MigrationReport", "migrate_legacy_catalog"]


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(
        migrate_legacy_catalog(
            args.source_root, args.output_root, overwrite=args.overwrite
        ).model_dump_json(indent=2)
    )
