"""Strict loader for UniDoor handle convex-collision supplements.

The normalized UniDoor component manifest is the authority that selects a
collision sidecar.  This module resolves that reference inside the catalog,
checks its identity and integrity, and exposes only collision-enabled convex
parts as immutable Python data.  It deliberately has no MuJoCo dependency;
the catalog compiler owns the eventual MJCF representation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "1.0"
_PACKAGE_KIND = "unidoor_component_collision_supplement"
_REPRESENTATION = "motrixsim_acd_convex_parts_v1"
_MIRROR_TRANSFORM_ID = "door-hinge-x-reflection-v1"
_IDENTITY_TRANSFORM_ID = "door-hinge-identity-v1"
_MIRROR_MATRIX = (
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

Vector3 = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class HandleCollisionPart:
    """One verified convex mesh in component-local metres."""

    name: str
    vertices_m: tuple[Vector3, ...]
    faces: tuple[Triangle, ...]
    geometry_sha256: str


@dataclass(frozen=True)
class HandleCollisionSupplement:
    """Verified collision data selected by one handle component manifest."""

    manifest_path: Path
    manifest_sha256: str
    asset_id: str
    source_id: str
    source_mesh_sha256: str
    model_version: str
    hinge_side: str
    actual_part_count: int
    topology_slot_count: int
    artifact_manifest_sha256: str
    artifact_collection_sha256: str
    artifact_asset_set_sha256: str
    enabled_parts: tuple[HandleCollisionPart, ...]


def load_handle_collision_supplement(
    catalog_root: Path,
    component_manifest: Mapping[str, Any],
    *,
    verify_hashes: bool,
) -> HandleCollisionSupplement | None:
    """Load the handle's ACD sidecar, returning ``None`` only when absent.

    A declared but malformed, missing, unsupported, or inconsistent sidecar
    always raises.  ``verify_hashes`` controls comparison of the sidecar file
    bytes with the component reference.  Internal geometry hashes and all
    manifest-to-sidecar identity links are always checked.
    """

    raw_reference = component_manifest.get("collision_supplement")
    if raw_reference is None:
        return None
    if not isinstance(raw_reference, Mapping):
        raise ValueError("handle collision_supplement must be an object")

    root = catalog_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UniDoor catalog root not found: {root}")
    asset_id = _required_text(component_manifest.get("asset_id"), "handle asset_id")
    if component_manifest.get("kind") != "handle":
        raise ValueError(f"Collision supplement component must be a handle: {asset_id}")
    source_id = _required_text(
        component_manifest.get("source_id"), f"{asset_id} source_id"
    )
    component_hinge_side = _component_hinge_side(component_manifest, asset_id)
    source_mesh_sha256 = _component_source_mesh_sha256(component_manifest, asset_id)

    reference = raw_reference
    _expect_equal(reference, "schema_version", _SCHEMA_VERSION, asset_id)
    _expect_equal(reference, "representation", _REPRESENTATION, asset_id)
    _expect_equal(reference, "asset_id", asset_id, asset_id)
    _expect_equal(reference, "asset_kind", "handle", asset_id)
    _expect_equal(reference, "source_mesh_sha256", source_mesh_sha256, asset_id)
    _expect_equal(reference, "hinge_side", component_hinge_side, asset_id)
    expected_manifest_sha256 = _digest(
        reference.get("sha256"), f"{asset_id} collision_supplement.sha256"
    )
    manifest_path = _sidecar_path(root, reference.get("path"), asset_id)
    manifest_sha256 = _file_sha256(manifest_path)
    if verify_hashes and manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"{asset_id} collision supplement sha256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha256}"
        )

    payload = _read_json(manifest_path)
    _expect_equal(payload, "schema_version", _SCHEMA_VERSION, asset_id)
    _expect_equal(payload, "package_kind", _PACKAGE_KIND, asset_id)
    _expect_equal(payload, "representation", _REPRESENTATION, asset_id)
    model_version = _required_text(
        payload.get("model_version"), f"{asset_id} collision model_version"
    )
    _expect_equal(payload, "asset_id", asset_id, asset_id)
    _expect_equal(payload, "asset_kind", "handle", asset_id)
    _expect_equal(payload, "source_id", source_id, asset_id)
    _expect_equal(payload, "source_mesh_sha256", source_mesh_sha256, asset_id)
    hinge_side = _validate_payload_handedness(
        payload, asset_id, expected_side=component_hinge_side
    )
    if hinge_side != component_hinge_side:
        raise ValueError(
            f"{asset_id} collision supplement hinge_side mismatch: "
            f"expected {component_hinge_side}, got {hinge_side}"
        )

    topology_slot_count = _strict_positive_int(
        payload.get("topology_slot_count"),
        f"{asset_id} collision supplement topology_slot_count",
    )
    actual_part_count = _strict_positive_int(
        payload.get("actual_part_count"),
        f"{asset_id} collision supplement actual_part_count",
    )
    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, list) or len(raw_parts) != topology_slot_count:
        raise ValueError(
            f"{asset_id} collision supplement parts do not match topology_slot_count"
        )
    if actual_part_count > topology_slot_count:
        raise ValueError(f"{asset_id} collision supplement part count is invalid")

    enabled_parts: list[HandleCollisionPart] = []
    enabled_geometry_hashes: list[str] = []
    disabled_seen = False
    for index, raw_part in enumerate(raw_parts):
        if not isinstance(raw_part, Mapping):
            raise ValueError(f"{asset_id} collision part {index} must be an object")
        expected_name = f"part_{index:03d}"
        _expect_equal(raw_part, "name", expected_name, asset_id)
        enabled = _strict_bool(
            raw_part.get("collision_enabled"),
            f"{asset_id}:{expected_name}.collision_enabled",
        )
        if disabled_seen and enabled:
            raise ValueError(
                f"{asset_id} collision-enabled topology slots must be contiguous"
            )
        disabled_seen = disabled_seen or not enabled
        part = _load_part(raw_part, asset_id, expected_name)
        if enabled:
            enabled_parts.append(part)
            enabled_geometry_hashes.append(part.geometry_sha256)

    if len(enabled_parts) != actual_part_count:
        raise ValueError(
            f"{asset_id} collision enabled-part count mismatch: "
            f"expected {actual_part_count}, got {len(enabled_parts)}"
        )
    if len(set(enabled_geometry_hashes)) != len(enabled_geometry_hashes):
        raise ValueError(f"{asset_id} collision supplement has duplicate enabled parts")

    artifact_manifest_sha256 = _digest(
        payload.get("artifact_manifest_sha256"),
        f"{asset_id} artifact_manifest_sha256",
    )
    artifact_collection_sha256 = _digest(
        payload.get("artifact_collection_sha256"),
        f"{asset_id} artifact_collection_sha256",
    )
    artifact_asset_set_sha256 = _digest(
        payload.get("artifact_asset_set_sha256"),
        f"{asset_id} artifact_asset_set_sha256",
    )
    expected_asset_set_sha256 = _json_sha256(
        {
            "asset_id": asset_id,
            "source_mesh_sha256": source_mesh_sha256,
            "part_geometry_sha256": enabled_geometry_hashes,
        }
    )
    if artifact_asset_set_sha256 != expected_asset_set_sha256:
        raise ValueError(f"{asset_id} collision supplement asset-set hash mismatch")

    reference_counts = {
        "actual_part_count": actual_part_count,
        "topology_slot_count": topology_slot_count,
    }
    for key, expected in reference_counts.items():
        actual = _strict_positive_int(
            reference.get(key), f"{asset_id} collision_supplement.{key}"
        )
        if actual != expected:
            raise ValueError(
                f"{asset_id} collision supplement {key} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    reference_hashes = {
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "artifact_collection_sha256": artifact_collection_sha256,
        "artifact_asset_set_sha256": artifact_asset_set_sha256,
    }
    for key, expected in reference_hashes.items():
        actual = _digest(reference.get(key), f"{asset_id} collision_supplement.{key}")
        if actual != expected:
            raise ValueError(
                f"{asset_id} collision supplement {key} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )

    return HandleCollisionSupplement(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        asset_id=asset_id,
        source_id=source_id,
        source_mesh_sha256=source_mesh_sha256,
        model_version=model_version,
        hinge_side=hinge_side,
        actual_part_count=actual_part_count,
        topology_slot_count=topology_slot_count,
        artifact_manifest_sha256=artifact_manifest_sha256,
        artifact_collection_sha256=artifact_collection_sha256,
        artifact_asset_set_sha256=artifact_asset_set_sha256,
        enabled_parts=tuple(enabled_parts),
    )


def _load_part(
    raw_part: Mapping[str, Any], asset_id: str, name: str
) -> HandleCollisionPart:
    vertices = _vertices(raw_part.get("vertices_m"), asset_id, name)
    faces = _faces(raw_part.get("faces"), len(vertices), asset_id, name)
    _validate_mesh_geometry(vertices, faces, asset_id, name)
    geometry_sha256 = _digest(
        raw_part.get("geometry_sha256"), f"{asset_id}:{name}.geometry_sha256"
    )
    expected_geometry_sha256 = _json_sha256(
        {
            "vertices_m": [list(vertex) for vertex in vertices],
            "faces": [list(face) for face in faces],
        }
    )
    if geometry_sha256 != expected_geometry_sha256:
        raise ValueError(f"{asset_id} collision part geometry hash mismatch: {name}")
    return HandleCollisionPart(
        name=name,
        vertices_m=vertices,
        faces=faces,
        geometry_sha256=geometry_sha256,
    )


def _vertices(value: Any, asset_id: str, name: str) -> tuple[Vector3, ...]:
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(f"{asset_id} collision part has invalid vertices: {name}")
    vertices = tuple(_vector3(row, f"{asset_id}:{name}.vertices_m") for row in value)
    if len(set(vertices)) != len(vertices):
        raise ValueError(f"{asset_id} collision part has duplicate vertices: {name}")
    return vertices


def _faces(
    value: Any, vertex_count: int, asset_id: str, name: str
) -> tuple[Triangle, ...]:
    if not isinstance(value, list) or len(value) < 4:
        raise ValueError(f"{asset_id} collision part has invalid faces: {name}")
    faces: list[Triangle] = []
    used_vertices: set[int] = set()
    undirected_edges: Counter[tuple[int, int]] = Counter()
    directed_edges: Counter[tuple[int, int]] = Counter()
    for raw_face in value:
        if not isinstance(raw_face, list) or len(raw_face) != 3:
            raise ValueError(f"{asset_id} collision part has invalid face: {name}")
        if any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_face
        ):
            raise ValueError(
                f"{asset_id} collision part face indices must be integers: {name}"
            )
        face = tuple(raw_face)
        if len(set(face)) != 3 or any(
            item < 0 or item >= vertex_count for item in face
        ):
            raise ValueError(
                f"{asset_id} collision part has invalid face index: {name}"
            )
        triangle: Triangle = face  # type: ignore[assignment]
        faces.append(triangle)
        used_vertices.update(triangle)
        for index in range(3):
            edge = (triangle[index], triangle[(index + 1) % 3])
            directed_edges[edge] += 1
            undirected_edges[tuple(sorted(edge))] += 1
    if len(set(faces)) != len(faces):
        raise ValueError(f"{asset_id} collision part has duplicate faces: {name}")
    if len({tuple(sorted(face)) for face in faces}) != len(faces):
        raise ValueError(
            f"{asset_id} collision part has duplicate unoriented faces: {name}"
        )
    if used_vertices != set(range(vertex_count)):
        raise ValueError(f"{asset_id} collision part has unused vertices: {name}")
    if any(count != 2 for count in undirected_edges.values()):
        raise ValueError(
            f"{asset_id} collision part is not a closed triangle mesh: {name}"
        )
    if any(
        directed_edges[(second, first)] != count
        for (first, second), count in directed_edges.items()
    ):
        raise ValueError(
            f"{asset_id} collision part has inconsistent face winding: {name}"
        )
    return tuple(faces)


def _validate_mesh_geometry(
    vertices: tuple[Vector3, ...],
    faces: tuple[Triangle, ...],
    asset_id: str,
    name: str,
) -> None:
    origin = vertices[0]
    signed_volume_times_six = 0.0
    face_normals: list[Vector3] = []
    edge_faces: dict[tuple[int, int], list[tuple[int, int]]] = {}
    face_neighbors: list[set[int]] = [set() for _ in faces]
    for face_index, (first, second, third) in enumerate(faces):
        a = _subtract(vertices[first], origin)
        b = _subtract(vertices[second], origin)
        c = _subtract(vertices[third], origin)
        normal = _cross(_subtract(b, a), _subtract(c, a))
        if _dot(normal, normal) == 0.0:
            raise ValueError(f"{asset_id} collision part has a degenerate face: {name}")
        face_normals.append(normal)
        signed_volume_times_six += _dot(a, _cross(b, c))
        for edge, opposite in (
            ((first, second), third),
            ((second, third), first),
            ((third, first), second),
        ):
            edge_faces.setdefault(tuple(sorted(edge)), []).append(
                (face_index, opposite)
            )
    if signed_volume_times_six <= 0.0:
        raise ValueError(
            f"{asset_id} collision part must have positive outward volume: {name}"
        )

    extent = max(
        max(vertex[axis] for vertex in vertices)
        - min(vertex[axis] for vertex in vertices)
        for axis in range(3)
    )
    face_tolerances = tuple(
        max(
            math.ulp(extent) * math.sqrt(_dot(normal, normal)) * 64.0,
            extent * math.sqrt(_dot(normal, normal)) * 1e-12,
        )
        for normal in face_normals
    )
    for adjacent in edge_faces.values():
        (first_face, first_opposite), (second_face, second_opposite) = adjacent
        face_neighbors[first_face].add(second_face)
        face_neighbors[second_face].add(first_face)
        _validate_convex_edge(
            vertices,
            faces[first_face],
            face_normals[first_face],
            face_tolerances[first_face],
            vertices[second_opposite],
            asset_id,
            name,
        )
        _validate_convex_edge(
            vertices,
            faces[second_face],
            face_normals[second_face],
            face_tolerances[second_face],
            vertices[first_opposite],
            asset_id,
            name,
        )

    visited = {0}
    pending = [0]
    while pending:
        face_index = pending.pop()
        for neighbor in face_neighbors[face_index] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    if len(visited) != len(faces):
        raise ValueError(
            f"{asset_id} collision part must be one connected mesh: {name}"
        )


def _validate_convex_edge(
    vertices: tuple[Vector3, ...],
    face: Triangle,
    normal: Vector3,
    tolerance: float,
    adjacent_opposite: Vector3,
    asset_id: str,
    name: str,
) -> None:
    face_point = vertices[face[0]]
    if _dot(normal, _subtract(adjacent_opposite, face_point)) > tolerance:
        raise ValueError(f"{asset_id} collision part is not convex: {name}")


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(first[index] - second[index] for index in range(3))  # type: ignore[return-value]


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(first[index] * second[index] for index in range(3))


def _vector3(value: Any, label: str) -> Vector3:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{label} must contain only numeric values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain only finite values")
    return result  # type: ignore[return-value]


def _component_source_mesh_sha256(
    component_manifest: Mapping[str, Any], asset_id: str
) -> str:
    outputs = component_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError(f"{asset_id} outputs must be an object")
    handle = outputs.get("handle")
    if not isinstance(handle, Mapping):
        raise ValueError(f"{asset_id} outputs.handle must be an object")
    return _digest(handle.get("sha256"), f"{asset_id} outputs.handle.sha256")


def _component_hinge_side(component_manifest: Mapping[str, Any], asset_id: str) -> str:
    handedness = component_manifest.get("handedness")
    if not isinstance(handedness, Mapping):
        raise ValueError(f"{asset_id} handedness must be an object")
    hinge_side = handedness.get("hinge_side")
    if hinge_side not in {"left", "right"}:
        raise ValueError(f"{asset_id} handedness.hinge_side is invalid")
    return str(hinge_side)


def _validate_payload_handedness(
    payload: Mapping[str, Any], asset_id: str, *, expected_side: str
) -> str:
    handedness = payload.get("handedness")
    if handedness is None and expected_side == "left":
        return "left"
    if not isinstance(handedness, Mapping):
        raise ValueError(f"{asset_id} collision supplement handedness is invalid")
    hinge_side = handedness.get("hinge_side")
    if hinge_side != expected_side:
        raise ValueError(f"{asset_id} collision supplement handedness side is invalid")
    expected_transform_id = (
        _MIRROR_TRANSFORM_ID if expected_side == "right" else _IDENTITY_TRANSFORM_ID
    )
    expected_matrix = _MIRROR_MATRIX if expected_side == "right" else _IDENTITY_MATRIX
    if handedness.get("transform_id") != expected_transform_id:
        raise ValueError(
            f"{asset_id} collision supplement handedness transform is invalid"
        )
    raw_matrix = handedness.get("matrix")
    if not isinstance(raw_matrix, list) or len(raw_matrix) != 3:
        raise ValueError(
            f"{asset_id} collision supplement handedness matrix is invalid"
        )
    matrix = tuple(
        _vector3(row, f"{asset_id} collision supplement handedness.matrix")
        for row in raw_matrix
    )
    if matrix != expected_matrix:
        raise ValueError(
            f"{asset_id} collision supplement handedness matrix is invalid"
        )
    identity_sha256 = _digest(
        handedness.get("identity_sha256"),
        f"{asset_id} collision supplement handedness.identity_sha256",
    )
    expected_identity_sha256 = _json_sha256(
        {
            "hinge_side": expected_side,
            "transform_id": expected_transform_id,
            "matrix": [list(row) for row in expected_matrix],
        }
    )
    if identity_sha256 != expected_identity_sha256:
        raise ValueError(
            f"{asset_id} collision supplement handedness identity mismatch"
        )
    return str(hinge_side)


def _sidecar_path(root: Path, value: Any, asset_id: str) -> Path:
    relative = _required_text(value, f"{asset_id} collision_supplement.path")
    if Path(relative).is_absolute() or "\\" in relative:
        raise ValueError(f"{asset_id} collision supplement path must be relative POSIX")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{asset_id} collision supplement escapes catalog root")
    if not path.is_file():
        raise FileNotFoundError(f"{asset_id} collision supplement not found: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid handle collision supplement: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"handle collision supplement must be an object: {path}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _expect_equal(
    value: Mapping[str, Any], key: str, expected: object, asset_id: str
) -> None:
    actual = value.get(key)
    if actual != expected:
        raise ValueError(
            f"{asset_id} collision supplement {key} mismatch: "
            f"expected {expected!r}, got {actual!r}"
        )


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strict_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _digest(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "HandleCollisionPart",
    "HandleCollisionSupplement",
    "load_handle_collision_supplement",
]
