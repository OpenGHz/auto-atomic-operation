"""Private compiler for the normalized UniDoor catalog format.

The generic adapter is the only caller of this module.  Host-scene merging,
namespacing and model loading stay in the generic composition pipeline; this
module only validates the selected catalog records and emits an MJCF fragment.

Only the normalized catalog contract is consumed here.  Source DAE/URDF files,
Praxis Python modules, and the optional ACD collision sidecar are not required
to compile the visual/AABB model.  ACD metadata remains available in the
component manifests for a future collision adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_COLLISION_CLASS = "unidoor_door_collision"
_MESH_NAMES = {
    "frame": "unidoor_frame_mesh",
    "panel": "unidoor_panel_mesh",
    "handle": "unidoor_handle_mesh",
    "lock": "unidoor_lock_mesh",
}


@dataclass(frozen=True)
class _UniDoorCatalogConfig:
    """Resolved adapter inputs consumed by the private catalog compiler."""

    catalog_root: Path
    door_id: str
    handle_id: str
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    verify_hashes: bool = True
    joint_specs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


def _build_unidoor_fragment(config: _UniDoorCatalogConfig) -> ET.Element:
    root = config.catalog_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"UniDoor catalog root not found: {root}")
    product_path = root / "product_space.json"
    if not product_path.is_file():
        raise FileNotFoundError(f"UniDoor product manifest not found: {product_path}")
    product = _read_json(product_path)
    door, door_ref = _load_component(
        root, product, "doors", config.door_id, config.verify_hashes
    )
    handle, handle_ref = _load_component(
        root, product, "handles", config.handle_id, config.verify_hashes
    )
    if door.get("kind") != "door" or handle.get("kind") != "handle":
        raise ValueError("UniDoor component kinds do not match door/handle dimensions")
    if door.get("status") != "pass" or handle.get("status") != "pass":
        raise ValueError("UniDoor components must have status=pass")

    door_geometry = _mapping(door.get("geometry"), "geometry")
    handle_geometry = _mapping(handle.get("geometry"), "geometry")
    root_translation_raw = door_geometry.get(
        "root_translation_m", door_geometry.get("root_translation")
    )
    if root_translation_raw is None:
        normalization = door.get("normalization")
        if isinstance(normalization, Mapping):
            root_translation_raw = normalization.get("root_translation")
    root_translation = _vector3(root_translation_raw, "root_translation_m")
    handle_position = _vector3(
        door_geometry.get("handle_position_m"), "handle_position_m"
    )
    grasp_offset = _vector3(handle_geometry.get("grasp_offset_m"), "grasp_offset_m")
    handle_bounds = _bounds(handle_geometry.get("handle_bounds_m"), "handle_bounds_m")
    panel_bounds, frame_bands = _door_collision_geometry(
        root, door, door_geometry, config.verify_hashes
    )

    door_outputs = _mapping(door.get("outputs"), "outputs")
    handle_outputs = _mapping(handle.get("outputs"), "outputs")
    frame_path = _output_path(root, door_outputs, "frame", config.verify_hashes)
    panel_path = _output_path(root, door_outputs, "panel", config.verify_hashes)
    handle_path = _output_path(root, handle_outputs, "handle", config.verify_hashes)
    lock_path = None
    if "lock" in handle_outputs:
        lock_path = _output_path(root, handle_outputs, "lock", config.verify_hashes)
    elif bool(handle.get("has_lock_mesh", False)):
        raise ValueError(
            f"{config.handle_id} declares has_lock_mesh but has no lock output"
        )

    door_hinge_axis, handle_hinge_axis, hinge_side = _hinge_axes(product, door, handle)

    # Keep references to the catalog records in custom metadata.  These are
    # useful when a compiled model is inspected without the source manifest.
    combination_id = f"{config.door_id}-{config.handle_id}"
    model_root = ET.Element("mujoco", {"model": f"aao_unidoor_{combination_id}"})
    ET.SubElement(model_root, "compiler", {"angle": "radian"})
    asset = ET.SubElement(model_root, "asset")
    ET.SubElement(
        asset,
        "mesh",
        {"name": _MESH_NAMES["frame"], "file": _absolute_posix(frame_path)},
    )
    ET.SubElement(
        asset,
        "mesh",
        {"name": _MESH_NAMES["panel"], "file": _absolute_posix(panel_path)},
    )
    ET.SubElement(
        asset,
        "mesh",
        {"name": _MESH_NAMES["handle"], "file": _absolute_posix(handle_path)},
    )
    if lock_path is not None:
        ET.SubElement(
            asset,
            "mesh",
            {"name": _MESH_NAMES["lock"], "file": _absolute_posix(lock_path)},
        )

    defaults = ET.SubElement(model_root, "default")
    collision_default = ET.SubElement(defaults, "default", {"class": _COLLISION_CLASS})
    ET.SubElement(
        collision_default,
        "geom",
        {
            "contype": "16",
            "conaffinity": "6",
            "friction": "1 0.05 0.005",
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
        },
    )

    worldbody = ET.SubElement(model_root, "worldbody")
    environment = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "unidoor_assembly",
            "pos": _fmt(config.position),
            "quat": _fmt(_xyzw_to_wxyz(config.orientation)),
        },
    )
    frame = ET.SubElement(environment, "body", {"name": "door_frame"})
    ET.SubElement(
        frame,
        "geom",
        {
            "name": "door_frame_visual",
            "type": "mesh",
            "mesh": _MESH_NAMES["frame"],
            "pos": _fmt(root_translation),
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
        },
    )
    for role in ("left", "right", "top"):
        band = frame_bands.get(role)
        if band is None:
            continue
        _append_box(
            frame, f"door_frame_{role}", band["position_m"], band["half_size_m"]
        )

    panel = ET.SubElement(environment, "body", {"name": "door_panel"})
    ET.SubElement(
        panel,
        "joint",
        _joint_attributes(
            config,
            "door",
            {
                "name": "door_hinge",
                "type": "hinge",
                "axis": _fmt(door_hinge_axis),
                "range": "0 1.5079644737231006",
                "springref": "0",
                "stiffness": "1",
                "damping": "1.2",
                "frictionloss": "0.2",
                "armature": "0.02",
                "limited": "true",
            },
        ),
    )
    panel_center = tuple(
        (panel_bounds[0][index] + panel_bounds[1][index]) / 2 for index in range(3)
    )
    panel_size = tuple(
        max(
            (panel_bounds[1][index] - panel_bounds[0][index]) / 2
            - (0.003 if index in (0, 2) else 0.0),
            0.0005,
        )
        for index in range(3)
    )
    _append_box(panel, "door_panel_collision", panel_center, panel_size, mass="28")
    ET.SubElement(
        panel,
        "geom",
        {
            "name": "door_panel_visual",
            "type": "mesh",
            "mesh": _MESH_NAMES["panel"],
            "pos": _fmt(root_translation),
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
        },
    )

    handle_body = ET.SubElement(
        panel, "body", {"name": "door_handle", "pos": _fmt(handle_position)}
    )
    ET.SubElement(
        handle_body,
        "joint",
        _joint_attributes(
            config,
            "handle",
            {
                "name": "handle_hinge",
                "type": "hinge",
                "axis": _fmt(handle_hinge_axis),
                "range": "0 0.65",
                "springref": "0",
                "stiffness": "3",
                "damping": "0.08",
                "armature": "0.002",
                "limited": "true",
            },
        ),
    )
    handle_center = tuple(
        (handle_bounds[0][index] + handle_bounds[1][index]) / 2 for index in range(3)
    )
    handle_size = tuple(
        max((handle_bounds[1][index] - handle_bounds[0][index]) / 2, 0.001)
        for index in range(3)
    )
    _append_box(
        handle_body, "handle_lever_collision", handle_center, handle_size, mass="0.25"
    )
    ET.SubElement(
        handle_body,
        "geom",
        {
            "name": "door_handle_visual",
            "type": "mesh",
            "mesh": _MESH_NAMES["handle"],
            "group": "2",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
        },
    )
    ET.SubElement(
        handle_body,
        "site",
        {
            "name": "handle_grasp_center",
            "type": "sphere",
            "pos": _fmt(grasp_offset),
            "size": "0.012",
            "rgba": "0.9 0.1 0.1 0.35",
        },
    )
    if lock_path is not None:
        lock_body = ET.SubElement(
            panel, "body", {"name": "door_lock", "pos": _fmt(handle_position)}
        )
        ET.SubElement(
            lock_body,
            "geom",
            {
                "name": "door_lock_visual",
                "type": "mesh",
                "mesh": _MESH_NAMES["lock"],
                "group": "2",
                "contype": "0",
                "conaffinity": "0",
                "density": "0",
            },
        )

    equality = ET.SubElement(model_root, "equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "name": "door_latch_lock",
            "joint1": "door_hinge",
            "polycoef": "0 0 0 0 0",
            "active": "true",
            "solref": "0.002 1",
            "solimp": "0.99 0.999 0.001",
        },
    )
    sensor = ET.SubElement(model_root, "sensor")
    for tag, name, joint in (
        ("jointpos", "door_angle", "door_hinge"),
        ("jointvel", "door_velocity", "door_hinge"),
        ("jointpos", "handle_angle", "handle_hinge"),
        ("jointvel", "handle_velocity", "handle_hinge"),
    ):
        ET.SubElement(sensor, tag, {"name": name, "joint": joint})
    ET.SubElement(
        sensor,
        "framepos",
        {
            "name": "handle_position",
            "objtype": "site",
            "objname": "handle_grasp_center",
        },
    )
    ET.SubElement(
        sensor,
        "framequat",
        {
            "name": "handle_orientation",
            "objtype": "site",
            "objname": "handle_grasp_center",
        },
    )
    custom = ET.SubElement(model_root, "custom")
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": "unidoor_width",
            "data": _fmt((panel_bounds[1][0] - panel_bounds[0][0],)),
        },
    )
    ET.SubElement(
        custom,
        "numeric",
        {
            "name": "unidoor_height",
            "data": _fmt((panel_bounds[1][2] - panel_bounds[0][2],)),
        },
    )
    ET.SubElement(
        custom,
        "numeric",
        {"name": "unidoor_handle_position", "data": _fmt(handle_position)},
    )
    ET.SubElement(
        custom, "numeric", {"name": "unidoor_grasp_offset", "data": _fmt(grasp_offset)}
    )
    ET.SubElement(
        custom, "text", {"name": "unidoor_combination_id", "data": combination_id}
    )
    ET.SubElement(
        custom, "text", {"name": "unidoor_door_asset_id", "data": config.door_id}
    )
    ET.SubElement(
        custom, "text", {"name": "unidoor_handle_asset_id", "data": config.handle_id}
    )
    ET.SubElement(custom, "text", {"name": "unidoor_hinge_side", "data": hinge_side})
    ET.SubElement(
        custom,
        "text",
        {"name": "unidoor_door_manifest", "data": _absolute_posix(door_ref)},
    )
    ET.SubElement(
        custom,
        "text",
        {"name": "unidoor_handle_manifest", "data": _absolute_posix(handle_ref)},
    )
    return model_root


def _load_component(
    root: Path,
    product: Mapping[str, Any],
    dimension: str,
    asset_id: str,
    verify_hashes: bool,
) -> tuple[dict[str, Any], Path]:
    components = product.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("product_space.json is missing components")
    refs = components.get(dimension)
    if not isinstance(refs, list):
        raise ValueError(f"product_space.json is missing components.{dimension}")
    reference = next(
        (
            item
            for item in refs
            if isinstance(item, Mapping) and item.get("asset_id") == asset_id
        ),
        None,
    )
    if reference is None:
        raise KeyError(f"{asset_id} is not present in product_space.json/{dimension}")
    expected_kind = "door" if dimension == "doors" else "handle"
    if reference.get("kind") != expected_kind:
        raise ValueError(
            f"Product reference {asset_id} has kind={reference.get('kind')!r}; "
            f"expected {expected_kind!r}"
        )
    if reference.get("status") != "pass":
        raise ValueError(f"Product reference {asset_id} must have status=pass")
    manifest_path = _safe_path(
        root, str(reference.get("manifest", "")), f"{asset_id} manifest"
    )
    if verify_hashes:
        _verify_hash(
            manifest_path, reference.get("manifest_sha256"), f"{asset_id} manifest"
        )
    manifest = _read_json(manifest_path)
    if manifest.get("asset_id") != asset_id:
        raise ValueError(
            f"Manifest asset_id mismatch: expected {asset_id}, got {manifest.get('asset_id')}"
        )
    return manifest, manifest_path


def _door_collision_geometry(
    root: Path,
    door: Mapping[str, Any],
    geometry: Mapping[str, Any],
    verify_hashes: bool,
) -> tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]],
    dict[str, dict[str, tuple[float, float, float]]],
]:
    supplement_ref = door.get("collision_supplement")
    if isinstance(supplement_ref, Mapping):
        path = _safe_path(
            root, str(supplement_ref.get("path", "")), "door collision supplement"
        )
        if verify_hashes:
            _verify_hash(
                path, supplement_ref.get("sha256"), "door collision supplement"
            )
        supplement = _read_json(path)
        if supplement.get("representation") != "door_box_primitives_v1":
            raise ValueError(f"Unsupported door collision representation: {path}")
        if supplement.get("asset_id") != door.get("asset_id"):
            raise ValueError(f"Door collision supplement asset_id mismatch: {path}")
        primitives = supplement.get("primitives")
        if isinstance(primitives, list):
            panel: (
                tuple[tuple[float, float, float], tuple[float, float, float]] | None
            ) = None
            bands: dict[str, dict[str, tuple[float, float, float]]] = {}
            for item in primitives:
                if not isinstance(item, Mapping):
                    continue
                name = str(item.get("name", ""))
                center = _vector3(item.get("center_m"), f"{name}.center_m")
                half = _vector3(item.get("half_size_m"), f"{name}.half_size_m")
                if any(value <= 0 for value in half):
                    raise ValueError(
                        f"Door collision half-size must be positive: {name}"
                    )
                if name == "door_panel_collision":
                    panel = (
                        tuple(center[index] - half[index] for index in range(3)),
                        tuple(center[index] + half[index] for index in range(3)),
                    )
                elif name.startswith("door_frame_"):
                    bands[name.removeprefix("door_frame_")] = {
                        "position_m": center,
                        "half_size_m": half,
                    }
            if panel is not None and bands:
                return panel, bands

    panel = _bounds(
        geometry.get("panel_collision_bounds_m", geometry.get("panel_bounds_m")),
        "panel_collision_bounds_m",
    )
    raw_bands = geometry.get("frame_collision_bands")
    if isinstance(raw_bands, Mapping):
        bands = {
            str(role): {
                "position_m": _vector3(
                    value.get("position_m"), f"frame {role}.position_m"
                ),
                "half_size_m": _vector3(
                    value.get("half_size_m"), f"frame {role}.half_size_m"
                ),
            }
            for role, value in raw_bands.items()
            if isinstance(value, Mapping)
        }
        if bands:
            return panel, bands
    frame = _bounds(geometry.get("frame_bounds_m"), "frame_bounds_m")
    return panel, _derive_frame_bands(frame, panel)


def _derive_frame_bands(
    frame: tuple[tuple[float, float, float], tuple[float, float, float]],
    panel: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> dict[str, dict[str, tuple[float, float, float]]]:
    frame_min, frame_max = frame
    panel_min, panel_max = panel
    center_y = (frame_min[1] + frame_max[1]) / 2
    half_y = (frame_max[1] - frame_min[1]) / 2
    center_z = (frame_min[2] + frame_max[2]) / 2
    half_z = (frame_max[2] - frame_min[2]) / 2
    bands: dict[str, dict[str, tuple[float, float, float]]] = {}
    left = panel_min[0] - frame_min[0]
    if left > 1e-6:
        bands["left"] = {
            "position_m": ((frame_min[0] + panel_min[0]) / 2, center_y, center_z),
            "half_size_m": (left / 2, half_y, half_z),
        }
    right = frame_max[0] - panel_max[0]
    if right > 1e-6:
        bands["right"] = {
            "position_m": ((panel_max[0] + frame_max[0]) / 2, center_y, center_z),
            "half_size_m": (right / 2, half_y, half_z),
        }
    top = frame_max[2] - panel_max[2]
    if top > 1e-6:
        bands["top"] = {
            "position_m": (
                (frame_min[0] + frame_max[0]) / 2,
                center_y,
                (panel_max[2] + frame_max[2]) / 2,
            ),
            "half_size_m": ((frame_max[0] - frame_min[0]) / 2, half_y, top / 2),
        }
    return bands


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest field {field_name!r} must be an object")
    return value


def _hinge_axes(
    product: Mapping[str, Any],
    door: Mapping[str, Any],
    handle: Mapping[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float], str]:
    """Resolve and validate the handedness contract for one assembly.

    The materialized right-hinge bundle uses a negative-Z door axis and a
    positive-Y lever axis.  Left-hinge catalogs use the mirrored negative-Y
    lever axis together with the canonical positive-Z door axis.  Any explicit side declared
    by the product or either component must agree before geometry is compiled.
    """

    def declared_side(value: Any, label: str) -> str | None:
        if isinstance(value, Mapping):
            raw = value.get("hinge_side")
        elif isinstance(value, str):
            raw = value
        else:
            return None
        if raw is None:
            return None
        side = str(raw).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError(
                f"{label}.hinge_side must be 'left' or 'right', got {raw!r}"
            )
        return side

    product_side = declared_side(product.get("handedness"), "product_space")
    if product_side is None:
        product_side = declared_side(product.get("hinge_side"), "product_space")
    if product_side is None:
        combination_space = product.get("combination_space")
        product_side = declared_side(combination_space, "combination_space")
    door_side = declared_side(door.get("handedness"), "door manifest")
    handle_side = declared_side(handle.get("handedness"), "handle manifest")
    sides = [
        side for side in (product_side, door_side, handle_side) if side is not None
    ]
    if not sides:
        raise ValueError("UniDoor manifests do not declare a hinge side")
    side = sides[0]
    if any(other != side for other in sides[1:]):
        raise ValueError(
            "UniDoor hinge-side mismatch: "
            f"product={product_side!r}, door={door_side!r}, handle={handle_side!r}"
        )
    door_axis = (0.0, 0.0, -1.0 if side == "right" else 1.0)
    handle_axis = (0.0, 1.0 if side == "right" else -1.0, 0.0)
    return door_axis, handle_axis, side


def _safe_path(root: Path, relative: str, label: str) -> Path:
    if not relative:
        raise ValueError(f"{label} path is empty")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes catalog root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _output_path(
    root: Path, outputs: Mapping[str, Any], role: str, verify_hashes: bool
) -> Path:
    output = _mapping(outputs.get(role), f"outputs.{role}")
    path = _safe_path(root, str(output.get("path", "")), f"{role} output")
    if verify_hashes:
        _verify_hash(path, output.get("sha256"), f"{role} output")
    return path


def _verify_hash(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or not expected:
        raise ValueError(f"{label} is missing sha256")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{label} sha256 mismatch: expected {expected}, got {digest}")


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{label} must contain three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result  # type: ignore[return-value]


def _bounds(
    value: Any, label: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{label} must contain lower and upper vectors")
    lower = _vector3(value[0], f"{label}[0]")
    upper = _vector3(value[1], f"{label}[1]")
    if any(upper[index] <= lower[index] for index in range(3)):
        raise ValueError(f"{label} must have positive extents")
    return lower, upper


def _append_box(
    parent: ET.Element,
    name: str,
    position: Sequence[float],
    size: Sequence[float],
    *,
    mass: str | None = None,
) -> None:
    attributes = {
        "name": name,
        "type": "box",
        "pos": _fmt(_vector3(position, f"{name}.position_m")),
        "size": _fmt(_vector3(size, f"{name}.half_size_m")),
        "group": "3",
        "class": _COLLISION_CLASS,
    }
    if mass is not None:
        attributes["mass"] = mass
    ET.SubElement(parent, "geom", attributes)


def _joint_attributes(
    config: _UniDoorCatalogConfig,
    role: str,
    defaults: Mapping[str, str],
) -> dict[str, str]:
    """Apply versioned template dynamics without accepting arbitrary XML attrs."""

    attributes = dict(defaults)
    spec = config.joint_specs.get(role)
    if not isinstance(spec, Mapping):
        return attributes
    for key in (
        "range",
        "springref",
        "stiffness",
        "damping",
        "frictionloss",
        "armature",
        "limited",
    ):
        value = spec.get(key)
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            attributes[key] = _fmt(value)
        elif isinstance(value, bool):
            attributes[key] = str(value).lower()
        else:
            attributes[key] = str(value)
    return attributes


def _xyzw_to_wxyz(value: Sequence[float]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("orientation must contain four values")
    x, y, z, w = (float(item) for item in value)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    return w / norm, x / norm, y / norm, z / norm


def _fmt(values: Sequence[float]) -> str:
    return " ".join(
        "0" if float(value) == 0 else format(float(value), ".15g") for value in values
    )


def _absolute_posix(path: Path) -> str:
    return path.resolve().as_posix()
