from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from omegaconf import OmegaConf


ROOT_SINGLETON_SECTIONS = (
    "compiler",
    "option",
    "size",
    "visual",
    "statistic",
)
ROOT_MERGED_SECTIONS = (
    "default",
    "asset",
    "contact",
    "equality",
    "tendon",
    "actuator",
    "sensor",
    "custom",
    "extension",
    "deformable",
    "keyframe",
)
PRE_WORLDBODY_SECTIONS = {
    "include",
    "compiler",
    "option",
    "size",
    "visual",
    "statistic",
    "default",
    "asset",
    "contact",
    "equality",
    "tendon",
    "actuator",
    "sensor",
    "custom",
    "extension",
    "deformable",
}
SKIP_NAMESPACE_SECTIONS = {"asset", "default"}
RUNTIME_REFERENCE_ATTRS = {
    "actuator",
    "body",
    "body1",
    "body2",
    "cranksite",
    "geom",
    "geom1",
    "geom2",
    "joint",
    "joint1",
    "joint2",
    "jointinparent",
    "objname",
    "site",
    "site1",
    "site2",
    "slidersite",
    "target",
    "tendon",
    "tendon1",
    "tendon2",
}


@dataclass(frozen=True)
class SourcePathContext:
    source_dir: Path
    asset_dir: Path | None
    mesh_dir: Path | None
    texture_dir: Path | None


@dataclass(frozen=True)
class PanelPlacement:
    row: int
    col: int
    slot_name: str
    object_name: str
    source_xml: Path


@dataclass(frozen=True)
class PanelAssemblyResult:
    output_path: Path
    placements: tuple[PanelPlacement, ...]


def generate_panel_assembly(
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> PanelAssemblyResult:
    """Generate a final MJCF by attaching configured objects onto a panel."""

    config_file = Path(config_path).expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    if not isinstance(raw, dict):
        raise TypeError(f"YAML root must be a mapping: {config_file}")

    panel_cfg = _require_mapping(raw, "panel")
    layout_cfg = _require_mapping(raw, "layout")
    placements_cfg = raw.get("placements")
    if not isinstance(placements_cfg, list):
        raise TypeError("`placements` must be a 2-D YAML list.")

    panel_xml_path = _resolve_config_path(
        config_file, _require_string(panel_cfg, "xml")
    )
    final_output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else _resolve_output_path(config_file, raw)
    )
    final_output.parent.mkdir(parents=True, exist_ok=True)

    x_coords = _require_number_list(layout_cfg, "x_coords")
    y_coords = _require_number_list(layout_cfg, "y_coords")
    coordinate_scale = float(layout_cfg.get("coordinate_scale", 1.0))
    base_pos = _scaled_xyz(
        layout_cfg.get("base_pos", [0.0, 0.0, 0.0]), coordinate_scale
    )
    face_axes = _face_axes(layout_cfg.get("face_axes", ["x", "z"]))
    default_quat = _quat_or_none(layout_cfg.get("default_quat")) or (1.0, 0.0, 0.0, 0.0)
    remove_root_joints_default = bool(layout_cfg.get("remove_root_joints", True))

    if len(placements_cfg) != len(y_coords):
        raise ValueError(
            "`placements` row count does not match layout `y_coords`: "
            f"{len(placements_cfg)} != {len(y_coords)}"
        )

    panel_tree = ET.parse(panel_xml_path)
    panel_root = panel_tree.getroot()
    if panel_root.tag != "mujoco":
        raise ValueError(f"Expected panel XML root to be <mujoco>: {panel_xml_path}")

    _rewrite_tree_paths(panel_root, panel_xml_path, final_output)

    panel_worldbody = panel_root.find("worldbody")
    if panel_worldbody is None:
        raise ValueError(f"Panel XML is missing <worldbody>: {panel_xml_path}")

    attach_to_body = panel_cfg.get("attach_to_body")
    if attach_to_body is None:
        attach_parent = panel_worldbody
    else:
        if not isinstance(attach_to_body, str) or not attach_to_body:
            raise TypeError(
                "`panel.attach_to_body` must be a non-empty string when provided."
            )
        attach_parent = _find_body_by_name(panel_worldbody, attach_to_body)
        if attach_parent is None:
            raise ValueError(
                f"Body '{attach_to_body}' was not found in panel XML: {panel_xml_path}"
            )

    placements: list[PanelPlacement] = []
    for row_idx, row in enumerate(placements_cfg):
        if not isinstance(row, list):
            raise TypeError(f"`placements[{row_idx}]` must be a YAML list.")
        if len(row) != len(x_coords):
            raise ValueError(
                f"`placements[{row_idx}]` column count does not match layout `x_coords`: "
                f"{len(row)} != {len(x_coords)}"
            )

        for col_idx, cell in enumerate(row):
            if cell is None:
                continue

            placement_cfg = _normalize_cell_config(
                cell, row_idx=row_idx, col_idx=col_idx
            )
            object_xml_path = _resolve_config_path(config_file, placement_cfg["xml"])
            object_tree = ET.parse(object_xml_path)
            object_root = object_tree.getroot()
            object_body_name = placement_cfg.get("body_name")

            _extract_object_body(object_root, object_body_name)
            object_root_copy = deepcopy(object_root)
            _rewrite_tree_paths(object_root_copy, object_xml_path, final_output)
            object_body_copy = _extract_object_body(object_root_copy, object_body_name)
            if _coerce_bool(
                placement_cfg.get("remove_root_joints"),
                default=remove_root_joints_default,
            ):
                _strip_root_joints(object_body_copy)

            _merge_root_dependencies(panel_root, object_root_copy)

            slot_name = (
                placement_cfg.get("slot_name") or f"panel_slot_r{row_idx}_c{col_idx}"
            )
            mounted_name = (
                placement_cfg.get("mounted_body_name")
                or f"{object_body_copy.get('name', object_xml_path.stem)}_r{row_idx}_c{col_idx}"
            )
            _namespace_object_model(
                object_root_copy,
                object_body=object_body_copy,
                mounted_body_name=mounted_name,
                namespace_suffix=f"r{row_idx}_c{col_idx}",
            )

            slot_pos = _apply_face_offsets(
                base_pos=base_pos,
                face_axes=face_axes,
                x_offset=x_coords[col_idx] * coordinate_scale,
                y_offset=y_coords[row_idx] * coordinate_scale,
            )
            pos_offset = _scaled_xyz(
                placement_cfg.get("pos_offset", [0.0, 0.0, 0.0]),
                coordinate_scale,
            )
            slot_pos = (
                slot_pos[0] + pos_offset[0],
                slot_pos[1] + pos_offset[1],
                slot_pos[2] + pos_offset[2],
            )
            slot_quat = _quat_or_none(placement_cfg.get("quat")) or default_quat

            slot_body = ET.Element(
                "body",
                {
                    "name": slot_name,
                    "pos": _format_vec(slot_pos),
                    "quat": _format_vec(slot_quat),
                },
            )
            slot_body.append(object_body_copy)
            attach_parent.append(slot_body)

            placements.append(
                PanelPlacement(
                    row=row_idx,
                    col=col_idx,
                    slot_name=slot_name,
                    object_name=mounted_name,
                    source_xml=object_xml_path,
                )
            )

    ET.indent(panel_tree, space="  ")
    panel_tree.write(final_output, encoding="utf-8", xml_declaration=True)
    return PanelAssemblyResult(output_path=final_output, placements=tuple(placements))


def _resolve_output_path(config_file: Path, raw: dict[str, Any]) -> Path:
    output_raw = raw.get("output_xml")
    if not isinstance(output_raw, str) or not output_raw:
        raise TypeError("`output_xml` must be provided as a non-empty string.")
    return _resolve_config_path(config_file, output_raw)


def _resolve_config_path(config_file: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (config_file.parent / path).resolve()
    return path


def _require_mapping(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"`{key}` must be a mapping.")
    return value


def _require_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"`{key}` must be a non-empty string.")
    return value


def _require_number_list(raw: dict[str, Any], key: str) -> list[float]:
    value = raw.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"`{key}` must be a non-empty list.")
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"`{key}` must only contain numbers.") from exc


def _scaled_xyz(raw: Any, scale: float) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise TypeError("Expected a 3-element position vector.")
    try:
        values = tuple(float(v) * scale for v in raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Position vectors must only contain numbers.") from exc
    return values


def _face_axes(raw: Any) -> tuple[int, int]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise TypeError("`layout.face_axes` must be a 2-element list like ['x', 'z'].")
    axis_map = {"x": 0, "y": 1, "z": 2}
    try:
        axes = tuple(axis_map[str(axis)] for axis in raw)
    except KeyError as exc:
        raise TypeError(
            "`layout.face_axes` only supports the axis names x, y, z."
        ) from exc
    if axes[0] == axes[1]:
        raise ValueError("`layout.face_axes` must contain two different axes.")
    return axes


def _quat_or_none(raw: Any) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise TypeError("Quaternion values must be a 4-element list.")
    try:
        return tuple(float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise TypeError("Quaternion values must only contain numbers.") from exc


def _normalize_cell_config(cell: Any, *, row_idx: int, col_idx: int) -> dict[str, Any]:
    if isinstance(cell, str):
        return {"xml": cell}
    if isinstance(cell, dict):
        xml_path = cell.get("xml")
        if not isinstance(xml_path, str) or not xml_path:
            raise TypeError(
                f"`placements[{row_idx}][{col_idx}].xml` must be a non-empty string."
            )
        return dict(cell)
    raise TypeError(
        f"`placements[{row_idx}][{col_idx}]` must be null, a string path, or a mapping."
    )


def _extract_object_body(root: ET.Element, body_name: str | None) -> ET.Element:
    if root.tag == "body":
        if body_name not in (None, "", root.get("name")):
            raise ValueError(
                f"Requested body '{body_name}' but the XML root body is '{root.get('name')}'."
            )
        return root

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Object XML must have a <worldbody> or root <body>.")

    bodies = [child for child in worldbody if child.tag == "body"]
    if body_name is None:
        if len(bodies) != 1:
            raise ValueError(
                "Object XML must contain exactly one top-level body when `body_name` is omitted."
            )
        return bodies[0]

    for body in bodies:
        if body.get("name") == body_name:
            return body

    raise ValueError(
        f"Body '{body_name}' was not found among the object's top-level bodies."
    )


def _strip_root_joints(body: ET.Element) -> None:
    removable = [child for child in list(body) if child.tag in {"joint", "freejoint"}]
    for child in removable:
        body.remove(child)


def _namespace_object_model(
    root: ET.Element,
    *,
    object_body: ET.Element,
    mounted_body_name: str,
    namespace_suffix: str,
) -> None:
    rename_map: dict[str, str] = {}

    def visit(element: ET.Element, *, skip_namespace: bool) -> None:
        current_skip = skip_namespace or element.tag in SKIP_NAMESPACE_SECTIONS
        if not current_skip and "name" in element.attrib:
            old_name = element.get("name")
            assert old_name is not None
            if element is object_body:
                new_name = mounted_body_name
            else:
                new_name = f"{old_name}__{namespace_suffix}"
            rename_map[old_name] = new_name
            element.set("name", new_name)

        for child in list(element):
            visit(child, skip_namespace=current_skip)

    visit(root, skip_namespace=False)

    for element in root.iter():
        for attr_name, attr_value in list(element.attrib.items()):
            if attr_name not in RUNTIME_REFERENCE_ATTRS:
                continue
            if attr_value in rename_map:
                element.set(attr_name, rename_map[attr_value])


def _merge_root_dependencies(target_root: ET.Element, source_root: ET.Element) -> None:
    for include in source_root.findall("include"):
        _append_unique_root_child(target_root, include)

    for section_name in ROOT_SINGLETON_SECTIONS:
        source_section = source_root.find(section_name)
        if source_section is None:
            continue
        if target_root.find(section_name) is None:
            _insert_root_child(target_root, deepcopy(source_section))

    for section_name in ROOT_MERGED_SECTIONS:
        source_section = source_root.find(section_name)
        if source_section is None:
            continue
        target_section = target_root.find(section_name)
        if target_section is None:
            target_section = ET.Element(section_name)
            _insert_root_child(target_root, target_section)
        for child in list(source_section):
            _append_unique_child(target_section, child)


def _append_unique_root_child(target_root: ET.Element, child: ET.Element) -> None:
    child_key = _element_key(child)
    for existing in target_root.findall(child.tag):
        if _element_key(existing) == child_key:
            if _elements_equal(existing, child):
                return
            raise ValueError(
                f"Conflicting top-level <{child.tag}> entry detected for key {child_key!r}."
            )
    _insert_root_child(target_root, deepcopy(child))


def _append_unique_child(target_parent: ET.Element, child: ET.Element) -> None:
    child_key = _element_key(child)
    for existing in target_parent.findall(child.tag):
        if _element_key(existing) == child_key:
            if _elements_equal(existing, child):
                return
            raise ValueError(
                f"Conflicting <{child.tag}> entry detected in <{target_parent.tag}> "
                f"for key {child_key!r}."
            )
    target_parent.append(deepcopy(child))


def _element_key(element: ET.Element) -> tuple[str, str, str]:
    if "name" in element.attrib:
        return element.tag, "name", element.attrib["name"]
    if "class" in element.attrib:
        return element.tag, "class", element.attrib["class"]
    if "file" in element.attrib:
        return element.tag, "file", element.attrib["file"]
    return element.tag, "xml", ET.tostring(element, encoding="unicode")


def _elements_equal(left: ET.Element, right: ET.Element) -> bool:
    return ET.tostring(left, encoding="unicode") == ET.tostring(
        right, encoding="unicode"
    )


def _find_body_by_name(root: ET.Element, body_name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    return None


def _rewrite_tree_paths(
    root: ET.Element, source_xml_path: Path, output_xml_path: Path
) -> None:
    source_context = _build_source_context(root, source_xml_path)
    output_dir = output_xml_path.parent.resolve()

    compiler = root.find("compiler")
    if compiler is not None:
        for attr_name, current_dir in (
            ("assetdir", source_context.asset_dir),
            ("meshdir", source_context.mesh_dir),
            ("texturedir", source_context.texture_dir),
        ):
            if current_dir is None:
                continue
            compiler.set(attr_name, _relative_path(current_dir, output_dir))

    for element in root.iter():
        for attr_name, attr_value in list(element.attrib.items()):
            if not attr_name.startswith("file"):
                continue
            rewritten = _rewrite_file_attribute(
                element=element,
                attr_name=attr_name,
                attr_value=attr_value,
                source_context=source_context,
                output_dir=output_dir,
            )
            if rewritten is not None:
                element.set(attr_name, rewritten)


def _build_source_context(root: ET.Element, source_xml_path: Path) -> SourcePathContext:
    source_dir = source_xml_path.parent.resolve()
    compiler = root.find("compiler")
    asset_dir = None
    mesh_dir = None
    texture_dir = None
    if compiler is not None:
        asset_dir = _resolve_optional_dir(source_dir, compiler.get("assetdir"))
        mesh_base = asset_dir or source_dir
        texture_base = asset_dir or source_dir
        mesh_dir = _resolve_optional_dir(mesh_base, compiler.get("meshdir"))
        texture_dir = _resolve_optional_dir(texture_base, compiler.get("texturedir"))
    return SourcePathContext(
        source_dir=source_dir,
        asset_dir=asset_dir,
        mesh_dir=mesh_dir,
        texture_dir=texture_dir,
    )


def _resolve_optional_dir(base: Path, raw_path: str | None) -> Path | None:
    if raw_path is None or raw_path == "":
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def _rewrite_file_attribute(
    *,
    element: ET.Element,
    attr_name: str,
    attr_value: str,
    source_context: SourcePathContext,
    output_dir: Path,
) -> str | None:
    if not attr_value:
        return None

    raw_path = Path(attr_value)
    if raw_path.is_absolute():
        return attr_value

    if element.tag == "include":
        source_base = source_context.source_dir
    elif element.tag == "texture":
        source_base = (
            source_context.texture_dir
            or source_context.asset_dir
            or source_context.source_dir
        )
    elif element.tag in {"mesh", "hfield", "skin"} and attr_name == "file":
        source_base = (
            source_context.mesh_dir
            or source_context.asset_dir
            or source_context.source_dir
        )
    else:
        source_base = source_context.source_dir

    resolved = (source_base / raw_path).resolve()
    return _relative_path(resolved, output_dir)


def _relative_path(path: Path, base_dir: Path) -> str:
    return Path(os.path.relpath(path, base_dir)).as_posix()


def _format_vec(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9f}".rstrip("0").rstrip(".") or "0" for value in values)


def _coerce_bool(raw: Any, *, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    raise TypeError("Boolean options must be true/false.")


def _insert_root_child(root: ET.Element, child: ET.Element) -> None:
    if child.tag in PRE_WORLDBODY_SECTIONS:
        worldbody_index = next(
            (idx for idx, item in enumerate(root) if item.tag == "worldbody"),
            len(root),
        )
        root.insert(worldbody_index, child)
        return
    root.append(child)


def _apply_face_offsets(
    *,
    base_pos: tuple[float, float, float],
    face_axes: tuple[int, int],
    x_offset: float,
    y_offset: float,
) -> tuple[float, float, float]:
    coords = list(base_pos)
    coords[face_axes[0]] += x_offset
    coords[face_axes[1]] += y_offset
    return coords[0], coords[1], coords[2]
