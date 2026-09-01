"""UniDoor adapter for the generic asset-assembly seam.

The adapter is intentionally the only module that knows the supplied UniDoor
catalog layout. The host composer sees only a generic ``SceneContribution``.
The catalog is read in place (OBJ bytes are never copied or changed), while
callers provide only a canonical package descriptor or package directory.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..config import AssetAssemblyLayerConfig
from ..contracts import SceneAssembler, SceneContribution
from ..normalization import apply_asset_normalization, resolve_bounds
from ..package import load_package_descriptor, validate_package_payload
from ._unidoor_catalog import _build_unidoor_fragment, _UniDoorCatalogConfig

_REFERENCE_ATTRIBUTES = {
    "mesh",
    "joint",
    "site",
    "objname",
    "body",
    "actuator",
    "sensor",
    "tendon",
    "target",
    "geom",
    "class",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
}


class UniDoorSceneAssembler(SceneAssembler):
    """Adapter object registered for the normalized UniDoor package view."""

    def assemble(self, config: AssetAssemblyLayerConfig) -> SceneContribution:
        return compile_unidoor_layer(config)


def compile_unidoor_layer(layer: AssetAssemblyLayerConfig) -> SceneContribution:
    """Compile a selected door/handle pair into a namespaced contribution."""

    catalog_root, descriptor, descriptor_path = _resolve_catalog_root(layer.package)
    package_warnings: tuple[str, ...] = ()
    if layer.verify_hashes:
        package_warnings = validate_package_payload(descriptor_path).warnings
    selection = dict(layer.selection)
    try:
        door_id = selection["door"]
        handle_id = selection["handle"]
    except KeyError as exc:
        raise ValueError(
            "unidoor.lever_door@1 requires selection roles 'door' and 'handle'"
        ) from exc

    explicit_axes = _explicit_joint_axes(layer, descriptor)
    catalog_config = _UniDoorCatalogConfig(
        catalog_root=catalog_root,
        door_id=door_id,
        handle_id=handle_id,
        position=layer.placement.position,
        orientation=layer.placement.orientation_xyzw,
        verify_hashes=layer.verify_hashes,
        joint_specs=_joint_specs(layer, descriptor),
    )
    compilation = _build_unidoor_fragment(catalog_config)
    fragment = compilation.fragment
    normalization_diagnostics = apply_asset_normalization(
        fragment,
        layer.scaling,
        layer.anchors,
        compilation.metadata,
    )
    _update_unidoor_custom_metrics(
        fragment,
        layer.scaling,
        compilation.metadata,
    )
    _remove_source_only_sections(fragment)
    _sanitize_provenance_metadata(fragment, catalog_root)
    _apply_joint_axes(fragment, explicit_axes)
    semantic_refs = _namespace_fragment(fragment, layer.namespace)
    dependencies = _collect_dependencies(
        fragment, catalog_root, door_id=door_id, handle_id=handle_id
    )
    dependencies.add(descriptor_path)
    descriptor_integrity = descriptor.get("integrity")
    lock_uri = (
        descriptor_integrity.get("lock")
        if isinstance(descriptor_integrity, Mapping)
        else None
    )
    if isinstance(lock_uri, str):
        lock_path = (descriptor_path.parent / lock_uri).resolve()
        if lock_path.is_file():
            dependencies.add(lock_path)
    diagnostics: list[str] = [*package_warnings, *normalization_diagnostics]
    return SceneContribution(
        fragment=fragment,
        semantic_refs=semantic_refs,
        dependencies=tuple(sorted(dependencies)),
        adapter=layer.adapter,
        diagnostics=tuple(diagnostics),
    )


def _update_unidoor_custom_metrics(
    fragment: ET.Element,
    scaling: Sequence[Any],
    metadata: Mapping[str, Any],
) -> None:
    """Keep UniDoor inspection metadata aligned with compiled geometry."""

    custom = fragment.find("custom")
    if custom is None:
        return

    def set_numeric(name: str, values: Sequence[float]) -> None:
        element = next(
            (
                item
                for item in custom
                if item.tag == "numeric" and item.get("name") == name
            ),
            None,
        )
        if element is not None:
            element.set(
                "data", " ".join(format(float(value), ".15g") for value in values)
            )

    for rule in scaling:
        if rule.source_bounds == "door.geometry.panel_bounds_m":
            bounds = resolve_bounds(metadata, rule.source_bounds)
            source_extent = (
                bounds[1]["xyz".index(rule.axis)] - bounds[0]["xyz".index(rule.axis)]
            )
            factor = float(rule.target_extent_m) / source_extent
            set_numeric("unidoor_width", ((bounds[1][0] - bounds[0][0]) * factor,))
            set_numeric("unidoor_height", (float(rule.target_extent_m),))
        elif rule.source_bounds == "handle.geometry.handle_bounds_m":
            set_numeric("unidoor_lever_length", (float(rule.target_extent_m),))

    handle = next(
        (item for item in fragment.iter("body") if item.get("name") == "door_handle"),
        None,
    )
    if handle is not None and handle.get("pos") is not None:
        set_numeric(
            "unidoor_handle_position",
            tuple(float(value) for value in handle.get("pos").split()),
        )
    grasp = next(
        (
            item
            for item in fragment.iter("site")
            if item.get("name") == "handle_grasp_center"
        ),
        None,
    )
    if grasp is not None and grasp.get("pos") is not None:
        set_numeric(
            "unidoor_grasp_offset",
            tuple(float(value) for value in grasp.get("pos").split()),
        )


def _resolve_catalog_root(
    package: Path,
) -> tuple[Path, Mapping[str, Any], Path]:
    package = package.expanduser().resolve()
    descriptor_path = (
        package / "scene_asset_package.json" if package.is_dir() else package
    )
    if not descriptor_path.is_file():
        raise FileNotFoundError(
            f"scene asset package descriptor not found: {descriptor_path}"
        )
    try:
        descriptor_model = load_package_descriptor(descriptor_path)
        descriptor = descriptor_model.model_dump(mode="python", by_alias=True)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"invalid scene asset package descriptor: {descriptor_path}"
        ) from exc
    payload = descriptor["payload_root"]
    relative = payload["uri"]
    root = (descriptor_path.parent / relative).resolve()
    if not root.is_dir() or not (root / "product_space.json").is_file():
        raise ValueError(
            "payload_root must resolve to a UniDoor catalog containing "
            "product_space.json"
        )
    return root, descriptor, descriptor_path


def _remove_source_only_sections(fragment: ET.Element) -> None:
    """Drop compiler metadata; host composition owns singleton sections."""

    for tag in ("compiler",):
        for element in list(fragment.findall(tag)):
            fragment.remove(element)


def _sanitize_provenance_metadata(fragment: ET.Element, catalog_root: Path) -> None:
    """Keep host-visible custom metadata relocatable and path-safe."""

    for text in fragment.iter("text"):
        data = text.get("data")
        if not data:
            continue
        candidate = Path(data)
        if not candidate.is_absolute():
            continue
        try:
            text.set("data", candidate.resolve().relative_to(catalog_root).as_posix())
        except ValueError:
            text.set("data", "<external-provenance>")


def _namespace_fragment(fragment: ET.Element, namespace: str) -> dict[str, str]:
    """Qualify every generated MJCF symbol and rewrite references to it."""

    prefix = f"{namespace}__"
    names: dict[str, str] = {}
    for element in fragment.iter():
        name = element.get("name")
        if name:
            names[name] = f"{prefix}{name}"
        if element.tag == "default":
            class_name = element.get("class")
            if class_name:
                names[class_name] = f"{prefix}{class_name}"
    for element in fragment.iter():
        name = element.get("name")
        if name in names:
            element.set("name", names[name])
        for key, value in list(element.attrib.items()):
            if key in _REFERENCE_ATTRIBUTES and value in names:
                element.set(key, names[value])

    return {
        "door.frame.body": names.get("door_frame", f"{prefix}door_frame"),
        "door.panel.body": names.get("door_panel", f"{prefix}door_panel"),
        "door.hinge.joint": names.get("door_hinge", f"{prefix}door_hinge"),
        "door.latch.constraint": names.get(
            "door_latch_lock", f"{prefix}door_latch_lock"
        ),
        "handle.body": names.get("door_handle", f"{prefix}door_handle"),
        "handle.hinge.joint": names.get("handle_hinge", f"{prefix}handle_hinge"),
        "handle.grasp.site": names.get(
            "handle_grasp_center", f"{prefix}handle_grasp_center"
        ),
        "handle.object": names.get("door_handle", f"{prefix}door_handle"),
    }


def _collect_dependencies(
    fragment: ET.Element,
    catalog_root: Path,
    *,
    door_id: str,
    handle_id: str,
) -> set[Path]:
    dependencies = {catalog_root / "product_space.json"}
    for mesh in fragment.iter("mesh"):
        file_name = mesh.get("file")
        if file_name:
            path = Path(file_name)
            if path.is_file():
                dependencies.add(path.resolve())
    selected = (
        catalog_root / "components" / "doors" / door_id,
        catalog_root / "components" / "handles" / handle_id,
    )
    for directory in selected:
        if directory.is_dir():
            dependencies.update(path for path in directory.rglob("*") if path.is_file())
    return dependencies


def _explicit_joint_axes(
    layer: AssetAssemblyLayerConfig, descriptor: Mapping[str, Any]
) -> dict[str, tuple[float, float, float]]:
    options = layer.options
    axes: Any = options.get("joint_axes") if isinstance(options, Mapping) else None
    if axes is None:
        templates = descriptor.get("assembly_templates")
        if isinstance(templates, (list, tuple)):
            for template in templates:
                if (
                    isinstance(template, Mapping)
                    and template.get("adapter") == layer.adapter
                ):
                    axes = template.get("joint_axes")
                    break
    if not isinstance(axes, Mapping) or not {"door", "handle"}.issubset(axes):
        raise ValueError(
            "unidoor.lever_door@1 requires explicit door and handle joint_axes "
            "in layer options or its package assembly template"
        )
    result: dict[str, tuple[float, float, float]] = {}
    for role in ("door", "handle"):
        value = axes[role]
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != 3
        ):
            raise ValueError(f"explicit joint axis for {role!r} must contain 3 values")
        vector = tuple(float(item) for item in value)
        if (
            not all(-1.0 <= item <= 1.0 for item in vector)
            or sum(item * item for item in vector) <= 1e-12
        ):
            raise ValueError(f"explicit joint axis for {role!r} is invalid")
        result[role] = vector
    return result


def _joint_specs(
    layer: AssetAssemblyLayerConfig, descriptor: Mapping[str, Any]
) -> Mapping[str, Mapping[str, Any]]:
    options = layer.options
    if isinstance(options, Mapping) and isinstance(options.get("joint_specs"), Mapping):
        return options["joint_specs"]
    templates = descriptor.get("assembly_templates")
    if isinstance(templates, (list, tuple)):
        for template in templates:
            if (
                isinstance(template, Mapping)
                and template.get("adapter") == layer.adapter
            ):
                specs = template.get("joint_specs")
                if isinstance(specs, Mapping) and {"door", "handle"}.issubset(specs):
                    return specs
    raise ValueError(
        "unidoor.lever_door@1 requires explicit door and handle joint_specs "
        "in layer options or its package assembly template"
    )


def _apply_joint_axes(
    fragment: ET.Element, axes: Mapping[str, tuple[float, float, float]]
) -> None:
    for joint in fragment.iter("joint"):
        name = joint.get("name")
        if name == "door_hinge":
            joint.set("axis", _format_vector(axes["door"]))
        elif name == "handle_hinge":
            joint.set("axis", _format_vector(axes["handle"]))


def _format_vector(values: tuple[float, float, float]) -> str:
    return " ".join("0" if value == 0 else format(value, ".15g") for value in values)


__all__ = ["UniDoorSceneAssembler", "compile_unidoor_layer"]
