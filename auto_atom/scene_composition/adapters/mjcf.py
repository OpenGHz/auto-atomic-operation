"""Adapter for pre-authored MJCF scene modules."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from ..config import MjcfLayerConfig


def _expand_includes(elem: ET.Element, source_dir: Path) -> None:
    """Inline nested MJCF includes before the layer is merged into its host."""

    has_include = False
    new_children: list[ET.Element] = []
    for child in list(elem):
        if child.tag == "include":
            has_include = True
            file_attr = child.get("file")
            if file_attr is None:
                continue
            include_path = (source_dir / file_attr).resolve()
            if not include_path.is_file():
                raise FileNotFoundError(f"included XML not found: {include_path}")
            include_root = ET.parse(include_path).getroot()
            if include_root.tag != "mujoco":
                raise ValueError(
                    "included root must be <mujoco>, "
                    f"got <{include_root.tag}>: {include_path}"
                )
            _expand_includes(include_root, include_path.parent)
            new_children.extend(list(include_root))
        else:
            _expand_includes(child, source_dir)
            new_children.append(child)
    if has_include:
        elem[:] = new_children


def _absolutize_asset_paths(root: ET.Element, source_dir: Path) -> None:
    """Make a layer's asset references independent of its merge location."""

    mesh_dir = source_dir
    texture_dir = source_dir
    for compiler in root.findall("compiler"):
        asset_dir = compiler.attrib.pop("assetdir", None)
        if asset_dir is not None:
            resolved_asset_dir = (source_dir / asset_dir).resolve()
            mesh_dir = resolved_asset_dir
            texture_dir = resolved_asset_dir
        mesh_dir_value = compiler.attrib.pop("meshdir", None)
        if mesh_dir_value is not None:
            mesh_dir = (source_dir / mesh_dir_value).resolve()
        texture_dir_value = compiler.attrib.pop("texturedir", None)
        if texture_dir_value is not None:
            texture_dir = (source_dir / texture_dir_value).resolve()

    def absolute(base: Path, value: str) -> str:
        path = Path(value)
        return value if path.is_absolute() else str((base / value).resolve())

    for mesh in root.iter("mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh.set("file", absolute(mesh_dir, file_name))
    for texture in root.iter("texture"):
        file_name = texture.get("file")
        if file_name:
            texture.set("file", absolute(texture_dir, file_name))
    for model in root.iter("model"):
        file_name = model.get("file")
        if file_name:
            model.set("file", absolute(source_dir, file_name))


def load_mjcf_fragment(config: MjcfLayerConfig) -> tuple[ET.Element, tuple[Path, ...]]:
    """Parse and normalise an MJCF layer without merging it into a host."""

    path = config.path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MJCF layer not found: {path}")
    root = ET.parse(path).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"MJCF layer root must be <mujoco>: {path}")
    _expand_includes(root, path.parent)
    _absolutize_asset_paths(root, path.parent)
    # Path-bearing compiler attributes were removed by the normalisation pass;
    # remaining attributes (angle, autolimits, balanceinertia, ...) are valid
    # scene-wide declarations and are merged by the host composer.
    return deepcopy(root), (path,)


__all__ = ["load_mjcf_fragment"]
