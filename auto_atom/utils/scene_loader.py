"""Compose a scene XML with one or more robot XMLs at load time.

The scene XML declares only task-specific geometry (tables, objects, cameras).
Robot XMLs are injected as ``<include>`` siblings directly under ``<mujoco>``
at load time, so the same scene file can be reused across robots without
duplicating XML.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import mujoco


def _absolutize_asset_paths(root: ET.Element, source_dir: Path) -> None:
    """In-place rewrite of relative <mesh>/<model>/<texture> file= and
    <compiler meshdir/texturedir/assetdir> attributes to absolute paths,
    using ``source_dir`` (the directory of the file these attributes were
    authored in) as the base.

    After this pass the element subtree is self-contained: it can be inlined
    into any other <mujoco> document without depending on which compiler's
    meshdir wins the parse-time merge or on the loader's CWD.
    """
    # Resolve compiler dir attributes against source_dir, then drop them so
    # the host scene's compiler is not perturbed.
    meshdir = source_dir
    texturedir = source_dir
    assetdir = source_dir
    for compiler in list(root.findall("compiler")):
        if "assetdir" in compiler.attrib:
            assetdir = (source_dir / compiler.attrib.pop("assetdir")).resolve()
            meshdir = assetdir
            texturedir = assetdir
        if "meshdir" in compiler.attrib:
            meshdir = (source_dir / compiler.attrib.pop("meshdir")).resolve()
        if "texturedir" in compiler.attrib:
            texturedir = (source_dir / compiler.attrib.pop("texturedir")).resolve()

    def _abs(base: Path, value: str) -> str:
        p = Path(value)
        return value if p.is_absolute() else str((base / value).resolve())

    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if f:
            mesh.set("file", _abs(meshdir, f))
    for tex in root.iter("texture"):
        f = tex.get("file")
        if f:
            tex.set("file", _abs(texturedir, f))
    # <model file=...> references sub-XMLs; resolve relative to source_dir
    # (not meshdir). Sub-XMLs are loaded as independent models with their
    # own compiler context, so we don't recurse here.
    for model in root.iter("model"):
        f = model.get("file")
        if f:
            model.set("file", _abs(source_dir, f))


def compose_scene_xml(scene_xml: Path, robot_xmls: Iterable[Path] = ()) -> str:
    """Return the composed XML string with robot XMLs inlined.

    Each robot XML is parsed, its asset paths and compiler dirs are
    absolutized against the robot file's own directory, and its top-level
    children are inlined into the scene's <mujoco> root. This makes the
    composed document independent of MuJoCo's parse-time meshdir merging and
    loader-CWD behavior — important when robot and scene have different
    meshdirs (e.g. ``assets/meshes/p7_arm`` vs ``assets/meshes``).
    """
    scene_xml = Path(scene_xml).resolve()
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"scene root must be <mujoco>, got <{root.tag}>: {scene_xml}")

    for robot in robot_xmls:
        robot_path = Path(robot).resolve()
        if not robot_path.exists():
            raise FileNotFoundError(f"robot XML not found: {robot_path}")
        robot_root = ET.parse(robot_path).getroot()
        if robot_root.tag != "mujoco":
            raise ValueError(
                f"robot root must be <mujoco>, got <{robot_root.tag}>: {robot_path}"
            )
        _absolutize_asset_paths(robot_root, robot_path.parent)
        # Inline robot's children into the scene root, preserving order.
        for child in list(robot_root):
            root.append(child)

    return ET.tostring(root, encoding="unicode")


def load_scene(
    scene_xml: str | Path,
    robot_xmls: Iterable[str | Path] = (),
) -> mujoco.MjModel:
    """Load a scene XML with zero or more robot XMLs injected as includes.

    When ``robot_xmls`` is empty, the scene is loaded directly via
    ``from_xml_path`` (fast path, no XML rewriting). Otherwise the composed
    XML is written to a temporary sibling of the scene file so relative paths
    inside the scene resolve unchanged; robot XML paths become absolute.
    """
    scene_xml = Path(scene_xml).resolve()
    robot_list = [Path(r) for r in robot_xmls]
    if not robot_list:
        return mujoco.MjModel.from_xml_path(str(scene_xml))

    composed = compose_scene_xml(scene_xml, robot_list)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=str(scene_xml.parent),
        prefix="._composed_",
        suffix=".xml",
        delete=False,
    ) as f:
        f.write(composed)
        composed_path = Path(f.name)
    try:
        return mujoco.MjModel.from_xml_path(str(composed_path))
    finally:
        composed_path.unlink(missing_ok=True)
