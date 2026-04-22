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


def compose_scene_xml(scene_xml: Path, robot_xmls: Iterable[Path] = ()) -> str:
    """Return the composed XML string with robot includes injected.

    Robot include paths are made absolute so MuJoCo can resolve them regardless
    of where the composed XML is written.
    """
    scene_xml = Path(scene_xml).resolve()
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"scene root must be <mujoco>, got <{root.tag}>: {scene_xml}")

    for i, robot in enumerate(robot_xmls):
        robot_path = Path(robot).resolve()
        if not robot_path.exists():
            raise FileNotFoundError(f"robot XML not found: {robot_path}")
        root.insert(i, ET.Element("include", {"file": str(robot_path)}))

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
