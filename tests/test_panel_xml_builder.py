from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.utils.panel_xml_builder import generate_panel_assembly


def test_generate_panel_assembly_from_demo_config(tmp_path) -> None:
    config_path = ROOT / "examples" / "panel_assembly" / "layout_7x3.yaml"
    output_path = tmp_path / "demo_panel_with_objects.xml"

    result = generate_panel_assembly(config_path, output_path=output_path)

    assert result.output_path == output_path
    assert len(result.placements) == 20

    root = ET.parse(output_path).getroot()
    top_left_slot = _find_body(root, "panel_slot_r0_c0")
    assert top_left_slot is not None
    assert top_left_slot.get("pos") == "-0.3498 0.03 0.12"

    offset_slot = _find_body(root, "panel_slot_r0_c3")
    assert offset_slot is not None
    assert offset_slot.get("pos") == "0 0.03 0.125"

    assert _find_body(root, "panel_slot_r1_c3") is None

    knob_body = _find_body(root, "green_knob_r0_c1")
    assert knob_body is not None
    assert knob_body.find("freejoint") is None
    assert knob_body.find("./geom[@name='green_knob_base__r0_c1']") is not None

    assert root.find("./asset/material[@name='amber_lamp_glow']") is not None
    assert root.find("./default/default[@class='amber_lamp_geom']") is not None

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(output_path))
    assert model.ngeom > 0


def test_generate_panel_assembly_rewrites_relative_file_paths(tmp_path) -> None:
    layout_dir = tmp_path / "layout"
    panel_dir = layout_dir / "panel"
    object_dir = layout_dir / "object"
    mesh_dir = object_dir / "meshes"
    output_dir = layout_dir / "generated"
    panel_dir.mkdir(parents=True)
    mesh_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    (panel_dir / "panel.xml").write_text(
        "<mujoco model='panel'><worldbody><body name='panel_mount'/></worldbody></mujoco>",
        encoding="utf-8",
    )
    (mesh_dir / "thing.stl").write_text(
        "solid thing\nendsolid thing\n", encoding="utf-8"
    )
    (object_dir / "thing.xml").write_text(
        """
<mujoco model="thing">
  <asset>
    <mesh name="thing_mesh" file="meshes/thing.stl"/>
  </asset>
  <worldbody>
    <body name="thing">
      <geom type="mesh" mesh="thing_mesh"/>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    config_path = layout_dir / "layout.yaml"
    config_path.write_text(
        """
panel:
  xml: panel/panel.xml
  attach_to_body: panel_mount

output_xml: generated/final.xml

layout:
  coordinate_scale: 1.0
  face_axes: [x, z]
  x_coords: [0.0]
  y_coords: [0.0]
  base_pos: [0.0, 0.0, 0.0]

placements:
  - - xml: object/thing.xml
""".strip(),
        encoding="utf-8",
    )

    result = generate_panel_assembly(config_path)
    root = ET.parse(result.output_path).getroot()

    mesh = root.find("./asset/mesh[@name='thing_mesh']")
    assert mesh is not None
    assert mesh.get("file") == "../object/meshes/thing.stl"


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None
