"""Generate w2_b1 panel assembly from objects/ XML files.

Reads switch XMLs from objects/ and assembles them into the final panel.
Edit objects/*.xml to customise, then re-run this script.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).parent

# 14 switches from Excel w2_b1 (positions with known coordinates)
# (class_name, x_mm, y_mm, wall_row, wall_col)
SWITCHES = [
    ("Knob_Quantity1_02",    -286.5,  160.0, 1, 10),
    ("Knob_Quantity1_01",    -356.5,  160.0, 1, 11),
    ("Toogle_Quantity1_01",   217.5,   50.0, 2,  2),
    ("Toogle_Quantity1_02",   111.5,   50.0, 2,  3),
    ("Toogle_Quantity1_03",  -100.5,   50.0, 2,  5),
    ("Toogle_Quantity3_01",  -206.5,   50.0, 2,  6),
    ("Switch_Quantity3_03",  -312.5,   50.0, 2,  7),
    ("Switch_Quantity1_01",   323.5,  -80.0, 3,  1),
    ("Switch_Quantity4_01",   217.5,  -80.0, 3,  2),
    ("Switch_Quantity1_03",   111.5,  -80.0, 3,  3),
    ("Switch_Quantity2_02",     5.5,  -80.0, 3,  4),
    ("Switch_Quantity3_02",  -100.5,  -80.0, 3,  5),
    ("Switch_Quantity1_05",  -206.5,  -80.0, 3,  6),
    ("Switch_Quantity1_04",  -312.5,  -80.0, 3,  7),
]

PANEL_HALF_X = 0.42
PANEL_HALF_Y = 0.025
PANEL_HALF_Z = 0.22
BASE_POS_Y = 30.0


def parse_switch_xml(xml_path: Path) -> tuple[list[ET.Element], list[ET.Element], str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    materials = list(root.iter("material"))
    body = root.find(".//worldbody/body")
    geoms = list(body.iter("geom")) if body is not None else []
    body_name = body.get("name", "") if body is not None else ""
    return materials, geoms, body_name


def generate_layout_yaml() -> str:
    lines = [
        "# w2_b1 switch wall layout",
        f"# {len(SWITCHES)} switches",
        "",
        "panel:",
        "  xml: w2_b1_panel_base.xml",
        "  attach_to_body: panel_mount",
        "",
        "output_xml: generated/w2_b1_panel_assembly.xml",
        "",
        "layout:",
        "  coordinate_scale: 0.001",
        "  face_axes: [x, z]",
        f"  base_pos: [0.0, {BASE_POS_Y}, 0.0]",
        "  default_quat: [0.7071068, -0.7071068, 0.0, 0.0]",
        "  remove_root_joints: true",
        "",
        "placements:",
    ]
    for idx, (cname, x, y, r, c) in enumerate(SWITCHES, 1):
        lines.append(f"  # {idx}. {cname} | wall ({r},{c})")
        lines.append(f"  - xml: objects/{cname}.xml")
        lines.append(f"    pos: [{x}, {y}]")
        lines.append(f"    row: {r}")
        lines.append(f"    col: {c}")
    return "\n".join(lines) + "\n"


def generate_panel_base_xml() -> str:
    return f"""<mujoco model="w2_b1_panel_base">
  <worldbody>
    <body name="panel_mount" pos="0 0 0">
      <geom name="panel_backplate" type="box" pos="0 0 0"
            size="{PANEL_HALF_X} {PANEL_HALF_Y} {PANEL_HALF_Z}"
            rgba="0.86 0.87 0.89 1" contype="1" conaffinity="1"/>
      <site name="panel_origin" pos="0 {PANEL_HALF_Y} 0" size="0.006" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def generate_assembly_xml() -> str:
    coord_scale = 0.001
    base_y = BASE_POS_Y * coord_scale
    quat = "0.7071068 -0.7071068 0 0"
    objects_dir = BASE_DIR / "objects"

    asset_lines: list[str] = []
    slot_lines: list[str] = []

    for cname, x_mm, y_mm, r, c in SWITCHES:
        xml_path = objects_dir / f"{cname}.xml"
        if not xml_path.exists():
            print(f"  WARNING: {xml_path.name} not found, skipping")
            continue
        materials, geoms, body_name = parse_switch_xml(xml_path)
        x_m = x_mm * coord_scale
        z_m = y_mm * coord_scale
        slot_name = f"panel_slot_r{r}_c{c}"
        assembled_body = f"{body_name}_r{r}_c{c}" if body_name else f"{cname}_r{r}_c{c}"

        for mat in materials:
            asset_lines.append(f"    {ET.tostring(mat, encoding='unicode').strip()}")

        slot_lines.append(f'      <body name="{slot_name}" pos="{x_m} {base_y} {z_m}" quat="{quat}">')
        slot_lines.append(f'        <body name="{assembled_body}">')
        for geom in geoms:
            orig = geom.get("name", "")
            if orig:
                geom.set("name", f"{orig}__{slot_name}")
            slot_lines.append(f"          {ET.tostring(geom, encoding='unicode').strip()}")
        slot_lines.append("        </body>")
        slot_lines.append("      </body>")

    return f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="w2_b1_panel_base">
  <asset>
{chr(10).join(asset_lines)}
  </asset>
  <worldbody>
    <body name="panel_mount" pos="0 0 0">
      <geom name="panel_backplate" type="box" pos="0 0 0" size="{PANEL_HALF_X} {PANEL_HALF_Y} {PANEL_HALF_Z}" rgba="0.86 0.87 0.89 1" contype="1" conaffinity="1" />
      <site name="panel_origin" pos="0 {PANEL_HALF_Y} 0" size="0.006" rgba="1 0 0 1" />
{chr(10).join(slot_lines)}
    </body>
  </worldbody>
</mujoco>"""


def main() -> None:
    objects_dir = BASE_DIR / "objects"
    if not objects_dir.exists():
        print(f"ERROR: {objects_dir} does not exist.")
        return
    (BASE_DIR / "generated").mkdir(exist_ok=True)

    (BASE_DIR / "layout_w2_b1.yaml").write_text(generate_layout_yaml(), encoding="utf-8")
    print("  Created layout_w2_b1.yaml")

    (BASE_DIR / "w2_b1_panel_base.xml").write_text(generate_panel_base_xml(), encoding="utf-8")
    print("  Created w2_b1_panel_base.xml")

    (BASE_DIR / "generated" / "w2_b1_panel_assembly.xml").write_text(generate_assembly_xml(), encoding="utf-8")
    print("  Created generated/w2_b1_panel_assembly.xml")

    print(f"\nDone. Assembly built from {len(SWITCHES)} switches.")


if __name__ == "__main__":
    main()
