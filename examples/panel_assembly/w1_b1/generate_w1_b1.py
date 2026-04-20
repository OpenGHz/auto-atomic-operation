"""Generate w1_b1 panel assembly from objects/ XML files.

Reads the 20 switch XML files from objects/ and assembles them into the
final panel. Edit the XMLs in objects/ directly to customise geometry,
materials, or colours, then re-run this script.

Creates / overwrites:
  - layout_w1_b1.yaml
  - w1_b1_panel_base.xml
  - generated/w1_b1_panel_assembly.xml
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# 20 switch definitions from Excel w1_b1
# (class_name, type, x_mm, y_mm, num_buttons, wall_pos)
# ---------------------------------------------------------------------------
SWITCHES = [
    ("Switch_Quantity2_07", "Switch",  349.8,  120.0, 2, (1,1,1)),
    ("Switch_Quantity2_08", "Switch",  233.2,  120.0, 2, (1,1,2)),
    ("Switch_Quantity2_09", "Switch",  116.6,  120.0, 2, (1,1,3)),
    ("Switch_Quantity2_10", "Switch",    0.0,  120.0, 2, (1,1,4)),
    ("Switch_Quantity4_02", "Switch", -116.6,  120.0, 4, (1,1,5)),
    ("Switch_Quantity1_45", "Switch", -233.2,  120.0, 1, (1,1,6)),
    ("Switch_Quantity1_46", "Switch", -349.8,  120.0, 1, (1,1,7)),
    ("Toogle_Quantity1_05", "Toggle",  349.8,    0.0, 1, (1,2,1)),
    ("Switch_Quantity4_03", "Switch",  233.2,    0.0, 4, (1,2,2)),
    ("Switch_Quantity1_47", "Switch",  116.6,    0.0, 1, (1,2,3)),
    ("Switch_Quantity2_11", "Switch", -116.6,    0.0, 2, (1,2,5)),
    ("Switch_Quantity1_48", "Switch", -233.2,    0.0, 1, (1,2,6)),
    ("Switch_Quantity1_49", "Switch", -349.8,    0.0, 1, (1,2,7)),
    ("Switch_Quantity3_06", "Switch",  349.8, -120.0, 3, (1,3,1)),
    ("Toogle_Quantity2_01", "Toggle",  233.2, -120.0, 2, (1,3,2)),
    ("Toogle_Quantity1_06", "Toggle",  116.6, -120.0, 1, (1,3,3)),
    ("Placeholder_31",      "Switch",    0.0, -120.0, 1, (1,3,4)),
    ("Switch_Quantity2_12", "Switch", -116.6, -120.0, 2, (1,3,5)),
    ("Switch_Quantity2_13", "Switch", -233.2, -120.0, 2, (1,3,6)),
    ("Switch_Quantity4_04", "Switch", -349.8, -120.0, 4, (1,3,7)),
]

# Panel backplate half-extents (m)
PANEL_HALF_X = 0.958
PANEL_HALF_Y = 0.025
PANEL_HALF_Z = 0.470

BASE_POS_Y = 30.0  # mm, placement offset from panel center


def parse_switch_xml(xml_path: Path) -> tuple[list[ET.Element], list[ET.Element], str]:
    """Parse a switch XML → (materials, geoms, body_name)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    materials = list(root.iter("material"))
    body = root.find(".//worldbody/body")
    geoms = list(body.iter("geom")) if body is not None else []
    body_name = body.get("name", "") if body is not None else ""
    return materials, geoms, body_name


def generate_layout_yaml() -> str:
    lines = [
        "# w1_b1 switch wall layout",
        "# 20 switches (various button counts), 3 rows x 7 columns",
        "# Geometry defined in objects/*.xml",
        "",
        "panel:",
        "  xml: w1_b1_panel_base.xml",
        "  attach_to_body: panel_mount",
        "",
        "output_xml: generated/w1_b1_panel_assembly.xml",
        "",
        "layout:",
        "  coordinate_scale: 0.001    # mm -> m",
        "  face_axes: [x, z]",
        f"  base_pos: [0.0, {BASE_POS_Y}, 0.0]",
        "  default_quat: [0.7071068, -0.7071068, 0.0, 0.0]",
        "  remove_root_joints: true",
        "",
        "placements:",
    ]
    for idx, (cname, stype, x, y, nbtn, wpos) in enumerate(SWITCHES, 1):
        w, r, c = wpos
        lines.append(f"  # {idx}. {cname} | {stype} | {nbtn} btn | wall {wpos}")
        lines.append(f"  - xml: objects/{cname}.xml")
        lines.append(f"    pos: [{x}, {y}]")
        lines.append(f"    row: {r}")
        lines.append(f"    col: {c}")
    return "\n".join(lines) + "\n"


def generate_panel_base_xml() -> str:
    return f"""<mujoco model="w1_b1_panel_base">
  <worldbody>
    <body name="panel_mount" pos="0 0 0">
      <geom
        name="panel_backplate"
        type="box"
        pos="0 0 0"
        size="{PANEL_HALF_X} {PANEL_HALF_Y} {PANEL_HALF_Z}"
        rgba="0.86 0.87 0.89 1"
        contype="1"
        conaffinity="1"
      />
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

    for idx, (cname, stype, x_mm, y_mm, nbtn, wpos) in enumerate(SWITCHES, 1):
        xml_path = objects_dir / f"{cname}.xml"
        if not xml_path.exists():
            print(f"  WARNING: {xml_path.name} not found, skipping #{idx}")
            continue

        materials, geoms, body_name = parse_switch_xml(xml_path)
        w, r, c = wpos
        x_m = x_mm * coord_scale
        z_m = y_mm * coord_scale
        slot_name = f"panel_slot_r{r}_c{c}"
        assembled_body = f"{body_name}_r{r}_c{c}" if body_name else f"{cname}_r{r}_c{c}"

        for mat in materials:
            asset_lines.append(f"    {ET.tostring(mat, encoding='unicode').strip()}")

        slot_lines.append(
            f'      <body name="{slot_name}" pos="{x_m} {base_y} {z_m}" quat="{quat}">'
        )
        slot_lines.append(f'        <body name="{assembled_body}">')
        for geom in geoms:
            orig = geom.get("name", "")
            if orig:
                geom.set("name", f"{orig}__{slot_name}")
            slot_lines.append(f"          {ET.tostring(geom, encoding='unicode').strip()}")
        slot_lines.append("        </body>")
        slot_lines.append("      </body>")

    return f"""<?xml version='1.0' encoding='utf-8'?>
<mujoco model="w1_b1_panel_base">
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

    layout_path = BASE_DIR / "layout_w1_b1.yaml"
    layout_path.write_text(generate_layout_yaml(), encoding="utf-8")
    print(f"  Created {layout_path.name}")

    base_path = BASE_DIR / "w1_b1_panel_base.xml"
    base_path.write_text(generate_panel_base_xml(), encoding="utf-8")
    print(f"  Created {base_path.name}")

    assembly_path = BASE_DIR / "generated" / "w1_b1_panel_assembly.xml"
    assembly_path.write_text(generate_assembly_xml(), encoding="utf-8")
    print(f"  Created generated/{assembly_path.name}")

    print(f"\nDone. Assembly built from {len(SWITCHES)} switches in objects/")


if __name__ == "__main__":
    main()
