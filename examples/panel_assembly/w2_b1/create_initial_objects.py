"""One-time script: create the initial switch XMLs for w2_b1 in objects/.

Each switch is 8.6cm x 8.6cm square, 7mm thick, default black metallic.
"""

from __future__ import annotations
from pathlib import Path

OBJECTS_DIR = Path(__file__).parent / "objects"

# Geometry (metres)
BTN_HALF = 0.043
BTN_FACE_HALF = 0.040
BEZEL_HALF_Z = 0.001
BTN_HALF_Z = 0.0035

# Default black metallic
BEZEL_RGBA = "0.05 0.05 0.07 1"
BTN_RGBA = "0.10 0.10 0.13 1"
SPECULAR = "0.7"
SHININESS = "0.8"

COLORS = {
    "white": {
        "bezel": "0.85 0.85 0.88 1",
        "button": "0.95 0.95 0.97 1",
        "specular": "0.85",
        "shininess": "0.9",
    },
    "dark_gray": {
        "bezel": "0.15 0.15 0.18 1",
        "button": "0.25 0.25 0.28 1",
        "specular": "0.75",
        "shininess": "0.85",
    },
    "gold": {
        "bezel": "0.65 0.53 0.15 1",
        "button": "0.83 0.69 0.22 1",
        "specular": "0.9",
        "shininess": "0.95",
    },
    "pink": {
        "bezel": "0.80 0.50 0.58 1",
        "button": "0.95 0.65 0.72 1",
        "specular": "0.8",
        "shininess": "0.9",
    },
    "placeholder": {
        "bezel": "0.30 0.30 0.33 1",
        "button": "0.60 0.60 0.63 1",
        "specular": "0.5",
        "shininess": "0.5",
    },
}


# 14 switches with known coordinates from Excel w2_b1
# (class_name, x_mm, y_mm, wall_pos)
SWITCHES = [
    ("Knob_Quantity1_02",    -286.5,  160.0, (2,1,10)),
    ("Knob_Quantity1_01",    -356.5,  160.0, (2,1,11)),
    ("Toogle_Quantity1_01",   217.5,   50.0, (2,2,2)),
    ("Toogle_Quantity1_02",   111.5,   50.0, (2,2,3)),
    ("Toogle_Quantity1_03",  -100.5,   50.0, (2,2,5)),
    ("Toogle_Quantity3_01",  -206.5,   50.0, (2,2,6)),
    ("Switch_Quantity3_03",  -312.5,   50.0, (2,2,7)),
    ("Switch_Quantity1_01",   323.5,  -80.0, (2,3,1)),
    ("Switch_Quantity4_01",   217.5,  -80.0, (2,3,2)),
    ("Switch_Quantity1_03",   111.5,  -80.0, (2,3,3)),
    ("Switch_Quantity2_02",     5.5,  -80.0, (2,3,4)),
    ("Switch_Quantity3_02",  -100.5,  -80.0, (2,3,5)),
    ("Switch_Quantity1_05",  -206.5,  -80.0, (2,3,6)),
    ("Switch_Quantity1_04",  -312.5,  -80.0, (2,3,7)),
]


def generate_switch_xml(class_name: str, color: str) -> str:
    c = COLORS[color]
    sp, sh = c["specular"], c["shininess"]
    return f"""<mujoco model="{class_name}">
  <asset>
    <material name="{class_name}_bezel" rgba="{c['bezel']}" specular="{sp}" shininess="{sh}"/>
    <material name="{class_name}_button" rgba="{c['button']}" specular="{sp}" shininess="{sh}"/>
  </asset>
  <worldbody>
    <body name="{class_name}">
      <geom name="{class_name}_bezel" type="box"
            pos="0 0 {BEZEL_HALF_Z}" size="{BTN_HALF} {BTN_HALF} {BEZEL_HALF_Z}"
            material="{class_name}_bezel"/>
      <geom name="{class_name}_button" type="box"
            pos="0 0 {BTN_HALF_Z}" size="{BTN_FACE_HALF} {BTN_FACE_HALF} {BTN_HALF_Z}"
            material="{class_name}_button"/>
    </body>
  </worldbody>
</mujoco>"""


def main() -> None:
    OBJECTS_DIR.mkdir(exist_ok=True)
    for class_name, x, y, color in SWITCHES:
        path = OBJECTS_DIR / f"{class_name}.xml"
        path.write_text(generate_switch_xml(class_name, color), encoding="utf-8")
        print(f"  {path.name}  ({color})")
    print(f"\nCreated {len(SWITCHES)} switch XMLs in {OBJECTS_DIR}")


if __name__ == "__main__":
    main()