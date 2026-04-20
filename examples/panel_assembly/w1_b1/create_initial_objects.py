"""One-time script: create the 20 initial switch XMLs in objects/.

Each switch is a simple 8.6cm x 8.6cm square, 7mm thick.
Colours are placeholder grey — edit objects/*.xml to set final colours.
"""

from __future__ import annotations
from pathlib import Path

OBJECTS_DIR = Path(__file__).parent / "objects"

# Geometry (metres)
BTN_HALF = 0.043        # 43mm half-width  (86mm full)
BTN_FACE_HALF = 0.040   # 40mm half (80mm inner face)
BEZEL_HALF_Z = 0.001    # bezel base half-thickness
BTN_HALF_Z = 0.0035     # button half-thickness (7mm total)

# Metallic colours: (bezel_rgba, button_rgba, specular, shininess)
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

# 20 switches: (class_name, x_mm, y_mm, color)
# 白色: 1-2,4-5,10-11,13-14,18  深灰: 3,7-9,12,15-16  金色: 6,20  粉色: 19
SWITCHES = [
    ("Switch_Quantity2_07",  349.8,  120.0, "white"),      # 1
    ("Switch_Quantity2_08",  233.2,  120.0, "white"),      # 2
    ("Switch_Quantity2_09",  116.6,  120.0, "dark_gray"),  # 3
    ("Switch_Quantity2_10",    0.0,  120.0, "white"),      # 4
    ("Switch_Quantity4_02", -116.6,  120.0, "white"),      # 5
    ("Switch_Quantity1_45", -233.2,  120.0, "gold"),       # 6
    ("Switch_Quantity1_46", -349.8,  120.0, "dark_gray"),  # 7
    ("Toogle_Quantity1_05",  349.8,    0.0, "dark_gray"),  # 8
    ("Switch_Quantity4_03",  233.2,    0.0, "dark_gray"),  # 9
    ("Switch_Quantity1_47",  116.6,    0.0, "white"),      # 10
    ("Switch_Quantity2_11", -116.6,    0.0, "white"),      # 11
    ("Switch_Quantity1_48", -233.2,    0.0, "dark_gray"),  # 12
    ("Switch_Quantity1_49", -349.8,    0.0, "white"),      # 13
    ("Switch_Quantity3_06",  349.8, -120.0, "white"),      # 14
    ("Toogle_Quantity2_01",  233.2, -120.0, "dark_gray"),  # 15
    ("Toogle_Quantity1_06",  116.6, -120.0, "dark_gray"),  # 16
    ("Placeholder_31",         0.0, -120.0, "placeholder"),# 17
    ("Switch_Quantity2_12", -116.6, -120.0, "white"),      # 18
    ("Switch_Quantity2_13", -233.2, -120.0, "pink"),       # 19
    ("Switch_Quantity4_04", -349.8, -120.0, "gold"),       # 20
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
