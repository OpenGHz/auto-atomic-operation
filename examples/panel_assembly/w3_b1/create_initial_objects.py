"""One-time script: create the initial switch XMLs for w3_b1 in objects/.

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

# 26 switches from Excel w3_b1
# (class_name, x_mm, y_mm)
# Row 1 (y=120): 13 switches; positions 1-5,10-13 have no class_name in Excel
SWITCHES = [
    ("w3b1_Switch_01",       349.8,  120.0),   #  1  (3,1,1)
    ("w3b1_Switch_02",       293.8,  120.0),   #  2  (3,1,2)
    ("w3b1_Switch_03",       237.8,  120.0),   #  3  (3,1,3)
    ("w3b1_Switch_04",       181.8,  120.0),   #  4  (3,1,4)
    ("w3b1_Switch_05",       125.8,  120.0),   #  5  (3,1,5)
    ("Switch_Quantity1_36",   69.8,  120.0),   #  6  (3,1,6)
    ("Switch_Quantity1_37",    9.8,  120.0),   #  7  (3,1,7)
    ("Switch_Quantity1_38",  -50.2,  120.0),   #  8  (3,1,8)
    ("Switch_Quantity1_39", -110.2,  120.0),   #  9  (3,1,9)
    ("w3b1_Switch_10",      -170.2,  120.0),   # 10  (3,1,10)
    ("w3b1_Switch_11",      -230.2,  120.0),   # 11  (3,1,11)
    ("w3b1_Switch_12",      -290.2,  120.0),   # 12  (3,1,12)
    ("w3b1_Switch_13",      -349.8,  120.0),   # 13  (3,1,13)
    # Row 2 (y=0): 6 switches (no position at col 4)
    ("Switch_Quantity3_04",  349.8,    0.0),   # 14  (3,2,1)
    ("Switch_Quantity3_05",  233.2,    0.0),   # 15  (3,2,2)
    ("Switch_Quantity2_03",  116.6,    0.0),   # 16  (3,2,3)
    ("Switch_Quantity1_40", -116.6,    0.0),   # 17  (3,2,5)
    ("Switch_Quantity1_41", -233.2,    0.0),   # 18  (3,2,6)
    ("Switch_Quantity1_42", -349.8,    0.0),   # 19  (3,2,7)
    # Row 3 (y=-120): 7 switches
    ("Switch_Quantity2_04",  349.8, -120.0),   # 20  (3,3,1)
    ("Switch_Quantity2_05",  233.2, -120.0),   # 21  (3,3,2)
    ("Toogle_Quantity4_01",  116.6, -120.0),   # 22  (3,3,3)
    ("Switch_Quantity2_06",    0.0, -120.0),   # 23  (3,3,4)
    ("Switch_Quantity1_43", -116.6, -120.0),   # 24  (3,3,5)
    ("Switch_Quantity1_44", -233.2, -120.0),   # 25  (3,3,6)
    ("Toogle_Quantity1_04", -349.8, -120.0),   # 26  (3,3,7)
]


def generate_switch_xml(class_name: str) -> str:
    return f"""<mujoco model="{class_name}">
  <asset>
    <material name="{class_name}_bezel" rgba="{BEZEL_RGBA}" specular="{SPECULAR}" shininess="{SHININESS}"/>
    <material name="{class_name}_button" rgba="{BTN_RGBA}" specular="{SPECULAR}" shininess="{SHININESS}"/>
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
    for class_name, x, y in SWITCHES:
        path = OBJECTS_DIR / f"{class_name}.xml"
        path.write_text(generate_switch_xml(class_name), encoding="utf-8")
        print(f"  {path.name}")
    print(f"\nCreated {len(SWITCHES)} switch XMLs in {OBJECTS_DIR}")


if __name__ == "__main__":
    main()
