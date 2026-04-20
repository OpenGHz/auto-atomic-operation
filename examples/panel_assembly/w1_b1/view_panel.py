"""View the w1_b1 panel assembly in MuJoCo interactive viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer

XML_PATH = Path(__file__).parent / "generated" / "w1_b1_panel_assembly.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
