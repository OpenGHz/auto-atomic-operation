"""Construction-time forwards must not generate contacts at ``qpos0``."""

from __future__ import annotations

from pathlib import Path

import mujoco

from auto_atom.basis.mjc.model_initialization import forward_without_contacts
from auto_atom.config_loader import load_task_file_hydra
from auto_atom.runtime import ComponentRegistry

ROOT = Path(__file__).resolve().parents[1]

_OVERLAPPING_BOXES = """
<mujoco>
  <worldbody>
    <geom type="box" size="0.1 0.1 0.1"/>
    <body>
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_forward_without_contacts_skips_contacts_and_restores_flags() -> None:
    model = mujoco.MjModel.from_xml_string(_OVERLAPPING_BOXES)
    data = mujoco.MjData(model)
    flags = model.opt.disableflags

    forward_without_contacts(model, data)

    assert data.ncon == 0
    assert model.opt.disableflags == flags
    mujoco.mj_forward(model, data)
    assert data.ncon > 0


def test_operator_buried_at_qpos0_constructs_at_home_pose() -> None:
    # At qpos0 the Robotiq freejoint sits at the world origin, inside the
    # hang_toothbrush_cup table, cup and floor. Generating those contacts
    # overflowed MuJoCo's constraint arena and crashed the process.
    ComponentRegistry.clear()
    try:
        task_file = load_task_file_hydra(
            "hang_toothbrush_cup",
            ROOT / "aao_configs",
            ["env.batch_size=1", "env.enabled_sensors=[pose,tactile]"],
        )
        env = ComponentRegistry.get_env(task_file.task.env_name)
        basis = env.envs[0] if hasattr(env, "envs") else env
        freejoint = mujoco.mj_name2id(
            basis.model, mujoco.mjtObj.mjOBJ_JOINT, "robotiq_freejoint"
        )
        address = basis.model.jnt_qposadr[freejoint]

        assert basis.data.qpos[address + 2] == 0.4
    finally:
        ComponentRegistry.clear()
