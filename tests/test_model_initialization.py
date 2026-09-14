"""Home linkage initialization is independent of temporary scene intersections."""

import mujoco
import numpy as np
import pytest

from auto_atom.basis.mjc.model_initialization import apply_initial_joint_positions


def _model(floor_height):
    return mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep="0.001"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 .1" pos="0 0 {floor_height}"/>
        <body pos="0 0 .03">
          <joint name="drive" axis="0 1 0"/>
          <geom type="box" size=".1 .02 .05" pos=".1 0 0"/>
        </body>
        <body pos=".4 0 .03">
          <joint name="passive" axis="0 1 0"/>
          <geom type="box" size=".1 .02 .05" pos=".1 0 0"/>
        </body>
      </worldbody>
      <equality><joint joint1="drive" joint2="passive"/></equality>
      <actuator><position name="servo" joint="drive" kp="100"/></actuator>
    </mujoco>
    """)


def test_home_linkage_matches_clear_scene_and_restores_contacts():
    clear = _model(-1)
    intersecting = _model(0)
    states = []
    for model in (clear, intersecting):
        data = mujoco.MjData(model)
        gravity = model.opt.gravity.copy()
        flags = model.opt.disableflags
        apply_initial_joint_positions(model, data, {"drive": 0.0}, [0])
        np.testing.assert_array_equal(model.opt.gravity, gravity)
        assert model.opt.disableflags == flags
        states.append(data)
    np.testing.assert_allclose(states[0].qpos, states[1].qpos, atol=1e-7)
    assert states[1].ncon > 0, (
        "The real scene contacts must be observable after initialization."
    )


def test_failed_home_solve_restores_physics_options(monkeypatch):
    model = _model(0)
    data = mujoco.MjData(model)
    gravity = model.opt.gravity.copy()
    flags = model.opt.disableflags

    def fail_step(model, data):
        raise RuntimeError("test interruption")

    monkeypatch.setattr(mujoco, "mj_step", fail_step)
    with pytest.raises(RuntimeError, match="test interruption"):
        apply_initial_joint_positions(model, data, {"drive": 0.0}, [0])
    np.testing.assert_array_equal(model.opt.gravity, gravity)
    assert model.opt.disableflags == flags
