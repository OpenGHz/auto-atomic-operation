from __future__ import annotations

import mujoco
import pytest
from omegaconf import OmegaConf

from examples import view_scene


def test_build_applies_sim_freq_and_holds_position_actuators(monkeypatch) -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.005" gravity="0 0 0"/>
          <worldbody>
            <body>
              <joint name="position_joint" type="hinge"/>
              <joint name="motor_joint" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
          <actuator>
            <position name="position_actuator" joint="position_joint"
                      gear="2" kp="100"/>
            <motor name="motor_actuator" joint="motor_joint"/>
            <intvelocity name="velocity_actuator" joint="motor_joint"
                         actrange="-1 1" ctrlrange="-1 1"/>
            <general name="affine_actuator" joint="motor_joint"
                     gainprm="7" biasprm="0 -8 -9"/>
          </actuator>
        </mujoco>
        """
    )
    monkeypatch.setattr(view_scene, "load_composed_scene", lambda _config: model)

    cfg = OmegaConf.create(
        {
            "env": {
                "scene": {"base": "unused.xml"},
                "sim_freq": 1250,
                "initial_joint_positions": {
                    "position_joint": 0.4,
                    "motor_joint": 0.3,
                },
                "operators": {
                    "arm": {
                        "arm_actuators": [
                            "position_actuator",
                            "motor_actuator",
                            "velocity_actuator",
                            "affine_actuator",
                        ],
                        "eef_actuators": [],
                        "root_body": "",
                    }
                },
            },
            "task": {},
            "task_operators": {},
        }
    )

    overrides = view_scene._extract_overrides(cfg)
    built_model, data = view_scene._build(overrides)

    assert overrides["sim_freq"] == pytest.approx(1250.0)
    assert overrides["actuator_names"] == [
        "position_actuator",
        "motor_actuator",
        "velocity_actuator",
        "affine_actuator",
    ]
    assert built_model.opt.timestep == pytest.approx(0.0008)
    assert data.qpos[0] == pytest.approx(0.4)
    assert data.qpos[1] == pytest.approx(0.3)
    assert data.actuator_length[0] == pytest.approx(0.8)
    assert data.ctrl[0] == pytest.approx(0.8)
    assert data.ctrl[1] == pytest.approx(0.0)
    assert data.ctrl[2] == pytest.approx(0.0)
    assert data.ctrl[3] == pytest.approx(0.0)
