from __future__ import annotations

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from examples import view_scene
from auto_atom.framework import PoseOverrideConfig


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


def test_build_resolves_object_then_operator_base_named_reference(monkeypatch) -> None:
    """Object placement is visible when an operator base uses its site frame."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="anchor" pos="1 0 0">
              <site name="anchor_site" pos="0.5 0 0"/>
            </body>
            <body name="arm_root" pos="0 2 0">
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    monkeypatch.setattr(view_scene, "load_composed_scene", lambda _config: model)

    overrides = {
        "scene": {"base": "unused.xml"},
        "sim_freq": None,
        "actuator_names": [],
        "ijp": {},
        "initial_pose": {
            "anchor": PoseOverrideConfig(position=[2.0, 0.0, 0.0]),
        },
        "op_bases": [
            (
                "arm_root",
                PoseOverrideConfig(
                    reference="anchor_site",
                    position=[0.0, 1.0, 0.0],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                ),
            )
        ],
        "operator_frames": {},
    }

    built_model, data = view_scene._build(overrides)
    anchor_site = view_scene._element_pose(built_model, data, "anchor_site")
    arm_root = view_scene._element_pose(built_model, data, "arm_root")

    # anchor's initial pose moves its site from x=1.5 to x=2.5.  The base
    # override is then interpreted in that updated frame.
    assert anchor_site.position[0, 0] == pytest.approx(2.5)
    assert arm_root.position[0, 0] == pytest.approx(2.5)
    assert arm_root.position[0, 1] == pytest.approx(1.0)


def test_build_applies_initial_pose_dependencies_in_topological_order(
    monkeypatch,
) -> None:
    """Viewer and runtime must resolve dependent object frames identically."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="first" pos="0 0 0">
              <geom type="sphere" size="0.01"/>
            </body>
            <body name="second" pos="0 0 0">
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    monkeypatch.setattr(view_scene, "load_composed_scene", lambda _config: model)

    # ``first`` is declared before its ``second`` reference on purpose.
    overrides = {
        "scene": {"base": "unused.xml"},
        "sim_freq": None,
        "actuator_names": [],
        "ijp": {},
        "initial_pose": {
            "first": PoseOverrideConfig(reference="second", position=[1.0, 0.0, 0.0]),
            "second": PoseOverrideConfig(position=[2.0, 0.0, 0.0]),
        },
        "op_bases": [],
        "operator_frames": {},
    }

    built_model, data = view_scene._build(overrides)
    first = view_scene._element_pose(built_model, data, "first")
    second = view_scene._element_pose(built_model, data, "second")
    np.testing.assert_allclose(second.position[0], [2.0, 0.0, 0.0])
    np.testing.assert_allclose(first.position[0], [3.0, 0.0, 0.0])


def test_build_applies_mounted_camera_initial_pose(monkeypatch) -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="mount" pos="1 2 0" quat="0.7071067812 0 0 0.7071067812">
              <camera name="mounted_cam" pos="1 0 0"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    monkeypatch.setattr(view_scene, "load_composed_scene", lambda _config: model)

    overrides = {
        "scene": {"base": "unused.xml"},
        "sim_freq": None,
        "actuator_names": [],
        "ijp": {},
        "initial_pose": {},
        "camera_initial_pose": {
            "mounted_cam": PoseOverrideConfig(
                position=[4.0, 5.0, 6.0],
                orientation=[0.0, 0.0, 0.0, 1.0],
            )
        },
        "op_bases": [],
        "operator_frames": {},
    }

    built_model, data = view_scene._build(overrides)
    camera_id = mujoco.mj_name2id(
        built_model, mujoco.mjtObj.mjOBJ_CAMERA, "mounted_cam"
    )
    np.testing.assert_allclose(data.cam_xpos[camera_id], [4.0, 5.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(
        view_scene.quaternion_from_matrix_3x3(data.cam_xmat[camera_id].reshape(3, 3)),
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-6,
    )


def test_build_applies_operator_base_after_freejoint_home(monkeypatch) -> None:
    """A base override must win over a mocap root's configured home qpos."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="mocap_root">
              <freejoint name="mocap_root_freejoint"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    monkeypatch.setattr(view_scene, "load_composed_scene", lambda _config: model)

    overrides = {
        "scene": {"base": "unused.xml"},
        "sim_freq": None,
        "actuator_names": [],
        "ijp": {
            "mocap_root_freejoint": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        },
        "initial_pose": {},
        "op_bases": [
            (
                "mocap_root",
                PoseOverrideConfig(
                    position=[4.0, 5.0, 6.0],
                    orientation=[0.0, 0.0, 0.0, 1.0],
                ),
            )
        ],
        "operator_frames": {},
    }

    built_model, data = view_scene._build(overrides)
    root_pose = view_scene._element_pose(built_model, data, "mocap_root")

    assert root_pose.position[0].tolist() == pytest.approx([4.0, 5.0, 6.0])


def test_extract_overrides_warns_when_eef_pose_cannot_be_applied(capsys) -> None:
    cfg = OmegaConf.create(
        {
            "env": {
                "scene": {"base": "unused.xml"},
                "operators": {
                    "arm": {
                        "root_body": "arm_root",
                        "pose_site": "eef_pose",
                    }
                },
            },
            "task": {},
            "task_operators": {
                "arm": {
                    "initial_state": {
                        "eef_pose": {
                            "position": [0.2, 0.0, 0.1],
                            "reference": "base",
                        }
                    }
                }
            },
        }
    )

    overrides = view_scene._extract_overrides(cfg)

    captured = capsys.readouterr()
    assert "eef_pose is configured but view_scene cannot apply" in captured.out
    assert len(overrides["op_eef_poses"]) == 1


def test_element_pose_for_joint_tracks_articulated_anchor() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hinge_body" pos="1 2 0">
              <joint name="hinge" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")

    initial = view_scene._element_pose(model, data, "hinge")
    data.qpos[model.jnt_qposadr[joint_id]] = 0.4
    mujoco.mj_forward(model, data)
    moved = view_scene._element_pose(model, data, "hinge")

    np.testing.assert_allclose(initial.position[0], [1.0, 2.0, 0.0])
    np.testing.assert_allclose(moved.position[0], [1.0, 2.0, 0.0])
    assert abs(float(np.dot(initial.orientation[0], moved.orientation[0]))) < 0.99
