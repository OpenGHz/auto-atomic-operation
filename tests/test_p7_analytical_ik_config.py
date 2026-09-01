from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from auto_atom.backend.mjc.ik.third_party_ik.p7_arm_v3_analytical_ik import (
    KDL_7DOF,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "aao_configs"


def _body_or_site_transform(
    data: mujoco.MjData, element_id: int, *, site: bool
) -> np.ndarray:
    transform = np.eye(4)
    if site:
        transform[:3, :3] = data.site_xmat[element_id].reshape(3, 3)
        transform[:3, 3] = data.site_xpos[element_id]
    else:
        transform[:3, :3] = data.xmat[element_id].reshape(3, 3)
        transform[:3, 3] = data.xpos[element_id]
    return transform


def test_v4_demo_incrementally_overrides_robot_and_kinematics() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        v3 = compose(config_name="open_door_unidoor_p7_v3_umi_v3")
        v4 = compose(config_name="open_door_unidoor_p7_v4_umi_v3")

    assert v4.task_name == "open_door_unidoor_p7_v4_umi_v3"
    assert v4.env.scene.layers[-1].path.endswith("p7_arm_v4_with_umi_gripper_v3.xml")
    v3_task = OmegaConf.to_container(v3.task, resolve=True)
    v4_task = OmegaConf.to_container(v4.task, resolve=True)
    # ``env_name`` is derived from each task name and intentionally changes so
    # the two instantiated environments can coexist in one process.
    assert v4_task.pop("env_name") != v3_task.pop("env_name")
    assert v4_task == v3_task
    assert OmegaConf.to_container(
        v4.task_operators, resolve=True
    ) == OmegaConf.to_container(v3.task_operators, resolve=True)

    kinematics = OmegaConf.to_container(
        v4.env.operators.arm.ik_params.kinematics, resolve=True
    )
    assert kinematics["d"][1] == pytest.approx(0.0662)
    assert kinematics["d"][5] == pytest.approx(0.32905)
    assert kinematics["joint_limits"][1] == pytest.approx([-2.8798, 0.83775])
    assert kinematics["joint_limits"][3] == pytest.approx([-2.5307, 0.08726])


def test_v4_configured_fk_matches_mujoco_flange() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        config = compose(config_name="open_door_unidoor_p7_v4_umi_v3")

    robot_path = ROOT / str(config.operator_robot_xml)
    model = mujoco.MjModel.from_xml_path(str(robot_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "p7_mount")
    flange_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tool_site")
    flange_in_root = np.linalg.inv(
        _body_or_site_transform(data, root_id, site=False)
    ) @ _body_or_site_transform(data, flange_id, site=True)

    params = OmegaConf.to_container(
        config.env.operators.arm.ik_params.kinematics, resolve=True
    )
    analytical_flange = KDL_7DOF(params).fk([0.0] * 7, use_tcp=False)
    np.testing.assert_allclose(analytical_flange, flange_in_root, atol=1e-5)


def test_kinematic_overrides_are_validated() -> None:
    default_solver = KDL_7DOF()
    np.testing.assert_allclose(
        default_solver.d,
        [0.0, 0.16452, 0.0, 0.249, 0.0, 0.329, 0.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        default_solver.JOINT_LIMITS[1], [-150 * np.pi / 180, 50 * np.pi / 180]
    )

    with pytest.raises(ValueError, match="joint_limits must be a finite 7x2 array"):
        KDL_7DOF({"joint_limits": [[-1.0, 1.0]]})

    with pytest.raises(ValueError, match="d must be a finite vector with 9 values"):
        KDL_7DOF({"d": [0.0] * 8})
