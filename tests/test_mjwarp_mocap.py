"""Physical mocap control and home-state equivalence on real MJWarp worlds."""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjwarp.arm_control import MjWarpArmControl  # noqa: E402
from auto_atom.backend.mjwarp.operator_handler import MjWarpOperatorHandler  # noqa: E402
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv  # noqa: E402
from auto_atom.config.env_config import EnvConfig  # noqa: E402
from auto_atom.execution_model import ControlSignal  # noqa: E402
from auto_atom.utils.pose import PoseState  # noqa: E402

_XML = """
<mujoco>
  <option timestep="0.004" gravity="0 0 0"/>
  <worldbody>
    <body name="target" mocap="true"/>
    <body name="root" pos="0.1 0.2 0.4">
      <freejoint name="root_free"/>
      <geom type="box" size="0.02 0.03 0.04" mass="1"/>
      <site name="tool" pos="0.04 0.02 -0.03" quat="0.92387953 0 0 0.38268343"/>
    </body>
  </worldbody>
  <equality><weld body1="target" body2="root" solref="0.01 1"/></equality>
</mujoco>
"""


@pytest.fixture(params=[False, True], ids=["mocap-first", "mocap-second"])
def handler(tmp_path, request):
    scene_path = tmp_path / "scene.xml"
    xml = (
        _XML.replace(
            'body1="target" body2="root"',
            'body1="root" body2="target" anchor="0.1 0.2 0.4"',
        )
        if request.param
        else _XML
    )
    scene_path.write_text(xml)
    config = EnvConfig.model_validate(
        {
            "scene": {"base": str(scene_path)},
            "batch_size": 2,
            "sim_freq": 1000,
            "update_freq": 50,
            "initial_joint_positions": {"root_free": [0.2, -0.1, 0.5, 1, 0, 0, 0]},
            "operators": {
                "arm": {
                    "name": "arm",
                    "root_body": "root",
                    "pose_site": "tool",
                    "mocap_body": "target",
                    "freejoint": "root_free",
                }
            },
        }
    )
    env = MjWarpObjectOnlyEnv(config)
    operator = env.get_operator_state("arm")
    arm = MjWarpArmControl(
        state=env.state,
        operator=operator,
        ik=None,
        n_substeps=env.n_substeps,
        position_tolerance=0.001,
        orientation_tolerance=0.01,
    )
    yield MjWarpOperatorHandler(state=env.state, operator=operator, arm=arm), env
    env.close()


def test_mocap_home_uses_configured_free_joint_and_virtual_base(handler):
    control, _ = handler
    np.testing.assert_allclose(control.get_base_pose().position, 0)
    np.testing.assert_allclose(
        control.state.get_body_pose_batch("root")[0],
        [[0.2, -0.1, 0.5]] * 2,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        control.state.get_mocap_pose("target")[0],
        [[0.1, -0.3, 0.1]] * 2,
        atol=1e-7,
    )


def test_mocap_reaches_rotated_tool_goal_without_moving_inactive_world(handler):
    control, _ = handler
    before = control.get_end_effector_pose()
    initial_qpos = control.state.data.qpos.numpy().copy()
    target_position = np.array([0.4, 0.2, 0.6])
    target_orientation = np.array([0.0, 0.0, 0.70710678, 0.70710678])
    mask = np.array([True, False])
    for _ in range(80):
        result = control.arm.move(target_position, target_orientation, world_mask=mask)
        if result.signals[0] == ControlSignal.REACHED:
            break
    assert result.signals[0] == ControlSignal.REACHED
    np.testing.assert_allclose(
        control.get_end_effector_pose().position[0], target_position, atol=0.001
    )
    np.testing.assert_array_equal(control.state.data.qpos.numpy()[1], initial_qpos[1])
    np.testing.assert_allclose(
        control.get_end_effector_pose().position[1], before.position[1]
    )


def test_mocap_home_and_masked_reset_restore_physical_and_target_poses(handler):
    control, env = handler
    original = control.get_end_effector_pose()
    target = PoseState(
        position=[[0.3, 0.1, 0.7], [0.4, -0.2, 0.6]],
        orientation=[[0, 0, 0, 1], [0, 0, 0, 1]],
    )
    control.set_home_end_effector_pose(target)
    np.testing.assert_allclose(
        control.get_end_effector_pose().position, target.position, atol=1e-6
    )
    env.reset(np.array([True, False]))
    np.testing.assert_allclose(
        control.get_end_effector_pose().position[0], original.position[0], atol=1e-6
    )
    np.testing.assert_allclose(
        control.get_end_effector_pose().position[1], target.position[1], atol=1e-6
    )
