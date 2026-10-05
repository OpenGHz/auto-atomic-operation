"""Grasp attachments on the MJWarp backend: per-world welds to the gripper."""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.basis.mjwarp.attachments import MjWarpGraspAttachments  # noqa: E402
from auto_atom.basis.mjwarp.state import (  # noqa: E402
    BATCHED_MODEL_FIELDS,
    MjWarpSceneState,
)
from auto_atom.config.env_config import GraspAttachmentConfig  # noqa: E402
from auto_atom.scene_composition import create_grasp_welds  # noqa: E402

# A gripper on three slide joints with one actuated finger, and a free cube
# beside it. Nothing on the gripper collides, so the weld alone carries the
# cube. The gripper frame site sits on a child of the joint body.
_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="gripper" pos="0 0 0.3">
      <joint name="gx" type="slide" axis="1 0 0" damping="50"/>
      <joint name="gy" type="slide" axis="0 1 0" damping="50"/>
      <joint name="gz" type="slide" axis="0 0 1" damping="50"/>
      <geom type="box" size="0.02 0.02 0.02" mass="1" contype="0" conaffinity="0"/>
      <body name="hand" pos="0 0 -0.05">
        <geom type="box" size="0.01 0.01 0.01" mass="0.1" contype="0" conaffinity="0"/>
        <site name="eef_pose"/>
        <body name="finger" pos="0 0 -0.01">
          <joint name="jf" type="slide" axis="1 0 0" range="0 0.02"/>
          <geom type="box" size="0.002 0.005 0.005" mass="0.01" contype="0"
                conaffinity="0"/>
        </body>
      </body>
    </body>
    <body name="cube" pos="0.1 0 0.025">
      <freejoint name="cube_joint"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="gx" kp="2000"/>
    <position name="ay" joint="gy" kp="2000"/>
    <position name="az" joint="gz" kp="2000"/>
    <position name="grip" joint="jf" kp="80"/>
  </actuator>
</mujoco>
"""

_BOTH = np.array([True, True])
_FIRST = np.array([True, False])
_SECOND = np.array([False, True])


@pytest.fixture(scope="module")
def host_model() -> "mujoco.MjModel":
    spec = mujoco.MjSpec.from_string(_SCENE)
    create_grasp_welds(
        spec,
        [GraspAttachmentConfig(operator="arm", object="cube").to_weld_element()],
    )
    return spec.compile()


def _state(host_model, batched=True) -> MjWarpSceneState:
    fields = BATCHED_MODEL_FIELDS + (("eq_data",) if batched else ())
    state = MjWarpSceneState(host_model, nworld=2, batched_fields=fields)
    # A different cube pose per world, so each world's weld differs.
    state.set_free_joint_pose(
        "cube_joint",
        np.array([[0.1, 0.0, 0.025], [0.12, 0.03, 0.025]]),
        np.array([[0.0, 0.0, 0.3826834, 0.9238795], [0.0, 0.0, 0.0, 1.0]]),
    )
    return state


def _attachments(state: MjWarpSceneState) -> MjWarpGraspAttachments:
    return MjWarpGraspAttachments(state, [("arm", "cube")])


def _cube_in_hand(state: MjWarpSceneState) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per world: the cube's position and rotation in the hand frame."""
    state.forward()
    hand, cube = state.body_id("hand"), state.body_id("cube")
    xpos = state.data.xpos.numpy()
    xquat = state.data.xquat.numpy()
    poses = []
    for world in range(state.nworld):
        hand_rotation, cube_rotation = np.zeros(9), np.zeros(9)
        mujoco.mju_quat2Mat(hand_rotation, xquat[world][hand].astype(np.float64))
        mujoco.mju_quat2Mat(cube_rotation, xquat[world][cube].astype(np.float64))
        hand_rotation = hand_rotation.reshape(3, 3)
        poses.append(
            (
                hand_rotation.T @ (xpos[world][cube] - xpos[world][hand]),
                hand_rotation.T @ cube_rotation.reshape(3, 3),
            )
        )
    return poses


def _cube_height(state: MjWarpSceneState) -> np.ndarray:
    return state.get_body_pose_batch("cube")[0][:, 2]


def _move_gripper(state: MjWarpSceneState, target, nstep: int = 600) -> None:
    state.set_ctrl(state.actuator_ids(("ax", "ay", "az")), np.asarray(target))
    state.step(nstep)


def _eq_active(state: MjWarpSceneState) -> np.ndarray:
    return state.data.eq_active.numpy()[:, 0].astype(bool)


# ----------------------------------------------------------------------
# Attachment store
# ----------------------------------------------------------------------


def test_each_world_holds_its_object_at_its_own_grasp_pose(host_model) -> None:
    state = _state(host_model)
    attachments = _attachments(state)
    grasp = _cube_in_hand(state)

    attachments.attach("arm", "cube", _BOTH)
    assert [attachments.attached_object("arm", w) for w in range(2)] == [
        "cube",
        "cube",
    ]
    # eq_data is batched, so each world's weld holds its own pose.
    eq_data = state.model.eq_data.numpy()
    assert not np.allclose(eq_data[0][0][3:10], eq_data[1][0][3:10])

    _move_gripper(state, [0.15, -0.1, 0.2])

    assert (_cube_height(state) > 0.2).all()
    for (position, rotation), (grasp_position, grasp_rotation) in zip(
        _cube_in_hand(state), grasp, strict=True
    ):
        np.testing.assert_allclose(position, grasp_position, atol=5e-4)
        np.testing.assert_allclose(rotation, grasp_rotation, atol=5e-3)


def test_attaching_in_one_world_leaves_the_others_free(host_model) -> None:
    state = _state(host_model)
    attachments = _attachments(state)

    attachments.attach("arm", "cube", _FIRST)
    np.testing.assert_array_equal(_eq_active(state), [True, False])
    np.testing.assert_array_equal(
        attachments.attached_bodies("arm"), [state.body_id("cube"), -1]
    )
    # A step masked to the other world keeps this world's weld.
    state.step(5, world_mask=_SECOND)
    np.testing.assert_array_equal(_eq_active(state), [True, False])

    _move_gripper(state, [0.0, 0.0, 0.2])

    height = _cube_height(state)
    assert height[0] > 0.2
    assert height[1] < 0.05


def test_release_and_reset_drop_the_weld(host_model) -> None:
    state = _state(host_model)
    attachments = _attachments(state)
    attachments.attach("arm", "cube", _BOTH)

    assert attachments.release("arm", _FIRST) == ["cube", None]
    np.testing.assert_array_equal(_eq_active(state), [False, True])
    assert attachments.attached_object("arm", 0) is None
    # Releasing again is a no-op.
    assert attachments.release("arm", _FIRST) == [None, None]

    attachments.reset(_SECOND)
    np.testing.assert_array_equal(_eq_active(state), [False, False])
    assert attachments.attached_object("arm", 1) is None

    # A released world drops its object.
    attachments.attach("arm", "cube", _SECOND)
    attachments.release("arm", _SECOND)
    _move_gripper(state, [0.0, 0.0, 0.2])
    assert (_cube_height(state) < 0.05).all()


def test_attachments_need_eq_data_batched_per_world(host_model) -> None:
    state = _state(host_model, batched=False)
    with pytest.raises(ValueError, match="eq_data batched per world"):
        _attachments(state)


def test_missing_or_undeclared_welds_are_errors(host_model) -> None:
    state = _state(host_model)
    with pytest.raises(ValueError, match="aao_grasp_attach__arm__plate"):
        MjWarpGraspAttachments(state, [("arm", "plate")])
    with pytest.raises(KeyError, match="No grasp attachment weld"):
        _attachments(state).attach("arm", "plate", _BOTH)
