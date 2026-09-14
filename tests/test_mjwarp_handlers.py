"""Contract tests for the MJWarp object handler.

``SceneBackend.apply_object_pose`` routes through ``ObjectHandler``, so this is
the whole transport mechanism for ``execution.mode: object_only``. The tests
therefore check the handler against the *contract* (the ABC, and the native
handler's observable behaviour) rather than against its own implementation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from auto_atom.backend.mjc.mujoco_backend import MujocoObjectHandler  # noqa: E402
from auto_atom.backend.mjwarp.handlers import MjWarpObjectHandler  # noqa: E402
from auto_atom.basis.mjwarp.state import MjWarpSceneState  # noqa: E402
from auto_atom.contracts import ObjectHandler  # noqa: E402
from auto_atom.utils.pose import PoseState  # noqa: E402

_SCENE_XML = """
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05"/>
    <body name="holder" pos="0.1 -0.2 0.3" quat="0.9238795 0 0.3826834 0">
      <geom name="holder_geom" type="box" size="0.05 0.04 0.03"/>
      <body name="nested_static" pos="0.06 0.01 0.02" quat="0.7071068 0 0 0.7071068">
        <geom name="nested_geom" type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
    <body name="mover" pos="-0.4 0.15 0.5">
      <freejoint name="mover_free"/>
      <geom name="mover_geom" type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def host_model():
    return mujoco.MjModel.from_xml_string(_SCENE_XML)


@pytest.fixture
def handler(host_model):
    state = MjWarpSceneState(host_model, nworld=2)
    return MjWarpObjectHandler(name="object", state=state, body_name="mover")


def test_satisfies_object_handler_contract(handler):
    assert isinstance(handler, ObjectHandler)
    assert handler.name == "object"
    assert handler.body_name == "mover"


def test_get_pose_returns_one_row_per_world(handler):
    pose = handler.get_pose()

    assert pose.batch_size == 2
    assert np.asarray(pose.position).shape == (2, 3)
    assert np.asarray(pose.orientation).shape == (2, 4)


def test_set_pose_gives_each_world_its_own_sample(handler):
    """A full batch is what randomization produces: one sample per env."""
    target = PoseState(
        position=np.array([[0.10, 0.20, 0.30], [-0.40, 0.50, 0.60]]),
        orientation=np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (2, 1)),
    )

    handler.set_pose(target)
    got = handler.get_pose()

    np.testing.assert_allclose(got.position, target.position, atol=1e-6)
    assert not np.allclose(got.position[0], got.position[1], atol=1e-6)


def test_set_pose_broadcasts_a_batch_of_one(handler):
    handler.set_pose(PoseState(position=np.array([[0.7, 0.8, 0.9]])))
    got = handler.get_pose()

    np.testing.assert_allclose(got.position[0], [0.7, 0.8, 0.9], atol=1e-6)
    np.testing.assert_allclose(got.position[1], [0.7, 0.8, 0.9], atol=1e-6)


def test_set_pose_honours_env_mask(handler):
    handler.set_pose(PoseState(position=np.array([[0.7, 0.8, 0.9]])))
    handler.set_pose(
        PoseState(position=np.array([[0.0, 0.0, 1.5]])),
        env_mask=np.array([True, False]),
    )
    got = handler.get_pose()

    np.testing.assert_allclose(got.position[0], [0.0, 0.0, 1.5], atol=1e-6)
    np.testing.assert_allclose(got.position[1], [0.7, 0.8, 0.9], atol=1e-6)


def test_empty_mask_is_a_no_op(handler):
    before = handler.get_pose()

    handler.set_pose(
        PoseState(position=np.array([[9.0, 9.0, 9.0]])),
        env_mask=np.array([False, False]),
    )
    after = handler.get_pose()

    np.testing.assert_allclose(after.position, before.position, atol=1e-9)


def test_mask_rejection_message_matches_the_native_handler(handler):
    """Both backends reject a wrong-shaped mask identically.

    A contract test pins the native message, so asserting the two against the
    same pattern keeps a caller that keys on it working across backends.
    """
    native = MujocoObjectHandler(
        name="object",
        env=SimpleNamespace(batch_size=2, envs=[]),
        body_name="mover",
    )
    pattern = r"env_mask must have shape \(2,\)"

    with pytest.raises(ValueError, match=pattern):
        native.set_pose(PoseState(), env_mask=np.asarray([True]))
    with pytest.raises(ValueError, match=pattern):
        handler.set_pose(PoseState(), env_mask=np.asarray([True]))


def test_static_body_transports_through_the_same_entry_point(host_model):
    """A body with no free joint is placed by the same handler API."""
    state = MjWarpSceneState(host_model, nworld=2)
    static_handler = MjWarpObjectHandler(
        name="rack", state=state, body_name="nested_static"
    )
    target = np.array([[0.42, -0.13, 0.37]])

    static_handler.set_pose(PoseState(position=target))
    got = static_handler.get_pose()

    for world in range(2):
        np.testing.assert_allclose(got.position[world], target[0], atol=1e-6)


def test_explicit_freejoint_override_is_used(host_model):
    state = MjWarpSceneState(host_model, nworld=1)
    overridden = MjWarpObjectHandler(
        name="object",
        state=state,
        body_name="mover",
        freejoint_name="mover_free",
    )

    overridden.set_pose(PoseState(position=np.array([[0.25, 0.35, 0.45]])))

    np.testing.assert_allclose(
        overridden.get_pose().position[0], [0.25, 0.35, 0.45], atol=1e-6
    )


def test_construction_requires_state_and_body_name(host_model):
    state = MjWarpSceneState(host_model, nworld=1)

    with pytest.raises(ValueError, match="requires an MjWarpSceneState"):
        MjWarpObjectHandler(name="object", body_name="mover")
    with pytest.raises(ValueError, match="requires a non-empty body_name"):
        MjWarpObjectHandler(name="object", state=state)
    # The ABC's own invariant still applies.
    with pytest.raises(ValueError, match="non-empty string"):
        MjWarpObjectHandler(name="", state=state, body_name="mover")
