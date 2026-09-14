"""Equivalence tests for the MJWarp object-only environment.

The comparisons run against the **real** ``rack_plate_p7_v4_umi_v3`` config in
``object_only`` mode rather than a synthetic scene, because that is what the
port has to reproduce: layered MJCF, an operator layer stripped at the Hydra
boundary, a config-declared object-mounted camera, and nested static scenery.

Every value is checked against the native ``UnifiedMujocoEnv``'s own answer, so
a divergence in convention or derivation fails here rather than surfacing later
as a placement that passes constraints on one backend and not the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from auto_atom.basis.mjc.mujoco_env import UnifiedMujocoEnv  # noqa: E402
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv  # noqa: E402
from auto_atom.config.env_config import EnvConfig  # noqa: E402
from auto_atom.contracts import (  # noqa: E402
    EnvProtocol,
    PoseConstraintEnvProtocol,
)
from auto_atom.execution_config import (  # noqa: E402
    prepare_task_config_for_instantiation,
)

_CONFIG_NAME = "rack_plate_p7_v4_umi_v3"
_REPO_ROOT = Path(__file__).resolve().parents[1]
# Constraint-relevant scene entities: the manipulated plate, the two randomized
# scenery bodies, and the placement target that rides the rack.
_ENTITIES = ("object", "rack", "plate_stand", "rack_target")


def _env_config(batch_size: int) -> EnvConfig:
    config_dir = str(_REPO_ROOT / "aao_configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name=_CONFIG_NAME,
            overrides=[
                "execution.mode=object_only",
                f"env.batch_size={batch_size}",
                "env.viewer=null",
            ],
        )
    prepared = prepare_task_config_for_instantiation(cfg)
    node = OmegaConf.to_container(prepared.env, resolve=True)
    node.pop("_target_", None)
    return EnvConfig.model_validate(node)


@pytest.fixture(scope="module")
def single_world_config() -> EnvConfig:
    return _env_config(1)


@pytest.fixture(scope="module")
def warp_env(single_world_config):
    env = MjWarpObjectOnlyEnv(single_world_config, njmax=512)
    yield env
    env.close()


@pytest.fixture(scope="module")
def native_env(single_world_config):
    env = UnifiedMujocoEnv(single_world_config)
    yield env
    env.close()


def test_satisfies_env_capability_protocols(warp_env):
    assert isinstance(warp_env, EnvProtocol)
    assert isinstance(warp_env, PoseConstraintEnvProtocol)
    assert warp_env.batch_size == 1


def test_configured_step_duration_matches_native(single_world_config, native_env):
    env = MjWarpObjectOnlyEnv(single_world_config, njmax=512)
    try:
        assert env.host_model.opt.timestep == pytest.approx(
            native_env.model.opt.timestep
        )
        before = env.state.data.time.numpy().copy()
        env.step(np.zeros(env.host_model.nu))
        np.testing.assert_allclose(
            env.state.data.time.numpy() - before,
            1.0 / env.config.update_freq,
            atol=1e-7,
        )
    finally:
        env.close()


def test_operator_layer_is_absent_under_object_only(warp_env):
    """object_only strips the operator, so its camera must not be present."""
    assert set(warp_env.camera_names()) == {"rack_camera_front", "plate_cam"}
    assert warp_env.object_camera_names == frozenset({"plate_cam"})


def test_object_mounted_camera_is_not_its_own_witness(warp_env, native_env):
    """plate_cam rides the plate, so it cannot witness it for visible_in."""
    assert warp_env._visibility_witness_camera_names() == (
        native_env._visibility_witness_camera_names()
    )
    assert "plate_cam" not in warp_env._visibility_witness_camera_names()


@pytest.mark.parametrize("entity", _ENTITIES)
def test_support_geometry_matches_native(warp_env, native_env, entity):
    got = warp_env.get_support_geometry(entity)
    want = native_env.get_support_geometry(entity)

    np.testing.assert_allclose(got.center, want.center, atol=1e-6)
    assert got.radius == pytest.approx(want.radius, abs=1e-6)


@pytest.mark.parametrize("camera", ["rack_camera_front", "plate_cam"])
def test_camera_model_matches_native(warp_env, native_env, camera):
    got = warp_env.get_camera_model(camera)
    want = native_env.get_camera_model(camera)

    assert (got.width, got.height) == (want.width, want.height)
    assert got.fovy_radians == pytest.approx(want.fovy_radians)
    assert got.near == pytest.approx(want.near)
    assert got.far == pytest.approx(want.far)
    np.testing.assert_allclose(
        np.asarray(got.pose.position), np.asarray(want.pose.position), atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(got.pose.orientation),
        np.asarray(want.pose.orientation),
        atol=1e-6,
    )


@pytest.mark.parametrize("body", ["object", "rack", "plate_stand"])
def test_body_pose_matches_native(warp_env, native_env, body):
    got_pos, got_quat = warp_env.get_body_pose(body)
    want_pos, want_quat = native_env.get_body_pose(body)

    assert got_pos.shape == (1, 3)
    np.testing.assert_allclose(got_pos[0], want_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat[0], want_quat, atol=1e-6)


@pytest.mark.parametrize("site", ["object_site", "rack_target_site"])
def test_site_pose_matches_native(warp_env, native_env, site):
    got_pos, got_quat = warp_env.get_site_pose(site)
    want_pos, want_quat = native_env.get_site_pose(site)

    np.testing.assert_allclose(got_pos[0], want_pos, atol=1e-6)
    np.testing.assert_allclose(got_quat[0], want_quat, atol=1e-6)


def test_element_pose_prefers_site_then_falls_back_to_body(warp_env, native_env):
    site_pose = warp_env.get_element_pose("object_site")
    want_pos, _ = native_env.get_site_pose("object_site")
    np.testing.assert_allclose(np.asarray(site_pose.position)[0], want_pos, atol=1e-6)

    body_pose = warp_env.get_element_pose("rack")
    want_body_pos, _ = native_env.get_body_pose("rack")
    np.testing.assert_allclose(
        np.asarray(body_pose.position)[0], want_body_pos, atol=1e-6
    )


def test_unknown_camera_raises_keyerror(warp_env):
    with pytest.raises(KeyError, match="Camera 'nope' not found"):
        warp_env.get_camera_model("nope")


def test_configured_camera_absent_from_an_injected_model_is_reported(
    single_world_config,
):
    """A camera the supplied model lacks fails at construction, with the list.

    This only reachable through the ``host_model=`` path. Going through scene
    composition cannot trigger it, because ``load_composed_scene`` *creates* a
    camera the config declares but the scene does not author -- so a renamed
    config camera silently exists rather than going missing.
    """
    minimal = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 0.05"/>
          <camera name="only_cam" pos="0 -1 0.5"/>
        </worldbody></mujoco>
        """
    )

    with pytest.raises(ValueError, match=r"not found in the compiled model"):
        MjWarpObjectOnlyEnv(single_world_config, host_model=minimal, njmax=512)


def test_injected_model_reports_the_available_camera_names(single_world_config):
    """The error names what *is* available, since that is the usual diagnosis."""
    minimal = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody>
          <geom name="floor" type="plane" size="1 1 0.05"/>
          <camera name="only_cam" pos="0 -1 0.5"/>
        </worldbody></mujoco>
        """
    )

    with pytest.raises(ValueError, match=r"only_cam"):
        MjWarpObjectOnlyEnv(single_world_config, host_model=minimal, njmax=512)


def test_interest_operations_are_validated(warp_env):
    warp_env.set_interest_objects_and_operations(["object"], ["pick"])

    with pytest.raises(ValueError, match="same length"):
        warp_env.set_interest_objects_and_operations(["object"], [])
    with pytest.raises(ValueError, match="not configured in operations"):
        warp_env.set_interest_objects_and_operations(["object"], ["teleport"])


# ----------------------------------------------------------------------
# Operator registration (physical mode keeps the operator layer)
# ----------------------------------------------------------------------


def _physical_env_config(batch_size: int) -> EnvConfig:
    """The same real config, but with the operator layer left in place.

    ``object_only`` strips ``task_operators`` at the Hydra boundary, so it cannot
    exercise registration at all. Physical mode keeps the P7 arm's actuators and
    its analytical IK factory, which is what registration has to consume.
    """
    config_dir = str(_REPO_ROOT / "aao_configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name=_CONFIG_NAME,
            overrides=[
                "execution.mode=physical",
                f"env.batch_size={batch_size}",
                "env.viewer=null",
            ],
        )
    prepared = prepare_task_config_for_instantiation(cfg)
    node = OmegaConf.to_container(prepared.env, resolve=True)
    node.pop("_target_", None)
    return EnvConfig.model_validate(node)


@pytest.fixture(scope="module")
def physical_env():
    return MjWarpObjectOnlyEnv(_physical_env_config(2))


def test_registration_binds_the_configured_operator(physical_env):
    """The real config's operator is registered, keyed by its config name."""
    assert physical_env.operator_names == ("arm",)

    operator = physical_env.get_operator_state("arm")
    assert operator.name == "arm"
    assert operator.joint_mode, "the P7 arm is actuator-driven"


def test_registration_constructs_the_real_ik_solver(physical_env):
    """ik_factory is invoked with the joint names its solver expects.

    This is the first point in the port where the real analytical IK solver is
    built, so a wrong joint-name list or a missing ik_params surfaces here rather
    than at the first control tick.
    """
    operator = physical_env.get_operator_state("arm")

    assert operator.ik_solver is not None
    assert hasattr(operator.ik_solver, "solve")


def test_registered_joint_names_match_native_derivation(physical_env):
    """Joint order defines the solver's mapping, so it must match native."""
    operator = physical_env.get_operator_state("arm")
    state = physical_env.state

    got = state.actuator_joint_names(operator.arm_actuator_ids)
    host = state.host_model
    want = [
        mujoco.mj_id2name(
            host,
            mujoco.mjtObj.mjOBJ_JOINT,
            int(host.actuator_trnid[int(actuator), 0]),
        )
        for actuator in operator.arm_actuator_ids
    ]

    assert got == want
    assert len(got) == 7, "the P7 is a 7-DOF arm"


def test_registered_base_pose_matches_the_root_body(physical_env):
    """The base frame is the root body's pose, per world."""
    operator = physical_env.get_operator_state("arm")
    want_pos, want_quat = physical_env.state.get_body_pose_batch(
        operator.root_body_name
    )

    np.testing.assert_allclose(operator.base_position, want_pos, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        operator.base_orientation, want_quat, rtol=1e-6, atol=1e-7
    )
    assert operator.base_position.shape == (2, 3), "per world, not shared"


def test_object_only_registers_nothing(warp_env):
    """The same code path costs nothing in the mode that has no operators.

    ``object_only`` clears ``task_operators`` at the Hydra boundary, so the
    registration loop finds no bindings -- which is why it can run
    unconditionally for both modes.
    """
    assert warp_env.operator_names == ()


def test_unknown_operator_names_what_is_registered(physical_env):
    with pytest.raises(KeyError, match="Registered: arm"):
        physical_env.get_operator_state("nonexistent")


# ----------------------------------------------------------------------
# Joint actions (JointActionEnvProtocol)
# ----------------------------------------------------------------------


def _joint_width(env, operator="arm"):
    state = env.get_operator_state(operator)
    return state.arm_actuator_ids.size + state.eef_actuator_ids.size


def test_satisfies_the_joint_action_protocol(physical_env):
    from auto_atom.contracts import JointActionEnvProtocol

    assert isinstance(physical_env, JointActionEnvProtocol)


def test_joint_action_writes_arm_then_eef_actuators(physical_env):
    """Action layout is arm actuators first, then eef, in config order.

    A transposed or offset mapping would still move the arm, just to the wrong
    joints, so this checks the commanded ctrl slot by slot.
    """
    operator = physical_env.get_operator_state("arm")
    width = _joint_width(physical_env)
    action = np.arange(1, width + 1, dtype=np.float64) * 0.01

    with physical_env.state.deferred_step():
        physical_env.apply_joint_action("arm", action)

    ctrl = physical_env.state.get_ctrl()[0]
    expected = np.concatenate([operator.arm_actuator_ids, operator.eef_actuator_ids])
    np.testing.assert_allclose(ctrl[expected], action, rtol=1e-5, atol=1e-6)


def test_joint_action_broadcasts_and_accepts_per_world_rows(physical_env):
    width = _joint_width(physical_env)
    operator = physical_env.get_operator_state("arm")
    slots = np.concatenate([operator.arm_actuator_ids, operator.eef_actuator_ids])

    with physical_env.state.deferred_step():
        physical_env.apply_joint_action("arm", np.full(width, 0.05))
    shared = physical_env.state.get_ctrl()
    np.testing.assert_allclose(shared[0][slots], shared[1][slots], atol=1e-7)

    per_world = np.stack([np.full(width, 0.02), np.full(width, -0.02)])
    with physical_env.state.deferred_step():
        physical_env.apply_joint_action("arm", per_world)
    got = physical_env.state.get_ctrl()
    np.testing.assert_allclose(got[0][slots], 0.02, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got[1][slots], -0.02, rtol=1e-5, atol=1e-6)


def test_joint_action_respects_a_world_mask(physical_env):
    width = _joint_width(physical_env)
    operator = physical_env.get_operator_state("arm")
    slots = np.concatenate([operator.arm_actuator_ids, operator.eef_actuator_ids])

    with physical_env.state.deferred_step():
        physical_env.apply_joint_action("arm", np.full(width, 0.11))
    with physical_env.state.deferred_step():
        physical_env.apply_joint_action(
            "arm", np.full(width, 0.33), env_mask=np.array([False, True])
        )

    got = physical_env.state.get_ctrl()
    np.testing.assert_allclose(got[0][slots], 0.11, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(got[1][slots], 0.33, rtol=1e-5, atol=1e-6)


def test_kinematic_action_pins_joints_exactly(physical_env):
    """Reset needs the exact pose, not whatever physics converges to."""
    operator = physical_env.get_operator_state("arm")
    width = _joint_width(physical_env)
    target = np.full(width, 0.07)

    physical_env.apply_joint_action("arm", target, kinematic=True)

    qpos_indices = np.concatenate(
        [operator.arm_qpos_indices, operator.eef_qpos_indices]
    )
    got = physical_env.state.get_joint_positions(qpos_indices)
    np.testing.assert_allclose(got[0], target[: qpos_indices.size], atol=1e-5)
    # And no momentum survives the teleport.
    dof_indices = np.concatenate([operator.arm_dof_indices, operator.eef_dof_indices])
    np.testing.assert_allclose(
        physical_env.state.get_joint_velocities(dof_indices), 0.0, atol=1e-7
    )


def test_one_step_per_tick_regardless_of_call_count(physical_env):
    """Two operators-worth of calls in one tick still advance physics once.

    This is the 3.8 contract at the env level: apply_joint_action requests a
    step, and the tick boundary collapses the requests.
    """
    width = _joint_width(physical_env)
    timestep = float(physical_env.state.host_model.opt.timestep)
    before = physical_env.state.data.time.numpy().copy()

    with physical_env.state.deferred_step():
        physical_env.apply_joint_action("arm", np.zeros(width))
        physical_env.apply_joint_action("arm", np.zeros(width))

    after = physical_env.state.data.time.numpy()
    np.testing.assert_allclose(after - before, timestep, rtol=1e-6)


def test_a_wrong_width_action_is_refused_not_truncated(physical_env):
    """Native truncates; this fails closed.

    A short action applied as action[:n] leaves the remaining joints holding a
    stale target, and an arm that moves partially is harder to diagnose than one
    that refuses the command.
    """
    with pytest.raises(ValueError, match="joint value"):
        physical_env.apply_joint_action("arm", np.zeros(3))


def test_a_wrong_batch_is_refused(physical_env):
    width = _joint_width(physical_env)

    with pytest.raises(ValueError, match="must be 1 or batch_size"):
        physical_env.apply_joint_action("arm", np.zeros((3, width)))


def test_joint_action_on_an_unknown_operator_is_refused(physical_env):
    with pytest.raises(KeyError, match="Registered: arm"):
        physical_env.apply_joint_action("nonexistent", np.zeros(1))


# ----------------------------------------------------------------------
# Policy actions (StepEnvProtocol)
# ----------------------------------------------------------------------


def test_satisfies_the_step_protocol(physical_env):
    from auto_atom.contracts import StepEnvProtocol

    assert isinstance(physical_env, StepEnvProtocol)


def test_policy_step_advances_one_control_update(physical_env):
    """One policy action is one control update, i.e. n_substeps physics steps.

    Native's step ends in update(), so a port that advanced a single step would
    run the episode at n_substeps times the intended rate.
    """
    nu = int(physical_env.host_model.nu)
    timestep = float(physical_env.state.host_model.opt.timestep)
    before = physical_env.state.data.time.numpy().copy()

    physical_env.step(np.zeros(nu))

    after = physical_env.state.data.time.numpy()
    np.testing.assert_allclose(
        after - before, timestep * physical_env.n_substeps, rtol=1e-5
    )


def test_policy_action_is_clamped_to_ctrlrange(physical_env):
    """This path clamps on write, unlike the actuator path (3.7).

    Native's step clamps before writing, so a recording that captures ctrl shows
    clamped values here. Divergence would make replayed actions differ.
    """
    nu = int(physical_env.host_model.nu)
    limited = np.asarray(physical_env.host_model.actuator_ctrllimited, dtype=bool)
    assert limited.any(), "config must have at least one limited actuator to test"
    high = np.asarray(physical_env.host_model.actuator_ctrlrange[:, 1])

    physical_env.step(np.full(nu, 1e6))

    got = physical_env.state.get_ctrl()[0]
    np.testing.assert_allclose(got[limited], high[limited], rtol=1e-5, atol=1e-6)


def test_policy_step_accepts_per_world_actions(physical_env):
    nu = int(physical_env.host_model.nu)
    rows = np.stack([np.full(nu, 0.01), np.full(nu, -0.01)])

    physical_env.step(rows)

    got = physical_env.state.get_ctrl()
    assert not np.allclose(got[0], got[1]), "each world keeps its own command"


def test_policy_step_respects_a_world_mask(physical_env):
    nu = int(physical_env.host_model.nu)
    physical_env.step(np.zeros(nu))
    baseline = physical_env.state.get_ctrl()[0].copy()

    physical_env.step(np.full(nu, 0.02), env_mask=np.array([False, True]))

    got = physical_env.state.get_ctrl()
    np.testing.assert_allclose(got[0], baseline, atol=1e-7)
    assert not np.allclose(got[1], baseline)


def test_policy_step_is_refused_inside_a_deferral(physical_env):
    """Deferral collapses steps to one, which would break n_substeps silently."""
    nu = int(physical_env.host_model.nu)

    with pytest.raises(RuntimeError, match="cannot be called inside a deferred_step"):
        with physical_env.state.deferred_step():
            physical_env.step(np.zeros(nu))


def test_policy_step_rejects_a_wrong_width(physical_env):
    with pytest.raises(ValueError, match="joint value"):
        physical_env.step(np.zeros(3))
