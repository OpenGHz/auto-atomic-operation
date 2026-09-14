"""Tests for applying operator initial_state on the MJWarp backend.

The end-to-end physical run (design doc 5.2) failed because nothing applied
initial_state: the arm sat at its raw MJCF pose. These tests pin the fix at the
level that failure occurred -- the operator's world pose after a reset -- against
the real rack_plate config rather than a synthetic scene, because the bug was a
gap between units that each passed their own tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from auto_atom.backend.mjwarp.backend import build_mjwarp_backend  # noqa: E402
from auto_atom.backend.mjwarp.initial_state import apply_initial_state  # noqa: E402
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv  # noqa: E402
from auto_atom.config.env_config import EnvConfig  # noqa: E402
from auto_atom.config.task import AutoAtomConfig, OperatorConfig  # noqa: E402
from auto_atom.execution_config import (  # noqa: E402
    prepare_task_config_for_instantiation,
)
from auto_atom.runtime import ComponentRegistry  # noqa: E402

_CONFIG_NAME = "rack_plate_p7_v4_umi_v3"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _build(batch_size: int = 2):
    ComponentRegistry.clear()
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "aao_configs"), version_base=None
    ):
        cfg = compose(
            config_name=_CONFIG_NAME,
            overrides=[
                "execution.mode=physical",
                f"env.batch_size={batch_size}",
                "env.viewer=null",
            ],
        )
    prepared = prepare_task_config_for_instantiation(cfg)
    env_node = OmegaConf.to_container(prepared.env, resolve=True)
    env_node.pop("_target_", None)
    MjWarpObjectOnlyEnv(EnvConfig.model_validate(env_node), njmax=512)

    task = AutoAtomConfig.model_validate(
        OmegaConf.to_container(prepared.task, resolve=True)
    )
    operators = {
        name: OperatorConfig.model_validate({"name": name, **(node or {})})
        for name, node in (
            OmegaConf.to_container(
                OmegaConf.select(prepared, "task_operators", default={}), resolve=True
            )
            or {}
        ).items()
    }
    backend = build_mjwarp_backend(task, operators)
    backend.setup(task)
    return backend, operators


@pytest.fixture(scope="module")
def built():
    backend, operators = _build()
    yield backend, operators
    backend.teardown()


# The config's base_pose, and the eef world pose that base ⊗ base-frame eef_pose
# composes to. The base is rotated +90° about z, so the base-frame offset
# [0.605, 0, 0.055] rotates to world [+0.105 in y, ...] rather than adding to x
# -- computed with base_to_world, verified to the millimetre against the run.
_CONFIG_BASE_POS = np.array([-0.20, -0.5, 0.075])
_CONFIG_EEF_WORLD = np.array([-0.20, 0.105, 0.130165])


def test_reset_moves_the_base_to_the_configured_pose(built):
    """Before this fix the base sat at the raw MJCF pose [1.0, 0.4, -0.3]."""
    backend, _ = built
    backend.reset()

    base = np.asarray(backend.get_operator_handler("arm").get_base_pose().position)
    for world in range(backend.batch_size):
        np.testing.assert_allclose(base[world], _CONFIG_BASE_POS, atol=1e-6)


def test_reset_places_the_eef_near_the_configured_home(built):
    """The eef ends up where base * base-frame eef_pose says, within IK error.

    This is the measurement that was wrong in 5.2: the eef started at
    [1.3489, 0.4, -0.0737] instead of near [-0.20, 0.105, 0.13].
    """
    backend, _ = built
    backend.reset()

    eef = np.asarray(
        backend.get_operator_handler("arm").get_end_effector_pose().position
    )
    for world in range(backend.batch_size):
        # A few cm of tolerance: the analytical IK solution is exact in joint
        # space but the eef site has a small tool offset the home pose realises.
        assert np.linalg.norm(eef[world] - _CONFIG_EEF_WORLD) < 0.05, (
            f"world {world} eef {eef[world]} far from {_CONFIG_EEF_WORLD}"
        )


def test_the_arm_no_longer_starts_far_from_the_work_area(built):
    """Regression guard for 5.2: eef must not be out near x=1.35 anymore."""
    backend, _ = built
    backend.reset()

    eef = np.asarray(
        backend.get_operator_handler("arm").get_end_effector_pose().position
    )
    assert np.all(eef[:, 0] < 0.5), f"eef x should be near the base, got {eef[:, 0]}"


def test_gripper_home_matches_the_configured_eef_value(built):
    """initial_state.eef: 0.0 must land on the gripper actuator."""
    backend, operators = built
    backend.reset()

    handler = backend.get_operator_handler("arm")
    configured = operators["arm"].initial_state.eef
    eef_id = handler.operator.eef_actuator_ids[0]
    ctrl = backend.env.state.get_ctrl()
    for world in range(backend.batch_size):
        assert ctrl[world][eef_id] == pytest.approx(float(configured), abs=1e-6)


def test_applying_twice_is_idempotent(built):
    """A second reset must land the base in the same place, not drift.

    override_base_pose recomputes the tool offset from the current eef each time;
    if the base write and the offset refresh disagreed, repeated resets would
    walk the base. They must not.
    """
    backend, _ = built
    backend.reset()
    first = np.asarray(
        backend.get_operator_handler("arm").get_base_pose().position
    ).copy()
    backend.reset()
    second = np.asarray(backend.get_operator_handler("arm").get_base_pose().position)

    np.testing.assert_allclose(second, first, atol=1e-6)


def test_partial_pose_overrides_are_refused_not_guessed():
    """A structured/partial override must raise, not resolve to something wrong.

    Resolving these means reproducing the native pose resolver; a silently wrong
    home pose is exactly the failure this round removes.
    """
    from types import SimpleNamespace

    backend, _ = _build()
    try:
        handler = backend.get_operator_handler("arm")
        # position omitted -> not the full-pose form.
        bad = SimpleNamespace(
            base_pose=SimpleNamespace(
                position=None, orientation=[0, 0, 0, 1.0], reference="world"
            ),
            eef_pose=None,
            eef=None,
            joint_positions={},
        )
        with pytest.raises(NotImplementedError, match="full-pose form"):
            apply_initial_state(handler, bad)
    finally:
        backend.teardown()


def test_joint_positions_initial_state_is_refused():
    """The target config uses eef_pose; joint homing is not implemented."""
    from types import SimpleNamespace

    backend, _ = _build()
    try:
        handler = backend.get_operator_handler("arm")
        bad = SimpleNamespace(
            base_pose=None,
            eef_pose=None,
            eef=None,
            joint_positions={"joint1": 0.1},
        )
        with pytest.raises(NotImplementedError, match="joint_positions"):
            apply_initial_state(handler, bad)
    finally:
        backend.teardown()
