"""Contract tests for the MJWarp object-only backend.

Built from the real ``rack_plate_p7_v4_umi_v3`` task file in ``object_only``
mode, so the whole chain under test is the production one: Hydra composition,
operator stripping, scene compilation, handler collection, and a reset that runs
constrained randomization through ``RandomizationHost``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")

from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from auto_atom.backend.mjwarp.backend import (  # noqa: E402
    MjWarpObjectOnlyBackend,
    build_mjwarp_object_only_backend,
)
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv  # noqa: E402
from auto_atom.config.env_config import EnvConfig  # noqa: E402
from auto_atom.config.task import AutoAtomConfig  # noqa: E402
from auto_atom.contracts import SceneBackend  # noqa: E402
from auto_atom.execution_config import (  # noqa: E402
    prepare_task_config_for_instantiation,
)
from auto_atom.runtime import ComponentRegistry  # noqa: E402

_CONFIG_NAME = "rack_plate_p7_v4_umi_v3"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _build(batch_size: int = 2) -> MjWarpObjectOnlyBackend:
    """Compose the real task file and build the backend it describes."""
    ComponentRegistry.clear()
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "aao_configs"), version_base=None
    ):
        cfg = compose(
            config_name=_CONFIG_NAME,
            overrides=[
                "execution.mode=object_only",
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
    backend = build_mjwarp_object_only_backend(task, {})
    backend.setup(task)
    return backend


@pytest.fixture
def backend():
    built = _build()
    yield built
    built.teardown()


def test_satisfies_scene_backend_contract(backend):
    assert isinstance(backend, SceneBackend)
    assert backend.batch_size == 2


def test_dt_per_update_matches_the_configured_substep_rate(backend):
    """timestep x (sim_freq / update_freq), as the native backend derives it."""
    timestep = float(backend.env.host_model.opt.timestep)
    assert backend.env.n_substeps == 40  # 1200 / 30
    assert backend.dt_per_update == pytest.approx(timestep * 40)


def test_handlers_cover_stage_targets_and_randomized_scenery(backend):
    """Randomized bodies need handlers even when no stage names them.

    rack and plate_stand are scenery: they never appear as a stage object, but
    randomization moves them, so pose get/set has to be available.
    """
    assert set(backend.object_handlers) == {
        "object",
        "rack",
        "rack_target",
        "plate_stand",
    }


def test_rigid_attachment_distinguishes_held_frames(backend):
    """The guard behind controlled_frame.kind='held_object'.

    object_site is a child of the plate, so it travels with it.
    rack_target_site rides the rack instead, so treating it as held would
    silently control the wrong frame during the place stage.
    """
    assert backend.is_element_rigidly_attached_to_object("object_site", "object")
    assert not backend.is_element_rigidly_attached_to_object(
        "rack_target_site", "object"
    )


def test_rigid_attachment_rejects_unknown_names(backend):
    with pytest.raises(KeyError):
        backend.is_element_rigidly_attached_to_object("object_site", "nope")
    with pytest.raises(KeyError):
        backend.is_element_rigidly_attached_to_object("nope", "object")


def test_reset_runs_constrained_randomization_per_world(backend):
    """The payoff: one shared model, independently randomized worlds.

    A successful sample also means evaluate_pose_constraints accepted it, so the
    configured visible_in constraint against rack_camera_front was exercised.
    """
    before = backend.get_object_handler("object").get_pose()
    backend.reset()
    after = backend.get_object_handler("object").get_pose()

    assert backend.reset_index == 1
    assert not np.allclose(before.position, after.position, atol=1e-4)
    # Each world drew its own sample rather than sharing one broadcast pose.
    assert not np.allclose(after.position[0], after.position[1], atol=1e-4)

    # Every sample lands inside the configured proposal box.
    positions = np.asarray(after.position)
    assert np.all(positions[:, 0] >= -1.050) and np.all(positions[:, 0] <= 1.050)
    assert np.all(positions[:, 1] >= -0.550) and np.all(positions[:, 1] <= 0.550)
    assert np.all(positions[:, 2] >= 0.130) and np.all(positions[:, 2] <= 0.370)


def test_reset_is_reproducible_for_a_fixed_seed():
    """Same seed, same samples -- the run stays replayable."""
    first = _build()
    try:
        first.reset()
        first_poses = np.asarray(first.get_object_handler("object").get_pose().position)
    finally:
        first.teardown()

    second = _build()
    try:
        second.reset()
        second_poses = np.asarray(
            second.get_object_handler("object").get_pose().position
        )
    finally:
        second.teardown()

    np.testing.assert_allclose(first_poses, second_poses, rtol=1e-6, atol=1e-7)


def test_operator_queries_fail_loudly(backend):
    """No operator exists, and a stub would read as a legitimate answer.

    An empty grasp state would look like "not grasping" and mask a
    misconfigured physical run, so every operator query raises instead.
    """
    for call in (
        lambda: backend.get_operator_handler("arm"),
        lambda: backend.is_operator_grasping("arm"),
        lambda: backend.is_object_grasped("arm", "object"),
        lambda: backend.get_grasped_object_name("arm", 0),
        lambda: backend.is_operator_contacting("arm", "object"),
        lambda: backend.get_operator_support_geometry("arm", "base"),
        lambda: backend.live_pose("arm.eef"),
        lambda: backend.set_target_pose(
            "operator_base",
            "arm",
            backend.get_object_handler("object").get_pose(),
            np.ones(backend.batch_size, dtype=bool),
        ),
    ):
        with pytest.raises(KeyError, match="object_only"):
            call()


def test_operator_contacts_returns_none_not_an_error(backend):
    """The contract defines None as 'contacts unobservable', so answer it.

    This is a diagnostic path the runtime calls opportunistically, unlike the
    queries above whose answers feed control decisions.
    """
    assert backend.get_operator_contacts("arm", 0) is None


def test_operator_names_is_empty_and_objects_are_listed(backend):
    assert backend.operator_names == ()
    assert set(backend.object_names) == set(backend.object_handlers)


def test_baseline_poses_are_recorded_for_objects_and_cameras(backend):
    assert backend.baseline_pose("object") is not None
    assert backend.baseline_pose("rack_camera_front") is not None
    assert backend.baseline_pose("no_such_target") is None


def test_builder_rejects_operators():
    """A non-empty task_operators means the task wants physical execution."""
    ComponentRegistry.clear()
    with initialize_config_dir(
        config_dir=str(_REPO_ROOT / "aao_configs"), version_base=None
    ):
        cfg = compose(
            config_name=_CONFIG_NAME,
            overrides=[
                "execution.mode=object_only",
                "env.batch_size=1",
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

    with pytest.raises(ValueError, match="no operators"):
        build_mjwarp_object_only_backend(task, {"arm": {}})


def test_element_pose_and_joint_angle_reach_the_scene(backend):
    """Both delegate to the env, which the place stage and arcs rely on."""
    pose = backend.get_element_pose("object_site")
    assert np.asarray(pose.position).shape[1] == 3

    assert backend.get_joint_angle("object_free") == pytest.approx(
        float(backend.env.state.data.qpos.numpy()[0][0])
    )
