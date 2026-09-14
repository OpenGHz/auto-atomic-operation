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
    _collect_object_names,
    build_mjwarp_object_only_backend,
)
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv  # noqa: E402
from auto_atom.config.env_config import EnvConfig  # noqa: E402
from auto_atom.config.reference import RandomizationReference  # noqa: E402
from auto_atom.config.task import AutoAtomConfig  # noqa: E402
from auto_atom.contracts import SceneBackend  # noqa: E402
from auto_atom.execution_config import (  # noqa: E402
    prepare_task_config_for_instantiation,
)
from auto_atom.runtime import ComponentRegistry  # noqa: E402

_CONFIG_NAME = "rack_plate_p7_v4_umi_v3"
_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("spec_form", ["range", "regions", "proposal"])
def test_object_collection_includes_operator_pose_references(spec_form):
    """Base/EEF and per-axis references can name otherwise unlisted scenery."""
    base = {
        "reference": "base_support",
        "x": {"range": [0, 0], "reference": "axis_support"},
    }
    eef = {"reference": "eef_support"}
    if spec_form == "regions":
        base = {"regions": [base, {"reference": "region_support"}]}
        eef = {"regions": [eef]}
    elif spec_form == "proposal":
        base = {"proposal": {"regions": [base, {"reference": "region_support"}]}}
        eef = {"proposal": eef}
    task = AutoAtomConfig.model_validate(
        {
            "env_name": "test",
            "stages": [],
            "initial_pose": {"initial_scenery": {"position": [0, 0, 0]}},
            "randomization": {
                "entities": {
                    "arm": {"base": base, "eef": eef},
                    "object": {"reference": "object_support"},
                    "missing_body": {"reference": "missing_reference"},
                }
            },
        }
    )
    bodies = {
        "object",
        "base_support",
        "axis_support",
        "eef_support",
        "region_support",
        "object_support",
        "initial_scenery",
    }
    expected = bodies if spec_form != "range" else bodies - {"region_support"}

    assert _collect_object_names(task, bodies.__contains__) == expected


@pytest.mark.parametrize("parts", [(), ("base",), ("eef",), ("base", "eef")])
def test_object_collection_ignores_reference_modes_and_absent_operator_specs(parts):
    """Reference modes must not become handlers, even with colliding body names."""
    task = AutoAtomConfig.model_validate(
        {
            "env_name": "test",
            "stages": [],
            "randomization": {
                "entities": {
                    "arm": {part: {"reference": "absolute_base"} for part in parts},
                    "object": {
                        "reference": "relative",
                        "x": {"range": [0, 0], "reference": "absolute_world"},
                    },
                }
            },
        }
    )
    bodies = {"object", *(reference.value for reference in RandomizationReference)}

    assert _collect_object_names(task, bodies.__contains__) == {"object"}


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


def test_reset_coalesces_kinematics_passes(backend):
    """Randomization's many writes must not each run their own pass.

    Measured on this config, eager writes cost 6 kernel launches for 5 entities;
    deferring brings it to 2 -- two rather than one because plate_cam is mounted
    on the randomized plate, so its write correctly forces a flush first.

    Counting real launches rather than calls to ``forward()`` is the point:
    deferral works by making the method return early, so a call count would show
    no change at all.
    """
    backend.reset()  # warm, so kernel compilation is not counted

    launches = {"n": 0}
    state = backend.env.state
    original = state._mjw.forward

    def counted(model, data):
        launches["n"] += 1
        original(model, data)

    state._mjw.forward = counted
    try:
        backend.reset()
    finally:
        state._mjw.forward = original

    assert 0 < launches["n"] <= 3


def test_reset_results_do_not_depend_on_deferral(backend):
    """Deferral is an optimisation, so disabling it must change nothing."""
    from contextlib import contextmanager

    from auto_atom.basis.mjwarp.state import MjWarpSceneState

    backend.reset()
    deferred = np.asarray(backend.get_object_handler("object").get_pose().position)

    @contextmanager
    def _no_deferral(self):
        yield

    original = MjWarpSceneState.deferred_forward
    MjWarpSceneState.deferred_forward = _no_deferral
    try:
        eager_backend = _build()
        try:
            eager_backend.reset()
            eager = np.asarray(
                eager_backend.get_object_handler("object").get_pose().position
            )
        finally:
            eager_backend.teardown()
    finally:
        MjWarpSceneState.deferred_forward = original

    np.testing.assert_allclose(deferred, eager, rtol=1e-6, atol=1e-7)


def test_builder_rejects_operators_the_env_cannot_drive():
    """A physical task composed against an object_only env is a mode mismatch.

    The builder used to refuse *all* operators because physical execution was
    unimplemented. It now assembles them, so the remaining error case is
    narrower and worth a specific message: object_only strips the operator MJCF
    layers, so the scene genuinely has no arm, and the previous behaviour was a
    bare "not registered" KeyError from inside the assembly.
    """
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

    with pytest.raises(ValueError, match="registered none"):
        build_mjwarp_object_only_backend(task, {"arm": {}})


def test_element_pose_and_joint_angle_reach_the_scene(backend):
    """Both delegate to the env, which the place stage and arcs rely on."""
    pose = backend.get_element_pose("object_site")
    assert np.asarray(pose.position).shape[1] == 3

    assert backend.get_joint_angle("object_free") == pytest.approx(
        float(backend.env.state.data.qpos.numpy()[0][0])
    )


# ----------------------------------------------------------------------
# Grasp and contact queries in physical mode
# ----------------------------------------------------------------------


def _build_physical(batch_size: int = 2):
    """Build the backend directly against a physical-mode env.

    The builder still refuses a non-empty operator mapping (that is 4c-3c-3d),
    so the backend is constructed directly here. That is the only way to exercise
    the operator surface today, and it is worth exercising now rather than
    waiting: these four methods answer stage post-conditions, so a wrong answer
    would silently pass or fail stages later.
    """
    from auto_atom.backend.mjwarp.handlers import MjWarpObjectHandler
    from auto_atom.config.randomization import ResolvedRandomizationConfig

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
    env = MjWarpObjectOnlyEnv(EnvConfig.model_validate(env_node), njmax=512)

    task = AutoAtomConfig.model_validate(
        OmegaConf.to_container(prepared.task, resolve=True)
    )
    handlers = {
        "object": MjWarpObjectHandler(
            name="object", state=env.state, body_name="object"
        )
    }
    backend = MjWarpObjectOnlyBackend(
        config=task,
        env=env,
        object_handlers=handlers,
        randomization=ResolvedRandomizationConfig.from_scope_config(task.randomization),
    )
    backend.setup(task)
    return backend


@pytest.fixture(scope="module")
def physical_backend():
    built = _build_physical()
    yield built
    built.teardown()


def test_grasp_queries_answer_per_world(physical_backend):
    """Every query returns one answer per world, which is what stages index."""
    got = physical_backend.is_object_grasped("arm", "object")

    assert got.shape == (physical_backend.batch_size,)
    assert got.dtype == bool


def test_nothing_is_grasped_at_rest(physical_backend):
    """The arm starts at its home pose, nowhere near the plate."""
    assert not physical_backend.is_object_grasped("arm", "object").any()
    assert not physical_backend.is_operator_grasping("arm").any()
    assert physical_backend.get_grasped_object_name("arm", 0) is None


def test_operator_is_not_contacting_at_rest(physical_backend):
    assert not physical_backend.is_operator_contacting("arm", "object").any()


def test_an_unknown_object_is_not_grasped_rather_than_an_error(physical_backend):
    """A stage may ask about an object this scene does not contain."""
    got = physical_backend.is_object_grasped("arm", "no_such_object")

    assert got.shape == (physical_backend.batch_size,)
    assert not got.any()
    assert not physical_backend.is_operator_contacting("arm", "no_such_object").any()


def test_grasp_queries_are_cached_per_operator(physical_backend):
    """Topology is static, so one build per operator -- not one per tick."""
    first = physical_backend._grasp_queries("arm")
    second = physical_backend._grasp_queries("arm")

    assert first is second


def test_unknown_operator_still_fails_loudly(physical_backend):
    """A stub answering 'not grasping' would be worse than an error."""
    with pytest.raises(KeyError):
        physical_backend.is_object_grasped("nonexistent", "object")


def test_grasped_object_index_is_bounds_checked(physical_backend):
    with pytest.raises(IndexError, match="env_index must be in"):
        physical_backend.get_grasped_object_name("arm", physical_backend.batch_size)


def test_object_only_operator_surface_still_refuses(backend):
    """object_only strips the operator layer, so these must fail, not answer."""
    for call in (
        lambda: backend.is_object_grasped("arm", "object"),
        lambda: backend.is_operator_grasping("arm"),
        lambda: backend.is_operator_contacting("arm", "object"),
        lambda: backend.get_grasped_object_name("arm", 0),
    ):
        with pytest.raises(KeyError):
            call()


# ----------------------------------------------------------------------
# Physical handler assembly through the builder
# ----------------------------------------------------------------------


def _build_via_builder(batch_size: int = 2):
    """Build in physical mode through the real builder, as a task file would."""
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
    # task_operators is a top-level node on the prepared config, not part of the
    # narrower AutoAtomConfig the backend receives, so the runtime passes it to
    # the builder separately -- which is why the builder takes it as an argument.
    from auto_atom.config.task import OperatorConfig

    operators = {
        name: OperatorConfig.model_validate({"name": name, **(node or {})})
        for name, node in (
            OmegaConf.to_container(
                OmegaConf.select(prepared, "task_operators", default={}), resolve=True
            )
            or {}
        ).items()
    }
    backend = build_mjwarp_object_only_backend(task, operators)
    backend.setup(task)
    return backend, operators


@pytest.fixture(scope="module")
def assembled():
    backend, operators = _build_via_builder()
    yield backend, operators
    backend.teardown()


def test_builder_assembles_an_operator_handler(assembled):
    """The seam the runtime asks for now exists in physical mode."""
    from auto_atom.contracts import OperatorHandler

    backend, _ = assembled
    handler = backend.get_operator_handler("arm")

    assert isinstance(handler, OperatorHandler)
    assert handler.name == "arm"


def test_assembled_handler_carries_both_control_halves(assembled):
    backend, _ = assembled
    handler = backend.get_operator_handler("arm")

    assert handler.arm is not None, "move_to_pose needs the arm half"
    assert handler.eef is not None, "control_eef needs the gripper half"
    assert handler.arm.ik.failure_streak(0) == 0


def test_control_parameters_come_from_the_task_config(assembled):
    """The task's control block reaches the state machines, not defaults."""
    backend, operators = assembled
    handler = backend.get_operator_handler("arm")
    control = (operators["arm"].model_extra or {}).get("control") or {}

    assert handler.arm.timeout_steps == int(control["timeout_steps"])
    assert handler.arm.n_substeps == backend.env.n_substeps
    assert handler.eef.n_substeps == backend.env.n_substeps
    assert handler.arm.max_linear_step == pytest.approx(
        float(control["cartesian_max_linear_step"])
    )
    tolerance = control["tolerance"]
    assert handler.arm.orientation_tolerance == pytest.approx(
        float(tolerance["orientation"])
    )
    # `placed` is nested under `tolerance` in this config; the builder accepts it
    # in either position, so the test reads it where the config actually puts it.
    assert handler.get_placed_tolerances()[0] == pytest.approx(
        float(tolerance["placed"]["position"])
    )


def test_ik_parameters_reach_the_operator_state(assembled):
    """joint_control_mode and max_joint_delta live on the task side."""
    backend, operators = assembled
    handler = backend.get_operator_handler("arm")
    ik_block = (operators["arm"].model_extra or {}).get("ik") or {}

    assert handler.operator.joint_control_mode == ik_block["joint_control_mode"]
    assert handler.operator.max_joint_delta == pytest.approx(
        float(ik_block["max_joint_delta"])
    )


def test_gripper_limits_are_derived_not_defaulted(assembled):
    """The real UMI claw is ctrlrange 0..0.0165, not robotiq's 0..0.82.

    Inheriting the robotiq default close value would command an unreachable
    target; inheriting the default tolerance would exceed the whole travel.
    """
    backend, operators = assembled
    handler = backend.get_operator_handler("arm")
    control = (operators["arm"].model_extra or {}).get("control") or {}
    configured_eef_tolerance = (control.get("tolerance") or {}).get("eef")

    assert handler.eef.eef_close_value < 0.1, "must not be robotiq's 0.82"
    assert handler.eef.eef_close_value > 0.0
    if configured_eef_tolerance is not None:
        # The config states one, so it wins over the derived value.
        assert handler.eef.eef_tolerance == pytest.approx(
            float(configured_eef_tolerance)
        )
    else:
        assert handler.eef.eef_tolerance < handler.eef.eef_close_value


def test_grasp_queries_work_through_the_assembled_backend(assembled):
    backend, _ = assembled

    assert not backend.is_operator_grasping("arm").any()
    assert backend.get_grasped_object_name("arm", 0) is None


def test_canonical_builder_alias_is_the_same_callable():
    from auto_atom.backend.mjwarp.backend import (
        build_mjwarp_backend,
        build_mjwarp_object_only_backend as legacy,
    )

    assert build_mjwarp_backend is legacy
