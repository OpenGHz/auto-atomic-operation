from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir

from auto_atom.basis.mjc.mujoco_basis import EnvConfig, MujocoBasis
from auto_atom.utils.scene_loader import load_scene


ROOT = Path(__file__).resolve().parents[1]


def _load_pick_and_place_model() -> mujoco.MjModel:
    return load_scene(
        ROOT / "assets/xmls/scenes/pick_and_place/demo.xml",
        [ROOT / "assets/xmls/robots/robotiq.xml"],
    )


def _make_visibility_env(model: mujoco.MjModel) -> MujocoBasis:
    env = MujocoBasis.__new__(MujocoBasis)
    env.model = model
    env._operators = {
        "arm": SimpleNamespace(
            root_body="robotiq_interface",
            mocap_body="robotiq_mocap",
        )
    }
    return env


def test_hide_operators_defaults_to_false() -> None:
    config = EnvConfig.model_validate({"model_path": "unused.xml"})

    assert config.hide_operators_in_camera is False


def test_pick_and_place_accepts_hide_operators_override() -> None:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(ROOT / "aao_configs"),
    ):
        cfg = compose(
            config_name="pick_and_place",
            overrides=["env.hide_operators_in_camera=true"],
        )

    assert cfg.env.hide_operators_in_camera is True


def test_operator_render_geoms_cover_complete_body_subtree() -> None:
    model = _load_pick_and_place_model()
    env = _make_visibility_env(model)

    hidden_geom_ids = env._resolve_operator_render_geom_ids()

    tactile_geom_ids = {
        geom_id
        for geom_id in range(model.ngeom)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
        and (name.startswith("left_touch_geom") or name.startswith("right_touch_geom"))
    }
    source_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "source_block_geom"
    )

    assert len(tactile_geom_ids) == 80
    assert len(hidden_geom_ids) == 103
    assert tactile_geom_ids <= hidden_geom_ids
    assert {int(model.geom_group[i]) for i in hidden_geom_ids} == {0, 2, 3}
    assert source_geom_id not in hidden_geom_ids


def test_real_mjv_scene_filter_does_not_mutate_physics_model() -> None:
    model = _load_pick_and_place_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = _make_visibility_env(model)
    env._camera_hidden_geom_ids = env._resolve_operator_render_geom_ids()

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "env0_cam")
    option = mujoco.MjvOption()
    perturb = mujoco.MjvPerturb()
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        perturb,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        scene,
    )

    geom_obj_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    hidden_type = int(mujoco.mjtGeom.mjGEOM_NONE)
    active_operator_geoms = [
        geom
        for geom in scene.geoms[: scene.ngeom]
        if int(geom.objtype) == geom_obj_type
        and int(geom.objid) in env._camera_hidden_geom_ids
        and int(geom.type) != hidden_type
    ]
    assert active_operator_geoms

    model_snapshot = {
        "geom_group": model.geom_group.copy(),
        "geom_rgba": model.geom_rgba.copy(),
        "geom_contype": model.geom_contype.copy(),
        "geom_conaffinity": model.geom_conaffinity.copy(),
        "body_mass": model.body_mass.copy(),
    }
    env._hide_operator_geoms_from_camera_scene(SimpleNamespace(scene=scene))

    assert all(int(geom.type) == hidden_type for geom in active_operator_geoms)
    assert all(
        np.array_equal(before, getattr(model, field))
        for field, before in model_snapshot.items()
    )

    fresh_scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(
        model,
        data,
        option,
        perturb,
        camera,
        mujoco.mjtCatBit.mjCAT_ALL,
        fresh_scene,
    )
    assert any(
        int(geom.objtype) == geom_obj_type
        and int(geom.objid) in env._camera_hidden_geom_ids
        and int(geom.type) != hidden_type
        for geom in fresh_scene.geoms[: fresh_scene.ngeom]
    )


def test_camera_scene_filter_removes_only_operator_geoms() -> None:
    hidden_geom = SimpleNamespace(
        objtype=int(mujoco.mjtObj.mjOBJ_GEOM),
        objid=7,
        type=int(mujoco.mjtGeom.mjGEOM_BOX),
    )
    visible_geom = SimpleNamespace(
        objtype=int(mujoco.mjtObj.mjOBJ_GEOM),
        objid=8,
        type=int(mujoco.mjtGeom.mjGEOM_SPHERE),
    )
    unrelated_site = SimpleNamespace(
        objtype=int(mujoco.mjtObj.mjOBJ_SITE),
        objid=7,
        type=int(mujoco.mjtGeom.mjGEOM_BOX),
    )
    renderer = SimpleNamespace(
        scene=SimpleNamespace(
            geoms=(hidden_geom, visible_geom, unrelated_site),
            ngeom=3,
        )
    )
    env = MujocoBasis.__new__(MujocoBasis)
    env._camera_hidden_geom_ids = frozenset({7})

    env._hide_operator_geoms_from_camera_scene(renderer)

    assert hidden_geom.type == int(mujoco.mjtGeom.mjGEOM_NONE)
    assert visible_geom.type == int(mujoco.mjtGeom.mjGEOM_SPHERE)
    assert unrelated_site.type == int(mujoco.mjtGeom.mjGEOM_BOX)
