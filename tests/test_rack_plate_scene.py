"""Regression tests for the migrated PlaceGen rack-plate assets."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from auto_atom.scene_composition import (  # noqa: E402
    MjcfLayerConfig,
    SceneComposer,
    SceneConfig,
    load_composed_scene,
)


_ROOT = Path(__file__).resolve().parents[1]
_SCENE = _ROOT / "assets/xmls/scenes/rack_plate/demo.xml"
_RACK_MESH = _ROOT / "assets/meshes/rack_plate/rack-plate-0.obj"
_PLATE_MESH = _ROOT / "assets/meshes/rack_plate/plate.obj"
_ROBOT_XML = _ROOT / "assets/xmls/robots/xf9600_mocap.xml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type.name}: {name}"
    return int(object_id)


def test_migrated_mesh_payloads_are_exact_and_plate_is_rack_local() -> None:
    assert _sha256(_RACK_MESH) == (
        "246f635a77aa42b724186150f736205a908c6fae85768cf06d5e91ccad4bf74d"
    )
    assert _sha256(_PLATE_MESH) == (
        "c55bb971d452b633f21f44fc32911e2cfe87c3d8a06cf4dc19e559a1164fc0d0"
    )
    assert _RACK_MESH.is_file()
    assert _PLATE_MESH.is_file()


def test_robotless_scene_loads_and_preserves_source_geometry_contract() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    assert model.nu == 0
    assert model.nq == 7
    assert model.nmesh == 2
    assert model.ncam == 5

    root = ET.parse(_SCENE).getroot()
    compiler = root.find("compiler")
    assert compiler is not None
    assert compiler.get("meshdir") == "../../../meshes"
    assert root.find("asset/mesh[@name='rack_plate_mesh']").get("file") == (
        "rack_plate/rack-plate-0.obj"
    )
    assert root.find("asset/mesh[@name='plate_mesh']").get("file") == (
        "rack_plate/plate.obj"
    )

    rack = _id(model, mujoco.mjtObj.mjOBJ_BODY, "rack")
    visual = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "rack_visual")
    assert model.geom_bodyid[visual] == rack
    assert model.geom_contype[visual] == 0
    assert model.geom_conaffinity[visual] == 0

    rib_ids = [
        _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"rack_rib_{index}") for index in range(9)
    ]
    assert all(model.geom_bodyid[geom_id] == rack for geom_id in rib_ids)
    assert all(model.geom_contype[geom_id] != 0 for geom_id in rib_ids)

    object_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    object_visual = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_visual")
    object_collision = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")
    assert model.geom_bodyid[object_visual] == object_body
    assert model.geom_bodyid[object_collision] == object_body
    assert model.geom_contype[object_visual] == 0
    assert model.geom_conaffinity[object_visual] == 0
    assert model.geom_contype[object_collision] != 0
    stand_back = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_stand_back")
    stand_front = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_stand_front")
    stand_left = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_stand_left_rail")
    stand_right = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate_stand_right_rail")
    np.testing.assert_allclose(
        model.geom_pos[stand_back], [-0.432, 0.0, 0.060], atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.geom_size[stand_back], [0.006, 0.020, 0.050], atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.geom_pos[stand_left], [-0.400, -0.065, 0.0465], atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.geom_pos[stand_right], [-0.400, 0.065, 0.0465], atol=1.0e-6
    )
    np.testing.assert_allclose(
        model.geom_size[stand_left], [0.040, 0.001, 0.014], atol=1.0e-6
    )
    np.testing.assert_array_equal(model.geom_contype[stand_front], 0)
    np.testing.assert_array_equal(model.geom_conaffinity[stand_front], 0)
    target_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "rack_target")
    np.testing.assert_allclose(
        model.body_pos[target_body], [-0.126322355, 0.0, 0.088], atol=1.0e-6
    )
    freejoint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
    address = int(model.jnt_qposadr[freejoint])
    np.testing.assert_allclose(
        model.qpos0[address : address + 7],
        [-0.4, 0.0, 0.130165074, 1.0, 0.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for guide in (stand_back, stand_left, stand_right):
        assert (
            mujoco.mj_geomDistance(model, data, object_collision, guide, 10.0, None)
            > 1.0e-4
        )

    for camera_name in (
        "rack_camera_front",
        "rack_camera_back",
        "rack_camera_left",
        "rack_camera_right",
        "rack_camera_top",
    ):
        _id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)


def test_host_can_be_compiled_with_an_ordered_robot_layer() -> None:
    config = SceneConfig(
        base=_SCENE,
        layers=(MjcfLayerConfig(path=_ROBOT_XML, role="operator"),),
    )
    artifact = SceneComposer().compile(config)
    assert "xf9600_interface" in artifact.xml
    assert _SCENE.resolve() in artifact.dependencies
    assert _ROBOT_XML.resolve() in artifact.dependencies

    model = load_composed_scene(config, artifact)
    assert model.nq > 7
    _id(model, mujoco.mjtObj.mjOBJ_BODY, "xf9600_interface")
