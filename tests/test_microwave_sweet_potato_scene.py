"""Regression tests for the migrated microwave and sweet-potato assets."""

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
_SCENE = _ROOT / "assets/xmls/scenes/microwave_sweet_potato/demo.xml"
_MESH_ROOT = _ROOT / "assets/meshes/microwave_sweet_potato"
_HULL_ROOT = _MESH_ROOT / "microwave/convex"
_POTATO_MESH = _MESH_ROOT / "sweet_potato/sweet_potato.obj"
_BODY_VISUAL = _MESH_ROOT / "microwave/microwave_body.obj"
_DOOR_VISUAL = _MESH_ROOT / "microwave/microwave_door.obj"
_ROBOT_XML = _ROOT / "assets/xmls/robots/umi_gripper_v3_mocap.xml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type.name}: {name}"
    return int(object_id)


def test_migrated_payloads_are_byte_identical_to_the_vendor_bundle() -> None:
    assert _sha256(_POTATO_MESH) == (
        "7312cab781ab477f215944e4a4808505ca0656d3bec243b5ecd11820154a51b3"
    )
    assert _sha256(_BODY_VISUAL) == (
        "8519021d027bfee55e0653d944ccc70830bf1e3a14d1181001f91e5a0eceba7b"
    )
    assert _sha256(_DOOR_VISUAL) == (
        "fa7df837a5035f1f46bea85c7e607cdd489936259127e4a0980187240c871323"
    )
    # One digest over "<sha256>  <relative path>" lines of all 116 hulls.
    lines = [
        f"{_sha256(path)}  {path.relative_to(_HULL_ROOT).as_posix()}"
        for path in sorted(_HULL_ROOT.rglob("*.obj"))
    ]
    assert len(lines) == 116
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == (
        "547c5caf6e93975d2196b84a28cdab10e79345c7eaec04af5cb75e723e8b9e69"
    )


def test_robotless_host_loads_with_relative_paths() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    # Door hinge + sweet-potato freejoint; no robot, actuator, or keyframe.
    assert model.nq == 8
    assert model.nu == 0
    assert model.nkey == 0
    assert model.nmesh == 119
    assert {model.camera(i).name for i in range(model.ncam)} == {
        "env0_cam",
        "env1_cam",
    }

    root = ET.parse(_SCENE).getroot()
    assert root.find("compiler").get("meshdir") == "../../../meshes"
    for xml_path in (_SCENE, *(_SCENE.parent / "includes").glob("*.xml")):
        for element in ET.parse(xml_path).getroot().iter():
            file_attr = element.get("file")
            if file_attr is not None:
                assert not Path(file_attr).is_absolute(), (xml_path, file_attr)


def test_microwave_visual_meshes_and_hidden_collision_hulls() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    cabinet = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave")
    door = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_door")
    # The cabinet is fixed to the world; only the door articulates.
    assert model.body_jntnum[cabinet] == 0
    assert int(model.body_parentid[door]) == cabinet

    for name, body in (
        ("microwave_body_visual", cabinet),
        ("microwave_door_visual", door),
    ):
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert int(model.geom_bodyid[geom]) == body
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        assert model.geom_contype[geom] == 0
        assert model.geom_conaffinity[geom] == 0
        assert model.geom_group[geom] == 0

    for prefix, body, count in (
        ("microwave_body_hull_", cabinet, 52),
        ("microwave_door_hull_", door, 64),
    ):
        for index in range(count):
            geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}{index:03d}")
            assert int(model.geom_bodyid[geom]) == body
            assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
            assert model.geom_contype[geom] == 1
            assert model.geom_conaffinity[geom] == 1
            # Hidden by default, as in the source collision class.
            assert model.geom_group[geom] == 3
            assert model.geom_condim[geom] == 4
    # Render-only geoms carry no mass; the hulls keep the source density.
    np.testing.assert_allclose(model.body_mass[door], 0.7094578826867317, rtol=1e-6)

    # The overlapping hinge seam is excluded, as in the source model.
    assert model.nexclude == 1
    assert {int(model.exclude_signature[0]) >> 16, int(model.exclude_signature[0]) & 0xFFFF} == {
        cabinet,
        door,
    }


def test_door_defaults_open_in_source_hinge_coordinates() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    hinge = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "microwave_door_hinge")
    address = int(model.jnt_qposadr[hinge])
    np.testing.assert_allclose(model.qpos0[address], -np.pi / 2, atol=1.0e-7)
    np.testing.assert_allclose(
        model.jnt_range[hinge], np.deg2rad([-120.0, 32.0]), atol=1.0e-7
    )

    data = mujoco.MjData(model)
    door = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_door")
    cabinet = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave")

    def door_bounds(angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Door hull AABB in the cabinet frame (which is world-aligned)."""
        data.qpos[address] = angle
        mujoco.mj_forward(model, data)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom]) != door:
                continue
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            vertices = model.mesh_vert[start : start + int(model.mesh_vertnum[mesh])]
            world = (
                vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            )
            local = world - data.xpos[cabinet]
            lo, hi = np.minimum(lo, local.min(0)), np.maximum(hi, local.max(0))
        return lo, hi

    # The default open door is swung clear of the cavity opening, which spans
    # cabinet-local x in [-0.172, 0.094].
    _, open_hi = door_bounds(float(model.qpos0[address]))
    assert open_hi[0] < -0.172

    # At the closed stop the door spans the opening in front of the cabinet.
    closed_lo, closed_hi = door_bounds(float(model.jnt_range[hinge][1]))
    assert closed_lo[0] < -0.172 and closed_hi[0] > 0.094
    assert -0.12 < closed_lo[1] and closed_hi[1] < 0.0

    # qpos=0 restores the source-authored door pose: identity in the cabinet.
    data.qpos[address] = 0.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[door], data.xpos[cabinet], atol=1.0e-6)
    np.testing.assert_allclose(data.xquat[door], [1.0, 0.0, 0.0, 0.0], atol=1.0e-6)


def test_sweet_potato_and_target_contract() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    potato = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato")
    freejoint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "sweet_potato_joint")
    assert int(model.jnt_bodyid[freejoint]) == potato
    _id(model, mujoco.mjtObj.mjOBJ_SITE, "sweet_potato_site")
    visual = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "sweet_potato_visual")
    collision = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "sweet_potato_collision")
    assert model.geom_contype[visual] == 0
    assert model.geom_conaffinity[visual] == 0
    assert model.geom_contype[collision] != 0
    assert model.geom_dataid[collision] == model.geom_dataid[visual]
    np.testing.assert_allclose(model.body_mass[potato], 0.2, atol=1.0e-9)
    # Inertia comes from the convex hull: the scan's raw enclosed volume is
    # an order of magnitude too small to be a physical solid.
    assert np.all(model.body_inertia[potato] > 1.0e-5)

    cabinet = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave")
    target = _id(model, mujoco.mjtObj.mjOBJ_BODY, "microwave_target")
    assert int(model.body_parentid[target]) == cabinet
    _id(model, mujoco.mjtObj.mjOBJ_SITE, "microwave_target_site")
    pad = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "microwave_target_pad")
    assert model.geom_contype[pad] == 0
    assert model.geom_conaffinity[pad] == 0


def test_grasp_frames_are_world_aligned_on_the_jaw_section() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    potato = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato")
    grasp = _id(model, mujoco.mjtObj.mjOBJ_SITE, "sweet_potato_grasp_site")
    assert int(model.site_bodyid[grasp]) == potato
    # In the settled pose the grasp site is world-aligned, 40 mm behind the
    # tuber centre along its long axis (world +Y) at the centre height.
    np.testing.assert_allclose(
        data.site_xmat[grasp].reshape(3, 3), np.eye(3), atol=1e-6
    )
    np.testing.assert_allclose(
        data.site_xpos[grasp] - data.xpos[potato], [-0.0034, -0.04, 0.0], atol=1e-6
    )
    # The static grasp frame starts on that site and has no physical presence.
    frame = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato_grasp_frame")
    assert int(model.body_parentid[frame]) == 0
    assert int(model.body_jntnum[frame]) == 0
    assert int(model.body_geomnum[frame]) == 0
    np.testing.assert_allclose(data.xpos[frame], data.site_xpos[grasp], atol=1e-6)
    np.testing.assert_allclose(data.xmat[frame].reshape(3, 3), np.eye(3), atol=1e-6)


def test_authored_sweet_potato_pose_is_at_rest() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    model.opt.timestep = 1.0 / 1200.0
    data = mujoco.MjData(model)
    potato = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato")
    mujoco.mj_forward(model, data)
    start = data.xpos[potato].copy()
    for _ in range(1200):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.xpos[potato] - start) < 5.0e-4


def test_released_sweet_potato_settles_on_the_target() -> None:
    model = mujoco.MjModel.from_xml_path(str(_SCENE))
    model.opt.timestep = 1.0 / 1200.0
    data = mujoco.MjData(model)
    potato = _id(model, mujoco.mjtObj.mjOBJ_BODY, "sweet_potato")
    target = _id(model, mujoco.mjtObj.mjOBJ_SITE, "microwave_target_site")
    freejoint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "sweet_potato_joint")
    address = int(model.jnt_qposadr[freejoint])
    mujoco.mj_forward(model, data)
    # Drop the tuber in its counter rest attitude 6 mm above the target.
    data.qpos[address : address + 3] = data.site_xpos[target] + [0.0, 0.0, 0.006]
    for _ in range(2400):
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.xpos[potato] - data.site_xpos[target]) < 0.01


def test_host_can_be_compiled_with_the_umi_operator_layer() -> None:
    config = SceneConfig(
        base=_SCENE,
        layers=(MjcfLayerConfig(path=_ROBOT_XML, role="operator"),),
    )
    artifact = SceneComposer().compile(config)
    assert "umi_interface" in artifact.xml
    assert _SCENE.resolve() in artifact.dependencies
    assert _ROBOT_XML.resolve() in artifact.dependencies

    model = load_composed_scene(config, artifact)
    assert model.nq > 8
    _id(model, mujoco.mjtObj.mjOBJ_BODY, "umi_interface")
    _id(model, mujoco.mjtObj.mjOBJ_SITE, "eef_pose")
