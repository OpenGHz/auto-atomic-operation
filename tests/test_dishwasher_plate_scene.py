"""Regression tests for the migrated Dishwasher031 and plate2 scene assets."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest

from auto_atom.scene_composition import (
    MjcfLayerConfig,
    SceneConfig,
    load_composed_scene,
)


_ROOT = Path(__file__).resolve().parents[1]
_SCENE_ROOT = _ROOT / "assets/xmls/scenes/dishwasher_plate"
_SCENE_FILE = "demo.xml"
_COLLISION_ROOT = _ROOT / "assets/collision/dishwasher_plate"
_ROBOT_XML = _ROOT / "assets/xmls/robots/robotiq.xml"

_ASSET_DIGESTS = {
    "assets/meshes/dishwasher_plate/dishwasher031/Body001.obj": (
        "c108d7bac6641ad2d02c384137bbadefc6c35ff4cc22c63714141ed554c0934f"
    ),
    "assets/meshes/dishwasher_plate/dishwasher031/door.obj": (
        "71367e965b5a325e3537a4b4f9a73bdeaff621bf39ac58a9e4aa8dcbe44f5ce7"
    ),
    "assets/meshes/dishwasher_plate/dishwasher031/button_lock.obj": (
        "7a16903533b6e65ff55688458857830003c21fb3855b5cfe5679c2401cec0201"
    ),
    "assets/meshes/dishwasher_plate/dishwasher031/button_power.obj": (
        "278ee26ee2bcc96a8949e12e0e140e3f8afdc0a4312ded0a2dbe3e4a5e4c3192"
    ),
    "assets/meshes/dishwasher_plate/dishwasher031/rack0.obj": (
        "985c9ae19ecc54944c125ad56bea5cecb5441673847a4837340936725f2b1c6c"
    ),
    "assets/meshes/dishwasher_plate/dishwasher031/rack1.obj": (
        "5204f9392cb5418b3e8ddf1920632619bdffe21f134a35cbcac5fc71f4e87dde"
    ),
    "assets/meshes/dishwasher_plate/dishwasher031/T_BC001.png": (
        "78f413b3a878102a210d843eb4a7c0d024b99219064a1c991648b5ae397dfca0"
    ),
    "assets/meshes/dishwasher_plate/plate2/plate2.obj": (
        "960f4113d5a9e6b123b836026f04889c45e30429ae4dda6bfc564f68e5757f93"
    ),
}

_DOOR_OPEN = 0.6035987755982988
_BUTTON_SPRINGREF = -0.0025

_MECHANISM_JOINTS = {
    "dishwasher_door_joint": (
        mujoco.mjtJoint.mjJNT_HINGE,
        "dishwasher_door",
        [1.0, 0.0, 0.0],
        [0.0, _DOOR_OPEN],
        _DOOR_OPEN,
    ),
    "dishwasher_button_lock_joint": (
        mujoco.mjtJoint.mjJNT_SLIDE,
        "dishwasher_button_lock",
        [0.0, 0.0, -1.0],
        [0.0, 0.002],
        0.0,
    ),
    "dishwasher_button_power_joint": (
        mujoco.mjtJoint.mjJNT_SLIDE,
        "dishwasher_button_power",
        [0.0, 0.0, -1.0],
        [0.0, 0.002],
        0.0,
    ),
    "dishwasher_rack0_joint": (
        mujoco.mjtJoint.mjJNT_SLIDE,
        "dishwasher_rack0",
        [0.0, -1.0, 0.0],
        [0.0, 0.33],
        0.0,
    ),
    "dishwasher_rack1_joint": (
        mujoco.mjtJoint.mjJNT_SLIDE,
        "dishwasher_rack1",
        [0.0, -1.0, 0.0],
        [0.0, 0.33],
        0.33,
    ),
}

_MECHANISM_ACTUATORS = {
    "dishwasher_door_actuator": ("dishwasher_door_joint", 50.0),
    "dishwasher_rack0_actuator": ("dishwasher_rack0_joint", 50.0),
    "dishwasher_rack1_actuator": ("dishwasher_rack1_joint", 50.0),
    "dishwasher_button_lock_actuator": ("dishwasher_button_lock_joint", 0.5),
    "dishwasher_button_power_actuator": ("dishwasher_button_power_joint", 0.5),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type.name}: {name}"
    return int(object_id)


def _assert_mechanism_contract(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    """The host retains one canonical, movable dishwasher mechanism."""

    dishwasher = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher")
    door = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher_door")
    assert model.body_parentid[door] == dishwasher
    for body_name in ("dishwasher_rack0", "dishwasher_rack1"):
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert model.body_parentid[body] == dishwasher
    for body_name in ("dishwasher_button_lock", "dishwasher_button_power"):
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert model.body_parentid[body] == door

    for joint_name, (
        kind,
        body_name,
        axis,
        limits,
        initial,
    ) in _MECHANISM_JOINTS.items():
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        qpos_address = int(model.jnt_qposadr[joint])
        assert model.jnt_type[joint] == kind
        assert model.jnt_bodyid[joint] == body
        assert model.jnt_limited[joint]
        np.testing.assert_allclose(model.jnt_axis[joint], axis, atol=1.0e-12)
        np.testing.assert_allclose(model.jnt_range[joint], limits, atol=1.0e-12)
        assert model.qpos0[qpos_address] == pytest.approx(initial)
        assert data.qpos[qpos_address] == pytest.approx(initial)

    # The upstream springref=-1 drives both buttons through their lower limit.
    # AAO uses a one-stroke-scale preload that holds the released stop against
    # gravity when the door is open.
    for joint_name in (
        "dishwasher_button_lock_joint",
        "dishwasher_button_power_joint",
    ):
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos_address = int(model.jnt_qposadr[joint])
        assert model.qpos_spring[qpos_address] == pytest.approx(_BUTTON_SPRINGREF)

    for actuator_name, (joint_name, gear) in _MECHANISM_ACTUATORS.items():
        actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT
        assert model.actuator_trnid[actuator, 0] == joint
        assert model.actuator_dyntype[actuator] == mujoco.mjtDyn.mjDYN_NONE
        assert model.actuator_gaintype[actuator] == mujoco.mjtGain.mjGAIN_FIXED
        assert model.actuator_biastype[actuator] == mujoco.mjtBias.mjBIAS_NONE
        assert model.actuator_gear[actuator, 0] == pytest.approx(gear)
        assert model.actuator_gainprm[actuator, 0] == pytest.approx(1.0)
        assert not model.actuator_ctrllimited[actuator]


def _direct_child_geoms(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.flatnonzero(model.geom_bodyid == body)


@pytest.mark.parametrize("relative_path, expected", _ASSET_DIGESTS.items())
def test_migrated_visual_payload_is_byte_identical(
    relative_path: str,
    expected: str,
) -> None:
    """Canonical copies retain the audited source bytes."""

    assert _sha256(_ROOT / relative_path) == expected


def test_demo_references_one_canonical_mechanism_definition() -> None:
    """The canonical demo consumes the shared dishwasher articulation XML."""

    expected_includes = {
        "includes/dishwasher031_assets.xml",
        "includes/dishwasher031_common.xml",
        "includes/dishwasher031_rack1_articulation.xml",
        "includes/dishwasher031_actuators.xml",
    }
    root = ET.parse(_SCENE_ROOT / _SCENE_FILE).getroot()
    includes = {node.attrib["file"] for node in root.findall(".//include")}
    assert expected_includes <= includes
    assert not any(
        joint.attrib.get("name", "").startswith("dishwasher_")
        for joint in root.findall(".//joint")
    )
    assert not any(
        motor.attrib.get("name", "").startswith("dishwasher_")
        for motor in root.findall(".//motor")
    )

    common = ET.parse(_SCENE_ROOT / "includes/dishwasher031_common.xml").getroot()
    rack1 = ET.parse(
        _SCENE_ROOT / "includes/dishwasher031_rack1_articulation.xml"
    ).getroot()
    actuators = ET.parse(_SCENE_ROOT / "includes/dishwasher031_actuators.xml").getroot()
    common_joints = {joint.attrib["name"] for joint in common.findall(".//joint")}
    rack1_joints = {joint.attrib["name"] for joint in rack1.findall(".//joint")}
    actuator_names = {motor.attrib["name"] for motor in actuators.findall(".//motor")}
    assert common_joints == set(_MECHANISM_JOINTS) - {"dishwasher_rack1_joint"}
    assert rack1_joints == {"dishwasher_rack1_joint"}
    assert actuator_names == set(_MECHANISM_ACTUATORS)


def test_robotless_host_and_final_composed_scene_load() -> None:
    """Validate the host and the public AAO host-plus-robot load path."""

    scene_path = _SCENE_ROOT / _SCENE_FILE
    host = mujoco.MjModel.from_xml_path(str(scene_path))
    host_data = mujoco.MjData(host)
    mujoco.mj_forward(host, host_data)
    assert host.nu == 5
    assert host.nkey == 0
    assert host.njnt == 6
    assert host.nq == 12
    assert host.nv == 11
    _assert_mechanism_contract(host, host_data)

    model = load_composed_scene(
        SceneConfig(
            base=scene_path,
            layers=(MjcfLayerConfig(path=_ROBOT_XML),),
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Five effort motors belong to the host; the reusable Robotiq position
    # actuator is appended by name. PlaceGen's three-axis gantry remains absent.
    assert model.nu == 6
    _assert_mechanism_contract(model, data)
    fingers_actuator = _id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        "fingers_actuator",
    )
    fingers_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
    assert fingers_actuator == 5
    assert model.actuator_trnid[fingers_actuator, 0] == fingers_joint
    for name in ("gantry_x", "gantry_y", "gantry_z"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) == -1

    plate_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate2_joint")
    assert model.jnt_type[plate_joint] == mujoco.mjtJoint.mjJNT_FREE

    door_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher_door")
    rack_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher_rack1")
    np.testing.assert_allclose(
        data.xpos[door_body],
        [0.0, -0.303386998368, 0.087427061032],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        data.xquat[door_body],
        [0.954803187366, 0.297238748139, 0.0, 0.0],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(data.xpos[rack_body], [0.0, -0.33, 0.0])

    target_site = _id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "dishwasher_rack1_target_site",
    )
    np.testing.assert_allclose(
        data.site_xpos[target_site],
        [0.024907, -0.3034355, 0.223671074],
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        data.site_xmat[target_site].reshape(3, 3),
        np.eye(3),
        atol=1.0e-12,
    )

    plate_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate2_collision")
    plate_axis = data.geom_xmat[plate_geom].reshape(3, 3)[:, 2]
    np.testing.assert_allclose(plate_axis, [0.0, 0.0, 1.0], atol=1.0e-12)

    wire_names = [
        name
        for geom_id in range(model.ngeom)
        if (
            (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id))
            and name.startswith("dishwasher_rack1_wire_")
        )
    ]
    assert len(wire_names) == 255

    for geom_name in (
        "dishwasher_body_visual_geom",
        "dishwasher_door_visual_geom",
        "dishwasher_button_lock_visual_geom",
        "dishwasher_button_power_visual_geom",
        "dishwasher_rack0_visual_geom",
        "dishwasher_rack1_visual_geom",
        "plate2_visual",
    ):
        geom_id = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert model.geom_contype[geom_id] == 0
        assert model.geom_conaffinity[geom_id] == 0

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.geom_xpos).all()


def test_effort_motors_apply_force_to_their_named_joints() -> None:
    """Each restored motor has a live, correctly geared joint transmission."""

    model = mujoco.MjModel.from_xml_path(str(_SCENE_ROOT / _SCENE_FILE))
    for actuator_name, (joint_name, gear) in _MECHANISM_ACTUATORS.items():
        data = mujoco.MjData(model)
        actuator = _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        dof_address = int(model.jnt_dofadr[joint])
        data.ctrl[actuator] = 0.2
        mujoco.mj_forward(model, data)
        assert data.actuator_force[actuator] == pytest.approx(0.2)
        assert data.qfrc_actuator[dof_address] == pytest.approx(0.2 * gear)


def test_articulated_visual_collision_and_target_frames_move_together() -> None:
    """Moving a mechanism joint cannot leave a world-space collision ghost."""

    model = mujoco.MjModel.from_xml_path(str(_SCENE_ROOT / _SCENE_FILE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    door_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher_door")
    door_geoms = _direct_child_geoms(model, "dishwasher_door")
    assert door_geoms.size == 5
    assert _id(model, mujoco.mjtObj.mjOBJ_GEOM, "dishwasher_door_visual_geom") in (
        door_geoms
    )
    door_positions = data.geom_xpos[door_geoms].copy()
    door_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "dishwasher_door_joint")
    data.qpos[model.jnt_qposadr[door_joint]] = 0.0
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.xpos[door_body], [0.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(
        data.xquat[door_body],
        [1.0, 0.0, 0.0, 0.0],
        atol=1.0e-12,
    )
    door_motion = np.linalg.norm(
        data.geom_xpos[door_geoms] - door_positions,
        axis=1,
    )
    assert door_motion.max() > 0.1

    for body_name, visual_name, collision_name in (
        (
            "dishwasher_button_lock",
            "dishwasher_button_lock_visual_geom",
            "dishwasher_button_lock_collision",
        ),
        (
            "dishwasher_button_power",
            "dishwasher_button_power_visual_geom",
            "dishwasher_button_power_collision",
        ),
    ):
        button_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        for geom_name in (visual_name, collision_name):
            geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            assert model.geom_bodyid[geom] == button_body

    # Reset the door before auditing the rack-only translation.
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    rack_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher_rack1")
    target_body = _id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "dishwasher_rack1_target",
    )
    assert model.body_parentid[target_body] == rack_body
    rack_geoms = _direct_child_geoms(model, "dishwasher_rack1")
    assert rack_geoms.size > 1
    rack_positions = data.geom_xpos[rack_geoms].copy()
    rack_rotations = data.geom_xmat[rack_geoms].copy()
    target_site = _id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "dishwasher_rack1_target_site",
    )
    target_position = data.site_xpos[target_site].copy()
    rack_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "dishwasher_rack1_joint")
    data.qpos[model.jnt_qposadr[rack_joint]] = 0.0
    mujoco.mj_forward(model, data)
    expected_delta = np.array([0.0, 0.33, 0.0])
    np.testing.assert_allclose(
        data.geom_xpos[rack_geoms] - rack_positions,
        np.broadcast_to(expected_delta, rack_positions.shape),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(data.geom_xmat[rack_geoms], rack_rotations, atol=1.0e-12)
    np.testing.assert_allclose(
        data.site_xpos[target_site] - target_position,
        expected_delta,
        atol=1.0e-12,
    )

    rack0_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "dishwasher_rack0")
    rack0_geoms = _direct_child_geoms(model, "dishwasher_rack0")
    assert rack0_geoms.size == 11
    assert np.all(model.geom_bodyid[rack0_geoms] == rack0_body)

    wire_geoms = np.array(
        [
            geom
            for geom in rack_geoms
            if (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom)) or ""
            ).startswith("dishwasher_rack1_wire_")
        ]
    )
    assert wire_geoms.size == 255


def test_ten_second_zero_control_mechanism_is_finite_and_nearly_stationary() -> None:
    """The default placement setup remains usable before joint locks exist."""

    model = mujoco.MjModel.from_xml_path(str(_SCENE_ROOT / _SCENE_FILE))
    data = mujoco.MjData(model)
    initial = data.qpos[:5].copy()
    for _ in range(5_000):
        mujoco.mj_step(model, data)

    assert np.isfinite(data.qpos).all()
    assert np.isfinite(data.qvel).all()
    np.testing.assert_allclose(data.qpos[:5], initial, atol=1.0e-3, rtol=0.0)
    for joint_name in (
        "dishwasher_button_lock_joint",
        "dishwasher_button_power_joint",
    ):
        joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qpos = data.qpos[model.jnt_qposadr[joint]]
        assert -1.0e-5 <= qpos <= 1.0e-5


def test_collision_manifests_bind_the_migrated_payload() -> None:
    """Generated planner resources still bind the exact canonical meshes."""

    proxy_path = _COLLISION_ROOT / "dishwasher031_rack1_wire_proxy.json"
    policy = json.loads(
        (_COLLISION_ROOT / "dishwasher031_rack1_vertical_policy.json").read_text()
    )
    proxy = json.loads(proxy_path.read_text())
    assert policy["proxy_sha256"] == _sha256(proxy_path)
    assert (
        proxy["source"]["sha256"]
        == _ASSET_DIGESTS["assets/meshes/dishwasher_plate/dishwasher031/rack1.obj"]
    )

    cover = json.loads((_COLLISION_ROOT / "plate2_vertical_cylinder.json").read_text())
    centers = _COLLISION_ROOT / cover["centers_resource"]
    assert cover["centers_sha256"] == _sha256(centers)
    assert (
        cover["source_mesh_sha256"]
        == _ASSET_DIGESTS["assets/meshes/dishwasher_plate/plate2/plate2.obj"]
    )
    assert cover["sphere_count"] == 388
