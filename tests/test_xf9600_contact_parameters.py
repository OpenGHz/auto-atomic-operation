"""Compiled-model regressions for XF9600 grasp contact parameters."""

from __future__ import annotations

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
_MOCAP_XML = _ROOT / "assets/xmls/robots/xf9600_mocap.xml"
_DISHWASHER_XML = _ROOT / "assets/xmls/scenes/dishwasher_plate/demo.xml"
_PAD_NAMES = (
    "eef_right_finger_pad_upper",
    "eef_right_finger_pad_lower",
    "eef_left_finger_pad_upper",
    "eef_left_finger_pad_lower",
)


def _id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    assert object_id >= 0, f"missing {object_type.name}: {name}"
    return int(object_id)


def test_compiled_xf9600_pad_and_mocap_weld_parameters() -> None:
    """Nested MJCF compilation retains pad friction and the tighter weld."""

    model = mujoco.MjModel.from_xml_path(str(_MOCAP_XML))

    for name in _PAD_NAMES:
        geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert model.geom_condim[geom] == 4
        assert model.geom_priority[geom] == 1
        np.testing.assert_allclose(
            model.geom_friction[geom],
            [1.0, 0.02, 0.01],
            atol=1.0e-12,
        )
        np.testing.assert_allclose(model.geom_solref[geom], [0.002, 1.0])

    mocap_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "xf9600_mocap")
    interface_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "xf9600_interface")
    welds = [
        equality
        for equality in range(model.neq)
        if (
            model.eq_type[equality] == mujoco.mjtEq.mjEQ_WELD
            and model.eq_obj1id[equality] == mocap_body
            and model.eq_obj2id[equality] == interface_body
        )
    ]
    assert len(welds) == 1
    np.testing.assert_allclose(model.eq_solref[welds[0]], [0.10, 1.0])


def test_plate_pad_contact_uses_xf9600_priority_parameters() -> None:
    """The actual plate-pad pair gets four contact dimensions and pad friction."""

    model = load_composed_scene(
        SceneConfig(
            base=_DISHWASHER_XML,
            layers=(MjcfLayerConfig(path=_MOCAP_XML),),
        )
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    plate = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "plate2_collision")
    pad = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "eef_right_finger_pad_upper")
    plate_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate2_joint")
    qpos_address = int(model.jnt_qposadr[plate_joint])

    # Put the real plate against the right pad's inner face with 1 mm of
    # penetration.  R_x(-90 deg) aligns the plate-cylinder axis with pad +Y.
    plate_position = data.geom_xpos[pad].copy()
    plate_position[1] -= model.geom_size[pad, 1] + model.geom_size[plate, 1] - 0.001
    data.qpos[qpos_address : qpos_address + 3] = plate_position
    data.qpos[qpos_address + 3 : qpos_address + 7] = [
        np.sqrt(0.5),
        -np.sqrt(0.5),
        0.0,
        0.0,
    ]
    mujoco.mj_forward(model, data)

    matching_contacts = [
        contact
        for contact in data.contact
        if {int(contact.geom1), int(contact.geom2)} == {plate, pad}
    ]
    assert matching_contacts, "expected a generated plate-pad contact"
    for contact in matching_contacts:
        assert contact.dim == 4
        assert contact.dist == pytest.approx(-0.001, abs=1.0e-12)
        # A 4D contact uses two sliding directions plus one torsional
        # direction.  The final two stored coefficients retain rolling values
        # even though condim=4 does not activate those directions.
        np.testing.assert_allclose(
            contact.friction,
            [1.0, 1.0, 0.02, 0.01, 0.01],
            atol=1.0e-12,
        )
