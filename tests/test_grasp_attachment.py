"""Grasp attachments: welds that hold a verified grasp target in the gripper."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from auto_atom.backend.mjc.mujoco_backend import (
    MujocoControlConfig,
    MujocoGraspConfig,
    MujocoObjectHandler,
    MujocoOperatorHandler,
    _validate_grasp_attachments,
)
from auto_atom.basis.mjc.mujoco_basis import MujocoBasis
from auto_atom.config.env_config import EnvConfig, GraspAttachmentConfig
from auto_atom.config.motion import EefControlConfig
from auto_atom.config.task import AutoAtomConfig
from auto_atom.execution_config import (
    declare_grasp_attachments,
    prepare_task_config_for_instantiation,
)
from auto_atom.runtime import ControlSignal
from auto_atom.scene_composition import (
    GraspWeldElementSpec,
    SceneConfig,
    create_grasp_welds,
    grasp_weld_name,
)

_ROOT = Path(__file__).resolve().parents[1]

# A gripper on three slide joints, and a free cube resting beside it. The
# gripper frame site sits on a child of the joint body, so a weld must resolve
# the site to its body.
_SCENE = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="gripper" pos="0 0 0.3">
      <joint name="gx" type="slide" axis="1 0 0" damping="50"/>
      <joint name="gy" type="slide" axis="0 1 0" damping="50"/>
      <joint name="gz" type="slide" axis="0 0 1" damping="50"/>
      <geom type="box" size="0.02 0.02 0.02" mass="1" contype="0" conaffinity="0"/>
      <body name="hand" pos="0 0 -0.05">
        <geom type="box" size="0.01 0.01 0.01" mass="0.1" contype="0" conaffinity="0"/>
        <site name="eef_pose"/>
      </body>
    </body>
    <body name="cube" pos="0.1 0 0.025" quat="0.9238795 0 0 0.3826834">
      <freejoint/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.2"/>
    </body>
  </worldbody>
  <actuator>
    <position name="ax" joint="gx" kp="2000"/>
    <position name="ay" joint="gy" kp="2000"/>
    <position name="az" joint="gz" kp="2000"/>
  </actuator>
</mujoco>
"""


def _scene(tmp_path: Path) -> SceneConfig:
    path = tmp_path / "scene.xml"
    path.write_text(_SCENE)
    return SceneConfig(base=path)


def _weld(frame: str = "eef_pose", object_name: str = "cube") -> GraspWeldElementSpec:
    return GraspAttachmentConfig(
        operator="arm", object=object_name, frame=frame
    ).to_weld_element()


def _basis(tmp_path: Path, frame: str = "eef_pose") -> MujocoBasis:
    return MujocoBasis(
        EnvConfig(
            scene=_scene(tmp_path),
            grasp_attachments=(
                GraspAttachmentConfig(operator="arm", object="cube", frame=frame),
            ),
        )
    )


def _body(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def _cube_in_hand(basis: MujocoBasis) -> tuple[np.ndarray, np.ndarray]:
    model, data = basis.model, basis.data
    mujoco.mj_kinematics(model, data)
    hand, cube = _body(model, "hand"), _body(model, "cube")
    rotation = data.xmat[hand].reshape(3, 3)
    position = rotation.T @ (data.xpos[cube] - data.xpos[hand])
    return position, rotation.T @ data.xmat[cube].reshape(3, 3)


# ----------------------------------------------------------------------
# Scene composition
# ----------------------------------------------------------------------


def test_weld_is_inactive_and_joins_the_frame_body_to_the_object() -> None:
    spec = mujoco.MjSpec.from_string(_SCENE)
    create_grasp_welds(spec, [_weld()])
    model = spec.compile()

    equality = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, grasp_weld_name("arm", "cube")
    )
    assert equality >= 0
    assert int(model.eq_type[equality]) == int(mujoco.mjtEq.mjEQ_WELD)
    assert not model.eq_active0[equality]
    # The site resolves to the body it sits on.
    assert int(model.eq_obj1id[equality]) == _body(model, "hand")
    assert int(model.eq_obj2id[equality]) == _body(model, "cube")
    np.testing.assert_allclose(model.eq_solref[equality], [0.004, 1.0])


def test_weld_prefers_the_gaussian_splatting_body() -> None:
    xml = _SCENE.replace(
        '<body name="cube" ',
        '<body name="cube_gs" pos="0 0.2 0.025"><freejoint/>'
        '<geom type="sphere" size="0.01"/></body>\n    <body name="cube" ',
    )
    spec = mujoco.MjSpec.from_string(xml)
    create_grasp_welds(spec, [_weld()])
    model = spec.compile()

    equality = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_EQUALITY, grasp_weld_name("arm", "cube")
    )
    assert int(model.eq_obj2id[equality]) == _body(model, "cube_gs")


@pytest.mark.parametrize(
    ("weld", "message"),
    [
        pytest.param(_weld(frame="nope"), "gripper frame 'nope'", id="frame"),
        pytest.param(_weld(object_name="nope"), "object 'nope'", id="object"),
    ],
)
def test_missing_weld_elements_fail_while_the_scene_is_built(weld, message) -> None:
    spec = mujoco.MjSpec.from_string(_SCENE)
    with pytest.raises(ValueError, match=message):
        create_grasp_welds(spec, [weld])


# ----------------------------------------------------------------------
# Runtime weld
# ----------------------------------------------------------------------


def test_attach_holds_the_object_where_it_is_until_released(tmp_path: Path) -> None:
    basis = _basis(tmp_path)
    model, data = basis.model, basis.data
    for _ in range(200):
        mujoco.mj_step(model, data)
    before = data.xpos[_body(model, "cube")].copy()
    grasp_position, grasp_rotation = _cube_in_hand(basis)

    basis.attach_grasped_object("arm", "cube")
    assert basis.attached_object("arm") == "cube"
    mujoco.mj_step(model, data)
    # Attaching does not move the object.
    np.testing.assert_allclose(data.xpos[_body(model, "cube")], before, atol=1e-6)

    # Lift and shift the gripper: the cube rides along, held at its grasp pose.
    data.ctrl[:] = [0.15, -0.1, 0.2]
    for _ in range(1500):
        mujoco.mj_step(model, data)
    position, rotation = _cube_in_hand(basis)
    assert float(data.xpos[_body(model, "cube")][2]) > 0.2
    np.testing.assert_allclose(position, grasp_position, atol=2e-4)
    np.testing.assert_allclose(rotation, grasp_rotation, atol=2e-3)

    assert basis.release_grasped_object("arm") == "cube"
    assert basis.attached_object("arm") is None
    for _ in range(500):
        mujoco.mj_step(model, data)
    # Released, it falls back to the floor.
    assert float(data.xpos[_body(model, "cube")][2]) < 0.05


def test_reset_releases_every_weld(tmp_path: Path) -> None:
    basis = _basis(tmp_path)
    basis.attach_grasped_object("arm", "cube")
    equality = mujoco.mj_name2id(
        basis.model, mujoco.mjtObj.mjOBJ_EQUALITY, grasp_weld_name("arm", "cube")
    )
    assert basis.data.eq_active[equality]

    basis.reset()

    assert basis.attached_object("arm") is None
    assert not basis.data.eq_active[equality]


def test_attaching_an_undeclared_object_is_an_error(tmp_path: Path) -> None:
    basis = _basis(tmp_path)
    with pytest.raises(KeyError, match="No grasp attachment weld"):
        basis.attach_grasped_object("arm", "plate")


# ----------------------------------------------------------------------
# Operator handler
# ----------------------------------------------------------------------


class _BatchOf:
    """One basis presented as the batched env an operator handler drives."""

    batch_size = 1

    def __init__(self, single: MujocoBasis) -> None:
        single._op_eef_qidx = {"arm": [0]}
        self.envs = [single]
        self.commands: list[float] = []

    def register_operator(self, *_args: object, **_kwargs: object) -> None:
        pass

    def step(self, ctrl: np.ndarray, env_mask: np.ndarray) -> None:
        self.commands.append(float(ctrl[0][0]))
        self.envs[0].data.ctrl[:] = ctrl[0]


def _handler(batch: _BatchOf, **grasp: object) -> MujocoOperatorHandler:
    handler = MujocoOperatorHandler(
        operator_name="arm",
        env=batch,
        eef_ctrl_index=0,
        eef_open_value=0.0,
        eef_close_value=0.3,
        control=MujocoControlConfig(
            grasp=MujocoGraspConfig(attach=True, settle_steps=1, **grasp)
        ),
    )
    # Stand in for the finger-contact check: the target counts as squeezed.
    handler._check_grasp_conditions = lambda _env, _target: {
        "left_contact": True,
        "right_contact": True,
        "lateral_ok": True,
    }
    return handler


def test_a_verified_grasp_attaches_and_opening_releases(tmp_path: Path) -> None:
    batch = _BatchOf(_basis(tmp_path))
    handler = _handler(batch, pre_release_settle_steps=2)
    cube = MujocoObjectHandler(name="cube", env=batch, body_name="cube")

    result = handler.control_eef(EefControlConfig(close=True), cube)
    assert result.signals[0] == ControlSignal.REACHED
    assert result.details[0]["attached_object"] == "cube"
    assert batch.envs[0].attached_object("arm") == "cube"
    # While welded, the object counts as grasped whatever its contacts show.
    handler._check_grasp_conditions = lambda _env, _target: {
        "left_contact": False,
        "right_contact": False,
        "lateral_ok": False,
    }
    assert handler._is_target_grasped(0, cube)

    # The weld survives the pre-release hold, then opening releases it.
    opening = EefControlConfig(close=False)
    handler.control_eef(opening, None)
    handler.control_eef(opening, None)
    assert batch.envs[0].attached_object("arm") == "cube"
    handler.control_eef(opening, None)
    assert batch.envs[0].attached_object("arm") is None
    assert not handler._is_target_grasped(0, cube)


def test_without_attach_a_grasp_leaves_the_object_free(tmp_path: Path) -> None:
    batch = _BatchOf(_basis(tmp_path))
    handler = _handler(batch)
    handler.control.grasp.attach = False
    cube = MujocoObjectHandler(name="cube", env=batch, body_name="cube")

    result = handler.control_eef(EefControlConfig(close=True), cube)

    assert result.signals[0] == ControlSignal.REACHED
    assert "attached_object" not in result.details[0]
    assert batch.envs[0].attached_object("arm") is None


# ----------------------------------------------------------------------
# Build-time validation
# ----------------------------------------------------------------------


def _stage_task(object_name: str = "cube") -> AutoAtomConfig:
    return AutoAtomConfig.model_validate(
        {
            "env_name": "grasp_attachment",
            "stages": [
                {
                    "object": object_name,
                    "operation": "pick",
                    "operator": "arm",
                    "param": {"eef": {"close": True}},
                }
            ],
        }
    )


def test_validation_accepts_welds_on_the_eef_body(tmp_path: Path) -> None:
    batch = _BatchOf(_basis(tmp_path))
    _validate_grasp_attachments(_stage_task(), {"arm": _handler(batch)}, batch)


def test_validation_rejects_a_weld_off_the_eef_body(tmp_path: Path) -> None:
    batch = _BatchOf(_basis(tmp_path, frame="gripper"))
    with pytest.raises(ValueError, match="is on body 'gripper'.*on body 'hand'"):
        _validate_grasp_attachments(_stage_task(), {"arm": _handler(batch)}, batch)


def test_validation_requires_a_weld_for_every_picked_object(tmp_path: Path) -> None:
    batch = _BatchOf(_basis(tmp_path))
    with pytest.raises(ValueError, match="no grasp attachment weld for 'plate'"):
        _validate_grasp_attachments(
            _stage_task("plate"), {"arm": _handler(batch)}, batch
        )


# ----------------------------------------------------------------------
# Config preparation
# ----------------------------------------------------------------------


def _config(attach: object, **extra: object):
    node = {
        "execution": {"mode": "physical"},
        "env": {"operators": {"arm": {"pose_site": "tool"}}},
        "task_operators": {"arm": {"control": {"grasp": {"attach": attach}}}},
        "task": {
            "stages": [
                {"object": "cube", "operation": "pick", "operator": "arm"},
                {"object": "shelf", "operation": "place", "operator": "arm"},
                {"object": "drawer", "operation": "pull"},
                {"object": "cube", "operation": "pick", "operator": "arm"},
            ]
        },
    }
    node.update(extra)
    return OmegaConf.create(node)


def test_attach_declares_one_weld_per_closing_stage_object() -> None:
    cfg = _config(True)
    declare_grasp_attachments(cfg)

    # Place does not close on its object; the operator-less pull resolves to
    # the only operator; the repeated pick is declared once.
    assert OmegaConf.to_container(cfg.env.grasp_attachments) == [
        {"operator": "arm", "object": "cube", "frame": "tool"},
        {"operator": "arm", "object": "drawer", "frame": "tool"},
    ]


def test_without_attach_no_weld_is_declared() -> None:
    cfg = _config(False)
    declare_grasp_attachments(cfg)
    assert OmegaConf.select(cfg, "env.grasp_attachments") is None


def test_listed_attachments_are_kept_and_not_repeated() -> None:
    cfg = _config(True)
    cfg.env.grasp_attachments = [
        {"operator": "arm", "object": "cube", "frame": "tool", "solref": [0.01, 1.0]}
    ]
    declare_grasp_attachments(cfg)
    entries = OmegaConf.to_container(cfg.env.grasp_attachments)
    assert [(entry["object"], entry.get("solref")) for entry in entries] == [
        ("cube", [0.01, 1.0]),
        ("drawer", None),
    ]


def test_object_only_execution_declares_no_weld() -> None:
    cfg = _config(True, execution={"mode": "object_only"})
    prepared = prepare_task_config_for_instantiation(cfg)
    assert OmegaConf.select(prepared, "env.grasp_attachments") is None
