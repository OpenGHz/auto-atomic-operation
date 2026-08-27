from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from auto_atom.framework import (
    AutoAtomConfig,
    EefControlConfig,
    PoseControlConfig,
    TaskFileConfig,
)
from auto_atom.backend.mjc.mujoco_backend import (
    MujocoObjectHandler,
    MujocoTaskBackend,
)
from auto_atom.runtime import (
    ContactObservation,
    ControlResult,
    ControlSignal,
    EnvProtocol,
    ObjectHandler,
    OperatorHandler,
    PoseState,
    SceneBackend,
    TaskFlowBuilder,
    TaskRunner,
)


class _ExternalEnv:
    batch_size = 1


class _ExternalObject(ObjectHandler):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pose = PoseState()

    def get_pose(self) -> PoseState:
        return self.pose

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        _ = env_mask
        self.pose = pose


class _ExternalOperator(OperatorHandler):
    @property
    def name(self) -> str:
        return "arm"

    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        _ = pose, target, env_mask
        return ControlResult.filled(1, ControlSignal.REACHED)

    def control_eef(
        self,
        eef: EefControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        _ = eef, target, env_mask
        return ControlResult.filled(1, ControlSignal.REACHED)

    def get_end_effector_pose(self) -> PoseState:
        return PoseState()

    def get_base_pose(self) -> PoseState:
        return PoseState()


class _ExternalBackend(SceneBackend):
    def __init__(self) -> None:
        self._env = _ExternalEnv()
        self._operator = _ExternalOperator()
        self._objects = {"block": _ExternalObject("block")}

    def get_env(self) -> EnvProtocol:
        return self._env

    @property
    def batch_size(self) -> int:
        return self._env.batch_size

    def setup(self, config: AutoAtomConfig) -> None:
        _ = config

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        _ = env_mask

    def teardown(self) -> None:
        pass

    def get_operator_handler(self, name: str) -> OperatorHandler:
        if name != "arm":
            raise KeyError(name)
        return self._operator

    def get_object_handler(self, name: str) -> Optional[ObjectHandler]:
        if not name:
            return None
        return self._objects[name]

    def is_element_rigidly_attached_to_object(
        self,
        element_name: str,
        object_name: str,
        env_index: int = 0,
    ) -> bool:
        _ = env_index
        if element_name not in self._objects:
            raise KeyError(element_name)
        if object_name not in self._objects:
            raise KeyError(object_name)
        return element_name == object_name

    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        _ = operator_name, object_name
        return np.asarray([False], dtype=bool)

    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        _ = operator_name
        return np.asarray([False], dtype=bool)

    def get_grasped_object_name(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[str]:
        _ = operator_name, env_index
        return None

    def is_operator_contacting(
        self,
        operator_name: str,
        object_name: str,
    ) -> np.ndarray:
        _ = operator_name, object_name
        return np.asarray([False], dtype=bool)

    def get_operator_contacts(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[list[ContactObservation]]:
        _ = operator_name, env_index
        return None


def test_eef_grasp_requirement_only_accepts_close_commands() -> None:
    assert EefControlConfig(close=True, require_grasp=True).require_grasp
    with pytest.raises(ValueError, match="require_grasp=true requires close=true"):
        EefControlConfig(close=False, require_grasp=True)


def _build_external_backend(task, operators) -> _ExternalBackend:
    _ = task, operators
    return _ExternalBackend()


def test_minimal_external_backend_satisfies_runner_contract() -> None:
    config = TaskFileConfig.model_validate(
        {
            "backend": _build_external_backend,
            "task": {"env_name": "external", "stages": []},
            "task_operators": {},
        }
    )
    runner = TaskRunner().from_config(config)
    try:
        assert runner.batch_size == 1
        assert isinstance(runner.get_env(), _ExternalEnv)
        assert runner._require_context().backend.is_element_rigidly_attached_to_object(
            "block",
            "block",
        )
        update = runner.reset()
        assert update.done.tolist() == [False]
        update = runner.update()
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        runner.close()


def test_mujoco_object_handler_rejects_wrong_mask_shape() -> None:
    handler = MujocoObjectHandler(
        name="block",
        env=SimpleNamespace(batch_size=2, envs=[]),
        body_name="block",
    )

    with np.testing.assert_raises_regex(ValueError, r"env_mask must have shape \(2,\)"):
        handler.set_pose(PoseState(), env_mask=np.asarray([True]))


def test_mujoco_backend_observes_named_operator_contacts() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="operator_root">
              <freejoint/>
              <geom name="finger_geom" type="sphere" size="0.1"
                    margin="0.02" gap="0.018"/>
            </body>
            <body name="door_panel">
              <freejoint/>
              <geom name="door_geom" type="box" size="0.1 0.1 0.1"/>
            </body>
            <body name="nearby_panel" pos="0.225 0 0">
              <freejoint/>
              <geom name="nearby_geom" type="box" size="0.1 0.1 0.1"
                    margin="0.02" gap="0.018"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(
        batch_size=1,
        envs=[SimpleNamespace(model=model, data=data)],
    )
    operator_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "operator_root",
    )
    operator = SimpleNamespace(
        get_operator_body_ids=lambda _model: frozenset({operator_body_id}),
    )
    door = MujocoObjectHandler(name="door", env=env, body_name="door_panel")
    nearby = MujocoObjectHandler(name="nearby", env=env, body_name="nearby_panel")
    backend = MujocoTaskBackend(
        env=env,
        operator_handlers={"arm": operator},
        object_handlers={"door": door, "nearby": nearby},
    )
    assert any(int(data.contact[index].efc_address) < 0 for index in range(data.ncon))

    contacts = backend.get_operator_contacts("arm", 0)

    assert contacts is not None
    assert len(contacts) == 1
    contact = contacts[0]
    assert contact.operator_body == "operator_root"
    assert contact.operator_geom == "finger_geom"
    assert contact.other_body == "door_panel"
    assert contact.other_geom == "door_geom"
    assert contact.signed_distance_m < 0.0
    assert contact.penetration_depth_m > 0.0
    assert len(contact.position_world_m) == 3
    assert contact.normal_force_n is not None
    assert np.isfinite(contact.normal_force_n)
    assert contact.normal_force_n >= 0.0
    assert backend.is_operator_contacting("arm", "door").tolist() == [True]
    assert backend.is_operator_contacting("arm", "nearby").tolist() == [False]


def test_mujoco_backend_validates_rigid_named_frame_ownership() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="plate">
              <freejoint/>
              <geom name="plate_geom" type="box" size="0.1 0.1 0.01"/>
              <site name="plate_site"/>
              <body name="fixed_child" pos="0.02 0 0">
                <geom type="sphere" size="0.005"/>
                <site name="fixed_site"/>
              </body>
              <body name="hinged_child" pos="0.04 0 0">
                <joint name="hinge" type="hinge"/>
                <geom type="sphere" size="0.005"/>
                <site name="hinged_site"/>
              </body>
            </body>
            <body name="rack" pos="0.5 0 0">
              <freejoint/>
              <geom type="sphere" size="0.01"/>
              <site name="rack_site"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    env = SimpleNamespace(
        batch_size=1,
        envs=[SimpleNamespace(model=model, data=data)],
    )
    backend = MujocoTaskBackend(
        env=env,
        operator_handlers={},
        object_handlers={
            "plate": MujocoObjectHandler(name="plate", env=env, body_name="plate"),
            "rack": MujocoObjectHandler(name="rack", env=env, body_name="rack"),
        },
    )

    assert backend.is_element_rigidly_attached_to_object("plate_site", "plate")
    assert backend.is_element_rigidly_attached_to_object("plate_geom", "plate")
    assert backend.is_element_rigidly_attached_to_object("fixed_site", "plate")
    assert not backend.is_element_rigidly_attached_to_object(
        "hinged_site",
        "plate",
    )
    assert not backend.is_element_rigidly_attached_to_object("rack_site", "plate")
    assert backend.get_element_pose("plate_geom").batch_size == 1
    with pytest.raises(KeyError, match="No site, body, geom, or joint"):
        backend.is_element_rigidly_attached_to_object("missing_site", "plate")


def test_object_handler_rejects_empty_identity() -> None:
    with np.testing.assert_raises_regex(ValueError, "non-empty string"):
        _ExternalObject("")


def test_task_flow_rejects_mismatched_operator_identity() -> None:
    backend = _ExternalBackend()
    backend._operator = SimpleNamespace(name="other")

    with np.testing.assert_raises_regex(ValueError, "mismatched identity"):
        TaskFlowBuilder._select_operator(SimpleNamespace(operator="arm"), backend)
