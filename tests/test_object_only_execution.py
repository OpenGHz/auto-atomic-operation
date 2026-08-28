"""Regression tests for the configuration-driven ``object_only`` mode.

The object-only path is intentionally tested at two seams:

* the Hydra boundary must remove operator-owned composition before an
  environment is instantiated; and
* the runner must preserve the task timeline while replacing physical EEF
  actions with logical acquire/release and bounded object motion.

These tests use the lightweight mock backend so they do not require MuJoCo or
an installed robot asset.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from auto_atom.execution_config import prepare_task_config_for_instantiation
from auto_atom.backend.mjc.mujoco_backend import MujocoObjectHandler
from auto_atom.framework import ExecutionMode, TaskFileConfig
from auto_atom.policy_eval import (
    ConfigDrivenDemoPolicy,
    PolicyActionFeedback,
    PolicyEvaluator,
)
from auto_atom.runtime import (
    ComponentRegistry,
    ControlSignal,
    PoseState,
    StageExecutionStatus,
    TaskRunner,
)
from auto_atom.utils.pose import quaternion_angular_distance


def _pose(
    position: tuple[float, float, float],
    *,
    reference: str = "world",
    controlled_frame: str | None = None,
    relative: bool = False,
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "position": list(position),
        "orientation": list(orientation),
        "reference": reference,
        "relative": relative,
    }
    if controlled_frame is not None:
        payload["controlled_frame"] = {"kind": controlled_frame}
    return payload


def _task_file(
    env_name: str,
    *,
    batch_size: int = 1,
    execution: dict[str, Any] | None = None,
    stages: list[dict[str, Any]] | None = None,
    operators: dict[str, Any] | None = None,
) -> TaskFileConfig:
    """Build a small pick/place task and register its mock environment."""

    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": batch_size},
    )
    if stages is None:
        stages = [
            {
                "name": "pick_item",
                "object": "item",
                "operation": "pick",
                "operator": "arm",
                "param": {
                    "pre_move": [_pose((0.0, 0.0, 0.3))],
                    "post_move": [_pose((0.0, 0.0, 0.4))],
                    "eef": {"close": True},
                },
            },
            {
                "name": "place_item",
                "object": "target",
                "operation": "place",
                "operator": "arm",
                "param": {
                    "pre_move": [
                        _pose(
                            (0.52, 0.0, 0.3),
                            controlled_frame="held_object",
                            orientation=(0.0, 0.0, 1.0, 0.0),
                        )
                    ],
                    "post_move": [_pose((0.52, 0.0, 0.5))],
                    "eef": {"close": False},
                    "placed_tolerance": {
                        "position": 0.01,
                        "orientation": 0.1,
                    },
                },
            },
        ]
    payload: dict[str, Any] = {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {"env_name": env_name, "stages": stages},
        "task_operators": operators if operators is not None else {"arm": {}},
    }
    if execution is not None:
        payload["execution"] = execution
    return TaskFileConfig.model_validate(payload)


@pytest.fixture(autouse=True)
def _clear_component_registry() -> None:
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_object_only_hydra_preparation_removes_operator_owned_composition() -> None:
    """Operator layers, cameras, and randomization never reach instantiation."""

    raw = OmegaConf.create(
        {
            "execution": {"mode": "object_only"},
            "env": {
                "scene": {
                    "layers": [
                        {"path": "scene.xml", "role": "scene"},
                        {"path": "robot.xml", "role": "operator"},
                        # Legacy paths remain recognized for old configs.
                        {"path": "assets/robots/legacy.xml"},
                    ]
                },
                "cameras": [
                    {"name": "head_cam", "role": "scene"},
                    {"name": "wrist_cam", "role": "operator"},
                    {"name": "eef_legacy_cam"},
                ],
                "enabled_sensors": ["camera", "pose", "tactile"],
                "operators": {"arm": {"name": "arm"}},
            },
            "task_operators": {"arm": {}, "second_arm": {}},
            "task": {
                "randomization": {
                    "arm": {"eef": {"x": [-0.1, 0.1]}},
                    "second_arm": {"eef": {"x": [-0.1, 0.1]}},
                    "item": {"x": [-0.01, 0.01]},
                },
                "camera_randomization": {
                    "wrist_cam": {"x": [-0.1, 0.1]},
                    "head_cam": {"x": [-0.1, 0.1]},
                },
            },
        }
    )
    original = deepcopy(OmegaConf.to_container(raw, resolve=False))

    prepared = prepare_task_config_for_instantiation(raw)

    assert [layer["path"] for layer in prepared.env.scene.layers] == ["scene.xml"]
    assert [camera["name"] for camera in prepared.env.cameras] == ["head_cam"]
    assert list(prepared.env.enabled_sensors) == ["camera"]
    assert prepared.env.operators == {}
    assert prepared.task_operators == {}
    assert set(prepared.task.randomization) == {"item"}
    assert set(prepared.task.camera_randomization) == {"head_cam"}
    # The preparation boundary must not mutate Hydra's source tree.
    assert OmegaConf.to_container(raw, resolve=False) == original


def test_physical_preparation_is_a_noop_copy() -> None:
    """The default mode keeps every operator-owned config entry intact."""

    raw = OmegaConf.create(
        {
            "execution": {"mode": "physical"},
            "env": {
                "scene": {"layers": [{"path": "robot.xml", "role": "operator"}]},
                "cameras": [{"name": "wrist_cam", "role": "operator"}],
                "enabled_sensors": ["camera", "pose"],
                "operators": {"arm": {}},
            },
            "task_operators": {"arm": {}},
            "task": {"randomization": {"arm": {"eef": {"x": [-1.0, 1.0]}}}},
        }
    )

    prepared = prepare_task_config_for_instantiation(raw)

    assert OmegaConf.to_container(prepared, resolve=False) == OmegaConf.to_container(
        raw, resolve=False
    )
    assert prepared is not raw


def test_object_only_runner_skips_eef_and_moves_object_in_bounded_steps() -> None:
    """EEF approach/retreat is inert; only held-object waypoints move the item."""

    config = _task_file(
        "object_only_bounded_motion",
        operators={},
        execution={
            "mode": "object_only",
            "object_motion": {
                "max_linear_step": 0.1,
                "max_angular_step": 0.2,
            },
        },
    )
    runner = TaskRunner().from_config(config)
    try:
        assert config.execution.mode == ExecutionMode.OBJECT_ONLY
        assert [plan.operator_name for plan in runner._plan] == [
            "object_only",
            "object_only",
        ]
        pick_actions = runner._materialize_stage_actions(runner._plan[0])
        place_actions = runner._materialize_stage_actions(runner._plan[1])
        assert [action.kind for action in pick_actions] == [
            "noop",
            "object_acquire",
            "noop",
        ]
        assert [action.kind for action in place_actions] == [
            "object_pose",
            "object_release",
            "noop",
        ]

        object_handler = runner._context.backend.objects["item"]
        previous = object_handler.get_pose().position[0].copy()
        previous_orientation = object_handler.get_pose().orientation[0].copy()
        update = runner.reset()
        assert update.done.tolist() == [False]
        assert update.details[0]["execution"]["mode"] == "object_only"

        movement_deltas: list[float] = []
        angular_deltas: list[float] = []
        action_events: list[tuple[str, str]] = []
        for _ in range(30):
            update = runner.update()
            current = object_handler.get_pose().position[0].copy()
            current_orientation = object_handler.get_pose().orientation[0].copy()
            movement_deltas.append(float(np.linalg.norm(current - previous)))
            angular_deltas.append(
                float(
                    quaternion_angular_distance(
                        previous_orientation,
                        current_orientation,
                    )
                )
            )
            previous = current
            previous_orientation = current_orientation
            action_events.append(
                (
                    str(update.details[0].get("action", "")),
                    str(update.details[0].get("event", "")),
                )
            )
            if bool(update.done[0]):
                break

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert update.status.tolist() == [StageExecutionStatus.SUCCEEDED]
        assert object_handler.get_pose().position[0] == pytest.approx([0.52, 0.0, 0.3])
        assert object_handler.get_pose().orientation[0] == pytest.approx(
            [0.0, 0.0, 1.0, 0.0]
        )
        assert max(movement_deltas) <= 0.1000001
        assert max(angular_deltas) <= 0.200001
        assert movement_deltas[0] == pytest.approx(0.0)
        assert any(action == "noop" for action, _event in action_events)
        assert any(event == "object_acquired" for _action, event in action_events)
        assert any(event == "object_released" for _action, event in action_events)
        assert runner._context.logical_carried_objects == {}
        assert all(
            record.details["execution_mode"] == "object_only"
            for record in runner.records
        )
    finally:
        runner.close()


def test_object_only_batch_mask_keeps_carry_and_motion_per_environment() -> None:
    """A masked update cannot move or acquire an unselected environment."""

    config = _task_file(
        "object_only_batch_mask",
        batch_size=2,
        operators={},
        execution={
            "mode": "object_only",
            "object_motion": {"max_linear_step": 0.2, "max_angular_step": 0.2},
        },
    )
    runner = TaskRunner().from_config(config)
    try:
        object_handler = runner._context.backend.objects["item"]
        object_handler.pose.position[:] = np.asarray(
            [[0.0, 0.0, 0.05], [0.1, 0.0, 0.05]],
            dtype=np.float64,
        )
        initial_env1 = object_handler.pose.position[1].copy()
        runner.reset()

        # First two selected updates consume pick's inert approach and acquire.
        runner.update(np.asarray([True, False]))
        runner.update(np.asarray([True, False]))
        assert runner._context.get_logical_carried_object(0) == "item"
        assert runner._context.get_logical_carried_object(1) is None
        np.testing.assert_allclose(object_handler.pose.position[1], initial_env1)

        # Finish env 0 while env 1 remains untouched.
        for _ in range(40):
            update = runner.update(np.asarray([True, False]))
            assert runner._context.get_logical_carried_object(1) is None
            np.testing.assert_allclose(object_handler.pose.position[1], initial_env1)
            if bool(update.done[0]):
                break
        assert runner._env_states[0].done is True
        assert runner._env_states[1].done is False

        # Resetting only env 0 clears only its logical carry state.
        runner.reset(np.asarray([True, False]))
        assert runner._context.get_logical_carried_object(0) is None
        assert runner._context.get_logical_carried_object(1) is None
        assert runner._env_states[0].done is False
        assert runner._env_states[1].done is False
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("stages", "match"),
    [
        pytest.param(
            [
                {
                    "name": "move",
                    "object": "item",
                    "operation": "move",
                    "operator": "arm",
                    "param": {"pre_move": [_pose((0.2, 0.0, 0.3))]},
                }
            ],
            "supports only pick/place",
            id="unsupported-operation",
        ),
        pytest.param(
            [
                {
                    "name": "place_without_held_waypoint",
                    "object": "target",
                    "operation": "place",
                    "operator": "arm",
                    "param": {
                        "pre_move": [_pose((0.2, 0.0, 0.3))],
                        "placed_tolerance": {
                            "position": 0.01,
                            "orientation": 0.1,
                        },
                    },
                }
            ],
            "requires at least one pre_move waypoint",
            id="place-without-held-waypoint",
        ),
        pytest.param(
            [
                {
                    "name": "pick",
                    "object": "item",
                    "operation": "pick",
                    "operator": "arm",
                    "param": {"pre_move": [_pose((0.2, 0.0, 0.3))]},
                },
                {
                    "name": "place_with_base_reference",
                    "object": "target",
                    "operation": "place",
                    "operator": "arm",
                    "param": {
                        "pre_move": [
                            _pose(
                                (0.2, 0.0, 0.3),
                                reference="base",
                                controlled_frame="held_object",
                            )
                        ],
                        "placed_tolerance": {
                            "position": 0.01,
                            "orientation": 0.1,
                        },
                    },
                },
            ],
            "cannot resolve held-object reference",
            id="operator-dependent-held-reference",
        ),
    ],
)
def test_object_only_invalid_plans_fail_before_backend_setup(
    stages: list[dict[str, Any]],
    match: str,
) -> None:
    config = _task_file(
        "object_only_invalid_plan",
        execution={"mode": "object_only"},
        stages=stages,
    )
    runner = TaskRunner()

    with pytest.raises(ValueError, match=match):
        runner.from_config(config)


def test_config_driven_policy_evaluator_uses_object_only_transport() -> None:
    """The policy adapter follows the same logical transport semantics."""

    config = _task_file(
        "object_only_policy_evaluator",
        operators={},
        execution={
            "mode": "object_only",
            "object_motion": {"max_linear_step": 0.2, "max_angular_step": 0.2},
        },
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        config
    )
    try:
        policy.reset()
        update = evaluator.reset()
        observed_actions: list[str] = []
        for _ in range(40):
            action = policy.act({}, update, evaluator)
            env_action = action.env_actions[0]
            assert env_action is not None
            observed_actions.append(env_action.action.kind)
            update = evaluator.update(action)
            if bool(update.done[0]):
                break

        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert observed_actions[:2] == ["noop", "object_acquire"]
        assert "object_pose" in observed_actions
        assert "object_release" in observed_actions
        assert all(
            record.details["evaluation_mode"] == "policy"
            for record in evaluator.records
        )
    finally:
        evaluator.close()


def test_external_policy_object_only_pick_does_not_probe_missing_operator() -> None:
    """External object-only feedback can acquire a logical object without an EEF."""

    config = _task_file(
        "object_only_external_pick",
        operators={},
        execution={"mode": "object_only"},
        stages=[
            {
                "name": "pick_item",
                "object": "item",
                "operation": "pick",
                "operator": "arm",
                "param": {"pre_move": [_pose((0.0, 0.0, 0.3))]},
            }
        ],
    )

    def apply_feedback(context: Any, action: Any, env_mask: Any = None):
        assert action is None
        assert np.asarray(env_mask, dtype=bool).tolist() == [True]
        context.acquire_logical_object(0, "item")
        return PolicyActionFeedback(
            signals=[ControlSignal.REACHED],
            details=[{"event": "external_logical_pick"}],
            stage_action_sequence_done=[True],
        )

    evaluator = PolicyEvaluator(action_applier=apply_feedback).from_config(config)
    try:
        evaluator.reset()
        update = evaluator.update(None)
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert evaluator.records[0].details["evaluation_mode"] == "policy"
    finally:
        evaluator.close()


def test_external_policy_object_only_place_uses_logical_postcondition() -> None:
    """An external policy can complete PLACE without an EEF completion goal."""
    config = _task_file(
        "object_only_external_place",
        operators={},
        execution={"mode": "object_only"},
        stages=[
            {
                "name": "pick_item",
                "object": "item",
                "operation": "pick",
                "operator": "arm",
                "param": {},
            },
            {
                "name": "place_item",
                "object": "target",
                "operation": "place",
                "operator": "arm",
                "param": {
                    "pre_move": [
                        _pose(
                            (0.2, 0.0, 0.3),
                            controlled_frame="held_object",
                        )
                    ],
                    "eef": {"close": False},
                    "placed_tolerance": {
                        "position": 0.001,
                        "orientation": 0.01,
                    },
                },
            },
        ],
    )
    calls = {"count": 0}

    def apply_feedback(context: Any, action: Any, env_mask: Any = None):
        assert action is None
        assert np.asarray(env_mask, dtype=bool).tolist() == [True]
        calls["count"] += 1
        if calls["count"] == 1:
            context.acquire_logical_object(0, "item")
            event = "external_logical_pick"
        else:
            context.apply_object_pose(
                "item",
                PoseState(position=(0.2, 0.0, 0.3)),
                env_mask=np.asarray([True]),
            )
            context.release_logical_object(0)
            event = "external_logical_place"
        return PolicyActionFeedback(
            signals=[ControlSignal.REACHED],
            details=[{"event": event}],
            stage_action_sequence_done=[True],
        )

    evaluator = PolicyEvaluator(action_applier=apply_feedback).from_config(config)
    try:
        evaluator.reset()
        update = evaluator.update(None)
        assert update.done.tolist() == [False]
        update = evaluator.update(None)
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
        assert [record.status for record in evaluator.records] == [
            StageExecutionStatus.SUCCEEDED,
            StageExecutionStatus.SUCCEEDED,
        ]
    finally:
        evaluator.close()


def test_external_policy_object_only_composes_relative_waypoints_before_callback() -> (
    None
):
    """Relative held-object goals are snapshotted before an external move."""
    config = _task_file(
        "object_only_external_relative_place",
        operators={},
        execution={"mode": "object_only"},
        stages=[
            {
                "name": "pick_item",
                "object": "item",
                "operation": "pick",
                "operator": "arm",
                "param": {},
            },
            {
                "name": "place_item",
                "object": "target",
                "operation": "place",
                "operator": "arm",
                "param": {
                    "pre_move": [
                        _pose(
                            (0.05, 0.0, 0.0),
                            controlled_frame="held_object",
                            relative=True,
                        ),
                        _pose(
                            (0.05, 0.0, 0.0),
                            controlled_frame="held_object",
                            relative=True,
                        ),
                    ],
                    "eef": {"close": False},
                    "placed_tolerance": {
                        "position": 0.001,
                        "orientation": None,
                    },
                },
            },
        ],
    )
    calls = {"count": 0}

    def apply_feedback(context: Any, action: Any, env_mask: Any = None):
        assert action is None
        assert np.asarray(env_mask, dtype=bool).tolist() == [True]
        calls["count"] += 1
        if calls["count"] == 1:
            context.acquire_logical_object(0, "item")
        else:
            # Mock objects start at x=0.4; two relative +0.05 waypoints
            # therefore compose to x=0.5.
            context.apply_object_pose(
                "item",
                PoseState(position=(0.5, -0.1, 0.05)),
                env_mask=np.asarray([True]),
            )
            context.release_logical_object(0)
        return PolicyActionFeedback(
            signals=[ControlSignal.REACHED],
            details=[{}],
            stage_action_sequence_done=[True],
        )

    evaluator = PolicyEvaluator(action_applier=apply_feedback).from_config(config)
    try:
        evaluator.reset()
        evaluator.update(None)
        update = evaluator.update(None)
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        evaluator.close()


def test_static_nested_mujoco_object_pose_is_interpreted_in_world_frame() -> None:
    """Kinematic object transport converts world goals to parent-local MJCF poses."""

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="parent" pos="1 2 0.3"
                  quat="0.9238795325 0 0 0.3826834324">
              <body name="child" pos="0.1 0 0.05">
                <geom type="sphere" size="0.01"/>
              </body>
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
    handler = MujocoObjectHandler(name="child", env=env, body_name="child")

    target_position = np.asarray([1.75, 2.25, 0.6], dtype=np.float64)
    target_orientation_xyzw = np.asarray(
        [0.0, 0.0, 0.2588190451, 0.9659258263],
        dtype=np.float64,
    )
    handler.set_pose(
        # PoseState accepts either one pose or a batched row.
        PoseState(
            position=target_position,
            orientation=target_orientation_xyzw,
        )
    )

    child_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "child")
    np.testing.assert_allclose(data.xpos[child_id], target_position, atol=1e-7)
    # MuJoCo stores quaternions as wxyz; the public contract is xyzw.
    np.testing.assert_allclose(
        data.xquat[child_id],
        [
            target_orientation_xyzw[3],
            target_orientation_xyzw[0],
            target_orientation_xyzw[1],
            target_orientation_xyzw[2],
        ],
        atol=1e-7,
    )


def test_object_only_default_placed_tolerance_is_released_only() -> None:
    """The schema's all-null placement tolerance must not become a float error."""
    stages = [
        {
            "name": "pick_item",
            "object": "item",
            "operation": "pick",
            "operator": "arm",
            "param": {"eef": {"close": True}},
        },
        {
            "name": "place_item",
            "object": "target",
            "operation": "place",
            "operator": "arm",
            "param": {
                "pre_move": [
                    _pose(
                        (0.2, 0.0, 0.3),
                        controlled_frame="held_object",
                    )
                ],
                "eef": {"close": False},
                # Omitted on purpose: PlacedToleranceConfig defaults to
                # [None, None, None] for both dimensions.
            },
        },
    ]
    config = _task_file(
        "object_only_default_tolerance",
        execution={"mode": "object_only"},
        stages=stages,
    )
    runner = TaskRunner().from_config(config)
    try:
        update = runner.reset()
        for _ in range(20):
            if bool(update.done[0]):
                break
            update = runner.update()
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        runner.close()


def test_object_only_nullable_axis_tolerance_is_checked_per_axis() -> None:
    """List-shaped placement tolerances retain their nullable-axis semantics."""
    stages = [
        {
            "name": "pick_item",
            "object": "item",
            "operation": "pick",
            "operator": "arm",
            "param": {"eef": {"close": True}},
        },
        {
            "name": "place_item",
            "object": "target",
            "operation": "place",
            "operator": "arm",
            "param": {
                "pre_move": [
                    _pose(
                        (0.2, 0.0, 0.3),
                        controlled_frame="held_object",
                    )
                ],
                "eef": {"close": False},
                "placed_tolerance": {
                    "position": [0.001, None, None],
                    "orientation": [None, None, None],
                },
            },
        },
    ]
    config = _task_file(
        "object_only_nullable_tolerance",
        execution={"mode": "object_only"},
        stages=stages,
    )
    runner = TaskRunner().from_config(config)
    try:
        update = runner.reset()
        for _ in range(20):
            if bool(update.done[0]):
                break
            update = runner.update()
        assert update.done.tolist() == [True]
        assert update.success.tolist() == [True]
    finally:
        runner.close()
