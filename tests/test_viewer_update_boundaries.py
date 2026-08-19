from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterator

import mujoco
import pytest

from auto_atom.basis.mjc import mujoco_basis
from auto_atom.basis.mjc.mujoco_basis import MujocoBasis, ViewerConfig
from auto_atom.framework import TaskFileConfig
from auto_atom.policy_eval import PolicyEvaluator
from auto_atom.runtime import ComponentRegistry, TaskRunner


@dataclass
class _FakeViewer:
    sync_calls: int = 0

    def is_running(self) -> bool:
        return True

    def sync(self) -> None:
        self.sync_calls += 1


def _minimal_mujoco_basis(*, step_delay: float = 0.03) -> MujocoBasis:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body>
              <joint type="hinge"/>
              <geom type="sphere" size="0.01"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    env = object.__new__(MujocoBasis)
    env.model = model
    env.data = mujoco.MjData(model)
    env.config = SimpleNamespace(viewer=ViewerConfig(step_delay=step_delay))
    env._ctrl_interp = False
    env._prev_ctrl = None
    env._n_substeps = 3
    env._pre_step_callbacks = []
    env._viewer_update_defer_depth = 0
    env._viewer_update_pending = False
    env._viewer = _FakeViewer()
    return env


def test_deferred_viewer_updates_preserve_physics_and_coalesce_sync_and_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dense = _minimal_mujoco_basis()
    deferred = _minimal_mujoco_basis()
    physics_steps = {id(dense.data): 0, id(deferred.data): 0}
    sleep_calls: list[float] = []
    real_mj_step = mujoco.mj_step

    def counting_mj_step(model, data) -> None:
        physics_steps[id(data)] += 1
        real_mj_step(model, data)

    monkeypatch.setattr(mujoco_basis.mujoco, "mj_step", counting_mj_step)
    monkeypatch.setattr(
        mujoco_basis.time,
        "sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    dense.update()
    dense.update()
    dense_sleep_calls = list(sleep_calls)
    sleep_calls.clear()

    with deferred.defer_viewer_updates():
        deferred.update()
        deferred.update()
        assert deferred._viewer.sync_calls == 0
        assert deferred._viewer_update_pending is True
        assert sleep_calls == []

    assert physics_steps[id(dense.data)] == 6
    assert physics_steps[id(deferred.data)] == 6
    assert deferred.data.time == pytest.approx(dense.data.time)
    assert dense._viewer.sync_calls == 2
    assert deferred._viewer.sync_calls == 1
    assert dense_sleep_calls == pytest.approx([0.03, 0.03])
    assert sleep_calls == []


def test_deferred_viewer_updates_are_nested_and_exception_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _minimal_mujoco_basis()
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        mujoco_basis.time,
        "sleep",
        lambda seconds: sleep_calls.append(float(seconds)),
    )

    with pytest.raises(RuntimeError, match="controller failed"):
        with env.defer_viewer_updates():
            with env.defer_viewer_updates():
                env.update()
                assert env._viewer_update_defer_depth == 2
                assert env._viewer.sync_calls == 0
                raise RuntimeError("controller failed")

    assert env._viewer_update_defer_depth == 0
    assert env._viewer_update_pending is False
    assert env._viewer.sync_calls == 1
    assert sleep_calls == []

    env.update()
    assert env._viewer.sync_calls == 2
    assert sleep_calls == pytest.approx([0.03])


@dataclass
class _DeferScopeRecorder:
    entries: int = 0
    exits: int = 0
    depth: int = 0
    max_depth: int = 0

    @contextmanager
    def defer(self) -> Iterator[None]:
        self.entries += 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        try:
            yield
        finally:
            self.depth -= 1
            self.exits += 1

    def clear(self) -> None:
        self.entries = 0
        self.exits = 0
        self.depth = 0
        self.max_depth = 0


def _pose(x: float) -> dict:
    return {
        "reference": "world",
        "position": [x, 0.0, 0.3],
        "orientation": [0.0, 0.0, 0.0, 1.0],
    }


def _task_payload(
    *,
    env_name: str,
    positions: tuple[float, ...] = (0.1, 0.2, 0.3),
    render_internal_updates: bool | None = None,
    update_boundary: str = "keypoint",
    interval_selection: dict | None = None,
) -> dict:
    ComponentRegistry.register_env(
        env_name,
        {"kind": "mock_env", "batch_size": 1},
    )
    execution = {"update_boundary": update_boundary}
    if render_internal_updates is not None:
        execution["render_internal_updates"] = render_internal_updates
    if interval_selection is not None:
        execution["interval_selection"] = interval_selection
    return {
        "backend": "auto_atom.mock.build_mock_backend",
        "task": {
            "env_name": env_name,
            "stages": [
                {
                    "name": "selected",
                    "object": "block",
                    "operation": "move",
                    "operator": "arm",
                    "param": {"pre_move": [_pose(position) for position in positions]},
                }
            ],
        },
        "execution": execution,
        "task_operators": {"arm": {}},
    }


def _runner(payload: dict) -> TaskRunner:
    return TaskRunner().from_config(TaskFileConfig.model_validate(payload))


def _attach_scope_recorder(runner: TaskRunner) -> _DeferScopeRecorder:
    assert runner._context is not None
    recorder = _DeferScopeRecorder()
    runner._context.backend.defer_viewer_updates = recorder.defer
    return recorder


@pytest.fixture(autouse=True)
def _clear_component_registry():
    ComponentRegistry.clear()
    yield
    ComponentRegistry.clear()


def test_interval_reset_defers_all_fast_forward_ticks_in_one_scope() -> None:
    interval = {
        "start": {"stage": "selected", "phase": "pre_move", "waypoint": 1},
        "stop": {"stage": "selected", "phase": "pre_move", "waypoint": 2},
    }
    runner = _runner(
        _task_payload(
            env_name="viewer_deferred_interval_reset",
            render_internal_updates=False,
            interval_selection=interval,
        )
    )
    recorder = _attach_scope_recorder(runner)
    try:
        update = runner.reset()

        assert recorder.entries == 1
        assert recorder.exits == 1
        assert recorder.depth == 0
        assert recorder.max_depth == 1
        assert update.phase == ["pre_move"]
        assert update.phase_step.tolist() == [1]
        assert update.details[0]["interval_selection"]["fast_forward_updates"] == 4
    finally:
        runner.close()


def test_keypoint_update_defers_all_internal_ticks_in_one_scope() -> None:
    runner = _runner(
        _task_payload(
            env_name="viewer_deferred_keypoint_update",
            render_internal_updates=False,
        )
    )
    recorder = _attach_scope_recorder(runner)
    try:
        runner.reset()
        recorder.clear()

        update = runner.update()

        assert recorder.entries == 1
        assert recorder.exits == 1
        assert recorder.depth == 0
        assert recorder.max_depth == 1
        assert update.details[0]["execution"]["event"] == "keypoint_reached"
        assert update.details[0]["execution"]["internal_updates"] == 2
    finally:
        runner.close()


def test_default_render_internal_updates_does_not_enter_deferred_viewer_scope() -> None:
    payload = _task_payload(env_name="viewer_dense_default")
    config = TaskFileConfig.model_validate(payload)
    assert config.execution.render_internal_updates is True

    runner = TaskRunner().from_config(config)
    recorder = _attach_scope_recorder(runner)
    try:
        runner.reset()
        update = runner.update()

        assert update.details[0]["execution"]["internal_updates"] == 2
        assert recorder.entries == 0
        assert recorder.exits == 0
        assert recorder.depth == 0
    finally:
        runner.close()


def test_policy_evaluator_rejects_deferred_viewer_updates() -> None:
    config = TaskFileConfig.model_validate(
        _task_payload(
            env_name="policy_deferred_viewer",
            render_internal_updates=False,
            update_boundary="control_tick",
            interval_selection=None,
        )
    )
    evaluator = PolicyEvaluator(action_applier=lambda *_args, **_kwargs: None)

    with pytest.raises(
        ValueError,
        match=(
            "execution.render_internal_updates=false is supported by "
            "TaskRunner/aao-demo only"
        ),
    ):
        evaluator.from_config(config)
