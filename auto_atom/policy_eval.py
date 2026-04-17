"""Policy-driven evaluator that reuses the task runner's result data classes."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .framework import (
    Operation,
    OperationConditionType,
    PoseControlConfig,
    TaskFileConfig,
)
from .runtime import (
    ControlSignal,
    ExecutionContext,
    ExecutionRecord,
    ExecutionSummary,
    ObjectHandler,
    OperatorHandler,
    PrimitiveAction,
    SceneBackend,
    StageExecutionPlan,
    StageExecutionStatus,
    TaskFlowBuilder,
    TaskRunner,
    TaskUpdate,
    _build_execution_summary,
    _check_stage_condition,
    _collect_reset_details,
    _EnvRuntimeState,
    load_task_file,
)
from .utils.pose import PoseState


@dataclass
class _PolicyStageState:
    plan: StageExecutionPlan
    operator: OperatorHandler
    target: Optional[ObjectHandler]
    initial_object_pose: Optional[PoseState] = None
    completion_pose: Optional[PoseControlConfig] = None


@dataclass
class ConfigDrivenEnvAction:
    """One primitive action emitted by the config-driven demo policy."""

    stage_index: int
    action: PrimitiveAction


@dataclass
class ConfigDrivenPolicyAction:
    """Batched primitive actions for all envs in one evaluator tick."""

    env_actions: List[Optional[ConfigDrivenEnvAction]]


@dataclass
class PolicyActionFeedback:
    """Optional per-env execution feedback returned by a policy action applier."""

    signals: List[Optional[ControlSignal]]
    details: List[Dict[str, Any]]
    stage_action_sequence_done: List[bool]


class ConfigDrivenDemoPolicy:
    """Policy that replays the same config-derived primitive actions as TaskRunner."""

    def __init__(self, builder: Optional[TaskFlowBuilder] = None) -> None:
        self.builder = builder or TaskFlowBuilder()
        self._cached_stage_indices: List[Optional[int]] = []
        self._cached_actions: List[List[PrimitiveAction]] = []
        self._action_indices: List[int] = []

    def reset(self) -> None:
        self._cached_stage_indices = []
        self._cached_actions = []
        self._action_indices = []

    def act(
        self,
        observation: Any,
        update: TaskUpdate,
        evaluator: "PolicyEvaluator",
    ) -> ConfigDrivenPolicyAction:
        _ = observation
        batch_size = evaluator.batch_size
        self._ensure_capacity(batch_size)
        env_actions: List[Optional[ConfigDrivenEnvAction]] = []

        for env_index in range(batch_size):
            if bool(update.done[env_index]):
                env_actions.append(None)
                continue

            stage_index = int(update.stage_index[env_index])
            if stage_index < 0:
                env_actions.append(None)
                continue

            actions = self._get_stage_actions(env_index, stage_index, evaluator)
            action_index = min(self._action_indices[env_index], len(actions) - 1)
            env_actions.append(
                ConfigDrivenEnvAction(
                    stage_index=stage_index,
                    action=actions[action_index],
                )
            )

        return ConfigDrivenPolicyAction(env_actions=env_actions)

    def action_applier(
        self,
        context: ExecutionContext,
        action: Any,
        env_mask: Optional[np.ndarray] = None,
    ) -> PolicyActionFeedback:
        if action is None:
            return PolicyActionFeedback(
                signals=[None for _ in range(context.backend.batch_size)],
                details=[{} for _ in range(context.backend.batch_size)],
                stage_action_sequence_done=[
                    False for _ in range(context.backend.batch_size)
                ],
            )
        if not isinstance(action, ConfigDrivenPolicyAction):
            raise TypeError(
                "ConfigDrivenDemoPolicy.action_applier expects "
                "ConfigDrivenPolicyAction."
            )
        mask = self._normalize_mask(context.backend.batch_size, env_mask)
        if len(action.env_actions) != context.backend.batch_size:
            raise ValueError(
                "ConfigDrivenPolicyAction batch size does not match backend batch size."
            )
        feedback = PolicyActionFeedback(
            signals=[None for _ in range(context.backend.batch_size)],
            details=[{} for _ in range(context.backend.batch_size)],
            stage_action_sequence_done=[
                False for _ in range(context.backend.batch_size)
            ],
        )

        for env_index, env_action in enumerate(action.env_actions):
            if not mask[env_index] or env_action is None:
                continue
            plan = context.plan[env_action.stage_index]
            operator = context.backend.get_operator_handler(plan.operator_name)
            target = context.backend.get_object_handler(plan.stage.object)
            result = TaskRunner._run_action(
                env_index=env_index,
                operator=operator,
                action=env_action.action,
                target=target,
                backend=context.backend,
                env_mask=self._single_env_mask(context.backend.batch_size, env_index),
            )
            if result.signals[env_index] == ControlSignal.REACHED:
                actions = self._cached_actions[env_index]
                next_index = self._action_indices[env_index] + 1
                feedback.stage_action_sequence_done[env_index] = next_index >= len(
                    actions
                )
                self._action_indices[env_index] = min(
                    next_index, max(len(actions) - 1, 0)
                )
            feedback.signals[env_index] = result.signals[env_index]
            feedback.details[env_index] = dict(result.details[env_index])
        return feedback

    def _ensure_capacity(self, batch_size: int) -> None:
        if len(self._cached_stage_indices) == batch_size:
            return
        self._cached_stage_indices = [None for _ in range(batch_size)]
        self._cached_actions = [[] for _ in range(batch_size)]
        self._action_indices = [0 for _ in range(batch_size)]

    def _get_stage_actions(
        self,
        env_index: int,
        stage_index: int,
        evaluator: "PolicyEvaluator",
    ) -> List[PrimitiveAction]:
        if self._cached_stage_indices[env_index] != stage_index:
            plan = evaluator.stage_plans[stage_index]
            self._cached_actions[env_index] = deepcopy(
                self.builder.build_actions(
                    plan.stage,
                    plan.last_orientation_before,
                )[0]
            )
            self._cached_stage_indices[env_index] = stage_index
            self._action_indices[env_index] = 0
        return self._cached_actions[env_index]

    @staticmethod
    def _normalize_mask(
        batch_size: int,
        env_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        if env_mask is None:
            return np.ones(batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if len(mask) != batch_size:
            raise ValueError(
                f"env_mask must have shape ({batch_size},), got {mask.shape}"
            )
        return mask

    @staticmethod
    def _single_env_mask(batch_size: int, env_index: int) -> np.ndarray:
        mask = np.zeros(batch_size, dtype=bool)
        mask[env_index] = True
        return mask


class PolicyEvaluator:
    """Evaluate external policy rollouts with the same update/record types as TaskRunner."""

    def __init__(
        self,
        *,
        action_applier: Callable[[ExecutionContext, Any, Optional[np.ndarray]], None],
        observation_getter: Optional[Callable[[ExecutionContext], Any]] = None,
        builder: Optional[TaskFlowBuilder] = None,
        default_position_tolerance: float = 0.01,
        default_orientation_tolerance: float = 0.08,
    ) -> None:
        self.action_applier = action_applier
        self.observation_getter = observation_getter
        self.builder = builder or TaskFlowBuilder()
        self.default_position_tolerance = float(default_position_tolerance)
        self.default_orientation_tolerance = float(default_orientation_tolerance)
        self._context: Optional[ExecutionContext] = None
        self._plan: List[StageExecutionPlan] = []
        self._records: List[ExecutionRecord] = []
        self._env_states: List[_EnvRuntimeState] = []
        self._policy_states: List[Optional[_PolicyStageState]] = []
        self._has_reset: np.ndarray = np.zeros(0, dtype=bool)
        self._sim_lock: threading.Lock = threading.Lock()
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop_event: Optional[threading.Event] = None
        self._pending_sim_loop_freq: float = 0.0

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)

    @property
    def batch_size(self) -> int:
        return self._require_context().backend.batch_size

    @property
    def stage_plans(self) -> List[StageExecutionPlan]:
        return list(self._plan)

    def from_yaml(
        self, path: str | Path, sim_loop_frequency: float = 0.0
    ) -> "PolicyEvaluator":
        return self.from_config(load_task_file(path), sim_loop_frequency)

    def from_config(
        self, config: TaskFileConfig, sim_loop_frequency: float = 0.0
    ) -> "PolicyEvaluator":
        backend = config.backend(config.task, config.task_operators)
        if not isinstance(backend, SceneBackend):
            raise TypeError(
                "Task file backend must be an instantiated SceneBackend. "
                f"Got {type(backend).__name__}."
            )
        self._context = ExecutionContext(
            config=config.task,
            backend=backend,
            task_file=config,
        )
        self._plan = self.builder.build(self._context)
        self._context.plan = self._plan
        self._context.backend.setup(self._context.config)
        self._env_states = [_EnvRuntimeState() for _ in range(backend.batch_size)]
        self._policy_states = [None for _ in range(backend.batch_size)]
        self._has_reset = np.zeros(backend.batch_size, dtype=bool)
        self._records = []
        self._pending_sim_loop_freq = float(sim_loop_frequency)
        return self

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        with self._sim_lock:
            context.backend.reset(mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._env_states[env_index] = _EnvRuntimeState()
                self._env_states[env_index].latest_details = _collect_reset_details(
                    env_index, context
                )
                self._policy_states[env_index] = None
        self._has_reset[mask] = True
        # self._set_interest_focus()
        if self._pending_sim_loop_freq > 0 and not self.sim_loop_running:
            self.start_sim_loop(frequency=self._pending_sim_loop_freq)
        return self._build_task_update()

    def get_observation(self) -> Any:
        context = self._require_context()
        with self._sim_lock:
            if self.observation_getter is not None:
                return self.observation_getter(context)
            backend = context.backend
            env = getattr(backend, "env", None)
            if env is not None and hasattr(env, "capture_observation"):
                return env.capture_observation()
        raise RuntimeError(
            "No observation_getter was provided and backend.env does not expose "
            "capture_observation()."
        )

    def update(self, action: Any, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        self._validate_update_mask(mask)
        with self._sim_lock:
            feedback = self.action_applier(context, action, mask)
            for env_index, enabled in enumerate(mask):
                if not enabled or self._env_states[env_index].done:
                    continue
                self._update_env(
                    env_index,
                    self._env_states[env_index],
                    context,
                    action_feedback=feedback,
                )
        # self._set_interest_focus()
        return self._build_task_update()

    def close(self) -> None:
        self.stop_sim_loop()
        if self._context is None:
            return
        self._context.backend.teardown()
        self._context = None
        self._plan = []
        self._records = []
        self._env_states = []
        self._policy_states = []
        self._has_reset = np.zeros(0, dtype=bool)

    # ------------------------------------------------------------------
    # Background simulation loop
    # ------------------------------------------------------------------

    @property
    def sim_lock(self) -> threading.Lock:
        """Lock held by the background sim loop during each physics step.

        Acquire this when reading/writing simulation state from the main
        thread while the loop is running.
        """
        return self._sim_lock

    def start_sim_loop(self, frequency: float = 60.0) -> None:
        """Start a background thread that advances physics at *frequency* Hz.

        Each iteration calls ``backend.env.update()`` which steps MuJoCo
        physics using whatever control values (``data.ctrl``) were last set,
        so the simulation keeps running without explicit ``update()`` calls.

        Parameters
        ----------
        frequency:
            Target update rate in Hz (default 60).
        """
        if self._sim_thread is not None:
            raise RuntimeError("Simulation loop is already running.")
        context = self._require_context()
        env = context.backend.env
        if not hasattr(env, "update"):
            raise RuntimeError(
                "Backend env does not expose an update() method. "
                "Background simulation loop is not supported for this backend."
            )
        self._sim_stop_event = threading.Event()
        self._sim_thread = threading.Thread(
            target=self._sim_loop_fn,
            args=(env, frequency),
            daemon=True,
        )
        self._sim_thread.start()

    def stop_sim_loop(self) -> None:
        """Stop the background simulation loop (no-op if not running)."""
        if self._sim_thread is None:
            return
        assert self._sim_stop_event is not None
        self._sim_stop_event.set()
        self._sim_thread.join()
        self._sim_thread = None
        self._sim_stop_event = None

    @property
    def sim_loop_running(self) -> bool:
        """Return whether the background simulation loop is active."""
        return self._sim_thread is not None and self._sim_thread.is_alive()

    def _sim_loop_fn(self, env: Any, frequency: float) -> None:
        """Background thread target: step physics at the requested rate."""
        assert self._sim_stop_event is not None
        dt = 1.0 / frequency
        while not self._sim_stop_event.is_set():
            t0 = time.monotonic()
            with self._sim_lock:
                env.update()
            elapsed = time.monotonic() - t0
            remaining = dt - elapsed
            if remaining > 0:
                self._sim_stop_event.wait(remaining)

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
        elapsed_time_sec: float = 0.0,
    ) -> ExecutionSummary:
        return _build_execution_summary(
            update=update or self._build_task_update(),
            records=self._records,
            total_stages=len(self._plan),
            max_updates=max_updates,
            updates_used=updates_used,
            elapsed_time_sec=elapsed_time_sec,
        )

    def _update_env(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
        action_feedback: Optional[PolicyActionFeedback] = None,
    ) -> None:
        if state.stage_cursor >= len(self._plan):
            state.done = True
            state.success = True
            state.latest_status = StageExecutionStatus.SUCCEEDED
            state.phase = None
            state.phase_step = None
            return

        policy_state = self._policy_states[env_index]
        if policy_state is None:
            plan = self._plan[state.stage_cursor]
            failure = (
                None
                if plan.stage.operation == Operation.PULL
                else _check_stage_condition(
                    env_index=env_index,
                    context=context,
                    plan=plan,
                    condition_type=OperationConditionType.PERFORM,
                )
            )
            if failure is not None:
                self._record_failure(env_index, plan, failure)
                state.done = True
                state.success = False
                state.latest_status = StageExecutionStatus.FAILED
                state.latest_details = failure
                return
            policy_state = self._start_stage(env_index, context, plan)
            self._policy_states[env_index] = policy_state
            state.latest_status = StageExecutionStatus.RUNNING
            state.phase = "policy"
            state.phase_step = None

        assert policy_state is not None
        if action_feedback is not None:
            signal = action_feedback.signals[env_index]
            if signal in {ControlSignal.TIMED_OUT, ControlSignal.FAILED}:
                details = {
                    "env_index": env_index,
                    **action_feedback.details[env_index],
                }
                failure = TaskRunner._build_action_failure_details(
                    policy_state.plan,
                    details,
                    signal,
                )
                self._record_failure(env_index, policy_state.plan, failure)
                self._policy_states[env_index] = None
                state.done = True
                state.success = False
                state.latest_status = StageExecutionStatus.FAILED
                state.latest_details = failure
                state.phase = None
                state.phase_step = None
                return
            if not action_feedback.stage_action_sequence_done[env_index]:
                state.latest_status = StageExecutionStatus.RUNNING
                state.latest_details = {
                    "event": "stage_action_sequence_running",
                    "env_index": env_index,
                    "evaluation_mode": "policy",
                    **action_feedback.details[env_index],
                }
                state.phase = "policy"
                state.phase_step = None
                return

        success_failure = _check_stage_condition(
            env_index=env_index,
            context=context,
            plan=policy_state.plan,
            condition_type=OperationConditionType.SUCCESS,
            initial_pose=policy_state.initial_object_pose,
            completion_pose=policy_state.completion_pose,
        )
        success_details = {
            "event": "stage_success_condition_met",
            "env_index": env_index,
            "evaluation_mode": "policy",
            "operator": policy_state.plan.operator_name,
            "operation": policy_state.plan.stage.operation.value,
            "target_object": policy_state.plan.stage.object,
        }

        if success_failure is None:
            self._records.append(
                ExecutionRecord(
                    env_index=env_index,
                    stage_index=policy_state.plan.stage_index,
                    stage_name=policy_state.plan.stage_name,
                    operator=policy_state.plan.operator_name,
                    operation=policy_state.plan.stage.operation.value,
                    target_object=policy_state.plan.stage.object,
                    blocking=policy_state.plan.stage.blocking,
                    status=StageExecutionStatus.SUCCEEDED,
                    details=success_details,
                )
            )
            state.stage_cursor += 1
            self._policy_states[env_index] = None
            state.latest_status = StageExecutionStatus.SUCCEEDED
            state.latest_details = success_details
            state.phase = None
            state.phase_step = None
            if state.stage_cursor >= len(self._plan):
                state.done = True
                state.success = True
            else:
                state.success = False
            return

        if (
            action_feedback is not None
            and action_feedback.stage_action_sequence_done[env_index]
        ):
            self._record_failure(env_index, policy_state.plan, success_failure)
            self._policy_states[env_index] = None
            state.done = True
            state.success = False
            state.latest_status = StageExecutionStatus.FAILED
            state.latest_details = success_failure
            state.phase = None
            state.phase_step = None
            return

        state.latest_status = StageExecutionStatus.RUNNING
        state.latest_details = {
            "event": "stage_success_condition_pending",
            "env_index": env_index,
            "evaluation_mode": "policy",
            **success_failure,
        }
        state.phase = "policy"
        state.phase_step = None

    def _start_stage(
        self,
        env_index: int,
        context: ExecutionContext,
        plan: StageExecutionPlan,
    ) -> _PolicyStageState:
        operator = context.backend.get_operator_handler(plan.operator_name)
        target = context.backend.get_object_handler(plan.stage.object)
        initial_object_pose: Optional[PoseState] = None
        if target is not None:
            initial_object_pose = target.get_pose().select(env_index)

        actions, _ = self.builder.build_actions(
            plan.stage, plan.last_orientation_before
        )
        completion_action = actions[-1] if actions else None
        completion_pose: Optional[PoseControlConfig] = None
        if completion_action is not None and completion_action.kind == "pose":
            completion_pose = _resolve_policy_completion_pose(
                env_index=env_index,
                operator=operator,
                target=target,
                backend=context.backend,
                action=completion_action,
                reference_site=plan.stage.site,
            )
        return _PolicyStageState(
            plan=plan,
            operator=operator,
            target=target,
            initial_object_pose=initial_object_pose,
            completion_pose=completion_pose,
        )

    def _record_failure(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        details: Dict[str, Any],
    ) -> None:
        self._records.append(
            ExecutionRecord(
                env_index=env_index,
                stage_index=plan.stage_index,
                stage_name=plan.stage_name,
                operator=plan.operator_name,
                operation=plan.stage.operation.value,
                target_object=plan.stage.object,
                blocking=plan.stage.blocking,
                status=StageExecutionStatus.FAILED,
                details=details,
            )
        )

    def _set_interest_focus(self) -> None:
        context = self._require_context()
        object_names: List[str] = []
        operation_names: List[str] = []
        for env_index, state in enumerate(self._env_states):
            policy_state = self._policy_states[env_index]
            if state.done or policy_state is None:
                object_names.append("")
                operation_names.append("")
            else:
                object_names.append(policy_state.plan.stage.object)
                operation_names.append(policy_state.plan.stage.operation.value)
        context.backend.set_interest_objects_and_operations(
            object_names, operation_names
        )

    def _build_task_update(self) -> TaskUpdate:
        stage_index: List[int] = []
        stage_name: List[str] = []
        status: List[StageExecutionStatus] = []
        done: List[bool] = []
        success: List[bool] = []
        details: List[Dict[str, Any]] = []
        phase: List[Optional[str]] = []
        phase_step: List[int] = []
        for env_index, state in enumerate(self._env_states):
            policy_state = self._policy_states[env_index]
            if policy_state is not None:
                stage_index.append(policy_state.plan.stage_index)
                stage_name.append(policy_state.plan.stage_name)
            elif state.stage_cursor < len(self._plan):
                stage_index.append(self._plan[state.stage_cursor].stage_index)
                stage_name.append(self._plan[state.stage_cursor].stage_name)
            else:
                stage_index.append(-1)
                stage_name.append("")
            status.append(state.latest_status)
            done.append(state.done)
            success.append(state.success)
            details.append(dict(state.latest_details))
            phase.append(state.phase)
            phase_step.append(-1 if state.phase_step is None else state.phase_step)
        return TaskUpdate(
            stage_index=np.asarray(stage_index, dtype=np.int64),
            stage_name=stage_name,
            status=np.asarray(status, dtype=object),
            done=np.asarray(done, dtype=bool),
            success=np.asarray(success, dtype=bool),
            details=details,
            phase=phase,
            phase_step=np.asarray(phase_step, dtype=np.int64),
        )

    def _normalize_mask(self, env_mask: Optional[np.ndarray]) -> np.ndarray:
        batch_size = self._require_context().backend.batch_size
        if env_mask is None:
            return np.ones(batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if len(mask) != batch_size:
            raise ValueError(
                f"env_mask must have shape ({batch_size},), got {mask.shape}"
            )
        return mask

    def _validate_update_mask(self, env_mask: np.ndarray) -> None:
        missing = np.flatnonzero(env_mask & ~self._has_reset)
        if missing.size == 0:
            return
        missing_str = ", ".join(str(int(i)) for i in missing.tolist())
        raise RuntimeError(
            "PolicyEvaluator.update() was called for envs that have not been reset: "
            f"[{missing_str}]. Call reset(env_mask=...) for those envs first."
        )

    def _require_context(self) -> ExecutionContext:
        if self._context is None:
            raise RuntimeError(
                "PolicyEvaluator is not initialized. Call from_yaml() first."
            )
        return self._context


def _resolve_policy_completion_pose(
    *,
    env_index: int,
    operator: OperatorHandler,
    target: Optional[ObjectHandler],
    backend: SceneBackend,
    action: Any,
    reference_site: Optional[str] = None,
) -> Optional[PoseControlConfig]:
    if action.pose is None:
        return None
    from .runtime import TaskRunner

    completion_action = deepcopy(action)
    return TaskRunner._resolve_pose_command(
        env_index=env_index,
        operator=operator,
        pose=completion_action.pose,
        target=target,
        backend=backend,
        action=completion_action,
        reference_site=reference_site,
    )
