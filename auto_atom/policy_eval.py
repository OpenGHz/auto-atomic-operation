"""Policy-driven evaluator that reuses the task runner's result data classes."""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config_loader import load_task_file
from .execution_timeline import ExecutionTimeline
from .framework import (
    PoseControlConfig,
    TaskFileConfig,
    UpdateBoundary,
)
from .runtime import (
    ActiveStageState,
    ControlSignal,
    EnvProtocol,
    ExecutionContext,
    ExecutionRecord,
    ExecutionSummary,
    InfoEnvProtocol,
    ObjectHandler,
    ObservationEnvProtocol,
    OperatorHandler,
    PrimitiveAction,
    SceneBackend,
    SimulationLoopEnvProtocol,
    StageExecutionPlan,
    StageExecutionStatus,
    TaskFlowBuilder,
    TaskRunner,
    TaskUpdate,
    _build_execution_summary,
    _collect_reset_details,
    _EnvRuntimeState,
    _teardown_backend_after_initialization_failure,
    require_env_capability,
)
from .stage_execution import PolicyStageFeedback, StageExecution


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
    stage_actions: List[Optional[List[PrimitiveAction]]] = field(default_factory=list)


class ConfigDrivenDemoPolicy:
    """Policy that replays the same config-derived primitive actions as TaskRunner."""

    def __init__(self, builder: Optional[TaskFlowBuilder] = None) -> None:
        self.builder = builder or TaskFlowBuilder()
        self._use_evaluator_timeline = builder is None
        self._cached_stage_indices: List[Optional[int]] = []
        self._cached_actions: List[List[PrimitiveAction]] = []

    def reset(self) -> None:
        self._cached_stage_indices = []
        self._cached_actions = []

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
            action_index = min(
                evaluator.stage_action_index(env_index, stage_index),
                len(actions) - 1,
            )
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
                stage_actions=[None for _ in range(context.backend.batch_size)],
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
            stage_actions=[None for _ in range(context.backend.batch_size)],
        )

        for env_index, env_action in enumerate(action.env_actions):
            if not mask[env_index] or env_action is None:
                continue
            plan = context.plan[env_action.stage_index]
            actions = self._cached_actions[env_index]
            feedback.stage_actions[env_index] = actions
            result = TaskRunner._run_stage_action(
                env_index=env_index,
                plan=plan,
                action=env_action.action,
                backend=context.backend,
                env_mask=self._single_env_mask(context.backend.batch_size, env_index),
            )
            if result.signals[env_index] == ControlSignal.REACHED:
                feedback.stage_action_sequence_done[env_index] = (
                    env_action.action is actions[-1]
                )
            feedback.signals[env_index] = result.signals[env_index]
            feedback.details[env_index] = dict(result.details[env_index])
        return feedback

    def _ensure_capacity(self, batch_size: int) -> None:
        if len(self._cached_stage_indices) == batch_size:
            return
        self._cached_stage_indices = [None for _ in range(batch_size)]
        self._cached_actions = [[] for _ in range(batch_size)]

    def _get_stage_actions(
        self,
        env_index: int,
        stage_index: int,
        evaluator: "PolicyEvaluator",
    ) -> List[PrimitiveAction]:
        if self._cached_stage_indices[env_index] != stage_index:
            plan = evaluator.stage_plans[stage_index]
            self._cached_actions[env_index] = (
                evaluator.materialize_policy_stage_actions(
                    None if self._use_evaluator_timeline else self.builder,
                    plan.stage_index,
                )
            )
            self._cached_stage_indices[env_index] = stage_index
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
        self._timeline: Optional[ExecutionTimeline] = None
        self._builder_timelines: Dict[int, ExecutionTimeline] = {}
        self._env_states: List[_EnvRuntimeState] = []
        self._stage_execution: Optional[StageExecution] = None
        self._has_reset: np.ndarray = np.zeros(0, dtype=bool)
        self._sim_lock: threading.Lock = threading.Lock()
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop_event: Optional[threading.Event] = None
        self._sim_loop_error: Optional[Exception] = None
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
        requested_sim_loop_frequency = float(sim_loop_frequency)
        if (
            not np.isfinite(requested_sim_loop_frequency)
            or requested_sim_loop_frequency < 0
        ):
            raise ValueError(
                "sim_loop_frequency must be non-negative and finite; "
                f"got {sim_loop_frequency}."
            )
        if config.execution.interval_selection is not None:
            raise ValueError(
                "execution.interval_selection is supported by TaskRunner/aao-demo only; "
                "PolicyEvaluator cannot fast-forward external policy actions during reset()."
            )
        if config.execution.update_boundary != UpdateBoundary.CONTROL_TICK:
            raise ValueError(
                "PolicyEvaluator only supports "
                "execution.update_boundary='control_tick'; got "
                f"{config.execution.update_boundary.value!r}."
            )
        if not config.execution.render_internal_updates:
            raise ValueError(
                "execution.render_internal_updates=false is supported by "
                "TaskRunner/aao-demo only."
            )
        backend = config.backend(config.task, config.task_operators)
        if not isinstance(backend, SceneBackend):
            raise TypeError(
                "Task file backend must be an instantiated SceneBackend. "
                f"Got {type(backend).__name__}."
            )
        try:
            env = require_env_capability(
                backend.get_env(),
                EnvProtocol,
                feature="PolicyEvaluator initialization",
                expected_batch_size=backend.batch_size,
            )
            if requested_sim_loop_frequency > 0:
                require_env_capability(
                    env,
                    SimulationLoopEnvProtocol,
                    feature="PolicyEvaluator background simulation",
                    expected_batch_size=backend.batch_size,
                )
            context = ExecutionContext(
                config=config.task,
                backend=backend,
                task_file=config,
            )
            timeline = self.builder.compile(
                context,
                validate_boundaries=False,
            )
            plan = list(timeline.stage_plans)
            context.plan = plan
            backend.setup(context.config)
            stage_execution = StageExecution(
                context,
                plan,
                actions_factory=lambda stage_plan: (
                    self._require_timeline().clone_stage_actions(stage_plan.stage_index)
                ),
                timeline=timeline,
                completion_pose_resolver=self._resolve_completion_pose,
            )
        except BaseException:
            _teardown_backend_after_initialization_failure(backend)
            raise

        self._context = context
        self._timeline = timeline
        self._builder_timelines = {id(self.builder): timeline}
        self._plan = plan
        self._stage_execution = stage_execution
        self._env_states = stage_execution.states
        self._has_reset = np.zeros(backend.batch_size, dtype=bool)
        self._records = self._stage_execution.records
        self._pending_sim_loop_freq = requested_sim_loop_frequency
        return self

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        self._raise_sim_loop_error()
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        with self._sim_lock:
            context.backend.reset(mask)
        self._require_stage_execution().reset(
            mask,
            lambda env_index: _collect_reset_details(env_index, context),
        )
        self._has_reset[mask] = True
        # self._set_interest_focus()
        if self._pending_sim_loop_freq > 0 and not self.sim_loop_running:
            self.start_sim_loop(frequency=self._pending_sim_loop_freq)
        return self._build_task_update()

    def get_observation(self) -> Any:
        self._raise_sim_loop_error()
        context = self._require_context()
        with self._sim_lock:
            if self.observation_getter is not None:
                return self.observation_getter(context)
            backend = context.backend
            env = require_env_capability(
                backend.get_env(),
                ObservationEnvProtocol,
                feature="PolicyEvaluator.get_observation() without an observation_getter",
                expected_batch_size=backend.batch_size,
            )
            return env.capture_observation()

    def get_env(self) -> EnvProtocol:
        """Return the underlying environment object managed by this evaluator."""
        backend = self._require_context().backend
        return require_env_capability(
            backend.get_env(),
            EnvProtocol,
            feature="PolicyEvaluator.get_env()",
            expected_batch_size=backend.batch_size,
        )

    def get_info(self) -> Dict[str, Any]:
        """Return environment metadata when the backend provides that capability."""
        self._raise_sim_loop_error()
        backend = self._require_context().backend
        with self._sim_lock:
            env = require_env_capability(
                backend.get_env(),
                InfoEnvProtocol,
                feature="PolicyEvaluator.get_info()",
                expected_batch_size=backend.batch_size,
            )
            return env.get_info()

    def update(self, action: Any, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        self._raise_sim_loop_error()
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
        self._sim_loop_error = None
        if self._context is None:
            return
        self._context.backend.teardown()
        self._context = None
        self._plan = []
        self._records = []
        self._env_states = []
        self._timeline = None
        self._builder_timelines = {}
        self._stage_execution = None
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

        Each iteration calls ``backend.get_env().update()`` which steps MuJoCo
        physics using whatever control values (``data.ctrl``) were last set,
        so the simulation keeps running without explicit ``update()`` calls.

        Parameters
        ----------
        frequency:
            Target update rate in Hz (default 60).
        """
        self._raise_sim_loop_error()
        if self._sim_thread is not None:
            raise RuntimeError("Simulation loop is already running.")
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(
                f"Simulation loop frequency must be positive and finite; got {frequency}."
            )
        context = self._require_context()
        backend = context.backend
        env = require_env_capability(
            backend.get_env(),
            SimulationLoopEnvProtocol,
            feature="PolicyEvaluator.start_sim_loop()",
            expected_batch_size=backend.batch_size,
        )
        self._sim_stop_event = threading.Event()
        self._sim_loop_error = None
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

    def _sim_loop_fn(
        self,
        env: SimulationLoopEnvProtocol,
        frequency: float,
    ) -> None:
        """Background thread target: step physics at the requested rate."""
        assert self._sim_stop_event is not None
        dt = 1.0 / frequency
        try:
            while not self._sim_stop_event.is_set():
                t0 = time.monotonic()
                with self._sim_lock:
                    env.update()
                elapsed = time.monotonic() - t0
                remaining = dt - elapsed
                if remaining > 0:
                    self._sim_stop_event.wait(remaining)
        except Exception as exc:
            self._sim_loop_error = exc
            self._sim_stop_event.set()

    def _raise_sim_loop_error(self) -> None:
        error = self._sim_loop_error
        if error is None:
            return
        self._sim_loop_error = None
        self.stop_sim_loop()
        raise RuntimeError("Background simulation loop failed.") from error

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
        elapsed_time_sec: float = 0.0,
    ) -> ExecutionSummary:
        self._raise_sim_loop_error()
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
        if state is not self._env_states[env_index] or context is not self._context:
            raise RuntimeError("Stage execution received state from another evaluator.")
        feedback = None
        if action_feedback is not None:
            stage_actions_by_env = getattr(action_feedback, "stage_actions", None)
            stage_actions = (
                stage_actions_by_env[env_index]
                if stage_actions_by_env is not None and len(stage_actions_by_env) > 0
                else None
            )
            sequence_done_by_env = getattr(
                action_feedback,
                "stage_action_sequence_done",
                None,
            )
            sequence_done = (
                bool(sequence_done_by_env[env_index])
                if sequence_done_by_env is not None and len(sequence_done_by_env) > 0
                else None
            )
            feedback = PolicyStageFeedback(
                signal=action_feedback.signals[env_index],
                details=dict(action_feedback.details[env_index]),
                stage_actions=stage_actions,
                stage_action_sequence_done=sequence_done,
            )
        self._require_stage_execution().advance_policy(env_index, feedback)
        return

    def _set_interest_focus(self) -> None:
        context = self._require_context()
        object_names: List[str] = []
        operation_names: List[str] = []
        for env_index, state in enumerate(self._env_states):
            active = state.active
            if state.done or active is None:
                object_names.append("")
                operation_names.append("")
            else:
                object_names.append(active.plan.stage.object)
                operation_names.append(active.plan.stage.operation.value)
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
            active = state.active
            if active is not None:
                stage_index.append(active.plan.stage_index)
                stage_name.append(active.plan.stage_name)
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

    def _require_stage_execution(self) -> StageExecution:
        if self._stage_execution is None:
            raise RuntimeError(
                "PolicyEvaluator is not initialized. Call from_yaml() first."
            )
        return self._stage_execution

    def _require_timeline(self) -> ExecutionTimeline:
        if self._timeline is None:
            raise RuntimeError(
                "PolicyEvaluator is not initialized. Call from_yaml() first."
            )
        return self._timeline

    def materialize_policy_stage_actions(
        self,
        builder: Optional[TaskFlowBuilder],
        stage_index: int,
    ) -> List[PrimitiveAction]:
        """Clone one builder's nominal stage and apply scripted randomization.

        The evaluator's own timeline is reused for the common builder. A
        custom ``ConfigDrivenDemoPolicy(builder=...)`` gets one additional
        cached compile, preserving its extension point without rebuilding on
        every environment or reset.
        """

        if builder is None:
            timeline = self._require_timeline()
        else:
            key = id(builder)
            timeline = self._builder_timelines.get(key)
            if timeline is None:
                timeline = builder.compile(
                    self._require_context(),
                    validate_boundaries=False,
                )
                self._builder_timelines[key] = timeline
        actions = timeline.clone_stage_actions(stage_index)
        TaskRunner._apply_waypoint_randomization(actions, self._require_context())
        return actions

    def stage_action_index(self, env_index: int, stage_index: int) -> int:
        state = self._env_states[env_index]
        active = state.active
        if active is None or active.plan.stage_index != stage_index:
            return 0
        return active.action_index

    def _resolve_completion_pose(
        self,
        env_index: int,
        active: ActiveStageState,
    ) -> Optional[PoseControlConfig]:
        if not active.actions:
            return None
        action = active.actions[-1]
        if action.kind != "pose":
            return None
        return _resolve_policy_completion_pose(
            env_index=env_index,
            operator=active.operator,
            target=active.target,
            backend=self._require_context().backend,
            action=action,
            reference_site=active.plan.stage.site,
        )


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
