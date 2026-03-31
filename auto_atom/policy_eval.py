"""Policy-driven evaluator that reuses the task runner's result data classes."""

from __future__ import annotations

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
    ExecutionContext,
    ExecutionRecord,
    ExecutionSummary,
    ObjectHandler,
    OperatorHandler,
    SceneBackend,
    StageExecutionPlan,
    StageExecutionStatus,
    TaskFlowBuilder,
    TaskUpdate,
    _EnvRuntimeState,
    _build_execution_summary,
    _check_stage_condition,
    _collect_reset_details,
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

    @property
    def records(self) -> List[ExecutionRecord]:
        return list(self._records)

    def from_yaml(self, path: str | Path) -> "PolicyEvaluator":
        return self.from_config(load_task_file(path))

    def from_config(self, config: TaskFileConfig) -> "PolicyEvaluator":
        backend = config.backend(config.task, config.operators)
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
        self._context.backend.setup(self._context.config)
        self._env_states = [_EnvRuntimeState() for _ in range(backend.batch_size)]
        self._policy_states = [None for _ in range(backend.batch_size)]
        self._records = []
        return self

    def reset(self, env_mask: Optional[np.ndarray] = None) -> TaskUpdate:
        context = self._require_context()
        mask = self._normalize_mask(env_mask)
        context.backend.reset(mask)
        for env_index, enabled in enumerate(mask):
            if enabled:
                self._env_states[env_index] = _EnvRuntimeState()
                self._env_states[env_index].latest_details = _collect_reset_details(
                    env_index, context
                )
                self._policy_states[env_index] = None
        self._set_interest_focus()
        return self._build_task_update()

    def get_observation(self) -> Any:
        context = self._require_context()
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
        self.action_applier(context, action, mask)
        for env_index, enabled in enumerate(mask):
            if not enabled or self._env_states[env_index].done:
                continue
            self._update_env(env_index, self._env_states[env_index], context)
        self._set_interest_focus()
        return self._build_task_update()

    def close(self) -> None:
        if self._context is None:
            return
        self._context.backend.teardown()
        self._context = None
        self._plan = []
        self._records = []
        self._env_states = []
        self._policy_states = []

    def summarize(
        self,
        update: Optional[TaskUpdate] = None,
        *,
        max_updates: Optional[int] = None,
        updates_used: int = 0,
    ) -> ExecutionSummary:
        return _build_execution_summary(
            update=update or self._build_task_update(),
            records=self._records,
            total_stages=len(self._plan),
            max_updates=max_updates,
            updates_used=updates_used,
        )

    def _update_env(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        context: ExecutionContext,
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
                state.success = None
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
        success: List[Optional[bool]] = []
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
            success=np.asarray(success, dtype=object),
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
    )
