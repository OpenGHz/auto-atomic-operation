"""Shared Stage execution state machine for runner adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from .framework import (
    OPERATION_CONDITIONS,
    Operation,
    OperationConditionType,
    OperationConstraint,
    PlacedToleranceConfig,
    PoseControlConfig,
    TaskPhase,
)
from .runtime import (
    ActiveStageState,
    ControlResult,
    ControlSignal,
    ExecutionContext,
    ExecutionRecord,
    PrimitiveAction,
    StageExecutionPlan,
    StageExecutionStatus,
    _EnvRuntimeState,
    _EnvUpdateEvent,
    _ResolvedTaskKeypoint,
)
from .utils.pose import (
    PoseState,
    orientation_within_tolerance_nullable,
    position_within_tolerance,
    position_within_tolerance_nullable,
    quaternion_angular_distance,
)

if TYPE_CHECKING:
    from .execution_timeline import ExecutionTimeline

StageActionsFactory = Callable[[StageExecutionPlan], List[PrimitiveAction]]
StageActionRunner = Callable[
    [int, StageExecutionPlan, PrimitiveAction, np.ndarray], ControlResult
]
CompletionPoseResolver = Callable[[int, ActiveStageState], Optional[PoseControlConfig]]
ResetDetailsFactory = Callable[[int], Dict[str, Any]]


@dataclass
class PolicyStageFeedback:
    """One environment's feedback for a policy-driven Stage execution tick."""

    signal: Optional[ControlSignal]
    details: Dict[str, Any] = field(default_factory=dict)
    stage_actions: Optional[List[PrimitiveAction]] = None
    stage_action_sequence_done: Optional[bool] = None


class StageExecution:
    """Own Stage state, Primitive progression, conditions, and records."""

    def __init__(
        self,
        context: ExecutionContext,
        plan: List[StageExecutionPlan],
        *,
        actions_factory: StageActionsFactory,
        action_runner: Optional[StageActionRunner] = None,
        completion_pose_resolver: Optional[CompletionPoseResolver] = None,
        timeline: Optional["ExecutionTimeline"] = None,
    ) -> None:
        self.context = context
        self.plan = plan
        self.actions_factory = actions_factory
        self.action_runner = action_runner
        self.completion_pose_resolver = completion_pose_resolver
        self.timeline = timeline
        self.states = [_EnvRuntimeState() for _ in range(context.backend.batch_size)]
        self.records: List[ExecutionRecord] = []

    def reset(
        self,
        env_mask: np.ndarray,
        details_factory: ResetDetailsFactory,
    ) -> None:
        for env_index, enabled in enumerate(env_mask):
            if not enabled:
                continue
            state = _EnvRuntimeState()
            state.latest_details = details_factory(env_index)
            self.states[env_index] = state

    def record_failure(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        details: Dict[str, Any],
    ) -> None:
        self.records.append(
            self._record(
                env_index,
                plan,
                StageExecutionStatus.FAILED,
                details,
            )
        )

    def advance_control(
        self,
        env_index: int,
        *,
        use_configured_identity: bool,
    ) -> _EnvUpdateEvent:
        state = self.states[env_index]
        start_event = self._ensure_started(env_index, state)
        if start_event is not None:
            return start_event
        active = state.active
        if active is None:
            return _EnvUpdateEvent()
        if not active.actions:
            raise RuntimeError(
                f"Stage '{active.plan.stage_name}' emitted no primitives."
            )
        if self.action_runner is None:
            raise RuntimeError("Stage execution has no Primitive action adapter.")

        action = active.actions[active.action_index]
        result = self.action_runner(
            env_index,
            active.plan,
            action,
            self._mask_for_env(env_index),
        )
        details = self._action_details(
            env_index,
            active,
            action,
            result.details[env_index],
        )
        return self._consume_configured_result(
            env_index,
            state,
            result.signals[env_index],
            details,
            mode="control",
            use_configured_identity=use_configured_identity,
        )

    def advance_policy(
        self,
        env_index: int,
        feedback: Optional[PolicyStageFeedback],
    ) -> _EnvUpdateEvent:
        state = self.states[env_index]
        if feedback is None:
            start_event = self._ensure_started(
                env_index,
                state,
                resolve_completion_pose=True,
            )
            if start_event is not None:
                return start_event
            return self._poll_external_policy(env_index, state)
        if feedback.stage_actions is None:
            return self._consume_external_feedback(env_index, state, feedback)

        start_event = self._ensure_started(
            env_index,
            state,
            actions_override=feedback.stage_actions,
        )
        if start_event is not None:
            return start_event
        if feedback.signal is None:
            self._set_policy_running(state, env_index, feedback.details)
            return _EnvUpdateEvent()
        return self._consume_configured_result(
            env_index,
            state,
            feedback.signal,
            dict(feedback.details),
            mode="policy",
            use_configured_identity=False,
        )

    def _consume_external_feedback(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        feedback: PolicyStageFeedback,
    ) -> _EnvUpdateEvent:
        start_event = self._ensure_started(
            env_index,
            state,
            resolve_completion_pose=True,
        )
        if start_event is not None:
            return start_event
        active = state.active
        if active is None:
            return _EnvUpdateEvent()

        if feedback.signal in {ControlSignal.TIMED_OUT, ControlSignal.FAILED}:
            failure = self._build_action_failure_details(
                active.plan,
                feedback.details,
                feedback.signal,
                env_index=env_index,
            )
            self.record_failure(env_index, active.plan, failure)
            self._set_failed(state, failure)
            return _EnvUpdateEvent(failed=True)

        if not feedback.stage_action_sequence_done:
            self._set_policy_running(state, env_index, feedback.details)
            return _EnvUpdateEvent()

        success_failure = self._final_stage_failure(env_index, active)
        if success_failure is not None:
            self.record_failure(env_index, active.plan, success_failure)
            self._set_failed(state, success_failure)
            return _EnvUpdateEvent(failed=True)

        details = self._policy_success_details(env_index, active.plan)
        self._record_success(env_index, active.plan, details)
        self._set_succeeded(state, details)
        return _EnvUpdateEvent(stage_succeeded=True)

    def _ensure_started(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        *,
        actions_override: Optional[List[PrimitiveAction]] = None,
        resolve_completion_pose: bool = False,
    ) -> Optional[_EnvUpdateEvent]:
        if state.stage_cursor >= len(self.plan):
            state.done = True
            state.success = True
            state.latest_status = StageExecutionStatus.SUCCEEDED
            state.phase = None
            state.phase_step = None
            return _EnvUpdateEvent()
        if state.active is not None:
            return None

        plan = self.plan[state.stage_cursor]
        failure = (
            None
            if plan.stage.operation == Operation.PULL
            else check_stage_condition(
                env_index=env_index,
                context=self.context,
                plan=plan,
                condition_type=OperationConditionType.PERFORM,
            )
        )
        if failure is not None:
            self.record_failure(env_index, plan, failure)
            self._set_failed(state, failure)
            return _EnvUpdateEvent(failed=True)

        active = self._start_stage(
            env_index,
            plan,
            actions_override=actions_override,
        )
        if resolve_completion_pose and self.completion_pose_resolver is not None:
            active.completion_pose = self.completion_pose_resolver(env_index, active)
        state.active = active
        state.latest_status = StageExecutionStatus.RUNNING
        return None

    def _start_stage(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        *,
        actions_override: Optional[List[PrimitiveAction]],
    ) -> ActiveStageState:
        backend = self.context.backend
        operator = backend.get_operator_handler(plan.operator_name)
        target = backend.get_object_handler(plan.stage.object)
        initial_object_pose = (
            None if target is None else target.get_pose().select(env_index)
        )
        held_object_name = (
            backend.get_grasped_object_name(plan.operator_name, env_index)
            if plan.stage.operation == Operation.PLACE
            else None
        )
        actions = (
            self.actions_factory(plan) if actions_override is None else actions_override
        )
        return ActiveStageState(
            plan=plan,
            operator=operator,
            target=target,
            actions=actions,
            initial_object_pose=initial_object_pose,
            held_object_name=held_object_name,
        )

    def _consume_configured_result(
        self,
        env_index: int,
        state: _EnvRuntimeState,
        signal: ControlSignal,
        details: Dict[str, Any],
        *,
        mode: str,
        use_configured_identity: bool,
    ) -> _EnvUpdateEvent:
        active = state.active
        if active is None:
            raise RuntimeError("Configured Stage feedback requires an active Stage.")
        action = active.actions[active.action_index]

        if signal == ControlSignal.RUNNING:
            self._set_running(
                state,
                active,
                env_index,
                details,
                mode=mode,
                use_configured_identity=use_configured_identity,
            )
            return _EnvUpdateEvent(control_tick=mode == "control")

        if signal in {ControlSignal.TIMED_OUT, ControlSignal.FAILED}:
            failure = self._build_action_failure_details(
                active.plan,
                details,
                signal,
                env_index=env_index,
            )
            self.record_failure(env_index, active.plan, failure)
            self._set_failed(state, failure)
            return _EnvUpdateEvent(control_tick=mode == "control", failed=True)

        if signal != ControlSignal.REACHED:
            raise RuntimeError(f"Unsupported control signal: {signal!r}")

        completed_position = self._resolve_completed_position(
            active.plan,
            action,
            active.action_index,
        )
        completed_keypoint = completed_position if action.completes_keypoint else None
        active.action_index += 1
        mid_failure = self._mid_stage_failure(env_index, active, action)
        if mid_failure is not None:
            self.record_failure(env_index, active.plan, mid_failure)
            self._set_failed(state, mid_failure)
            return self._event_for_completed_action(
                mode=mode,
                completed_position=completed_position,
                completed_keypoint=completed_keypoint,
                failed=True,
            )

        if active.action_index < len(active.actions):
            self._set_running(
                state,
                active,
                env_index,
                details,
                mode=mode,
                use_configured_identity=use_configured_identity,
            )
            return self._event_for_completed_action(
                mode=mode,
                completed_position=completed_position,
                completed_keypoint=completed_keypoint,
            )

        success_failure = self._final_stage_failure(
            env_index,
            active,
            press_checked_at_eef=any(action.kind == "eef" for action in active.actions),
        )
        if success_failure is not None:
            self.record_failure(env_index, active.plan, success_failure)
            self._set_failed(state, success_failure)
            return self._event_for_completed_action(
                mode=mode,
                completed_position=completed_position,
                completed_keypoint=completed_keypoint,
                failed=True,
            )

        success_details = (
            details
            if mode == "control"
            else self._policy_success_details(env_index, active.plan)
        )
        self._record_success(env_index, active.plan, success_details)
        self._set_succeeded(state, success_details)
        return self._event_for_completed_action(
            mode=mode,
            completed_position=completed_position,
            completed_keypoint=completed_keypoint,
            stage_succeeded=True,
        )

    def _poll_external_policy(
        self,
        env_index: int,
        state: _EnvRuntimeState,
    ) -> _EnvUpdateEvent:
        active = state.active
        if active is None:
            return _EnvUpdateEvent()
        success_failure = self._final_stage_failure(env_index, active)
        if success_failure is None:
            details = self._policy_success_details(env_index, active.plan)
            self._record_success(env_index, active.plan, details)
            self._set_succeeded(state, details)
            return _EnvUpdateEvent(stage_succeeded=True)
        self._set_policy_running(
            state,
            env_index,
            {
                "event": "stage_success_condition_pending",
                **success_failure,
            },
        )
        return _EnvUpdateEvent()

    def _mid_stage_failure(
        self,
        env_index: int,
        active: ActiveStageState,
        completed_action: PrimitiveAction,
    ) -> Optional[Dict[str, Any]]:
        if completed_action.kind != "eef":
            return None
        eef = completed_action.eef
        if eef is not None and eef.require_grasp:
            object_name = active.plan.stage.object
            target_grasped = bool(
                object_name
                and self.context.backend.is_object_grasped(
                    active.operator.name,
                    object_name,
                )[env_index]
            )
            if not target_grasped:
                details = _condition_failure_details(
                    env_index=env_index,
                    context=self.context,
                    plan=active.plan,
                    condition_type=OperationConditionType.SUCCESS,
                    constraint=OperationConstraint.GRASPED,
                    is_grasping=bool(
                        self.context.backend.is_operator_grasping(active.operator.name)[
                            env_index
                        ]
                    ),
                    completion_pose=None,
                    target_object_pose=None,
                    held_object_name=None,
                )
                details["is_target_grasped"] = False
                details["failure_reason"] = (
                    f"operator is not grasping required target '{object_name}'"
                )
                return details
        operation = active.plan.stage.operation
        if operation == Operation.PULL:
            return check_stage_condition(
                env_index=env_index,
                context=self.context,
                plan=active.plan,
                condition_type=OperationConditionType.PERFORM,
                initial_pose=active.initial_object_pose,
            )
        if operation == Operation.PICK and not bool(
            self.context.backend.is_operator_grasping(active.operator.name)[env_index]
        ):
            return check_stage_condition(
                env_index=env_index,
                context=self.context,
                plan=active.plan,
                condition_type=OperationConditionType.SUCCESS,
                initial_pose=active.initial_object_pose,
            )
        if operation == Operation.PRESS:
            return check_stage_condition(
                env_index=env_index,
                context=self.context,
                plan=active.plan,
                condition_type=OperationConditionType.SUCCESS,
                initial_pose=active.initial_object_pose,
            )
        return None

    def _final_stage_failure(
        self,
        env_index: int,
        active: ActiveStageState,
        *,
        press_checked_at_eef: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if press_checked_at_eef and active.plan.stage.operation == Operation.PRESS:
            return None
        target_object_pose = self._target_object_pose(env_index, active)
        return check_stage_condition(
            env_index=env_index,
            context=self.context,
            plan=active.plan,
            condition_type=OperationConditionType.SUCCESS,
            initial_pose=active.initial_object_pose,
            completion_pose=(
                active.completion_pose or self._completion_pose_from_active(active)
            ),
            target_object_pose=target_object_pose,
            held_object_name=active.held_object_name,
        )

    def _target_object_pose(
        self,
        env_index: int,
        active: ActiveStageState,
    ) -> Optional[PoseState]:
        if active.plan.stage.operation != Operation.PLACE:
            return None
        reference = getattr(active.plan.stage.param, "placed_reference", "object")
        target_name = active.plan.stage.object
        if reference == "object" and target_name:
            target = self.context.backend.get_object_handler(target_name)
            return None if target is None else target.get_pose().select(env_index)
        return self._pre_move_end_pose(active)

    def _set_running(
        self,
        state: _EnvRuntimeState,
        active: ActiveStageState,
        env_index: int,
        details: Dict[str, Any],
        *,
        mode: str,
        use_configured_identity: bool,
    ) -> None:
        state.latest_status = StageExecutionStatus.RUNNING
        if mode == "policy":
            self._set_policy_running(state, env_index, details)
            return
        phase, phase_step = self._action_phase(
            active.actions,
            active.action_index,
            use_configured_identity=use_configured_identity,
        )
        state.latest_details = details
        state.phase = phase
        state.phase_step = phase_step

    @staticmethod
    def _set_policy_running(
        state: _EnvRuntimeState,
        env_index: int,
        details: Dict[str, Any],
    ) -> None:
        state.latest_status = StageExecutionStatus.RUNNING
        event = details.get("event", "stage_action_sequence_running")
        state.latest_details = {
            "event": event,
            "env_index": env_index,
            "evaluation_mode": "policy",
            **details,
        }
        state.phase = "policy"
        state.phase_step = None

    @staticmethod
    def _set_failed(state: _EnvRuntimeState, details: Dict[str, Any]) -> None:
        state.active = None
        state.done = True
        state.success = False
        state.latest_status = StageExecutionStatus.FAILED
        state.latest_details = details
        state.phase = None
        state.phase_step = None
        state.reported_keypoint = None

    def _set_succeeded(
        self,
        state: _EnvRuntimeState,
        details: Dict[str, Any],
    ) -> None:
        state.stage_cursor += 1
        state.active = None
        state.latest_status = StageExecutionStatus.SUCCEEDED
        state.latest_details = details
        state.phase = None
        state.phase_step = None
        if state.stage_cursor >= len(self.plan):
            state.done = True
            state.success = True
        else:
            state.success = False

    def _record_success(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        details: Dict[str, Any],
    ) -> None:
        self.records.append(
            self._record(
                env_index,
                plan,
                StageExecutionStatus.SUCCEEDED,
                details,
            )
        )

    @staticmethod
    def _record(
        env_index: int,
        plan: StageExecutionPlan,
        status: StageExecutionStatus,
        details: Dict[str, Any],
    ) -> ExecutionRecord:
        return ExecutionRecord(
            env_index=env_index,
            stage_index=plan.stage_index,
            stage_name=plan.stage_name,
            operator=plan.operator_name,
            operation=plan.stage.operation.value,
            target_object=plan.stage.object,
            blocking=plan.stage.blocking,
            status=status,
            details=details,
        )

    def _mask_for_env(self, env_index: int) -> np.ndarray:
        mask = np.zeros(self.context.backend.batch_size, dtype=bool)
        mask[env_index] = True
        return mask

    def _action_details(
        self,
        env_index: int,
        active: ActiveStageState,
        action: PrimitiveAction,
        result_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        details = {
            "env_index": env_index,
            "action": action.kind,
            "action_index": active.action_index,
            **result_details,
        }
        if action.kind != "pose" or action.pose is None or action.pose.arc is None:
            return details
        arc = action.pose.arc
        arc_details: Dict[str, Any] = {
            "pivot": (
                arc.pivot
                if isinstance(arc.pivot, str)
                else [float(value) for value in arc.pivot]
            ),
            "axis": [float(value) for value in arc.axis],
            "angle": float(arc.angle),
            "absolute": bool(arc.absolute),
        }
        if arc.absolute and isinstance(arc.pivot, str):
            try:
                current_joint = float(
                    self.context.backend.get_joint_angle(arc.pivot, env_index)
                )
                arc_details["current_joint_angle"] = current_joint
                arc_details["target_joint_angle"] = float(arc.angle)
                arc_details["delta_joint_angle"] = float(arc.angle) - current_joint
            except (KeyError, NotImplementedError):
                pass
        elif action.arc_cumulative_angle is not None:
            arc_details["cumulative_angle"] = float(action.arc_cumulative_angle)
        details["action"] = "arc"
        details["arc"] = arc_details
        return details

    def _resolve_completed_position(
        self,
        plan: StageExecutionPlan,
        action: PrimitiveAction,
        action_index: int,
    ) -> Optional[_ResolvedTaskKeypoint]:
        if self.timeline is not None:
            # External policy adapters may provide a runtime action sequence
            # whose shape differs from the nominal timeline.  In that case
            # the compiled lookup is intentionally best-effort.
            try:
                compiled = self.timeline.keypoint_for_action(
                    plan.stage_index,
                    action_index,
                )
            except IndexError:
                compiled = None
            if compiled is not None and (
                compiled.phase == action.phase and compiled.waypoint == action.waypoint
            ):
                return compiled
        if not isinstance(action.phase, TaskPhase) or not isinstance(
            action.waypoint, int
        ):
            return None
        return _ResolvedTaskKeypoint(
            stage_index=plan.stage_index,
            stage_name=plan.stage_name,
            phase=action.phase,
            waypoint=action.waypoint,
        )

    @staticmethod
    def _event_for_completed_action(
        *,
        mode: str,
        completed_position: Optional[_ResolvedTaskKeypoint],
        completed_keypoint: Optional[_ResolvedTaskKeypoint],
        failed: bool = False,
        stage_succeeded: bool = False,
    ) -> _EnvUpdateEvent:
        return _EnvUpdateEvent(
            control_tick=mode == "control",
            primitive_reached=True,
            keypoint_reached=completed_keypoint is not None,
            stage_succeeded=stage_succeeded,
            failed=failed,
            completed_position=completed_position,
            completed_keypoint=completed_keypoint,
        )

    @staticmethod
    def _policy_success_details(
        env_index: int,
        plan: StageExecutionPlan,
    ) -> Dict[str, Any]:
        return {
            "event": "stage_success_condition_met",
            "env_index": env_index,
            "evaluation_mode": "policy",
            "operator": plan.operator_name,
            "operation": plan.stage.operation.value,
            "target_object": plan.stage.object,
        }

    @staticmethod
    def _build_action_failure_details(
        plan: StageExecutionPlan,
        details: Dict[str, Any],
        signal: ControlSignal,
        *,
        env_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        enriched = dict(details)
        if env_index is not None:
            enriched.setdefault("env_index", env_index)
        enriched.setdefault("failure_stage", "execution")
        enriched.setdefault("operator", plan.operator_name)
        enriched.setdefault("operation", plan.stage.operation.value)
        enriched.setdefault("target_object", plan.stage.object)
        if signal == ControlSignal.TIMED_OUT:
            enriched.setdefault("failure_category", "controller_timeout")
            enriched.setdefault(
                "failure_reason",
                "primitive action did not finish before timeout",
            )
        elif signal == ControlSignal.FAILED:
            enriched.setdefault("failure_category", "controller_failure")
            enriched.setdefault("failure_reason", "primitive action reported failure")
        else:
            enriched.setdefault("failure_category", "execution_failure")
            enriched.setdefault(
                "failure_reason",
                "primitive action failed during execution",
            )
        return enriched

    @staticmethod
    def _completion_pose_from_active(
        active: ActiveStageState,
    ) -> Optional[PoseControlConfig]:
        for action in reversed(active.actions):
            if action.kind == "pose" and action.resolved_pose is not None:
                return action.resolved_pose
        return None

    @staticmethod
    def _pre_move_end_pose(active: ActiveStageState) -> Optional[PoseState]:
        last_pose = None
        for action in active.actions:
            if action.kind == "eef":
                break
            if action.kind == "pose" and action.resolved_pose is not None:
                last_pose = action.resolved_pose
        if last_pose is None:
            return None
        return PoseState(
            position=np.asarray(last_pose.position, dtype=np.float64).reshape(1, 3),
            orientation=np.asarray(last_pose.orientation, dtype=np.float64).reshape(
                1, 4
            ),
        )

    @staticmethod
    def _action_phase(
        actions: List[PrimitiveAction],
        action_index: int,
        *,
        use_configured_identity: bool = False,
    ) -> tuple[str, Optional[int]]:
        if not actions:
            return "complete", None
        index = min(action_index, len(actions) - 1)
        action = actions[index]
        if use_configured_identity and isinstance(action.phase, TaskPhase):
            return action.phase.value, action.waypoint
        kinds = [item.kind for item in actions]
        first_eef = next((i for i, kind in enumerate(kinds) if kind == "eef"), None)
        if action.kind == "eef":
            return "eef", None
        if first_eef is None or index < first_eef:
            return "pre_move", index
        post_index = sum(1 for kind in kinds[first_eef + 1 : index] if kind == "pose")
        return "post_move", post_index


def check_stage_condition(
    env_index: int,
    context: ExecutionContext,
    plan: StageExecutionPlan,
    condition_type: OperationConditionType,
    initial_pose: Optional[PoseState] = None,
    completion_pose: Optional[PoseControlConfig] = None,
    target_object_pose: Optional[PoseState] = None,
    held_object_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    constraints = OPERATION_CONDITIONS.get(plan.stage.operation)
    if not constraints:
        return None
    constraint = constraints.get(condition_type)
    if constraint is None or constraint == OperationConstraint.NONE:
        return None

    operator_name = plan.operator_name
    object_name = plan.stage.object
    backend = context.backend
    is_grasping = bool(backend.is_operator_grasping(operator_name)[env_index])

    if constraint == OperationConstraint.GRASPED:
        satisfied = is_grasping
    elif constraint == OperationConstraint.RELEASED:
        satisfied = not is_grasping
    elif constraint == OperationConstraint.CONTACTED:
        satisfied = bool(
            backend.is_operator_contacting(operator_name, object_name)[env_index]
        )
    elif constraint == OperationConstraint.DISPLACED:
        threshold = getattr(plan.stage.param, "displacement_threshold", None)
        kwargs = {"threshold": float(threshold)} if threshold is not None else {}
        satisfied = (
            bool(
                backend.is_object_displaced(object_name, initial_pose, **kwargs)[
                    env_index
                ]
            )
            if initial_pose is not None and object_name
            else True
        )
    elif constraint == OperationConstraint.REACHED:
        operator = backend.get_operator_handler(operator_name)
        waypoint_tolerance = (
            getattr(completion_pose, "tolerance", None) if completion_pose else None
        )
        operator_position, operator_orientation = operator.get_reached_tolerances()
        position_tolerance = (
            waypoint_tolerance.position
            if waypoint_tolerance is not None
            and waypoint_tolerance.position is not None
            else operator_position
        )
        orientation_tolerance = (
            waypoint_tolerance.orientation
            if waypoint_tolerance is not None
            and waypoint_tolerance.orientation is not None
            else operator_orientation
        )
        if completion_pose is None:
            satisfied = False
        else:
            current_pose = operator.get_end_effector_pose().select(env_index)
            position_difference = np.asarray(
                current_pose.position[0], dtype=np.float64
            ) - np.asarray(completion_pose.position, dtype=np.float64)
            orientation_error = float(
                quaternion_angular_distance(
                    current_pose.orientation[0],
                    np.asarray(completion_pose.orientation, dtype=np.float64),
                )
            )
            satisfied = position_within_tolerance(
                position_difference,
                position_tolerance,
            ) and orientation_error <= float(orientation_tolerance)
    elif constraint == OperationConstraint.PLACED:
        satisfied = _placed_condition_satisfied(
            env_index=env_index,
            context=context,
            plan=plan,
            is_grasping=is_grasping,
            target_object_pose=target_object_pose,
            held_object_name=held_object_name,
        )
    else:
        satisfied = True

    if satisfied:
        return None
    return _condition_failure_details(
        env_index=env_index,
        context=context,
        plan=plan,
        condition_type=condition_type,
        constraint=constraint,
        is_grasping=is_grasping,
        completion_pose=completion_pose,
        target_object_pose=target_object_pose,
        held_object_name=held_object_name,
    )


def _placed_condition_satisfied(
    *,
    env_index: int,
    context: ExecutionContext,
    plan: StageExecutionPlan,
    is_grasping: bool,
    target_object_pose: Optional[PoseState],
    held_object_name: Optional[str],
) -> bool:
    if is_grasping:
        return False
    if target_object_pose is None:
        return True
    if not held_object_name:
        return False
    handler = context.backend.get_object_handler(held_object_name)
    if handler is None:
        return False
    current = handler.get_pose()
    position_difference = np.asarray(
        current.position[env_index], dtype=np.float64
    ) - np.asarray(target_object_pose.position[0], dtype=np.float64)
    control = plan.stage.param
    stage_tolerance: Optional[PlacedToleranceConfig] = getattr(
        control,
        "placed_tolerance",
        None,
    )
    operator_position, operator_orientation = context.backend.get_operator_handler(
        plan.operator_name
    ).get_placed_tolerances()
    stage_position = stage_tolerance.position if stage_tolerance is not None else None
    stage_orientation = (
        stage_tolerance.orientation if stage_tolerance is not None else None
    )
    position_tolerance = (
        stage_position
        if _is_configured(stage_position)
        else operator_position
        if _is_configured(operator_position)
        else None
    )
    orientation_tolerance = (
        stage_orientation
        if _is_configured(stage_orientation)
        else operator_orientation
        if _is_configured(operator_orientation)
        else None
    )
    return position_within_tolerance_nullable(
        position_difference,
        position_tolerance,
    ) and orientation_within_tolerance_nullable(
        current.orientation[env_index],
        target_object_pose.orientation[0],
        orientation_tolerance,
    )


def _is_configured(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, np.ndarray)):
        return any(item is not None for item in value)
    return True


def _condition_failure_details(
    *,
    env_index: int,
    context: ExecutionContext,
    plan: StageExecutionPlan,
    condition_type: OperationConditionType,
    constraint: OperationConstraint,
    is_grasping: bool,
    completion_pose: Optional[PoseControlConfig],
    target_object_pose: Optional[PoseState],
    held_object_name: Optional[str],
) -> Dict[str, Any]:
    object_name = plan.stage.object
    phase = (
        "precondition"
        if condition_type == OperationConditionType.PERFORM
        else "postcondition"
    )
    failure_category, failure_reason = {
        OperationConstraint.GRASPED: (
            "missing_grasp",
            "operator is not grasping the required object",
        ),
        OperationConstraint.RELEASED: (
            "unexpected_grasp",
            "operator is still grasping when it should be empty-handed",
        ),
        OperationConstraint.CONTACTED: (
            "no_contact",
            f"operator end-effector is not in contact with '{object_name}'",
        ),
        OperationConstraint.DISPLACED: (
            "no_displacement",
            f"object '{object_name}' has not been displaced beyond the threshold",
        ),
        OperationConstraint.REACHED: (
            "target_not_reached",
            "operator end-effector is not within tolerance of the target pose",
        ),
        OperationConstraint.PLACED: (
            "placement_failed",
            "held object is not within tolerance of the target position",
        ),
    }.get(constraint, ("condition_mismatch", "stage condition is not satisfied"))
    details: Dict[str, Any] = {
        "event": f"stage_{phase}_failed",
        "failure_stage": phase,
        "failure_category": failure_category,
        "failure_reason": failure_reason,
        "condition_type": condition_type.value,
        "required_constraint": constraint.value,
        "operator": plan.operator_name,
        "operation": plan.stage.operation.value,
        "target_object": object_name,
        "is_operator_grasping": is_grasping,
        "env_index": env_index,
    }
    if constraint == OperationConstraint.REACHED:
        _add_reached_failure_details(
            details,
            env_index,
            context,
            plan,
            completion_pose,
        )
    elif constraint == OperationConstraint.PLACED:
        details["placed_reference"] = getattr(
            plan.stage.param,
            "placed_reference",
            "object",
        )
        _add_placed_failure_details(
            details,
            env_index,
            context,
            target_object_pose,
            held_object_name,
        )
    return details


def _add_reached_failure_details(
    details: Dict[str, Any],
    env_index: int,
    context: ExecutionContext,
    plan: StageExecutionPlan,
    completion_pose: Optional[PoseControlConfig],
) -> None:
    details["completion_pose_available"] = completion_pose is not None
    if completion_pose is None:
        return
    operator = context.backend.get_operator_handler(plan.operator_name)
    current_pose = operator.get_end_effector_pose().select(env_index)
    details["target_pose"] = completion_pose.model_dump(mode="json")
    details["current_pose"] = {
        "position": [float(value) for value in current_pose.position[0]],
        "orientation": [float(value) for value in current_pose.orientation[0]],
    }
    details["position_error"] = float(
        np.linalg.norm(
            np.asarray(current_pose.position[0], dtype=np.float64)
            - np.asarray(completion_pose.position, dtype=np.float64)
        )
    )
    details["orientation_error"] = float(
        quaternion_angular_distance(
            current_pose.orientation[0],
            np.asarray(completion_pose.orientation, dtype=np.float64),
        )
    )


def _add_placed_failure_details(
    details: Dict[str, Any],
    env_index: int,
    context: ExecutionContext,
    target_object_pose: Optional[PoseState],
    held_object_name: Optional[str],
) -> None:
    details["held_object"] = held_object_name or ""
    if not held_object_name or target_object_pose is None:
        return
    handler = context.backend.get_object_handler(held_object_name)
    if handler is None:
        return
    current = handler.get_pose()
    details["target_position"] = [
        float(value) for value in target_object_pose.position[0]
    ]
    details["current_position"] = [
        float(value) for value in current.position[env_index]
    ]
    details["position_error"] = float(
        np.linalg.norm(
            np.asarray(current.position[env_index], dtype=np.float64)
            - np.asarray(target_object_pose.position[0], dtype=np.float64)
        )
    )
    details["target_orientation"] = [
        float(value) for value in target_object_pose.orientation[0]
    ]
    details["current_orientation"] = [
        float(value) for value in current.orientation[env_index]
    ]
    details["orientation_error"] = float(
        quaternion_angular_distance(
            current.orientation[env_index],
            target_object_pose.orientation[0],
        )
    )
