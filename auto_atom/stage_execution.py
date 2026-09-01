"""Shared Stage execution state machine for runner adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .execution_model import (
    ActiveStageState,
    ControlResult,
    ControlSignal,
    ExecutionRecord,
    ExecutionTimelineProtocol,
    PrimitiveAction,
    ResolvedMotionGoal,
    StageExecutionPlan,
    StageExecutionStatus,
    _EnvRuntimeState,
    _EnvUpdateEvent,
    _ResolvedTaskKeypoint,
)
from .framework import (
    OPERATION_CONDITIONS,
    AxisAlignmentOrientationGoalConfig,
    ControlledFrameKind,
    Operation,
    OperationConditionType,
    OperationConstraint,
    PlacedToleranceConfig,
    PoseControlConfig,
    TaskPhase,
)
from .motion_goal import motion_goal_errors, resolve_object_motion_goal
from .pose_goal import axis_alignment_error, resolve_axis_in_world
from .utils.pose import (
    PoseState,
    orientation_within_tolerance_nullable,
    position_within_tolerance,
    position_within_tolerance_nullable,
    quaternion_angular_distance,
)

StageActionsFactory = Callable[[StageExecutionPlan], List[PrimitiveAction]]
StageActionRunner = Callable[
    [int, StageExecutionPlan, PrimitiveAction, np.ndarray], ControlResult
]
CompletionPoseResolver = Callable[[int, ActiveStageState], Optional[ResolvedMotionGoal]]
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
        context: Any,
        plan: List[StageExecutionPlan],
        *,
        actions_factory: StageActionsFactory,
        action_runner: Optional[StageActionRunner] = None,
        completion_pose_resolver: Optional[CompletionPoseResolver] = None,
        timeline: Optional[ExecutionTimelineProtocol] = None,
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
            self.context.clear_env_grasp_bindings(env_index)
            self.context.clear_env_logical_object(env_index)
            state = _EnvRuntimeState()
            state.latest_details = details_factory(env_index)
            self.states[env_index] = state

    def prepare_policy(
        self,
        env_index: int,
        *,
        resolve_completion_pose: bool,
    ) -> _EnvUpdateEvent:
        """Start and validate a policy stage before its action is applied."""
        state = self.states[env_index]
        start_event = self._ensure_started(
            env_index,
            state,
            resolve_completion_pose=resolve_completion_pose,
        )
        if start_event is not None:
            return start_event

        # External object-only policies do not run the materialized primitive
        # that normally resolves a held-object waypoint.  Resolve and cache
        # that semantic completion goal before the policy gets a chance to
        # move the object, so relative/static references retain their start
        # snapshot and the later postcondition check remains meaningful.
        if resolve_completion_pose and self.context.is_object_only:
            active = state.active
            if active is not None:
                completion_failure = self._resolve_external_completion_goal(
                    env_index,
                    active,
                )
                if completion_failure is not None:
                    completion_failure = self.record_failure(
                        env_index,
                        active.plan,
                        completion_failure,
                    )
                    self._set_failed(state, completion_failure)
                    return _EnvUpdateEvent(failed=True)
        return _EnvUpdateEvent()

    def record_failure(
        self,
        env_index: int,
        plan: StageExecutionPlan,
        details: Dict[str, Any],
    ) -> Dict[str, Any]:
        enriched = dict(details)
        if self.context.is_object_only:
            # There is intentionally no physical operator in this mode, so a
            # contact query would only manufacture an ``Unknown operator``
            # diagnostic and obscure the actual object-only failure.
            contact_snapshot = {
                "status": "not_applicable",
                "contacts": [],
            }
        else:
            try:
                contacts = self.context.backend.get_operator_contacts(
                    plan.operator_name,
                    env_index,
                )
                contact_snapshot = {
                    "status": "unsupported" if contacts is None else "observed",
                    "contacts": (
                        []
                        if contacts is None
                        else [contact.to_dict() for contact in contacts]
                    ),
                }
            except Exception as exc:  # diagnostic failure must not mask task failure
                contact_snapshot = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "contacts": [],
                }
        enriched["operator_contact_snapshot"] = contact_snapshot
        self.records.append(
            self._record(
                env_index,
                plan,
                StageExecutionStatus.FAILED,
                enriched,
            )
        )
        return enriched

    def terminate_unfinished_at_update_limit(self, max_updates: int) -> None:
        """Fail every unfinished environment after the rollout budget is spent."""
        if max_updates < 0:
            raise ValueError("max_updates must be non-negative")
        for env_index, state in enumerate(self.states):
            if state.done:
                continue
            plan = (
                state.active.plan
                if state.active is not None
                else self.plan[state.stage_cursor]
                if state.stage_cursor < len(self.plan)
                else None
            )
            if plan is None:
                state.done = True
                state.success = True
                state.latest_status = StageExecutionStatus.SUCCEEDED
                state.latest_details = {"event": "task_succeeded"}
                continue
            details = self.record_failure(
                env_index,
                plan,
                {
                    "event": "rollout_update_limit_reached",
                    "failure_stage": "rollout",
                    "failure_category": "max_updates_reached",
                    "failure_reason": (
                        f"reached max_updates={max_updates} before task completion"
                    ),
                    "env_index": env_index,
                    "max_updates": max_updates,
                },
            )
            self._set_failed(state, details)

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

        active = state.active
        if (
            active is not None
            and active.action_index == 0
            and active.actions is not feedback.stage_actions
        ):
            # PolicyEvaluator now starts the Stage before applying an action so
            # preconditions and an already-held PLACE binding exist in time.
            # Adopt the config-driven policy's materialized actions before
            # consuming its first result; these carry randomization and the
            # resolved goal produced by the action that just ran.
            active.actions = feedback.stage_actions
            active.completion_motion_goal = None

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
            use_configured_identity=self.context.is_object_only,
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

        binding_failure = self._synchronize_external_grasp_binding(
            env_index,
            active,
        )
        if binding_failure is not None:
            binding_failure = self.record_failure(
                env_index,
                active.plan,
                binding_failure,
            )
            self._set_failed(state, binding_failure)
            return _EnvUpdateEvent(failed=True)
        completion_failure = self._resolve_external_completion_goal(
            env_index,
            active,
        )
        if completion_failure is not None:
            completion_failure = self.record_failure(
                env_index,
                active.plan,
                completion_failure,
            )
            self._set_failed(state, completion_failure)
            return _EnvUpdateEvent(failed=True)

        if feedback.signal in {ControlSignal.TIMED_OUT, ControlSignal.FAILED}:
            failure = self._build_action_failure_details(
                active.plan,
                feedback.details,
                feedback.signal,
                env_index=env_index,
            )
            failure = self.record_failure(env_index, active.plan, failure)
            self._set_failed(state, failure)
            return _EnvUpdateEvent(failed=True)

        if not feedback.stage_action_sequence_done:
            self._set_policy_running(state, env_index, feedback.details)
            return _EnvUpdateEvent()

        success_failure = self._final_stage_failure(env_index, active)
        if success_failure is not None:
            success_failure = self.record_failure(
                env_index,
                active.plan,
                success_failure,
            )
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
        if self.context.is_object_only:
            carried = self.context.get_logical_carried_object(env_index)
            if plan.stage.operation == Operation.PICK:
                failure = (
                    {
                        "event": "stage_precondition_failed",
                        "failure_stage": "precondition",
                        "failure_category": "unexpected_logical_carry",
                        "failure_reason": (
                            f"environment already carries {carried!r} before pick"
                        ),
                        "execution_mode": "object_only",
                        "env_index": env_index,
                    }
                    if carried is not None
                    else None
                )
            elif plan.stage.operation == Operation.PLACE:
                failure = (
                    {
                        "event": "stage_precondition_failed",
                        "failure_stage": "precondition",
                        "failure_category": "missing_logical_carry",
                        "failure_reason": "place requires a preceding logical pick",
                        "execution_mode": "object_only",
                        "env_index": env_index,
                    }
                    if carried is None
                    else None
                )
            else:
                failure = {
                    "event": "stage_precondition_failed",
                    "failure_stage": "precondition",
                    "failure_category": "unsupported_object_only_operation",
                    "failure_reason": ("object_only supports only pick/place stages"),
                    "execution_mode": "object_only",
                    "env_index": env_index,
                }
        else:
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
            failure = self.record_failure(env_index, plan, failure)
            self._set_failed(state, failure)
            return _EnvUpdateEvent(failed=True)

        active = self._start_stage(
            env_index,
            plan,
            actions_override=actions_override,
        )
        state.active = active
        if resolve_completion_pose and self.completion_pose_resolver is not None:
            try:
                active.completion_motion_goal = self.completion_pose_resolver(
                    env_index,
                    active,
                )
            except (KeyError, NotImplementedError, RuntimeError, ValueError) as error:
                failure = self.record_failure(
                    env_index,
                    plan,
                    self._completion_resolution_failure(
                        env_index,
                        active,
                        error,
                    ),
                )
                self._set_failed(state, failure)
                return _EnvUpdateEvent(failed=True)
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
        operator = (
            None
            if self.context.is_object_only
            else backend.get_operator_handler(plan.operator_name)
        )
        target = backend.get_object_handler(plan.stage.object)
        initial_object_pose = (
            None if target is None else target.get_pose().select(env_index)
        )
        held_object_name = (
            self.context.get_logical_carried_object(env_index)
            if self.context.is_object_only and plan.stage.operation == Operation.PLACE
            else backend.get_grasped_object_name(plan.operator_name, env_index)
            if plan.stage.operation == Operation.PLACE
            else None
        )
        if (
            not self.context.is_object_only
            and held_object_name
            and self.context.get_grasp_binding(
                env_index,
                plan.operator_name,
            )
            is None
        ):
            # A task may begin at an already-grasped PLACE stage or arrive
            # here through an external policy.  The PERFORM condition has
            # already verified that the operator is holding something, so a
            # missing binding can be bootstrapped exactly once at this edge.
            self.context.capture_grasp_binding(
                env_index,
                plan.operator_name,
                held_object_name,
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
            failure = self.record_failure(env_index, active.plan, failure)
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
            mid_failure = self.record_failure(env_index, active.plan, mid_failure)
            self._set_failed(state, mid_failure)
            return self._event_for_completed_action(
                mode=mode,
                completed_position=completed_position,
                completed_keypoint=completed_keypoint,
                failed=True,
            )

        binding_failure = self._update_grasp_binding_after_eef(
            env_index,
            active,
            action,
        )
        if binding_failure is not None:
            binding_failure = self.record_failure(
                env_index,
                active.plan,
                binding_failure,
            )
            self._set_failed(state, binding_failure)
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
            success_failure = self.record_failure(
                env_index,
                active.plan,
                success_failure,
            )
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

    def _update_grasp_binding_after_eef(
        self,
        env_index: int,
        active: ActiveStageState,
        completed_action: PrimitiveAction,
    ) -> Optional[Dict[str, Any]]:
        """Capture verified grasps and clear bindings after verified release."""
        if completed_action.kind != "eef" or completed_action.eef is None:
            return None
        operator_name = active.plan.operator_name
        if not completed_action.eef.close:
            # The accepted open primitive is the lifecycle boundary.  PLACED
            # keeps its held identity and semantic goal on ActiveStageState,
            # while the binding itself must no longer authorize held-object
            # commands after release.
            self.context.clear_grasp_binding(env_index, operator_name)
            return None

        object_name = self.context.backend.get_grasped_object_name(
            operator_name,
            env_index,
        )
        if completed_action.eef.require_grasp:
            object_name = active.plan.stage.object
        if not object_name:
            return None

        existing = self.context.get_grasp_binding(env_index, operator_name)
        if existing is not None:
            if existing.object_name == object_name:
                # Never silently rebind a retained grasp: doing so would make
                # attachment slip look like a new nominal transform.
                return None
            return {
                "event": "grasp_binding_identity_mismatch",
                "failure_stage": "execution",
                "failure_category": "grasp_binding_identity_mismatch",
                "failure_reason": (
                    "operator already has a binding for "
                    f"{existing.object_name!r} but verified {object_name!r}"
                ),
                "env_index": env_index,
                "operator": operator_name,
                "operation": active.plan.stage.operation.value,
                "target_object": active.plan.stage.object,
            }
        try:
            self.context.capture_grasp_binding(
                env_index,
                operator_name,
                object_name,
            )
        except (KeyError, RuntimeError, ValueError) as error:
            return {
                "event": "grasp_binding_capture_failed",
                "failure_stage": "execution",
                "failure_category": "grasp_binding_capture_failed",
                "failure_reason": str(error),
                "env_index": env_index,
                "operator": operator_name,
                "operation": active.plan.stage.operation.value,
                "target_object": active.plan.stage.object,
            }
        return None

    def _synchronize_external_grasp_binding(
        self,
        env_index: int,
        active: ActiveStageState,
    ) -> Optional[Dict[str, Any]]:
        """Synchronize binding state at an external-policy observation edge."""
        # ``object_only`` deliberately instantiates no operator handlers.  An
        # external policy may still drive the scene directly (for example by
        # calling ``context.acquire_logical_object``/``apply_object_pose``),
        # but there is no physical grasp state to synchronize here.  Skipping
        # this probe keeps that path backend-independent and avoids turning a
        # valid object-only policy update into ``Unknown operator 'object_only'``.
        if self.context.is_object_only:
            return None
        operator_name = active.plan.operator_name
        is_grasping = bool(
            self.context.backend.is_operator_grasping(operator_name)[env_index]
        )
        object_name = (
            self.context.backend.get_grasped_object_name(
                operator_name,
                env_index,
            )
            if is_grasping
            else None
        )
        existing = self.context.get_grasp_binding(env_index, operator_name)
        if object_name is None:
            if existing is not None and active.plan.stage.operation in {
                Operation.PLACE,
                Operation.RELEASE,
            }:
                self.context.clear_grasp_binding(env_index, operator_name)
            return None
        if existing is not None:
            if existing.object_name == object_name:
                return None
            return {
                "event": "grasp_binding_identity_mismatch",
                "failure_stage": "execution",
                "failure_category": "grasp_binding_identity_mismatch",
                "failure_reason": (
                    "operator already has a binding for "
                    f"{existing.object_name!r} but now holds {object_name!r}"
                ),
                "env_index": env_index,
                "operator": operator_name,
                "operation": active.plan.stage.operation.value,
                "target_object": active.plan.stage.object,
            }
        try:
            self.context.capture_grasp_binding(
                env_index,
                operator_name,
                object_name,
            )
        except (KeyError, RuntimeError, ValueError) as error:
            return {
                "event": "grasp_binding_capture_failed",
                "failure_stage": "execution",
                "failure_category": "grasp_binding_capture_failed",
                "failure_reason": str(error),
                "env_index": env_index,
                "operator": operator_name,
                "operation": active.plan.stage.operation.value,
                "target_object": active.plan.stage.object,
            }
        return None

    def _resolve_external_completion_goal(
        self,
        env_index: int,
        active: ActiveStageState,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a deferred external-policy goal once its binding exists."""
        # Object-only execution has no operator or EEF completion pose.  Its
        # external-policy contract is expressed through logical object state
        # and held-object waypoints, so the physical completion resolver must
        # not reject a stage merely because that EEF goal is unavailable.
        if self.context.is_object_only:
            if active.plan.stage.operation != Operation.PLACE:
                return None
            held_object_name = active.held_object_name
            if not held_object_name:
                return self._completion_resolution_failure(
                    env_index,
                    active,
                    RuntimeError("object-only place has no logically carried object"),
                )
            carried_handler = self.context.backend.get_object_handler(held_object_name)
            if carried_handler is None:
                return self._completion_resolution_failure(
                    env_index,
                    active,
                    KeyError(f"Unknown carried object {held_object_name!r}"),
                )
            virtual_object_pose = carried_handler.get_pose().select(env_index)
            for action in active.actions:
                if action.kind != "object_pose" or action.pose is None:
                    continue
                if action.resolved_object_motion_goal is not None:
                    virtual_object_pose = (
                        action.resolved_object_motion_goal.object_world_pose
                    )
                    continue
                try:
                    action.resolved_object_motion_goal = resolve_object_motion_goal(
                        env_index=env_index,
                        object_name=held_object_name,
                        pose=action.pose,
                        target=active.target,
                        backend=self.context.backend,
                        reference_site=active.plan.stage.site,
                        current_object_pose=virtual_object_pose,
                    )
                    virtual_object_pose = (
                        action.resolved_object_motion_goal.object_world_pose
                    )
                except (
                    KeyError,
                    NotImplementedError,
                    RuntimeError,
                    ValueError,
                ) as error:
                    return self._completion_resolution_failure(
                        env_index,
                        active,
                        error,
                    )
            return None
        if (
            active.completion_motion_goal is not None
            or self.completion_pose_resolver is None
        ):
            return None
        try:
            active.completion_motion_goal = self.completion_pose_resolver(
                env_index,
                active,
            )
        except (KeyError, NotImplementedError, RuntimeError, ValueError) as error:
            return self._completion_resolution_failure(
                env_index,
                active,
                error,
            )
        if (
            active.completion_motion_goal is None
            and self._requires_resolved_completion_goal(active)
        ):
            return self._completion_resolution_failure(
                env_index,
                active,
                RuntimeError(
                    "configured semantic completion goal could not be resolved"
                ),
            )
        return None

    @staticmethod
    def _requires_resolved_completion_goal(active: ActiveStageState) -> bool:
        success_constraint = OPERATION_CONDITIONS.get(
            active.plan.stage.operation,
            {},
        ).get(OperationConditionType.SUCCESS)
        if success_constraint == OperationConstraint.REACHED:
            return any(action.kind == "pose" for action in active.actions)
        if active.plan.stage.operation != Operation.PLACE:
            return False
        before_eef = True
        has_held_goal = False
        for action in active.actions:
            if action.kind == "eef":
                before_eef = False
            if (
                before_eef
                and action.kind == "pose"
                and action.pose is not None
                and action.pose.controlled_frame.kind == ControlledFrameKind.HELD_OBJECT
            ):
                has_held_goal = True
        return has_held_goal or (
            getattr(active.plan.stage.param, "placed_reference", "object") == "pre_move"
        )

    @staticmethod
    def _completion_resolution_failure(
        env_index: int,
        active: ActiveStageState,
        error: Exception,
    ) -> Dict[str, Any]:
        return {
            "event": "motion_goal_resolution_failed",
            "failure_stage": "execution",
            "failure_category": "motion_goal_resolution_failed",
            "failure_reason": str(error),
            "env_index": env_index,
            "operator": active.plan.operator_name,
            "operation": active.plan.stage.operation.value,
            "target_object": active.plan.stage.object,
        }

    def _poll_external_policy(
        self,
        env_index: int,
        state: _EnvRuntimeState,
    ) -> _EnvUpdateEvent:
        active = state.active
        if active is None:
            return _EnvUpdateEvent()
        binding_failure = self._synchronize_external_grasp_binding(
            env_index,
            active,
        )
        if binding_failure is not None:
            binding_failure = self.record_failure(
                env_index,
                active.plan,
                binding_failure,
            )
            self._set_failed(state, binding_failure)
            return _EnvUpdateEvent(failed=True)
        completion_failure = self._resolve_external_completion_goal(
            env_index,
            active,
        )
        if completion_failure is not None:
            completion_failure = self.record_failure(
                env_index,
                active.plan,
                completion_failure,
            )
            self._set_failed(state, completion_failure)
            return _EnvUpdateEvent(failed=True)
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
        if self.context.is_object_only:
            if (
                completed_action.kind == "object_acquire"
                and self.context.get_logical_carried_object(env_index)
                != active.plan.stage.object
            ):
                return {
                    "event": "logical_acquire_failed",
                    "failure_stage": "execution",
                    "failure_category": "logical_acquire_failed",
                    "failure_reason": "logical pick did not acquire its target",
                    "execution_mode": "object_only",
                    "env_index": env_index,
                }
            return None
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
                    completion_motion_goal=None,
                    target_object_pose=None,
                    target_motion_goal=None,
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
        if self.context.is_object_only:
            return self._final_object_only_stage_failure(env_index, active)
        if press_checked_at_eef and active.plan.stage.operation == Operation.PRESS:
            return None
        completion_motion_goal = (
            active.completion_motion_goal
            or self._completion_motion_goal_from_active(active)
        )
        target_motion_goal = self._target_motion_goal(active)
        target_object_pose = self._target_object_pose(
            env_index,
            active,
            target_motion_goal=target_motion_goal,
        )
        return check_stage_condition(
            env_index=env_index,
            context=self.context,
            plan=active.plan,
            condition_type=OperationConditionType.SUCCESS,
            initial_pose=active.initial_object_pose,
            completion_pose=(
                active.completion_pose or self._completion_pose_from_active(active)
            ),
            completion_motion_goal=completion_motion_goal,
            target_object_pose=target_object_pose,
            target_motion_goal=target_motion_goal,
            held_object_name=active.held_object_name,
        )

    def _final_object_only_stage_failure(
        self,
        env_index: int,
        active: ActiveStageState,
    ) -> Optional[Dict[str, Any]]:
        """Validate logical acquire/release and the final transported pose."""
        stage = active.plan.stage
        carried = self.context.get_logical_carried_object(env_index)
        if stage.operation == Operation.PICK:
            if carried == stage.object:
                return None
            return {
                "event": "stage_postcondition_failed",
                "failure_stage": "postcondition",
                "failure_category": "missing_logical_carry",
                "failure_reason": "logical pick did not leave the target carried",
                "execution_mode": "object_only",
                "env_index": env_index,
                "carried_object": carried or "",
            }

        if stage.operation != Operation.PLACE:
            return {
                "event": "stage_postcondition_failed",
                "failure_stage": "postcondition",
                "failure_category": "unsupported_object_only_operation",
                "failure_reason": "object_only supports only pick/place stages",
                "execution_mode": "object_only",
                "env_index": env_index,
            }
        if carried is not None:
            return {
                "event": "stage_postcondition_failed",
                "failure_stage": "postcondition",
                "failure_category": "logical_release_failed",
                "failure_reason": "logical place did not release the carried object",
                "execution_mode": "object_only",
                "env_index": env_index,
                "carried_object": carried,
            }
        held_name = active.held_object_name
        if not held_name:
            return {
                "event": "stage_postcondition_failed",
                "failure_stage": "postcondition",
                "failure_category": "missing_logical_carry",
                "failure_reason": "place lost its carried object identity",
                "execution_mode": "object_only",
                "env_index": env_index,
            }
        final_goal = next(
            (
                action.resolved_object_motion_goal
                for action in reversed(active.actions)
                if action.kind == "object_pose"
                and action.resolved_object_motion_goal is not None
            ),
            None,
        )
        if final_goal is None:
            return {
                "event": "stage_postcondition_failed",
                "failure_stage": "postcondition",
                "failure_category": "missing_object_goal",
                "failure_reason": "place has no resolved held-object goal",
                "execution_mode": "object_only",
                "env_index": env_index,
            }
        handler = self.context.backend.get_object_handler(held_name)
        if handler is None:
            return {
                "event": "stage_postcondition_failed",
                "failure_stage": "postcondition",
                "failure_category": "unknown_carried_object",
                "failure_reason": f"unknown carried object {held_name!r}",
                "execution_mode": "object_only",
                "env_index": env_index,
            }
        current = handler.get_pose().select(env_index)
        configured = final_goal.configured_pose
        frame_name = configured.controlled_frame.frame
        if frame_name is not None:
            try:
                current = self.context.backend.get_element_pose(
                    frame_name,
                    env_index,
                )
            except (KeyError, NotImplementedError, RuntimeError, ValueError) as error:
                return {
                    "event": "stage_postcondition_failed",
                    "failure_stage": "postcondition",
                    "failure_category": "object_frame_unavailable",
                    "failure_reason": str(error),
                    "execution_mode": "object_only",
                    "env_index": env_index,
                    "carried_object": held_name,
                }

        position_difference = np.asarray(
            current.position[0],
            dtype=np.float64,
        ) - np.asarray(
            final_goal.controlled_world_pose.position[0],
            dtype=np.float64,
        )
        position_error = float(np.linalg.norm(position_difference))
        tolerance = getattr(stage.param, "placed_tolerance", None)
        position_tolerance = tolerance.position if tolerance is not None else None
        orientation_tolerance = tolerance.orientation if tolerance is not None else None
        position_ok = position_within_tolerance_nullable(
            position_difference,
            position_tolerance,
        )
        orientation_goal = configured.orientation_goal
        if not _is_configured(orientation_tolerance):
            orientation_error = 0.0
            orientation_ok = True
        elif isinstance(orientation_goal, AxisAlignmentOrientationGoalConfig):
            # A partial orientation goal constrains only the configured axis;
            # in-plane twist is intentionally free.  PlacedToleranceConfig
            # documents list tolerances as invalid for this goal kind.
            if final_goal.target_axis_world is None or isinstance(
                orientation_tolerance, (list, np.ndarray)
            ):
                orientation_error = float("inf")
                orientation_ok = False
            else:
                current_axis_world = resolve_axis_in_world(
                    orientation_goal.controlled_axis,
                    current,
                )
                orientation_error = axis_alignment_error(
                    current_axis_world,
                    final_goal.target_axis_world,
                    orientation_goal.direction,
                )
                orientation_ok = orientation_error <= float(orientation_tolerance)
        else:
            orientation_error = float(
                quaternion_angular_distance(
                    current.orientation[0],
                    final_goal.controlled_world_pose.orientation[0],
                )
            )
            orientation_ok = orientation_within_tolerance_nullable(
                current.orientation[0],
                final_goal.controlled_world_pose.orientation[0],
                orientation_tolerance,
            )
        if position_ok and orientation_ok:
            return None
        return {
            "event": "stage_postcondition_failed",
            "failure_stage": "postcondition",
            "failure_category": "placement_failed",
            "failure_reason": "carried object is outside the placement tolerance",
            "execution_mode": "object_only",
            "env_index": env_index,
            "carried_object": held_name,
            "position_error": position_error,
            "orientation_error": orientation_error,
            "position_tolerance": position_tolerance,
            "orientation_tolerance": orientation_tolerance,
        }

    def _target_object_pose(
        self,
        env_index: int,
        active: ActiveStageState,
        *,
        target_motion_goal: Optional[ResolvedMotionGoal],
    ) -> Optional[PoseState]:
        if active.plan.stage.operation != Operation.PLACE:
            return None
        if (
            target_motion_goal is not None
            and target_motion_goal.controlled_object_name is not None
        ):
            return target_motion_goal.controlled_world_pose
        reference = getattr(active.plan.stage.param, "placed_reference", "object")
        target_name = active.plan.stage.object
        if reference == "object" and target_name:
            target = self.context.backend.get_object_handler(target_name)
            return None if target is None else target.get_pose().select(env_index)
        if target_motion_goal is not None:
            return target_motion_goal.controlled_world_pose
        return self._pre_move_end_pose(active)

    @staticmethod
    def _target_motion_goal(
        active: ActiveStageState,
    ) -> Optional[ResolvedMotionGoal]:
        if active.plan.stage.operation != Operation.PLACE:
            return None
        reference = getattr(active.plan.stage.param, "placed_reference", "object")
        last_goal: Optional[ResolvedMotionGoal] = None
        last_held_goal: Optional[ResolvedMotionGoal] = None
        for action in active.actions:
            if action.kind == "eef":
                break
            goal = action.resolved_motion_goal
            if goal is not None:
                last_goal = goal
                if goal.controlled_object_name is not None:
                    last_held_goal = goal
        if last_held_goal is not None:
            return last_held_goal
        if (
            active.completion_motion_goal is not None
            and active.completion_motion_goal.controlled_object_name is not None
        ):
            return active.completion_motion_goal
        if reference == "pre_move" or not active.plan.stage.object:
            return last_goal
        return None

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
            "angle": None if arc.angle is None else float(arc.angle),
            "arc_length": (None if arc.arc_length is None else float(arc.arc_length)),
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
    def _completion_motion_goal_from_active(
        active: ActiveStageState,
    ) -> Optional[ResolvedMotionGoal]:
        for action in reversed(active.actions):
            if action.kind == "pose" and action.resolved_motion_goal is not None:
                return action.resolved_motion_goal
        return None

    @staticmethod
    def _pre_move_end_pose(active: ActiveStageState) -> Optional[PoseState]:
        last_pose = None
        for action in active.actions:
            if action.kind == "eef":
                break
            if action.kind == "pose" and action.resolved_motion_goal is not None:
                last_pose = action.resolved_motion_goal.controlled_world_pose
            elif action.kind == "pose" and action.resolved_pose is not None:
                last_pose = action.resolved_pose
        if last_pose is None:
            return None
        if isinstance(last_pose, PoseState):
            return last_pose
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
    context: Any,
    plan: StageExecutionPlan,
    condition_type: OperationConditionType,
    initial_pose: Optional[PoseState] = None,
    completion_pose: Optional[PoseControlConfig] = None,
    completion_motion_goal: Optional[ResolvedMotionGoal] = None,
    target_object_pose: Optional[PoseState] = None,
    target_motion_goal: Optional[ResolvedMotionGoal] = None,
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
    target_grasp_required = (
        constraint == OperationConstraint.GRASPED
        and plan.stage.operation in {Operation.PICK, Operation.PULL}
    )
    target_grasped: Optional[bool] = None

    if constraint == OperationConstraint.GRASPED:
        if target_grasp_required:
            target_grasped = bool(
                object_name
                and backend.is_object_grasped(operator_name, object_name)[env_index]
            )
            satisfied = target_grasped
        else:
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
            completion_motion_goal.configured_pose.tolerance
            if completion_motion_goal is not None
            else getattr(completion_pose, "tolerance", None)
            if completion_pose
            else None
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
        if completion_motion_goal is not None:
            try:
                position_difference, orientation_error, _ = motion_goal_errors(
                    env_index=env_index,
                    operator=operator,
                    backend=backend,
                    goal=completion_motion_goal,
                    require_held=(
                        completion_motion_goal.configured_pose.controlled_frame.kind.value
                        == "held_object"
                    ),
                )
            except (KeyError, NotImplementedError, RuntimeError, ValueError):
                satisfied = False
            else:
                satisfied = position_within_tolerance(
                    position_difference,
                    position_tolerance,
                ) and orientation_error <= float(orientation_tolerance)
        elif completion_pose is None:
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
            target_motion_goal=target_motion_goal,
            held_object_name=held_object_name,
        )
    else:
        satisfied = True

    if satisfied:
        return None
    details = _condition_failure_details(
        env_index=env_index,
        context=context,
        plan=plan,
        condition_type=condition_type,
        constraint=constraint,
        is_grasping=is_grasping,
        completion_pose=completion_pose,
        completion_motion_goal=completion_motion_goal,
        target_object_pose=target_object_pose,
        target_motion_goal=target_motion_goal,
        held_object_name=held_object_name,
    )
    if target_grasp_required:
        details["is_target_grasped"] = bool(target_grasped)
        details["failure_reason"] = (
            f"operator is not grasping required target '{object_name}'"
        )
    return details


def _placed_condition_satisfied(
    *,
    env_index: int,
    context: Any,
    plan: StageExecutionPlan,
    is_grasping: bool,
    target_object_pose: Optional[PoseState],
    target_motion_goal: Optional[ResolvedMotionGoal],
    held_object_name: Optional[str],
) -> bool:
    if is_grasping:
        return False
    if target_object_pose is None:
        return True
    if not held_object_name:
        return False
    if (
        target_motion_goal is not None
        and target_motion_goal.controlled_object_name is not None
        and target_motion_goal.controlled_object_name != held_object_name
    ):
        return False
    handler = context.backend.get_object_handler(held_object_name)
    if handler is None:
        return False
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

    semantic_goal = (
        target_motion_goal
        if target_motion_goal is not None
        and target_motion_goal.controlled_object_name == held_object_name
        else None
    )
    if semantic_goal is not None:
        try:
            position_difference, orientation_error, _ = motion_goal_errors(
                env_index=env_index,
                operator=context.backend.get_operator_handler(plan.operator_name),
                backend=context.backend,
                goal=semantic_goal,
                require_held=False,
            )
        except (KeyError, NotImplementedError, RuntimeError, ValueError):
            return False
        position_ok = position_within_tolerance_nullable(
            position_difference,
            position_tolerance,
        )
        if not _is_configured(orientation_tolerance):
            orientation_ok = True
        elif isinstance(orientation_tolerance, (list, np.ndarray)):
            # Euler component masks are not a geometric axis tolerance.
            return False
        else:
            orientation_ok = orientation_error <= float(orientation_tolerance)
        return position_ok and orientation_ok

    current = handler.get_pose()
    position_difference = np.asarray(
        current.position[env_index], dtype=np.float64
    ) - np.asarray(target_object_pose.position[0], dtype=np.float64)
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
    context: Any,
    plan: StageExecutionPlan,
    condition_type: OperationConditionType,
    constraint: OperationConstraint,
    is_grasping: bool,
    completion_pose: Optional[PoseControlConfig],
    completion_motion_goal: Optional[ResolvedMotionGoal],
    target_object_pose: Optional[PoseState],
    target_motion_goal: Optional[ResolvedMotionGoal],
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
            "configured controlled frame is not within tolerance of the target",
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
            completion_motion_goal,
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
            plan,
            target_object_pose,
            target_motion_goal,
            held_object_name,
        )
    return details


def _add_reached_failure_details(
    details: Dict[str, Any],
    env_index: int,
    context: Any,
    plan: StageExecutionPlan,
    completion_pose: Optional[PoseControlConfig],
    completion_motion_goal: Optional[ResolvedMotionGoal],
) -> None:
    details["completion_pose_available"] = (
        completion_pose is not None or completion_motion_goal is not None
    )
    if completion_motion_goal is not None:
        operator = context.backend.get_operator_handler(plan.operator_name)
        details["target_pose"] = {
            "position": [
                float(value)
                for value in completion_motion_goal.controlled_world_pose.position[0]
            ],
            "orientation": [
                float(value)
                for value in completion_motion_goal.controlled_world_pose.orientation[0]
            ],
        }
        details["controlled_frame"] = (
            completion_motion_goal.configured_pose.controlled_frame.model_dump(
                mode="json"
            )
        )
        try:
            position_error, orientation_error, current_pose = motion_goal_errors(
                env_index=env_index,
                operator=operator,
                backend=context.backend,
                goal=completion_motion_goal,
                require_held=False,
            )
        except (KeyError, NotImplementedError, RuntimeError, ValueError) as error:
            details["motion_goal_error"] = str(error)
            return
        details["current_pose"] = {
            "position": [float(value) for value in current_pose.position[0]],
            "orientation": [float(value) for value in current_pose.orientation[0]],
        }
        details["position_error"] = float(np.linalg.norm(position_error))
        details["orientation_error"] = float(orientation_error)
        return
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
    context: Any,
    plan: StageExecutionPlan,
    target_object_pose: Optional[PoseState],
    target_motion_goal: Optional[ResolvedMotionGoal],
    held_object_name: Optional[str],
) -> None:
    details["held_object"] = held_object_name or ""
    if not held_object_name or target_object_pose is None:
        return
    handler = context.backend.get_object_handler(held_object_name)
    if handler is None:
        return
    if (
        target_motion_goal is not None
        and target_motion_goal.controlled_object_name == held_object_name
    ):
        details["controlled_frame"] = (
            target_motion_goal.configured_pose.controlled_frame.model_dump(mode="json")
        )
        try:
            position_error, orientation_error, current_pose = motion_goal_errors(
                env_index=env_index,
                operator=context.backend.get_operator_handler(plan.operator_name),
                backend=context.backend,
                goal=target_motion_goal,
                require_held=False,
            )
        except (KeyError, NotImplementedError, RuntimeError, ValueError) as error:
            details["motion_goal_error"] = str(error)
            return
        details["target_position"] = [
            float(value)
            for value in target_motion_goal.controlled_world_pose.position[0]
        ]
        details["current_position"] = [
            float(value) for value in current_pose.position[0]
        ]
        details["position_error"] = float(np.linalg.norm(position_error))
        details["target_orientation"] = [
            float(value)
            for value in target_motion_goal.controlled_world_pose.orientation[0]
        ]
        details["current_orientation"] = [
            float(value) for value in current_pose.orientation[0]
        ]
        details["orientation_error"] = float(orientation_error)
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
