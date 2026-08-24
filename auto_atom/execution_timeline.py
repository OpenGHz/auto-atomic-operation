"""Compiled static ordering for Stage, Keypoint, and Primitive execution.

The execution timeline is a static seam. It owns the ordering and identity
needed by update boundaries and interval selection, while mutable controller
state (resolved poses, arc snapshots, cursors, and randomized waypoints)
remains in per-environment runtime action lists.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional

from .framework import (
    IntervalSelectionConfig,
    KeypointSide,
    TaskKeypointConfig,
    TaskPhase,
    UpdateBoundary,
    _phase_waypoint_count,
)
from .runtime import (
    PrimitiveAction,
    StageExecutionPlan,
    _EnvUpdateEvent,
    _ResolvedTaskKeypoint,
)

if TYPE_CHECKING:
    from .runtime import ExecutionContext, TaskFlowBuilder


@dataclass(frozen=True)
class CompiledKeypoint:
    """One configured keypoint and its flattened primitive span."""

    identity: _ResolvedTaskKeypoint
    primitive_start: int
    primitive_stop: int

    @property
    def primitive_count(self) -> int:
        return self.primitive_stop - self.primitive_start


@dataclass(frozen=True)
class ExecutionTimeline:
    """Immutable nominal ordering compiled from one builder pass."""

    stage_plans: tuple[StageExecutionPlan, ...]
    keypoints: tuple[CompiledKeypoint, ...]
    _stage_action_templates: tuple[tuple[PrimitiveAction, ...], ...] = field(repr=False)
    interval_selection: Optional[IntervalSelectionConfig] = None
    update_boundary: UpdateBoundary = UpdateBoundary.CONTROL_TICK
    max_internal_updates_per_update: int = 10_000
    _keypoint_indices: Mapping[_ResolvedTaskKeypoint, int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _stage_ranges: tuple[tuple[int, int], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _primitive_keypoints: tuple[Optional[_ResolvedTaskKeypoint], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    _configured_keypoint_indices: Mapping[
        tuple[str, TaskPhase, int], tuple[int, ...]
    ] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def compile(
        cls,
        builder: "TaskFlowBuilder",
        context: "ExecutionContext",
        *,
        validate_boundaries: bool = True,
    ) -> "ExecutionTimeline":
        """Compile plans, nominal actions, and boundary metadata once."""

        execution = context.task_file.execution
        compiled_plans: list[StageExecutionPlan] = []
        templates: list[tuple[PrimitiveAction, ...]] = []
        compiled_keypoints: list[CompiledKeypoint] = []
        stage_ranges: list[tuple[int, int]] = []
        last_orientation = None
        primitive_offset = 0
        strict = bool(
            validate_boundaries
            and (
                execution.update_boundary == UpdateBoundary.KEYPOINT
                or execution.interval_selection is not None
            )
        )

        for stage_index, stage in enumerate(context.config.stages):
            operator_name = builder._select_operator(stage, context.backend)
            actions, last_orientation = builder.build_actions(
                stage,
                last_orientation,
            )
            plan = StageExecutionPlan(
                stage_index=stage_index,
                stage=stage,
                operator_name=operator_name,
            )
            compiled_plans.append(plan)

            # Deep-copy the complete list so relative arc primitives retain
            # their intentional shared ArcExecutionSnapshot alias.
            nominal_actions = tuple(deepcopy(actions))
            templates.append(nominal_actions)
            stage_ranges.append(
                (primitive_offset, primitive_offset + len(nominal_actions))
            )
            if strict:
                _validate_stage_actions(builder, plan, list(nominal_actions))
            compiled_keypoints.extend(
                _collect_keypoints(
                    plan,
                    list(nominal_actions),
                    primitive_offset=primitive_offset,
                )
            )
            primitive_offset += len(nominal_actions)

        resolved_keypoints = tuple(compiled_keypoints)
        identities = tuple(item.identity for item in resolved_keypoints)
        primitive_keypoints: list[Optional[_ResolvedTaskKeypoint]] = [
            None
        ] * primitive_offset
        for item in resolved_keypoints:
            for primitive_index in range(item.primitive_start, item.primitive_stop):
                primitive_keypoints[primitive_index] = item.identity
        configured_indices: dict[tuple[str, TaskPhase, int], list[int]] = {}
        for index, identity in enumerate(identities):
            key = (identity.stage_name, identity.phase, identity.waypoint)
            configured_indices.setdefault(key, []).append(index)
        selection = execution.interval_selection if validate_boundaries else None
        if selection is not None:
            _validate_interval_selection(builder, selection, identities)

        return cls(
            stage_plans=tuple(compiled_plans),
            keypoints=resolved_keypoints,
            _stage_action_templates=tuple(templates),
            interval_selection=selection,
            update_boundary=execution.update_boundary,
            max_internal_updates_per_update=int(
                execution.max_internal_updates_per_update
            ),
            _keypoint_indices=MappingProxyType(
                {identity: index for index, identity in enumerate(identities)}
            ),
            _stage_ranges=tuple(stage_ranges),
            _primitive_keypoints=tuple(primitive_keypoints),
            _configured_keypoint_indices=MappingProxyType(
                {key: tuple(value) for key, value in configured_indices.items()}
            ),
        )

    def clone_stage_actions(self, stage_index: int) -> list[PrimitiveAction]:
        """Return an isolated, complete runtime copy for one stage."""

        try:
            if stage_index < 0:
                raise IndexError
            template = self._stage_action_templates[stage_index]
        except IndexError as exc:
            raise IndexError(f"Unknown stage index {stage_index}.") from exc
        return deepcopy(template)

    def stage_action_range(self, stage_index: int) -> range:
        try:
            if stage_index < 0:
                raise IndexError
            start, stop = self._stage_ranges[stage_index]
        except IndexError as exc:
            raise IndexError(f"Unknown stage index {stage_index}.") from exc
        return range(start, stop)

    def keypoint_index(self, keypoint: _ResolvedTaskKeypoint) -> int:
        try:
            return self._keypoint_indices[keypoint]
        except KeyError as exc:
            raise RuntimeError(
                "Completed keypoint is absent from the resolved interval plan"
            ) from exc

    def keypoint_for_action(
        self,
        stage_index: int,
        action_index: int,
    ) -> Optional[_ResolvedTaskKeypoint]:
        try:
            if stage_index < 0:
                raise IndexError
            start, stop = self._stage_ranges[stage_index]
            flattened_index = start + action_index
            if action_index < 0 or action_index >= stop - start:
                raise IndexError
        except IndexError as exc:
            raise IndexError(
                f"Unknown action {action_index} for stage {stage_index}."
            ) from exc
        return self._primitive_keypoints[flattened_index]

    def interval_keypoint_index(self, configured: TaskKeypointConfig) -> int:
        matches = self._configured_keypoint_indices.get(
            (configured.stage, configured.phase, configured.waypoint),
            (),
        )
        if len(matches) != 1:
            raise RuntimeError(
                "Interval keypoints must be resolved exactly once before execution"
            )
        return matches[0]

    def boundary_order_index(self, configured: TaskKeypointConfig) -> int:
        if configured.side is None:
            raise RuntimeError("Interval endpoint side was not resolved")
        return 2 * self.interval_keypoint_index(configured) + int(
            configured.side == KeypointSide.AFTER
        )

    def boundary_state_index(self, configured: TaskKeypointConfig) -> int:
        if configured.side is None:
            raise RuntimeError("Interval endpoint side was not resolved")
        return self.interval_keypoint_index(configured) + int(
            configured.side == KeypointSide.AFTER
        )

    def boundary_keypoint(
        self,
        configured: TaskKeypointConfig,
    ) -> _ResolvedTaskKeypoint:
        return self.keypoints[self.interval_keypoint_index(configured)].identity

    def completed_interval_state_index(
        self,
        completed: _ResolvedTaskKeypoint,
    ) -> int:
        return self.keypoint_index(completed) + 1

    def reached_update_boundary(self, event: _EnvUpdateEvent) -> Optional[str]:
        if self.update_boundary == UpdateBoundary.CONTROL_TICK and event.control_tick:
            return "control_tick"
        if self.update_boundary == UpdateBoundary.PRIMITIVE and event.primitive_reached:
            return "primitive_reached"
        if self.update_boundary == UpdateBoundary.KEYPOINT and event.keypoint_reached:
            return "keypoint_reached"
        if self.update_boundary == UpdateBoundary.STAGE and event.stage_succeeded:
            return "stage_succeeded"
        return None


def _collect_keypoints(
    plan: StageExecutionPlan,
    actions: list[PrimitiveAction],
    *,
    primitive_offset: int,
) -> list[CompiledKeypoint]:
    result: list[CompiledKeypoint] = []
    index = 0
    while index < len(actions):
        action = actions[index]
        if not isinstance(action.phase, TaskPhase) or not isinstance(
            action.waypoint, int
        ):
            index += 1
            continue
        identity = _ResolvedTaskKeypoint(
            stage_index=plan.stage_index,
            stage_name=plan.stage_name,
            phase=action.phase,
            waypoint=action.waypoint,
        )
        end = index + 1
        while end < len(actions):
            next_action = actions[end]
            if (next_action.phase, next_action.waypoint) != (
                action.phase,
                action.waypoint,
            ):
                break
            end += 1
        result.append(
            CompiledKeypoint(
                identity=identity,
                primitive_start=primitive_offset + index,
                primitive_stop=primitive_offset + end,
            )
        )
        index = end
    return result


def _validate_stage_actions(
    builder: "TaskFlowBuilder",
    plan: StageExecutionPlan,
    actions: list[PrimitiveAction],
) -> None:
    groups: list[tuple[tuple[TaskPhase, int], list[int]]] = []
    for action_index, action in enumerate(actions):
        if not isinstance(action.phase, TaskPhase) or not isinstance(
            action.waypoint, int
        ):
            raise ValueError(
                "Keypoint-aware execution requires every action emitted by "
                f"{type(builder).__name__} to define a TaskPhase phase "
                f"and integer waypoint; {plan.stage_name} action "
                f"{action_index} does not"
            )
        count = _phase_waypoint_count(plan.stage, action.phase)
        if action.waypoint < 0 or action.waypoint >= count:
            raise ValueError(
                f"{type(builder).__name__} emitted invalid keypoint "
                f"{plan.stage_name}.{action.phase.value}[{action.waypoint}] for "
                f"action {action_index}"
            )
        if not isinstance(action.completes_keypoint, bool):
            raise ValueError(
                f"{type(builder).__name__} must emit a boolean "
                f"completes_keypoint for {plan.stage_name} action "
                f"{action_index}"
            )
        identity = (action.phase, action.waypoint)
        if not groups or groups[-1][0] != identity:
            if any(group_identity == identity for group_identity, _ in groups):
                raise ValueError(
                    f"{type(builder).__name__} emitted non-contiguous primitives "
                    f"for keypoint {plan.stage_name}.{action.phase.value}"
                    f"[{action.waypoint}]"
                )
            groups.append((identity, []))
        groups[-1][1].append(action_index)

    for (phase, waypoint), indices in groups:
        completion_indices = [
            index for index in indices if actions[index].completes_keypoint
        ]
        if completion_indices != [indices[-1]]:
            raise ValueError(
                f"{type(builder).__name__} must mark only the final primitive "
                f"of keypoint {plan.stage_name}.{phase.value}[{waypoint}] with "
                "completes_keypoint=True"
            )


def _validate_interval_selection(
    builder: "TaskFlowBuilder",
    selection: IntervalSelectionConfig,
    keypoints: tuple[_ResolvedTaskKeypoint, ...],
) -> None:
    def resolve(field_name: str, configured: TaskKeypointConfig) -> int:
        matches = [
            index
            for index, keypoint in enumerate(keypoints)
            if keypoint.matches(configured)
        ]
        if not matches:
            raise ValueError(
                f"execution.interval_selection.{field_name} is not emitted by "
                f"{type(builder).__name__}: {configured.stage}."
                f"{configured.phase.value}[{configured.waypoint}]"
            )
        if len(matches) > 1:
            raise ValueError(
                f"execution.interval_selection.{field_name} is emitted more than "
                f"once by {type(builder).__name__}: {configured.stage}."
                f"{configured.phase.value}[{configured.waypoint}]"
            )
        return matches[0]

    start_order = 2 * resolve("start", selection.start) + int(
        selection.start.side == KeypointSide.AFTER
    )
    stop_order = 2 * resolve("stop", selection.stop) + int(
        selection.stop.side == KeypointSide.AFTER
    )
    if start_order > stop_order:
        raise ValueError(
            "execution.interval_selection.start must not come after "
            "execution.interval_selection.stop in the active TaskFlowBuilder"
        )
