"""task configuration models (split from the former auto_atom.framework monolith)."""

import math
from collections.abc import Mapping
from typing import Dict, List, Optional, Tuple, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from auto_atom.config.execution import (
    ExecutionConfig,
    KeypointRangeConfig,
    KeypointSide,
    TaskKeypointConfig,
    TaskPhase,
)
from auto_atom.config.motion import StageConfig
from auto_atom.config.operations import Operation
from auto_atom.config.pose import PoseOverrideConfig
from auto_atom.config.randomization import RandomizationScopeConfig


class AutoAtomConfig(BaseModel):
    """Configuration for the AutoAtom operator."""

    model_config = ConfigDict(extra="forbid")

    stages: List[StageConfig]
    """A list of StageConfig objects, each representing a stage of the AutoAtom operator. The stages are executed in the order they are defined in the list."""
    env_name: str
    """The registered environment name used to resolve the basis environment instance for the selected scene."""
    seed: int | None = None
    """The run's root seed, shared by every randomness source (scene
    randomization, waypoint randomization, and camera noise).

    ``None`` means "no run seed was chosen": the run stays random, but it is
    resolved once into a concrete entropy-derived seed that is logged and
    reported, so the run can be replayed afterwards with
    ``task.seed=<that value>``. Any integer — **including ``0``** — is used
    verbatim; ``0`` is not a sentinel for "unseeded"."""
    initial_pose: Dict[str, PoseOverrideConfig] = Field(default_factory=dict)
    """Per-object initial pose overrides applied after the backend reset and
    before randomization. Keys are logical object names exposed by the selected
    backend."""
    randomization: RandomizationScopeConfig = Field(
        default_factory=RandomizationScopeConfig
    )
    """Global randomization scope applied at each reset.

    The scope groups the global default ``distribution`` / ``constraints``,
    the per-entity ``entities`` map and the per-camera ``cameras`` map.
    Objects accept either a direct ``PoseRandomRange`` or an advanced
    ``RandomizationSpec``. Operators use ``OperatorRandomizationConfig`` with
    explicit ``base`` and/or ``eef`` sub-entries.

    Bare entity ranges inherit the scope-wide ``distribution`` and
    ``constraints`` defaults; advanced specs are fully explicit. The placement
    strategy is scope-wide (``randomization.strategy``). Camera entries never
    inherit those defaults because cameras have no distribution or collision
    semantics.
    """
    camera_initial_pose: Dict[str, PoseOverrideConfig] = Field(default_factory=dict)
    """Per-camera initial pose overrides applied at each reset, before
    camera randomization records its defaults.

    Keys are logical camera names exposed by the selected backend. Each entry
    may set ``position`` and/or ``orientation`` (4-float quaternion xyzw or
    3-float Euler roll/pitch/yaw in radians). Omitted components preserve the
    backend-provided reset value.

    Example YAML::

        camera_initial_pose:
          env1_cam:
            position: [2.4, 0.6, -0.1]
            orientation: [-0.5, 0.5, 0.5, 0.5]   # xyzw
    """
    randomization_debug: bool = False

    @field_validator(
        "initial_pose",
        "randomization",
        "camera_initial_pose",
        mode="before",
    )
    @classmethod
    def _strip_none_keys(cls, v: object) -> object:
        """Remove ``None``-valued keys from nested mapping entries.

        Hydra/OmegaConf merges override ``key: null`` as ``key: None``
        rather than deleting the key.  Stripping them here lets child
        configs cleanly switch between forms without triggering Pydantic
        ``extra="forbid"`` errors.
        """
        if not isinstance(v, dict):
            return v

        def _strip(value: object) -> object:
            if isinstance(value, dict):
                return {
                    key: _strip(nested)
                    for key, nested in value.items()
                    if nested is not None
                }
            if isinstance(value, list):
                return [_strip(item) for item in value]
            return value

        return _strip(v)

    """When True the first N resets cycle through extreme poses (each axis at its min/max, then all-min and all-max) before switching to random sampling.  Use this to verify that configured ranges are not too large."""


class OperatorInitialState(BaseModel, frozen=True):
    """Optional override for an operator's home control state applied at reset.

    ``joint_positions`` uses the operator model's declared joint-space
    coordinates. The selected backend maps them to native state. The gripper
    remains controlled by the separate ``eef`` field.
    """

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    joint_positions: Dict[str, Union[float, List[float]]] = Field(default_factory=dict)
    """Operator joint-space coordinates, keyed by logical joint name.

    Values are applied through the operator home seam after the environment
    reset. A value may be a scalar or a non-empty sequence as declared by the
    operator model. The backend maps them to native state; they remain distinct
    from EEF user-space controls.
    """

    @field_validator("joint_positions", mode="after")
    @classmethod
    def _validate_joint_positions(
        cls, value: Dict[str, Union[float, List[float]]]
    ) -> Dict[str, Union[float, List[float]]]:
        for name, position in value.items():
            if not str(name).strip():
                raise ValueError("joint_positions keys must be non-empty names")
            values = position if isinstance(position, list) else [position]
            if not values:
                raise ValueError(
                    f"joint_positions['{name}'] must contain at least one value"
                )
            if not all(math.isfinite(float(component)) for component in values):
                raise ValueError(
                    f"joint_positions['{name}'] must contain only finite values"
                )
        return value

    eef_pose: Optional[
        Union[Tuple[float, float, float, float, float, float], PoseOverrideConfig]
    ] = None
    """Override for the operator's home end-effector pose.

    Supports two input forms:
    1. Compact six-value form: [x, y, z, yaw, pitch, roll]
    2. Structured dict: {position: [x,y,z], orientation: [roll,pitch,yaw] or [x,y,z,w]}
       - Both position and orientation are optional in structured format
       - orientation can be Euler angles (3 floats) or quaternion (4 floats)

    When omitted the backend-provided reset value is kept."""

    @field_validator("eef_pose", mode="before")
    @classmethod
    def _validate_legacy_eef_shape(cls, value: object) -> object:
        """Require the flat legacy EEF form to contain exactly six values."""
        if value is None or isinstance(
            value, (PoseOverrideConfig, Mapping, str, bytes)
        ):
            return value
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            return value
        if length != 6:
            raise ValueError(
                "eef_pose legacy form must contain exactly six values: "
                "[x, y, z, yaw, pitch, roll]"
            )
        return value

    @field_validator("eef_pose", mode="after")
    @classmethod
    def _validate_eef_values(
        cls,
        value: Optional[
            Union[Tuple[float, float, float, float, float, float], PoseOverrideConfig]
        ],
    ) -> Optional[
        Union[Tuple[float, float, float, float, float, float], PoseOverrideConfig]
    ]:
        """Reject non-finite values in the flat EEF form."""
        if value is None or isinstance(value, PoseOverrideConfig):
            return value
        if not all(math.isfinite(component) for component in value):
            raise ValueError("eef_pose must contain only finite values")
        return value

    eef: Optional[float] = None
    """Override value for the end-effector/gripper control.
    When omitted the backend-provided reset value is kept."""

    @field_validator("eef", mode="after")
    @classmethod
    def _validate_eef_control(cls, value: Optional[float]) -> Optional[float]:
        """Reject non-finite gripper controls before they reach a backend."""
        if value is not None and not math.isfinite(value):
            raise ValueError("eef must be finite")
        return value

    base_pose: Optional[PoseOverrideConfig] = None
    """Override for the operator's base pose.

    ``reference: world`` expresses the pose in the world frame. A named scene
    frame exposed by the backend expresses the base pose relative to that
    frame; the resolved world pose is sampled at setup/reset and then held
    fixed.
    """

    @model_validator(mode="after")
    def _validate_home_pose_sources(self) -> Self:
        if self.joint_positions and self.eef_pose is not None:
            raise ValueError(
                "joint_positions cannot be combined with eef_pose; configure "
                "one canonical arm home representation"
            )
        return self


class OperatorConfig(BaseModel):
    """Configuration for constructing an operator instance from YAML."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    """The unique operator name referenced by task stages.
    Defaults to empty; populated from the dict key in ``TaskFileConfig.task_operators``
    during validation, so YAML entries do not need to repeat the name."""

    initial_state: Optional[OperatorInitialState] = None
    """Optional initial control state applied to this operator on every reset.
    Overrides the backend-provided reset values for the specified fields."""


def _phase_waypoint_count(stage: StageConfig, phase: TaskPhase) -> int:
    """Return the number of selectable configured points in a stage phase."""
    if phase == TaskPhase.PRE_MOVE:
        return len(stage.param.pre_move)
    if phase == TaskPhase.POST_MOVE:
        return len(stage.param.post_move)
    if stage.operation in {
        Operation.GRASP,
        Operation.RELEASE,
        Operation.PICK,
        Operation.PLACE,
        Operation.PULL,
        Operation.PRESS,
    }:
        return 1
    if stage.operation == Operation.PUSH and stage.param.eef is not None:
        return 1
    return 0


class TaskFileConfig(BaseModel):
    """Top-level YAML schema for a runnable task file."""

    model_config = ConfigDict(
        use_attribute_docstrings=True,
        extra="allow",
        populate_by_name=True,
    )

    backend: ImportString
    """The backend to execute this task file. The backend should be registered in the ComponentRegistry and should be compatible with the selected scene."""
    task: AutoAtomConfig
    """The task-level configuration describing stages, scene, and environment selection."""
    execution: ExecutionConfig = ExecutionConfig()
    """Runner execution policy. Defaults preserve one-control-tick updates over
    the complete task."""
    task_operators: Dict[str, OperatorConfig] = {}
    """The operator definitions available to the selected backend for this task file,
    keyed by operator name. Using a mapping (rather than a list) lets Hydra overrides
    target individual operators by key, e.g. ``task_operators.arm.control.tolerance.position=0.01``.
    Use ``task_operators`` in YAML; ``env.operators`` is reserved for environment-level operator bindings."""

    @model_validator(mode="before")
    @classmethod
    def _reject_top_level_execution_fields(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        for field_name in (
            "interval_selection",
            "keypoint_selection",
            "update_boundary",
            "render_internal_updates",
            "max_internal_updates_per_update",
            "max_fast_forward_updates",
        ):
            if field_name in value:
                target = (
                    "execution.interval_selection.max_fast_forward_updates"
                    if field_name == "max_fast_forward_updates"
                    else f"execution.{field_name}"
                )
                raise ValueError(
                    f"Top-level {field_name} is not supported; use {target} instead"
                )
        return value

    @field_validator("task_operators", mode="after")
    @classmethod
    def _populate_operator_names(
        cls, value: Dict[str, OperatorConfig]
    ) -> Dict[str, OperatorConfig]:
        for key, op in value.items():
            if not op.name:
                op.name = key
            elif op.name != key:
                raise ValueError(
                    f"task_operators key '{key}' does not match operator name '{op.name}'. "
                    "Either omit the name field or make it match the key."
                )
        return value

    @model_validator(mode="after")
    def _validate_interval_selection(self) -> "TaskFileConfig":
        selection = self.execution.interval_selection
        if selection is None:
            return self

        stages_by_name: Dict[str, List[Tuple[int, StageConfig]]] = {}
        for index, stage in enumerate(self.task.stages):
            effective_name = stage.name or f"stage_{index}"
            stages_by_name.setdefault(effective_name, []).append((index, stage))

        phase_order = {
            TaskPhase.PRE_MOVE: 0,
            TaskPhase.EEF: 1,
            TaskPhase.POST_MOVE: 2,
        }
        side_order = {
            KeypointSide.BEFORE: 0,
            KeypointSide.AFTER: 1,
        }

        def resolve(
            field_name: str,
            keypoint: TaskKeypointConfig,
        ) -> Tuple[int, int, int, int]:
            matches = stages_by_name.get(keypoint.stage, [])
            if not matches:
                available = ", ".join(stages_by_name) or "<none>"
                raise ValueError(
                    f"execution.interval_selection.{field_name}.stage "
                    f"{keypoint.stage!r} "
                    f"does not match a task stage; available stages: {available}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"execution.interval_selection.{field_name}.stage "
                    f"{keypoint.stage!r} "
                    "is ambiguous because multiple stages use that name"
                )

            stage_index, stage = matches[0]
            count = _phase_waypoint_count(stage, keypoint.phase)
            if count == 0:
                raise ValueError(
                    f"execution.interval_selection.{field_name} references phase "
                    f"{keypoint.phase.value!r}, but stage {keypoint.stage!r} "
                    "does not execute that phase"
                )
            if keypoint.waypoint >= count:
                raise ValueError(
                    f"execution.interval_selection.{field_name}.waypoint "
                    f"{keypoint.waypoint} is out of range for "
                    f"{keypoint.stage}.{keypoint.phase.value}; expected 0..{count - 1}"
                )
            if keypoint.side is None:
                raise ValueError(
                    f"execution.interval_selection.{field_name}.side was not resolved"
                )
            return (
                stage_index,
                phase_order[keypoint.phase],
                int(keypoint.waypoint),
                side_order[keypoint.side],
            )

        start_order = resolve("start", selection.start)
        stop_order = resolve("stop", selection.stop)
        if start_order > stop_order:
            raise ValueError(
                "execution.interval_selection.start must not come after "
                "execution.interval_selection.stop in task execution order"
            )
        return self

    @model_validator(mode="after")
    def _validate_keypoint_selection(self) -> "TaskFileConfig":
        entries = self.execution.keypoint_selection
        if not entries:
            return self

        stages_by_name: Dict[str, List[Tuple[int, StageConfig]]] = {}
        for index, stage in enumerate(self.task.stages):
            effective_name = stage.name or f"stage_{index}"
            stages_by_name.setdefault(effective_name, []).append((index, stage))

        phase_order = {
            TaskPhase.PRE_MOVE: 0,
            TaskPhase.EEF: 1,
            TaskPhase.POST_MOVE: 2,
        }

        def resolve(
            field_name: str,
            entry: KeypointRangeConfig,
        ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
            matches = stages_by_name.get(entry.stage, [])
            if not matches:
                available = ", ".join(stages_by_name) or "<none>"
                raise ValueError(
                    f"execution.{field_name}.stage {entry.stage!r} does not match "
                    f"a task stage; available stages: {available}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"execution.{field_name}.stage {entry.stage!r} is ambiguous "
                    "because multiple stages use that name"
                )

            stage_index, stage = matches[0]
            if entry.phase is None:
                selected = [
                    (phase_order[phase], waypoint)
                    for phase in phase_order
                    for waypoint in range(_phase_waypoint_count(stage, phase))
                ]
                if not selected:
                    raise ValueError(
                        f"execution.{field_name} selects stage {entry.stage!r}, "
                        "which does not execute any keypoint"
                    )
            else:
                count = _phase_waypoint_count(stage, entry.phase)
                if count == 0:
                    raise ValueError(
                        f"execution.{field_name} references phase "
                        f"{entry.phase.value!r}, but stage {entry.stage!r} "
                        "does not execute that phase"
                    )
                if entry.waypoint is None:
                    selected = [
                        (phase_order[entry.phase], waypoint)
                        for waypoint in range(count)
                    ]
                else:
                    if entry.waypoint >= count:
                        raise ValueError(
                            f"execution.{field_name}.waypoint {entry.waypoint} is "
                            f"out of range for {entry.stage}.{entry.phase.value}; "
                            f"expected 0..{count - 1}"
                        )
                    selected = [(phase_order[entry.phase], int(entry.waypoint))]

            return (
                (stage_index, *min(selected)),
                (stage_index, *max(selected)),
            )

        previous_stop: Optional[Tuple[int, int, int]] = None
        for index, entry in enumerate(entries):
            start, stop = resolve(f"keypoint_selection[{index}]", entry)
            if previous_stop is not None and start <= previous_stop:
                raise ValueError(
                    f"execution.keypoint_selection[{index}] must select keypoints "
                    "after the previous entry in task execution order"
                )
            previous_stop = stop
        return self
