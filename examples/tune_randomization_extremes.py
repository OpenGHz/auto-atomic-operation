"""Interactively inspect task randomization extreme cases.

Loads a task config via Hydra, extracts ``task.randomization`` plus the
initial pose/state values that define its defaults, opens the MuJoCo viewer,
and provides a small tkinter panel for switching between extreme
randomization cases. This helps verify whether configured ranges push objects
or operators outside a reasonable workspace.

Usage::

    python examples/tune_randomization_extremes.py
    python examples/tune_randomization_extremes.py --config-name cup_on_coaster
"""

from __future__ import annotations

import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable, Dict, List, Optional

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend, _resolve_arm_pose
from auto_atom.framework import (
    AutoAtomConfig,
    InitialPoseConfig,
    OperatorConfig,
    OperatorInitialState,
    OperatorRandomizationConfig,
    PoseRandomRange,
    PoseReference,
    RandomizationReference,
)
from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import TaskRunner
from auto_atom.utils.pose import (
    PoseState,
    compose_pose,
    euler_to_quaternion,
    inverse_pose,
    quaternion_to_rpy,
)

AXES = ("x", "y", "z", "roll", "pitch", "yaw")
POSITION_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class ReloadedTuningConfig:
    randomization: Dict[str, PoseRandomRange | OperatorRandomizationConfig]
    initial_poses: Dict[str, InitialPoseConfig]
    operator_initial_states: Dict[str, OperatorInitialState]


def _fmt(values, precision: int = 6) -> str:
    return ", ".join(f"{float(v):.{precision}f}" for v in values)


def _axis_range(rand_range: PoseRandomRange, axis: str) -> tuple[float, float]:
    raw = getattr(rand_range, axis, None)
    if raw is None:
        return (0.0, 0.0)
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        lo = 0.0 if raw[0] is None else float(raw[0])
        hi = 0.0 if raw[1] is None else float(raw[1])
        return (lo, hi)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return (0.0, 0.0)
    return (value, value)


def _with_offsets(
    base_pose: PoseState,
    offsets: Dict[str, float],
    rand_range: PoseRandomRange,
) -> PoseState:
    pose = base_pose.broadcast_to(base_pose.batch_size)
    position = pose.position.copy()
    orientation = pose.orientation.copy()
    is_absolute = rand_range.reference in (
        RandomizationReference.ABSOLUTE_WORLD,
        RandomizationReference.ABSOLUTE_BASE,
    )
    for env_index in range(pose.batch_size):
        for axis_index, axis in enumerate(POSITION_AXES):
            if axis not in offsets:
                continue
            if is_absolute:
                position[env_index, axis_index] = float(offsets[axis])
            else:
                position[env_index, axis_index] += float(offsets[axis])
        roll, pitch, yaw = quaternion_to_rpy(orientation[env_index])
        if is_absolute:
            roll = roll if "roll" not in offsets else float(offsets["roll"])
            pitch = pitch if "pitch" not in offsets else float(offsets["pitch"])
            yaw = yaw if "yaw" not in offsets else float(offsets["yaw"])
        else:
            roll += float(offsets.get("roll", 0.0))
            pitch += float(offsets.get("pitch", 0.0))
            yaw += float(offsets.get("yaw", 0.0))
        orientation[env_index] = euler_to_quaternion((roll, pitch, yaw))
    return PoseState(position=position, orientation=orientation)


def _parse_tuning_config(cfg: DictConfig) -> ReloadedTuningConfig:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")
    task_raw = raw.get("task")
    if not isinstance(task_raw, dict):
        raise TypeError("Config task must be a mapping.")
    task_cfg = AutoAtomConfig.model_validate(task_raw)

    operator_initial_states: Dict[str, OperatorInitialState] = {}
    operators_raw = raw.get("task_operators") or {}
    if not isinstance(operators_raw, dict):
        raise TypeError("Config task_operators must be a mapping.")
    for name, op_raw in operators_raw.items():
        if not isinstance(op_raw, dict):
            continue
        op_cfg = OperatorConfig.model_validate({**op_raw, "name": name})
        if op_cfg.initial_state is not None:
            operator_initial_states[name] = op_cfg.initial_state

    return ReloadedTuningConfig(
        randomization=dict(task_cfg.randomization),
        initial_poses=dict(task_cfg.initial_pose),
        operator_initial_states=operator_initial_states,
    )


def _apply_operator_initial_states(
    backend: MujocoTaskBackend,
    operator_initial_states: Dict[str, OperatorInitialState],
) -> None:
    for name, initial_state in operator_initial_states.items():
        handler = backend.operator_handlers.get(name)
        if handler is None:
            print(f"[reload_initial_state] skip unknown operator={name}")
            continue

        if initial_state.base_pose is not None:
            pose = _resolve_arm_pose(
                initial_state.base_pose,
                handler.get_base_pose().select(0),
            )
            pose = pose.broadcast_to(backend.batch_size)
            handler.env.override_operator_base_pose(
                handler.operator_name,
                pose.position,
                pose.orientation,
            )

        if initial_state.arm is not None:
            arm_config = initial_state.arm
            pose = _resolve_arm_pose(
                arm_config,
                handler.get_end_effector_pose().select(0),
            )
            if (
                not isinstance(arm_config, list)
                and arm_config.reference == PoseReference.BASE
            ):
                pos_w, quat_w = handler.env.base_to_world(
                    handler.operator_name,
                    np.asarray(pose.position, dtype=np.float32),
                    np.asarray(pose.orientation, dtype=np.float32),
                )
                pose = PoseState(position=pos_w, orientation=quat_w)
            handler.set_home_end_effector_pose(pose)

        if initial_state.eef is not None:
            handler._home_ctrl[:, handler.eef_ctrl_index] = float(initial_state.eef)
            handler.home()


@dataclass(frozen=True)
class RandomizationTarget:
    key: str
    label: str
    rand_range: PoseRandomRange
    get_default_pose: Callable[[], PoseState]
    apply_pose: Callable[[PoseState], None]
    get_current_pose: Callable[[], PoseState]
    get_base_pose: Optional[Callable[[], PoseState]] = None


@dataclass(frozen=True)
class ExtremeCase:
    name: str
    description: str
    offsets_by_target: Dict[str, Dict[str, float]]


def _collect_cli_overrides(argv: List[str]) -> List[str]:
    overrides: List[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--config-name", "--config-path"}:
            skip_next = True
            continue
        if arg.startswith("--config-name=") or arg.startswith("--config-path="):
            continue
        if arg == "--multirun" or arg.startswith("hydra."):
            continue
        if "=" in arg:
            overrides.append(arg)
            continue
        if index > 0 and argv[index - 1] in {"--config-name", "--config-path"}:
            continue
    return overrides


class RandomizationInspector:
    def __init__(
        self,
        root: tk.Tk,
        backend: MujocoTaskBackend,
        operator_initial_states: Optional[Dict[str, OperatorInitialState]] = None,
        reload_randomization_callback: Optional[Callable[[], None]] = None,
        full_reload_callback: Optional[Callable[[], None]] = None,
    ):
        self.root = root
        self.backend = backend
        self.env = backend.env
        self.operator_initial_states = dict(operator_initial_states or {})
        self.reload_randomization_callback = reload_randomization_callback
        self.full_reload_callback = full_reload_callback
        self.targets = self._collect_targets()
        self.cases = self._build_cases()
        self.case_index = 0
        self.rng = np.random.default_rng()

        root.title("Tune Randomization Extremes")
        root.geometry("760x720")

        outer = ttk.Frame(root, padding=10)
        outer.pack(fill="both", expand=True)

        summary = ttk.LabelFrame(outer, text="Randomization Summary")
        summary.pack(fill="x", pady=(0, 8))
        self.summary_text = tk.Text(summary, height=12, wrap="word")
        self.summary_text.pack(fill="x", padx=6, pady=6)
        self.summary_text.insert("1.0", self._summary_text())
        self.summary_text.config(state="disabled")

        controls = ttk.LabelFrame(outer, text="Extreme Cases")
        controls.pack(fill="x", pady=(0, 8))

        top = ttk.Frame(controls)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Case:").pack(side="left")
        self.case_var = tk.StringVar(value=self.cases[0].name)
        self.case_combo = ttk.Combobox(
            top,
            textvariable=self.case_var,
            state="readonly",
            values=[case.name for case in self.cases],
            width=54,
        )
        self.case_combo.pack(side="left", padx=6, fill="x", expand=True)
        self.case_combo.bind("<<ComboboxSelected>>", self._on_case_selected)

        buttons = ttk.Frame(controls)
        buttons.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Button(buttons, text="Prev", command=self.prev_case).pack(side="left")
        ttk.Button(buttons, text="Apply", command=self.apply_selected_case).pack(
            side="left", padx=6
        )
        ttk.Button(buttons, text="Next", command=self.next_case).pack(side="left")
        ttk.Button(
            buttons, text="Random Sample", command=self.apply_random_sample
        ).pack(side="left", padx=(12, 0))
        ttk.Button(buttons, text="Reset Default", command=self.reset_default).pack(
            side="left", padx=6
        )
        if self.reload_randomization_callback is not None:
            ttk.Button(
                buttons,
                text="Reload Randomization",
                command=self.reload_randomization_callback,
            ).pack(side="left", padx=(12, 0))
        if self.full_reload_callback is not None:
            ttk.Button(
                buttons,
                text="Full Reload",
                command=self.full_reload_callback,
            ).pack(side="left", padx=6)

        self.desc_var = tk.StringVar(value=self.cases[0].description)
        ttk.Label(controls, textvariable=self.desc_var, wraplength=700).pack(
            fill="x", padx=6, pady=(0, 6)
        )

        state = ttk.LabelFrame(outer, text="Current Poses")
        state.pack(fill="both", expand=True)
        self.state_text = tk.Text(state, height=20, wrap="word")
        self.state_text.pack(fill="both", expand=True, padx=6, pady=6)

        self.reset_default()

    def reload_randomization(
        self,
        tuning_config: ReloadedTuningConfig,
        preferred_case_name: Optional[str] = None,
    ) -> None:
        self._apply_reloaded_defaults(tuning_config)
        self.targets = self._collect_targets()
        self.cases = self._build_cases()
        self.summary_text.config(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", self._summary_text())
        self.summary_text.config(state="disabled")
        self.case_combo["values"] = [case.name for case in self.cases]
        self.case_index = 0
        if preferred_case_name is not None:
            for index, case in enumerate(self.cases):
                if case.name == preferred_case_name:
                    self.case_index = index
                    break
        self.case_var.set(self.cases[self.case_index].name)
        self.desc_var.set(self.cases[self.case_index].description)
        self.apply_selected_case()

    def _apply_reloaded_defaults(self, tuning_config: ReloadedTuningConfig) -> None:
        self.backend.randomization = dict(tuning_config.randomization)
        self.backend.initial_poses = dict(tuning_config.initial_poses)
        self.operator_initial_states = dict(tuning_config.operator_initial_states)

        self.backend.env.reset()
        for operator in self.backend.operator_handlers.values():
            operator.home()
        _apply_operator_initial_states(self.backend, self.operator_initial_states)
        if self.backend.initial_poses:
            self.backend._apply_initial_poses()  # type: ignore[attr-defined]
        if self.backend.camera_initial_poses:
            self.backend._apply_camera_initial_poses()  # type: ignore[attr-defined]

        self.backend._default_object_poses.clear()  # type: ignore[attr-defined]
        self.backend._default_operator_base_poses.clear()  # type: ignore[attr-defined]
        self.backend._default_operator_eef_poses.clear()  # type: ignore[attr-defined]
        self.backend._default_camera_poses.clear()  # type: ignore[attr-defined]
        self.backend._record_default_poses()  # type: ignore[attr-defined]
        self.backend.env.refresh_viewer()

    def _collect_targets(self) -> List[RandomizationTarget]:
        targets: List[RandomizationTarget] = []
        for name, rand in self.backend.randomization.items():
            if name in self.backend.object_handlers:
                if isinstance(rand, OperatorRandomizationConfig):
                    continue
                handler = self.backend.object_handlers[name]
                targets.append(
                    RandomizationTarget(
                        key=f"object:{name}",
                        label=f"object {name}",
                        rand_range=rand,
                        get_default_pose=lambda n=name,
                        h=handler: self.backend._default_object_poses.get(  # type: ignore[attr-defined]
                            n, h.get_pose()
                        ),
                        apply_pose=lambda pose, h=handler: h.set_pose(pose),
                        get_current_pose=lambda h=handler: h.get_pose(),
                        get_base_pose=None,
                    )
                )
                continue

            if name not in self.backend.operator_handlers:
                continue

            handler = self.backend.operator_handlers[name]
            if isinstance(rand, OperatorRandomizationConfig):
                if rand.base is not None:
                    targets.append(
                        RandomizationTarget(
                            key=f"operator-base:{name}",
                            label=f"operator {name} base",
                            rand_range=rand.base,
                            get_default_pose=lambda n=name,
                            h=handler: self.backend._default_operator_base_poses.get(  # type: ignore[attr-defined]
                                n, h.get_base_pose()
                            ),
                            apply_pose=lambda pose, h=handler: h.set_pose(pose),
                            get_current_pose=lambda h=handler: h.get_base_pose(),
                            get_base_pose=None,
                        )
                    )
                if rand.eef is not None:
                    targets.append(
                        RandomizationTarget(
                            key=f"operator-eef:{name}",
                            label=f"operator {name} eef",
                            rand_range=rand.eef,
                            get_default_pose=lambda n=name,
                            h=handler: self.backend._default_operator_eef_poses.get(  # type: ignore[attr-defined]
                                n, h.get_end_effector_pose()
                            ),
                            apply_pose=lambda pose,
                            h=handler: h.set_home_end_effector_pose(pose),
                            get_current_pose=lambda h=handler: h.get_end_effector_pose(),
                            get_base_pose=lambda h=handler: h.get_base_pose(),
                        )
                    )
            else:
                targets.append(
                    RandomizationTarget(
                        key=f"operator-eef:{name}",
                        label=f"operator {name} eef",
                        rand_range=rand,
                        get_default_pose=lambda n=name,
                        h=handler: self.backend._default_operator_eef_poses.get(  # type: ignore[attr-defined]
                            n, h.get_end_effector_pose()
                        ),
                        apply_pose=lambda pose, h=handler: h.set_home_end_effector_pose(
                            pose
                        ),
                        get_current_pose=lambda h=handler: h.get_end_effector_pose(),
                        get_base_pose=lambda h=handler: h.get_base_pose(),
                    )
                )
        existing_keys = {target.key for target in targets}
        zero_range = PoseRandomRange()
        for name, initial_state in self.operator_initial_states.items():
            handler = self.backend.operator_handlers.get(name)
            if handler is None:
                continue
            if (
                initial_state.base_pose is not None
                and f"operator-base:{name}" not in existing_keys
            ):
                targets.append(
                    RandomizationTarget(
                        key=f"operator-base:{name}",
                        label=f"operator {name} base",
                        rand_range=zero_range,
                        get_default_pose=lambda n=name,
                        h=handler: self.backend._default_operator_base_poses.get(  # type: ignore[attr-defined]
                            n, h.get_base_pose()
                        ),
                        apply_pose=lambda pose, h=handler: h.set_pose(pose),
                        get_current_pose=lambda h=handler: h.get_base_pose(),
                        get_base_pose=None,
                    )
                )
                existing_keys.add(f"operator-base:{name}")
            if (
                initial_state.arm is not None or initial_state.eef is not None
            ) and f"operator-eef:{name}" not in existing_keys:
                targets.append(
                    RandomizationTarget(
                        key=f"operator-eef:{name}",
                        label=f"operator {name} eef",
                        rand_range=zero_range,
                        get_default_pose=lambda n=name,
                        h=handler: self.backend._default_operator_eef_poses.get(  # type: ignore[attr-defined]
                            n, h.get_end_effector_pose()
                        ),
                        apply_pose=lambda pose, h=handler: h.set_home_end_effector_pose(
                            pose
                        ),
                        get_current_pose=lambda h=handler: h.get_end_effector_pose(),
                        get_base_pose=lambda h=handler: h.get_base_pose(),
                    )
                )
                existing_keys.add(f"operator-eef:{name}")
        return targets

    def _build_cases(self) -> List[ExtremeCase]:
        cases: List[ExtremeCase] = [
            ExtremeCase(
                name="default",
                description="No randomization offset. Restore every randomized target to its default pose.",
                offsets_by_target={},
            )
        ]

        non_zero_targets = [
            target
            for target in self.targets
            if any(
                _axis_range(target.rand_range, axis)[0]
                != _axis_range(target.rand_range, axis)[1]
                for axis in AXES
            )
        ]
        if non_zero_targets:
            all_min = {
                target.key: {
                    axis: _axis_range(target.rand_range, axis)[0]
                    for axis in AXES
                    if _axis_range(target.rand_range, axis) != (0.0, 0.0)
                }
                for target in non_zero_targets
            }
            all_max = {
                target.key: {
                    axis: _axis_range(target.rand_range, axis)[1]
                    for axis in AXES
                    if _axis_range(target.rand_range, axis) != (0.0, 0.0)
                }
                for target in non_zero_targets
            }
            cases.append(
                ExtremeCase(
                    name="all-min",
                    description="Apply every randomized axis at its minimum value at the same time.",
                    offsets_by_target=all_min,
                )
            )
            cases.append(
                ExtremeCase(
                    name="all-max",
                    description="Apply every randomized axis at its maximum value at the same time.",
                    offsets_by_target=all_max,
                )
            )

        for target in self.targets:
            for axis in AXES:
                axis_min, axis_max = _axis_range(target.rand_range, axis)
                if axis_min == axis_max:
                    continue
                cases.append(
                    ExtremeCase(
                        name=f"{target.label} {axis}=min",
                        description=f"Only {target.label} uses {axis} minimum {axis_min:.6f}; all other axes stay at default.",
                        offsets_by_target={target.key: {axis: float(axis_min)}},
                    )
                )
                cases.append(
                    ExtremeCase(
                        name=f"{target.label} {axis}=max",
                        description=f"Only {target.label} uses {axis} maximum {axis_max:.6f}; all other axes stay at default.",
                        offsets_by_target={target.key: {axis: float(axis_max)}},
                    )
                )

        return cases

    def _summary_text(self) -> str:
        if not self.targets:
            return "No supported task.randomization entries found in this config."
        lines = []
        for target in self.targets:
            parts = []
            for axis in AXES:
                lo, hi = _axis_range(target.rand_range, axis)
                if lo == 0.0 and hi == 0.0:
                    continue
                parts.append(f"{axis}=[{lo:.6f}, {hi:.6f}]")
            if target.rand_range.collision_radius != 0.05:
                parts.append(
                    f"collision_radius={float(target.rand_range.collision_radius):.6f}"
                )
            lines.append(f"{target.label}: " + (", ".join(parts) or "all zero"))
        return "\n".join(lines)

    def _set_state_text(self, text: str) -> None:
        self.state_text.config(state="normal")
        self.state_text.delete("1.0", "end")
        self.state_text.insert("1.0", text)
        self.state_text.config(state="disabled")

    def _refresh_state_text(
        self, title: str, case: Optional[ExtremeCase] = None
    ) -> None:
        lines = [title]
        if case is not None:
            lines.append(f"case: {case.name}")
            lines.append(case.description)
        lines.append("")
        for target in self.targets:
            pose = target.get_current_pose().select(0)
            roll, pitch, yaw = quaternion_to_rpy(pose.orientation[0])
            lines.append(target.label)
            lines.append(f"  position: [{_fmt(pose.position[0])}]")
            lines.append(f"  quat(xyzw): [{_fmt(pose.orientation[0])}]")
            lines.append(f"  rpy: [{_fmt((roll, pitch, yaw))}]")
            if target.key.startswith("operator-"):
                _, _, operator_name = target.key.partition(":")
                handler = self.backend.operator_handlers.get(operator_name)
                if handler is not None:
                    eef_ctrl = float(handler._home_ctrl[0, handler.eef_ctrl_index])
                    lines.append(f"  eef_ctrl: {eef_ctrl:.6f}")
            if case is not None and target.key in case.offsets_by_target:
                offsets = case.offsets_by_target[target.key]
                lines.append(
                    "  offsets: "
                    + ", ".join(
                        f"{axis}={value:.6f}" for axis, value in offsets.items()
                    )
                )
            lines.append("")
        self._set_state_text("\n".join(lines).rstrip() + "\n")

    @staticmethod
    def _sampled_pose_key(target: RandomizationTarget) -> Optional[str]:
        prefix, _, name = target.key.partition(":")
        if prefix == "object":
            return name
        if prefix == "operator-base":
            return f"{name}.base"
        if prefix == "operator-eef":
            return f"{name}.eef"
        return None

    def _sorted_targets_for_apply(self) -> List[RandomizationTarget]:
        """Order targets so entity-name-referenced entries resolve after their
        referents (delta-carry depends on the referenced pose being sampled)."""
        try:
            entity_order = self.backend._randomization_order()
        except Exception:
            return list(self.targets)
        order_index = {name: idx for idx, name in enumerate(entity_order)}

        def sort_key(target: RandomizationTarget) -> tuple:
            _, _, entity = target.key.partition(":")
            # within one operator entity, base must be applied before eef so
            # that an eef referencing "<name>.base" sees the sampled base.
            attr_priority = 0 if target.key.startswith("operator-base:") else 1
            return (order_index.get(entity, len(order_index)), attr_priority)

        return sorted(self.targets, key=sort_key)

    def _apply_case(self, case: ExtremeCase) -> None:
        for target in self.targets:
            target.apply_pose(target.get_default_pose())
        sampled_poses: Dict[str, PoseState] = {}
        for target in self._sorted_targets_for_apply():
            offsets = case.offsets_by_target.get(target.key) or {}
            reference = target.rand_range.reference
            if (
                reference == RandomizationReference.ABSOLUTE_BASE
                and target.get_base_pose is not None
            ):
                base_world = target.get_base_pose()
                default_in_base = compose_pose(
                    inverse_pose(base_world),
                    target.get_default_pose(),
                )
                sampled_in_base = _with_offsets(
                    default_in_base,
                    offsets,
                    target.rand_range,
                )
                sampled_pose = compose_pose(base_world, sampled_in_base)
            elif isinstance(reference, RandomizationReference):
                sampled_pose = _with_offsets(
                    target.get_default_pose(),
                    offsets,
                    target.rand_range,
                )
            else:
                # Entity-name reference: carry the referenced entity's delta
                # onto this target's default pose so the entry follows its
                # referent even when it has no offsets of its own.
                base_pose = self.backend._resolve_reference_base_pose(
                    reference,
                    sampled_poses,
                    target.get_default_pose(),
                )
                sampled_pose = _with_offsets(base_pose, offsets, target.rand_range)
            target.apply_pose(sampled_pose)
            sample_key = self._sampled_pose_key(target)
            if sample_key is not None:
                sampled_poses[sample_key] = sampled_pose
        self.env.refresh_viewer()
        self.case_var.set(case.name)
        self.desc_var.set(case.description)
        self._refresh_state_text("Applied extreme case.", case)
        print(f"[randomization_case] {case.name}")

    def _on_case_selected(self, _event=None) -> None:
        name = self.case_var.get()
        for index, case in enumerate(self.cases):
            if case.name == name:
                self.case_index = index
                self.apply_selected_case()
                return

    def apply_selected_case(self) -> None:
        self._apply_case(self.cases[self.case_index])

    def prev_case(self) -> None:
        self.case_index = (self.case_index - 1) % len(self.cases)
        self.apply_selected_case()

    def next_case(self) -> None:
        self.case_index = (self.case_index + 1) % len(self.cases)
        self.apply_selected_case()

    def reset_default(self) -> None:
        self.case_index = 0
        self._apply_case(self.cases[0])

    def apply_random_sample(self) -> None:
        offsets_by_target: Dict[str, Dict[str, float]] = {}
        for target in self.targets:
            offsets = {}
            for axis in AXES:
                low, high = _axis_range(target.rand_range, axis)
                if low == 0.0 and high == 0.0:
                    continue
                offsets[axis] = float(self.rng.uniform(low, high))
            if offsets:
                offsets_by_target[target.key] = offsets
        case = ExtremeCase(
            name="random-sample",
            description="A fresh random sample drawn uniformly from each configured range.",
            offsets_by_target=offsets_by_target,
        )
        self._apply_case(case)


class RandomizationInspectorApp:
    def __init__(
        self,
        root: tk.Tk,
        initial_cfg: DictConfig,
        config_name: str,
        overrides: List[str],
    ):
        self.root = root
        self.initial_cfg = initial_cfg
        self.config_name = config_name
        self.overrides = overrides
        self.runner: Optional[TaskRunner] = None
        self.backend: Optional[MujocoTaskBackend] = None
        self.inspector: Optional[RandomizationInspector] = None

    def _load_cfg(self) -> DictConfig:
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            config_dir=str(get_config_dir()),
            version_base=None,
            job_name="tune_randomization_extremes_reload",
        ):
            return compose(config_name=self.config_name, overrides=self.overrides)

    def _extract_tuning_config(self, cfg: DictConfig) -> ReloadedTuningConfig:
        return _parse_tuning_config(cfg)

    def _start_backend(self) -> None:
        cfg = self.initial_cfg
        task_file = prepare_task_file(cfg)
        runner = TaskRunner().from_config(task_file)
        backend = runner._context.backend
        if not isinstance(backend, MujocoTaskBackend):
            runner.close()
            raise TypeError("Only MujocoTaskBackend is supported.")
        backend.reset()
        backend.env.refresh_viewer()
        tuning_config = self._extract_tuning_config(cfg)
        self.runner = runner
        self.backend = backend
        self.inspector = RandomizationInspector(
            self.root,
            backend,
            operator_initial_states=tuning_config.operator_initial_states,
            reload_randomization_callback=self.reload_randomization,
            full_reload_callback=self.full_reload,
        )

    def reload_randomization(self) -> None:
        print(f"[reload_randomization] config_name={self.config_name}")
        if self.backend is None or self.inspector is None:
            self._start_backend()
            return
        preferred_case_name = self.inspector.case_var.get()
        cfg = self._load_cfg()
        tuning_config = self._extract_tuning_config(cfg)
        self.inspector.reload_randomization(
            tuning_config,
            preferred_case_name=preferred_case_name,
        )

    def full_reload(self) -> None:
        print(f"[full_reload] config_name={self.config_name}")
        preferred_case_name = (
            self.inspector.case_var.get() if self.inspector is not None else None
        )
        cfg = self._load_cfg()
        self.initial_cfg = cfg
        if self.runner is not None:
            self.runner.close()
            self.runner = None
            self.backend = None
            self.inspector = None
        for child in self.root.winfo_children():
            child.destroy()
        self._start_backend()
        if self.inspector is not None:
            self.inspector.reload_randomization(
                self._extract_tuning_config(cfg),
                preferred_case_name=preferred_case_name,
            )

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None
            self.backend = None
            self.inspector = None


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    root = tk.Tk()
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name or "pick_and_place"
    overrides = _collect_cli_overrides(sys.argv[1:])
    app = RandomizationInspectorApp(root, cfg, config_name, overrides)
    try:
        app.reload_randomization()

        def tick():
            if app.backend is not None:
                app.backend.env.refresh_viewer()
            root.after(50, tick)

        root.after(50, tick)
        root.mainloop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
