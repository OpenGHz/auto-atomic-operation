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

import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass, field
from tkinter import ttk
from typing import Callable, Dict, List, Optional

import hydra
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.framework import (
    AutoAtomConfig,
    OperatorConfig,
    OperatorInitialState,
    OperatorRandomizationConfig,
    PoseOverrideConfig,
    PoseRandomizationSpec,
    PoseRandomRange,
    PoseReference,
    RandomizationReference,
    pose_randomization_regions,
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


def _enable_high_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
    except Exception:
        return
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def _env_float(name: str) -> Optional[float]:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return None
    try:
        return float(raw_value)
    except ValueError:
        print(f"[ui] ignore invalid {name}={raw_value!r}")
        return None


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        print(f"[ui] ignore invalid {name}={raw_value!r}")
        return default


def _clamped(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _detected_tk_scaling(root: tk.Tk) -> Optional[float]:
    dpi_values = []
    try:
        width_mm = float(root.winfo_screenmmwidth())
        height_mm = float(root.winfo_screenmmheight())
        if width_mm > 0:
            dpi_values.append(root.winfo_screenwidth() / (width_mm / 25.4))
        if height_mm > 0:
            dpi_values.append(root.winfo_screenheight() / (height_mm / 25.4))
    except tk.TclError:
        dpi_values = []
    dpi_values = [dpi for dpi in dpi_values if 60.0 <= dpi <= 360.0]
    if dpi_values:
        return sum(dpi_values) / len(dpi_values) / 72.0
    try:
        return float(root.winfo_fpixels("1i")) / 72.0
    except tk.TclError:
        return None


def _preferred_font_family(
    root: tk.Tk,
    candidates: tuple[str, ...],
    fallback: str,
) -> str:
    try:
        available_families = set(tkfont.families(root))
    except tk.TclError:
        return fallback
    for family in candidates:
        if family in available_families:
            return family
    return fallback


def _configure_font(font_name: str, family: str, size: int) -> None:
    try:
        font = tkfont.nametofont(font_name)
    except tk.TclError:
        return
    font.configure(family=family, size=size)


def _configure_tk_dpi_and_fonts(root: tk.Tk) -> None:
    root.update_idletasks()
    scaling = _env_float("AAO_TK_SCALING")
    if scaling is None:
        scaling = _detected_tk_scaling(root)
    if scaling is not None:
        root.tk.call("tk", "scaling", _clamped(scaling, 1.0, 3.0))

    default_font = tkfont.nametofont("TkDefaultFont")
    fixed_font = tkfont.nametofont("TkFixedFont")
    default_family = _preferred_font_family(
        root,
        ("Noto Sans", "Source Sans 3", "DejaVu Sans", "Liberation Sans", "Arial"),
        str(default_font.cget("family")),
    )
    fixed_family = _preferred_font_family(
        root,
        (
            "Noto Sans Mono",
            "Source Code Pro",
            "DejaVu Sans Mono",
            "Liberation Mono",
            "Consolas",
        ),
        str(fixed_font.cget("family")),
    )
    default_size = _env_int("AAO_TK_FONT_SIZE", 10)
    text_size = _env_int("AAO_TK_TEXT_FONT_SIZE", default_size)

    for font_name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
    ):
        _configure_font(font_name, default_family, default_size)
    _configure_font("TkHeadingFont", default_family, default_size + 1)
    _configure_font("TkFixedFont", fixed_family, text_size)

    style = ttk.Style(root)
    style.configure(".", font="TkDefaultFont")
    style.configure("TLabelFrame.Label", font="TkDefaultFont")


@dataclass(frozen=True)
class ReloadedTuningConfig:
    randomization: Dict[str, PoseRandomizationSpec | OperatorRandomizationConfig]
    initial_poses: Dict[str, PoseOverrideConfig]
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


def _region_label(target: RandomizationTarget, region_index: int) -> str:
    """Format a target label, disambiguating multi-region entries."""
    if len(target.regions) == 1:
        return target.label
    return f"{target.label} [region {region_index}]"


def _active_region_axes(rand_range: PoseRandomRange) -> tuple[str, ...]:
    """Return every explicitly configured axis, including fixed zero ranges.

    ``None`` means "leave the baseline unchanged", while an explicit
    ``[0, 0]`` in an absolute reference means "set this axis to zero".  Those
    cases must remain distinguishable in generated extreme and random cases.
    """
    return tuple(axis for axis in AXES if getattr(rand_range, axis, None) is not None)


def _sample_region_index(
    rng: np.random.Generator,
    region_count: int,
) -> int:
    """Sample an equiprobable region index, with a small test-double fallback."""
    if region_count <= 0:
        raise ValueError("Randomization region lists must not be empty")
    if region_count == 1:
        return 0
    integers = getattr(rng, "integers", None)
    if callable(integers):
        return int(integers(0, region_count))
    sampled = int(float(rng.uniform(0.0, float(region_count))))
    return max(0, min(sampled, region_count - 1))


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


@dataclass(frozen=True)
class RandomizationTarget:
    key: str
    label: str
    rand_range: PoseRandomizationSpec
    get_default_pose: Callable[[], PoseState]
    apply_pose: Callable[[PoseState], None]
    get_current_pose: Callable[[], PoseState]
    get_base_pose: Optional[Callable[[], PoseState]] = None

    @property
    def regions(self) -> tuple[PoseRandomRange, ...]:
        """Return the candidate regions for this physical target."""
        return pose_randomization_regions(self.rand_range)


@dataclass(frozen=True)
class ExtremeCase:
    name: str
    description: str
    offsets_by_target: Dict[str, Dict[str, float]]
    region_indices_by_target: Dict[str, int] = field(default_factory=dict)


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
        self.env = backend.get_env()
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
        self.summary_text = tk.Text(
            summary,
            height=12,
            wrap="word",
            font="TkFixedFont",
        )
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
        self.state_text = tk.Text(state, height=20, wrap="word", font="TkFixedFont")
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
        self.backend.operator_initial_states = dict(self.operator_initial_states)

        self.backend.get_env().reset()
        for operator in self.backend.operator_handlers.values():
            operator.home()
        if self.backend.initial_poses:
            self.backend._apply_initial_poses()  # type: ignore[attr-defined]
        self.backend.apply_operator_initial_states(home=True)
        if self.backend.camera_initial_poses:
            self.backend._apply_camera_initial_poses()  # type: ignore[attr-defined]

        self.backend._default_object_poses.clear()  # type: ignore[attr-defined]
        self.backend._default_operator_base_poses.clear()  # type: ignore[attr-defined]
        self.backend._default_operator_eef_poses.clear()  # type: ignore[attr-defined]
        self.backend._default_camera_poses.clear()  # type: ignore[attr-defined]
        self.backend._record_default_poses()  # type: ignore[attr-defined]
        self.backend.get_env().refresh_viewer()

    def _collect_targets(self) -> List[RandomizationTarget]:
        self.backend._validate_randomization_configuration()  # type: ignore[attr-defined]
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
                            get_default_pose=self._make_eef_default_getter(
                                name, handler
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
                        get_default_pose=self._make_eef_default_getter(name, handler),
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
                initial_state.eef_pose is not None or initial_state.eef is not None
            ) and f"operator-eef:{name}" not in existing_keys:
                targets.append(
                    RandomizationTarget(
                        key=f"operator-eef:{name}",
                        label=f"operator {name} eef",
                        rand_range=zero_range,
                        get_default_pose=self._make_eef_default_getter(name, handler),
                        apply_pose=lambda pose, h=handler: h.set_home_end_effector_pose(
                            pose
                        ),
                        get_current_pose=lambda h=handler: h.get_end_effector_pose(),
                        get_base_pose=lambda h=handler: h.get_base_pose(),
                    )
                )
                existing_keys.add(f"operator-eef:{name}")
        return targets

    def _make_eef_default_getter(
        self,
        name: str,
        handler,
    ) -> Callable[[], PoseState]:
        """Return a closure that yields the operator's default EEF pose
        re-anchored to the operator's **current** base, by delegating to
        ``MujocoTaskBackend._operator_default_eef_following_base`` so the
        runtime sampler and this tool agree on the same semantics.
        """
        backend = self.backend

        def _getter() -> PoseState:
            poses = []
            for env_index in range(backend.batch_size):
                follow_default, _ = backend._operator_default_eef_following_base(  # type: ignore[attr-defined]
                    name, handler, env_index, sampled_poses=None
                )
                poses.append(follow_default)
            return PoseState(
                position=np.stack([np.asarray(p.position[0]) for p in poses], axis=0),
                orientation=np.stack(
                    [np.asarray(p.orientation[0]) for p in poses], axis=0
                ),
            )

        return _getter

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
            if any(_active_region_axes(region) for region in target.regions)
        ]
        if non_zero_targets:
            selected_regions = {target.key: 0 for target in non_zero_targets}
            all_min = {
                target.key: {
                    axis: _axis_range(target.regions[0], axis)[0]
                    for axis in _active_region_axes(target.regions[0])
                }
                for target in non_zero_targets
            }
            all_max = {
                target.key: {
                    axis: _axis_range(target.regions[0], axis)[1]
                    for axis in _active_region_axes(target.regions[0])
                }
                for target in non_zero_targets
            }
            cases.append(
                ExtremeCase(
                    name="all-min",
                    description="Apply every randomized axis at its minimum value at the same time.",
                    offsets_by_target=all_min,
                    region_indices_by_target=selected_regions,
                )
            )
            cases.append(
                ExtremeCase(
                    name="all-max",
                    description="Apply every randomized axis at its maximum value at the same time.",
                    offsets_by_target=all_max,
                    region_indices_by_target=selected_regions.copy(),
                )
            )

        for target in self.targets:
            for region_index, rand_range in enumerate(target.regions):
                active_axes = _active_region_axes(rand_range)
                region_name = _region_label(target, region_index)
                region_selection = {target.key: region_index}

                if len(target.regions) > 1:
                    region_min = {
                        axis: _axis_range(rand_range, axis)[0] for axis in active_axes
                    }
                    region_max = {
                        axis: _axis_range(rand_range, axis)[1] for axis in active_axes
                    }
                    cases.append(
                        ExtremeCase(
                            name=f"{region_name} all-min",
                            description=(
                                f"Apply all {region_name} axes at their minimum "
                                "values; all other targets stay at default."
                            ),
                            offsets_by_target={target.key: region_min},
                            region_indices_by_target=region_selection,
                        )
                    )
                    cases.append(
                        ExtremeCase(
                            name=f"{region_name} all-max",
                            description=(
                                f"Apply all {region_name} axes at their maximum "
                                "values; all other targets stay at default."
                            ),
                            offsets_by_target={target.key: region_max},
                            region_indices_by_target=region_selection.copy(),
                        )
                    )

                for axis in active_axes:
                    axis_min, axis_max = _axis_range(rand_range, axis)
                    cases.append(
                        ExtremeCase(
                            name=f"{region_name} {axis}=min",
                            description=(
                                f"Only {region_name} uses {axis} minimum "
                                f"{axis_min:.6f}; all other axes stay at default."
                            ),
                            offsets_by_target={target.key: {axis: float(axis_min)}},
                            region_indices_by_target=region_selection,
                        )
                    )
                    cases.append(
                        ExtremeCase(
                            name=f"{region_name} {axis}=max",
                            description=(
                                f"Only {region_name} uses {axis} maximum "
                                f"{axis_max:.6f}; all other axes stay at default."
                            ),
                            offsets_by_target={target.key: {axis: float(axis_max)}},
                            region_indices_by_target=region_selection.copy(),
                        )
                    )

        return cases

    def _summary_text(self) -> str:
        if not self.targets:
            return "No supported task.randomization entries found in this config."
        lines = []
        for target in self.targets:
            for region_index, rand_range in enumerate(target.regions):
                reference = rand_range.reference
                reference_label = (
                    reference.value
                    if isinstance(reference, RandomizationReference)
                    else reference
                )
                parts = [f"reference={reference_label}"]
                for axis in _active_region_axes(rand_range):
                    lo, hi = _axis_range(rand_range, axis)
                    parts.append(f"{axis}=[{lo:.6f}, {hi:.6f}]")
                if rand_range.collision_radius != 0.05:
                    parts.append(
                        f"collision_radius={float(rand_range.collision_radius):.6f}"
                    )
                lines.append(
                    f"{_region_label(target, region_index)}: "
                    + (", ".join(parts) or "all zero")
                )
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
            if case is not None and target.key in case.region_indices_by_target:
                region_index = case.region_indices_by_target[target.key]
                if 0 <= region_index < len(target.regions):
                    lines.append(f"  selected_region: {region_index}")
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
        action_order = self.backend._randomization_order()
        order_index = {name: idx for idx, name in enumerate(action_order)}

        def sort_key(target: RandomizationTarget) -> tuple:
            action_key = self._sampled_pose_key(target)
            attr_priority = 0 if target.key.startswith("operator-base:") else 1
            return (
                order_index.get(action_key, len(order_index)),
                attr_priority,
            )

        return sorted(self.targets, key=sort_key)

    def _apply_case(self, case: ExtremeCase) -> None:
        # Reset every target to its default in the same dependency order we
        # later use to apply offsets (base before eef, etc.). For
        # operator-eef targets the "default" is computed against the
        # operator's *current* base via ``_make_eef_default_getter``, so
        # the base must already have been reset to its own default before
        # the eef default is queried — otherwise the eef would re-anchor
        # to whatever base happened to be left over from the previous
        # case and we'd see the very bug this ordering is meant to avoid.
        sorted_targets = self._sorted_targets_for_apply()
        for target in sorted_targets:
            target.apply_pose(target.get_default_pose())
        sampled_poses: Dict[str, PoseState] = {}
        for target in self._sorted_targets_for_apply():
            offsets = case.offsets_by_target.get(target.key) or {}
            region_index = case.region_indices_by_target.get(target.key, 0)
            if not 0 <= region_index < len(target.regions):
                raise ValueError(
                    f"Case '{case.name}' selects invalid region {region_index} "
                    f"for target '{target.key}' (expected 0..{len(target.regions) - 1})"
                )
            rand_range = target.regions[region_index]
            reference = rand_range.reference
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
                    rand_range,
                )
                sampled_pose = compose_pose(base_world, sampled_in_base)
            elif isinstance(reference, RandomizationReference):
                sampled_pose = _with_offsets(
                    target.get_default_pose(),
                    offsets,
                    rand_range,
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
                sampled_pose = _with_offsets(base_pose, offsets, rand_range)
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
        region_indices_by_target: Dict[str, int] = {}
        for target in self.targets:
            region_index = _sample_region_index(self.rng, len(target.regions))
            region_indices_by_target[target.key] = region_index
            rand_range = target.regions[region_index]
            offsets = {}
            for axis in _active_region_axes(rand_range):
                low, high = _axis_range(rand_range, axis)
                offsets[axis] = float(self.rng.uniform(low, high))
            if offsets:
                offsets_by_target[target.key] = offsets
        case = ExtremeCase(
            name="random-sample",
            description=(
                "A fresh random sample drawn uniformly from one selected "
                "region per target and each configured range."
            ),
            offsets_by_target=offsets_by_target,
            region_indices_by_target=region_indices_by_target,
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
        # Surfacing borderline IK solutions is the whole point of this tool —
        # force the joint-limit-proximity warning on regardless of the env's
        # default (which is off, since it's noise during normal demos).
        backend.get_env().set_joint_limit_warning_enabled(True)
        backend.reset()
        backend.get_env().refresh_viewer()
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
    _enable_high_dpi_awareness()
    root = tk.Tk()
    _configure_tk_dpi_and_fonts(root)
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name or "pick_and_place"
    overrides = _collect_cli_overrides(sys.argv[1:])
    app = RandomizationInspectorApp(root, cfg, config_name, overrides)
    try:
        app.reload_randomization()

        def tick():
            if app.backend is not None:
                app.backend.get_env().refresh_viewer()
            root.after(50, tick)

        root.after(50, tick)
        root.mainloop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
