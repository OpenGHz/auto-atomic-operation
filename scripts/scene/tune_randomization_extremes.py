"""Interactively inspect task randomization extreme cases.

Loads a task config via Hydra, opens the MuJoCo viewer, and provides a small
tkinter panel for switching between extreme randomization cases. This helps
verify whether configured ranges push objects or operators outside a
reasonable workspace.

The tool only chooses values (each range's min, max, or midpoint). Every case
is applied by the backend's own reset with those values in place of the draws,
so the scene is exactly what the runtime produces for them; "Random Sample" is
a plain runtime reset. The Left / Right arrow keys step to the previous / next
case while the panel has keyboard focus.

Usage::

    python scripts/scene/tune_randomization_extremes.py
    python scripts/scene/tune_randomization_extremes.py task=cup_on_coaster
    python scripts/scene/tune_randomization_extremes.py task=open_door embodiment=p7_xf9600

Reloads re-compose ``aao_configs/config.yaml`` with the same command-line
overrides (including ``task=`` / ``embodiment=``). "Reload Randomization"
reconfigures the live backend and falls back to a full reload when the new
config needs different handlers.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import ttk

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from auto_atom.backend.mjc.mujoco_backend import (
    BackendRebuildRequired,
    MujocoTaskBackend,
)
from auto_atom.config.randomization import PoseRandomRange, pose_randomization_regions
from auto_atom.config.reference import RandomizationReference
from auto_atom.config_loader import compose_task_config
from auto_atom.randomization import POSITION_AXES, ROTATION_AXES
from auto_atom.randomization_executor import (
    FixedRandomization,
    FixedSample,
    RandomizationExecutor,
)
from auto_atom.runner.common import (
    get_config_dir,
    get_run_name,
    prepare_task_file,
    prepare_task_sections,
)
from auto_atom.runtime import TaskRunner
from auto_atom.utils.pose import quaternion_to_rpy


def _enable_high_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
    except ImportError:
        return
    # Newest API first. Older Windows lacks the function (AttributeError) or
    # the whole DLL (OSError), so fall through to the next one.
    try:
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def _env_float(name: str) -> float | None:
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


def _detected_tk_scaling(root: tk.Tk) -> float | None:
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


AXES = (*POSITION_AXES, *ROTATION_AXES)
DEFAULT_COLLISION_RADIUS = PoseRandomRange.model_fields["collision_radius"].default


def _fmt(values, precision: int = 6) -> str:
    return ", ".join(f"{float(v):.{precision}f}" for v in values)


def _reference_label(reference: RandomizationReference | str) -> str:
    return (
        reference.value if isinstance(reference, RandomizationReference) else reference
    )


def _configured_axes(region: PoseRandomRange) -> tuple[str, ...]:
    """Every axis a reset draws for ``region``, including fixed ``[0, 0]`` ranges.

    An unset axis keeps its baseline, while an explicit ``[0, 0]`` in an
    absolute reference sets that axis to zero, so the two stay distinct.
    """
    return tuple(axis for axis in AXES if region.axis_range(axis) is not None)


def _midpoint(bounds: tuple[float, float]) -> float:
    return 0.5 * (float(bounds[0]) + float(bounds[1]))


def _region_values(
    region: PoseRandomRange,
    pick: Callable[[tuple[float, float]], float],
) -> dict[str, float]:
    """One value per configured axis of ``region``, chosen by ``pick``."""
    return {
        axis: float(pick(region.axis_range(axis))) for axis in _configured_axes(region)
    }


def _region_label(
    label: str,
    regions: tuple[PoseRandomRange, ...],
    region_index: int,
) -> str:
    """Format a target label, disambiguating multi-region entries."""
    if len(regions) == 1:
        return label
    return f"{label} [region {region_index}]"


@dataclass(frozen=True)
class ExtremeCase:
    """One scene to inspect: a runtime reset with chosen values, or its baseline."""

    name: str
    description: str
    fixed: FixedRandomization | None
    """The values the reset applies; ``None`` suspends randomization instead."""


def _collect_cli_overrides(argv: list[str]) -> list[str]:
    overrides: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg in {"--config-name", "--config-path"}:
            skip_next = True
            continue
        if arg.startswith(("--config-name=", "--config-path=")):
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
    """Tk panel that shows a task's randomization at chosen values.

    The inspector only chooses values. Every case is applied by the backend's
    own ``reset()``, with the executor's value source switched to the case's
    :class:`FixedRandomization` (or suspended for the baseline), so what it
    shows is exactly what the runtime produces for those values.
    """

    def __init__(
        self,
        root: tk.Tk,
        backend: MujocoTaskBackend,
        reload_randomization_callback: Callable[[], None] | None = None,
        full_reload_callback: Callable[[], None] | None = None,
    ):
        self.root = root
        self.backend = backend
        self.env = backend.get_env()
        self.reload_randomization_callback = reload_randomization_callback
        self.full_reload_callback = full_reload_callback
        self.cases = self._build_cases()
        self.case_index = 0

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
        # Arrow keys step through the cases from anywhere in the panel. Binding
        # the toplevel replaces the previous inspector's keys after a reload.
        root.bind("<Left>", lambda _event: self.prev_case())
        root.bind("<Right>", lambda _event: self.next_case())

        self.desc_var = tk.StringVar(value=self.cases[0].description)
        ttk.Label(controls, textvariable=self.desc_var, wraplength=700).pack(
            fill="x", padx=6, pady=(0, 6)
        )

        state = ttk.LabelFrame(outer, text="Current Poses")
        state.pack(fill="both", expand=True)
        self.state_text = tk.Text(state, height=20, wrap="word", font="TkFixedFont")
        self.state_text.pack(fill="both", expand=True, padx=6, pady=6)

        self.reset_default()

    @property
    def executor(self) -> RandomizationExecutor:
        """Read on every use: a reconfigured backend has a new executor."""
        return self.backend.randomization_executor

    def _targets(self) -> dict[str, tuple[PoseRandomRange, ...]]:
        """Each action a reset samples, with its candidate regions."""
        return {
            label: pose_randomization_regions(action.randomization)
            for label, action in self.executor.samplable_actions().items()
        }

    def _joints(self) -> dict[str, tuple[float, float]]:
        return dict(self.executor.scope.joints)

    def reload_randomization(self, preferred_case_name: str | None = None) -> None:
        """Rebuild the cases for the backend's current configuration."""
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
        self.apply_selected_case()

    def _build_cases(self) -> list[ExtremeCase]:
        """Cases that each push one thing to an extreme.

        Whatever a case does not push sits at its range midpoint (first region
        for multi-region targets), so every case is a sample the runtime could
        draw — the baseline itself is the separate ``default`` case.
        """
        targets = self._targets()
        joints = self._joints()

        def fixed(
            poses: dict[str, FixedSample] | None = None,
            joint_values: dict[str, float] | None = None,
        ) -> FixedRandomization:
            samples = {
                label: FixedSample(0, _region_values(regions[0], _midpoint))
                for label, regions in targets.items()
            }
            samples.update(poses or {})
            values = {name: _midpoint(bounds) for name, bounds in joints.items()}
            values.update(joint_values or {})
            return FixedRandomization(poses=samples, joints=values)

        cases = [
            ExtremeCase(
                name="default",
                description=(
                    "The reset baseline: the runtime reset with randomization "
                    "suspended."
                ),
                fixed=None,
            )
        ]
        if not targets and not joints:
            return cases
        cases.append(
            ExtremeCase(
                name="center",
                description=(
                    "Every randomized axis and joint at its range midpoint "
                    "(first region of multi-region targets)."
                ),
                fixed=fixed(),
            )
        )
        for bound_name, pick in (("min", min), ("max", max)):
            cases.append(
                ExtremeCase(
                    name=f"all-{bound_name}",
                    description=(
                        f"Every randomized axis and joint at its {bound_name}imum "
                        "at the same time (first region of multi-region targets)."
                    ),
                    fixed=fixed(
                        {
                            label: FixedSample(0, _region_values(regions[0], pick))
                            for label, regions in targets.items()
                        },
                        {name: float(pick(bounds)) for name, bounds in joints.items()},
                    ),
                )
            )

        for label, regions in targets.items():
            for region_index, region in enumerate(regions):
                region_name = _region_label(label, regions, region_index)
                if len(regions) > 1:
                    for bound_name, pick in (("min", min), ("max", max)):
                        cases.append(
                            ExtremeCase(
                                name=f"{region_name} all-{bound_name}",
                                description=(
                                    f"All {region_name} axes at their "
                                    f"{bound_name}imum; everything else at its "
                                    "midpoint."
                                ),
                                fixed=fixed(
                                    {
                                        label: FixedSample(
                                            region_index,
                                            _region_values(region, pick),
                                        )
                                    }
                                ),
                            )
                        )
                for axis in _configured_axes(region):
                    for bound_name, pick in (("min", min), ("max", max)):
                        value = float(pick(region.axis_range(axis)))
                        values = _region_values(region, _midpoint)
                        values[axis] = value
                        cases.append(
                            ExtremeCase(
                                name=f"{region_name} {axis}={bound_name}",
                                description=(
                                    f"{region_name} {axis} at its {bound_name}imum "
                                    f"{value:.6f}; everything else at its midpoint."
                                ),
                                fixed=fixed({label: FixedSample(region_index, values)}),
                            )
                        )

        for name, bounds in joints.items():
            for bound_name, pick in (("min", min), ("max", max)):
                value = float(pick(bounds))
                cases.append(
                    ExtremeCase(
                        name=f"joint {name}={bound_name}",
                        description=(
                            f"Joint {name} at its {bound_name}imum {value:.6f}; "
                            "everything else at its midpoint."
                        ),
                        fixed=fixed(joint_values={name: value}),
                    )
                )
        return cases

    def _summary_text(self) -> str:
        targets = self._targets()
        joints = self._joints()
        if not targets and not joints:
            return "No supported task.randomization entries found in this config."
        lines = []
        for label, regions in targets.items():
            for region_index, region in enumerate(regions):
                parts = [f"reference={_reference_label(region.reference)}"]
                for axis in _configured_axes(region):
                    lo, hi = region.axis_range(axis)
                    axis_reference = region.axis_reference(axis)
                    suffix = (
                        ""
                        if axis_reference == region.reference
                        else f"@{_reference_label(axis_reference)}"
                    )
                    parts.append(f"{axis}=[{lo:.6f}, {hi:.6f}]{suffix}")
                if region.collision_radius != DEFAULT_COLLISION_RADIUS:
                    parts.append(
                        f"collision_radius={float(region.collision_radius):.6f}"
                    )
                lines.append(
                    f"{_region_label(label, regions, region_index)}: "
                    + ", ".join(parts)
                )
        for name, (low, high) in joints.items():
            lines.append(f"joint {name}: [{low:.6f}, {high:.6f}]")
        return "\n".join(lines)

    def _displayed_labels(self) -> list[str]:
        """Randomized targets, plus operator parts an initial state places."""
        labels = list(self._targets())
        for name, initial_state in self.backend.operator_initial_states.items():
            if name not in self.backend.operator_handlers:
                continue
            if initial_state.base_pose is not None and f"{name}.base" not in labels:
                labels.append(f"{name}.base")
            if (
                initial_state.eef_pose is not None or initial_state.eef is not None
            ) and f"{name}.eef" not in labels:
                labels.append(f"{name}.eef")
        return labels

    def _set_state_text(self, text: str) -> None:
        self.state_text.config(state="normal")
        self.state_text.delete("1.0", "end")
        self.state_text.insert("1.0", text)
        self.state_text.config(state="disabled")

    def _refresh_state_text(self, title: str, case: ExtremeCase | None = None) -> None:
        lines = [title]
        if case is not None:
            lines.append(f"case: {case.name}")
            lines.append(case.description)
        lines.append("")
        fixed = case.fixed if case is not None else None
        targets = self._targets()
        for label in self._displayed_labels():
            pose = self.backend.live_pose(label).select(0)
            roll, pitch, yaw = quaternion_to_rpy(pose.orientation[0])
            sample = fixed.poses.get(label) if fixed is not None else None
            lines.append(label)
            if sample is not None and len(targets.get(label, ())) > 1:
                lines.append(f"  region: {sample.region_index}")
            lines.append(f"  position: [{_fmt(pose.position[0])}]")
            lines.append(f"  quat(xyzw): [{_fmt(pose.orientation[0])}]")
            lines.append(f"  rpy: [{_fmt((roll, pitch, yaw))}]")
            owner, _, part = label.partition(".")
            handler = self.backend.operator_handlers.get(owner) if part else None
            if handler is not None:
                eef_ctrl = float(handler._home_ctrl[0, handler.eef_ctrl_index])
                lines.append(f"  eef_ctrl: {eef_ctrl:.6f}")
            if sample is not None and sample.axis_values:
                lines.append(
                    "  values: "
                    + ", ".join(
                        f"{axis}={value:.6f}"
                        for axis, value in sample.axis_values.items()
                    )
                )
            lines.append("")
        if fixed is not None and fixed.joints:
            lines.append(
                "joints: "
                + ", ".join(
                    f"{name}={value:.6f}" for name, value in fixed.joints.items()
                )
            )
        self._set_state_text("\n".join(lines).rstrip() + "\n")

    def _apply_case(self, case: ExtremeCase) -> None:
        executor = self.executor
        with (
            executor.suspended()
            if case.fixed is None
            else executor.fixed_samples(case.fixed)
        ):
            self.backend.reset()
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
        """One plain runtime reset: its own draws, rejection, and constraints."""
        self.backend.reset()
        description = (
            "One runtime reset: the backend's own draws, collision rejection, "
            "and constraints."
        )
        self.case_var.set("random-sample")
        self.desc_var.set(description)
        self._refresh_state_text(f"Applied a runtime reset. {description}")
        print("[randomization_case] random-sample")


class RandomizationInspectorApp:
    def __init__(
        self,
        root: tk.Tk,
        initial_cfg: DictConfig,
        run_name: str,
        overrides: list[str],
    ):
        self.root = root
        self.initial_cfg = initial_cfg
        self.run_name = run_name
        self.overrides = overrides
        self.runner: TaskRunner | None = None
        self.backend: MujocoTaskBackend | None = None
        self.inspector: RandomizationInspector | None = None

    def _load_cfg(self) -> DictConfig:
        GlobalHydra.instance().clear()
        # The CLI overrides carry the task=/embodiment=/... group selections.
        return compose_task_config(
            overrides=self.overrides, config_dir=get_config_dir()
        )

    def _start_backend(self) -> None:
        task_file = prepare_task_file(self.initial_cfg)
        runner = TaskRunner().from_config(task_file)
        backend = runner._context.backend
        if not isinstance(backend, MujocoTaskBackend):
            runner.close()
            raise TypeError("Only MujocoTaskBackend is supported.")
        # Surfacing borderline IK solutions is the whole point of this tool —
        # force the joint-limit-proximity warning on regardless of the env's
        # default (which is off, since it's noise during normal demos).
        backend.get_env().set_joint_limit_warning_enabled(True)
        self.runner = runner
        self.backend = backend
        self.inspector = RandomizationInspector(
            self.root,
            backend,
            reload_randomization_callback=self.reload_randomization,
            full_reload_callback=self.full_reload,
        )

    def reload_randomization(self) -> None:
        print(f"[reload_randomization] run={self.run_name}")
        if self.backend is None or self.inspector is None:
            self._start_backend()
            return
        preferred_case_name = self.inspector.case_var.get()
        cfg = self._load_cfg()
        task, operators = prepare_task_sections(cfg)
        try:
            self.backend.reconfigure(task, operators)
        except BackendRebuildRequired as exc:
            print(f"[reload_randomization] {exc} Falling back to a full reload.")
            self._rebuild(cfg, preferred_case_name)
            return
        self.inspector.reload_randomization(preferred_case_name=preferred_case_name)

    def full_reload(self) -> None:
        print(f"[full_reload] run={self.run_name}")
        preferred_case_name = (
            self.inspector.case_var.get() if self.inspector is not None else None
        )
        self._rebuild(self._load_cfg(), preferred_case_name)

    def _rebuild(self, cfg: DictConfig, preferred_case_name: str | None) -> None:
        self.initial_cfg = cfg
        self.close()
        for child in self.root.winfo_children():
            child.destroy()
        self._start_backend()
        if self.inspector is not None and preferred_case_name is not None:
            self.inspector.reload_randomization(preferred_case_name=preferred_case_name)
        # The rebuilt scene opens a new viewer window, which takes the keyboard
        # focus; the reload was requested from this panel, so hand it back for
        # the arrow keys to keep working.
        self.root.focus_force()

    def close(self) -> None:
        if self.runner is not None:
            self.runner.close()
            self.runner = None
            self.backend = None
            self.inspector = None


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _enable_high_dpi_awareness()
    root = tk.Tk()
    _configure_tk_dpi_and_fonts(root)
    overrides = _collect_cli_overrides(sys.argv[1:])
    app = RandomizationInspectorApp(root, cfg, get_run_name(), overrides)
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
