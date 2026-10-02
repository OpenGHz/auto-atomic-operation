"""Console entry point that introspects the tasks in ``aao_configs/``.

Runnable tasks are the options of the ``task`` config group
(``aao_configs/task/*.yaml``; ``_``-prefixed fragments are skipped). Each task
is reported once for its default embodiment and once more for every
embodiment it has a dedicated adaptation for
(``aao_configs/adapt/<task>/<embodiment>.yaml``) — the validated
task x embodiment combinations. A variant is kept when, after Hydra
composition, it exposes a non-empty ``task.stages`` list.

For each variant it reports the Hydra overrides that select it, the objects it
manipulates, the operations it performs, and a workflow generated from the
ordered stages.

Usage::

    aao-info                 # list every runnable task variant
    aao-info pick_and_place  # only the named task(s)
    aao-info --json          # machine-readable output
    aao-info --verbose       # also report variants skipped as non-tasks
"""

from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from pydantic import BaseModel

from auto_atom.config_loader import PRIMARY_CONFIG_NAME, task_overrides
from auto_atom.execution_config import prepare_task_config_for_instantiation

from .common import get_config_dir

# Human-readable phrasing for each operation, used to generate the workflow.
# Keys match the values of ``auto_atom.config.operations.Operation``.
_OPERATION_PHRASE = {
    "move": "move to {obj}",
    "grasp": "grasp {obj}",
    "release": "release {obj}",
    "pick": "pick up {obj}",
    "place": "place onto {obj}",
    "push": "push {obj}",
    "pull": "pull {obj}",
    "press": "press {obj}",
}

# Phrasing when a stage carries no target object (target pose comes from param).
_OPERATION_PHRASE_NO_OBJECT = {
    "move": "move to a target pose",
    "grasp": "close the gripper",
    "release": "open the gripper",
    "pick": "pick up (no target object)",
    "place": "place down at a target pose",
    "push": "push to a target pose",
    "pull": "pull to a target pose",
    "press": "press at a target pose",
}


class StageInfo(BaseModel):
    """One stage of a task, as declared in ``task.stages``."""

    index: int
    name: str = ""
    object: str = ""
    operation: str = ""
    operator: str = ""
    site: Optional[str] = None

    def describe(self) -> str:
        """Return a human-readable phrase for this stage's action."""
        op = self.operation
        if self.object:
            template = _OPERATION_PHRASE.get(op, "{op} {obj}".replace("{op}", op))
            return template.format(obj=self.object)
        return _OPERATION_PHRASE_NO_OBJECT.get(op, f"{op} (target pose)")


class OperatorInfo(BaseModel):
    """The operating subject (actor) that performs a task's stages."""

    name: str
    """The operator name referenced by stages / declared in ``task_operators``."""
    model: str = ""
    """The robot model the operator is embodied as — the ``mjcf`` scene-layer
    XML stem (e.g. ``robotiq``, ``airbot_play_with_g2p``). Set when the scene
    loads a single robot; left empty when the robot is ambiguous."""


class TaskInfo(BaseModel):
    """Introspected metadata for one runnable task x embodiment variant."""

    task: str
    """The ``task`` config group option (what you pass as ``task=``)."""
    embodiment: str = ""
    """The ``embodiment`` config group option this variant was composed with."""
    default_embodiment: bool = True
    """Whether ``embodiment`` is the task's own default (no override needed)."""
    scene_name: str = ""
    """The scene the task runs in (``scene_name`` in the composed config)."""
    renders: List[str] = []
    """``render`` options with assets for this scene (``render_assets/scene/<render>``);
    native ``mujoco`` rendering is always available for simulated tasks."""
    operators: List[OperatorInfo] = []
    """Operating subjects — the operator(s) that perform the stages, each with
    its robot model when known."""
    robots: List[str] = []
    """Robot models loaded by the scene, from ``mjcf`` layer XML stems."""
    declared_objects: List[str] = []
    """Objects declared in ``env.mask_objects``."""
    stage_objects: List[str] = []
    """Objects actually referenced across the stages (order-preserving, unique)."""
    declared_operations: List[str] = []
    """Operations declared in ``env.operations``."""
    stage_operations: List[str] = []
    """Operations actually used across the stages (order-preserving, unique)."""
    stages: List[StageInfo] = []

    @property
    def objects(self) -> List[str]:
        """Best available object list: declared if present, else stage-derived.

        Deduplicated for display; the raw ``declared_objects`` /
        ``stage_objects`` fields back the mismatch check.
        """
        return _unique_preserving_order(self.declared_objects or self.stage_objects)

    @property
    def operations(self) -> List[str]:
        """Best available operation list: declared if present, else stage-derived.

        Deduplicated for display; the raw ``declared_operations`` /
        ``stage_operations`` fields back the mismatch check.
        """
        return _unique_preserving_order(
            self.declared_operations or self.stage_operations
        )

    @property
    def object_mismatch(self) -> bool:
        """True when declared objects and stage objects disagree (as sets)."""
        if not self.declared_objects:
            return False
        return set(self.declared_objects) != set(self.stage_objects)

    @property
    def operation_mismatch(self) -> bool:
        """True when declared operations and stage operations disagree (as sets)."""
        if not self.declared_operations:
            return False
        return set(self.declared_operations) != set(self.stage_operations)

    @property
    def workflow(self) -> List[str]:
        """Ordered, human-readable description of the task flow."""
        return [stage.describe() for stage in self.stages]

    @property
    def overrides(self) -> List[str]:
        """Hydra overrides selecting this variant, e.g. for ``aao-demo``."""
        selected = [f"task={self.task}"]
        if self.embodiment and not self.default_embodiment:
            selected.append(f"embodiment={self.embodiment}")
        return selected

    @property
    def label(self) -> str:
        """Short display / sort key: the task plus any non-default embodiment."""
        return " ".join(self.overrides).replace("task=", "", 1)


def _unique_preserving_order(values: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _as_str_list(value: object) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v) for v in value]


# Width the transient progress line is padded to, so a shorter line fully
# overwrites a longer previous one before the carriage return.
_PROGRESS_WIDTH = 80


def _write_progress(current: int, total: int, name: str) -> None:
    line = f"Composing configs [{current}/{total}]: {name}"
    if len(line) > _PROGRESS_WIDTH:
        line = line[: _PROGRESS_WIDTH - 1] + "…"
    sys.stderr.write("\r" + line.ljust(_PROGRESS_WIDTH))
    sys.stderr.flush()


def _clear_progress() -> None:
    sys.stderr.write("\r" + " " * _PROGRESS_WIDTH + "\r")
    sys.stderr.flush()


def _option_stems(directory: Path) -> List[str]:
    """Selectable config options in ``directory`` (``_`` fragments skipped)."""
    return sorted(
        p.stem for p in directory.glob("*.yaml") if not p.name.startswith("_")
    )


def discover_task_names(config_dir: Path) -> List[str]:
    """Return every option of the ``task`` config group."""
    return _option_stems(config_dir / "task")


def discover_adapted_embodiments(config_dir: Path, task: str) -> List[str]:
    """Embodiments with a dedicated ``adapt/<task>/<embodiment>.yaml``."""
    return _option_stems(config_dir / "adapt" / task)


def _scene_renders(config_dir: Optional[Path], scene_name: str) -> List[str]:
    if not scene_name:
        return []
    renders = ["mujoco"]
    if config_dir is not None:
        assets = config_dir / "render_assets" / "scene"
        renders += sorted(p.parent.name for p in assets.glob(f"*/{scene_name}.yaml"))
    return renders


def load_task_info(
    task: str,
    embodiment: Optional[str] = None,
    config_dir: Optional[Path] = None,
) -> Optional[TaskInfo]:
    """Compose ``task`` (optionally on ``embodiment``) and return its info.

    Must run inside an initialized Hydra config-dir context. Returns ``None``
    when the composition declares no stages. Raises on composition errors so
    the caller can decide whether to report or swallow them.
    """
    overrides = [f"embodiment={embodiment}"] if embodiment else []
    cfg = compose(
        config_name=PRIMARY_CONFIG_NAME,
        overrides=task_overrides(task, overrides),
        return_hydra_config=True,
    )
    choices = OmegaConf.to_container(cfg.hydra.runtime.choices)
    OmegaConf.set_struct(cfg, False)
    del cfg["hydra"]
    prepared = prepare_task_config_for_instantiation(cfg)
    try:
        data = OmegaConf.to_container(prepared, resolve=True)
    except Exception:  # noqa: BLE001 — e.g. unset env vars; names stay readable
        data = OmegaConf.to_container(prepared, resolve=False)
    if not isinstance(data, dict):
        return None

    task_cfg = data.get("task") or {}
    raw_stages = task_cfg.get("stages") if isinstance(task_cfg, dict) else None
    if not raw_stages:
        return None

    stages: List[StageInfo] = []
    for i, st in enumerate(raw_stages):
        st = st if isinstance(st, dict) else {}
        stages.append(
            StageInfo(
                index=i,
                name=str(st.get("name", "") or ""),
                object=str(st.get("object", "") or ""),
                operation=str(st.get("operation", "") or ""),
                operator=str(st.get("operator", "") or ""),
                site=st.get("site"),
            )
        )

    env = data.get("env") or {}
    env = env if isinstance(env, dict) else {}

    # Robot models are ordinary ordered ``mjcf`` layers.
    scene_data = env.get("scene") or {}
    layers = scene_data.get("layers", []) if isinstance(scene_data, dict) else []
    mjcf_paths = [
        layer.get("path")
        for layer in layers
        if isinstance(layer, dict) and layer.get("kind") == "mjcf"
    ]
    robots = _unique_preserving_order([Path(str(p)).stem for p in mjcf_paths])

    # Operating subjects: names come from the stages (the actors that act) plus
    # any declared in task_operators / env.operators. The robot model is only
    # attached when the scene loads exactly one robot (otherwise ambiguous).
    task_operators = data.get("task_operators")
    env_operators = env.get("operators")
    operator_names = _unique_preserving_order(
        [s.operator for s in stages]
        + (list(task_operators) if isinstance(task_operators, dict) else [])
        + (list(env_operators) if isinstance(env_operators, dict) else [])
    )
    sole_model = robots[0] if len(robots) == 1 else ""
    operators = [OperatorInfo(name=n, model=sole_model) for n in operator_names]

    selected_embodiment = str(choices.get("embodiment") or "")
    return TaskInfo(
        task=task,
        embodiment=selected_embodiment,
        default_embodiment=embodiment is None,
        scene_name=str(data.get("scene_name", "") or ""),
        renders=_scene_renders(config_dir, str(data.get("scene_name", "") or "")),
        operators=operators,
        robots=robots,
        declared_objects=_as_str_list(env.get("mask_objects")),
        stage_objects=_unique_preserving_order([s.object for s in stages]),
        declared_operations=_as_str_list(env.get("operations")),
        stage_operations=_unique_preserving_order([s.operation for s in stages]),
        stages=stages,
    )


def collect_task_infos(
    config_dir: Path,
    name_patterns: Optional[List[str]] = None,
    *,
    verbose: bool = False,
    progress: Optional[bool] = None,
) -> List[TaskInfo]:
    """Compose every matching task variant and keep the ones with stages.

    ``name_patterns`` is a list of ``fnmatch`` globs (e.g. ``open_door*``)
    matched against the ``task`` group options; when omitted, every task is
    considered. An exact name is just a glob that matches itself, so
    composition happens only for tasks that match. Each task is composed on
    its default embodiment and on every embodiment it has an adaptation for.
    Composition runs under a single Hydra context.

    ``progress`` shows a transient ``[i/total]`` line on stderr while composing
    (composition is the slow part). ``None`` (default) auto-enables it only when
    stderr is a TTY, so piped/redirected output stays clean.
    """
    tasks = discover_task_names(config_dir)
    if name_patterns:
        tasks = [
            name
            for name in tasks
            if any(fnmatch(name, pattern) for pattern in name_patterns)
        ]
    candidates: List[tuple[str, Optional[str]]] = [
        (task, embodiment)
        for task in tasks
        for embodiment in [None, *discover_adapted_embodiments(config_dir, task)]
    ]
    infos: List[TaskInfo] = []

    if progress is None:
        progress = sys.stderr.isatty()
    total = len(candidates)

    # A single Hydra init serves every compose() call in the loop.
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        for i, (task, embodiment) in enumerate(candidates, start=1):
            name = task if embodiment is None else f"{task} embodiment={embodiment}"
            if progress:
                _write_progress(i, total, name)
            try:
                info = load_task_info(task, embodiment, config_dir)
            except Exception as exc:  # noqa: BLE001 — report, never abort the sweep
                if verbose:
                    if progress:
                        _clear_progress()
                    print(
                        f"[skip] {name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                continue
            if info is None:
                if verbose:
                    if progress:
                        _clear_progress()
                    print(
                        f"[skip] {name}: no task.stages (not a task)",
                        file=sys.stderr,
                    )
                continue
            if not info.default_embodiment and any(
                other.task == task and other.embodiment == info.embodiment
                for other in infos
            ):
                continue  # the adaptation targets the task's own default
            infos.append(info)

    if progress:
        _clear_progress()

    infos.sort(key=lambda t: (t.task, not t.default_embodiment, t.embodiment))
    return infos


def filter_task_infos(
    infos: List[TaskInfo],
    *,
    operations: Optional[List[str]] = None,
    objects: Optional[List[str]] = None,
    scenes: Optional[List[str]] = None,
    robots: Optional[List[str]] = None,
) -> List[TaskInfo]:
    """Keep task infos matching every provided filter category.

    Categories are AND-combined; values within a category are OR-combined:

    - ``operations``: keep a task that uses at least one listed operation
      (case-insensitive, exact match against the task's operations).
    - ``objects``: keep a task that references an object containing at least
      one listed substring (case-insensitive).
    - ``scenes``: keep a task whose ``scene_name`` matches at least one glob.
    - ``robots``: keep a task whose embodiment name or robot model contains at
      least one listed substring (case-insensitive).
    """

    def keep(info: TaskInfo) -> bool:
        if operations:
            wanted = {op.lower() for op in operations}
            if not (wanted & {op.lower() for op in info.operations}):
                return False
        if objects:
            needles = [o.lower() for o in objects]
            haystack = [o.lower() for o in info.objects]
            if not any(needle in obj for needle in needles for obj in haystack):
                return False
        if scenes:
            if not any(fnmatch(info.scene_name, pattern) for pattern in scenes):
                return False
        if robots:
            needles = [r.lower() for r in robots]
            haystack = [r.lower() for r in [info.embodiment, *info.robots] if r]
            if not any(needle in robot for needle in needles for robot in haystack):
                return False
        return True

    return [info for info in infos if keep(info)]


def render_text(infos: List[TaskInfo]) -> str:
    """Render task infos as a readable, grouped text report."""
    if not infos:
        return "No runnable task configs found."

    lines: List[str] = [f"Runnable task variants ({len(infos)}):", ""]
    for info in infos:
        header = info.label
        if info.scene_name and info.scene_name != info.task:
            header += f"  (scene: {info.scene_name})"
        lines.append(header)
        lines.append(f"  run:        aao-demo {' '.join(info.overrides)}")
        if info.embodiment:
            suffix = " (default)" if info.default_embodiment else ""
            lines.append(f"  embodiment: {info.embodiment}{suffix}")
        if len(info.renders) > 1:
            lines.append(f"  render:     {' | '.join(info.renders)}")
        if info.operators:
            actors = ", ".join(
                f"{op.name} ({op.model})" if op.model else op.name
                for op in info.operators
            )
            lines.append(f"  operators:  {actors}")
        # Show an explicit robots line unless it is the single-robot case
        # already displayed inline on the operators line.
        if info.robots and (len(info.robots) != 1 or not info.operators):
            lines.append(f"  robots:     {', '.join(info.robots)}")
        lines.append(f"  objects:    {', '.join(info.objects) or '(none)'}")
        if info.object_mismatch:
            lines.append(
                f"    note: env.mask_objects {info.declared_objects} "
                f"differs from stage objects {info.stage_objects}"
            )
        lines.append(f"  operations: {', '.join(info.operations) or '(none)'}")
        if info.operation_mismatch:
            lines.append(
                f"    note: env.operations {info.declared_operations} "
                f"differs from stage operations {info.stage_operations}"
            )
        lines.append("  workflow:")
        for i, (stage, phrase) in enumerate(zip(info.stages, info.workflow), start=1):
            label = f" [{stage.name}]" if stage.name else ""
            lines.append(f"    {i}. {phrase}{label}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_json(infos: List[TaskInfo]) -> str:
    """Render task infos as a JSON array (workflow included)."""
    import json

    payload = []
    for info in infos:
        entry = info.model_dump()
        entry["overrides"] = info.overrides
        entry["objects"] = info.objects
        entry["operations"] = info.operations
        entry["workflow"] = info.workflow
        payload.append(entry)
    return json.dumps(payload, indent=2, ensure_ascii=False)


# Vocabulary fields, in display order. Each maps to the sorted set of values
# seen across the tasks — a controlled vocabulary for keyword-driven retrieval.
_VOCAB_LABELS = {
    "tasks": "tasks",
    "embodiments": "embodiments",
    "scenes": "scenes",
    "operators": "operators",
    "robots": "robots",
    "objects": "objects",
    "operations": "operations",
    "stage_names": "stage names",
}


def build_vocabulary(infos: List[TaskInfo]) -> "dict[str, List[str]]":
    """Aggregate every task field into sorted, de-duplicated value sets.

    Instead of a per-task view, this collapses all tasks into one glossary:
    each field maps to the union of its values across ``infos``. Objects and
    operations union both the declared and stage-derived sources so no term is
    missed. Useful as a controlled vocabulary an agent can search against.
    """
    buckets: "dict[str, set[str]]" = {key: set() for key in _VOCAB_LABELS}
    for info in infos:
        buckets["tasks"].add(info.task)
        if info.embodiment:
            buckets["embodiments"].add(info.embodiment)
        if info.scene_name:
            buckets["scenes"].add(info.scene_name)
        buckets["operators"].update(op.name for op in info.operators)
        buckets["robots"].update(info.robots)
        buckets["objects"].update(info.declared_objects)
        buckets["objects"].update(info.stage_objects)
        buckets["operations"].update(info.declared_operations)
        buckets["operations"].update(info.stage_operations)
        buckets["stage_names"].update(s.name for s in info.stages if s.name)
    # Drop unresolved interpolation placeholders (e.g. ``${object_name}`` from
    # template configs) — they are noise in a keyword vocabulary.
    return {
        key: sorted(v for v in values if "${" not in v)
        for key, values in buckets.items()
    }


def render_vocab_text(vocab: "dict[str, List[str]]") -> str:
    """Render the aggregated vocabulary as a readable, wrapped glossary."""
    import textwrap

    n_tasks = len(vocab.get("tasks", []))
    if n_tasks == 0:
        return "No runnable task configs found."

    lines: List[str] = [
        f"Task vocabulary ({n_tasks} task{'s' if n_tasks != 1 else ''}):",
        "",
    ]
    for key, label in _VOCAB_LABELS.items():
        values = vocab.get(key, [])
        lines.append(f"{label} ({len(values)}):")
        if values:
            lines.append(
                textwrap.fill(
                    ", ".join(values),
                    width=78,
                    initial_indent="  ",
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        else:
            lines.append("  (none)")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_vocab_json(vocab: "dict[str, List[str]]") -> str:
    """Render the aggregated vocabulary as a JSON object of sorted value lists."""
    import json

    return json.dumps(vocab, indent=2, ensure_ascii=False)


def _flatten_csv(values: Optional[List[str]]) -> List[str]:
    """Split comma-separated filter values so ``-o pick,place`` works too."""
    out: List[str] = []
    for value in values or []:
        out.extend(part.strip() for part in value.split(",") if part.strip())
    return out


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aao-info",
        description="Introspect and filter runnable tasks in aao_configs/task/.",
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        metavar="PATTERN",
        help=(
            "Glob pattern(s) matched against task names, e.g. 'open_door*' "
            "(an exact name matches itself); default: all runnable tasks."
        ),
    )
    parser.add_argument(
        "-o",
        "--operation",
        action="append",
        default=[],
        metavar="OP",
        help="Keep tasks that use this operation (repeatable / comma-separated).",
    )
    parser.add_argument(
        "-b",
        "--object",
        dest="objects",
        action="append",
        default=[],
        metavar="OBJ",
        help="Keep tasks referencing an object containing this substring "
        "(repeatable / comma-separated).",
    )
    parser.add_argument(
        "-s",
        "--scene",
        action="append",
        default=[],
        metavar="GLOB",
        help="Keep tasks whose scene name matches this glob (repeatable).",
    )
    parser.add_argument(
        "-r",
        "--robot",
        action="append",
        default=[],
        metavar="MODEL",
        help="Keep tasks whose embodiment or robot model contains this substring "
        "(repeatable / comma-separated).",
    )
    parser.add_argument(
        "--vocab",
        "--keywords",
        dest="vocab",
        action="store_true",
        help="Aggregate all fields into a keyword vocabulary (sorted value sets) "
        "instead of a per-task report — a glossary for retrieval.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit JSON instead of readable text."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Config directory (default: ./aao_configs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Report configs skipped as non-tasks or on composition errors.",
    )
    parser.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        default=None,
        help="Disable the progress line (auto-shown only on a TTY otherwise).",
    )
    args = parser.parse_args(argv)

    config_dir = args.config_dir or get_config_dir()
    if not config_dir.is_dir():
        parser.error(f"config directory not found: {config_dir}")

    infos = collect_task_infos(
        config_dir,
        args.patterns or None,
        verbose=args.verbose,
        progress=args.progress,
    )
    infos = filter_task_infos(
        infos,
        operations=_flatten_csv(args.operation) or None,
        objects=_flatten_csv(args.objects) or None,
        scenes=_flatten_csv(args.scene) or None,
        robots=_flatten_csv(args.robot) or None,
    )

    if args.vocab:
        vocab = build_vocabulary(infos)
        if args.json:
            print(render_vocab_json(vocab))
        else:
            print(render_vocab_text(vocab), end="")
    elif args.json:
        print(render_json(infos))
    else:
        print(render_text(infos), end="")


if __name__ == "__main__":
    main()
