#!/usr/bin/env python3
"""Check whether MJWarp can run the models an ``aao-demo`` config would build.

MJWarp (``mujoco_warp``) is a GPU build of MuJoCo.  Adding it as a backend is
only worth scoping if it accepts our scenes in the first place, and its
``put_model`` feature gate rejects a handful of model traits that no enum table
documents -- notably non-zero ``margin`` on mesh CCD pairs.  This script answers
that question empirically instead of by inspection.

It runs in two stages because MJWarp normally lives in a different environment
than the project:

``export``
    Compile each execution mode's scene through the real Hydra +
    scene-composition path and save it as a ``.mjb`` plus a JSON manifest.
    Must run in the project environment.

``probe``
    Load those models into MJWarp: ``put_model`` (the feature gate), per-world
    ``batch_sizes``, ``put_data``, ``step``, contact/sensor reads, the host
    round-trip, and the batch renderer against the config's own camera specs.
    Must run in an environment with ``mujoco_warp`` installed.

``both`` (the default) runs ``export`` in-process and re-executes this file for
``probe`` under ``--probe-python``.

Examples::

    # One shot, project env for export and a probe venv for the MJWarp half.
    python scripts/check_mjwarp_compat.py --probe-python /tmp/mjw/venv/bin/python

    # Stages separately.
    python scripts/check_mjwarp_compat.py export --config-name rack_plate_p7_v4_umi_v3
    /tmp/mjw/venv/bin/python scripts/check_mjwarp_compat.py probe outputs/mjwarp-compat

A ``.mjb`` is tied to the MuJoCo version that wrote it, so the manifest records
that version and ``probe`` refuses a mismatch rather than failing obscurely.

To create a probe environment::

    python -m venv /tmp/mjw/venv
    /tmp/mjw/venv/bin/pip install mujoco-warp "mujoco==<project version>"

Exit status is 0 when every model passes, 1 when any model is rejected, and 2
for a usage or environment error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_CONFIG = "rack_plate_p7_v4_umi_v3"
DEFAULT_OUT_DIR = Path("outputs/mjwarp-compat")
DEFAULT_MODES = ("physical", "object_only")
DEFAULT_NWORLD = 2

# Model fields the randomizer writes per replica today (one MjModel per env).
# On MJWarp a single shared model can only express that variation if these are
# ``*``-batched, so the probe checks them explicitly.
BATCHED_FIELDS = (
    "body_pos",
    "body_quat",
    "geom_size",
    "geom_rgba",
    "cam_pos",
    "cam_quat",
    "cam_fovy",
    "jnt_range",
    "qpos0",
)

MANIFEST_NAME = "manifest.json"


# ----------------------------------------------------------------------
# Stage 1: export
# ----------------------------------------------------------------------


def export_models(
    config_name: str,
    out_dir: Path,
    modes: Sequence[str],
) -> Path:
    """Compile each mode's scene and write ``.mjb`` files plus a manifest.

    The scene is built through ``prepare_task_config_for_instantiation`` and
    ``load_composed_scene`` so operator layers, config-declared cameras and
    ``execution.mode`` stripping behave exactly as they do in a real run.
    """
    import mujoco
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from auto_atom.config.env_config import EnvConfig
    from auto_atom.execution_config import prepare_task_config_for_instantiation
    from auto_atom.scene_composition import load_composed_scene

    config_dir = Path.cwd() / "aao_configs"
    if not config_dir.is_dir():
        raise SystemExit(
            f"aao_configs not found at {config_dir}; run from the project root."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []

    for mode in modes:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=config_name, overrides=[f"execution.mode={mode}"])
        prepared = prepare_task_config_for_instantiation(cfg)
        env_node = OmegaConf.to_container(prepared.env, resolve=True)
        env_node.pop("_target_", None)
        env_config = EnvConfig.model_validate(env_node)

        model = load_composed_scene(
            env_config.scene,
            cameras=env_config.camera_elements(),
        )
        path = out_dir / f"{config_name}.{mode}.mjb"
        mujoco.mj_saveModel(model, str(path), None)

        # Camera specs travel with the model so the probe can render exactly
        # what the config asks for instead of guessing a resolution.
        cameras = [
            {
                "name": camera.name,
                "width": camera.width,
                "height": camera.height,
                "enable_color": camera.enable_color,
                "enable_depth": camera.enable_depth,
                "enable_mask": camera.enable_mask,
                "enable_heat_map": camera.enable_heat_map,
            }
            for camera in env_config.cameras
        ]
        entry = {
            "mode": mode,
            "model": path.name,
            "batch_size": int(env_config.batch_size),
            "cameras": cameras,
            "stats": {
                "nbody": int(model.nbody),
                "ngeom": int(model.ngeom),
                "nu": int(model.nu),
                "nq": int(model.nq),
                "nsensor": int(model.nsensor),
                "ncam": int(model.ncam),
            },
        }
        entries.append(entry)
        stats = entry["stats"]
        print(
            f"  {mode:12s} -> {path.name}  "
            + " ".join(f"{key}={value}" for key, value in stats.items())
        )

    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "config_name": config_name,
                "mujoco_version": mujoco.__version__,
                "models": entries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"  manifest -> {manifest_path}")
    return manifest_path


# ----------------------------------------------------------------------
# Stage 2: probe
# ----------------------------------------------------------------------


@dataclass
class ProbeResult:
    """One model's outcome, in the order the port would hit each step."""

    mode: str
    model: str
    steps: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.blockers

    def record(self, name: str, outcome: str) -> None:
        self.steps[name] = outcome

    def fail(self, name: str, exc: BaseException) -> None:
        detail = f"{type(exc).__name__}: {exc}"
        self.steps[name] = detail
        self.blockers.append(f"{name}: {detail}")


def report_feature_gate(mjm: Any, result: ProbeResult) -> None:
    """Print and record the model traits MJWarp's ``put_model`` rejects."""
    import mujoco
    import numpy as np

    print(
        f"    integrator={mujoco.mjtIntegrator(mjm.opt.integrator).name}"
        f" solver={mujoco.mjtSolver(mjm.opt.solver).name}"
        f" cone={mujoco.mjtCone(mjm.opt.cone).name}"
        f" noslip_iterations={mjm.opt.noslip_iterations}"
    )
    sensors = sorted({mujoco.mjtSensor(int(t)).name for t in mjm.sensor_type})
    equalities = sorted({mujoco.mjtEq(int(t)).name for t in mjm.eq_type})
    print(f"    sensors={sensors or ['<none>']} equalities={equalities or ['<none>']}")

    for label, array in (
        ("body", mjm.body_plugin),
        ("actuator", mjm.actuator_plugin),
        ("sensor", mjm.sensor_plugin),
    ):
        count = int((np.asarray(array) != -1).sum())
        if count:
            note = f"{count} {label} plugin(s) — MJWarp does not support plugins"
            result.blockers.append(note)
            print(f"    !! {note}")

    # Non-zero margin on mesh/box CCD pairs is the check that actually bites,
    # and it is invisible in the supported-enum tables.
    margins = np.asarray(mjm.geom_margin)
    nonzero = int((margins != 0.0).sum())
    if nonzero:
        multiccd_on = not (
            mjm.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        )
        note = (
            f"{nonzero}/{mjm.ngeom} geoms carry non-zero margin "
            f"(values {sorted(set(margins[margins != 0].round(6).tolist()))}); "
            f"MULTICCD is {'enabled' if multiccd_on else 'disabled'}"
        )
        result.notes.append(note)
        print(f"    {note}")

    # friction 0 with condim>=4 is a documented NaN risk under contact.
    risky = int(
        (
            (np.asarray(mjm.geom_friction)[:, 0] < 1e-5)
            & (np.asarray(mjm.geom_condim) >= 4)
        ).sum()
    )
    if risky:
        note = f"{risky} geoms have friction<MJ_MINMU with condim>=4 (NaN risk)"
        result.notes.append(note)
        print(f"    {note}")


def probe_physics(
    mjm: Any,
    nworld: int,
    njmax: int | None,
    result: ProbeResult,
) -> Any:
    """Run put_model / put_data / step and read back contacts and sensors.

    Returns ``(model, data)`` when stepping succeeded, else ``None`` -- the
    renderer probe needs a stepped Data to render a settled scene.
    """
    import mujoco
    import mujoco_warp as mjw
    import numpy as np

    try:
        model = mjw.put_model(mjm)
        result.record("put_model", "ok")
        print("    put_model: ok")
    except Exception as exc:
        result.fail("put_model", exc)
        print(f"    put_model: FAILED — {type(exc).__name__}: {exc}")
        return None

    batchable: list[str] = []
    for name in BATCHED_FIELDS:
        try:
            mjw.put_model(mjm, batch_sizes={name: nworld})
            batchable.append(name)
        except Exception as exc:
            result.notes.append(f"{name} not batchable: {type(exc).__name__}: {exc}")
    result.record("batchable", f"{len(batchable)}/{len(BATCHED_FIELDS)}")
    print(f"    batchable: {len(batchable)}/{len(BATCHED_FIELDS)} {batchable}")

    if batchable:
        try:
            model = mjw.put_model(mjm, batch_sizes={f: nworld for f in batchable})
            result.record("put_model_batched", "ok")
            print(f"    put_model(batched, nworld={nworld}): ok")
        except Exception as exc:
            result.fail("put_model_batched", exc)
            print(f"    put_model(batched): FAILED — {type(exc).__name__}: {exc}")
            return None

    try:
        host_data = mujoco.MjData(mjm)
        mujoco.mj_forward(mjm, host_data)
        kwargs: dict[str, Any] = {"nworld": nworld}
        if njmax is not None:
            kwargs["njmax"] = njmax
        data = mjw.put_data(mjm, host_data, **kwargs)
        result.record("put_data", "ok")
        print(f"    put_data: ok naconmax={data.naconmax} njmax={data.njmax}")
    except Exception as exc:
        result.fail("put_data", exc)
        print(f"    put_data: FAILED — {type(exc).__name__}: {exc}")
        return None

    try:
        mjw.step(model, data)
        # MJWarp has no data.ncon: contacts live in one flat naconmax pool that
        # every world shares, tagged by contact.worldid.
        nacon = int(data.nacon.numpy()[0])
        worldid = data.contact.worldid.numpy()[:nacon]
        per_world = [int((worldid == w).sum()) for w in range(nworld)]
        result.record("step", "ok")
        result.record("contacts", f"nacon={nacon} per_world={per_world}")
        print(f"    step: ok nacon={nacon} per_world={per_world}")
    except Exception as exc:
        result.fail("step", exc)
        print(f"    step: FAILED — {type(exc).__name__}: {exc}")
        return None

    if mjm.nsensor:
        sensordata = data.sensordata.numpy()
        finite = bool(np.isfinite(sensordata).all())
        result.record("sensordata", f"shape={sensordata.shape} finite={finite}")
        print(f"    sensordata: shape={sensordata.shape} finite={finite}")
        if not finite:
            result.blockers.append("sensordata contains non-finite values after step")

    if mjm.nu:
        try:
            data.ctrl.assign(np.zeros((nworld, mjm.nu), dtype=np.float32))
            mjw.step(model, data)
            result.record("ctrl_write", "ok")
            print(f"    ctrl write + step: ok (nu={mjm.nu})")
        except Exception as exc:
            result.fail("ctrl_write", exc)
            print(f"    ctrl write: FAILED — {type(exc).__name__}: {exc}")

    try:
        out = mujoco.MjData(mjm)
        mjw.get_data_into(out, mjm, data, 0)
        result.record("get_data_into", f"ok ncon={out.ncon}")
        print(f"    get_data_into(world 0): ok ncon={out.ncon}")
    except Exception as exc:
        result.fail("get_data_into", exc)
        print(f"    get_data_into: FAILED — {type(exc).__name__}: {exc}")

    return (model, data)


def probe_renderer(
    mjm: Any,
    entry: dict[str, Any],
    stepped: Any,
    nworld: int,
    result: ProbeResult,
) -> None:
    """Render every model camera at the resolution the task config asks for."""
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    if not mjm.ncam:
        result.record("render", "skipped (no cameras)")
        print("    render: skipped (model has no cameras)")
        return

    model, data = stepped
    names = [
        mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(mjm.ncam)
    ]
    specs = {camera["name"]: camera for camera in entry.get("cameras", ())}
    # Fall back to the model's own resolution for a camera the config does not
    # declare (the scene may author its own).
    resolutions = [
        (
            int(specs[name]["width"]),
            int(specs[name]["height"]),
        )
        if name in specs
        else (int(mjm.cam_resolution[i][0]) or 64, int(mjm.cam_resolution[i][1]) or 64)
        for i, name in enumerate(names)
    ]

    try:
        context = mjw.create_render_context(
            mjm,
            nworld=nworld,
            cam_res=resolutions,
            render_rgb=True,
            render_depth=True,
            render_seg=True,
        )
        mjw.render(model, data, context)
        result.record("render", "ok")
        print(f"    render: ok ({mjm.ncam} cameras)")
    except Exception as exc:
        result.fail("render", exc)
        print(f"    render: FAILED — {type(exc).__name__}: {exc}")
        return

    for index, name in enumerate(names):
        width, height = resolutions[index]
        try:
            rgb = wp.zeros((nworld, height, width), dtype=wp.vec3)
            depth = wp.zeros((nworld, height, width), dtype=wp.float32)
            segmentation = wp.zeros((nworld, height, width), dtype=wp.vec2i)
            mjw.get_rgb(context, index, rgb)
            mjw.get_depth(context, index, 1.0, depth)
            mjw.get_segmentation(context, index, segmentation)
            depth_np = depth.numpy()
            seg_np = segmentation.numpy()
            coverage = float((seg_np[..., 0] >= 0).mean())
            ids = int(np.unique(seg_np[..., 0]).size)
            print(
                f"      {name!r:24s} {width}x{height} "
                f"rgb_mean={rgb.numpy().mean():.4f} "
                f"depth=[{depth_np.min():.3f},{depth_np.max():.3f}] "
                f"seg_ids={ids} coverage={coverage:.1%}"
            )
            if coverage == 0.0:
                result.notes.append(f"camera {name!r} rendered an empty scene")
        except Exception as exc:
            result.fail(f"render[{name}]", exc)
            print(f"      {name!r}: FAILED — {type(exc).__name__}: {exc}")


def probe_model(
    model_path: Path,
    entry: dict[str, Any],
    nworld: int,
    njmax: int | None,
    disable_multiccd: bool,
) -> ProbeResult:
    """Probe one exported model end to end."""
    import mujoco

    result = ProbeResult(mode=entry["mode"], model=model_path.name)
    mjm = mujoco.MjModel.from_binary_path(str(model_path))
    stats = " ".join(f"{key}={value}" for key, value in entry["stats"].items())
    print(f"  host model: {stats}")

    if disable_multiccd:
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
        result.notes.append("probed with MULTICCD disabled (--disable-multiccd)")
        print("    note: MULTICCD disabled for this probe")

    report_feature_gate(mjm, result)
    stepped = probe_physics(mjm, nworld, njmax, result)
    if stepped is not None:
        probe_renderer(mjm, entry, stepped, nworld, result)
    return result


def probe_models(
    out_dir: Path,
    nworld: int,
    njmax: int | None,
    disable_multiccd: bool,
) -> list[ProbeResult]:
    """Load the manifest and probe every model it lists."""
    import mujoco

    manifest_path = out_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(
            f"{manifest_path} not found; run the 'export' stage in the project "
            "environment first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    written_by = manifest.get("mujoco_version")
    if written_by and written_by != mujoco.__version__:
        raise SystemExit(
            f"model files were written by mujoco {written_by} but this "
            f"environment has {mujoco.__version__}; a .mjb is version-locked. "
            f"Install mujoco=={written_by} here, or re-export."
        )

    import mujoco_warp as mjw
    import warp as wp

    print(
        f"mujoco={mujoco.__version__} warp={wp.__version__} "
        f"mujoco_warp={mjw.__version__} device={wp.get_device()}"
    )
    print()

    results: list[ProbeResult] = []
    for entry in manifest["models"]:
        print(f"=== {entry['mode']} ({entry['model']}) ===")
        try:
            results.append(
                probe_model(
                    out_dir / entry["model"],
                    entry,
                    nworld,
                    njmax,
                    disable_multiccd,
                )
            )
        except Exception as exc:
            traceback.print_exc()
            result = ProbeResult(mode=entry["mode"], model=entry["model"])
            result.fail("probe", exc)
            results.append(result)
        print()
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def print_summary(results: Sequence[ProbeResult]) -> None:
    print("=== summary ===")
    for result in results:
        verdict = "PASS" if result.passed else "BLOCKED"
        print(f"  {result.mode:12s} {verdict}")
        for blocker in result.blockers:
            print(f"      blocker: {blocker}")
        for note in result.notes:
            print(f"      note:    {note}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="both",
        choices=("export", "probe", "both"),
        help="Which stage to run (default: both).",
    )
    parser.add_argument(
        "out_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for exported models (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--config-name",
        default=DEFAULT_CONFIG,
        help=f"aao_configs task file to export (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        help="execution.mode to export; repeatable "
        f"(default: {' '.join(DEFAULT_MODES)}).",
    )
    parser.add_argument(
        "--nworld",
        type=int,
        default=DEFAULT_NWORLD,
        help=f"Worlds to probe (default: {DEFAULT_NWORLD}).",
    )
    parser.add_argument(
        "--njmax",
        type=int,
        default=None,
        help="Constraint-row budget passed to put_data. Raise this when the "
        "probe reports 'nefc overflow'.",
    )
    parser.add_argument(
        "--disable-multiccd",
        action="store_true",
        help="Set mjDSBL_MULTICCD before put_model. Clears the non-zero-margin "
        "mesh-CCD rejection, at the cost of one contact per CCD pair.",
    )
    parser.add_argument(
        "--probe-python",
        type=Path,
        default=None,
        help="Interpreter with mujoco_warp installed, used for the probe stage "
        "of 'both'. Defaults to the current interpreter.",
    )
    return parser


def run_probe_subprocess(
    python: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> int:
    """Re-execute this file's probe stage under a different interpreter."""
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "probe",
        str(out_dir),
        "--nworld",
        str(args.nworld),
    ]
    if args.njmax is not None:
        command += ["--njmax", str(args.njmax)]
    if args.disable_multiccd:
        command.append("--disable-multiccd")
    print(f"probing via {python}")
    print()
    return subprocess.call(command)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir
    modes = tuple(args.modes) if args.modes else DEFAULT_MODES

    if args.stage in ("export", "both"):
        print(f"=== export ({args.config_name}) ===")
        export_models(args.config_name, out_dir, modes)
        print()

    if args.stage == "export":
        return 0

    if args.stage == "both" and args.probe_python is not None:
        if not args.probe_python.is_file():
            raise SystemExit(f"--probe-python not found: {args.probe_python}")
        return run_probe_subprocess(args.probe_python, out_dir, args)

    try:
        import mujoco_warp  # noqa: F401
    except ImportError:
        raise SystemExit(
            "mujoco_warp is not installed in this interpreter. Install it here, "
            "or pass --probe-python pointing at an interpreter that has it."
        ) from None

    results = probe_models(out_dir, args.nworld, args.njmax, args.disable_multiccd)
    print_summary(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as exc:
        if isinstance(exc.code, str):
            print(f"error: {exc.code}", file=sys.stderr)
            sys.exit(2)
        raise
