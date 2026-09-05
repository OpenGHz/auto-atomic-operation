"""Launch interactive MuJoCo viewers over the canonical backend interface."""

from __future__ import annotations

import sys

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from pydantic_settings import CliApp

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.backend.mjc.viewer import (
    gaussian_config,
    print_model_summary,
    run_gs_synced_viewer,
    run_native_viewer,
)
from auto_atom.runner.common import get_config_dir, prepare_task_file
from auto_atom.runtime import construct_scene_backend


class ViewSceneCliConfig(BaseModel, frozen=True):
    """Script-owned options parsed before Hydra handles task composition."""

    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    debug: bool = False
    """Print full tracebacks for Gaussian render and reload errors."""

    show_object_frames: bool = False
    """Show MuJoCo body coordinate frames immediately when the viewer opens."""


_CLI_CONFIG = ViewSceneCliConfig()


def _parse_script_cli_config(argv: list[str]) -> ViewSceneCliConfig:
    """Consume script-owned flags before Hydra parses the remaining argv."""

    script_args: list[str] = []
    stripped: list[str] = []
    hydra_separator_seen = False
    owned_flags = {
        "--debug",
        "--no-debug",
        "--show-object-frames",
        "--no-show-object-frames",
    }
    for arg in argv:
        if arg == "--":
            hydra_separator_seen = True
            stripped.append(arg)
        elif not hydra_separator_seen and arg in owned_flags:
            script_args.append(arg)
        else:
            stripped.append(arg)
    argv[:] = stripped
    return CliApp.run(ViewSceneCliConfig, cli_args=script_args)


def _without_embedded_viewer(cfg: DictConfig) -> DictConfig:
    """Copy a Hydra config while leaving viewer ownership to this script."""

    isolated = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.update(isolated, "env.viewer", None, merge=False, force_add=True)
    return isolated


def _load_backend(cfg: DictConfig) -> MujocoTaskBackend:
    """Create one backend and run its canonical setup/reset lifecycle."""

    task_file = prepare_task_file(_without_embedded_viewer(cfg))
    backend = construct_scene_backend(
        task_file,
        feature="view_scene initialization",
    )
    if not isinstance(backend, MujocoTaskBackend):
        backend.teardown()
        raise TypeError(
            f"view_scene requires the MuJoCo backend; got {type(backend).__name__}."
        )
    try:
        backend.setup(task_file.task)
        backend.reset()
    except BaseException:
        backend.teardown()
        raise
    return backend


def _compose_config_from_disk(
    config_dir: str,
    config_name: str,
    cli_overrides: list[str] | None = None,
) -> DictConfig:
    """Re-compose a task config for viewer reload."""

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=cli_overrides or [])


@hydra.main(
    config_path=str(get_config_dir()),
    config_name="pick_and_place",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    hydra_cfg = HydraConfig.get()
    config_name = hydra_cfg.job.config_name
    cli_overrides = list(hydra_cfg.overrides.task)
    config_dir = next(
        (
            source.path
            for source in hydra_cfg.runtime.config_sources
            if source.provider == "main"
        ),
        None,
    )
    if config_dir is None:
        raise RuntimeError("Could not resolve absolute config dir from HydraConfig.")

    backend = _load_backend(cfg)

    def reload_backend() -> MujocoTaskBackend:
        reloaded = _compose_config_from_disk(config_dir, config_name, cli_overrides)
        return _load_backend(reloaded)

    try:
        print_model_summary(backend)
        gs_cfg = gaussian_config(backend)
        if gs_cfg is not None:
            backend = run_gs_synced_viewer(
                backend,
                gs_cfg,
                reload_callback=reload_backend,
                debug=_CLI_CONFIG.debug,
                show_object_frames=_CLI_CONFIG.show_object_frames,
            )
        else:
            backend = run_native_viewer(
                backend,
                reload_callback=reload_backend,
                show_object_frames=_CLI_CONFIG.show_object_frames,
            )
    except KeyboardInterrupt:
        print("[info] interrupted; closing viewer.", flush=True)
    finally:
        backend.teardown()


if __name__ == "__main__":
    _CLI_CONFIG = _parse_script_cli_config(sys.argv)
    main()
