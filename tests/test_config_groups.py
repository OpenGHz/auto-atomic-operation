"""Composition checks for the ``aao_configs`` config-group layout."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from auto_atom.config_loader import compose_task_config, describe_run
from auto_atom.execution_config import (
    normalize_env_collections,
    prepare_task_config_for_instantiation,
)
from auto_atom.runner.task_info import (
    discover_adapted_embodiments,
    discover_task_names,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "aao_configs"

TASK_VARIANTS = [
    (task, embodiment)
    for task in discover_task_names(CONFIG_DIR)
    for embodiment in [None, *discover_adapted_embodiments(CONFIG_DIR, task)]
]


def _prepared(task: str, overrides: list[str] | None = None) -> dict:
    cfg = compose_task_config(task, overrides, CONFIG_DIR)
    prepared = prepare_task_config_for_instantiation(cfg)
    data = OmegaConf.to_container(prepared, resolve=True)
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    ("task", "embodiment"),
    TASK_VARIANTS,
    ids=[f"{t}-{e or 'default'}" for t, e in TASK_VARIANTS],
)
def test_every_task_variant_composes_into_flat_env_lists(
    task: str, embodiment: str | None
) -> None:
    data = _prepared(task, [f"embodiment={embodiment}"] if embodiment else [])

    assert data["task"]["stages"], "a runnable task declares stages"
    env = data["env"]
    assert "camera_defaults" not in env and "camera_roles" not in env
    cameras = env.get("cameras", [])
    assert isinstance(cameras, list)
    for camera in cameras:
        assert camera["name"], camera
        assert {"width", "height", "enable_color", "enable_depth"} <= camera.keys()
    layers = (env.get("scene") or {}).get("layers", [])
    assert isinstance(layers, list) and None not in layers


def test_task_discovery_skips_fragments() -> None:
    tasks = discover_task_names(CONFIG_DIR)
    assert "pick_and_place" in tasks
    assert not [name for name in tasks if name.startswith("_")]
    assert "_press_button" not in discover_adapted_embodiments(
        CONFIG_DIR, "press_blue_button"
    )


@pytest.mark.parametrize(
    "task",
    sorted(
        {
            task
            for task, _ in TASK_VARIANTS
            if (
                CONFIG_DIR
                / "render_assets"
                / "scene"
                / "gs"
                / f"{_prepared(task).get('scene_name', '')}.yaml"
            ).exists()
        }
    ),
)
def test_gs_render_merges_scene_assets(task: str) -> None:
    env = _prepared(task, ["render=gs"])["env"]

    assert env["_target_"].endswith("BatchedGSUnifiedMujocoEnv")
    render = env["gaussian_render"]
    assert render["background_ply"]
    assert render["body_gaussians"]


def _differing_keys(native: object, gs: object, path: str = "") -> set[str]:
    if isinstance(native, dict) and isinstance(gs, dict):
        keys: set[str] = set()
        for key in native.keys() | gs.keys():
            if key not in native or key not in gs:
                keys.add(f"{path}{key}")
            else:
                keys |= _differing_keys(native[key], gs[key], f"{path}{key}.")
        return keys
    return set() if native == gs else {path.rstrip(".")}


SHARED_LAYOUT_VARIANTS = [
    ("arrange_flowers", None),
    ("arrange_flowers", "p7_g2p"),
    ("cup_on_coaster", None),
    ("cup_on_coaster", "p7_g2p"),
    ("cup_on_coaster", "p7_v3_umi_v3"),
]


@pytest.mark.parametrize(
    ("task", "embodiment"),
    SHARED_LAYOUT_VARIANTS,
    ids=[f"{t}-{e or 'default'}" for t, e in SHARED_LAYOUT_VARIANTS],
)
def test_shared_layout_scene_differs_only_in_gs_rendering(
    task: str, embodiment: str | None
) -> None:
    # scene/<task>.yaml selects demo_gs.xml for both renderers.
    overrides = [f"embodiment={embodiment}"] if embodiment else []
    native = _prepared(task, overrides)
    gs = _prepared(task, [*overrides, "render=gs"])

    assert native["env"]["scene"]["base"].endswith(f"/{task}/demo_gs.xml")
    assert _differing_keys(native, gs) == {
        "env._target_",
        "env.gaussian_render",
        "gs_dir",
    }


def test_embodiment_swap_and_adaptation() -> None:
    robotiq = _prepared("pick_and_place")
    xf9600 = _prepared("pick_and_place", ["embodiment=xf9600_mocap"])

    assert robotiq["env"]["scene"]["layers"][0]["path"].endswith("/robotiq.xml")
    assert xf9600["env"]["scene"]["layers"][0]["path"].endswith("/xf9600_mocap.xml")
    assert xf9600["env"]["sim_freq"] == 1200
    # adapt/pick_and_place/xf9600_mocap.yaml retunes the grasp height.
    pick_z = [
        stage["param"]["pre_move"][1]["position"][2]
        for stage in (
            robotiq["task"]["stages"][0],
            xf9600["task"]["stages"][0],
        )
    ]
    assert pick_z == [0.006, 0.045]


def test_backend_and_platform_select_without_append_prefix() -> None:
    data = _prepared("pick_and_place", ["backend=warp", "platform=egl"])

    assert data["backend"].endswith("build_mjwarp_backend")
    assert data["env"]["_target_"].endswith("MjWarpObjectOnlyEnv")


def test_observation_and_camera_layout_groups() -> None:
    data = _prepared(
        "cup_on_coaster", ["observation=rgb_only", "camera_layout=no_operator"]
    )
    cameras = data["env"]["cameras"]

    assert [camera["name"] for camera in cameras] == ["env1_cam", "env0_cam"]
    assert {(c["width"], c["height"], c["enable_depth"]) for c in cameras} == {
        (640, 480, False)
    }


def test_mock_task_clears_simulated_groups() -> None:
    data = _prepared("mock")

    assert data["backend"] == "auto_atom.mock.build_mock_backend"
    assert set(data["env"]) == {"_target_", "kind", "name"}
    assert set(data["task_operators"]) == {"arm_a", "observer"}


def test_normalize_env_collections_slots() -> None:
    cfg = OmegaConf.create(
        {
            "res": {"width": 320, "height": 240},
            "env": {
                "camera_defaults": "${res}",
                "camera_roles": ["scene"],
                "cameras": {
                    "wrist": {"name": "wrist_cam", "role": "operator"},
                    "env0": None,
                    "env1": {"name": "env1_cam", "height": 120},
                },
                "scene": {"layers": {"robot": {"kind": "mjcf"}, "door": None}},
            },
        }
    )

    normalize_env_collections(cfg)

    assert OmegaConf.to_container(cfg.env, resolve=True) == {
        "cameras": [{"width": 320, "height": 120, "name": "env1_cam"}],
        "scene": {"layers": [{"kind": "mjcf"}]},
    }


def test_describe_run_names_non_default_dimensions() -> None:
    assert (
        describe_run(
            {"task": "pick_and_place", "embodiment": "xf9600_mocap", "render": "mujoco"}
        )
        == "pick_and_place__xf9600_mocap"
    )
    assert (
        describe_run(
            {"task": "press_blue_button", "embodiment": "p7_g2p", "render": "gs"}
        )
        == "press_blue_button__p7_g2p__gs"
    )
    assert (
        describe_run({"task": "mock", "embodiment": None, "render": "mujoco"}) == "mock"
    )
