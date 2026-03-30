from pathlib import Path
import sys

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


def main() -> None:
    ComponentRegistry.clear()
    config_dir = ROOT / "examples" / "mujoco"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="pick_and_place",
            overrides=[
                "env.env.config.batch_size=1",
                "env.env.config.viewer=null",
            ],
        )
    instantiate(cfg.env)
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict)
    raw["task"]["randomization"]["arm"] = {
        "base": {
            "x": [0.012, 0.012],
            "y": [-0.007, -0.007],
        },
        "eef": {
            "z": [0.02, 0.02],
        },
    }
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(raw))

    try:
        backend = runner._context.backend
        operator = backend.get_operator_handler("arm")
        default_eef = backend._default_operator_eef_poses["arm"].select(0)

        update = runner.reset()
        details = update.details[0]["initial_poses"]["arm"]
        assert "base_pose" in details
        assert "eef_pose" in details

        base_pose = operator.get_base_pose().select(0)
        eef_pose = operator.get_end_effector_pose().select(0)

        assert abs(float(base_pose.position[0, 0]) - 0.012) < 1e-6
        assert abs(float(base_pose.position[0, 1]) + 0.007) < 1e-6
        assert abs(float(base_pose.position[0, 2])) < 1e-6
        assert (
            abs(
                float(eef_pose.position[0, 2])
                - float(default_eef.position[0, 2])
                - 0.02
            )
            < 5e-3
        )
    finally:
        runner.close()


def test_direct_operator_randomization_details() -> None:
    ComponentRegistry.clear()
    config_dir = ROOT / "examples" / "mujoco"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="pick_and_place",
            overrides=[
                "env.env.config.batch_size=1",
                "env.env.config.viewer=null",
            ],
        )
    instantiate(cfg.env)
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict)
    raw["task"]["randomization"]["arm"] = {
        "x": [0.01, 0.01],
        "y": [0.0, 0.0],
    }
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(raw))

    try:
        update = runner.reset()
        details = update.details[0]["initial_poses"]["arm"]
        assert "base_pose" in details
        assert "eef_pose" in details
    finally:
        runner.close()


def test_initial_poses_without_randomization() -> None:
    ComponentRegistry.clear()
    config_dir = ROOT / "examples" / "mujoco"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="pick_and_place",
            overrides=[
                "env.env.config.batch_size=1",
                "env.env.config.viewer=null",
            ],
        )
    instantiate(cfg.env)
    raw = OmegaConf.to_container(cfg, resolve=False)
    assert isinstance(raw, dict)
    raw["task"]["randomization"] = {}
    runner = TaskRunner().from_config(TaskFileConfig.model_validate(raw))

    try:
        update = runner.reset()
        initial_poses = update.details[0]["initial_poses"]
        assert "arm" in initial_poses
        assert "source_block" in initial_poses
        assert "target_pedestal" in initial_poses
        assert "base_pose" in initial_poses["arm"]
        assert "eef_pose" in initial_poses["arm"]
    finally:
        runner.close()


if __name__ == "__main__":
    main()
    test_direct_operator_randomization_details()
    test_initial_poses_without_randomization()
