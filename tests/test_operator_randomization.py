from pathlib import Path
import sys

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from auto_atom.config.randomization import OperatorRandomizationConfig, PoseRandomRange
from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


def _load_task_file(overrides: list[str] | None = None):
    config_dir = ROOT / "aao_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="pick_and_place",
            overrides=["env.batch_size=1", "env.viewer=null", *(overrides or [])],
        )
    return prepare_task_file(cfg)


def main() -> None:
    task_file = _load_task_file()
    task_file.task.randomization.entities["arm"] = (
        OperatorRandomizationConfig.model_validate(
            {
                "base": {
                    "x": [0.012, 0.012],
                    "y": [-0.007, -0.007],
                },
                "eef": {
                    "z": [0.02, 0.02],
                },
            }
        )
    )
    runner = TaskRunner().from_config(task_file)

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


def test_direct_operator_randomization_rejected() -> None:
    task_file = _load_task_file()
    task_file.task.randomization.entities["arm"] = PoseRandomRange.model_validate(
        {
            "x": [0.01, 0.01],
            "y": [0.0, 0.0],
        }
    )
    runner = TaskRunner().from_config(task_file)

    try:
        with pytest.raises(TypeError, match="nested form"):
            runner.reset()
    finally:
        runner.close()


def test_initial_poses_without_randomization() -> None:
    task_file = _load_task_file()
    task_file.task.randomization.entities = {}
    runner = TaskRunner().from_config(task_file)

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


def test_operator_auto_collision_radius_resolves_from_real_geometry() -> None:
    """Auto base/eef radii derive from real model geometry at reset."""
    task_file = _load_task_file()
    task_file.task.randomization.entities = {
        "arm": OperatorRandomizationConfig.model_validate(
            {
                "base": {
                    "x": [0.012, 0.012],
                    "y": [-0.007, -0.007],
                    "collision_radius": -1,
                },
                "eef": {
                    "x": [0.0, 0.0],
                    "y": [0.0, 0.0],
                    "z": [0.0, 0.0],
                    "collision_radius": -1,
                    "collision_margin": 0.005,
                },
            }
        )
    }
    runner = TaskRunner().from_config(task_file)
    try:
        backend = runner._context.backend
        update = runner.reset()
        details = update.details[0]["initial_poses"]["arm"]
        assert "base_pose" in details
        assert "eef_pose" in details
        cache = backend.randomization_executor._auto_radius_cache
        base_auto = cache[("operator_base", "arm", 0)]
        eef_auto = cache[("operator_eef", "arm", 0)]
        # pick_and_place's ``arm`` is a mocap gripper whose root body
        # (robotiq_interface) carries no geoms → base footprint is 0 (exempt).
        assert base_auto == 0.0
        # The EEF assembly (gripper/fingers under robotiq_base) has real geoms
        # around the EEF site → a positive conservative radius plus margin.
        assert eef_auto > 0.05
    finally:
        runner.close()


if __name__ == "__main__":
    main()
    test_direct_operator_randomization_rejected()
    test_initial_poses_without_randomization()
    test_operator_auto_collision_radius_resolves_from_real_geometry()
