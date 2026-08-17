from pathlib import Path
import sys

import numpy as np
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.runner.common import prepare_task_file
from auto_atom.runtime import ComponentRegistry, TaskRunner


def test_franka_task_uses_stable_closed_loop_setup() -> None:
    """Franka keeps its validated home pose and refines IK error each step."""
    config_dir = ROOT / "aao_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="pick_and_place_franka")

    assert cfg.task_operators.arm.ik.joint_control_mode == "per_step_ik"
    eef_randomization = cfg.task.randomization.arm.eef
    assert list(eef_randomization.x) == [0.0, 0.0]
    assert list(eef_randomization.y) == [0.0, 0.0]
    assert list(eef_randomization.z) == [0.0, 0.0]


def main() -> None:
    config_dir = ROOT / "aao_configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="pick_and_place",
            overrides=["env.batch_size=2", "env.viewer=null"],
        )
    task_file = prepare_task_file(cfg)
    runner = TaskRunner().from_config(task_file)

    try:
        update = runner.reset()
        assert update.stage_name == ["pick_source", "pick_source"]

        while not bool(np.all(update.done)):
            update = runner.update()

        assert bool(np.all(update.success))
        assert len(runner.records) == 4
        assert all(record.status.value == "succeeded" for record in runner.records)
        assert {record.env_index for record in runner.records} == {0, 1}
        assert {record.stage_name for record in runner.records} == {
            "pick_source",
            "place_source",
        }
    finally:
        runner.close()


if __name__ == "__main__":
    main()
