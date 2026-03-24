"""Run a demo using the Mujoco backend.

Config files live in the ``mujoco/`` subdirectory. The default is
``mujoco/pick_and_place.yaml``. Switch tasks with ``--config-name``.
Any value can be overridden from the command line via Hydra, e.g.::

    # Run the default pick-and-place demo
    python run_demo.py

    # Run other tasks
    python run_demo.py --config-name cup_on_coaster
    python run_demo.py --config-name stack_color_blocks
    python run_demo.py --config-name mock

    # Override individual values
    python run_demo.py task.seed=0
    python run_demo.py "task.randomization.cup.x=[-0.03,0.03]"
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from auto_atom.runtime import ComponentRegistry, TaskFileConfig, TaskRunner


@hydra.main(config_path="mujoco", config_name="pick_and_place", version_base=None)
def main(cfg: DictConfig) -> None:
    raw = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(raw, dict):
        raise TypeError("Config root must be a mapping.")

    ComponentRegistry.clear()
    if "env" in cfg and cfg.env is not None:
        instantiate(cfg.env)

    task_file = TaskFileConfig.model_validate(raw)
    runner = TaskRunner().from_config(task_file)

    try:
        print("Reset task")
        print(runner.reset())
        print()

        while True:
            update = runner.update()
            print(update)
            input("Press Enter to continue...")
            if update.done:
                break

        print()
        print("Execution records:")
        for record in runner.records:
            print(record)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
