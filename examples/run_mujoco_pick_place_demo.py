"""Run a simple pick-and-place demo using the Mujoco backend.

Pose randomization is configured in mujoco_pick_place_demo.yaml under
``task.randomization`` and requires no code changes to adjust.
"""

from pathlib import Path
from auto_atom.runtime import ComponentRegistry, TaskRunner


def main() -> None:
    config_path = Path(__file__).with_name("mujoco_pick_place_demo.yaml")
    ComponentRegistry.clear()
    runner = TaskRunner().from_yaml(config_path)

    try:
        print("Reset task (randomization loaded from YAML)")
        print(runner.reset())
        print()

        backend = runner._require_context().backend
        source_pose = backend.get_object_handler("source_block").get_pose()
        print(f"source_block pose after reset: {source_pose}")
        print()

        while True:
            update = runner.update()
            print(update)
            if update.done:
                break

        source_pose = backend.get_object_handler("source_block").get_pose()
        target_pose = backend.get_object_handler("target_pedestal").get_pose()

        print()
        print("Final poses:")
        print("source_block:", source_pose)
        print("target_pedestal:", target_pose)
        print()
        print("Execution records:")
        for record in runner.records:
            print(record)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
