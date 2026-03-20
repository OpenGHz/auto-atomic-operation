"""Run the YAML-driven mock task flow example."""

from pathlib import Path
from auto_atom.runtime import ComponentRegistry, TaskRunner


def main() -> None:
    config_path = Path(__file__).with_name("mock_task.yaml")
    ComponentRegistry.clear()
    runner = TaskRunner().from_yaml(config_path)

    print("Reset task")
    print(runner.reset())
    print()

    while True:
        update = runner.update()
        print(update)
        if update.done:
            break

    print()
    print("Execution records:")
    for record in runner.records:
        print(record)

    runner.close()


if __name__ == "__main__":
    main()
