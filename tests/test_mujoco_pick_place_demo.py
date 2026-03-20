from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.runtime import ComponentRegistry, TaskRunner


def main() -> None:
    config_path = ROOT / "examples" / "mujoco_pick_place_demo.yaml"
    ComponentRegistry.clear()
    runner = TaskRunner().from_yaml(config_path)

    try:
        update = runner.reset()
        assert update.stage_name == "pick_source"

        while True:
            update = runner.update()
            if update.done:
                break

        assert update.success is True

        backend = runner._require_context().backend
        source_pose = backend.get_object_handler("source_block").get_pose()
        target_pose = backend.get_object_handler("target_pedestal").get_pose()

        target_top_z = target_pose.position[2] + 0.04
        assert abs(source_pose.position[0] - target_pose.position[0]) < 0.03
        assert abs(source_pose.position[1] - target_pose.position[1]) < 0.03
        assert abs(source_pose.position[2] - target_top_z) < 0.04
    finally:
        runner.close()


if __name__ == "__main__":
    main()
