"""Run a simple pick-and-place demo using the Mujoco backend."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import time
import mujoco.viewer
from auto_atom.runtime import ComponentRegistry, TaskRunner
from auto_atom.sim.backend.mujoco_backend import build_mujoco_backend


def build_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.register_backend("mujoco", build_mujoco_backend)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run without the Mujoco viewer.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.03,
        help="Sleep time in seconds after each task update when the viewer is enabled.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="How long to keep the viewer open after the task finishes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(__file__).with_name("mujoco_pick_place_demo.yaml")
    runner = TaskRunner(registry=build_registry()).from_yaml(config_path)
    backend = runner._require_context().backend  # demo-only introspection

    try:
        viewer_context = (
            nullcontext()
            if args.no_viewer
            else mujoco.viewer.launch_passive(backend.env.model, backend.env.data)
        )
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = [1.04, -5.6, 0.88]
                viewer.cam.distance = 0.55
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -28
                viewer.sync()
    
            print("Reset task")
            print(runner.reset())
            print()
            if viewer is not None:
                viewer.sync()
                time.sleep(args.step_delay)

            while True:
                update = runner.update()
                print(update)
                if viewer is not None:
                    viewer.sync()
                    time.sleep(args.step_delay)
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

            if viewer is not None and args.hold_seconds > 0.0:
                deadline = time.time() + args.hold_seconds
                while time.time() < deadline and viewer.is_running():
                    viewer.sync()
                    time.sleep(min(args.step_delay, 0.05))
    finally:
        runner.close()


if __name__ == "__main__":
    main()
