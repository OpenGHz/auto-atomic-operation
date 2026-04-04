"""Remote policy evaluation client — mirrors policy_eval_example.py.

Replays a recorded demo through a remote PolicyEvaluator server.

Usage::

    # Terminal 1: start the server
    python examples/policy_eval_server.py

    # Terminal 2: run this client
    python examples/policy_eval_client.py
    python examples/policy_eval_client.py --host 10.0.0.5 --port 9999
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from auto_atom import TaskUpdate
from auto_atom.ipc import RemotePolicyEvaluator

CONFIG_NAME = "press_three_buttons"
DEMO_PATH = Path("assets/demos") / f"{CONFIG_NAME}.npz"


def load_demo(path: Path) -> dict:
    data = np.load(path)
    keys = [str(k) for k in data["low_dim_keys"]]
    arrays: dict[str, np.ndarray] = {}
    for idx, key in enumerate(keys):
        arrays[key] = np.asarray(data[f"low_dim_data__{idx}"], dtype=np.float32)
    return {
        "position": arrays["action/arm/pose/position"],
        "orientation": arrays["action/arm/pose/orientation"],
        "gripper": arrays["action/eef/joint_state/position"],
    }


class RecordedDemoPolicy:
    def __init__(self, demo: dict) -> None:
        self.positions = demo["position"]
        self.orientations = demo["orientation"]
        self.grippers = demo["gripper"]
        self._max = len(self.positions) - 1
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: Any, update: TaskUpdate) -> dict:
        i = min(self._step, self._max)
        self._step += 1
        return {
            "position": self.positions[i],
            "orientation": self.orientations[i],
            "gripper": self.grippers[i],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote policy evaluation client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=18861)
    args = parser.parse_args()

    if not DEMO_PATH.exists():
        raise FileNotFoundError(
            f"Demo not found: {DEMO_PATH}\n"
            f"Record first: python examples/record_demo.py "
            f"--config-name {CONFIG_NAME} env.batch_size=1"
        )

    demo = load_demo(DEMO_PATH)
    print(f"Loaded {len(demo['position'])} steps from {DEMO_PATH}")

    evaluator = RemotePolicyEvaluator(host=args.host, port=args.port)
    evaluator.from_config(CONFIG_NAME, overrides=["env.batch_size=1"])

    max_updates = len(demo["position"]) + 50
    policy = RecordedDemoPolicy(demo)

    try:
        info = evaluator.get_info()
        print(
            f"Env: {info.get('env_name', '?')}, cameras: {list(info.get('cameras', {}).keys())}"
        )

        policy.reset()
        update = evaluator.reset()
        plans = evaluator.stage_plans
        print(f"Stages: {[p['stage_name'] for p in plans] if plans else '(unknown)'}")

        step = -1
        for step in range(max_updates):
            obs = evaluator.get_observation()
            action = policy.act(obs, update)
            update = evaluator.update(action)
            if update.done.all():
                break

        summary = evaluator.summarize(max_updates=max_updates, updates_used=step + 1)
        print(
            f"\nCompleted {summary.completed_stage_count} stages "
            f"in {summary.updates_used} steps"
        )
        for r in evaluator.records:
            print(f"  {r.stage_name}: {r.status.value}")
        print(f"Success: {list(summary.final_success)}")
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
