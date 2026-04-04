"""PolicyEvaluator example: replay a recorded demo as a policy.

Demonstrates the full record -> evaluate pipeline:

1. Record a demo::

    python examples/record_demo.py --config-name press_three_buttons env.batch_size=1

2. Evaluate by replaying the recorded actions::

    python examples/policy_eval_example.py

The policy feeds back the recorded EEF poses + gripper values through
``env.apply_pose_action()``.  See ``docs/action_space.md`` for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from auto_atom import (
    ExecutionContext,
    PolicyEvaluator,
    TaskUpdate,
    load_task_file_hydra,
)

CONFIG_NAME = "press_three_buttons"
DEMO_PATH = Path("assets/demos") / f"{CONFIG_NAME}.npz"


# ---------------------------------------------------------------------------
# Load demo
# ---------------------------------------------------------------------------


def load_demo(path: Path) -> dict:
    """Load pose trace + gripper from a recorded NPZ."""
    data = np.load(path)
    keys = [str(k) for k in data["low_dim_keys"]]

    arrays: dict[str, np.ndarray] = {}
    for idx, key in enumerate(keys):
        arrays[key] = np.asarray(data[f"low_dim_data__{idx}"], dtype=np.float32)

    return {
        "position": arrays["action/arm/pose/position"],  # (T, 3)
        "orientation": arrays["action/arm/pose/orientation"],  # (T, 4)
        "gripper": arrays["action/eef/joint_state/position"],  # (T, 1)
    }


# ---------------------------------------------------------------------------
# Action applier / observation getter
# ---------------------------------------------------------------------------


def action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    """One-liner: apply pose + gripper to the env."""
    if action is not None:
        context.backend.env.apply_pose_action(
            "arm",
            action["position"],
            action["orientation"],
            action["gripper"],
        )


def observation_getter(context: ExecutionContext) -> dict:
    return context.backend.env.capture_observation()


# ---------------------------------------------------------------------------
# Recorded demo policy
# ---------------------------------------------------------------------------


class RecordedDemoPolicy:
    """Replays recorded EEF poses + gripper step by step."""

    def __init__(self, demo: dict) -> None:
        self.positions = demo["position"]
        self.orientations = demo["orientation"]
        self.grippers = demo["gripper"]
        self._max = len(self.positions) - 1
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(
        self, observation: Any, update: TaskUpdate, evaluator: PolicyEvaluator
    ) -> dict:
        i = min(self._step, self._max)
        self._step += 1
        return {
            "position": self.positions[i],
            "orientation": self.orientations[i],
            "gripper": self.grippers[i],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not DEMO_PATH.exists():
        raise FileNotFoundError(
            f"Demo not found: {DEMO_PATH}\n"
            f"Record first: python examples/record_demo.py "
            f"--config-name {CONFIG_NAME} env.batch_size=1"
        )

    task_file = load_task_file_hydra(CONFIG_NAME, overrides=["env.batch_size=1"])
    demo = load_demo(DEMO_PATH)
    print(f"Loaded {len(demo['position'])} steps from {DEMO_PATH}")

    policy = RecordedDemoPolicy(demo)
    evaluator = PolicyEvaluator(
        action_applier=action_applier,
        observation_getter=observation_getter,
    ).from_config(task_file)

    max_updates = len(demo["position"]) + 50
    try:
        policy.reset()
        update = evaluator.reset()
        print(f"Stages: {[s.name for s in task_file.task.stages]}")

        step = -1
        for step in range(max_updates):
            obs = evaluator.get_observation()
            action = policy.act(obs, update=update, evaluator=evaluator)
            update = evaluator.update(action)
            if update.done.all():
                break

        summary = evaluator.summarize(
            update, max_updates=max_updates, updates_used=step + 1
        )
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
