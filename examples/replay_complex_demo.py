"""PolicyEvaluator example: replay a recorded demo as a policy.

Demonstrates the full record -> evaluate pipeline:

1. Record a demo (any batch_size)::

    python examples/record_demo.py --config-name press_three_buttons_gs

2. Evaluate by replaying the recorded actions::

    python examples/replay_complex_demo.py

3. Evaluate with recorded initial object poses restored::

    python examples/replay_complex_demo.py --restore-initial-poses

The policy feeds back the recorded EEF poses + gripper values through
``env.apply_pose_action()``.  See ``docs/action_space.md`` for details.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from auto_atom import (
    ExecutionContext,
    PolicyEvaluator,
    TaskUpdate,
    load_task_file_hydra,
)
from auto_atom.framework import InitialPoseConfig

CONFIG_NAME = "press_three_buttons_gs"
DEMO_PATH = Path("outputs/records/demos") / f"{CONFIG_NAME}.npz"


# ---------------------------------------------------------------------------
# Load demo
# ---------------------------------------------------------------------------


def load_demo(path: Path) -> dict:
    """Load pose trace + gripper (and optional base trace) from a recorded NPZ.

    Arrays are stored as ``(T, B, dim)`` where *B* is the recording
    batch size.  Indexing by step yields ``(B, dim)`` — ready for
    ``apply_pose_action``.
    """
    data = np.load(path)
    keys = [str(k) for k in data["low_dim_keys"]]

    arrays: dict[str, np.ndarray] = {}
    for idx, key in enumerate(keys):
        arrays[key] = np.asarray(data[f"low_dim_data__{idx}"], dtype=np.float32)

    pos = arrays["action/arm/pose/position"]  # (T, B, 3)
    ori = arrays["action/arm/pose/orientation"]  # (T, B, 4)
    grip = arrays["action/gripper/joint_state/position"]  # (T, B, 1)

    batch_size = pos.shape[1]
    demo: dict = {
        "position": pos,  # (T, B, 3)
        "orientation": ori,  # (T, B, 4)
        "gripper": grip,  # (T, B, 1)
        "batch_size": batch_size,
    }

    # Optional base trajectory keys.
    base_pos_key = "action/arm/base/position"
    base_ori_key = "action/arm/base/orientation"
    if base_pos_key in arrays and base_ori_key in arrays:
        demo["base_position"] = arrays[base_pos_key]  # (T, B, 3)
        demo["base_orientation"] = arrays[base_ori_key]  # (T, B, 4)

    return demo


def load_initial_poses(json_path: Path) -> List[Dict[str, InitialPoseConfig]]:
    """Load per-env object initial poses from a recorded demo JSON.

    The JSON's ``reset_update.details[env].initial_poses`` stores the
    poses that were active right after the recording session's reset.
    Returns a list (one dict per env) mapping object names to
    ``InitialPoseConfig`` instances ready for
    ``backend.initial_poses.update()``.
    """
    meta = json.loads(json_path.read_text())
    per_env: List[Dict[str, InitialPoseConfig]] = []
    for detail in meta["reset_update"]["details"]:
        env_poses: Dict[str, InitialPoseConfig] = {}
        for name, pose_data in detail.get("initial_poses", {}).items():
            if "base_pose" in pose_data:
                continue  # skip operator entries
            env_poses[name] = InitialPoseConfig(
                position=pose_data["position"],
                orientation=pose_data["orientation"],
            )
        per_env.append(env_poses)
    return per_env


def apply_recorded_initial_poses(
    evaluator: PolicyEvaluator,
    per_env: List[Dict[str, InitialPoseConfig]],
) -> None:
    """Write per-env initial poses into the backend before reset.

    All envs share the same ``backend.initial_poses`` dict, so when
    ``batch_size == 1`` the single env's poses are written directly.
    For ``batch_size > 1`` only ``env 0``'s poses are used (the
    recorded batch may differ from the evaluation batch).
    """
    if not per_env:
        return
    evaluator._context.backend.initial_poses.update(per_env[0])


# ---------------------------------------------------------------------------
# Action applier / observation getter
# ---------------------------------------------------------------------------


def action_applier(
    context: ExecutionContext, action: Any, env_mask: Optional[np.ndarray] = None
) -> None:
    """Apply base pose, EEF pose and gripper to the env."""
    if action is None:
        return
    env = context.backend.env

    # Move the robot base if the action contains base commands.
    if "base_position" in action and "base_orientation" in action:
        env.set_operator_base_pose(
            "arm",
            action["base_position"],
            action["base_orientation"],
            env_mask=env_mask,
        )

    env.apply_pose_action(
        "arm",
        action["position"],
        action["orientation"],
        action["gripper"],
        env_mask=env_mask,
    )


def observation_getter(context: ExecutionContext) -> dict:
    return context.backend.env.capture_observation()


# ---------------------------------------------------------------------------
# Recorded demo policy
# ---------------------------------------------------------------------------


class RecordedDemoPolicy:
    """Replays recorded EEF poses + gripper (and optional base) step by step.

    Each array is ``(T, B, dim)`` so indexing by step yields ``(B, dim)``.
    """

    def __init__(self, demo: dict) -> None:
        self.positions = demo["position"]  # (T, B, 3)
        self.orientations = demo["orientation"]  # (T, B, 4)
        self.grippers = demo["gripper"]  # (T, B, 1)
        self.base_positions = demo.get("base_position")  # (T, B, 3) or None
        self.base_orientations = demo.get("base_orientation")  # (T, B, 4) or None
        self._max = len(self.positions) - 1
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(
        self, observation: Any, update: TaskUpdate, evaluator: PolicyEvaluator
    ) -> dict:
        i = min(self._step, self._max)
        self._step += 1
        action: dict[str, Any] = {
            "position": self.positions[i],  # (B, 3)
            "orientation": self.orientations[i],  # (B, 4)
            "gripper": self.grippers[i],  # (B, 1)
        }
        if self.base_positions is not None:
            action["base_position"] = self.base_positions[i]  # (B, 3)
            action["base_orientation"] = self.base_orientations[i]  # (B, 4)
        return action


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--restore-initial-poses",
        action="store_true",
        help="Restore object initial poses from the recorded demo JSON "
        "so objects start exactly where they were during recording.",
    )
    args = parser.parse_args()

    if not DEMO_PATH.exists():
        raise FileNotFoundError(
            f"Demo not found: {DEMO_PATH}\n"
            f"Record first: python examples/record_demo.py "
            f"--config-name {CONFIG_NAME}"
        )

    demo = load_demo(DEMO_PATH)
    batch_size = demo["batch_size"]
    task_file = load_task_file_hydra(
        CONFIG_NAME, overrides=[f"env.batch_size={batch_size}"]
    )
    print(
        f"Loaded {len(demo['position'])} steps from {DEMO_PATH} "
        f"(batch_size={batch_size})"
    )

    # Optionally load recorded initial object poses from the companion JSON.
    recorded_initial_poses: List[Dict[str, InitialPoseConfig]] = []
    if args.restore_initial_poses:
        json_path = DEMO_PATH.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(
                f"Demo JSON not found: {json_path}\n"
                "The JSON file is required for --restore-initial-poses."
            )
        recorded_initial_poses = load_initial_poses(json_path)
        obj_names = (
            list(recorded_initial_poses[0].keys()) if recorded_initial_poses else []
        )
        print(f"Restoring initial poses for: {obj_names}")

    policy = RecordedDemoPolicy(demo)
    evaluator = PolicyEvaluator(
        action_applier=action_applier,
        observation_getter=observation_getter,
    ).from_config(task_file, 10)

    max_updates = len(demo["position"]) + 50
    try:
        # Write recorded initial poses into the backend before reset so
        # that objects start where they were during the recording session.
        if recorded_initial_poses:
            apply_recorded_initial_poses(evaluator, recorded_initial_poses)

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
