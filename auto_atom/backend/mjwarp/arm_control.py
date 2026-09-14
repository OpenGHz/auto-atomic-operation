"""Arm motion (``move_to_pose``) for the MJWarp backend.

This is the second half of :class:`~auto_atom.contracts.OperatorHandler`,
composing the pieces landed in earlier rounds: frame conversion, the IK caller
with its per-world failure streak, cartesian step shaping, the joint-delta
clamp, and the actuator write.

One distinction carries the correctness of the whole thing, and it is the one
the native handler documents at its own call site: **shaping changes only the
command sent on this tick, never the completion test.** The command is a clamped
sub-target a bounded distance away; completion is always measured against the
final waypoint. Measuring completion against the sub-target would advance the
stage as soon as the arm reached the first few millimetres of a long move.

Like the gripper path, this advances a complete control update for the selected
worlds before evaluating completion. The scene-state adapter preserves inactive
worlds when the runtime dispatches one environment at a time.

``solve_once_interpolate`` is not implemented. It plans a joint trajectory once
per waypoint and advances it without re-solving, which is a genuinely different
control strategy rather than a variation, so it is refused explicitly instead of
silently behaving like ``per_step_ik`` -- the same choice made for mocap mode in
:mod:`auto_atom.backend.mjwarp.operator_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from auto_atom.backend.mjwarp.frames import world_to_base
from auto_atom.backend.mjwarp.ik import MjWarpIkCaller
from auto_atom.backend.mjwarp.motion_shaping import (
    StallScaler,
    clamp_cartesian_step,
    clamp_joint_delta,
    effective_step_bounds,
)
from auto_atom.backend.mjwarp.operator_state import (
    MjWarpOperatorState,
    get_eef_pose_in_world,
)
from auto_atom.basis.mjwarp.state import MjWarpSceneState
from auto_atom.execution_model import ControlResult, ControlSignal
from auto_atom.utils.pose import position_within_tolerance, quaternion_angular_distance


@dataclass
class MjWarpArmControl:
    """Per-world cartesian arm motion driven by per-step IK."""

    state: MjWarpSceneState
    operator: MjWarpOperatorState
    ik: MjWarpIkCaller

    position_tolerance: Any = 0.01
    orientation_tolerance: float = 0.08
    timeout_steps: int = 100
    max_linear_step: float = 0.0
    max_angular_step: float = 0.0
    adaptive_step_scaling: bool = False
    ik_unreachable_threshold: int = 30
    n_substeps: int = 1

    _steps: np.ndarray = field(init=False, repr=False)
    _last_command_key: List[Optional[str]] = field(init=False, repr=False)
    _scaler: StallScaler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.operator.joint_control_mode != "per_step_ik":
            raise ValueError(
                f"Operator '{self.operator.name}' uses joint_control_mode "
                f"'{self.operator.joint_control_mode}', which the MJWarp backend "
                "does not implement. Only 'per_step_ik' is supported: "
                "'solve_once_interpolate' plans a joint trajectory once per "
                "waypoint and advances it without re-solving, a different "
                "control strategy rather than a variation."
            )
        nworld = self.state.nworld
        self._steps = np.zeros(nworld, dtype=np.int64)
        self._last_command_key = [None] * nworld
        self._scaler = StallScaler(nworld=nworld, enabled=self.adaptive_step_scaling)

    def move(
        self,
        desired_position: np.ndarray,
        desired_orientation: np.ndarray,
        *,
        waypoint_linear_step: float = 0.0,
        waypoint_angular_step: float = 0.0,
        waypoint_position_tolerance: Any = None,
        waypoint_orientation_tolerance: Optional[float] = None,
        command_key: str = "",
        world_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the arm one control tick toward a world-frame goal.

        ``command_key`` identifies the waypoint being commanded. When it changes
        the per-world step counter and stall history reset, because the previous
        waypoint's progress says nothing about this one.
        """
        nworld = self.state.nworld
        mask = self._normalize_mask(world_mask)
        signals = np.asarray([ControlSignal.RUNNING] * nworld, dtype=object)
        details: List[Dict[str, Any]] = [{} for _ in range(nworld)]
        worlds = np.flatnonzero(mask)
        if worlds.size == 0:
            return ControlResult(signals=signals, details=details)

        goal_position = np.asarray(desired_position, dtype=np.float64).reshape(-1)
        goal_orientation = np.asarray(desired_orientation, dtype=np.float64).reshape(-1)

        current_positions, current_orientations = get_eef_pose_in_world(
            self.state, self.operator
        )
        arm_qpos = self.state.get_joint_positions(self.operator.arm_qpos_indices)

        commanded: List[Tuple[int, np.ndarray]] = []
        for world in worlds:
            if self._last_command_key[world] != command_key:
                self._last_command_key[world] = command_key
                self._steps[world] = 0
                self._scaler.reset(world)

            targets = self._solve_for_world(
                world=world,
                current_position=current_positions[world],
                current_orientation=current_orientations[world],
                goal_position=goal_position,
                goal_orientation=goal_orientation,
                seed=arm_qpos[world],
                waypoint_linear_step=waypoint_linear_step,
                waypoint_angular_step=waypoint_angular_step,
            )
            if targets is not None:
                commanded.append((world, targets))
            self._steps[world] += 1

        # One ctrl write per world, then a single step request for the tick.
        for world, targets in commanded:
            world_only = np.zeros(nworld, dtype=bool)
            world_only[world] = True
            self.state.set_ctrl(
                self.operator.arm_actuator_ids, targets, world_mask=world_only
            )
        self.state.step(self.n_substeps, world_mask=mask)

        self._evaluate(
            worlds=worlds,
            goal_position=goal_position,
            goal_orientation=goal_orientation,
            waypoint_position_tolerance=waypoint_position_tolerance,
            waypoint_orientation_tolerance=waypoint_orientation_tolerance,
            signals=signals,
            details=details,
        )
        return ControlResult(signals=signals, details=details)

    def _solve_for_world(
        self,
        *,
        world: int,
        current_position: np.ndarray,
        current_orientation: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray,
        seed: np.ndarray,
        waypoint_linear_step: float,
        waypoint_angular_step: float,
    ) -> Optional[np.ndarray]:
        """Shape the goal, solve IK, and clamp the joint step for one world.

        Returns the joint command, or ``None`` when IK fails -- in which case no
        ctrl write happens and the arm holds its previous command, as native
        does. Holding rather than zeroing matters: a zeroed command would drop
        the arm under gravity on an IK miss.
        """
        position_error = float(np.linalg.norm(current_position - goal_position))
        orientation_error = quaternion_angular_distance(
            current_orientation, goal_orientation
        )
        scale = self._scaler.update(world, position_error, orientation_error)
        linear_bound, angular_bound = effective_step_bounds(
            waypoint_linear_step,
            waypoint_angular_step,
            self.max_linear_step,
            self.max_angular_step,
            scale,
        )
        shaped_position, shaped_orientation = clamp_cartesian_step(
            current_position,
            current_orientation,
            goal_position,
            goal_orientation,
            linear_bound,
            angular_bound,
        )

        target_position_b, target_orientation_b = world_to_base(
            shaped_position,
            shaped_orientation,
            self.operator.base_position[world],
            self.operator.base_orientation[world],
        )
        solution = self.ik.solve(
            world, target_position_b, target_orientation_b, seed, "per_step_ik"
        )
        if solution is None:
            return None
        return clamp_joint_delta(solution, seed, self.operator.max_joint_delta)

    def _evaluate(
        self,
        *,
        worlds: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray,
        waypoint_position_tolerance: Any,
        waypoint_orientation_tolerance: Optional[float],
        signals: np.ndarray,
        details: List[Dict[str, Any]],
    ) -> None:
        """Decide each world's signal, measured against the FINAL waypoint.

        Deliberately not against the shaped sub-target: reaching a clamped
        sub-step is not reaching the waypoint, and testing against it would
        advance the stage after a few millimetres of a long move.
        """
        positions, orientations = get_eef_pose_in_world(self.state, self.operator)
        position_tolerance = (
            waypoint_position_tolerance
            if waypoint_position_tolerance is not None
            else self.position_tolerance
        )
        orientation_tolerance = (
            waypoint_orientation_tolerance
            if waypoint_orientation_tolerance is not None
            else self.orientation_tolerance
        )

        for world in worlds:
            difference = positions[world] - goal_position
            position_error = float(np.linalg.norm(difference))
            orientation_error = quaternion_angular_distance(
                orientations[world], goal_orientation
            )
            position_ok = bool(
                position_within_tolerance(difference, position_tolerance)
            )
            orientation_ok = orientation_error <= orientation_tolerance
            streak = self.ik.failure_streak(world)
            steps = int(self._steps[world])

            details[world] = {
                "event": "pose_reached" if position_ok and orientation_ok else "moving",
                "operator": self.operator.name,
                "position_error": position_error,
                "orientation_error": orientation_error,
                "steps": steps,
                "ik_failure_streak": streak,
                "step_scale": self._scaler.scale(world),
            }

            if position_ok and orientation_ok:
                signals[world] = ControlSignal.REACHED
                self._steps[world] = 0
            elif streak >= self.ik_unreachable_threshold:
                # Persistent IK failure: fail now with a specific category
                # rather than burning the stage timeout watching a frozen arm,
                # so an unreachable target is distinguishable from slow motion.
                details[world]["event"] = "ik_unreachable"
                details[world]["failure_category"] = "ik_unreachable"
                details[world]["failure_reason"] = (
                    f"IK failed for {streak} consecutive control steps; target "
                    "pose is outside the arm's reachable workspace"
                )
                signals[world] = ControlSignal.FAILED
                self._steps[world] = 0
            elif steps >= self.timeout_steps:
                details[world]["event"] = "move_timeout"
                signals[world] = ControlSignal.TIMED_OUT
            else:
                signals[world] = ControlSignal.RUNNING

    def _normalize_mask(self, world_mask: Optional[np.ndarray]) -> np.ndarray:
        nworld = self.state.nworld
        if world_mask is None:
            return np.ones(nworld, dtype=bool)
        mask = np.asarray(world_mask, dtype=bool).reshape(-1)
        if mask.shape != (nworld,):
            raise ValueError(
                f"world_mask must have shape ({nworld},), got {mask.shape}"
            )
        return mask

    def reset(self, worlds: Optional[Sequence[int]] = None) -> None:
        """Clear motion progress, e.g. at an episode reset."""
        indices = range(self.state.nworld) if worlds is None else worlds
        for world in indices:
            self._steps[world] = 0
            self._last_command_key[world] = None
            self._scaler.reset(world)
            self.ik.reset(world)
