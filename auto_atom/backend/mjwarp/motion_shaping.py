"""Cartesian step shaping and stall scaling for MJWarp arm motion.

``move_to_pose`` does not send the final waypoint straight to IK. It sends a
*clamped sub-target* a bounded distance from the current pose, so the arm sweeps
toward the goal instead of asking IK for a large jump that would either fail or
snap through an unintended branch. Two pieces of that shaping are pure functions
of the current and desired poses, so they live here and are tested on their own:

* :func:`clamp_cartesian_step` -- bound how far the commanded sub-target may sit
  from where the tool is now, in position and in angle.
* :class:`StallScaler` -- shrink the step when progress stalls and grow it back
  when progress resumes.

The distinction that makes this correct, and which the native handler documents
at its call site: **shaping changes only the command for this tick, never the
completion test.** Completion is always measured against the final waypoint, so
reaching the first clamped sub-step does not advance the stage after a few
millimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from auto_atom.utils.pose import quaternion_angular_distance
from auto_atom.utils.transformations import quaternion_slerp


def clamp_cartesian_step(
    current_position: np.ndarray,
    current_orientation: np.ndarray,
    desired_position: np.ndarray,
    desired_orientation: np.ndarray,
    max_linear_step: float,
    max_angular_step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bound a commanded sub-target's distance from the current pose.

    A non-positive bound disables that axis of clamping, which is how the native
    config expresses "no limit" (``cartesian_max_linear_step: 0.0``). Position is
    clamped by scaling along the straight line to the goal; orientation by
    slerping a fraction of the way, so a large rotation is approached at a
    bounded angular rate rather than in one jump.

    Returns the goal unchanged when it is already inside both bounds.
    """
    position = np.asarray(desired_position, dtype=np.float64).copy()
    orientation = np.asarray(desired_orientation, dtype=np.float64).copy()
    current_position = np.asarray(current_position, dtype=np.float64)
    current_orientation = np.asarray(current_orientation, dtype=np.float64)

    if max_linear_step > 0.0:
        delta = position - current_position
        distance = float(np.linalg.norm(delta))
        if distance > max_linear_step:
            position = current_position + delta * (max_linear_step / distance)

    if max_angular_step > 0.0:
        angle = quaternion_angular_distance(current_orientation, orientation)
        if angle > max_angular_step:
            orientation = quaternion_slerp(
                current_orientation,
                orientation,
                fraction=max_angular_step / angle,
            )

    return position, orientation


def clamp_joint_delta(
    solved: np.ndarray,
    seed: np.ndarray,
    max_delta: float,
) -> np.ndarray:
    """Bound how far one control step may move any single joint.

    IK is free to return a solution on a different kinematic branch than the
    seed -- elbow flipped, wrist rolled the other way -- which is a valid pose
    for the target but reaches it by sweeping the arm through a large motion.
    Clamping the per-step displacement suppresses that: the arm walks toward the
    new branch over several ticks instead of lunging at it.

    The clamp scales the *whole* delta vector by the worst single joint, rather
    than clipping each joint independently, so the intermediate pose stays on
    the straight line in joint space toward the solution. Clipping per joint
    would bend that path and can steer the arm somewhere neither pose intended.

    A non-positive ``max_delta`` disables the clamp.
    """
    solved = np.asarray(solved, dtype=np.float64)
    seed = np.asarray(seed, dtype=np.float64)
    if max_delta <= 0.0:
        return solved
    delta = solved - seed
    if delta.size == 0:
        return solved
    largest = float(np.max(np.abs(delta)))
    if largest > max_delta:
        return seed + delta * (max_delta / largest)
    return solved


@dataclass
class StallScaler:
    """Per-world step scaling that reacts to progress.

    Contact-heavy motion can stall for reasons a smaller step will not fix, so
    this is opt-in (native's ``adaptive_step_scaling``); when disabled the scale
    stays at 1.0 and the caller's clamp bounds apply unmodified.

    The thresholds come from the native handler: progress counts as a 1e-4 m
    improvement in position *or* a 1e-3 rad improvement in orientation, eight
    consecutive stalled ticks halve the scale (floored at 0.1), and each
    improvement grows it by 10% (capped at 1.0). Growth is slower than
    shrinkage on purpose -- backing off hard and recovering gently avoids
    oscillating around a hard spot.
    """

    nworld: int
    enabled: bool = True
    stall_patience: int = 8

    _best_position_error: np.ndarray = field(init=False, repr=False)
    _best_orientation_error: np.ndarray = field(init=False, repr=False)
    _stall_count: np.ndarray = field(init=False, repr=False)
    _scale: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.nworld < 1:
            raise ValueError(f"nworld must be >= 1; got {self.nworld}.")
        self._best_position_error = np.full(self.nworld, np.inf, dtype=np.float64)
        self._best_orientation_error = np.full(self.nworld, np.inf, dtype=np.float64)
        self._stall_count = np.zeros(self.nworld, dtype=np.int64)
        self._scale = np.ones(self.nworld, dtype=np.float64)

    def scale(self, world: int) -> float:
        """Current step-scale multiplier for one world."""
        return float(self._scale[self._check_world(world)])

    def stall_count(self, world: int) -> int:
        return int(self._stall_count[self._check_world(world)])

    def reset(self, world: int) -> None:
        """Forget progress history, e.g. when the commanded waypoint changes.

        Carrying the previous waypoint's best errors into a new one would read
        as an immediate stall, since the new target's initial error is unrelated
        to the old target's best.
        """
        index = self._check_world(world)
        self._best_position_error[index] = np.inf
        self._best_orientation_error[index] = np.inf
        self._stall_count[index] = 0
        self._scale[index] = 1.0

    def update(
        self, world: int, position_error: float, orientation_error: float
    ) -> float:
        """Record this tick's errors and return the scale to use.

        Returns 1.0 unchanged when scaling is disabled, so a caller can always
        multiply by the result without branching.
        """
        index = self._check_world(world)
        if not self.enabled:
            return 1.0

        improved = position_error < (
            self._best_position_error[index] - 1e-4
        ) or orientation_error < (self._best_orientation_error[index] - 1e-3)

        if improved:
            self._best_position_error[index] = min(
                self._best_position_error[index], position_error
            )
            self._best_orientation_error[index] = min(
                self._best_orientation_error[index], orientation_error
            )
            self._stall_count[index] = 0
            self._scale[index] = min(1.0, self._scale[index] * 1.1)
        else:
            self._stall_count[index] += 1
            if self._stall_count[index] >= self.stall_patience:
                self._scale[index] = max(0.1, self._scale[index] * 0.5)
                self._stall_count[index] = 0

        return float(self._scale[index])

    def _check_world(self, world: int) -> int:
        index = int(world)
        if not 0 <= index < self.nworld:
            raise IndexError(f"world must be in [0, {self.nworld}); got {index}.")
        return index


def effective_step_bounds(
    waypoint_linear: float,
    waypoint_angular: float,
    default_linear: float,
    default_angular: float,
    scale: float,
) -> Tuple[float, float]:
    """Resolve step bounds for a tick: waypoint override, else default, times scale.

    A waypoint's own bound wins when it is positive, matching native's
    ``pose.max_linear_step if pose.max_linear_step > 0.0 else
    self.control.cartesian_max_linear_step``: zero means "not specified here",
    not "no motion allowed".
    """
    linear = waypoint_linear if waypoint_linear > 0.0 else default_linear
    angular = waypoint_angular if waypoint_angular > 0.0 else default_angular
    return float(linear) * float(scale), float(angular) * float(scale)
