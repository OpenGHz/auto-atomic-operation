"""IK invocation for the MJWarp backend.

The IK solvers themselves are host-side numpy and backend-agnostic
(``solve(target, seed) -> Optional[np.ndarray]``), so nothing about solving
changes here. What changes is the bookkeeping around the call, and it changes in
one specific way: **failure streaks are per world**.

Native keeps one ``_OperatorState`` per replica, so its ``ik_failure_streak`` is
naturally per env. MJWarp has one operator state serving every world, so the
streak has to become an array indexed by world -- otherwise one world's
unreachable target would trip the unreachable-threshold for all of them, failing
stages that were tracking fine. Same shape of difference as the per-world
contact scan and per-world ``ctrl``.

Two cross-cutting concerns stay in this one place, as they do natively:

* **Failure logging**, throttled by streak count. Silently dropping IK failures
  is what makes a stuck arm impossible to diagnose from the terminal, but a
  per-step IK loop against an unreachable target would flood the log, so the
  rate decays: report at 1, then on every 10th failure within each decade (10,
  20, ..., 100, 200, ..., 1000, 2000, ...).
* **Recovery logging**, so a transient miss is visibly transient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from auto_atom.utils.pose import PoseState

logger = logging.getLogger(__name__)


def should_log_failure(streak: int) -> bool:
    """Whether a failure at this streak length is reported.

    Reports at 1, then on every 10th failure inside each decade: 10, 20, ...,
    90, 100, 200, ..., 900, 1000, 2000, ... Dense early for fast feedback,
    sparse later so a long stuck loop does not flood the log.
    """
    if streak <= 0:
        return False
    if streak == 1:
        return True
    return streak >= 10 and streak % (10 ** int(np.log10(streak))) == 0


@dataclass
class MjWarpIkCaller:
    """Wraps an IK solver with per-world failure accounting.

    ``operator_name`` only appears in log messages; the caller owns which
    operator this belongs to.
    """

    solver: Any
    nworld: int
    operator_name: str = "arm"

    _failure_streak: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.nworld < 1:
            raise ValueError(f"nworld must be >= 1; got {self.nworld}.")
        self._failure_streak = np.zeros(self.nworld, dtype=np.int64)

    def failure_streak(self, world: int) -> int:
        """Consecutive IK failures for one world.

        ``move_to_pose`` reads this to declare a target unreachable instead of
        burning the whole stage timeout watching a frozen arm.
        """
        return int(self._failure_streak[self._check_world(world)])

    def reset(self, world: Optional[int] = None) -> None:
        """Clear the streak for one world, or all of them."""
        if world is None:
            self._failure_streak[:] = 0
            return
        self._failure_streak[self._check_world(world)] = 0

    def solve(
        self,
        world: int,
        target_position_b: np.ndarray,
        target_orientation_b: np.ndarray,
        seed: np.ndarray,
        context: str = "per_step_ik",
    ) -> Optional[np.ndarray]:
        """Solve for one world, tracking that world's failure streak.

        Returns the joint solution, or ``None`` when IK fails -- in which case
        the caller holds its current command rather than moving, which is what
        native does and why the failure has to be logged rather than swallowed.
        """
        index = self._check_world(world)
        target = PoseState(
            position=tuple(float(v) for v in np.asarray(target_position_b).reshape(-1)),
            orientation=tuple(
                float(v) for v in np.asarray(target_orientation_b).reshape(-1)
            ),
        )
        solution = self.solver.solve(target, seed)

        if solution is None:
            self._failure_streak[index] += 1
            streak = int(self._failure_streak[index])
            if should_log_failure(streak):
                logger.warning(
                    "[%s] IK failed for operator '%s' world %d (consecutive "
                    "failures=%d). Target in base frame: pos=%s, quat=%s. "
                    "Seed qpos=%s. The pose is likely outside the arm's reachable "
                    "workspace; the operator will hold its current ctrl until a "
                    "solvable target arrives.",
                    context,
                    self.operator_name,
                    world,
                    streak,
                    np.array2string(
                        np.asarray(target_position_b).reshape(-1), precision=4
                    ),
                    np.array2string(
                        np.asarray(target_orientation_b).reshape(-1), precision=4
                    ),
                    np.array2string(np.asarray(seed), precision=4),
                )
            return None

        if self._failure_streak[index] > 0:
            logger.info(
                "[%s] IK recovered for operator '%s' world %d after %d "
                "consecutive failures.",
                context,
                self.operator_name,
                world,
                int(self._failure_streak[index]),
            )
            self._failure_streak[index] = 0
        return solution

    def _check_world(self, world: int) -> int:
        index = int(world)
        if not 0 <= index < self.nworld:
            raise IndexError(f"world must be in [0, {self.nworld}); got {index}.")
        return index
