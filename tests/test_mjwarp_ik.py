"""Tests for MJWarp IK invocation and per-world failure accounting.

The per-world streak is the point of this layer: one world's unreachable target
must not fail stages in the others. The log-throttle rule is tested directly
because it is the kind of arithmetic that looks right and silently isn't.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from auto_atom.backend.mjwarp.ik import MjWarpIkCaller, should_log_failure


class _Solver:
    """Returns a fixed solution, or None to force a failure."""

    def __init__(self, solution=None):
        self.solution = solution
        self.calls = []

    def solve(self, target, seed):
        self.calls.append((target, np.asarray(seed).copy()))
        return self.solution


def _caller(solution=None, nworld=2):
    return MjWarpIkCaller(solver=_Solver(solution), nworld=nworld)


def test_throttle_reports_first_failure_then_decades():
    """1, then every 10th inside each decade: 10..90, 100..900, 1000..."""
    assert should_log_failure(1)
    assert not any(should_log_failure(s) for s in (2, 5, 9))
    assert should_log_failure(10) and should_log_failure(20)
    assert not should_log_failure(11)
    assert should_log_failure(100) and should_log_failure(200)
    assert not should_log_failure(110), "110 is not a multiple of 100"
    assert should_log_failure(1000) and should_log_failure(2000)
    assert not should_log_failure(0)


def test_solution_is_returned_and_seed_forwarded():
    solution = np.array([0.1, 0.2, 0.3])
    caller = _caller(solution)
    seed = np.array([0.0, 0.1, 0.2])

    got = caller.solve(0, np.array([0.4, 0.0, 0.3]), np.array([0, 0, 0, 1.0]), seed)

    np.testing.assert_array_equal(got, solution)
    np.testing.assert_array_equal(caller.solver.calls[0][1], seed)


def test_target_is_passed_as_a_pose_in_base_frame():
    """The solver receives the target it was given, not a re-derived one."""
    caller = _caller(np.zeros(3))
    position = np.array([0.4, -0.1, 0.3])
    orientation = np.array([0.0, 0.0, 0.0, 1.0])

    caller.solve(0, position, orientation, np.zeros(3))

    target = caller.solver.calls[0][0]
    np.testing.assert_allclose(np.asarray(target.position).reshape(-1), position)
    np.testing.assert_allclose(np.asarray(target.orientation).reshape(-1), orientation)


def test_failures_accumulate_per_world_independently():
    """One world's unreachable target must not fail the others.

    Native gets this for free with one operator state per replica; MJWarp
    shares one state across worlds, so the streak has to be per world or the
    unreachable-threshold would trip for worlds that were tracking fine.
    """
    caller = _caller(None, nworld=2)

    for _ in range(3):
        caller.solve(0, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))
    caller.solve(1, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    assert caller.failure_streak(0) == 3
    assert caller.failure_streak(1) == 1


def test_success_clears_only_that_world_streak():
    caller = MjWarpIkCaller(solver=_Solver(None), nworld=2)
    for world in (0, 1):
        caller.solve(world, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))
    assert caller.failure_streak(0) == 1 and caller.failure_streak(1) == 1

    caller.solver.solution = np.zeros(3)
    caller.solve(0, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    assert caller.failure_streak(0) == 0
    assert caller.failure_streak(1) == 1, "world 1 keeps its streak"


def test_recovery_is_logged_so_a_transient_miss_is_visible(caplog):
    caller = MjWarpIkCaller(solver=_Solver(None), nworld=1)
    caller.solve(0, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    caller.solver.solution = np.zeros(3)
    with caplog.at_level(logging.INFO, logger="auto_atom.backend.mjwarp.ik"):
        caller.solve(0, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    assert "IK recovered" in caplog.text


def test_failure_is_logged_with_the_world_and_streak(caplog):
    caller = _caller(None, nworld=2)

    with caplog.at_level(logging.WARNING, logger="auto_atom.backend.mjwarp.ik"):
        caller.solve(1, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    assert "IK failed" in caplog.text
    assert "world 1" in caplog.text


def test_repeated_failures_are_not_logged_every_time(caplog):
    """A per-step IK loop against an unreachable target must not flood."""
    caller = _caller(None, nworld=1)

    with caplog.at_level(logging.WARNING, logger="auto_atom.backend.mjwarp.ik"):
        for _ in range(9):
            caller.solve(0, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    assert caplog.text.count("IK failed") == 1, "only the first of 9"


def test_reset_clears_streaks():
    caller = _caller(None, nworld=2)
    for world in (0, 1):
        caller.solve(world, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))

    caller.reset(0)
    assert caller.failure_streak(0) == 0 and caller.failure_streak(1) == 1

    caller.reset()
    assert caller.failure_streak(1) == 0


def test_world_index_is_bounds_checked():
    caller = _caller(np.zeros(3), nworld=2)

    with pytest.raises(IndexError, match=r"world must be in \[0, 2\)"):
        caller.solve(2, np.zeros(3), np.array([0, 0, 0, 1.0]), np.zeros(3))
    with pytest.raises(IndexError):
        caller.failure_streak(-1)


def test_nworld_must_be_positive():
    with pytest.raises(ValueError, match="nworld must be >= 1"):
        MjWarpIkCaller(solver=_Solver(), nworld=0)
