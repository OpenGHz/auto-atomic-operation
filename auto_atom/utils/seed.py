"""Seeding policy shared by every randomness source in AAO.

AAO has exactly two seed states: a **fixed** run seed, or **no** run seed. The
absence of a run seed is spelled ``None`` — never ``0`` — so every integer,
including ``0``, is a usable seed.

An unseeded run resolves to one concrete entropy-derived integer, which lets a
caller log or record it: the run stays random, but it becomes replayable after
the fact with ``task.seed=<recorded value>``. Resolving once (instead of
passing ``None`` down to each generator) is what makes that value reportable.

Every reset draws from its own stream, derived from the run seed and the
reset's 1-based number (:func:`reset_generator`). What one episode consumes
therefore never shifts the next one, so episode ``N`` of a seed is the same
whether or not the episodes before it ran to completion.
"""

from __future__ import annotations

import numpy as np

__all__ = ["reset_generator", "resolve_run_seed"]


def resolve_run_seed(seed: int | None) -> int:
    """Return the concrete seed a randomness source must be built from.

    ``None`` means "no run seed was chosen" and resolves to an entropy-derived
    integer. Any integer, including ``0``, is returned unchanged: ``0`` is a
    valid seed, not a sentinel for "unseeded".
    """
    if seed is None:
        return int(np.random.SeedSequence().entropy)
    return int(seed)


def reset_generator(seed: int, reset_index: int) -> np.random.Generator:
    """The generator reset number ``reset_index`` of run ``seed`` draws from.

    A child stream of the run seed (``SeedSequence`` spawn key), so streams of
    different resets are statistically independent.
    """
    if reset_index < 0:
        raise ValueError(f"reset_index must be non-negative, got {reset_index}")
    return np.random.default_rng(
        np.random.SeedSequence(int(seed), spawn_key=(int(reset_index),))
    )
