"""Seeding policy shared by every randomness source in AAO.

AAO has exactly two seed states: a **fixed** run seed, or **no** run seed. The
absence of a run seed is spelled ``None`` — never ``0`` — so every integer,
including ``0``, is a usable seed.

An unseeded run resolves to one concrete entropy-derived integer, which lets a
caller log or record it: the run stays random, but it becomes replayable after
the fact with ``task.seed=<recorded value>``. Resolving once (instead of
passing ``None`` down to each generator) is what makes that value reportable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["resolve_run_seed"]


def resolve_run_seed(seed: int | None) -> int:
    """Return the concrete seed a randomness source must be built from.

    ``None`` means "no run seed was chosen" and resolves to an entropy-derived
    integer. Any integer, including ``0``, is returned unchanged: ``0`` is a
    valid seed, not a sentinel for "unseeded".
    """
    if seed is None:
        return int(np.random.SeedSequence().entropy)
    return int(seed)
