"""Seed semantics: ``None`` means "unseeded", any integer (including 0) is a seed.

``task.seed`` used to overload ``0`` as a sentinel for "no seed chosen", which
made the default run unreproducible and made ``0`` — a perfectly good seed —
unreachable from YAML. Every randomness source now takes ``int | None`` and
resolves it through :func:`auto_atom.utils.seed.resolve_run_seed`.
"""

from __future__ import annotations

import numpy as np

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.basis.mjc.camera_noise import CameraNoiseProcessor
from auto_atom.config.task import AutoAtomConfig
from auto_atom.runtime import ExecutionContext
from auto_atom.utils.seed import resolve_run_seed

_CAMERA = "env1_cam"
_UNSET = object()


class _FakeEnv:
    """Records the camera-noise root seed the backend hands over."""

    def __init__(self) -> None:
        self.noise_seed: object = _UNSET

    def set_camera_noise_seed(self, seed: int | None) -> None:
        self.noise_seed = seed


def _backend(seed: int | None) -> tuple[MujocoTaskBackend, _FakeEnv]:
    """Build a backend without a simulator.

    Both construction-time validators return early when no initial poses and no
    operator initial states are configured, so an empty backend is enough to
    exercise ``__post_init__`` and the RNG wiring it owns.
    """
    env = _FakeEnv()
    backend = MujocoTaskBackend(
        env=env,  # type: ignore[arg-type]
        operator_handlers={},
        object_handlers={},
        random_seed=seed,
    )
    return backend, env


def _context(seed: int | None) -> ExecutionContext:
    """Build the runner-side randomness owner without a backend."""
    return ExecutionContext(
        config=AutoAtomConfig(stages=[], env_name="seed_semantics", seed=seed),
        backend=None,  # type: ignore[arg-type]
        task_file=None,  # type: ignore[arg-type]
    )


def test_omitted_seed_is_none_and_zero_stays_a_real_seed() -> None:
    base = {"stages": [], "env_name": "seed_semantics"}
    assert AutoAtomConfig.model_validate(base).seed is None
    assert AutoAtomConfig.model_validate({**base, "seed": None}).seed is None
    assert AutoAtomConfig.model_validate({**base, "seed": 0}).seed == 0
    assert AutoAtomConfig.model_validate({**base, "seed": 42}).seed == 42


def test_resolve_run_seed_keeps_every_integer_and_resolves_none() -> None:
    assert resolve_run_seed(0) == 0
    assert resolve_run_seed(42) == 42
    assert resolve_run_seed(-7) == -7
    first, second = resolve_run_seed(None), resolve_run_seed(None)
    assert isinstance(first, int)
    assert first != second, "an unseeded run must stay random"


def test_seed_zero_is_deterministic_for_runner_side_randomness() -> None:
    a = _context(0).random_generator.uniform(0.0, 1.0, 3)
    b = _context(0).random_generator.uniform(0.0, 1.0, 3)
    assert np.array_equal(a, b)


def test_unseeded_runner_side_randomness_stays_random() -> None:
    a = _context(None).random_generator.uniform(0.0, 1.0, 3)
    b = _context(None).random_generator.uniform(0.0, 1.0, 3)
    assert not np.array_equal(a, b)


def test_seed_zero_reaches_backend_and_camera_noise_unchanged() -> None:
    first, first_env = _backend(0)
    second, _ = _backend(0)
    assert first.random_seed == 0
    assert first_env.noise_seed == 0
    assert np.array_equal(
        first.rng.uniform(0.0, 1.0, 3),
        second.rng.uniform(0.0, 1.0, 3),
    )


def test_unseeded_backend_resolves_to_a_reported_seed(caplog) -> None:
    logger_name = MujocoTaskBackend.__name__
    with caplog.at_level("WARNING", logger=logger_name):
        backend, env = _backend(None)

    resolved = backend.random_seed
    assert isinstance(resolved, int), "an unseeded run must still be reportable"
    assert env.noise_seed == resolved
    assert str(resolved) in caplog.text, "the warning must name the run's seed"
    assert "task.seed=" in caplog.text, "the warning must say how to replay it"


def test_unseeded_backends_do_not_share_a_seed() -> None:
    first, _ = _backend(None)
    second, _ = _backend(None)
    assert first.random_seed != second.random_seed


def test_camera_noise_root_seed_follows_the_same_policy() -> None:
    specs = {_CAMERA: object()}
    assert CameraNoiseProcessor(specs, seed=0)._seed == 0
    assert CameraNoiseProcessor(specs, seed=42)._seed == 42
    assert CameraNoiseProcessor(specs)._seed != CameraNoiseProcessor(specs)._seed, (
        "an unseeded processor must stay random"
    )
