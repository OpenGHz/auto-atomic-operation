"""Addressed resets: a numbered reset depends on (seed, number, retry) alone.

Covers the seed primitives, the runner-side generators, the randomization
executor (against a fake batched host, no simulator), camera noise, the MuJoCo
backend, and ``PolicyEvaluator.reset(address=...)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest

import auto_atom.mock as mock
from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.basis.mjc.camera_noise import CameraNoiseProcessor
from auto_atom.basis.mjc.mujoco_env import KeyCreator
from auto_atom.config.env_config import (
    CameraNoiseConfig,
    CameraSpec,
    RGBNoiseConfig,
    TemporalNoiseConfig,
)
from auto_atom.config.randomization import (
    RandomizationScopeConfig,
    ResolvedRandomizationConfig,
)
from auto_atom.config.task import AutoAtomConfig
from auto_atom.config_loader import load_task_file_hydra
from auto_atom.contracts import (
    AddressableResetHost,
    CameraModel,
    PoseConstraintReport,
    SupportGeometry,
)
from auto_atom.policy_eval import ConfigDrivenDemoPolicy, PolicyEvaluator
from auto_atom.randomization_executor import RandomizationExecutor
from auto_atom.runtime import ComponentRegistry, ExecutionContext
from auto_atom.utils.pose import PoseState
from auto_atom.utils.seed import ResetAddress, reset_generator

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "aao_configs")
CUP = "cup"
CAMERA = "camera_color_optical_frame"


# ---------------------------------------------------------------------------
#  Seed primitives and runner-side generators
# ---------------------------------------------------------------------------


def test_retry_zero_is_the_counted_reset_and_retries_get_their_own_stream() -> None:
    counted = np.random.default_rng(np.random.SeedSequence(5, spawn_key=(3,)))

    assert reset_generator(5, 3).random() == counted.random()
    assert reset_generator(5, 3, retry=0).random() == reset_generator(5, 3).random()
    assert reset_generator(5, 3, retry=1).random() != reset_generator(5, 3).random()
    assert reset_generator(5, 3, retry=1).random() != reset_generator(5, 4).random()


@pytest.mark.parametrize(
    "fields", [{"reset_index": 0}, {"reset_index": 1, "retry": -1}]
)
def test_reset_address_is_one_based_with_a_non_negative_retry(fields: dict) -> None:
    with pytest.raises(ValueError):
        ResetAddress(**fields)


def _context(seed: int = 11) -> ExecutionContext:
    return ExecutionContext(
        config=AutoAtomConfig(stages=[], env_name="addressing", seed=seed),
        backend=mock.MockSceneBackend(env_name="addressing", batch_size=3),
        task_file=None,  # type: ignore[arg-type]
    )


def test_addressed_begin_reset_takes_the_given_number() -> None:
    context = _context()

    context.begin_reset(ResetAddress(7, retry=2))
    addressed = context.random_generator.random()
    context.begin_reset()

    assert addressed == reset_generator(11, 7, retry=2).random()
    assert context.reset_count == 8
    assert context.random_generator.random() == reset_generator(11, 8).random()


def test_an_addressed_env_keeps_its_generator_while_others_reset() -> None:
    context = _context()
    first = np.asarray([True, False, False])
    second = np.asarray([False, True, False])

    context.begin_reset(ResetAddress(1))
    context.bind_env_generators(first, addressed=True)
    context.begin_reset(ResetAddress(2))
    context.bind_env_generators(second, addressed=True)

    assert context.waypoint_generator(0).random() == reset_generator(11, 1).random()
    assert context.waypoint_generator(1).random() == reset_generator(11, 2).random()
    # An unbound env and a counted reset use the latest reset's generator.
    assert context.waypoint_generator(2) is context.random_generator
    context.begin_reset()
    context.bind_env_generators(first, addressed=False)
    assert context.waypoint_generator(0) is context.random_generator


# ---------------------------------------------------------------------------
#  Executor against a batched fake host
# ---------------------------------------------------------------------------


class _BatchedHost:
    """A batched host whose resets can be numbered, like the MuJoCo backend."""

    def __init__(self, batch_size: int, seed: int = 3) -> None:
        self._batch_size = batch_size
        self._seed = seed
        self.reset_index = 0
        self.reset_address: Optional[ResetAddress] = None
        self._rng = reset_generator(seed, 0)
        self._next: Optional[ResetAddress] = None
        self.poses: Dict[str, PoseState] = {
            CUP: PoseState(position=(0.5, 0.0, 0.0)).broadcast_to(batch_size)
        }

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def object_names(self) -> Set[str]:
        return {CUP}

    @property
    def operator_names(self) -> Set[str]:
        return set()

    def set_reset_address(self, address: ResetAddress) -> None:
        self._next = address

    def begin(self) -> None:
        address, self._next = self._next, None
        self.reset_address = address
        self.reset_index = (
            self.reset_index + 1 if address is None else address.reset_index
        )
        retry = 0 if address is None else address.retry
        self._rng = reset_generator(self._seed, self.reset_index, retry)

    def live_pose(self, label: str) -> PoseState:
        pose = self.poses[label]
        return PoseState(
            position=pose.position.copy(), orientation=pose.orientation.copy()
        )

    def baseline_pose(self, label: str) -> Optional[PoseState]:
        return PoseState(position=(0.5, 0.0, 0.0)).broadcast_to(self._batch_size)

    def camera_names(self) -> List[str]:
        return []

    def object_camera_names(self) -> Set[str]:
        return set()

    def operator_camera_names(self) -> Set[str]:
        return set()

    def set_target_pose(self, kind: str, owner: str, pose: PoseState, env_mask) -> None:
        mask = np.asarray(env_mask, dtype=bool)
        self.poses[owner].position[mask] = pose.position[mask]
        self.poses[owner].orientation[mask] = pose.orientation[mask]

    def get_camera_model(self, camera_name: str, env_index: int) -> CameraModel:
        raise KeyError(camera_name)

    def get_support_geometry(self, entity_name: str, env_index: int) -> SupportGeometry:
        return SupportGeometry(center=(0.0, 0.0, 0.0), radius=0.01)

    def evaluate_pose_constraints(self, candidate_poses, **kwargs):
        return PoseConstraintReport(valid=True)

    def record_reset_diagnostics(self, env_index, diagnostics) -> None:
        pass


def _scope(generator: str) -> ResolvedRandomizationConfig:
    distribution: Dict[str, Any] = {"generator": generator}
    if generator == "poisson_disk":
        distribution["spacing"] = 0.02
    return ResolvedRandomizationConfig.from_scope_config(
        RandomizationScopeConfig.model_validate(
            {
                "entities": {
                    CUP: {
                        "proposal": {
                            "x": [-0.2, 0.2],
                            "y": [-0.2, 0.2],
                            "yaw": [-1, 1],
                        },
                        "distribution": distribution,
                    }
                }
            }
        )
    )


def _addressed_scene(
    generator: str,
    *,
    batch_size: int,
    row: int,
    address: ResetAddress,
    resets_before: int = 0,
) -> np.ndarray:
    """Cup pose after an addressed reset of ``row``, after other resets."""
    host = _BatchedHost(batch_size)
    executor = RandomizationExecutor(host, _scope(generator))
    for _ in range(resets_before):
        host.begin()
        executor.apply_randomization(np.ones(batch_size, dtype=bool))
    host.set_reset_address(address)
    host.begin()
    mask = np.zeros(batch_size, dtype=bool)
    mask[row] = True
    executor.apply_randomization(mask)
    pose = host.poses[CUP]
    return np.concatenate([pose.position[row], pose.orientation[row]])


GENERATORS = ["iid", "sobol", "halton", "latin_hypercube", "poisson_disk"]


@pytest.mark.parametrize("generator", GENERATORS)
def test_addressed_scene_ignores_row_batch_size_and_earlier_resets(
    generator: str,
) -> None:
    address = ResetAddress(4)

    alone = _addressed_scene(generator, batch_size=1, row=0, address=address)
    elsewhere = _addressed_scene(
        generator, batch_size=3, row=2, address=address, resets_before=5
    )

    np.testing.assert_array_equal(alone, elsewhere)


@pytest.mark.parametrize("generator", GENERATORS)
def test_addressed_scenes_differ_by_number_and_by_retry(generator: str) -> None:
    def scene(address: ResetAddress) -> np.ndarray:
        return _addressed_scene(generator, batch_size=1, row=0, address=address)

    assert not np.array_equal(scene(ResetAddress(4)), scene(ResetAddress(5)))
    assert not np.array_equal(scene(ResetAddress(4)), scene(ResetAddress(4, 1)))


@pytest.mark.parametrize("generator", ["iid", "sobol", "halton"])
def test_an_addressed_reset_equals_the_counted_reset_of_a_batch_of_one(
    generator: str,
) -> None:
    host = _BatchedHost(1)
    executor = RandomizationExecutor(host, _scope(generator))
    counted = []
    for _ in range(3):
        host.begin()
        executor.apply_randomization(np.ones(1, dtype=bool))
        counted.append(host.poses[CUP].position[0].copy())

    addressed = _addressed_scene(
        generator, batch_size=1, row=0, address=ResetAddress(3)
    )

    np.testing.assert_array_equal(addressed[:3], counted[2])


def test_an_addressed_reset_clears_the_cross_reset_state() -> None:
    host = _BatchedHost(1)
    executor = RandomizationExecutor(host, _scope("poisson_disk"))
    host.begin()
    executor.apply_randomization(np.ones(1, dtype=bool))
    assert executor._poisson_streams

    host.set_reset_address(ResetAddress(9))
    host.begin()
    executor.begin_reset()

    assert not executor._poisson_streams
    assert not executor.history


# ---------------------------------------------------------------------------
#  Camera noise
# ---------------------------------------------------------------------------


def _noise_processor() -> CameraNoiseProcessor:
    spec = CameraSpec(
        name=CAMERA,
        noise=CameraNoiseConfig(
            rgb=RGBNoiseConfig(
                gaussian_std=0.05,
                temporal=TemporalNoiseConfig(jitter_std=0.02, ar1_coefficient=0.8),
            )
        ),
    )
    return CameraNoiseProcessor({CAMERA: spec}, seed=31)


def _capture(processor: CameraNoiseProcessor, batch_size: int = 2) -> np.ndarray:
    key_creator = KeyCreator(False)
    key = key_creator.create_color_key(CAMERA)
    return processor.process_batched_observation(
        {
            key: {
                "data": np.full((batch_size, 4, 4, 3), 0.5, dtype=np.float32),
                "t": np.zeros(batch_size),
            }
        },
        key_creator,
        structured=False,
        batch_size=batch_size,
    )[key]["data"]


def test_an_addressed_episode_has_the_same_noise_in_every_row() -> None:
    in_row_0 = _noise_processor()
    in_row_0.reset([0], episode=5, retry=0)
    in_row_1 = _noise_processor()
    _capture(in_row_1)
    in_row_1.reset([1], episode=5, retry=0)

    np.testing.assert_array_equal(_capture(in_row_0)[0], _capture(in_row_1)[1])

    retried = _noise_processor()
    retried.reset([0], episode=5, retry=1)
    fresh = _noise_processor()
    fresh.reset([0], episode=5, retry=0)
    assert not np.array_equal(_capture(retried)[0], _capture(fresh)[0])


def test_a_held_row_repeats_its_frame_and_does_not_advance() -> None:
    held = _noise_processor()
    held.reset([0, 1], episode=2, retry=0)
    plain = _noise_processor()
    plain.reset([0, 1], episode=2, retry=0)

    first = _capture(held)
    held.hold([0])
    repeat = _capture(held)
    after = _capture(held)
    _capture(plain)
    expected_next = _capture(plain)

    np.testing.assert_array_equal(repeat[0], first[0])
    np.testing.assert_array_equal(after[0], expected_next[0])
    # Row 1 was not held: it advanced every capture.
    assert not np.array_equal(repeat[1], first[1])


def test_holding_a_row_before_its_first_frame_has_no_effect() -> None:
    held = _noise_processor()
    held.reset([0], episode=2, retry=0)
    plain = _noise_processor()
    plain.reset([0], episode=2, retry=0)

    held.hold([0])

    np.testing.assert_array_equal(_capture(held)[0], _capture(plain)[0])


# ---------------------------------------------------------------------------
#  MuJoCo backend and the evaluator
# ---------------------------------------------------------------------------


class _FakeEnv:
    def __init__(self, unseeded: tuple = ()) -> None:
        self.unseeded_randomness = unseeded
        self.noise_episodes: List[tuple] = []

    def set_camera_noise_seed(self, seed: int | None) -> None:
        pass

    def set_camera_noise_episode(self, episode, mask, *, retry=None) -> None:
        self.noise_episodes.append((episode, np.flatnonzero(mask).tolist(), retry))

    def reset(self, env_mask=None) -> None:
        pass

    def refresh_viewer(self) -> None:
        pass


def test_mujoco_backend_numbers_the_next_reset_only(monkeypatch) -> None:
    monkeypatch.setattr(MujocoTaskBackend, "batch_size", 2)
    monkeypatch.setattr(MujocoTaskBackend, "_record_default_poses", lambda self: None)
    env = _FakeEnv()
    backend = MujocoTaskBackend(
        env=env,  # type: ignore[arg-type]
        operator_handlers={},
        object_handlers={},
        random_seed=13,
    )
    assert isinstance(backend, AddressableResetHost)

    backend.set_reset_address(ResetAddress(6, retry=1))
    backend.reset(np.asarray([False, True]))
    addressed = (backend.reset_index, backend.reset_address, backend.rng.random())
    backend.reset()

    assert addressed == (6, ResetAddress(6, 1), reset_generator(13, 6, 1).random())
    assert (backend.reset_index, backend.reset_address) == (7, None)
    assert env.noise_episodes == [(6, [1], 1), (7, [0, 1], None)]
    backend.set_reset_address(ResetAddress(1))
    with pytest.raises(ValueError, match="exactly one env"):
        backend.reset()


def test_mujoco_backend_refuses_an_address_with_unseeded_randomness() -> None:
    backend = MujocoTaskBackend(
        env=_FakeEnv(unseeded=("the GS background pool",)),  # type: ignore[arg-type]
        operator_handlers={},
        object_handlers={},
        random_seed=13,
    )

    with pytest.raises(ValueError, match="GS background pool"):
        backend.set_reset_address(ResetAddress(1))


def test_policy_evaluator_addresses_one_env_at_a_time() -> None:
    ComponentRegistry.clear()
    task_file = load_task_file_hydra(
        "policy_eval_mock", config_dir=CONFIG_DIR, overrides=["env.batch_size=2"]
    )
    policy = ConfigDrivenDemoPolicy()
    evaluator = PolicyEvaluator(action_applier=policy.action_applier).from_config(
        task_file
    )
    try:
        with pytest.raises(ValueError, match="exactly one env"):
            evaluator.reset(address=ResetAddress(1))
        evaluator.reset(np.asarray([False, True]), address=ResetAddress(3))
        assert evaluator.context.reset_count == 3
        assert evaluator.context.backend.reset_index == 3
    finally:
        evaluator.close()
        ComponentRegistry.clear()
