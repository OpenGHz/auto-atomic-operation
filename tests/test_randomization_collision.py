from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

import auto_atom.backend.mjc.mujoco_backend as mujoco_backend_module
from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.framework import PoseRandomRange, RandomizationReference
from auto_atom.utils.pose import PoseState


class SequenceRNG:
    def __init__(self, values: Iterable[float]) -> None:
        self._values = list(values)

    def uniform(self, low: float, high: float) -> float:
        if not self._values:
            raise AssertionError("SequenceRNG exhausted")
        value = float(self._values.pop(0))
        if value < min(low, high) or value > max(low, high):
            raise AssertionError(
                f"Sample {value} is outside requested range [{low}, {high}]"
            )
        return value


@dataclass
class DummyEnv:
    batch_size: int = 1


@dataclass
class DummyObjectHandler:
    name: str
    pose: PoseState

    def get_pose(self) -> PoseState:
        return self.pose

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        pose = pose.broadcast_to(self.pose.batch_size)
        mask = (
            np.ones(self.pose.batch_size, dtype=bool)
            if env_mask is None
            else np.asarray(env_mask, dtype=bool).reshape(-1)
        )
        updated_pos = self.pose.position.copy()
        updated_ori = self.pose.orientation.copy()
        updated_pos[mask] = pose.position[mask]
        updated_ori[mask] = pose.orientation[mask]
        self.pose = PoseState(position=updated_pos, orientation=updated_ori)


@dataclass
class DummyOperatorHandler:
    operator_name: str
    base_pose: PoseState
    eef_pose: PoseState

    @property
    def name(self) -> str:
        return self.operator_name

    def get_base_pose(self) -> PoseState:
        return self.base_pose

    def get_end_effector_pose(self) -> PoseState:
        return self.eef_pose

    def set_pose(self, pose: PoseState, env_mask: Optional[np.ndarray] = None) -> None:
        self.base_pose = _masked_pose_update(self.base_pose, pose, env_mask)

    def set_home_end_effector_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        self.eef_pose = _masked_pose_update(self.eef_pose, pose, env_mask)


def _masked_pose_update(
    current: PoseState,
    update: PoseState,
    env_mask: Optional[np.ndarray],
) -> PoseState:
    update = update.broadcast_to(current.batch_size)
    mask = (
        np.ones(current.batch_size, dtype=bool)
        if env_mask is None
        else np.asarray(env_mask, dtype=bool).reshape(-1)
    )
    updated_pos = current.position.copy()
    updated_ori = current.orientation.copy()
    updated_pos[mask] = update.position[mask]
    updated_ori[mask] = update.orientation[mask]
    return PoseState(position=updated_pos, orientation=updated_ori)


def _make_backend(
    randomization: Dict[str, PoseRandomRange],
    object_positions: Dict[str, tuple[float, float, float]],
) -> MujocoTaskBackend:
    object_handlers = {
        name: DummyObjectHandler(
            name=name,
            pose=PoseState(
                position=np.asarray([position], dtype=np.float64),
                orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            ),
        )
        for name, position in object_positions.items()
    }
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers=object_handlers,
        randomization=randomization,
    )
    backend._default_object_poses = {
        name: handler.get_pose() for name, handler in object_handlers.items()
    }
    return backend


def test_collision_rejection_resamples_overlapping_objects() -> None:
    backend = _make_backend(
        randomization={
            "vase": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.3),
                y=(0.0, 0.0),
                collision_radius=0.05,
            ),
            "vase2": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.3),
                y=(0.0, 0.0),
                collision_radius=0.05,
            ),
        },
        object_positions={
            "vase": (0.0, 0.0, 0.0),
            "vase2": (0.0, 0.0, 0.0),
        },
    )
    backend._rng = SequenceRNG([0.0, 0.0, 0.0, 0.0, 0.2, 0.0])

    backend._apply_randomization(np.asarray([True], dtype=bool))

    vase_pos = backend.object_handlers["vase"].get_pose().position[0]
    vase2_pos = backend.object_handlers["vase2"].get_pose().position[0]
    assert np.allclose(vase_pos[:2], [0.0, 0.0])
    assert np.allclose(vase2_pos[:2], [0.2, 0.0])


def test_reference_chain_skips_collision_rejection_with_ancestor() -> None:
    backend = _make_backend(
        randomization={
            "vase": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.0),
                y=(0.0, 0.0),
                collision_radius=0.05,
            ),
            "flower": PoseRandomRange(
                reference="vase",
                x=(0.0, 0.0),
                y=(0.0, 0.0),
                collision_radius=0.05,
            ),
        },
        object_positions={
            "vase": (0.0, 0.0, 0.0),
            "flower": (0.0, 0.0, 0.0),
        },
    )
    backend._rng = SequenceRNG([0.0, 0.0, 0.0, 0.0])

    backend._apply_randomization(np.asarray([True], dtype=bool))

    flower_pos = backend.object_handlers["flower"].get_pose().position[0]
    assert np.allclose(flower_pos[:2], [0.0, 0.0])


def test_direct_operator_randomization_updates_home_eef_pose() -> None:
    handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=PoseState(
            position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
        eef_pose=PoseState(
            position=np.asarray([[0.2, 0.0, 0.3]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": handler},
        object_handlers={},
        randomization={
            "arm": PoseRandomRange(
                x=(0.1, 0.1),
                y=(0.0, 0.0),
                collision_radius=0.1,
            )
        },
    )
    backend._default_operator_base_poses = {"arm": handler.get_base_pose()}
    backend._default_operator_eef_poses = {"arm": handler.get_end_effector_pose()}
    backend._rng = SequenceRNG([0.1, 0.0])

    backend._apply_randomization(np.asarray([True], dtype=bool))

    eef_pos = handler.get_end_effector_pose().position[0]
    assert np.allclose(eef_pos, [0.3, 0.0, 0.3])


def test_collision_rejection_warns_after_attempts_exhausted(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        mujoco_backend_module,
        "_MAX_COLLISION_REJECTION_ATTEMPTS",
        3,
    )
    backend = _make_backend(
        randomization={
            "vase": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.0),
                y=(0.0, 0.0),
                collision_radius=0.05,
            ),
            "vase2": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.0),
                y=(0.0, 0.0),
                collision_radius=0.05,
            ),
        },
        object_positions={
            "vase": (0.0, 0.0, 0.0),
            "vase2": (0.0, 0.0, 0.0),
        },
    )
    backend._rng = SequenceRNG([0.0] * 8)

    with caplog.at_level(logging.WARNING):
        backend._apply_randomization(np.asarray([True], dtype=bool))

    assert "Collision rejection exhausted for 'vase2'" in caplog.text
    vase2_pos = backend.object_handlers["vase2"].get_pose().position[0]
    assert np.allclose(vase2_pos[:2], [0.0, 0.0])
