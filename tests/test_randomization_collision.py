from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

import pytest

import auto_atom.backend.mjc.mujoco_backend as mujoco_backend_module
from auto_atom.backend.mjc.mujoco_backend import (
    MujocoTaskBackend,
    _CollisionParticipant,
)
from auto_atom.framework import (
    AutoAtomConfig,
    OperatorRandomizationConfig,
    PoseControlConfig,
    PoseRandomizationConfig,
    PoseRandomRange,
    RandomizationReference,
)
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
    randomization: Dict[str, PoseRandomRange | PoseRandomizationConfig],
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


def test_child_collision_resamples_reference_component() -> None:
    handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=PoseState(
            position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
        eef_pose=PoseState(
            position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = _make_backend(
        randomization={
            "arm": OperatorRandomizationConfig(
                eef=PoseRandomRange(
                    x=(0.0, 0.0),
                    y=(0.0, 0.0),
                    collision_radius=0.10,
                ),
            ),
            "vase": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.3),
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
    backend.operator_handlers = {"arm": handler}
    backend._default_operator_base_poses = {"arm": handler.get_base_pose()}
    backend._default_operator_eef_poses = {"arm": handler.get_end_effector_pose()}
    backend._rng = SequenceRNG(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
        ]
    )

    backend._apply_randomization(np.asarray([True], dtype=bool))

    vase_pos = backend.object_handlers["vase"].get_pose().position[0]
    flower_pos = backend.object_handlers["flower"].get_pose().position[0]
    assert np.allclose(vase_pos[:2], [0.3, 0.0])
    assert np.allclose(flower_pos[:2], [0.3, 0.0])


def test_direct_operator_randomization_raises_type_error() -> None:
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

    with pytest.raises(TypeError, match="nested form"):
        backend._apply_randomization(np.asarray([True], dtype=bool))


def test_direct_operator_multi_region_randomization_raises_type_error() -> None:
    pose = PoseState(
        position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=pose,
        eef_pose=pose,
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": handler},
        object_handlers={},
        randomization={
            "arm": PoseRandomizationConfig(
                regions=[PoseRandomRange(), PoseRandomRange()]
            )
        },
    )

    with pytest.raises(TypeError, match="nested form"):
        backend._apply_randomization(np.asarray([True], dtype=bool))


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


def test_legacy_single_range_and_multi_region_config_are_accepted() -> None:
    legacy_range = PoseRandomRange(x=(0.0, 0.0))
    multi_region = PoseRandomizationConfig(
        regions=[
            PoseRandomRange(x=(0.0, 0.0)),
            PoseRandomRange(x=(1.0, 1.0)),
        ]
    )

    config = AutoAtomConfig.model_validate(
        {
            "stages": [],
            "env_name": "randomization_test",
            "randomization": {
                "legacy": legacy_range.model_dump(),
                "multi": multi_region.model_dump(),
            },
        }
    )

    assert isinstance(config.randomization["legacy"], PoseRandomRange)
    assert isinstance(config.randomization["multi"], PoseRandomizationConfig)
    assert len(config.randomization["multi"].regions) == 2


def test_empty_randomization_regions_are_rejected() -> None:
    with pytest.raises(ValueError):
        PoseRandomizationConfig(regions=[])


def test_operator_nested_randomization_strips_hydra_null_switches() -> None:
    config = AutoAtomConfig.model_validate(
        {
            "stages": [],
            "env_name": "randomization_test",
            "randomization": {
                "arm": {
                    "base": {
                        "regions": None,
                        "x": [0.1, 0.2],
                    },
                    "eef": {
                        "x": None,
                        "regions": [
                            {
                                "x": [0.3, 0.4],
                                "y": None,
                            }
                        ],
                    },
                }
            },
        }
    )

    arm = config.randomization["arm"]
    assert isinstance(arm, OperatorRandomizationConfig)
    assert isinstance(arm.base, PoseRandomRange)
    assert arm.base.x == (0.1, 0.2)
    assert isinstance(arm.eef, PoseRandomizationConfig)
    assert arm.eef.regions[0].x == (0.3, 0.4)


def test_camera_and_waypoint_randomization_reject_regions() -> None:
    with pytest.raises(ValueError):
        AutoAtomConfig.model_validate(
            {
                "stages": [],
                "env_name": "randomization_test",
                "camera_randomization": {"camera": {"regions": [{"x": [0.0, 1.0]}]}},
            }
        )

    with pytest.raises(ValueError):
        PoseControlConfig.model_validate(
            {"randomization": {"regions": [{"x": [0.0, 1.0]}]}}
        )


def test_disjoint_object_regions_use_each_region_configuration() -> None:
    backend = _make_backend(
        randomization={
            "block": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(
                        reference=RandomizationReference.RELATIVE,
                        x=(0.1, 0.1),
                        y=(0.2, 0.2),
                        collision_radius=0.0,
                    ),
                    PoseRandomRange(
                        reference=RandomizationReference.ABSOLUTE_WORLD,
                        x=(1.5, 1.5),
                        y=(-0.4, -0.4),
                        collision_radius=0.0,
                    ),
                ]
            )
        },
        object_positions={"block": (1.0, 1.0, 0.0)},
    )
    backend._rng = SequenceRNG([0.0, 0.1, 0.2, 1.0, 1.5, -0.4])

    backend._apply_randomization(np.asarray([True], dtype=bool))
    first_position = backend.object_handlers["block"].get_pose().position[0].copy()
    backend._apply_randomization(np.asarray([True], dtype=bool))
    second_position = backend.object_handlers["block"].get_pose().position[0].copy()

    assert np.allclose(first_position, [1.1, 1.2, 0.0])
    assert np.allclose(second_position, [1.5, -0.4, 0.0])


def test_multi_region_selection_is_equiprobable() -> None:
    backend = _make_backend(
        randomization={
            "block": PoseRandomizationConfig(
                regions=[PoseRandomRange(), PoseRandomRange()]
            )
        },
        object_positions={"block": (0.0, 0.0, 0.0)},
    )
    regions = backend.randomization["block"]
    assert isinstance(regions, PoseRandomizationConfig)

    backend._rng = np.random.default_rng(123)
    selected = [backend._select_randomization_region(regions) for _ in range(10_000)]

    first_count = sum(region is regions.regions[0] for region in selected)
    second_count = sum(region is regions.regions[1] for region in selected)
    assert first_count + second_count == 10_000
    assert 0.45 <= first_count / 10_000 <= 0.55
    assert 0.45 <= second_count / 10_000 <= 0.55


def test_collision_retry_reselects_a_multi_region_target() -> None:
    backend = _make_backend(
        randomization={
            "obstacle": PoseRandomRange(
                reference=RandomizationReference.ABSOLUTE_WORLD,
                x=(0.0, 0.0),
                y=(0.0, 0.0),
                collision_radius=0.2,
            ),
            "block": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(
                        reference=RandomizationReference.ABSOLUTE_WORLD,
                        x=(0.0, 0.0),
                        y=(0.0, 0.0),
                        collision_radius=0.2,
                    ),
                    PoseRandomRange(
                        reference=RandomizationReference.ABSOLUTE_WORLD,
                        x=(1.0, 1.0),
                        y=(0.0, 0.0),
                        collision_radius=0.2,
                    ),
                ]
            ),
        },
        object_positions={
            "obstacle": (0.0, 0.0, 0.0),
            "block": (0.0, 0.0, 0.0),
        },
    )
    backend._rng = SequenceRNG([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

    backend._apply_randomization(np.asarray([True], dtype=bool))

    block_position = backend.object_handlers["block"].get_pose().position[0]
    assert np.allclose(block_position[:2], [1.0, 0.0])


def test_operator_eef_multi_region_randomization() -> None:
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
            "arm": OperatorRandomizationConfig(
                eef=PoseRandomizationConfig(
                    regions=[
                        PoseRandomRange(
                            reference=RandomizationReference.RELATIVE,
                            x=(0.1, 0.1),
                            collision_radius=0.0,
                        ),
                        PoseRandomRange(
                            reference=RandomizationReference.ABSOLUTE_WORLD,
                            x=(1.0, 1.0),
                            collision_radius=0.0,
                        ),
                    ]
                )
            )
        },
    )
    backend._default_operator_base_poses = {"arm": handler.get_base_pose()}
    backend._default_operator_eef_poses = {"arm": handler.get_end_effector_pose()}
    backend._rng = SequenceRNG([0.0, 0.1, 1.0, 1.0])

    backend._apply_randomization(np.asarray([True], dtype=bool))
    first_position = handler.get_end_effector_pose().position[0].copy()
    backend._apply_randomization(np.asarray([True], dtype=bool))
    second_position = handler.get_end_effector_pose().position[0].copy()

    assert np.allclose(first_position, [0.3, 0.0, 0.3])
    assert np.allclose(second_position, [1.0, 0.0, 0.3])


def test_multi_region_references_are_all_dependencies() -> None:
    pose = PoseState(
        position=np.zeros((1, 3), dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    handlers = {
        name: DummyObjectHandler(name=name, pose=pose)
        for name in ("anchor_a", "anchor_b", "block")
    }
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers=handlers,
        randomization={
            "anchor_a": PoseRandomRange(),
            "anchor_b": PoseRandomRange(),
            "block": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(reference="anchor_a"),
                    PoseRandomRange(reference="anchor_b"),
                ]
            ),
        },
    )

    dependencies = backend._randomization_dependencies()

    assert dependencies["block"] == {"anchor_a", "anchor_b"}


def test_operator_multi_region_references_are_all_dependencies() -> None:
    pose = PoseState(
        position=np.zeros((1, 3), dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    object_handlers = {
        name: DummyObjectHandler(name=name, pose=pose)
        for name in ("base_anchor", "eef_anchor")
    }
    operator_handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=pose,
        eef_pose=pose,
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": operator_handler},
        object_handlers=object_handlers,
        randomization={
            "base_anchor": PoseRandomRange(),
            "eef_anchor": PoseRandomRange(),
            "arm": OperatorRandomizationConfig(
                base=PoseRandomizationConfig(
                    regions=[PoseRandomRange(reference="base_anchor")]
                ),
                eef=PoseRandomizationConfig(
                    regions=[PoseRandomRange(reference="eef_anchor")]
                ),
            ),
        },
    )

    dependencies = backend._randomization_dependencies()

    assert dependencies["arm.base"] == {"base_anchor"}
    assert dependencies["arm.eef"] == {"arm.base", "eef_anchor"}


def test_batched_regions_preserve_per_environment_radius_and_ancestors() -> None:
    pose = PoseState(
        position=np.zeros((2, 3), dtype=np.float64),
        orientation=np.tile(
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            (2, 1),
        ),
    )
    handlers = {
        name: DummyObjectHandler(name=name, pose=pose)
        for name in ("anchor_a", "anchor_b", "block")
    }
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=2),
        operator_handlers={},
        object_handlers=handlers,
        randomization={
            "block": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(reference="anchor_a", collision_radius=0.1),
                    PoseRandomRange(reference="anchor_b", collision_radius=0.2),
                ]
            )
        },
    )
    backend._default_object_poses = {
        name: handler.get_pose() for name, handler in handlers.items()
    }
    backend._rng = SequenceRNG([0.0, 1.0])
    _poses, actions = backend._sample_randomization_component(
        ["block"],
        np.asarray([True, True], dtype=bool),
        {},
        [],
    )

    assert len(actions) == 1
    action = actions[0]
    assert np.allclose(action.radius, [0.1, 0.2])
    assert action.ancestors == [{"anchor_a"}, {"anchor_b"}]

    participant = _CollisionParticipant(
        owner="block",
        label="block",
        pose=action.pose,
        radius=action.radius,
        ancestors=action.ancestors,
    )
    assert (
        backend._find_collision_participant(
            owner_name="anchor_a",
            env_index=0,
            candidate_pose=action.pose.select(0),
            collision_radius=0.1,
            ancestors=set(),
            collision_participants=[participant],
        )
        is None
    )
    assert (
        backend._find_collision_participant(
            owner_name="anchor_a",
            env_index=1,
            candidate_pose=action.pose.select(1),
            collision_radius=0.1,
            ancestors=set(),
            collision_participants=[participant],
        )
        is participant
    )


def test_batched_region_randomization_respects_environment_mask() -> None:
    pose = PoseState(
        position=np.asarray([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
        orientation=np.tile(
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            (2, 1),
        ),
    )
    handler = DummyObjectHandler(name="block", pose=pose)
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=2),
        operator_handlers={},
        object_handlers={"block": handler},
        randomization={
            "block": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(
                        reference=RandomizationReference.ABSOLUTE_WORLD,
                        x=(1.0, 1.0),
                        collision_radius=0.0,
                    ),
                    PoseRandomRange(
                        reference=RandomizationReference.ABSOLUTE_WORLD,
                        x=(2.0, 2.0),
                        collision_radius=0.0,
                    ),
                ]
            )
        },
    )
    backend._default_object_poses = {"block": handler.get_pose()}
    backend._rng = SequenceRNG([1.0, 2.0])

    backend._apply_randomization(np.asarray([False, True], dtype=bool))

    assert np.allclose(
        handler.get_pose().position,
        [[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )


def test_batched_regions_preserve_only_selected_transitive_ancestors() -> None:
    pose = PoseState(
        position=np.zeros((2, 3), dtype=np.float64),
        orientation=np.tile(
            np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            (2, 1),
        ),
    )
    handlers = {
        name: DummyObjectHandler(name=name, pose=pose)
        for name in ("root", "anchor", "block")
    }
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=2),
        operator_handlers={},
        object_handlers=handlers,
        randomization={
            "root": PoseRandomRange(collision_radius=0.0),
            "anchor": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(collision_radius=0.0),
                    PoseRandomRange(reference="root", collision_radius=0.0),
                ]
            ),
            "block": PoseRandomRange(reference="anchor", collision_radius=0.0),
        },
    )
    backend._default_object_poses = {
        name: handler.get_pose() for name, handler in handlers.items()
    }
    backend._rng = SequenceRNG([0.0, 1.0])

    _poses, actions = backend._sample_randomization_component(
        ["root", "anchor", "block"],
        np.asarray([True, True], dtype=bool),
        {},
        [],
    )

    actions_by_label = {action.label: action for action in actions}
    assert actions_by_label["anchor"].ancestors == [set(), {"root"}]
    assert actions_by_label["block"].ancestors == [
        {"anchor"},
        {"anchor", "root"},
    ]


def test_all_regions_are_validated_before_region_selection() -> None:
    backend = _make_backend(
        randomization={
            "block": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(collision_radius=0.0),
                    PoseRandomRange(
                        reference=RandomizationReference.ABSOLUTE_BASE,
                        collision_radius=0.0,
                    ),
                ]
            )
        },
        object_positions={"block": (0.0, 0.0, 0.0)},
    )
    backend._rng = SequenceRNG([0.0])

    with pytest.raises(ValueError, match=r"region 1 cannot use 'absolute_base'"):
        backend._apply_randomization(np.asarray([True], dtype=bool))

    assert backend._rng._values == [0.0]


def test_multi_reference_dependency_order_follows_declaration_order() -> None:
    pose = PoseState(
        position=np.zeros((1, 3), dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    handlers = {
        name: DummyObjectHandler(name=name, pose=pose)
        for name in ("child", "anchor_b", "anchor_a")
    }
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers=handlers,
        randomization={
            "child": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(reference="anchor_b"),
                    PoseRandomRange(reference="anchor_a"),
                ]
            ),
            "anchor_b": PoseRandomRange(),
            "anchor_a": PoseRandomRange(),
        },
    )

    assert backend._randomization_order() == ["anchor_b", "anchor_a", "child"]


def test_unknown_multi_region_target_does_not_connect_known_components(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pose = PoseState(
        position=np.zeros((1, 3), dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    handlers = {name: DummyObjectHandler(name=name, pose=pose) for name in ("a", "b")}
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={},
        object_handlers=handlers,
        randomization={
            "a": PoseRandomRange(collision_radius=0.0),
            "b": PoseRandomRange(collision_radius=0.0),
            "ghost": PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(reference="a", collision_radius=0.0),
                    PoseRandomRange(reference="b", collision_radius=0.0),
                ]
            ),
        },
    )
    backend._default_object_poses = {
        name: handler.get_pose() for name, handler in handlers.items()
    }

    dependencies = backend._randomization_dependencies()
    components = backend._randomization_components(
        backend._randomization_order(),
        dependencies,
    )
    with caplog.at_level(logging.WARNING):
        backend._apply_randomization(np.asarray([True], dtype=bool))

    assert dependencies == {"a": set(), "b": set()}
    assert components == [["a"], ["b"]]
    assert sum("ghost" in record.getMessage() for record in caplog.records) == 1


def test_operator_eef_own_base_reference_uses_selected_base_ancestors() -> None:
    root_pose = PoseState(
        position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    root_handler = DummyObjectHandler(name="root", pose=root_pose)
    arm_handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=root_pose,
        eef_pose=PoseState(
            position=np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": arm_handler},
        object_handlers={"root": root_handler},
        randomization={
            "arm": OperatorRandomizationConfig(
                base=PoseRandomRange(reference="root", collision_radius=0.0),
                eef=PoseRandomRange(reference="arm.base", collision_radius=0.0),
            ),
            "root": PoseRandomRange(x=(0.5, 0.5), collision_radius=0.0),
        },
    )
    backend._default_object_poses = {"root": root_handler.get_pose()}
    backend._default_operator_base_poses = {"arm": arm_handler.get_base_pose()}
    backend._default_operator_eef_poses = {"arm": arm_handler.get_end_effector_pose()}
    backend._rng = SequenceRNG([0.5])

    dependencies = backend._randomization_dependencies()
    assert dependencies["arm.base"] == {"root"}
    assert dependencies["arm.eef"] == {"arm.base"}
    order = backend._randomization_order()
    _poses, actions = backend._sample_randomization_component(
        order,
        np.asarray([True], dtype=bool),
        {},
        [],
    )

    actions_by_label = {action.label: action for action in actions}
    assert actions_by_label["arm.base"].ancestors == {"root"}
    assert actions_by_label["arm.eef"].ancestors == {"arm", "root"}
    assert np.allclose(actions_by_label["arm.base"].pose.position[0], [0.5, 0.0, 0.0])
    assert np.allclose(actions_by_label["arm.eef"].pose.position[0], [0.7, 0.0, 0.0])


def test_operator_base_and_eef_dependencies_can_interleave_an_object() -> None:
    base_pose = PoseState(
        position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    child_handler = DummyObjectHandler(
        name="child",
        pose=PoseState(
            position=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    arm_handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=base_pose,
        eef_pose=PoseState(
            position=np.asarray([[0.2, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": arm_handler},
        object_handlers={"child": child_handler},
        randomization={
            "arm": OperatorRandomizationConfig(
                base=PoseRandomRange(x=(0.5, 0.5), collision_radius=0.0),
                eef=PoseRandomRange(reference="child", collision_radius=0.0),
            ),
            "child": PoseRandomRange(reference="arm.base", collision_radius=0.0),
        },
    )
    backend._default_object_poses = {"child": child_handler.get_pose()}
    backend._default_operator_base_poses = {"arm": arm_handler.get_base_pose()}
    backend._default_operator_eef_poses = {"arm": arm_handler.get_end_effector_pose()}
    backend._rng = SequenceRNG([0.5])

    order = backend._randomization_order()
    _poses, actions = backend._sample_randomization_component(
        order,
        np.asarray([True], dtype=bool),
        {},
        [],
    )

    assert order == ["arm.base", "child", "arm.eef"]
    actions_by_label = {action.label: action for action in actions}
    assert actions_by_label["child"].ancestors == {"arm"}
    assert actions_by_label["arm.eef"].ancestors == {"arm", "child"}
    assert np.allclose(actions_by_label["child"].pose.position[0], [1.5, 0.0, 0.0])
    assert np.allclose(actions_by_label["arm.eef"].pose.position[0], [0.7, 0.0, 0.0])


def test_operator_base_and_eef_select_regions_independently() -> None:
    handler = DummyOperatorHandler(
        operator_name="arm",
        base_pose=PoseState(
            position=np.asarray([[1.0, 2.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
        eef_pose=PoseState(
            position=np.asarray([[1.2, 2.0, 0.3]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        ),
    )
    backend = MujocoTaskBackend(
        env=DummyEnv(batch_size=1),
        operator_handlers={"arm": handler},
        object_handlers={},
        randomization={
            "arm": OperatorRandomizationConfig(
                base=PoseRandomizationConfig(
                    regions=[
                        PoseRandomRange(x=(0.1, 0.1), collision_radius=0.0),
                        PoseRandomRange(x=(0.2, 0.2), collision_radius=0.0),
                    ]
                ),
                eef=PoseRandomizationConfig(
                    regions=[
                        PoseRandomRange(z=(0.1, 0.1), collision_radius=0.0),
                        PoseRandomRange(
                            reference=RandomizationReference.ABSOLUTE_BASE,
                            x=(0.5, 0.5),
                            y=(0.0, 0.0),
                            z=(0.4, 0.4),
                            collision_radius=0.0,
                        ),
                    ]
                ),
            )
        },
    )
    backend._default_operator_base_poses = {"arm": handler.get_base_pose()}
    backend._default_operator_eef_poses = {"arm": handler.get_end_effector_pose()}
    backend._rng = SequenceRNG([0.0, 0.1, 1.0, 0.5, 0.0, 0.4])

    backend._apply_randomization(np.asarray([True], dtype=bool))

    assert np.allclose(handler.get_base_pose().position[0], [1.1, 2.0, 0.0])
    assert np.allclose(
        handler.get_end_effector_pose().position[0],
        [1.6, 2.0, 0.4],
    )
