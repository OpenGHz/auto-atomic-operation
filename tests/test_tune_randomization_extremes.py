from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auto_atom.framework import PoseRandomizationConfig, PoseRandomRange
from auto_atom.utils.pose import PoseState
from examples.tune_randomization_extremes import (
    ExtremeCase,
    RandomizationInspector,
    RandomizationTarget,
    _sample_region_index,
)


@dataclass
class _DummyEnv:
    batch_size: int = 1

    def refresh_viewer(self) -> None:
        pass


@dataclass
class _DummyObject:
    pose: PoseState

    def get_pose(self) -> PoseState:
        return self.pose

    def set_pose(self, pose: PoseState, env_mask=None) -> None:
        self.pose = pose.broadcast_to(self.pose.batch_size)


class _SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def uniform(self, low: float, high: float) -> float:
        value = float(self.values.pop(0))
        assert low <= value <= high
        return value


class _DummyBackend:
    def __init__(self, handler: _DummyObject, randomization) -> None:
        self.batch_size = 1
        self.env = _DummyEnv()
        self.object_handlers = {"block": handler}
        self.operator_handlers = {}
        self.randomization = randomization
        self._default_object_poses = {"block": handler.get_pose()}
        self.validation_calls = 0

    def _validate_randomization_configuration(self) -> None:
        self.validation_calls += 1

    def get_env(self):
        return self.env

    def _randomization_order(self):
        return ["block"]

    def _resolve_reference_base_pose(self, _reference, _sampled, default_pose):
        return default_pose


def _make_inspector() -> tuple[RandomizationInspector, _DummyObject]:
    handler = _DummyObject(
        PoseState(
            position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        )
    )
    randomization = {
        "block": PoseRandomizationConfig(
            regions=[
                PoseRandomRange(x=(0.1, 0.2), y=(0.3, 0.4)),
                PoseRandomRange(
                    reference="absolute_world",
                    x=(1.0, 1.2),
                    y=(-0.2, -0.1),
                ),
            ]
        )
    }
    backend = _DummyBackend(handler, randomization)
    inspector = object.__new__(RandomizationInspector)
    inspector.backend = backend
    inspector.env = backend.env
    inspector.targets = [
        RandomizationTarget(
            key="object:block",
            label="object block",
            rand_range=randomization["block"],
            get_default_pose=lambda: backend._default_object_poses["block"],
            apply_pose=handler.set_pose,
            get_current_pose=handler.get_pose,
        )
    ]
    inspector.case_var = type("_Var", (), {"set": lambda self, _value: None})()
    inspector.desc_var = type("_Var", (), {"set": lambda self, _value: None})()
    inspector._refresh_state_text = lambda *_args, **_kwargs: None
    return inspector, handler


def test_build_cases_covers_each_region_without_duplicate_physical_targets() -> None:
    inspector, _handler = _make_inspector()

    cases = inspector._build_cases()
    names = {case.name for case in cases}

    for region_index in (0, 1):
        assert f"object block [region {region_index}] x=min" in names
        assert f"object block [region {region_index}] x=max" in names
        assert f"object block [region {region_index}] y=min" in names
        assert f"object block [region {region_index}] y=max" in names
    assert all(
        len(case.region_indices_by_target) <= len(inspector.targets) for case in cases
    )


def test_apply_case_uses_only_selected_region() -> None:
    inspector, handler = _make_inspector()
    handler.pose = PoseState(
        position=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    inspector.backend._default_object_poses["block"] = handler.get_pose()
    case = ExtremeCase(
        name="region-one",
        description="select region one",
        offsets_by_target={"object:block": {"x": 1.1}},
        region_indices_by_target={"object:block": 1},
    )

    inspector._apply_case(case)

    assert np.allclose(handler.get_pose().position[0], [1.1, 0.0, 0.0])


def test_build_cases_includes_reference_only_regions() -> None:
    inspector, _handler = _make_inspector()
    original_target = inspector.targets[0]
    inspector.targets = [
        RandomizationTarget(
            key=original_target.key,
            label=original_target.label,
            rand_range=PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(reference="anchor_a"),
                    PoseRandomRange(reference="anchor_b"),
                ]
            ),
            get_default_pose=original_target.get_default_pose,
            apply_pose=original_target.apply_pose,
            get_current_pose=original_target.get_current_pose,
        )
    ]

    names = {case.name for case in inspector._build_cases()}

    for region_index in (0, 1):
        assert f"object block [region {region_index}] all-min" in names
        assert f"object block [region {region_index}] all-max" in names


def test_summary_reports_each_region_reference() -> None:
    inspector, _handler = _make_inspector()

    summary = inspector._summary_text()

    assert "object block [region 0]: reference=relative" in summary
    assert "object block [region 1]: reference=absolute_world" in summary


def test_collect_targets_reuses_backend_validation() -> None:
    inspector, _handler = _make_inspector()
    inspector.operator_initial_states = {}

    targets = inspector._collect_targets()

    assert inspector.backend.validation_calls == 1
    assert [target.key for target in targets] == ["object:block"]


def test_random_sample_records_one_equiprobable_region_selection() -> None:
    inspector, handler = _make_inspector()
    inspector.rng = _SequenceRng([1.9, 1.1, -0.15])

    captured: list[ExtremeCase] = []
    inspector._apply_case = captured.append
    inspector.apply_random_sample()

    assert len(captured) == 1
    case = captured[0]
    assert case.region_indices_by_target == {"object:block": 1}
    assert case.offsets_by_target["object:block"] == {"x": 1.1, "y": -0.15}
    assert handler.get_pose().position[0, 0] == 0.0


def test_inspector_region_selection_is_equiprobable() -> None:
    rng = np.random.default_rng(123)

    selected = [_sample_region_index(rng, 2) for _ in range(10_000)]

    first_count = selected.count(0)
    assert 0.45 <= first_count / len(selected) <= 0.55


def test_random_sample_applies_explicit_absolute_zero_axis() -> None:
    inspector, handler = _make_inspector()
    handler.pose = PoseState(
        position=np.asarray([[0.7, 0.2, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )
    inspector.backend._default_object_poses["block"] = handler.get_pose()
    original_target = inspector.targets[0]
    inspector.targets = [
        RandomizationTarget(
            key=original_target.key,
            label=original_target.label,
            rand_range=PoseRandomizationConfig(
                regions=[
                    PoseRandomRange(
                        reference="absolute_world",
                        x=(0.0, 0.0),
                    )
                ]
            ),
            get_default_pose=original_target.get_default_pose,
            apply_pose=original_target.apply_pose,
            get_current_pose=original_target.get_current_pose,
        )
    ]
    inspector.rng = _SequenceRng([0.0])

    inspector.apply_random_sample()

    assert handler.get_pose().position[0, 0] == 0.0
    assert handler.get_pose().position[0, 1] == 0.2
