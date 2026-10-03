from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from auto_atom.backend.mjc.mujoco_backend import MujocoTaskBackend
from auto_atom.config.randomization import (
    PoseRandomizationConfig,
    PoseRandomRange,
    ResolvedRandomizationConfig,
    ResolvedRandomizationScope,
)
from auto_atom.randomization_executor import FixedSample
from auto_atom.utils.pose import PoseState
from scripts.scene.tune_randomization_extremes import (
    ExtremeCase,
    RandomizationInspector,
)


class _Object:
    def __init__(self, name: str, position: tuple[float, float, float]) -> None:
        self.name = name
        self.pose = PoseState(position=np.asarray([position], dtype=np.float64))

    def get_pose(self) -> PoseState:
        # A simulator read is a fresh object; the executor fills reads in place
        # as sample buffers, so they must not alias the stored pose.
        return PoseState(
            position=self.pose.position.copy(),
            orientation=self.pose.orientation.copy(),
        )

    def set_pose(self, pose: PoseState, env_mask=None) -> None:
        self.pose = pose.broadcast_to(self.pose.batch_size)


class _ResettableEnv:
    """Restores every object on reset, as the simulator's own reset does."""

    batch_size = 1

    def __init__(self, objects: dict[str, _Object]) -> None:
        self.envs = [SimpleNamespace()]
        self._objects = objects
        self._initial = {name: obj.pose for name, obj in objects.items()}
        self.joint_writes: list[dict[str, float]] = []

    def reset(self, env_mask=None) -> None:
        for name, obj in self._objects.items():
            obj.pose = self._initial[name]

    def refresh_viewer(self) -> None:
        pass

    def set_scene_joint_positions(self, joint_names, positions, env_mask) -> None:
        self.joint_writes.append(dict(zip(joint_names, positions[0], strict=True)))


class _SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)

    def uniform(self, low: float, high: float) -> float:
        value = float(self.values.pop(0))
        assert min(low, high) <= value <= max(low, high)
        return value


def _backend(
    entities: dict,
    positions: dict[str, tuple[float, float, float]],
    *,
    joints: dict[str, tuple[float, float]] | None = None,
) -> MujocoTaskBackend:
    objects = {name: _Object(name, position) for name, position in positions.items()}
    return MujocoTaskBackend(
        env=_ResettableEnv(objects),
        operator_handlers={},
        object_handlers=objects,
        randomization=ResolvedRandomizationConfig(
            scope=ResolvedRandomizationScope(
                entities=dict(entities),
                joints=dict(joints or {}),
            )
        ),
        random_seed=0,
    )


def _inspector(backend: MujocoTaskBackend) -> RandomizationInspector:
    """The inspector without its Tk widgets."""
    inspector = object.__new__(RandomizationInspector)
    inspector.backend = backend
    inspector.env = backend.get_env()
    inspector.case_var = SimpleNamespace(set=lambda _value: None)
    inspector.desc_var = SimpleNamespace(set=lambda _value: None)
    inspector._refresh_state_text = lambda *_args, **_kwargs: None
    inspector.cases = inspector._build_cases()
    inspector.case_index = 0
    return inspector


def _case(inspector: RandomizationInspector, name: str) -> ExtremeCase:
    (case,) = [case for case in inspector.cases if case.name == name]
    return case


def _two_region_block() -> dict:
    return {
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


def test_single_axis_case_holds_everything_else_at_midpoints() -> None:
    inspector = _inspector(
        _backend(
            {
                "block": PoseRandomRange(x=(0.1, 0.3), y=(-0.2, 0.0)),
                "cup": PoseRandomRange(x=(0.0, 0.4)),
            },
            {"block": (0.0, 0.0, 0.0), "cup": (1.0, 0.0, 0.0)},
        )
    )

    case = _case(inspector, "block x=max")

    assert case.fixed is not None
    assert case.fixed.poses["block"] == FixedSample(0, {"x": 0.3, "y": -0.1})
    assert case.fixed.poses["cup"] == FixedSample(0, {"x": pytest.approx(0.2)})


def test_cases_cover_each_region_of_a_multi_region_target() -> None:
    inspector = _inspector(_backend(_two_region_block(), {"block": (0.0, 0.0, 0.0)}))
    names = [case.name for case in inspector.cases]

    assert names[:4] == ["default", "center", "all-min", "all-max"]
    for region_index in (0, 1):
        for suffix in ("all-min", "all-max", "x=min", "x=max", "y=min", "y=max"):
            assert f"block [region {region_index}] {suffix}" in names


def test_every_case_is_applied_by_the_runtime_reset_without_drawing() -> None:
    backend = _backend(_two_region_block(), {"block": (0.0, 0.0, 0.0)})
    inspector = _inspector(backend)
    backend._rng = _SequenceRng([])  # a fixed or suspended reset draws nothing

    for case in inspector.cases:
        inspector._apply_case(case)


def test_case_values_go_through_the_selected_region_frame() -> None:
    backend = _backend(_two_region_block(), {"block": (0.0, 0.0, 0.0)})
    inspector = _inspector(backend)

    inspector._apply_case(_case(inspector, "block [region 1] x=max"))
    np.testing.assert_allclose(
        backend.live_pose("block").position[0], [1.2, -0.15, 0.0]
    )

    inspector._apply_case(_case(inspector, "block [region 0] y=min"))
    np.testing.assert_allclose(backend.live_pose("block").position[0], [0.15, 0.3, 0.0])


def test_default_case_is_the_reset_baseline() -> None:
    backend = _backend(_two_region_block(), {"block": (0.5, 0.0, 0.0)})
    inspector = _inspector(backend)

    inspector._apply_case(_case(inspector, "all-max"))
    inspector._apply_case(_case(inspector, "default"))

    np.testing.assert_allclose(backend.live_pose("block").position[0], [0.5, 0.0, 0.0])


def test_random_sample_is_a_plain_runtime_reset() -> None:
    backend = _backend(
        {"block": PoseRandomRange(x=(0.1, 0.3), y=(-0.2, 0.0))},
        {"block": (1.0, 0.0, 0.0)},
    )
    inspector = _inspector(backend)
    backend._rng = _SequenceRng([0.25, -0.05])

    inspector.apply_random_sample()

    np.testing.assert_allclose(
        backend.live_pose("block").position[0], [1.25, -0.05, 0.0]
    )


def test_explicit_absolute_zero_axis_is_set_to_zero() -> None:
    backend = _backend(
        {
            "block": PoseRandomRange(
                reference="absolute_world", x=(0.0, 0.0), y=(0.1, 0.3)
            )
        },
        {"block": (0.5, 0.0, 0.0)},
    )
    inspector = _inspector(backend)

    inspector._apply_case(_case(inspector, "center"))

    np.testing.assert_allclose(backend.live_pose("block").position[0], [0.0, 0.2, 0.0])


def test_joint_cases_push_one_joint_and_hold_the_rest_at_midpoints() -> None:
    backend = _backend(
        {"block": PoseRandomRange(x=(0.0, 0.2))},
        {"block": (0.0, 0.0, 0.0)},
        joints={"hinge": (-1.0, 0.0), "slide": (0.0, 0.1)},
    )
    inspector = _inspector(backend)

    inspector._apply_case(_case(inspector, "joint hinge=min"))

    assert backend.get_env().joint_writes[-1] == {
        "hinge": -1.0,
        "slide": pytest.approx(0.05),
    }
    np.testing.assert_allclose(backend.live_pose("block").position[0], [0.1, 0.0, 0.0])


def test_summary_reports_each_region_reference() -> None:
    inspector = _inspector(_backend(_two_region_block(), {"block": (0.0, 0.0, 0.0)}))

    summary = inspector._summary_text()

    assert "block [region 0]: reference=relative" in summary
    assert "block [region 1]: reference=absolute_world" in summary
