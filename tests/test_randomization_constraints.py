"""Contract tests for the backend-neutral randomization constraint evaluator.

These exercise ``RandomizationConstraintEvaluator`` through a minimal fake
backend (toy camera model + toy support geometry) instead of MuJoCo. They are
the executable form of the claim that a new backend inherits ``visible_in`` /
``separated`` semantics by implementing only ``get_camera_model`` and
``get_support_geometry``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.config.randomization import RandomizationConstraintConfig
from auto_atom.contracts import CameraModel, SupportGeometry
from auto_atom.randomization import RandomizationConstraintEvaluator
from auto_atom.utils.pose import PoseState

# A 640x480 pinhole camera at the world origin with an identity orientation,
# which is the camera convention used throughout: the camera looks along -z
# with +y up and +x right, and ``fovy_radians=1.0`` keeps the arithmetic
# readable.
CAMERA = CameraModel(
    name="head_cam",
    pose=PoseState(
        position=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    ),
    width=640,
    height=480,
    fovy_radians=1.0,
    near=0.1,
    far=10.0,
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> PoseState:
    return PoseState(
        position=np.asarray([[x, y, z]], dtype=np.float64),
        orientation=np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
    )


class _FakeEnv:
    """Minimal environment exposing only the two reads a backend must supply."""

    def __init__(self, radii: dict[str, float]) -> None:
        self._radii = radii
        self.camera_calls: list[str] = []
        self.geometry_calls: list[str] = []

    def get_camera_model(self, camera_name: str) -> CameraModel:
        self.camera_calls.append(camera_name)
        return CAMERA

    def get_support_geometry(self, entity_name: str) -> SupportGeometry:
        self.geometry_calls.append(entity_name)
        radius = self._radii[entity_name]
        return SupportGeometry(
            center=np.zeros(3, dtype=np.float64),
            radius=radius,
        )

    def evaluator(self) -> RandomizationConstraintEvaluator:
        return RandomizationConstraintEvaluator()

    def evaluate(
        self,
        evaluator: RandomizationConstraintEvaluator,
        candidate_poses,
        constraints,
        *,
        target_names=None,
        ancestors=None,
    ):
        return evaluator.evaluate(
            candidate_poses,
            constraints,
            camera_model_of=self.get_camera_model,
            support_geometry_of=self.get_support_geometry,
            all_camera_names=("head_cam",),
            ancestors=ancestors,
            target_names=target_names,
        )


def test_in_front_of_camera_is_visible_and_behind_it_is_not() -> None:
    env = _FakeEnv({"cup": 0.02})
    evaluator = env.evaluator()
    constraints = RandomizationConstraintConfig(
        visible_in={"cameras": "all", "geometry": "center"}
    )

    assert env.evaluate(evaluator, {"cup": _pose(z=-2.0)}, constraints).valid

    behind = env.evaluate(evaluator, {"cup": _pose(z=2.0)}, constraints)
    assert not behind.valid
    assert behind.violations == ("cup:outside_depth:head_cam",)


def test_bounding_sphere_rejects_a_target_that_only_leaves_the_frame() -> None:
    constraints = RandomizationConstraintConfig(
        visible_in={"cameras": "all", "geometry": "bounding_sphere"}
    )
    pose = _pose(z=-1.5)

    # The centre is well inside the frustum, but an inflating sphere is not:
    # at depth 1.5 a radius of 1.0 covers more than half the image width.
    big = _FakeEnv({"cup": 1.0})
    report = big.evaluate(big.evaluator(), {"cup": pose}, constraints)
    assert not report.valid
    assert report.violations == ("cup:outside_view:head_cam",)

    small = _FakeEnv({"cup": 0.01})
    assert small.evaluate(small.evaluator(), {"cup": pose}, constraints).valid


def test_cameras_all_requires_the_backend_camera_set() -> None:
    env = _FakeEnv({"cup": 0.02})
    constraints = RandomizationConstraintConfig(
        visible_in={"cameras": "all", "geometry": "center"}
    )

    with pytest.raises(ValueError, match="all needs the backend's camera set"):
        env.evaluator().evaluate(
            {"cup": _pose(z=-2.0)},
            constraints,
            camera_model_of=env.get_camera_model,
            support_geometry_of=env.get_support_geometry,
        )


def test_separation_reports_the_tightest_gap() -> None:
    env = _FakeEnv({"cup": 0.1, "saucer": 0.1})
    evaluator = env.evaluator()
    constraints = RandomizationConstraintConfig(separated={"scope": "randomized"})

    report = env.evaluate(
        evaluator,
        {"cup": _pose(0.0), "saucer": _pose(0.5)},
        constraints,
    )
    assert report.valid
    assert report.minimum_clearance == pytest.approx(0.5 - 0.1 - 0.1)

    report = env.evaluate(
        evaluator,
        {"cup": _pose(0.0), "saucer": _pose(0.15)},
        constraints,
    )
    assert not report.valid
    assert report.violations == ("cup:collides:saucer",)
    assert report.minimum_clearance == pytest.approx(-0.05)


def test_separation_clearance_is_a_pure_surface_gap() -> None:
    env = _FakeEnv({"cup": 0.1, "saucer": 0.1})
    constraints = RandomizationConstraintConfig(
        separated={"scope": "randomized", "clearance": 0.4}
    )

    report = env.evaluate(
        env.evaluator(),
        {"cup": _pose(0.0), "saucer": _pose(0.5)},
        constraints,
    )
    assert not report.valid
    assert report.minimum_clearance == pytest.approx(0.5 - 0.1 - 0.1 - 0.4)


def test_ancestor_pairs_are_exempt_from_separation() -> None:
    env = _FakeEnv({"arm": 0.2, "gripper": 0.2})
    constraints = RandomizationConstraintConfig(separated={"scope": "randomized"})

    assert not env.evaluate(
        env.evaluator(),
        {"arm": _pose(0.0), "gripper": _pose(0.1)},
        constraints,
    ).valid

    exempt = env.evaluate(
        env.evaluator(),
        {"arm": _pose(0.0), "gripper": _pose(0.1)},
        constraints,
        ancestors={"arm": {"gripper"}},
    )
    # Exempted pairs leave the running minimum untouched.
    assert exempt.valid
    assert exempt.minimum_clearance == float("inf")


def test_visibility_only_checks_the_sampled_target() -> None:
    env = _FakeEnv({"cup": 0.02, "saucer": 0.02})
    constraints = RandomizationConstraintConfig(
        visible_in={"cameras": "all", "geometry": "center"}
    )

    report = env.evaluate(
        env.evaluator(),
        {"cup": _pose(z=-2.0), "saucer": _pose(z=2.0)},
        constraints,
        target_names={"cup"},
    )
    assert report.valid


def test_camera_model_and_support_radius_resolve_once_per_reset() -> None:
    env = _FakeEnv({"cup": 0.02})
    evaluator = env.evaluator()
    constraints = RandomizationConstraintConfig(
        visible_in={"cameras": "all", "geometry": "bounding_sphere"}
    )

    for _ in range(3):
        env.evaluate(evaluator, {"cup": _pose(z=-2.0)}, constraints)
    assert env.camera_calls == ["head_cam"]
    assert env.geometry_calls == ["cup"]

    evaluator.reset()
    env.evaluate(evaluator, {"cup": _pose(z=-2.0)}, constraints)
    assert env.camera_calls == ["head_cam", "head_cam"]
    assert env.geometry_calls == ["cup", "cup"]


def test_separation_geometry_is_cached_per_reset() -> None:
    env = _FakeEnv({"cup": 0.1, "saucer": 0.1})
    evaluator = env.evaluator()
    constraints = RandomizationConstraintConfig(separated={"scope": "randomized"})

    for _ in range(3):
        env.evaluate(
            evaluator,
            {"cup": _pose(0.0), "saucer": _pose(0.5)},
            constraints,
        )
    assert sorted(env.geometry_calls) == ["cup", "saucer"]


def test_unimplemented_modes_fail_closed() -> None:
    env = _FakeEnv({"cup": 0.02})

    with pytest.raises(NotImplementedError, match="segmentation visibility"):
        env.evaluate(
            env.evaluator(),
            {"cup": _pose(z=-2.0)},
            RandomizationConstraintConfig(
                visible_in={"cameras": "all", "mode": "segmentation"}
            ),
        )

    with pytest.raises(NotImplementedError, match="support-hull visibility"):
        env.evaluate(
            env.evaluator(),
            {"cup": _pose(z=-2.0)},
            RandomizationConstraintConfig(
                visible_in={"cameras": "all", "geometry": "support_hull"}
            ),
        )

    with pytest.raises(NotImplementedError, match="scene-scope separation"):
        env.evaluate(
            env.evaluator(),
            {"cup": _pose(z=-2.0)},
            RandomizationConstraintConfig(separated={"scope": "scene"}),
        )


def test_absent_constraints_are_unconstrained() -> None:
    env = _FakeEnv({})
    report = env.evaluator().evaluate(
        {"cup": _pose(0.0)},
        None,
        camera_model_of=env.get_camera_model,
        support_geometry_of=env.get_support_geometry,
    )
    assert report.valid
    assert env.camera_calls == []
    assert env.geometry_calls == []
