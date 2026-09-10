"""Contract test for the backend-neutral randomization executor.

The point of this file is the *interface*. ``_RecordingHost`` implements only
the ``RandomizationHost`` capabilities — read a pose, write a pose, read a
recorded baseline, read geometry, report diagnostics — and a complete
randomization reset runs against it: the executor compiles the plan, selects
regions, resolves references, samples, and enforces ordering on its own.

The host therefore declares no plan, no strategy, no sampler, and no camera
randomization config. If the executor ever consults the host for randomization
policy again, these tests fail loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auto_atom.config.randomization import (
    OperatorRandomizationConfig,
    PoseRandomRange,
    RandomizationConstraintConfig,
    RandomizationFailureConfig,
    RandomizationSpec,
    RandomizationStrategy,
    RandomizationVisibilityConfig,
    ResolvedRandomizationConfig,
    ResolvedRandomizationScope,
)
from auto_atom.contracts import (
    CameraModel,
    RandomizationConstraintReport,
    SupportGeometry,
)
from auto_atom.randomization import RandomizationFailureError
from auto_atom.randomization_executor import RandomizationExecutor
from auto_atom.utils.pose import PoseState

ARM = "arm"
CUP = "cup"
PLATE = "plate"
CAMERA = "head_cam"

BASE_POSITIONS = {CUP: 5.0, PLATE: 7.0}


def _default_constraints(
    *,
    visible: bool = False,
) -> RandomizationConstraintConfig:
    return RandomizationConstraintConfig(
        failure=RandomizationFailureConfig(max_attempts=3),
        visible_in=(
            RandomizationVisibilityConfig(cameras=[CAMERA]) if visible else None
        ),
    )


def _entities(*, visible: bool = False) -> Dict[str, object]:
    """One operator, one standalone object, and a reference-connected pair.

    ``plate`` references ``cup``, so those two form a single two-member
    component — the shape that makes the placement strategy observable. With
    ``visible`` the ``cup`` entry asks for a rigid ``visible_in`` requirement, so
    the executor's deterministic preflight has something to reject.
    """
    return {
        ARM: OperatorRandomizationConfig(base=PoseRandomRange(x=(0.0, 0.0))),
        CUP: RandomizationSpec(
            proposal=PoseRandomRange(x=(0.0, 0.0)),
            constraints=_default_constraints(visible=visible),
        ),
        PLATE: RandomizationSpec(
            proposal=PoseRandomRange.model_validate(
                {"x": {"range": [0.0, 0.0], "reference": CUP}}
            ),
            constraints=_default_constraints(),
        ),
    }


def _config(
    strategy: RandomizationStrategy,
    *,
    visible: bool = False,
) -> ResolvedRandomizationConfig:
    """A resolved config; cameras inherit the scope, so they are listed here."""
    return ResolvedRandomizationConfig(
        scope=ResolvedRandomizationScope(
            entities=_entities(visible=visible),
            cameras={CAMERA: PoseRandomRange(x=(0.0, 0.0))},
            strategy=strategy,
        )
    )


class _RecordingHost:
    """A backend that implements only the randomization host capabilities."""

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)
        self.events: List[str] = []
        self.applied: List[str] = []
        self.diagnostics: List[dict] = []
        self.poses: Dict[str, PoseState] = {
            f"{ARM}.base": PoseState(position=(0.0, 0.0, 0.0)),
            f"{ARM}.eef": PoseState(position=(0.0, 0.0, 0.0)),
            CUP: PoseState(position=(BASE_POSITIONS[CUP], 0.0, 0.0)),
            PLATE: PoseState(position=(BASE_POSITIONS[PLATE], 0.0, 0.0)),
            CAMERA: PoseState(),
        }
        # Recorded reset baselines. Objects and the camera have one; operator
        # parts are left unrecorded, so the live-pose fallback is exercised too.
        self._baselines: Dict[str, PoseState] = {
            CUP: self.poses[CUP],
            PLATE: self.poses[PLATE],
            CAMERA: self.poses[CAMERA],
        }

    # --- host surface -----------------------------------------------------
    @property
    def batch_size(self) -> int:
        return 1

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def seed(self) -> Optional[int]:
        return 0

    @property
    def episode_index(self) -> int:
        return 0

    @property
    def object_names(self) -> Set[str]:
        return {CUP, PLATE}

    @property
    def operator_names(self) -> Set[str]:
        return {ARM}

    def live_pose(self, label: str) -> PoseState:
        if label not in self.poses:
            raise KeyError(label)
        return self.poses[label]

    def baseline_pose(self, label: str) -> Optional[PoseState]:
        return self._baselines.get(label)

    def get_camera_pose(self, camera_name: str) -> PoseState:
        return self.poses[CAMERA]

    def set_camera_pose(self, camera_name: str, pose: PoseState, env_mask) -> None:
        self.events.append("camera")
        self.poses[CAMERA] = pose

    def camera_names(self) -> List[str]:
        return [CAMERA]

    def get_camera_model(self, camera_name: str, env_index: int) -> CameraModel:
        # A degenerate clip range: every box is provably outside the frustum, so
        # a ``visible_in`` region is deterministically infeasible. That is what
        # lets a test observe the executor's preflight without a simulator.
        return CameraModel(
            name=camera_name,
            pose=PoseState(),
            width=640,
            height=480,
            fovy_radians=1.0,
            near=5.0,
            far=1.0,
        )

    def get_support_geometry(self, entity_name: str, env_index: int) -> SupportGeometry:
        return SupportGeometry(center=(0.0, 0.0, 0.0), radius=0.01)

    def get_operator_support_geometry(
        self,
        operator_name: str,
        part: str,
        env_index: int,
    ) -> SupportGeometry:
        return SupportGeometry(center=(0.0, 0.0, 0.0), radius=0.01)

    def set_target_pose(self, kind, owner, pose, env_mask) -> None:
        # The host addresses an element part, not a randomization action: it
        # never learns which label the executor sampled.
        label = (
            owner
            if kind == "object"
            else f"{owner}.{'base' if kind == 'operator_base' else 'eef'}"
        )
        self.events.append(f"apply:{label}")
        self.applied.append(label)
        self.poses[label] = pose

    def evaluate_pose_constraints(self, candidate_poses, **kwargs):
        return RandomizationConstraintReport(valid=True)

    def record_reset_diagnostics(self, env_index, diagnostics) -> None:
        self.events.append(f"diagnostic:{diagnostics.get('generator', '')}")
        self.diagnostics.append(dict(diagnostics))


def _operator_label() -> str:
    return f"{ARM}.base"


def _reset(host: _RecordingHost, config: ResolvedRandomizationConfig) -> None:
    executor = RandomizationExecutor(host, config)
    executor.apply_randomization(np.asarray([True], dtype=bool))


def test_host_needs_no_randomization_policy_to_run_a_reset() -> None:
    host = _RecordingHost()
    # The host exposes capabilities only: no strategy, no plan, no camera config.
    assert not hasattr(host, "randomization_strategy")
    assert not hasattr(host, "randomization_plan")
    assert not hasattr(host, "camera_randomization")
    assert not hasattr(host, "sample_target")

    executor = RandomizationExecutor(host, _config(RandomizationStrategy.RSA))
    executor.apply_randomization(np.asarray([True], dtype=bool))

    # The executor compiled the plan and drove every target itself.
    assert set(host.applied) == set(executor.plan.actions)
    assert host.applied[0] == _operator_label()


def test_reset_orders_operators_before_cameras_before_objects() -> None:
    host = _RecordingHost()

    _reset(host, _config(RandomizationStrategy.RSA))

    camera = host.events.index("camera")
    operator_apply = host.events.index(f"apply:{_operator_label()}")
    object_apply = host.events.index(f"apply:{CUP}")
    # Operators shape the frame that mounted cameras and object references see,
    # so their poses are written first; cameras are final before objects are
    # sampled, because object visibility constraints are evaluated against them.
    assert operator_apply < camera < object_apply


def test_executor_owned_preflight_rejects_a_region_that_can_never_be_visible() -> None:
    host = _RecordingHost()
    executor = RandomizationExecutor(
        host,
        _config(RandomizationStrategy.RSA, visible=True),
    )

    with pytest.raises(RandomizationFailureError) as error:
        executor.apply_randomization(np.asarray([True], dtype=bool))

    assert error.value.attempts == 0
    assert error.value.target == CUP
    # The preflight runs against the final camera poses, before any object is
    # written, and reports through the host's diagnostics sink.
    assert host.events.index("camera") < host.events.index(
        "diagnostic:deterministic_infeasible"
    )
    assert f"apply:{CUP}" not in host.events
    assert host.diagnostics[0]["generator"] == "deterministic_infeasible"


def test_placement_strategy_comes_from_the_config() -> None:
    rsa_host = _RecordingHost()
    rsa = RandomizationExecutor(rsa_host, _config(RandomizationStrategy.RSA))
    # The hard-sphere RSA loop is the one that consumes dependency edges.
    assert rsa.plan.strategy == RandomizationStrategy.RSA
    assert PLATE in rsa.plan.dependencies
    assert rsa.plan.dependencies[PLATE] == {CUP}
    rsa.apply_randomization(np.asarray([True], dtype=bool))

    joint_host = _RecordingHost()
    joint = RandomizationExecutor(
        joint_host,
        _config(RandomizationStrategy.JOINT_REJECTION),
    )
    assert joint.plan.strategy == RandomizationStrategy.JOINT_REJECTION
    joint.apply_randomization(np.asarray([True], dtype=bool))

    # Both strategies place the same targets; only the placement differs.
    assert joint_host.applied == rsa_host.applied


def test_camera_randomization_comes_from_the_config() -> None:
    host = _RecordingHost()
    config = _config(RandomizationStrategy.RSA)
    executor = RandomizationExecutor(host, config)

    assert executor.camera_randomization == config.scope.cameras

    executor.apply_randomization(np.asarray([True], dtype=bool))

    assert "camera" in host.events


def test_camera_randomization_requires_only_pose_access() -> None:
    """A camera is driven by the config, not by a host randomization hook."""
    host = _RecordingHost()
    executor = RandomizationExecutor(host, _config(RandomizationStrategy.RSA))

    executor.apply_camera_randomization(np.asarray([True], dtype=bool))

    assert host.events == ["camera"]


def test_reset_is_reproducible_across_identical_hosts() -> None:
    """A reference-connected reset repeats exactly, from host state alone."""
    first, second = _RecordingHost(), _RecordingHost()
    config = _config(RandomizationStrategy.RSA)

    _reset(first, config)
    _reset(second, config)

    assert first.applied == second.applied
    for label in (CUP, PLATE, CAMERA):
        assert np.allclose(first.poses[label].position, second.poses[label].position)
