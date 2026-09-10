"""Contract test for the backend-neutral randomization executor.

The point of this file is the *interface*: a host that implements only
``RandomizationHost`` runs a complete randomization reset. It deliberately
declares **no placement strategy** — the effective policy travels on the
compiled :class:`RandomizationPlan` — so a new backend never has to know or
expose randomization policy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set

import numpy as np

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
)
from auto_atom.config.reference import RandomizationReference
from auto_atom.contracts import RandomizationConstraintReport
from auto_atom.randomization import (
    compile_randomization_plan,
    RandomizationPlan,
)
from auto_atom.randomization_executor import (
    PendingRandomizationAction,
    RandomizationExecutor,
)
from auto_atom.utils.pose import PoseState

ARM = "arm"
CUP = "cup"
PLATE = "plate"


def _spec(label: str, **kwargs) -> RandomizationSpec:
    spec = RandomizationSpec(
        proposal=PoseRandomRange(x=(0.0, 0.0)),
        constraints=RandomizationConstraintConfig(
            failure=RandomizationFailureConfig(max_attempts=3)
        ),
    )
    return spec


def _entities() -> Dict[str, object]:
    """One operator, one standalone object, and a reference-connected pair.

    ``plate`` references ``cup``, so those two form a single two-member
    component — the shape that makes the plan's placement strategy observable.
    """
    return {
        ARM: OperatorRandomizationConfig(base=PoseRandomRange(x=(0.0, 0.0))),
        CUP: _spec(CUP),
        PLATE: RandomizationSpec(
            proposal=PoseRandomRange.model_validate(
                {"x": {"range": [0.0, 0.0], "reference": CUP}}
            ),
            constraints=RandomizationConstraintConfig(
                failure=RandomizationFailureConfig(max_attempts=3)
            ),
        ),
    }


def _plan(strategy: RandomizationStrategy) -> RandomizationPlan:
    return compile_randomization_plan(
        _entities(),
        object_names={CUP, PLATE},
        operator_names={ARM},
        strategy=strategy,
    )


class _RecordingHost:
    """A backend that implements only the executor's host surface.

    There is intentionally no ``randomization_strategy`` member: if the
    executor ever consults the host for placement policy again, this host
    fails loudly.
    """

    def __init__(self, plan: RandomizationPlan) -> None:
        self._plan = plan
        self._rng = np.random.default_rng(0)
        self.events: List[str] = []
        self.applied: List[str] = []
        self.diagnostics: List[dict] = []
        self.dependency_queries = 0
        self.positions = {ARM: 0.0, CUP: 5.0, PLATE: 7.0}

    # --- host surface -----------------------------------------------------
    @property
    def randomization_rng(self) -> np.random.Generator:
        return self._rng

    @property
    def randomization_reset_index(self) -> int:
        return 0

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def object_names(self) -> Set[str]:
        return {CUP, PLATE}

    @property
    def operator_names(self) -> Set[str]:
        return {ARM}

    def randomization_plan(self) -> RandomizationPlan:
        return self._plan

    def action_dependencies(self) -> Mapping[str, Set[str]]:
        self.dependency_queries += 1
        return self._plan.dependencies

    def template_pose(self, label: str) -> PoseState:
        return PoseState()

    def sample_target(
        self,
        action,
        env_index: int,
        working_poses: Dict[str, PoseState],
        *,
        candidate_index: int,
    ):
        self.events.append(f"sample:{action.label}")
        pose = PoseState(position=(self.positions[action.owner], 0.0, 0.0))
        return {action.label: pose}, [
            PendingRandomizationAction(
                kind=action.kind,
                owner=action.owner,
                label=action.label,
                pose=pose,
                radius=0.01,
                # A real sampler always reports its frame references; the
                # executor requires them for ancestor computation.
                references=(RandomizationReference.RELATIVE,),
                constraints=action.randomization.constraints,
            )
        ]

    def evaluate_constraints(self, candidate_poses, **kwargs):
        return RandomizationConstraintReport(valid=True)

    def record_randomization_diagnostics(self, env_index, diagnostics) -> None:
        self.diagnostics.append(dict(diagnostics))

    def begin_randomization_episode(self) -> None:
        self.events.append("begin")

    def apply_action(self, action, env_mask) -> None:
        self.events.append(f"apply:{action.label}")
        self.applied.append(action.label)

    def apply_camera_randomization(self, env_mask) -> None:
        self.events.append("camera")

    def run_visibility_preflight(self, env_mask) -> None:
        self.events.append("preflight")


def _operator_label(plan: RandomizationPlan) -> str:
    return next(
        label for label, action in plan.actions.items() if action.kind != "object"
    )


def test_plan_carries_the_effective_strategy() -> None:
    assert _plan(RandomizationStrategy.RSA).strategy == RandomizationStrategy.RSA
    assert (
        _plan(RandomizationStrategy.JOINT_REJECTION).strategy
        == RandomizationStrategy.JOINT_REJECTION
    )
    # Group generation is a consequence of the strategy, also recorded on the plan.
    assert any(
        len(component) > 1 for component in _plan(RandomizationStrategy.RSA).components
    )
    assert _plan(RandomizationStrategy.JOINT_REJECTION).groups == {}


def test_host_needs_no_placement_strategy_to_run_a_reset() -> None:
    host = _RecordingHost(_plan(RandomizationStrategy.RSA))
    assert not hasattr(host, "randomization_strategy")

    RandomizationExecutor(host).apply_randomization(np.asarray([True], dtype=bool))

    # Every sampled pose reached the scene, operators first.
    assert set(host.applied) == set(host._plan.actions)
    assert host.applied[0] == _operator_label(host._plan)


def test_reset_orders_operators_before_cameras_before_objects() -> None:
    host = _RecordingHost(_plan(RandomizationStrategy.RSA))

    RandomizationExecutor(host).apply_randomization(np.asarray([True], dtype=bool))

    assert host.events[0] == "begin"
    camera = host.events.index("camera")
    preflight = host.events.index("preflight")
    operator_apply = host.events.index(f"apply:{_operator_label(host._plan)}")
    object_apply = host.events.index(f"apply:{CUP}")
    # Operators shape the frame that mounted cameras and object references see,
    # so they are sampled and applied first; the deterministic visibility
    # preflight then runs against the final camera poses, before objects.
    assert operator_apply < camera < preflight < object_apply


def test_component_placement_strategy_comes_from_the_plan() -> None:
    rsa_host = _RecordingHost(_plan(RandomizationStrategy.RSA))
    RandomizationExecutor(rsa_host).apply_randomization(np.asarray([True], dtype=bool))
    # The hard-sphere RSA loop is the only caller that reads dependency edges.
    assert rsa_host.dependency_queries > 0

    joint_host = _RecordingHost(_plan(RandomizationStrategy.JOINT_REJECTION))
    RandomizationExecutor(joint_host).apply_randomization(
        np.asarray([True], dtype=bool)
    )
    assert joint_host.dependency_queries == 0
    assert joint_host.applied == rsa_host.applied
