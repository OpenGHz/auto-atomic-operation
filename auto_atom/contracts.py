"""Backend-neutral contracts, protocols, and data models.

This module owns the simulation-agnostic interfaces shared by task runners,
scene backends, and simulator environments:

* handler ABCs (``ObjectHandler`` / ``OperatorHandler``) and contact data;
* environment capability protocols (``EnvProtocol`` and its narrow variants);
* camera / support-geometry / randomization constraint data models;
* the :class:`SceneBackend` abstract base and the ``require_env_capability``
  structural-check helpers used across backends.

It is intentionally free of any concrete runner or simulator implementation
so every backend (native MuJoCo, Gaussian-Splatting rendering, mock, …) shares
one contract surface.  Split out of the former ``auto_atom.runtime`` monolith.
"""

from __future__ import annotations

import inspect
import logging
import operator
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np

from auto_atom.config.motion import EefControlConfig, PoseControlConfig
from auto_atom.config.primitives import Position
from auto_atom.config.task import AutoAtomConfig, TaskFileConfig
from auto_atom.execution_model import ControlResult
from auto_atom.utils.pose import PoseState

logger = logging.getLogger(__name__)


@dataclass
class ObjectHandler(ABC):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(
                f"ObjectHandler.name must be a non-empty string; got {self.name!r}."
            )

    @abstractmethod
    def get_pose(self) -> PoseState:
        """Return the object's batched world pose."""

    @abstractmethod
    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> None:
        """Set the object's batched world pose for selected environments."""


@dataclass(frozen=True)
class ContactObservation:
    """One backend-neutral contact between an operator and the scene."""

    operator_body: str
    operator_geom: str
    other_body: str
    other_geom: str
    position_world_m: Position
    signed_distance_m: float
    penetration_depth_m: float
    normal_force_n: Optional[float] = None
    tangential_force_n: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-native representation for execution records."""
        return {
            "operator_body": self.operator_body,
            "operator_geom": self.operator_geom,
            "other_body": self.other_body,
            "other_geom": self.other_geom,
            "position_world_m": [float(value) for value in self.position_world_m],
            "signed_distance_m": float(self.signed_distance_m),
            "penetration_depth_m": float(self.penetration_depth_m),
            "normal_force_n": (
                None if self.normal_force_n is None else float(self.normal_force_n)
            ),
            "tangential_force_n": (
                None
                if self.tangential_force_n is None
                else float(self.tangential_force_n)
            ),
        }


@runtime_checkable
class IKSolver(Protocol):
    def solve(
        self,
        target_pose_in_base: PoseState,
        current_qpos: np.ndarray,
    ) -> Optional[np.ndarray]: ...


class OperatorHandler(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name used by stage configs."""

    @abstractmethod
    def move_to_pose(
        self,
        pose: PoseControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance motion toward the desired pose for selected envs."""

    @abstractmethod
    def control_eef(
        self,
        eef: EefControlConfig,
        target: Optional[ObjectHandler],
        env_mask: Optional[np.ndarray] = None,
    ) -> ControlResult:
        """Advance the end-effector toward the desired state for selected envs."""

    @abstractmethod
    def get_end_effector_pose(self) -> PoseState:
        """Return batched world poses for the operator end-effector."""

    @abstractmethod
    def get_base_pose(self) -> PoseState:
        """Return batched world poses for the operator base."""

    def get_reached_tolerances(self) -> tuple[Any, Any]:
        """Return default position and orientation tolerances for REACHED.

        Backends whose operator configuration exposes different tolerances
        should override this method.  Keeping this on the public handler
        contract prevents Stage execution from depending on a backend's
        private configuration object.
        """
        return 0.01, 0.08

    def get_placed_tolerances(self) -> tuple[Any, Any]:
        """Return default position and orientation tolerances for PLACED.

        ``None`` means that the corresponding placement component is not
        constrained unless the Stage supplies an explicit tolerance.
        """
        return None, None

    def set_pose(
        self,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
    ) -> None:
        raise NotImplementedError

    def set_home_joint_positions(
        self,
        joint_positions: Mapping[str, object],
        env_mask: Optional[np.ndarray] = None,  # noqa: ARG002
        *,
        apply_home: bool = True,  # noqa: ARG002
    ) -> None:
        """Set raw arm-joint qpos values restored by :meth:`home`.

        This optional capability keeps operator-scoped initial joints out of
        the generic environment config. Backends that do not expose joint-mode
        operators may leave the default implementation unchanged.
        """
        raise NotImplementedError(
            f"Operator '{self.name}' does not support home joint positions"
        )


@runtime_checkable
class EnvProtocol(Protocol):
    """Core batched environment interface returned by ``SceneBackend.get_env()``.

    Environment features such as stepping, observations, and simulation-loop
    updates are optional capabilities represented by the narrower protocols
    below. Callers should request only the capability they actually use.
    """

    @property
    def batch_size(self) -> int:
        """Number of environments represented by this object."""
        ...


@runtime_checkable
class StepEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for applying batched policy actions."""

    def step(
        self,
        action: np.ndarray,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
    ) -> None: ...


@runtime_checkable
class ObservationEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for capturing policy observations."""

    def capture_observation(self) -> Dict[str, Dict[str, Any]]: ...


@runtime_checkable
class JointActionEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for directly applying operator joint actions."""

    def apply_joint_action(
        self,
        operator: str,
        action: Any,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
        kinematic: bool = False,
    ) -> None: ...


@runtime_checkable
class PoseActionEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for directly applying operator pose actions."""

    def apply_pose_action(
        self,
        operator: str,
        position: Any,
        orientation: Any,
        gripper: Any = None,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
    ) -> None: ...


@runtime_checkable
class KinematicPoseActionEnvProtocol(PoseActionEnvProtocol, Protocol):
    """Pose-action capability that also supports kinematic application."""

    def apply_pose_action(
        self,
        operator: str,
        position: Any,
        orientation: Any,
        gripper: Any = None,
        /,
        *,
        env_mask: Optional[np.ndarray] = None,
        kinematic: bool = False,
    ) -> None: ...


@runtime_checkable
class SimulationLoopEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for independently advancing simulation state."""

    def update(self) -> None: ...


@runtime_checkable
class InfoEnvProtocol(EnvProtocol, Protocol):
    """Environment capability for returning serializable metadata."""

    def get_info(self) -> Dict[str, Any]: ...


@dataclass(frozen=True)
class CameraModel:
    """Backend-neutral pinhole camera model for reset constraints."""

    name: str
    pose: PoseState
    width: int
    height: int
    fovy_radians: float
    near: float = 0.0
    far: float = float("inf")


@dataclass(frozen=True)
class SupportGeometry:
    """Conservative world-frame-independent support geometry for an entity."""

    center: np.ndarray
    radius: float
    points: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if center.shape != (3,):
            raise ValueError("SupportGeometry.center must contain three values")
        if self.radius < 0.0:
            raise ValueError("SupportGeometry.radius must be non-negative")
        if self.points is not None:
            points = np.asarray(self.points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("SupportGeometry.points must have shape (N, 3)")
            object.__setattr__(self, "points", points)
        object.__setattr__(self, "center", center)


@dataclass(frozen=True)
class PoseConstraintReport:
    """Result of evaluating one candidate pose set against hard constraints."""

    valid: bool
    violations: tuple[str, ...] = ()
    minimum_clearance: float = float("inf")


@runtime_checkable
class PoseConstraintEnvProtocol(EnvProtocol, Protocol):
    """Optional environment capability for camera/geometry constraints."""

    def get_camera_model(self, camera_name: str) -> CameraModel: ...

    def get_support_geometry(self, entity_name: str) -> SupportGeometry: ...

    def evaluate_pose_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int = 0,
        constraints: Any = None,
    ) -> PoseConstraintReport: ...


_EnvCapabilityT = TypeVar("_EnvCapabilityT")
_ENV_CAPABILITY_CACHE: Dict[
    tuple[int, type[Any]],
    weakref.ReferenceType[object],
] = {}


def require_env_capability(
    env: object,
    capability: type[_EnvCapabilityT],
    *,
    feature: str,
    expected_batch_size: Optional[int] = None,
) -> _EnvCapabilityT:
    """Return *env* narrowed to *capability* or raise a clear runtime error.

    A successful structural/signature check is cached for the lifetime of a
    weak-referenceable environment. Environment capability methods are
    therefore expected to remain stable after construction; batch-size values
    are still checked on every call.
    """
    cached = _is_environment_capability_cached(env, capability)
    if not cached:
        if not (
            getattr(capability, "_is_protocol", False)
            and getattr(capability, "_is_runtime_protocol", False)
        ):
            raise TypeError(
                f"{capability!r} is not a runtime-checkable environment protocol."
            )
        required_members = _environment_protocol_members(capability)
        missing_members = []
        for member in required_members:
            try:
                inspect.getattr_static(env, member)
            except AttributeError:
                missing_members.append(member)
        if missing_members:
            missing = (
                f" Missing attributes: {', '.join(missing_members)}."
                if missing_members
                else ""
            )
            raise RuntimeError(
                f"{feature} requires environment capability {capability.__name__}; "
                f"got {type(env).__name__}.{missing}"
            )

    narrowed = cast(_EnvCapabilityT, env)
    try:
        environment_batch_size = getattr(narrowed, "batch_size")
    except Exception as exc:
        raise RuntimeError(
            f"{feature} requires a readable environment batch_size; "
            f"{type(env).__name__}.batch_size raised "
            f"{type(exc).__name__}: {exc}."
        ) from exc
    actual_batch_size = _require_positive_batch_size(
        environment_batch_size,
        owner="environment",
        feature=feature,
    )
    normalized_expected_batch_size = (
        None
        if expected_batch_size is None
        else _require_positive_batch_size(
            expected_batch_size,
            owner="backend",
            feature=feature,
        )
    )
    if (
        normalized_expected_batch_size is not None
        and actual_batch_size != normalized_expected_batch_size
    ):
        raise RuntimeError(
            f"{feature} requires environment batch_size to match "
            f"backend.batch_size; got {actual_batch_size} and "
            f"{normalized_expected_batch_size}."
        )
    if not cached:
        _validate_environment_protocol_signatures(
            narrowed,
            capability,
            feature=feature,
        )
        _cache_environment_capability(env, capability)
    return narrowed


def _is_environment_capability_cached(
    env: object,
    capability: type[Any],
) -> bool:
    reference = _ENV_CAPABILITY_CACHE.get((id(env), capability))
    return reference is not None and reference() is env


def _cache_environment_capability(
    env: object,
    capability: type[Any],
) -> None:
    key = (id(env), capability)

    def discard(reference: weakref.ReferenceType[object]) -> None:
        if _ENV_CAPABILITY_CACHE.get(key) is reference:
            _ENV_CAPABILITY_CACHE.pop(key, None)

    try:
        reference = weakref.ref(env, discard)
    except TypeError:
        return
    _ENV_CAPABILITY_CACHE[key] = reference


def _require_positive_batch_size(
    value: object,
    *,
    owner: str,
    feature: str,
) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise RuntimeError(
            f"{feature} requires {owner} batch_size to be an integer; got {value!r}."
        )
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise RuntimeError(
            f"{feature} requires {owner} batch_size to be an integer; got {value!r}."
        ) from exc
    if normalized <= 0:
        raise RuntimeError(
            f"{feature} requires {owner} batch_size to be positive; got {normalized}."
        )
    return normalized


def _environment_protocol_members(capability: type[Any]) -> List[str]:
    members: set[str] = set()
    for base in capability.__mro__:
        members.update(
            name
            for name in getattr(base, "__annotations__", {})
            if not name.startswith("_")
        )
        members.update(
            name
            for name, value in vars(base).items()
            if not name.startswith("_")
            and (callable(value) or isinstance(value, property))
        )
    return sorted(members)


def _validate_environment_protocol_signatures(
    env: object,
    capability: type[Any],
    *,
    feature: str,
) -> None:
    checked: set[str] = set()
    for base in capability.__mro__:
        for member_name, expected_member in vars(base).items():
            if (
                member_name.startswith("_")
                or member_name in checked
                or not callable(expected_member)
            ):
                continue
            checked.add(member_name)
            try:
                actual_member = getattr(env, member_name)
            except Exception as exc:
                raise RuntimeError(
                    f"{feature} requires readable "
                    f"{capability.__name__}.{member_name}; "
                    f"{type(env).__name__}.{member_name} raised "
                    f"{type(exc).__name__}: {exc}."
                ) from exc
            if not callable(actual_member):
                raise RuntimeError(
                    f"{feature} requires callable "
                    f"{capability.__name__}.{member_name}; got "
                    f"{type(env).__name__}.{member_name}="
                    f"{actual_member!r}."
                )
            callable_implementation = getattr(actual_member, "__call__", None)
            if any(
                inspect.iscoroutinefunction(candidate)
                or inspect.isasyncgenfunction(candidate)
                or inspect.isgeneratorfunction(candidate)
                for candidate in (actual_member, callable_implementation)
                if candidate is not None
            ):
                raise RuntimeError(
                    f"{feature} requires synchronous "
                    f"{capability.__name__}.{member_name}; got "
                    f"{type(env).__name__}.{member_name}."
                )
            expected_signature = inspect.signature(expected_member)
            try:
                actual_signature = inspect.signature(actual_member)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{feature} requires introspectable "
                    f"{capability.__name__}.{member_name}; wrap "
                    f"{type(env).__name__}.{member_name} in a Python method "
                    "with an explicit signature."
                ) from exc

            for args, kwargs in _protocol_signature_calls(expected_signature):
                try:
                    actual_signature.bind(*args, **kwargs)
                except TypeError as exc:
                    raise RuntimeError(
                        f"{feature} requires {capability.__name__}.{member_name} "
                        f"with a signature compatible with {expected_signature}; "
                        f"got {type(env).__name__}.{member_name}{actual_signature}."
                    ) from exc


def _protocol_signature_calls(
    signature: inspect.Signature,
) -> List[tuple[List[object], Dict[str, object]]]:
    sentinel = object()
    parameters = list(signature.parameters.values())
    if parameters and parameters[0].name in {"self", "cls"}:
        parameters = parameters[1:]

    minimal_args: List[object] = []
    minimal_kwargs: Dict[str, object] = {}
    full_args: List[object] = []
    full_kwargs: Dict[str, object] = {}
    for parameter in parameters:
        required = parameter.default is inspect.Parameter.empty
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            if required:
                minimal_args.append(sentinel)
            full_args.append(sentinel)
        elif parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if required:
                minimal_args.append(sentinel)
                full_args.append(sentinel)
            else:
                full_kwargs[parameter.name] = sentinel
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            if required:
                minimal_kwargs[parameter.name] = sentinel
            full_kwargs[parameter.name] = sentinel

    return [
        (minimal_args, minimal_kwargs),
        (full_args, full_kwargs),
    ]


class SceneBackend(ABC):
    @abstractmethod
    def get_env(self) -> EnvProtocol:
        """Return the stable basis environment exposed to runners and policies.

        This method is called after backend construction and before ``setup``;
        the environment object and its capability methods must already exist.
        """

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of envs in the backend batch."""

    @abstractmethod
    def setup(self, config: AutoAtomConfig) -> None:
        """Prepare task state after environment and handler construction."""

    @abstractmethod
    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        """Reset selected envs for a new run."""

    @abstractmethod
    def teardown(self) -> None:
        """Release backend resources after execution."""

    @abstractmethod
    def get_operator_handler(self, name: str) -> OperatorHandler:
        """Resolve an operator handler by name."""

    @abstractmethod
    def get_object_handler(self, name: str) -> Optional[ObjectHandler]:
        """Resolve an object handler by name. Empty names may return None."""

    def apply_object_pose(
        self,
        object_name: str,
        pose: PoseState,
        env_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Apply a kinematic object pose through the backend-neutral contract."""
        handler = self.get_object_handler(object_name)
        if handler is None:
            raise KeyError(f"Unknown object {object_name!r}.")
        handler.set_pose(pose, env_mask=env_mask)

    @abstractmethod
    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        """Return whether the operator is currently grasping the given object."""

    @abstractmethod
    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        """Return whether the operator is currently grasping any object."""

    @abstractmethod
    def get_grasped_object_name(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[str]:
        """Return the object grasped by an operator in one environment.

        Return ``None`` when the operator is empty-handed.  This is the
        stable public lookup used by PLACE validation; implementations may
        keep their object registries private.
        """

    @property
    def dt_per_update(self) -> float:
        """Simulation time advanced per update() call, in seconds.

        Backends that track physics time should override this.
        Returns 0.0 by default (unknown).
        """
        return 0.0

    @contextmanager
    def defer_viewer_updates(self) -> Iterator[None]:
        """Defer viewer refreshes until a compound runner update completes.

        Backends without an interactive viewer keep the default no-op
        implementation. Viewer-backed implementations should coalesce all
        refresh requests inside this context into one final refresh.
        """
        yield

    def is_object_displaced(
        self,
        object_name: str,
        original_pose: PoseState,
        threshold: float = 0.01,
    ) -> np.ndarray:
        handler = self.get_object_handler(object_name)
        if handler is None:
            return np.zeros(self.batch_size, dtype=bool)
        current = handler.get_pose()
        if original_pose.batch_size != self.batch_size:
            original_pose = original_pose.broadcast_to(self.batch_size)
        delta = np.linalg.norm(
            np.asarray(current.position, dtype=np.float64)
            - np.asarray(original_pose.position, dtype=np.float64),
            axis=1,
        )
        return delta > threshold

    @abstractmethod
    def is_operator_contacting(
        self,
        operator_name: str,
        object_name: str,
    ) -> np.ndarray:
        """Return whether the operator contacts the object in each environment."""

    @abstractmethod
    def get_operator_contacts(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[List[ContactObservation]]:
        """Observe current contacts between an operator and non-operator bodies.

        ``None`` means that the backend does not support contact observations;
        an empty list means that it observed the environment and found no
        current external contacts.
        """

    @abstractmethod
    def is_element_rigidly_attached_to_object(
        self,
        element_name: str,
        object_name: str,
        env_index: int = 0,
    ) -> bool:
        """Return whether a named frame is in the object's rigid subtree.

        Implementations must return ``False`` for an existing but unrelated or
        articulated element, and raise ``KeyError`` for unknown element or
        object names. This ownership check is the backend-independent guard for
        ``controlled_frame.kind='held_object'`` named frames.
        """

    def get_element_pose(self, name: str, env_index: int = 0) -> PoseState:  # noqa: ARG002
        raise NotImplementedError(
            f"Backend does not support named element lookup (requested '{name}')."
        )

    def get_joint_angle(self, name: str, env_index: int = 0) -> float:  # noqa: ARG002
        raise NotImplementedError(
            f"Backend does not support joint angle lookup (requested '{name}')."
        )

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        """Notify the backend about the current task-focus objects and operations."""

    @property
    def rng(self) -> Optional[np.random.Generator]:
        """The backend-owned RNG, when it has one.

        Runner-owned seeded randomness is used when a backend returns ``None``.
        This keeps waypoint randomization deterministic without reaching into a
        backend's private state.
        """
        return None

    def get_camera_poses(self, env_index: int) -> Dict[str, PoseState]:
        """Return the camera poses this reset ended up with.

        Reported in reset diagnostics so a run can be reconstructed from what
        the cameras actually saw. Backends without cameras return an empty
        mapping.
        """
        return {}

    def get_reset_diagnostics(self, env_index: int = 0) -> Dict[str, Any]:
        """Return diagnostics produced by the most recent reset.

        A backend reports here why a placement was hard (exhausted attempts,
        provably empty constraint intersection), so a failing run can be
        explained without re-running it.
        """
        return {}

    # Pose-constraint evaluation (`get_camera_model`,
    # `get_support_geometry`, `get_operator_support_geometry`,
    # `evaluate_pose_constraints`) is deliberately *not* part of this contract:
    # it is what the feasibility layer needs in order to decide whether a
    # candidate placement is legal, and it is declared there instead, by
    # ``RandomizationHost``. A backend that supports constrained randomization
    # implements those members structurally; a backend that does not needs to
    # know nothing about them.


def _teardown_backend_after_initialization_failure(backend: SceneBackend) -> None:
    try:
        backend.teardown()
    except Exception:
        logger.exception("Backend teardown failed after initialization error.")


def construct_scene_backend(
    config: TaskFileConfig,
    *,
    feature: str,
) -> SceneBackend:
    """Construct and validate the backend declared by one task file.

    This is the common lifecycle entry point for runners and simulator tools.
    It intentionally does not call :meth:`SceneBackend.setup`: task runners
    compile and attach their execution timeline first, while scene-only
    clients (for example ``view_scene``) can set up immediately.  The backend
    remains the owner of the simulator environment and its lifecycle; this
    helper only centralizes construction and contract validation.
    """

    backend = config.backend(config.task, config.task_operators)
    if not isinstance(backend, SceneBackend):
        raise TypeError(
            "Task file backend must be an instantiated SceneBackend. "
            f"Got {type(backend).__name__}."
        )
    try:
        require_env_capability(
            backend.get_env(),
            EnvProtocol,
            feature=feature,
            expected_batch_size=backend.batch_size,
        )
    except BaseException:
        _teardown_backend_after_initialization_failure(backend)
        raise
    return backend
