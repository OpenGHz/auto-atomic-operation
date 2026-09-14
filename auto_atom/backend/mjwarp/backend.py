"""MJWarp scene backend for object-only execution.

This is the object a task file's ``backend:`` entry constructs. It satisfies
:class:`~auto_atom.contracts.SceneBackend` and, structurally,
``RandomizationHost``, which together are the whole surface the runtime and the
randomization layer address a simulator through.

Scope is deliberately ``execution.mode: object_only``. That mode strips the
operator layer at the Hydra boundary, so there is no arm, no actuator, no IK and
no tactile sensor in the compiled model -- and the runtime substitutes its own
``ObjectOnlyOperatorHandler`` rather than asking the backend for one. Operator
queries here therefore fail loudly instead of fabricating an answer: an empty
grasp state would look like a legitimate "not grasping" result and silently mask
a misconfigured physical run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

from auto_atom.backend.mjwarp.handlers import MjWarpObjectHandler
from auto_atom.basis.mjwarp.env import MjWarpObjectOnlyEnv
from auto_atom.config.randomization import ResolvedRandomizationConfig
from auto_atom.config.task import AutoAtomConfig
from auto_atom.contracts import (
    CameraModel,
    ContactObservation,
    ObjectHandler,
    OperatorHandler,
    PoseConstraintReport,
    SceneBackend,
    SupportGeometry,
)
from auto_atom.utils.pose import PoseState
from auto_atom.utils.seed import resolve_run_seed

logger = logging.getLogger(__name__)

_OPERATOR_UNSUPPORTED = (
    "The MJWarp backend currently supports execution.mode=object_only, which "
    "has no embodied operator. Operator '{name}' cannot be resolved. Physical "
    "execution on MJWarp is not implemented yet (see "
    "docs/design/mjwarp-backend-design.md round 4)."
)


class MjWarpObjectOnlyBackend(SceneBackend):
    """Backend that transports objects kinematically on MJWarp worlds."""

    def __init__(
        self,
        config: AutoAtomConfig,
        env: MjWarpObjectOnlyEnv,
        object_handlers: Mapping[str, MjWarpObjectHandler],
        randomization: Optional["ResolvedRandomizationConfig"] = None,
    ) -> None:
        self.config = config
        self.env = env
        self.object_handlers: Dict[str, MjWarpObjectHandler] = dict(object_handlers)
        # The *resolved* config, not the raw scope: `.applies` folds the master
        # switch and emptiness into one decision, and a backend is not supposed
        # to hold a raw randomization config at all.
        self.randomization = (
            randomization
            if randomization is not None
            else ResolvedRandomizationConfig.from_scope_config(config.randomization)
        )

        self._reset_index = 0
        self._seed = resolve_run_seed(config.seed)
        self._rng = np.random.default_rng(self._seed)
        self._baseline_object_poses: Dict[str, PoseState] = {}
        self._baseline_camera_poses: Dict[str, PoseState] = {}
        self._last_reset_diagnostics: Dict[int, List[Dict[str, Any]]] = {}
        self._randomization_executor: Any = None

    # ------------------------------------------------------------------
    # SceneBackend lifecycle
    # ------------------------------------------------------------------

    def get_env(self) -> MjWarpObjectOnlyEnv:
        return self.env

    @property
    def batch_size(self) -> int:
        return self.env.batch_size

    def setup(self, config: AutoAtomConfig) -> None:
        self._record_baseline_poses()

    def teardown(self) -> None:
        self.env.close()

    def reset(self, env_mask: Optional[np.ndarray] = None) -> None:
        """Restore the scene, then apply randomization to the masked worlds."""
        mask = self._normalize_mask(env_mask)
        self._reset_index += 1
        self._last_reset_diagnostics.clear()
        self.env.reset()
        if not (self._baseline_object_poses or self._baseline_camera_poses):
            self._record_baseline_poses()
        if self.randomization.applies:
            self.randomization_executor.apply_randomization(mask)

    def _normalize_mask(self, env_mask: Optional[np.ndarray]) -> np.ndarray:
        if env_mask is None:
            return np.ones(self.batch_size, dtype=bool)
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape != (self.batch_size,):
            raise ValueError(
                f"env_mask must have shape ({self.batch_size},), got {mask.shape}"
            )
        return mask

    def _record_baseline_poses(self) -> None:
        """Snapshot what a reset returns to, before randomization moves it."""
        for name, handler in self.object_handlers.items():
            self._baseline_object_poses[name] = handler.get_pose()
        for camera_name in self.camera_names():
            self._baseline_camera_poses[camera_name] = self.get_camera_pose(camera_name)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def get_object_handler(self, name: str) -> Optional[ObjectHandler]:
        if not name:
            return None
        return self.object_handlers.get(name)

    def get_operator_handler(self, name: str) -> OperatorHandler:
        """Always fails: object-only execution has no embodied operator.

        The runtime substitutes its own ``ObjectOnlyOperatorHandler`` for this
        mode, so reaching here means something expected a physical operator.
        Failing loudly beats returning a stub whose empty grasp state would read
        as a legitimate "not grasping".
        """
        raise KeyError(_OPERATOR_UNSUPPORTED.format(name=name))

    # ------------------------------------------------------------------
    # Grasp and contact queries
    # ------------------------------------------------------------------

    def is_object_grasped(self, operator_name: str, object_name: str) -> np.ndarray:
        raise KeyError(_OPERATOR_UNSUPPORTED.format(name=operator_name))

    def is_operator_grasping(self, operator_name: str) -> np.ndarray:
        raise KeyError(_OPERATOR_UNSUPPORTED.format(name=operator_name))

    def get_grasped_object_name(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[str]:
        raise KeyError(_OPERATOR_UNSUPPORTED.format(name=operator_name))

    def is_operator_contacting(
        self,
        operator_name: str,
        object_name: str,
    ) -> np.ndarray:
        raise KeyError(_OPERATOR_UNSUPPORTED.format(name=operator_name))

    def get_operator_contacts(
        self,
        operator_name: str,
        env_index: int,
    ) -> Optional[List[ContactObservation]]:
        """``None`` -- no operator exists, so contacts are unobservable.

        Unlike the queries above this is a diagnostic path that the runtime
        calls opportunistically, and the contract already defines ``None`` as
        "this backend does not support contact observations", so answering it is
        honest rather than a fabrication. Object-only stage execution skips it
        anyway.
        """
        return None

    @property
    def dt_per_update(self) -> float:
        """Simulation time one ``update()`` advances, in seconds.

        Derived as the native backend derives it: model timestep times the
        substep count implied by ``sim_freq / update_freq``.
        """
        return float(self.env.host_model.opt.timestep) * self.env.n_substeps

    # ------------------------------------------------------------------
    # Frame queries
    # ------------------------------------------------------------------

    def get_element_pose(self, name: str, env_index: int = 0) -> PoseState:
        return self.env.get_element_pose(name, env_index)

    def get_joint_angle(self, name: str, env_index: int = 0) -> float:
        return self.env.state.get_joint_angle(name, env_index)

    def is_element_rigidly_attached_to_object(
        self,
        element_name: str,
        object_name: str,
        env_index: int = 0,
    ) -> bool:
        """Whether a named frame sits in the object's rigid subtree.

        This is the guard for ``controlled_frame.kind='held_object'``, which the
        rack_plate place stage uses. The walk climbs from the element's body to
        the object's body and rejects any body carrying a joint on the way:
        such a body stays in the subtree but its frame can move relative to the
        object's root, so it is not rigidly attached.
        """
        import mujoco

        handler = self.get_object_handler(object_name)
        if handler is None:
            raise KeyError(f"Unknown object {object_name!r}.")

        host_model = self.env.host_model
        object_body_id = mujoco.mj_name2id(
            host_model, mujoco.mjtObj.mjOBJ_BODY, handler.body_name
        )
        if object_body_id < 0:
            raise KeyError(
                f"Object {object_name!r} refers to missing body {handler.body_name!r}."
            )

        current = self._named_element_body_id(host_model, element_name)
        while current != object_body_id:
            if current <= 0:
                return False
            if int(host_model.body_jntnum[current]) > 0:
                return False
            current = int(host_model.body_parentid[current])
        return True

    @staticmethod
    def _named_element_body_id(host_model: Any, name: str) -> int:
        """Body owning a named site, body, geom or joint.

        Same resolution order as the native backend, so a task addressing a
        frame by name gets the same answer on either backend.
        """
        import mujoco

        site_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return int(host_model.site_bodyid[site_id])
        body_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
        geom_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            return int(host_model.geom_bodyid[geom_id])
        joint_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            return int(host_model.body_parentid[int(host_model.jnt_bodyid[joint_id])])
        raise KeyError(
            f"No site, body, geom, or joint named {name!r} found in the MuJoCo model."
        )

    def set_interest_objects_and_operations(
        self,
        object_names: List[str],
        operation_names: List[str],
    ) -> None:
        self.env.set_interest_objects_and_operations(object_names, operation_names)

    def get_camera_poses(self, env_index: int) -> Dict[str, PoseState]:
        return {
            name: self.get_camera_pose(name, env_index) for name in self.camera_names()
        }

    def get_reset_diagnostics(self, env_index: int = 0) -> Dict[str, Any]:
        entries = self._last_reset_diagnostics.get(env_index, [])
        return {"randomization": list(entries)} if entries else {}

    # ------------------------------------------------------------------
    # RandomizationHost
    # ------------------------------------------------------------------

    @property
    def object_names(self) -> Sequence[str]:
        return tuple(self.object_handlers)

    @property
    def operator_names(self) -> Sequence[str]:
        """Empty: object-only execution has no operator to randomize.

        The executor drops operator entries when this is empty, which is the
        same outcome the Hydra boundary already produced by removing them from
        ``task.randomization.entities``.
        """
        return ()

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @property
    def seed(self) -> Optional[int]:
        """The resolved run seed, so an unseeded run stays replayable.

        ``resolve_run_seed`` reports a concrete value even when the config left
        the seed unset, which keeps the Poisson-disk lattice stable across this
        run's resets while still allowing ``task.seed=<value>`` replay.
        """
        return self._seed

    @property
    def reset_index(self) -> int:
        return self._reset_index

    @property
    def randomization_executor(self) -> Any:
        """The randomization policy bound to this backend, built lazily."""
        if self._randomization_executor is None:
            from auto_atom.randomization_executor import RandomizationExecutor

            self._randomization_executor = RandomizationExecutor(
                self,
                self.randomization,
                logger=logging.getLogger(MjWarpObjectOnlyBackend.__name__),
            )
        return self._randomization_executor

    def live_pose(self, label: str) -> PoseState:
        """Current pose of one randomization target."""
        from auto_atom.randomization import parse_entity_reference

        owner, attribute = parse_entity_reference(label)
        if attribute is not None:
            raise KeyError(_OPERATOR_UNSUPPORTED.format(name=owner))
        handler = self.get_object_handler(owner)
        if handler is None:
            raise KeyError(f"Unknown target '{label}'.")
        return handler.get_pose()

    def baseline_pose(self, label: str) -> Optional[PoseState]:
        """Recorded reset baseline for one target, or ``None`` if unrecorded."""
        for recorded in (self._baseline_object_poses, self._baseline_camera_poses):
            pose = recorded.get(label)
            if pose is not None:
                return pose
        return None

    def set_target_pose(
        self,
        kind: str,
        owner: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None:
        """Write one sampled target pose into the masked worlds."""
        if kind == "object":
            handler = self.object_handlers.get(owner)
            if handler is None:
                raise KeyError(f"Unknown object '{owner}'.")
            handler.set_pose(pose, env_mask)
            return
        if kind in {"operator_base", "operator_eef"}:
            raise KeyError(_OPERATOR_UNSUPPORTED.format(name=owner))
        raise ValueError(f"Unknown target part '{kind}' for '{owner}'.")

    def camera_names(self) -> List[str]:
        return self.env.camera_names()

    def object_camera_names(self) -> frozenset[str]:
        return self.env.object_camera_names

    def get_camera_pose(self, camera_name: str, env_index: int = 0) -> PoseState:
        return self.env.get_camera_pose(camera_name, env_index)

    def set_camera_pose(
        self,
        camera_name: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None:
        self.env.set_camera_pose(camera_name, pose, env_mask)

    def get_camera_mount_pose(self, camera_name: str) -> PoseState:
        return self.env.get_camera_mount_pose(camera_name)

    def set_camera_mount_pose(
        self,
        camera_name: str,
        pose: PoseState,
        env_mask: np.ndarray,
    ) -> None:
        self.env.set_camera_mount_pose(camera_name, pose, env_mask)

    def get_camera_model(self, camera_name: str, env_index: int = 0) -> CameraModel:
        return self.env.get_camera_model(camera_name, env_index)

    def get_support_geometry(
        self,
        entity_name: str,
        env_index: int = 0,
    ) -> SupportGeometry:
        return self.env.get_support_geometry(entity_name, env_index)

    def get_operator_support_geometry(
        self,
        operator_name: str,
        part: str,
        env_index: int = 0,
    ) -> SupportGeometry:
        raise KeyError(_OPERATOR_UNSUPPORTED.format(name=operator_name))

    def evaluate_pose_constraints(
        self,
        candidate_poses: Mapping[str, PoseState],
        *,
        env_index: int = 0,
        constraints: Any = None,
        ancestors: Optional[Mapping[str, Set[str]]] = None,
        target_names: Optional[Set[str]] = None,
    ) -> PoseConstraintReport:
        return self.env.evaluate_pose_constraints(
            candidate_poses,
            env_index=env_index,
            constraints=constraints,
            ancestors=ancestors,
            target_names=target_names,
        )

    def record_reset_diagnostics(
        self,
        env_index: int,
        diagnostics: Mapping[str, Any],
    ) -> None:
        self._last_reset_diagnostics.setdefault(env_index, []).append(dict(diagnostics))


def _collect_object_names(
    config: AutoAtomConfig,
    body_exists: Any,
) -> Set[str]:
    """Names needing an object handler, resolved as the native builder does.

    Stage targets are the obvious source, but a randomized or initially-posed
    body may never appear in a stage and still needs a handler for pose
    get/set -- so those are collected too, then filtered to bodies the compiled
    model actually has.
    """
    from auto_atom.randomization import declared_randomization_references

    names = {stage.object for stage in config.stages if stage.object}

    candidates: Set[str] = set(config.randomization.entities)
    for entity_range in config.randomization.entities.values():
        for reference in declared_randomization_references(entity_range):
            if isinstance(reference, str):
                candidates.add(reference)
    candidates.update(config.initial_pose)

    names.update(name for name in candidates if body_exists(name))
    return {name for name in names if name}


def build_mjwarp_object_only_backend(
    task: Any,
    operators: Any = None,
    *,
    njmax: Optional[int] = None,
) -> MjWarpObjectOnlyBackend:
    """Construct the MJWarp object-only backend from a task file.

    ``operators`` is accepted for signature parity with the native builders and
    must be empty: ``execution.mode: object_only`` clears ``task_operators`` at
    the Hydra boundary, so a non-empty mapping means the task file is asking for
    physical execution, which this backend does not implement.
    """
    import mujoco

    from auto_atom.runtime import ComponentRegistry

    config = (
        task
        if isinstance(task, AutoAtomConfig)
        else AutoAtomConfig.model_validate(task)
    )
    if operators:
        raise ValueError(
            "The MJWarp backend supports execution.mode=object_only, which has "
            f"no operators, but received: {sorted(operators)}. Physical "
            "execution on MJWarp is not implemented yet."
        )

    env = ComponentRegistry.get_env(config.env_name)
    if not isinstance(env, MjWarpObjectOnlyEnv):
        raise TypeError(
            f"Environment '{config.env_name}' must be an MjWarpObjectOnlyEnv, "
            f"got {type(env).__name__}."
        )
    if njmax is not None:
        logger.debug("njmax=%s was applied at environment construction", njmax)

    def body_exists(name: str) -> bool:
        return mujoco.mj_name2id(env.host_model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

    object_handlers = {
        name: MjWarpObjectHandler(name=name, state=env.state, body_name=name)
        for name in sorted(_collect_object_names(config, body_exists))
    }
    return MjWarpObjectOnlyBackend(
        config,
        env,
        object_handlers,
        randomization=ResolvedRandomizationConfig.from_scope_config(
            config.randomization
        ),
    )
