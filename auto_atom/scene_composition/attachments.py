"""Materialize grasp-attachment welds into a parsed scene spec.

A grasp attachment holds a grasped object rigidly in an operator's gripper: a
body weld between the gripper and the object that the scene carries inactive
and a backend activates, at the object's current pose relative to the gripper,
once it has verified the grasp.  Equality constraints cannot be added to a
compiled model, so every weld a task may need is created on the editable
:class:`mujoco.MjSpec` before compilation, as config-declared cameras are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import mujoco

__all__ = ["GraspWeldElementSpec", "create_grasp_welds", "grasp_weld_name"]


def grasp_weld_name(operator: str, object_name: str) -> str:
    """Equality name of the weld holding ``object_name`` in ``operator``."""
    return f"aao_grasp_attach__{operator}__{object_name}"


@dataclass(frozen=True)
class GraspWeldElementSpec:
    """One inactive gripper-to-object weld the scene must carry."""

    operator: str
    """Logical operator holding the object."""
    object: str
    """Logical object name; its ``<name>_gs`` body if there is one, else its
    body, is welded, as backends resolve object bodies."""
    frame: str
    """Site or body of the gripper; a site resolves to the body it sits on."""
    solref: tuple[float, float]
    """Constraint ``solref`` of the weld."""
    solimp: tuple[float, float, float, float, float]
    """Constraint ``solimp`` of the weld."""

    @property
    def name(self) -> str:
        return grasp_weld_name(self.operator, self.object)


def create_grasp_welds(
    spec: mujoco.MjSpec,
    welds: Sequence[GraspWeldElementSpec],
) -> None:
    """Add every weld in ``welds`` to ``spec``, inactive.

    Raises ``ValueError`` when a gripper frame or object body is missing, so a
    misconfigured attachment fails while the scene is built rather than at the
    first grasp.
    """
    for weld in welds:
        gripper = _frame_body(spec, weld.frame)
        if gripper is None:
            raise ValueError(
                f"Grasp attachment for operator '{weld.operator}': gripper frame "
                f"'{weld.frame}' is neither a site nor a body of the scene."
            )
        target = _object_body(spec, weld.object)
        if target is None:
            raise ValueError(
                f"Grasp attachment for operator '{weld.operator}': object "
                f"'{weld.object}' has no body '{weld.object}_gs' or "
                f"'{weld.object}' in the scene."
            )
        spec.add_equality(
            name=weld.name,
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=gripper,
            name2=target,
            active=False,
            solref=list(weld.solref),
            solimp=list(weld.solimp),
        )


def _frame_body(spec: mujoco.MjSpec, frame: str) -> Optional[str]:
    site = _find(spec.site, frame)
    if site is not None:
        return site.parent.name
    body = _find(spec.body, frame)
    return None if body is None else body.name


def _object_body(spec: mujoco.MjSpec, name: str) -> Optional[str]:
    for candidate in (f"{name}_gs", name):
        body = _find(spec.body, candidate)
        if body is not None:
            return body.name
    return None


def _find(lookup: Any, name: str) -> Any:
    """``spec.site`` / ``spec.body`` lookup that tolerates a missing name."""
    try:
        return lookup(name)
    except (KeyError, ValueError):
        return None
