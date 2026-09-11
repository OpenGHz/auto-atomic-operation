"""Materialize config-declared MJCF cameras into a parsed scene spec.

Cameras used to be authorable only in the MJCF: the environment resolved every
``env.cameras[].name`` against the compiled model and failed when the scene did
not define that name.  A camera a task config already describes completely — the
frame it is mounted on, its mount pose, its optics — does not need an MJCF
element authored next to that frame, so this module creates the missing ones on
the editable :class:`mujoco.MjSpec` before the model is compiled.

Creation happens on the spec rather than by rewriting XML text because the mount
frame can live inside an attached sub-model (``<attach model="umi_gripper_v3"/>``
expands into the spec, not into the scene XML), and because a camera the config
declares is a config concern: the scene XML keeps describing only the scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

_IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CameraElementSpec:
    """One camera the composed scene may have to create.

    The record is deliberately backend-neutral about where it comes from: the
    config layer maps its camera model onto it, and this module turns it into an
    MJCF camera element.  ``parent_frame`` names the site or body the camera is
    mounted on (empty keeps the world frame) and ``position`` / ``orientation``
    are the mount pose expressed in that frame; ``orientation`` follows the YAML
    ``[x, y, z, w]`` convention so no caller has to reorder quaternions.
    """

    name: str
    """Camera name; the key the environment resolves its configuration under."""
    parent_frame: str = ""
    """Site or body whose frame the mount pose is expressed in."""
    position: tuple[float, float, float] | None = None
    """Mount position ``[x, y, z]`` in metres relative to ``parent_frame``."""
    orientation: tuple[float, float, float, float] | None = None
    """Mount orientation ``[x, y, z, w]`` relative to ``parent_frame``."""
    fovy_deg: float | None = None
    """Vertical field of view in degrees; ``None`` keeps MuJoCo's default."""
    object_mount: bool = False
    """Also accept the ``<name>_gs`` body of a logical object as the mount.

    Gaussian-splatting scenes name an object's body ``cup_gs`` while the rest of
    the stack keeps calling the object ``cup``; object-mounted cameras resolve
    the same alias ``MujocoBasis`` applies.
    """


def create_camera_elements(
    spec: mujoco.MjSpec,
    elements: Sequence[CameraElementSpec],
) -> tuple[str, ...]:
    """Create every declared camera the scene does not already define.

    A camera the scene defines keeps its authored element — the declaration then
    contributes only the config-side aspects (channels, calibration, noise,
    randomization), which is the pre-existing behaviour.  A missing one is
    created on the body its ``parent_frame`` resolves to, so MuJoCo derives its
    world pose from that body on every forward pass.

    Returns the created names so the caller can report them.  Raises when a
    camera has to be created without a complete mount pose, or when its
    ``parent_frame`` does not resolve: both are configuration bugs that would
    otherwise surface as a camera stuck at the frame origin.
    """
    defined = {camera.name for camera in spec.cameras}
    created: list[str] = []
    for element in elements:
        if element.name in defined:
            continue
        if element.position is None or element.orientation is None:
            raise ValueError(
                f"Camera '{element.name}' is not defined in the composed scene, so "
                "it has to declare its mount: set calibration.extrinsics.position "
                "and calibration.extrinsics.orientation, or author the camera "
                "element in the MJCF."
            )
        mount = _resolve_mount(spec, element)
        body, frame_position, frame_quaternion = mount
        position, quaternion = _mount_pose(
            frame_position,
            frame_quaternion,
            element,
        )
        body.add_camera(
            name=element.name,
            pos=position.tolist(),
            quat=quaternion.tolist(),
            **({} if element.fovy_deg is None else {"fovy": float(element.fovy_deg)}),
        )
        defined.add(element.name)
        created.append(element.name)
        logger.info(
            "Created declared camera '%s' on body '%s'",
            element.name,
            body.name,
        )
    return tuple(created)


def _resolve_mount(
    spec: mujoco.MjSpec,
    element: CameraElementSpec,
) -> tuple[mujoco.MjsBody, np.ndarray, np.ndarray]:
    """Resolve ``parent_frame`` to a body plus the frame's pose inside it.

    Sites resolve to their owning body, so the mount pose composes with the
    site's own local pose.  The lookup order mirrors ``MujocoBasis``: site, then
    body, then — for object-mounted cameras — the object's ``_gs`` body.
    """
    if not element.parent_frame:
        return (
            spec.worldbody,
            np.zeros(3, dtype=np.float64),
            np.array(_IDENTITY_QUAT, dtype=np.float64),
        )
    site = spec.site(element.parent_frame)
    if site is not None:
        return (
            site.parent,
            np.asarray(site.pos, dtype=np.float64),
            _normalize_quaternion(np.asarray(site.quat, dtype=np.float64)),
        )
    body = spec.body(element.parent_frame)
    if body is None and element.object_mount:
        body = spec.body(f"{element.parent_frame}_gs")
    if body is None:
        raise ValueError(
            f"Camera '{element.name}': parent_frame '{element.parent_frame}' is "
            "neither a site nor a body in the composed scene."
        )
    return (
        body,
        np.zeros(3, dtype=np.float64),
        np.array(_IDENTITY_QUAT, dtype=np.float64),
    )


def _mount_pose(
    frame_position: np.ndarray,
    frame_quaternion: np.ndarray,
    element: CameraElementSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the declared mount pose with the mount frame's local pose."""

    assert element.position is not None
    assert element.orientation is not None
    position = np.asarray(element.position, dtype=np.float64)
    orientation_xyzw = np.asarray(element.orientation, dtype=np.float64)
    camera_quaternion = _normalize_quaternion(
        np.array(
            [
                orientation_xyzw[3],
                orientation_xyzw[0],
                orientation_xyzw[1],
                orientation_xyzw[2],
            ]
        )
    )
    rotated = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(rotated, position, frame_quaternion)
    body_quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(body_quaternion, frame_quaternion, camera_quaternion)
    return frame_position + rotated, body_quaternion


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("camera mount orientation must be a non-zero quaternion")
    return quaternion / norm


__all__ = ["CameraElementSpec", "create_camera_elements"]
