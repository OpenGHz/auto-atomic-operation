"""Clearance of a gripper and its held object placed at hypothetical poses.

The check never steps or mutates the live simulation. Geoms that move with
the end effector are captured as rigid templates relative to the EEF frame;
a candidate EEF pose places them by writing world poses into a scratch
``MjData`` copy, where ``mj_geomDistance`` measures them against obstacle
geoms. ``mj_geomDistance`` reads ``geom_xpos`` / ``geom_xmat`` from the data
it is given, so no kinematics pass is needed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import mujoco
import numpy as np

from ...contracts import ClearanceReport


@dataclass(frozen=True)
class GeomTemplate:
    """Geoms held rigidly in one frame, as poses relative to that frame."""

    geom_ids: np.ndarray
    """Geom ids, shape ``(k,)``."""
    positions: np.ndarray
    """Geom positions in the frame, shape ``(k, 3)``."""
    rotations: np.ndarray
    """Geom orientations in the frame, shape ``(k, 3, 3)``."""


def subtree_bodies(model: mujoco.MjModel, root: int) -> set[int]:
    """Body ids of ``root`` and every body below it."""
    bodies = {int(root)}
    for body in range(int(root) + 1, model.nbody):
        if int(model.body_parentid[body]) in bodies:
            bodies.add(body)
    return bodies


def probe_geoms(model: mujoco.MjModel, bodies: Iterable[int]) -> np.ndarray:
    """Colliding geoms and render meshes of ``bodies``.

    Render meshes are included so a candidate cannot clip visibly; small
    non-colliding primitives such as tactile cells are not.
    """
    body_set = set(bodies)
    return np.asarray(
        [
            geom
            for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) in body_set
            and (
                int(model.geom_contype[geom]) != 0
                or int(model.geom_conaffinity[geom]) != 0
                or int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_MESH)
            )
        ],
        dtype=np.int64,
    )


def obstacle_geoms(model: mujoco.MjModel, bodies: Iterable[int]) -> np.ndarray:
    """Colliding geoms of ``bodies``."""
    body_set = set(bodies)
    return np.asarray(
        [
            geom
            for geom in range(model.ngeom)
            if int(model.geom_bodyid[geom]) in body_set
            and (
                int(model.geom_contype[geom]) != 0
                or int(model.geom_conaffinity[geom]) != 0
            )
        ],
        dtype=np.int64,
    )


def capture_template(
    data: mujoco.MjData,
    geom_ids: np.ndarray,
    frame_position: np.ndarray,
    frame_rotation: np.ndarray,
) -> GeomTemplate:
    """Record the current poses of ``geom_ids`` relative to a world frame."""
    inverse = np.asarray(frame_rotation, dtype=np.float64).T
    positions = (
        np.asarray(data.geom_xpos[geom_ids], dtype=np.float64)
        - np.asarray(frame_position, dtype=np.float64)
    ) @ inverse.T
    rotations = inverse @ np.asarray(data.geom_xmat[geom_ids]).reshape(-1, 3, 3)
    return GeomTemplate(
        geom_ids=np.asarray(geom_ids, dtype=np.int64),
        positions=positions,
        rotations=rotations,
    )


def check_paths(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    held: Sequence[GeomTemplate],
    released: Sequence[Sequence[GeomTemplate]],
    eef_paths: Sequence[tuple[np.ndarray, np.ndarray]],
    open_paths: Optional[Sequence[tuple[np.ndarray, np.ndarray]]],
    obstacles: np.ndarray,
    margin: float,
    *,
    report_range: float = 0.02,
) -> ClearanceReport:
    """Return the first path whose samples all keep ``margin``.

    ``held`` templates (EEF-relative) are placed at every sample of a path,
    and the matching entry of ``released`` at every sample of the matching
    entry of ``open_paths``. Each path is ``(positions (n, 3), rotations (n, 3, 3))``
    of the EEF frame in world. Distances are resolved up to ``margin +
    report_range``. When no path is clear, the report describes what blocked
    the first one. Held paths end deepest in the scene, so the open path is
    measured first and the held path from its end backwards: blocked
    candidates fail fast.
    """
    scratch = copy.copy(data)
    cap = float(margin) + float(report_range)
    obstacle_positions = np.asarray(data.geom_xpos[obstacles], dtype=np.float64)
    obstacle_radii = np.asarray(model.geom_rbound[obstacles], dtype=np.float64)
    first_block: Optional[ClearanceReport] = None
    for path_index, (positions, rotations) in enumerate(eef_paths):
        samples = []
        if open_paths is not None:
            open_positions, open_rotations = open_paths[path_index]
            samples.extend(
                (
                    released[path_index],
                    open_positions[index],
                    open_rotations[index],
                    True,
                    index,
                )
                for index in range(len(open_positions))
            )
        samples.extend(
            (held, positions[index], rotations[index], False, index)
            for index in range(len(positions) - 1, -1, -1)
        )
        report = _measure(
            model,
            scratch,
            samples,
            obstacles,
            obstacle_positions,
            obstacle_radii,
            margin,
            cap,
        )
        if report.clear:
            return ClearanceReport(
                clear=True,
                distance=report.distance,
                path_index=path_index,
                sample_index=report.sample_index,
                released=report.released,
                probe_geom=report.probe_geom,
                obstacle_geom=report.obstacle_geom,
                checked_paths=path_index + 1,
            )
        if first_block is None:
            first_block = report
    if first_block is None:
        return ClearanceReport(clear=False, distance=cap, checked_paths=0)
    return ClearanceReport(
        clear=False,
        distance=first_block.distance,
        sample_index=first_block.sample_index,
        released=first_block.released,
        probe_geom=first_block.probe_geom,
        obstacle_geom=first_block.obstacle_geom,
        checked_paths=len(eef_paths),
    )


def _measure(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    samples: Sequence[tuple[Sequence[GeomTemplate], np.ndarray, np.ndarray, bool, int]],
    obstacles: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_radii: np.ndarray,
    margin: float,
    cap: float,
) -> ClearanceReport:
    """Closest approach over one path's samples, stopping below ``margin``."""
    best = ClearanceReport(clear=True, distance=cap)
    for templates, frame_position, frame_rotation, released, sample_index in samples:
        rotation = np.asarray(frame_rotation, dtype=np.float64)
        origin = np.asarray(frame_position, dtype=np.float64)
        for template in templates:
            positions = template.positions @ rotation.T + origin
            scratch.geom_xpos[template.geom_ids] = positions
            scratch.geom_xmat[template.geom_ids] = (
                rotation @ template.rotations
            ).reshape(-1, 9)
            reach = (
                np.asarray(model.geom_rbound[template.geom_ids])[:, None]
                + obstacle_radii[None, :]
                + cap
            )
            near = (
                np.linalg.norm(
                    positions[:, None, :] - obstacle_positions[None, :, :], axis=2
                )
                < reach
            )
            for probe_index, obstacle_index in zip(*np.nonzero(near), strict=True):
                probe = int(template.geom_ids[probe_index])
                obstacle = int(obstacles[obstacle_index])
                distance = float(
                    mujoco.mj_geomDistance(model, scratch, probe, obstacle, cap, None)
                )
                if distance < best.distance:
                    best = ClearanceReport(
                        clear=distance >= margin,
                        distance=distance,
                        sample_index=sample_index,
                        released=released,
                        probe_geom=_geom_name(model, probe),
                        obstacle_geom=_geom_name(model, obstacle),
                    )
                    if not best.clear:
                        return best
    return best


def _geom_name(model: mujoco.MjModel, geom: int) -> Optional[str]:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"
