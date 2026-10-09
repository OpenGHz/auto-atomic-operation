"""GPU batch rendering for the MJWarp environment.

MJWarp renders every camera across every world in one ray-traced pass, which is
a different shape from the native path's per-replica ``mujoco.Renderer``. Three
differences from MuJoCo's offscreen renderer are handled here, each measured
rather than assumed:

* **Depth comes back normalised, never metric.** ``get_depth`` computes
  ``clamp(value / depth_scale, 0, 1)``, so passing ``1.0`` silently clamps every
  distance beyond a metre. Metric depth requires passing the far plane as the
  scale and multiplying back; doing that drops the disagreement with native
  depth from 3.16 m to 2.3 mm on a 100 m range.
* **A ray that hits nothing reports 0.0, not the far plane.** Native MuJoCo
  reports the far plane there. Left alone, empty background would read as
  "zero distance", i.e. closer than anything real.
* **A ``RenderContext`` fixes one ``znear`` at creation and has no ``zfar``.**
  The native path switches clip range per output stream, so cameras are grouped
  by requested range and one context is built per distinct group.
* **The ray tracer draws every geom of the rendered groups, invisible ones
  included.** MuJoCo leaves a geom whose colour has zero alpha out of the scene;
  MJWarp drew such geoms as opaque black, in colour, depth and segmentation
  alike. Scenes use them for collision proxies (``rack_plate`` has 11), so they
  occluded what native shows. Contexts are built from a render-only model copy
  that keeps exactly the geoms MuJoCo's own scene contains
  (:func:`natively_drawn_geom_ids`).

Image orientation and segmentation encoding need no conversion: measured
centroids agree with the native renderer to four decimals, and segmentation is
already MuJoCo's ``(object_id, object_type)`` with ``(-1, -1)`` for background.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import mujoco

    from auto_atom.basis.mjwarp.state import MjWarpSceneState

# Geom groups both renderers draw: MuJoCo's default ``MjvOption.geomgroup``,
# which the native env renders with, and MJWarp's default
# ``enabled_geom_groups``.
RENDERED_GEOM_GROUPS: Tuple[int, ...] = (0, 1, 2)
# A group outside RENDERED_GEOM_GROUPS, where the render model parks the geoms
# native does not draw.
_UNRENDERED_GEOM_GROUP = 5


def natively_drawn_geom_ids(model: "mujoco.MjModel") -> frozenset:
    """Geoms MuJoCo's own renderer draws with the native env's visual options.

    Asked of ``mjv_updateScene`` rather than re-derived, so every rule MuJoCo
    applies (group switches, zero alpha after resolving the material) carries
    over. Visibility is read once, from the model as compiled: a geom whose
    alpha is changed per world later keeps the visibility it had.
    """
    import mujoco

    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=max(1000, 2 * model.ngeom))
    mujoco.mjv_updateScene(
        model,
        data,
        mujoco.MjvOption(),
        None,
        mujoco.MjvCamera(),
        int(mujoco.mjtCatBit.mjCAT_ALL),
        scene,
    )
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    return frozenset(
        int(geom.objid)
        for geom in scene.geoms[: scene.ngeom]
        if int(geom.objtype) == geom_type
    )


def render_model(model: "mujoco.MjModel") -> "mujoco.MjModel":
    """A copy of ``model`` whose rendered groups hold only natively drawn geoms."""
    import copy

    drawn = natively_drawn_geom_ids(model)
    hidden = [
        geom
        for geom in range(model.ngeom)
        if geom not in drawn and int(model.geom_group[geom]) in RENDERED_GEOM_GROUPS
    ]
    if not hidden:
        return model
    copied = copy.copy(model)
    copied.geom_group[hidden] = _UNRENDERED_GEOM_GROUP
    return copied


@dataclass(frozen=True)
class CameraRenderRequest:
    """What one camera needs rendered, and with which clip range."""

    name: str
    width: int
    height: int
    want_color: bool
    want_segmentation: bool
    want_depth: bool
    rgb_clip_range_m: Optional[Tuple[float, float]]
    depth_clip_range_m: Optional[Tuple[float, float]]

    @property
    def needs_rgb_pass(self) -> bool:
        return self.want_color or self.want_segmentation


@dataclass
class _ContextGroup:
    """One ``RenderContext`` plus the cameras and streams it serves."""

    clip_range_m: Tuple[float, float]
    camera_names: List[str] = field(default_factory=list)
    render_rgb: List[bool] = field(default_factory=list)
    render_depth: List[bool] = field(default_factory=list)
    render_seg: List[bool] = field(default_factory=list)
    resolutions: List[Tuple[int, int]] = field(default_factory=list)
    context: Any = None


class MjWarpBatchRenderer:
    """Renders every requested camera stream for every world.

    Contexts are built lazily on the first render, because context creation
    compiles kernels and a scene may be constructed without ever capturing an
    observation -- notably the ``object_only`` execution path, which never
    renders unless a caller explicitly asks for one.
    """

    def __init__(
        self,
        state: "MjWarpSceneState",
        requests: Sequence[CameraRenderRequest],
    ) -> None:
        self.state = state
        self.requests = {request.name: request for request in requests}
        self._groups: Optional[List[_ContextGroup]] = None

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _resolve_clip_range(
        self,
        requested: Optional[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """A camera's effective metric clip range, falling back to the model."""
        if requested is None:
            return self.state.default_clip_range_m()
        return (float(requested[0]), float(requested[1]))

    def _build_groups(self) -> List[_ContextGroup]:
        """Group cameras by clip range, one ``RenderContext`` per distinct range.

        A camera whose RGB and depth ranges differ appears in two groups, with
        only the relevant streams enabled in each -- that is the only way to give
        each stream its own near plane when a context fixes ``znear`` once.
        """
        grouped: Dict[Tuple[float, float], _ContextGroup] = {}

        def slot(
            group: _ContextGroup,
            name: str,
            request: CameraRenderRequest,
        ) -> int:
            if name in group.camera_names:
                return group.camera_names.index(name)
            group.camera_names.append(name)
            group.render_rgb.append(False)
            group.render_depth.append(False)
            group.render_seg.append(False)
            group.resolutions.append((request.width, request.height))
            return len(group.camera_names) - 1

        for name, request in self.requests.items():
            if request.needs_rgb_pass:
                clip = self._resolve_clip_range(request.rgb_clip_range_m)
                group = grouped.setdefault(clip, _ContextGroup(clip_range_m=clip))
                index = slot(group, name, request)
                group.render_rgb[index] = request.want_color
                group.render_seg[index] = request.want_segmentation
            if request.want_depth:
                clip = self._resolve_clip_range(request.depth_clip_range_m)
                group = grouped.setdefault(clip, _ContextGroup(clip_range_m=clip))
                index = slot(group, name, request)
                group.render_depth[index] = True

        import mujoco_warp as mjw

        model = render_model(self.state.host_model)
        for group in grouped.values():
            group.context = mjw.create_render_context(
                model,
                nworld=self.state.nworld,
                cam_res=list(group.resolutions),
                render_rgb=list(group.render_rgb),
                render_depth=list(group.render_depth),
                render_seg=list(group.render_seg),
                # cam_active accepts camera names directly, so no id lookup is
                # needed and the group's own ordering becomes the stream index.
                cam_active=list(group.camera_names),
                enabled_geom_groups=list(RENDERED_GEOM_GROUPS),
            )
        return list(grouped.values())

    @property
    def groups(self) -> List[_ContextGroup]:
        if self._groups is None:
            self._groups = self._build_groups()
        return self._groups

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Render every requested stream, keyed by camera then stream.

        Returns ``{camera: {"color"|"depth"|"segmentation": array}}`` with a
        leading world axis on every array. Colour is ``uint8`` and depth is
        metres, matching what the native renderer hands back.
        """
        import mujoco_warp as mjw
        import warp as wp

        results: Dict[str, Dict[str, np.ndarray]] = {name: {} for name in self.requests}
        nworld = self.state.nworld

        for group in self.groups:
            mjw.render(self.state.model, self.state.data, group.context)
            near_m, far_m = group.clip_range_m

            for index, name in enumerate(group.camera_names):
                request = self.requests[name]
                width, height = group.resolutions[index]

                if group.render_rgb[index]:
                    buffer = wp.zeros((nworld, height, width), dtype=wp.vec3)
                    mjw.get_rgb(group.context, index, buffer)
                    # MJWarp returns linear float RGB in [0, 1]; the native
                    # renderer hands back uint8, and downstream consumers
                    # (encoders, mask overlays) assume that.
                    results[name]["color"] = np.clip(
                        buffer.numpy() * 255.0, 0.0, 255.0
                    ).astype(np.uint8)

                if group.render_seg[index]:
                    buffer = wp.zeros((nworld, height, width), dtype=wp.vec2i)
                    mjw.get_segmentation(group.context, index, buffer)
                    results[name]["segmentation"] = np.asarray(
                        buffer.numpy(), dtype=np.int32
                    )

                if group.render_depth[index]:
                    buffer = wp.zeros((nworld, height, width), dtype=wp.float32)
                    # get_depth computes clamp(value / scale, 0, 1), so the far
                    # plane is the scale that keeps the full range representable
                    # and multiplying back recovers metres.
                    mjw.get_depth(group.context, index, far_m, buffer)
                    depth = buffer.numpy() * far_m
                    # A ray that hits nothing comes back as 0.0, which would read
                    # as "closer than everything". Native reports the far plane.
                    depth = np.where(depth <= 0.0, far_m, depth)
                    results[name]["depth"] = np.clip(depth, near_m, far_m)

        return results
