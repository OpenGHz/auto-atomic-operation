"""Generic cold-path geometry transforms for compiled asset contributions.

The scene composer never applies these transforms in a simulator control tick.
An adapter supplies component metadata and a declarative layer recipe selects
the MJCF bodies/meshes to transform.  The same machinery can therefore serve
doors, handles, tools, fixtures, or any other asset family without adding a
vendor branch to the composer.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any

from .config import AssetAnchorConfig, AssetScaleRuleConfig

Bounds3 = tuple[tuple[float, float, float], tuple[float, float, float]]


def apply_asset_normalization(
    fragment: ET.Element,
    scaling: Sequence[AssetScaleRuleConfig],
    anchors: Sequence[AssetAnchorConfig],
    metadata: Mapping[str, Any],
) -> tuple[str, ...]:
    """Apply configured scales and anchor projections to an MJCF fragment.

    ``metadata`` is adapter-owned component data.  Rules refer to it through
    dotted paths, while body and mesh names remain the only XML-specific part
    of the declarative recipe.  Source metadata is never modified.
    """

    scale_by_source: dict[str, float] = {}
    diagnostics: list[str] = []
    all_body_targets = {body for rule in scaling for body in rule.bodies}
    for rule in scaling:
        bounds = resolve_bounds(metadata, rule.source_bounds)
        axis = "xyz".index(rule.axis)
        source_extent = bounds[1][axis] - bounds[0][axis]
        if source_extent <= 0.0 or not math.isfinite(source_extent):
            raise ValueError(
                f"asset scale source extent must be positive: {rule.source_bounds}"
            )
        factor = float(rule.target_extent_m) / source_extent
        previous = scale_by_source.get(rule.source_bounds)
        if previous is not None and not math.isclose(
            previous, factor, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"conflicting scale factors for source bounds: {rule.source_bounds}"
            )
        scale_by_source[rule.source_bounds] = factor
        matched_meshes = _scale_meshes(fragment, rule, factor)
        if rule.required:
            missing_meshes = set(rule.meshes) - matched_meshes
            missing_prefixes = tuple(
                prefix
                for prefix in rule.mesh_prefixes
                if not any(name.startswith(prefix) for name in matched_meshes)
            )
            if missing_meshes or missing_prefixes:
                raise ValueError(
                    "asset scale mesh targets not found: "
                    f"names={sorted(missing_meshes)}, prefixes={missing_prefixes}"
                )
        for body_name in rule.bodies:
            body = _find_named(fragment, "body", body_name)
            if body is None:
                if rule.required:
                    raise ValueError(f"asset scale body not found: {body_name}")
                continue
            _scale_body_owned_geometry(
                body,
                factor,
                all_body_targets,
                set(rule.preserve_bodies),
            )
        diagnostics.append(
            f"scaled {rule.source_bounds} to {float(rule.target_extent_m):.9g}m "
            f"(factor={factor:.9g})"
        )

    for anchor in anchors:
        transformed_bounds = None
        if anchor.source_bounds is not None:
            source_bounds = resolve_bounds(metadata, anchor.source_bounds)
            transformed_bounds = _transform_bounds(
                source_bounds, anchor.source_bounds, scale_by_source
            )
        position: dict[str, float] = {}
        for axis_name, coordinate in anchor.coordinates.items():
            axis = "xyz".index(axis_name)
            if coordinate.value_m is not None:
                value = float(coordinate.value_m)
            else:
                if transformed_bounds is None:
                    raise ValueError(
                        f"asset anchor source bounds missing for projected coordinate: {axis_name}"
                    )
                edge = 0 if coordinate.edge == "min" else 1
                value = transformed_bounds[edge][axis] * float(
                    coordinate.multiplier
                ) + float(coordinate.offset_m)
            if not math.isfinite(value):
                raise ValueError(
                    f"resolved asset anchor coordinate is not finite: {axis_name}"
                )
            position[axis_name] = value
        for body_name in anchor.bodies:
            body = _find_named(fragment, "body", body_name)
            if body is None:
                if anchor.required:
                    raise ValueError(f"asset anchor body not found: {body_name}")
                continue
            current = _parse_vector(
                body.get("pos", "0 0 0"), 3, f"body {body_name}.pos"
            )
            updated = tuple(
                position.get("xyz"[index], current[index]) for index in range(3)
            )
            body.set("pos", _format_vector(updated))
        diagnostics.append(f"positioned anchor bodies: {', '.join(anchor.bodies)}")
    return tuple(diagnostics)


def resolve_bounds(metadata: Mapping[str, Any], path: str) -> Bounds3:
    """Resolve a dotted metadata path containing lower/upper 3-vectors."""

    value: Any = metadata
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"asset normalization metadata path not found: {path}")
        value = value[part]
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(
            f"asset normalization bounds must contain lower and upper vectors: {path}"
        )
    lower = _parse_vector(value[0], 3, f"{path}[0]")
    upper = _parse_vector(value[1], 3, f"{path}[1]")
    if any(upper[index] <= lower[index] for index in range(3)):
        raise ValueError(
            f"asset normalization bounds must have positive extents: {path}"
        )
    return lower, upper


def _transform_bounds(
    bounds: Bounds3, source_path: str, scale_by_source: Mapping[str, float]
) -> Bounds3:
    factor = scale_by_source.get(source_path, 1.0)
    return (
        tuple(value * factor for value in bounds[0]),
        tuple(value * factor for value in bounds[1]),
    )  # type: ignore[return-value]


def _scale_meshes(
    fragment: ET.Element, rule: AssetScaleRuleConfig, factor: float
) -> set[str]:
    selected = set(rule.meshes)
    prefixes = tuple(rule.mesh_prefixes)
    matched: set[str] = set()
    for mesh in fragment.iter("mesh"):
        name = mesh.get("name", "")
        if name not in selected and not any(
            name.startswith(prefix) for prefix in prefixes
        ):
            continue
        current = _parse_vector(mesh.get("scale", "1 1 1"), 3, f"mesh {name}.scale")
        mesh.set("scale", _format_vector(tuple(value * factor for value in current)))
        matched.add(name)
    return matched


def _scale_body_owned_geometry(
    body: ET.Element,
    factor: float,
    all_body_targets: set[str],
    preserved_bodies: set[str],
) -> None:
    """Scale a body's local geometry while leaving placement roots untouched."""

    for element in list(body):
        if element.tag == "body":
            if element.get("name") in preserved_bodies:
                continue
            if element.get("name") not in all_body_targets:
                _scale_body_owned_geometry(
                    element, factor, all_body_targets, preserved_bodies
                )
                _scale_attribute(element, "pos", factor, (3,))
            continue
        if element.tag in {"geom", "site", "inertial", "camera", "light", "joint"}:
            _scale_attribute(element, "pos", factor, (3,))
            _scale_attribute(element, "size", factor, (1, 2, 3))
            _scale_attribute(element, "fromto", factor, (6,))
        if element.tag == "inertial":
            _scale_attribute(element, "diaginertia", factor * factor, (3,))
            _scale_attribute(element, "fullinertia", factor * factor, (6,))


def _scale_attribute(
    element: ET.Element, name: str, factor: float, sizes: tuple[int, ...]
) -> None:
    value = element.get(name)
    if value is None:
        return
    vector = _parse_vector_sizes(value, sizes, f"{element.tag}.{name}")
    element.set(name, _format_vector(tuple(item * factor for item in vector)))


def _find_named(root: ET.Element, tag: str, name: str) -> ET.Element | None:
    return next(
        (element for element in root.iter(tag) if element.get("name") == name), None
    )


def _parse_vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    return _parse_vector_sizes(value, (size,), label)


def _parse_vector_sizes(
    value: Any, sizes: tuple[int, ...], label: str
) -> tuple[float, ...]:
    if isinstance(value, str):
        values = value.split()
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = list(value)
    else:
        raise ValueError(f"{label} must contain {sizes} numeric values")
    if len(values) not in sizes:
        raise ValueError(f"{label} must contain {sizes} numeric values")
    result = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain finite values")
    return result


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(
        "0" if float(value) == 0 else format(float(value), ".15g") for value in values
    )


__all__ = ["apply_asset_normalization", "resolve_bounds"]
