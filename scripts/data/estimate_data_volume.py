#!/usr/bin/env python3
"""Estimate how many points a real-robot data collection covers and how long it takes.

Every dimension of the collection space (a position axis, an orientation
angle, a grasp parameter, ...) is discretized at a resolution, and the number
of distinct points is the product over all dimensions. The YAML config nests
dimensions in groups so their subtotals are reported too:

    rate_hz: 30          # points recorded per second by one station
    stations: 1          # robots collecting in parallel (default 1)
    hours_per_day: 24    # collection hours per calendar day (default 24)
    dimensions:
      position:
        x: {range: [-0.5, 0.5], resolution: 0.001}
        ...
      orientation:
        roll: {range: 180, resolution: 1}
        ...
      gripper: {count: 2}

A dimension gives ``range`` as a span or as ``[low, high]`` bounds, plus a
``resolution`` in the same unit, and counts ``ceil(span / resolution)`` points
(no extra point for the far end); a discrete dimension gives ``count``
directly. Any other mapping is a group whose count is the product of its
members. Spans and resolutions are read as exact decimals, so ``1 / 0.001`` is
1000 rather than 999.999...

    python scripts/data/estimate_data_volume.py
    python scripts/data/estimate_data_volume.py path/to/config.yaml

Without an argument the config next to this script, ``data_volume.yaml``, is
used.
"""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).with_name("data_volume.yaml")

CONFIG_KEYS = frozenset({"rate_hz", "stations", "hours_per_day", "dimensions"})
DIMENSION_KEYS = frozenset({"range", "resolution", "count"})

# (power of ten, name), largest first.
SHORT_SCALE = (
    (24, "Sp"),
    (21, "S"),
    (18, "Qi"),
    (15, "Q"),
    (12, "T"),
    (9, "B"),
    (6, "M"),
    (3, "K"),
)
CHINESE_SCALE = (
    (28, "穰"),
    (24, "秭"),
    (20, "垓"),
    (16, "京"),
    (12, "兆"),
    (8, "亿"),
    (4, "万"),
)
DAYS_PER_YEAR = Fraction("365.25")


@dataclass(frozen=True)
class Dimension:
    """A dimension, or a group of them whose ``count`` is their product."""

    name: str
    count: int
    members: tuple[Dimension, ...] = ()


@dataclass(frozen=True)
class Estimate:
    dimensions: Dimension
    rate_hz: Fraction
    stations: int
    hours_per_day: Fraction

    @property
    def points(self) -> int:
        return self.dimensions.count

    @property
    def seconds(self) -> Fraction:
        """Time spent collecting, with every station recording at ``rate_hz``."""
        return self.points / (self.rate_hz * self.stations)

    @property
    def days(self) -> Fraction:
        """Calendar days when collecting ``hours_per_day`` hours a day."""
        return self.seconds / (3600 * self.hours_per_day)


def _number(value: Any, path: str) -> Fraction:
    # str() first so a float reads as the decimal written in the YAML; PyYAML
    # also leaves exponents without a dot (``1e-3``) as strings.
    if not isinstance(value, (int, float, str)) or isinstance(value, bool):
        raise ValueError(f"{path}: expected a number, got {value!r}")
    try:
        return Fraction(str(value))
    except ValueError:
        raise ValueError(f"{path}: expected a number, got {value!r}") from None


def _positive(value: Any, path: str) -> Fraction:
    number = _number(value, path)
    if number <= 0:
        raise ValueError(f"{path}: must be positive, got {value!r}")
    return number


def _positive_int(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{path}: must be a positive integer, got {value!r}")
    return value


def _span(value: Any, path: str) -> Fraction:
    if not isinstance(value, list):
        return _positive(value, path)
    if len(value) != 2:
        raise ValueError(f"{path}: bounds must be [low, high], got {value!r}")
    low, high = (_number(bound, path) for bound in value)
    if high <= low:
        raise ValueError(f"{path}: high must exceed low, got {value!r}")
    return high - low


def _dimension_count(spec: dict, path: str) -> int:
    keys = set(spec)
    if keys == {"count"}:
        return _positive_int(spec["count"], f"{path}.count")
    if keys != {"range", "resolution"}:
        raise ValueError(
            f"{path}: a dimension takes either count or range + resolution, "
            f"got {sorted(keys)}"
        )
    span = _span(spec["range"], f"{path}.range")
    ratio = span / _positive(spec["resolution"], f"{path}.resolution")
    count = math.ceil(ratio)
    if count != ratio:
        warnings.warn(
            f"{path}: range / resolution = {float(ratio):g} is not an integer, "
            f"counting {count} points",
            stacklevel=2,
        )
    return count


def _parse(name: str, spec: Any, path: str) -> Dimension:
    if not isinstance(spec, dict) or not spec:
        raise ValueError(f"{path}: expected a non-empty mapping, got {spec!r}")
    if DIMENSION_KEYS & set(spec):
        return Dimension(name, _dimension_count(spec, path))
    members = tuple(
        _parse(str(key), value, f"{path}.{key}") for key, value in spec.items()
    )
    return Dimension(name, math.prod(member.count for member in members), members)


def estimate(config: Any) -> Estimate:
    """Count the points a parsed YAML config spans and time collecting them."""
    if not isinstance(config, dict):
        raise ValueError(f"config: expected a mapping, got {config!r}")
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"config: unknown keys {sorted(unknown)}")
    for key in ("rate_hz", "dimensions"):
        if key not in config:
            raise ValueError(f"config: missing {key}")
    hours_per_day = _positive(config.get("hours_per_day", 24), "hours_per_day")
    if hours_per_day > 24:
        raise ValueError(f"hours_per_day: at most 24, got {config['hours_per_day']!r}")
    return Estimate(
        dimensions=_parse("total", config["dimensions"], "dimensions"),
        rate_hz=_positive(config["rate_hz"], "rate_hz"),
        stations=_positive_int(config.get("stations", 1), "stations"),
        hours_per_day=hours_per_day,
    )


def format_scientific(value: Fraction | int) -> str:
    """Integers below a million as they are, anything else as ``m.mmme+XX``."""
    if value == int(value) and abs(value) < 10**6:
        return str(int(value))
    mantissa, exponent = f"{float(value):.6e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{exponent}"


def format_scaled(value: Fraction | int, scale: tuple[tuple[int, str], ...]) -> str:
    """Write ``value`` in the largest unit of ``scale`` it reaches, e.g. 4.5927 S."""
    for power, unit in scale:
        if value >= 10**power:
            return f"{float(Fraction(value, 10**power)):.7g} {unit}"
    return f"{float(value):.7g}"


def _plain(value: Fraction) -> str:
    return str(int(value)) if value == int(value) else f"{float(value):g}"


def _width(text: str) -> int:
    """Terminal columns ``text`` takes; CJK characters take two."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def report(result: Estimate) -> str:
    """Render the per-group point counts and the collection time as a table."""

    def row(label: str, value: Fraction | int) -> tuple[str, ...]:
        return (
            label,
            format_scientific(value),
            format_scaled(value, SHORT_SCALE),
            format_scaled(value, CHINESE_SCALE),
        )

    def walk(group: Dimension, depth: int) -> None:
        for member in group.members:
            rows.append(row("  " * depth + member.name, member.count))
            walk(member, depth + 1)

    rows: list[tuple[str, ...] | str] = [("points", "scientific", "short", "Chinese")]
    walk(result.dimensions, 0)
    rows.append(row("total", result.points))
    rows.append("")
    rows.append(
        f"time at {_plain(result.rate_hz)} Hz x {result.stations} station(s), "
        f"{_plain(result.hours_per_day)} h/day"
    )
    rows.append(row("seconds", result.seconds))
    rows.append(row("hours", result.seconds / 3600))
    rows.append(row("days", result.days))
    rows.append(row("years", result.days / DAYS_PER_YEAR))

    table = [cells for cells in rows if isinstance(cells, tuple)]
    widths = [max(_width(cells[col]) for cells in table) for col in range(4)]
    lines = []
    for cells in rows:
        if isinstance(cells, str):
            lines.append(cells)
            continue
        if cells[0] == "total":
            lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
        padded = [cells[0] + " " * (widths[0] - _width(cells[0]))]
        padded += [
            " " * (w - _width(c)) + c
            for c, w in zip(cells[1:], widths[1:], strict=True)
        ]
        lines.append("  ".join(padded))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML config (default: {DEFAULT_CONFIG.name} next to this script)",
    )
    args = parser.parse_args(argv)
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = estimate(config)
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)
    print(report(result))


if __name__ == "__main__":
    main()
