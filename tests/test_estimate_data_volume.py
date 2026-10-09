"""Tests for the real-robot data volume estimator."""

from __future__ import annotations

from fractions import Fraction

import pytest
import yaml

from scripts.data.estimate_data_volume import (
    CHINESE_SCALE,
    DEFAULT_CONFIG,
    SHORT_SCALE,
    estimate,
    format_scaled,
    format_scientific,
    main,
)


def _config(dimensions: dict, **extra) -> dict:
    return {"rate_hz": 30, "dimensions": dimensions, **extra}


def test_default_config_reproduces_the_hand_estimate() -> None:
    result = estimate(yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8")))

    assert result.points == 4_592_700_000_000_000_000_000
    assert result.days == 1_771_875_000_000_000
    space, grasp = result.dimensions.members
    position, orientation = space.members
    assert position.count == 10**9
    assert orientation.count == 5_832_000
    assert grasp.count == 787_500


def test_bounds_and_span_count_the_same_points() -> None:
    result = estimate(
        _config(
            {
                "bounds": {"range": [-0.5, 0.5], "resolution": 0.001},
                "span": {"range": 1.0, "resolution": 0.001},
                "exponent": {"range": 1, "resolution": "1e-3"},
            }
        )
    )

    assert [member.count for member in result.dimensions.members] == [1000] * 3


def test_count_gives_a_discrete_dimension() -> None:
    result = estimate(
        _config({"gripper": {"count": 2}, "x": {"range": 3, "resolution": 1}})
    )

    assert result.points == 6


def test_partial_step_rounds_up_with_a_warning() -> None:
    with pytest.warns(UserWarning, match=r"dimensions\.x: .* counting 4 points"):
        result = estimate(_config({"x": {"range": 1, "resolution": 0.3}}))

    assert result.points == 4


def test_stations_and_hours_per_day_scale_calendar_time() -> None:
    dimensions = {"x": {"count": 30 * 3600 * 24}}

    assert estimate(_config(dimensions)).days == 1
    assert estimate(_config(dimensions, stations=4)).days == Fraction(1, 4)
    assert estimate(_config(dimensions, hours_per_day=8)).days == 3


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_config({"x": {"range": [1, 0], "resolution": 0.1}}), "high must exceed low"),
        (_config({"x": {"range": [0, 1, 2], "resolution": 0.1}}), r"\[low, high\]"),
        (_config({"x": {"range": 1}}), "either count or range"),
        (_config({"x": {"range": 1, "resolution": 1, "count": 2}}), "either count"),
        (_config({"x": {"range": 1, "resolution": 0}}), "must be positive"),
        (_config({"x": {"count": 1.5}}), "positive integer"),
        (_config({"x": {"range": "wide", "resolution": 1}}), "expected a number"),
        (_config({"x": {}}), "non-empty mapping"),
        (_config({"x": {"count": 1}}, stations=0), "stations"),
        (_config({"x": {"count": 1}}, hours_per_day=25), "at most 24"),
        (_config({"x": {"count": 1}}, rate=30), "unknown keys"),
        ({"dimensions": {"x": {"count": 1}}}, "missing rate_hz"),
    ],
)
def test_invalid_config_names_the_offending_key(config: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate(config)


def test_formats_use_the_largest_unit_reached() -> None:
    total = 4_592_700_000_000_000_000_000

    assert format_scientific(total) == "4.5927e+21"
    assert format_scientific(787_500) == "787500"
    assert format_scaled(total, SHORT_SCALE) == "4.5927 S"
    assert format_scaled(total, CHINESE_SCALE) == "45.927 垓"
    assert format_scaled(5_832 * 10**12, CHINESE_SCALE) == "5832 兆"
    assert format_scaled(35, SHORT_SCALE) == "35"


def test_main_reports_total_and_days(tmp_path, capsys) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(_config({"x": {"range": 1, "resolution": 0.3}})),
        encoding="utf-8",
    )

    main([str(path)])

    captured = capsys.readouterr()
    assert "warning: dimensions.x" in captured.err
    lines = captured.out.splitlines()
    assert [line.split()[:2] for line in lines if line.startswith(("x ", "total"))] == [
        ["x", "4"],
        ["total", "4"],
    ]
    assert any(line.startswith("days") for line in lines)
