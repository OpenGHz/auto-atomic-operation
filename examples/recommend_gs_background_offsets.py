"""Estimate initial GS background xyz offsets from background PLY geometry.

This reproduces the heuristic used earlier to populate
`aao_configs/gs_mixin.yaml`:

1. Build a Z histogram over all points.
2. Take the top-N densest bins.
3. Choose the dense bin whose center is closest to a reference tabletop Z.
4. Gather a narrow band around that Z as the candidate tabletop plane.
5. Estimate the tabletop center from the median point in that band.
6. Align that estimated center to a target tabletop reference pose to obtain
   the recommended xyz offset.

Examples:
    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        examples/recommend_gs_background_offsets.py

    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        examples/recommend_gs_background_offsets.py assets/gs/backgrounds/table.ply

    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        examples/recommend_gs_background_offsets.py \
        --target-center 0.45 0.06 \
        --target-z 0.0742 \
        --yaml
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gaussian_renderer.core.util_gau import load_ply

DEFAULT_INPUT = Path("assets/gs/backgrounds")


@dataclass(frozen=True)
class BackgroundOffsetEstimate:
    name: str
    path: Path
    peak_z: float
    plane_center_median: tuple[float, float, float]
    plane_center_mean: tuple[float, float, float]
    recommended_offset: tuple[float, float, float]
    band_points: int
    used_points: int
    bin_width: float


def _round_tuple(values, ndigits: int = 4) -> tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in values)


def _iter_ply_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".ply":
            raise ValueError(f"Expected a .ply file, got: {path}")
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    return sorted(p for p in path.glob("*.ply") if p.is_file())


def estimate_background_offset(
    path: Path,
    *,
    target_center_xy: tuple[float, float] = (0.45, 0.06),
    target_tabletop_z: float = 0.0742,
    histogram_bins: int = 400,
    top_dense_bins: int = 30,
    min_band_half_width: float = 0.02,
    local_radius_xy: float = 1.5,
    min_local_points: int = 5000,
) -> BackgroundOffsetEstimate:
    gaussians = load_ply(str(path))
    xyz = np.asarray(gaussians.xyz, dtype=np.float32)
    z = xyz[:, 2]

    hist, edges = np.histogram(z, bins=histogram_bins)
    mids = (edges[:-1] + edges[1:]) / 2.0
    top_indices = hist.argsort()[-top_dense_bins:]
    chosen_idx = min(
        top_indices, key=lambda idx: abs(float(mids[idx] - target_tabletop_z))
    )

    peak_z = float(mids[chosen_idx])
    bin_width = float(edges[1] - edges[0])
    band_half_width = max(bin_width * 1.5, min_band_half_width)
    band = xyz[np.abs(z - peak_z) < band_half_width]

    band_median_xy = np.median(band[:, :2], axis=0)
    local = band[
        (np.abs(band[:, 0] - band_median_xy[0]) < local_radius_xy)
        & (np.abs(band[:, 1] - band_median_xy[1]) < local_radius_xy)
    ]
    used = local if len(local) > min_local_points else band

    plane_center_median = np.median(used, axis=0)
    plane_center_mean = np.mean(used, axis=0)

    target = np.array(
        [target_center_xy[0], target_center_xy[1], target_tabletop_z],
        dtype=np.float32,
    )
    recommended_offset = target - plane_center_median

    return BackgroundOffsetEstimate(
        name=path.stem,
        path=path,
        peak_z=peak_z,
        plane_center_median=_round_tuple(plane_center_median),
        plane_center_mean=_round_tuple(plane_center_mean),
        recommended_offset=_round_tuple(recommended_offset),
        band_points=int(len(band)),
        used_points=int(len(used)),
        bin_width=round(bin_width, 6),
    )


def _print_verbose(est: BackgroundOffsetEstimate) -> None:
    print(f"=== {est.name} ===")
    print(f"path: {est.path}")
    print(f"peak_z: {est.peak_z:.4f}")
    print(f"band_points: {est.band_points}")
    print(f"used_points: {est.used_points}")
    print(f"bin_width: {est.bin_width:.6f}")
    print(f"plane_center_median: {list(est.plane_center_median)}")
    print(f"plane_center_mean: {list(est.plane_center_mean)}")
    print(f"recommended_offset: {list(est.recommended_offset)}")


def _print_yaml(estimates: list[BackgroundOffsetEstimate]) -> None:
    print("background_offsets:")
    for est in estimates:
        x, y, z = est.recommended_offset
        print(f"  {est.name}: [{x:.4f}, {y:.4f}, {z:.4f}]")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help="A background .ply file or a directory containing background .ply files.",
    )
    parser.add_argument(
        "--target-center",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        default=(0.45, 0.06),
        help="Target tabletop center xy in MuJoCo world.",
    )
    parser.add_argument(
        "--target-z",
        type=float,
        default=0.0742,
        help="Target tabletop top-surface z in MuJoCo world.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=400,
        help="Number of z histogram bins.",
    )
    parser.add_argument(
        "--top-dense-bins",
        type=int,
        default=30,
        help="How many dense z bins to consider before choosing the one nearest target-z.",
    )
    parser.add_argument(
        "--min-band-half-width",
        type=float,
        default=0.02,
        help="Minimum half-width of the candidate tabletop z band.",
    )
    parser.add_argument(
        "--local-radius-xy",
        type=float,
        default=1.5,
        help="Half-width of the local xy window around the band median.",
    )
    parser.add_argument(
        "--min-local-points",
        type=int,
        default=5000,
        help="Use the local xy subset only if it contains at least this many points.",
    )
    parser.add_argument(
        "--yaml",
        action="store_true",
        help="Print only a YAML-ready background_offsets block.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    paths = _iter_ply_files(args.path)
    estimates = [
        estimate_background_offset(
            path,
            target_center_xy=tuple(args.target_center),
            target_tabletop_z=float(args.target_z),
            histogram_bins=int(args.histogram_bins),
            top_dense_bins=int(args.top_dense_bins),
            min_band_half_width=float(args.min_band_half_width),
            local_radius_xy=float(args.local_radius_xy),
            min_local_points=int(args.min_local_points),
        )
        for path in paths
    ]

    if args.yaml:
        _print_yaml(estimates)
        return

    for idx, est in enumerate(estimates):
        if idx:
            print()
        _print_verbose(est)


if __name__ == "__main__":
    main()
