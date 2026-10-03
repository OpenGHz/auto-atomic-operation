"""Visualize standard or compressed Gaussian-splatting PLY files.

This script supports:
1. Standard point-cloud PLY files with explicit x/y/z and optional RGB fields.
2. Compressed SuperSplat-style PLY files with chunk metadata and packed vertex
   fields like ``packed_position`` and ``packed_color``.

Examples:
    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        scripts/render/gs/visualize_ply.py third_party/3dgs/backgrounds/franka_table.ply

    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        scripts/render/gs/visualize_ply.py third_party/3dgs/backgrounds --limit 80000

    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        scripts/render/gs/visualize_ply.py third_party/3dgs/backgrounds/airbot_play_background.ply \
        --save /tmp/airbot_background.png --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


def _decode_uint_to_range(
    packed: np.ndarray,
    bit_offset: int,
    bit_count: int,
    vmin: np.ndarray,
    vmax: np.ndarray,
) -> np.ndarray:
    mask = (1 << bit_count) - 1
    quantized = (packed >> bit_offset) & mask
    denom = max(mask, 1)
    return vmin + (quantized.astype(np.float32) / denom) * (vmax - vmin)


def _load_standard_vertices(vertex_data) -> tuple[np.ndarray, np.ndarray | None]:
    names = set(vertex_data.data.dtype.names or [])
    xyz_names = ("x", "y", "z")
    if not all(name in names for name in xyz_names):
        raise ValueError("Standard vertex layout requires x/y/z fields.")

    xyz = np.column_stack([vertex_data[name].astype(np.float32) for name in xyz_names])

    rgb_candidates = [
        ("red", "green", "blue"),
        ("r", "g", "b"),
    ]
    colors = None
    for candidate in rgb_candidates:
        if all(name in names for name in candidate):
            colors = (
                np.column_stack(
                    [vertex_data[name].astype(np.float32) for name in candidate]
                )
                / 255.0
            )
            break

    return xyz, colors


def _load_compressed_vertices(ply: PlyData) -> tuple[np.ndarray, np.ndarray | None]:
    chunk = ply["chunk"].data
    vertex = ply["vertex"].data
    vertex_count = len(vertex)
    chunk_count = len(chunk)

    if chunk_count == 0:
        raise ValueError("Compressed PLY has no chunks.")

    vertices_per_chunk = int(np.ceil(vertex_count / chunk_count))
    chunk_index = np.arange(vertex_count, dtype=np.int64) // vertices_per_chunk
    chunk_index = np.clip(chunk_index, 0, chunk_count - 1)

    min_xyz = np.column_stack([chunk["min_x"], chunk["min_y"], chunk["min_z"]]).astype(
        np.float32
    )[chunk_index]
    max_xyz = np.column_stack([chunk["max_x"], chunk["max_y"], chunk["max_z"]]).astype(
        np.float32
    )[chunk_index]

    packed_position = vertex["packed_position"].astype(np.uint32)
    x = _decode_uint_to_range(packed_position, 21, 11, min_xyz[:, 0], max_xyz[:, 0])
    y = _decode_uint_to_range(packed_position, 11, 10, min_xyz[:, 1], max_xyz[:, 1])
    z = _decode_uint_to_range(packed_position, 0, 11, min_xyz[:, 2], max_xyz[:, 2])
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    colors = None
    if "packed_color" in (vertex.dtype.names or ()):
        min_rgb = np.column_stack(
            [chunk["min_r"], chunk["min_g"], chunk["min_b"]]
        ).astype(np.float32)[chunk_index]
        max_rgb = np.column_stack(
            [chunk["max_r"], chunk["max_g"], chunk["max_b"]]
        ).astype(np.float32)[chunk_index]
        packed_color = vertex["packed_color"].astype(np.uint32)
        r = _decode_uint_to_range(packed_color, 24, 8, min_rgb[:, 0], max_rgb[:, 0])
        g = _decode_uint_to_range(packed_color, 16, 8, min_rgb[:, 1], max_rgb[:, 1])
        b = _decode_uint_to_range(packed_color, 8, 8, min_rgb[:, 2], max_rgb[:, 2])
        colors = np.clip(np.column_stack([r, g, b]), 0.0, 1.0).astype(np.float32)

    return xyz, colors


def load_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"{path} does not contain a vertex element.")

    vertex = ply["vertex"]
    names = set(vertex.data.dtype.names or [])
    if {"x", "y", "z"}.issubset(names):
        return _load_standard_vertices(vertex)

    if "chunk" in ply and "packed_position" in names:
        return _load_compressed_vertices(ply)

    raise ValueError(
        f"Unsupported PLY vertex layout in {path}. Available fields: {sorted(names)}"
    )


def sample_points(
    xyz: np.ndarray,
    colors: np.ndarray | None,
    limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(xyz) <= limit:
        return xyz, colors

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(xyz), size=limit, replace=False)
    indices.sort()
    if colors is None:
        return xyz[indices], None
    return xyz[indices], colors[indices]


def set_equal_axes(ax, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    if radius == 0.0:
        radius = 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize_file(
    path: Path,
    limit: int,
    point_size: float,
    elev: float,
    azim: float,
    seed: int,
    save: Path | None,
    show: bool,
    backend: str,
) -> None:
    xyz, colors = load_ply_points(path)
    xyz, colors = sample_points(xyz, colors, limit=limit, seed=seed)

    if backend == "open3d":
        if o3d is None:
            raise ModuleNotFoundError(
                "open3d is not installed in the current Python environment."
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        else:
            normalized = xyz - xyz.min(axis=0, keepdims=True)
            denom = np.maximum(normalized.ptp(axis=0, keepdims=True), 1e-6)
            pcd.colors = o3d.utility.Vector3dVector(
                (normalized / denom).astype(np.float64)
            )

        if save is not None:
            print(
                "open3d backend does not save screenshots directly; falling back to matplotlib for --save."
            )
        if show:
            o3d.visualization.draw_geometries([pcd], window_name=path.name)
            return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        s=point_size,
        c=colors if colors is not None else xyz[:, 2],
        cmap=None if colors is not None else "viridis",
        linewidths=0,
        alpha=0.9,
    )
    ax.set_title(f"{path.name} ({len(xyz):,} sampled points)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    set_equal_axes(ax, xyz)
    fig.tight_layout()

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=200)
        print(f"Saved figure to {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def iter_ply_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.glob("*.ply") if p.is_file())
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="PLY file or a directory of PLY files.")
    parser.add_argument(
        "--limit",
        type=int,
        default=100000,
        help="Maximum number of points to visualize per file.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.2,
        help="Matplotlib scatter point size.",
    )
    parser.add_argument("--elev", type=float, default=25.0, help="Camera elevation.")
    parser.add_argument("--azim", type=float, default=35.0, help="Camera azimuth.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output image path. Directory mode requires --save-dir instead.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for saved images when visualizing a folder.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "matplotlib", "open3d"),
        default="auto",
        help="Visualization backend. 'auto' prefers open3d when available and interactive.",
    )
    args = parser.parse_args()

    ply_files = iter_ply_files(args.path)
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found under {args.path}")

    if len(ply_files) > 1 and args.save is not None:
        raise ValueError("--save can only be used with a single input file.")

    for ply_file in ply_files:
        save_path = args.save
        if args.save_dir is not None:
            save_path = args.save_dir / f"{ply_file.stem}.png"
        backend = args.backend
        if backend == "auto":
            backend = (
                "open3d"
                if (o3d is not None and not args.no_show and save_path is None)
                else "matplotlib"
            )
        print(f"Visualizing {ply_file}")
        visualize_file(
            path=ply_file,
            limit=args.limit,
            point_size=args.point_size,
            elev=args.elev,
            azim=args.azim,
            seed=args.seed,
            save=save_path,
            show=not args.no_show,
            backend=backend,
        )


if __name__ == "__main__":
    main()
