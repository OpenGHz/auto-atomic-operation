#!/usr/bin/env python3
"""Convert between base and EEF orientations.

This script converts base euler/quat to both base and EEF representations.
"""

import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def base_to_eef(tool_in_base: R, base_rot: R) -> R:
    """Convert base rotation to EEF world rotation."""
    return base_rot * tool_in_base


def print_rotation(name: str, rot: R):
    """Print rotation in multiple formats."""
    # Quaternion wxyz
    q_xyzw = rot.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # Intrinsic (uppercase in scipy): rotate about body axes
    intr_zyx = rot.as_euler("ZYX")  # intrinsic ZYX = extrinsic xyz
    intr_xyz = rot.as_euler("XYZ")  # intrinsic XYZ = extrinsic zyx

    # Extrinsic (lowercase in scipy): rotate about fixed world axes
    extr_zyx = rot.as_euler("zyx")  # extrinsic zyx = intrinsic XYZ
    extr_xyz = rot.as_euler("xyz")  # extrinsic xyz = intrinsic ZYX

    print(f"\n{name}:")
    print(
        f"  Quat (wxyz): [{q_wxyz[0]:9.6f}, {q_wxyz[1]:9.6f}, {q_wxyz[2]:9.6f}, {q_wxyz[3]:9.6f}]"
    )
    print(
        f"  Quat (xyzw): [{q_xyzw[0]:9.6f}, {q_xyzw[1]:9.6f}, {q_xyzw[2]:9.6f}, {q_xyzw[3]:9.6f}]"
    )
    print(f"  --- Intrinsic (body-frame axes) ---")
    print(
        f"  Intrinsic ZYX (ctrl): yaw={intr_zyx[0]:7.4f}, pitch={intr_zyx[1]:7.4f}, roll={intr_zyx[2]:7.4f}"
    )
    print(
        f"  Intrinsic XYZ:        [{intr_xyz[0]:7.4f}, {intr_xyz[1]:7.4f}, {intr_xyz[2]:7.4f}]"
    )
    print(f"  --- Extrinsic (fixed-frame axes) ---")
    print(
        f"  Extrinsic ZYX:        [{extr_zyx[0]:7.4f}, {extr_zyx[1]:7.4f}, {extr_zyx[2]:7.4f}]"
    )
    print(
        f"  Extrinsic XYZ:        [{extr_xyz[0]:7.4f}, {extr_xyz[1]:7.4f}, {extr_xyz[2]:7.4f}]"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Convert base orientation to base and EEF representations.",
        epilog="Examples:\n"
        "  python base_eef_converter.py 0 0 0           # euler XYZ (default)\n"
        "  python base_eef_converter.py 0 1.5707 0 --zyx  # intrinsic ZYX (MuJoCo ctrl)\n"
        "  python base_eef_converter.py 1 0 0 0         # quat wxyz\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "values",
        nargs="+",
        type=float,
        help="3 values for euler or 4 values for quat wxyz",
    )
    parser.add_argument(
        "--zyx",
        action="store_true",
        help="Use intrinsic ZYX euler convention (yaw pitch roll) instead of intrinsic XYZ",
    )
    parser.add_argument(
        "--tool-in-base",
        nargs=4,
        type=float,
        default=[0.70710678, 0.70710678, 0, 0],
        help="Tool orientation in base frame as quat xyzw (default: [0.70710678, 0.70710678, 0, 0])",
    )
    args = parser.parse_args()

    tool_in_base = R.from_quat(args.tool_in_base)

    if len(args.values) == 3:
        if args.zyx:
            yaw, pitch, roll = args.values
            base_rot = R.from_euler("ZYX", [yaw, pitch, roll])
            print(f"Input: Intrinsic ZYX (yaw={yaw}, pitch={pitch}, roll={roll})")
        else:
            x, y, z = args.values
            base_rot = R.from_euler("XYZ", [x, y, z])
            print(f"Input: Intrinsic XYZ ({x}, {y}, {z})")
    elif len(args.values) == 4:
        w, x, y, z = args.values
        base_rot = R.from_quat([x, y, z, w])
        print(f"Input: Quat wxyz ({w}, {x}, {y}, {z})")
    else:
        parser.error(f"Expected 3 (euler) or 4 (quat) values, got {len(args.values)}")

    eef_rot = base_to_eef(tool_in_base, base_rot)

    print("\n" + "=" * 60)
    print_rotation("BASE", base_rot)
    print_rotation("EEF", eef_rot)
    print("=" * 60)


if __name__ == "__main__":
    main()
