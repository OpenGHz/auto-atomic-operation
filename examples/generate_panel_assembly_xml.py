"""Generate a panel MJCF by attaching configured objects to mounting slots.

Example:
    /home/ghz/.mini_conda3/envs/airbot_play_data/bin/python \
        examples/generate_panel_assembly_xml.py \
        examples/panel_assembly/layout_7x3.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from auto_atom.utils.panel_xml_builder import generate_panel_assembly


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the panel assembly YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional override for the generated XML output path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = generate_panel_assembly(args.config, output_path=args.output)

    print(f"Generated XML: {result.output_path}")
    print(f"Mounted objects: {len(result.placements)}")
    for placement in result.placements:
        print(
            f"  row={placement.row} col={placement.col} "
            f"slot={placement.slot_name} body={placement.object_name} "
            f"xml={placement.source_xml}"
        )


if __name__ == "__main__":
    main()
