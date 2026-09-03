"""Part 5: generate Lecture-8 sine-box matrix eigenstate results."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_scattering.box_basis import generate_part5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Part 5 particle-in-a-box sine-basis matrix results."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--profile",
        choices=("quick", "reference", "full"),
        default="reference",
        help="quick lowers J/L/N; reference and full use the teacher's displayed parameters.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    created = generate_part5(args.output_dir, profile=args.profile)
    elapsed = time.perf_counter() - started
    print(f"Part 5 generated {len(created)} figures in {elapsed:.2f} s")
    for path in created:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
