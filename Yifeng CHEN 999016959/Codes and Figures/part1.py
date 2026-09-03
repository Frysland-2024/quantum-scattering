"""Part 1: generate Gaussian-state and free-propagation results."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_scattering.figures import generate_part1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Part 1 Gaussian and free-wave-packet figures."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--profile",
        choices=("quick", "reference", "full"),
        default="reference",
        help="Control animation frame counts.",
    )
    parser.add_argument("--skip-gifs", action="store_true", help="Generate PNG files only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    created = generate_part1(
        args.output_dir,
        profile=args.profile,
        include_gifs=not args.skip_gifs,
    )
    elapsed = time.perf_counter() - started
    print(f"Part 1 generated {len(created)} figures in {elapsed:.2f} s")
    for path in created:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
