"""Command-line interface for reproducing individual project parts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .figures import generate_all, generate_part1, generate_part2, generate_part3, generate_part4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-scattering",
        description="Reproduce the lecture-note one-dimensional scattering results.",
    )
    parser.add_argument(
        "part",
        nargs="?",
        choices=("all", "part1", "part2", "part3", "part4"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--profile",
        choices=("quick", "reference", "full"),
        default="reference",
        help="quick lowers plot grids/frame counts; full raises animation frame counts",
    )
    parser.add_argument("--skip-gifs", action="store_true", help="Generate static results only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    include_gifs = not args.skip_gifs
    if args.part == "all":
        created = generate_all(args.output_dir, profile=args.profile, include_gifs=include_gifs)
    elif args.part == "part1":
        created = generate_part1(args.output_dir, profile=args.profile, include_gifs=include_gifs)
    elif args.part == "part2":
        created = generate_part2(args.output_dir)
    elif args.part == "part3":
        created = generate_part3(args.output_dir)
    else:
        created = generate_part4(args.output_dir, profile=args.profile, include_gifs=include_gifs)
    elapsed = time.perf_counter() - started
    print(f"Generated {len(created)} artifacts in {elapsed:.2f} s under {args.output_dir.resolve()}")
    for path in created:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

