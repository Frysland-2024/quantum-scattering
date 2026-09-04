"""Part 2: generate the finite-cutoff Dirac-delta results."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_scattering.figures import generate_part2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Part 2 regularized-Dirac-delta figures and data."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    created = generate_part2(args.output_dir)
    elapsed = time.perf_counter() - started
    print(f"Part 2 generated {len(created)} artifacts in {elapsed:.2f} s")
    for path in created:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
