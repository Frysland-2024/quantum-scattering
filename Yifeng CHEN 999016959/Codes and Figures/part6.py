"""Part 6: generate complex-scaling, Breit--Wigner, and wavefunction figures."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from quantum_scattering.complex_scaling import generate_part6
from quantum_scattering.resonance_wavefunction import generate_resonance_wavefunctions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the six Part 6 complex-scaling and resonance figures."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--profile",
        choices=("quick", "reference", "full"),
        default="full",
        help="quick uses J=400; reference/full use the teacher's J=2000, L=200, N=10000.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    created = generate_part6(args.output_dir, profile=args.profile)
    created.extend(
        generate_resonance_wavefunctions(args.output_dir, profile=args.profile)
    )
    figures = [
        Path(path)
        for path in created
        if Path(path).suffix.lower() == ".png" and Path(path).exists()
    ]
    elapsed = time.perf_counter() - started
    print(f"Part 6 generated {len(figures)} figures in {elapsed:.2f} s")
    for path in figures:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
