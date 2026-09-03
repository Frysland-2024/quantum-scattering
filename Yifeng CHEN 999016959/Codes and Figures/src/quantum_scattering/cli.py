"""Command-line interface for reproducing individual project parts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .figures import generate_all, generate_part1, generate_part2, generate_part3, generate_part4
from .box_basis import generate_part5
from .complex_scaling import generate_part6
from .resonance_wavefunction import generate_resonance_wavefunctions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantum-scattering",
        description="Reproduce the lecture-note one-dimensional scattering results.",
    )
    parser.add_argument(
        "part",
        nargs="?",
        choices=("all", "part1", "part2", "part3", "part4", "part5", "part6"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--profile",
        choices=("quick", "reference", "full"),
        default="reference",
        help=(
            "quick lowers grids, frame counts, and Part-6 basis size; "
            "reference/full retain the report-scale Part-6 parameters"
        ),
    )
    parser.add_argument("--skip-gifs", action="store_true", help="Generate static results only")
    return parser


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def cleanup_non_deliverables(root: Path) -> None:
    """Remove development/audit metadata while keeping required project outputs.

    In particular, Part 2's delta_convergence.csv is a required deliverable and
    must never be removed here.
    """
    root = root.resolve()

    # Top-level generator bookkeeping is not part of the report deliverables.
    _remove_if_present(root / "manifest.json")

    # Part 3: keep the five PNG figures; remove only the numerical audit JSON.
    _remove_if_present(root / "part3_stationary_scattering" / "resonance_summary.json")

    # Part 4: keep all PNG/GIF results; remove diagnostics and manifest files.
    part4 = root / "part4_wavepacket_scattering"
    for path in part4.glob("*_diagnostics.json"):
        _remove_if_present(path)
    _remove_if_present(part4 / "part4_manifest.json")

    # Part 5: keep the five PNG figures only.
    part5 = root / "part5_box_basis"
    _remove_if_present(part5 / "part5_diagnostics.json")
    _remove_if_present(part5 / "part5_manifest.json")

    # Part 6: keep the six PNG figures only.
    part6 = root / "part6_complex_scaling"
    _remove_if_present(part6 / "part6_diagnostics.json")
    _remove_if_present(part6 / "part6_manifest.json")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    include_gifs = not args.skip_gifs
    if args.part == "all":
        created = generate_all(args.output_dir, profile=args.profile, include_gifs=include_gifs)
        created.extend(
            generate_resonance_wavefunctions(args.output_dir, profile=args.profile)
        )
    elif args.part == "part1":
        created = generate_part1(args.output_dir, profile=args.profile, include_gifs=include_gifs)
    elif args.part == "part2":
        created = generate_part2(args.output_dir)
    elif args.part == "part3":
        created = generate_part3(args.output_dir)
    elif args.part == "part4":
        created = generate_part4(args.output_dir, profile=args.profile, include_gifs=include_gifs)
    elif args.part == "part5":
        created = generate_part5(args.output_dir, profile=args.profile)
    else:
        created = generate_part6(args.output_dir, profile=args.profile)
        created.extend(
            generate_resonance_wavefunctions(args.output_dir, profile=args.profile)
        )

    cleanup_non_deliverables(args.output_dir)
    created = [path for path in created if Path(path).exists()]

    elapsed = time.perf_counter() - started
    print(f"Generated {len(created)} deliverables in {elapsed:.2f} s under {args.output_dir.resolve()}")
    for path in created:
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
