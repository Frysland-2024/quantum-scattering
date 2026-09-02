from __future__ import annotations

from pathlib import Path
import sys

import part3_stationary_scattering as part3
import part6_complex_scaling_impl as impl


def _output_dir_from_argv() -> Path:
    try:
        index = sys.argv.index("--output-dir")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError):
        return impl.DEFAULT_OUTPUT.resolve()


def _part3_peaks() -> dict[str, float]:
    """Use the Part 3 numerical scan directly instead of an output metadata file."""
    (e1, _), (e2, _) = part3.find_resonance_peaks()
    return {"E1": e1, "E2": e2}


def _cleanup_metadata(output: Path) -> None:
    for name in ("part6_diagnostics.json", "part6_manifest.json"):
        try:
            (output / name).unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    output = _output_dir_from_argv()
    _cleanup_metadata(output)
    impl.read_part3_peaks = _part3_peaks
    impl.main()
    _cleanup_metadata(output)
    print("Part 6 final outputs: four PNG figures only (no diagnostics/manifest JSON).")


if __name__ == "__main__":
    main()
