from __future__ import annotations

from pathlib import Path
import sys

import part5_box_basis_impl as impl


RENAME_MAP = {
    "harmonic_oscillator_validation.png": "part5_harmonic_validation.png",
    "short_range_all_states.png": "part5_box_spectrum_all_states.png",
    "three_localized_states.png": "part5_localized_states.png",
    "short_range_first11_combined.png": "part5_short_range_first11_combined.png",
    "short_range_first11_separate.png": "part5_short_range_first11_panels.png",
}


def _output_dir_from_argv() -> Path:
    try:
        index = sys.argv.index("--output-dir")
        return Path(sys.argv[index + 1]).resolve()
    except (ValueError, IndexError):
        return impl.DEFAULT_OUTPUT.resolve()


def _cleanup_metadata(output: Path) -> None:
    diagnostics = output / "part5_diagnostics.json"
    try:
        diagnostics.unlink()
    except FileNotFoundError:
        pass


def _normalize_output_names(output: Path) -> None:
    for source_name, target_name in RENAME_MAP.items():
        source = output / source_name
        target = output / target_name
        if source.exists():
            if target.exists():
                target.unlink()
            source.replace(target)


def main() -> None:
    output = _output_dir_from_argv()
    _cleanup_metadata(output)
    impl.main()
    _normalize_output_names(output)
    _cleanup_metadata(output)
    print("Part 5 final outputs: five PNG figures only (no diagnostics JSON).")


if __name__ == "__main__":
    main()
