from pathlib import Path

import part4_wavepacket_scattering_impl as impl


OUT = impl.OUT


def cleanup_diagnostics() -> None:
    """Keep Part 4 deliverables limited to the report PNG/GIF outputs."""
    for path in OUT.glob("part4_*_diagnostics.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    cleanup_diagnostics()
    impl.main()
    cleanup_diagnostics()
    print("Part 4 final outputs: PNG/GIF only (no diagnostics JSON).")


if __name__ == "__main__":
    main()
