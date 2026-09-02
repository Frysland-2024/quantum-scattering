import part3_stationary_scattering as part3
import part4_wavepacket_scattering_impl as impl


OUT = impl.OUT


def part3_peaks() -> dict[str, float]:
    """Use the Part 3 numerical scan directly instead of an output metadata file."""
    (e1, _), (e2, _) = part3.find_resonance_peaks()
    return {"E1": e1, "E2": e2}


def cleanup_diagnostics() -> None:
    """Keep Part 4 deliverables limited to the report PNG/GIF outputs."""
    for path in OUT.glob("part4_*_diagnostics.json"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    cleanup_diagnostics()
    impl.saved_part3_peaks = part3_peaks
    impl.scenario_catalog.cache_clear()
    impl.main()
    cleanup_diagnostics()
    print("Part 4 final outputs: PNG/GIF only (no diagnostics JSON).")


if __name__ == "__main__":
    main()
