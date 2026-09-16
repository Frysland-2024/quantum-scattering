from __future__ import annotations

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

import part3_stationary_scattering as part3
from quantum_scattering.complex_scaling import (
    ComplexScalingConfig,
    ComplexSpectrum,
    ResonanceTrack,
    breit_wigner,
    identify_resonance_tracks,
    nearest_track_value,
    plot_window,
    solve_complex_spectrum,
)
from quantum_scattering.core import ensure_dir
from quantum_scattering.scattering import appendix_v_scan


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "part6"
DEFAULT_ANGLES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
REAL_LIMITS = (-0.35, 3.0)
IMAG_LIMITS = (-2.0, 0.10)
COLORS = ("C0", "C1", "C2", "C3", "C4", "C5")
MARKERS = ("o", "s", "^", "D", "v", "*")
BW_COLORS = ("red", "green", "dodgerblue", "darkorange", "magenta")
ORANGE = "#ef6c00"
EXPECTED_RESONANCE_COUNT = 5


def configuration(quick: bool) -> ComplexScalingConfig:
    if quick:
        return ComplexScalingConfig(400, 200.0, 2401)
    return ComplexScalingConfig(2000, 200.0, 10000)


def parse_angles(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("angles must be comma-separated numbers") from exc
    if len(values) < 2 or any(b <= a for a, b in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("angles must be strictly increasing")
    if not np.isclose(values[0], 0.0):
        raise argparse.ArgumentTypeError("the angle family must start at theta=0")
    return values


def read_part3_peaks() -> dict[str, float]:
    """Use the Part 3 numerical scan directly; no metadata file is needed."""
    (e1, _), (e2, _) = part3.find_resonance_peaks()
    return {"E1": e1, "E2": e2}


def load_part3_transmission() -> tuple[np.ndarray, np.ndarray]:
    peaks = read_part3_peaks()
    energies = np.unique(
        np.r_[
            np.linspace(0.002, 3.0, 2200),
            np.linspace(peaks["E1"] - 0.004, peaks["E1"] + 0.004, 1600),
            np.linspace(peaks["E2"] - 0.10, peaks["E2"] + 0.10, 1500),
            peaks["E1"],
            peaks["E2"],
        ]
    )
    scan = appendix_v_scan(energies, dx=0.0025)
    return energies, np.abs(scan.T) ** 2


def solve_family(
    angles: tuple[float, ...], config: ComplexScalingConfig
) -> list[ComplexSpectrum]:
    return [solve_complex_spectrum(theta, config) for theta in angles]


def banner(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.012,
        0.985,
        text,
        color=ORANGE,
        fontsize=20,
        fontweight="bold",
        va="top",
    )


def scatter_spectrum(
    ax: plt.Axes,
    spectrum: ComplexSpectrum,
    slot: int,
    size: float = 32.0,
) -> None:
    values = plot_window(spectrum.eigenvalues, REAL_LIMITS, IMAG_LIMITS)
    ax.scatter(
        values.real,
        values.imag,
        s=size,
        marker=MARKERS[slot % len(MARKERS)],
        facecolors="none",
        edgecolors=COLORS[slot % len(COLORS)],
        linewidths=1.05,
        label=rf"$\theta={spectrum.theta:.2f}$",
    )


def apply_complex_axes(ax: plt.Axes) -> None:
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set(
        xlim=REAL_LIMITS,
        ylim=IMAG_LIMITS,
        xlabel=r"Re($E$)",
        ylabel=r"Im($E$)",
    )


def plot_combined_spectrum(
    output: Path,
    spectra: list[ComplexSpectrum],
    config: ComplexScalingConfig,
) -> Path:
    fig, ax = plt.subplots(figsize=(14.0, 7.3))
    banner(fig, "complex eigenvalues (plotted in one picture for various values of theta)")
    for slot, spectrum in enumerate(spectra):
        scatter_spectrum(ax, spectrum, slot, size=34.0)
    apply_complex_axes(ax)
    ax.set_title(
        rf"Complex-scaling spectra for multiple $\theta$, $J={config.basis_size}$",
        fontsize=15,
    )
    ax.legend(loc="lower left", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = output / "part6_complex_eigenvalues_all.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_individual_spectra(
    output: Path,
    spectra: list[ComplexSpectrum],
    config: ComplexScalingConfig,
) -> Path:
    columns = 3
    rows = int(np.ceil(len(spectra) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(14.0, 7.4), squeeze=False)
    banner(fig, "complex eigenvalues (separately for each theta)")
    for slot, (ax, spectrum) in enumerate(zip(axes.ravel(), spectra)):
        scatter_spectrum(ax, spectrum, slot, size=25.0)
        apply_complex_axes(ax)
        ax.set_title(rf"$\theta={spectrum.theta:.2f}$ rad", fontsize=11)
    for ax in axes.ravel()[len(spectra):]:
        ax.axis("off")
    fig.suptitle(
        rf"Complex-scaling individual spectra, $J={config.basis_size}$",
        fontsize=15,
        y=0.95,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output / "part6_complex_eigenvalues_separate.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def broad_energy_grid(part3_energy: np.ndarray) -> np.ndarray:
    mask = (part3_energy >= 0.0) & (part3_energy <= 3.0)
    return part3_energy[mask]


def format_pole(pole: complex, decimals: int = 5) -> str:
    return f"{pole.real:.{decimals}f}{pole.imag:+.{decimals}f}i"


def plot_all_breit_wigner(
    output: Path,
    spectra: list[ComplexSpectrum],
    tracks: list[ResonanceTrack],
    part3_energy: np.ndarray,
    part3_transmission: np.ndarray,
    peaks: dict[str, float],
) -> Path:
    energy = broad_energy_grid(part3_energy)
    transmission = np.interp(energy, part3_energy, part3_transmission)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13.2, 9.2))
    banner(fig, "Breit-Wigner profiles (all)")
    top.plot(
        energy,
        transmission,
        "k",
        lw=1.2,
        label="Hermitian QM Transmission profile",
    )
    for slot, track in enumerate(tracks):
        top.plot(
            energy,
            breit_wigner(energy, track.pole),
            color=BW_COLORS[slot % len(BW_COLORS)],
            linestyle=":",
            lw=1.6,
            label=rf"Breit-Wigner at $E={format_pole(track.pole)}$",
        )
    for key in ("E1", "E2"):
        peak_energy = peaks[key]
        peak_value = float(np.interp(peak_energy, part3_energy, part3_transmission))
        top.scatter([peak_energy], [peak_value], color="red", s=20, zorder=6)
        top.annotate(
            f"E={peak_energy:.6f}\nT={peak_value:.6f}",
            (peak_energy, peak_value),
            xytext=(8, -12),
            textcoords="offset points",
            color="red",
            fontsize=9,
        )
    top.set(
        xlim=(0.0, 3.0),
        ylim=(-0.05, 1.05),
        ylabel=r"$|T(E)|^2$",
        title="Breit-Wigner Transmission Approximation based on non-Hermitian Complex-scaling",
    )
    top.legend(loc="lower left", fontsize=7.2)

    for slot, spectrum in enumerate(spectra):
        scatter_spectrum(bottom, spectrum, slot, size=12.0)
    bottom.scatter(
        [track.pole.real for track in tracks],
        [track.pole.imag for track in tracks],
        color="red",
        s=22,
        label="Breit-Wigner selected",
        zorder=8,
    )
    apply_complex_axes(bottom)
    bottom.set_ylim(-1.75, 0.10)
    bottom.legend(loc="lower left", fontsize=7.2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    path = output / "part6_breit_wigner_all.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def plot_breit_wigner_zoom(
    output: Path,
    angles: tuple[float, ...],
    tracks: list[ResonanceTrack],
    part3_energy: np.ndarray,
    part3_transmission: np.ndarray,
    peaks: dict[str, float],
) -> Path:
    if len(tracks) < 2:
        raise RuntimeError("the first two resonance tracks were not identified")
    first, second = tracks[:2]
    windows = (
        (first, peaks["E1"], max(0.0005, 8.5 * (-first.pole.imag))),
        (second, peaks["E2"], max(0.14, 4.5 * (-second.pole.imag))),
    )
    titles = (
        "Transmission profile around first resonance peak",
        "Around the second resonance peak",
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    banner(fig, "Breit-Wigner profiles (detailed zoom)")
    for resonance_number, (ax, (track, peak_energy, half_window), title) in enumerate(
        zip(axes, windows, titles),
        start=1,
    ):
        energy = np.linspace(peak_energy - half_window, peak_energy + half_window, 1800)
        exact = np.interp(energy, part3_energy, part3_transmission)
        ax.plot(energy, exact, "k", lw=1.2, label="Hermitian method")
        peak_value = float(np.interp(peak_energy, part3_energy, part3_transmission))
        ax.scatter(
            [peak_energy],
            [peak_value],
            color="red",
            s=24,
            zorder=6,
            label=rf"$\theta=0.00$, $E_{resonance_number}={peak_energy:.6f}$",
        )
        for slot, theta in enumerate(angles[1:], start=1):
            pole = nearest_track_value(track, theta)
            ax.plot(
                energy,
                breit_wigner(energy, pole),
                linestyle=":",
                color=BW_COLORS[(slot - 1) % len(BW_COLORS)],
                lw=1.5,
                label=rf"$\theta={theta:.2f}$, $E_{resonance_number}={format_pole(pole, 6)}$",
            )
        ax.set(
            xlabel=r"Re($E$)",
            ylabel=r"$|T(E)|^2$",
            ylim=(-0.04, 1.05),
            title=title,
        )
        ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output / "part6_breit_wigner_zoom.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 6 non-Hermitian complex scaling and Breit-Wigner resonances"
    )
    parser.add_argument("--quick", action="store_true", help="J=400 validation profile")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only",
        choices=("all", "spectra", "breit-wigner"),
        default="all",
        help="limit figure generation; spectra are always solved",
    )
    parser.add_argument(
        "--angles",
        type=parse_angles,
        default=DEFAULT_ANGLES,
        help="comma-separated rotation angles (default: 0,0.05,...,0.25)",
    )
    parser.add_argument("--basis-size", type=int, help="override profile J")
    parser.add_argument("--box-length", type=float, help="override profile L")
    parser.add_argument("--quadrature-points", type=int, help="override profile N")
    args = parser.parse_args()

    output = ensure_dir(args.output_dir.resolve())

    base = configuration(args.quick)
    config = ComplexScalingConfig(
        args.basis_size or base.basis_size,
        args.box_length or base.box_length,
        args.quadrature_points or base.quadrature_points,
        base.mass,
        base.hbar,
    )
    config.validate()
    angles = tuple(args.angles)
    spectra = solve_family(angles, config)
    tracks = identify_resonance_tracks(spectra)
    if len(tracks) != EXPECTED_RESONANCE_COUNT:
        raise RuntimeError(
            f"expected exactly {EXPECTED_RESONANCE_COUNT} resonance poles, found {len(tracks)}"
        )
    print(
        "identified poles: "
        + "; ".join(
            f"{track.label}={format_pole(track.pole, 8)}"
            for track in tracks
        )
    )

    peaks = read_part3_peaks()
    if args.only in ("all", "spectra"):
        plot_combined_spectrum(output, spectra, config)
        plot_individual_spectra(output, spectra, config)

    if args.only in ("all", "breit-wigner"):
        part3_energy, part3_transmission = load_part3_transmission()
        plot_all_breit_wigner(
            output,
            spectra,
            tracks,
            part3_energy,
            part3_transmission,
            peaks,
        )
        plot_breit_wigner_zoom(
            output,
            angles,
            tracks,
            part3_energy,
            part3_transmission,
            peaks,
        )

    print(f"Part 6 outputs: {output}")


if __name__ == "__main__":
    main()
