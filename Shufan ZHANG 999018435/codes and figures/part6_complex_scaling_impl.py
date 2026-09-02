from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import time

import matplotlib.pyplot as plt
import numpy as np

from quantum_scattering.complex_scaling import (
    ComplexScalingConfig,
    ComplexSpectrum,
    ResonanceTrack,
    breit_wigner,
    distance_from_continuum_ray,
    identify_resonance_tracks,
    nearest_track_value,
    plot_window,
    solve_complex_spectrum,
)
from quantum_scattering.core import ensure_dir
from quantum_scattering.scattering import appendix_v_scan


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "part6"
PART3_SUMMARY = ROOT / "outputs" / "part3" / "part3_peak_summary.txt"
DEFAULT_ANGLES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
REAL_LIMITS = (-0.35, 3.0)
IMAG_LIMITS = (-2.0, 0.10)
COLORS = ("C0", "C1", "C2", "C3", "C4", "C5")
MARKERS = ("o", "s", "^", "D", "v", "*")
BW_COLORS = ("red", "green", "dodgerblue", "darkorange", "magenta")
ORANGE = "#ef6c00"
REFERENCE_BOUND_ENERGY = -0.297959638
REFERENCE_POLES = (
    0.620971 - 0.000058j,
    1.327197 - 0.015447j,
    1.78458 - 0.17375j,
    2.12442 - 0.56479j,
    2.45549 - 1.11153j,
)


def configuration(quick: bool) -> ComplexScalingConfig:
    if quick:
        # J=400 already resolves the five displayed poles to better than 1e-5.
        return ComplexScalingConfig(400, 200.0, 2401)
    # The values printed in the teacher's Part 6 screenshots.
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
    peaks: dict[str, float] = {}
    try:
        for line in PART3_SUMMARY.read_text(encoding="utf-8").splitlines():
            for key in ("E1", "E2"):
                if line.startswith(key + "="):
                    peaks[key] = float(line.split("=", 1)[1].split(",", 1)[0])
    except (OSError, ValueError):
        pass
    peaks.setdefault("E1", 0.6209703037)
    peaks.setdefault("E2", 1.3288237771)
    return peaks


def load_part3_transmission() -> tuple[np.ndarray, np.ndarray, str]:
    """Recompute the Part 3 transmission on its original deterministic grid."""
    peaks = read_part3_peaks()
    energies = np.unique(
        np.r_[
            np.linspace(0.002, 3.0, 2200),
            np.linspace(peaks["E1"] - 0.004, peaks["E1"] + 0.004, 1600),
            np.linspace(peaks["E2"] - 0.10, peaks["E2"] + 0.10, 1500),
            peaks["E1"], peaks["E2"],
        ]
    )
    scan = appendix_v_scan(energies, dx=0.0025)
    return energies, np.abs(scan.T) ** 2, "recomputed with appendix_v_scan(dx=0.0025)"


def solve_family(
    angles: tuple[float, ...], config: ComplexScalingConfig
) -> tuple[list[ComplexSpectrum], dict[str, float]]:
    spectra: list[ComplexSpectrum] = []
    runtimes: dict[str, float] = {}
    for theta in angles:
        started = time.perf_counter()
        spectrum = solve_complex_spectrum(theta, config)
        elapsed = time.perf_counter() - started
        spectra.append(spectrum)
        runtimes[f"theta_{theta:.2f}_seconds"] = elapsed
        print(f"theta={theta:.2f}: {config.basis_size} eigenvalues in {elapsed:.2f} s")
    return spectra, runtimes


def banner(fig: plt.Figure, text: str) -> None:
    fig.text(0.012, 0.985, text, color=ORANGE, fontsize=20, fontweight="bold", va="top")


def scatter_spectrum(ax: plt.Axes, spectrum: ComplexSpectrum, slot: int, size: float = 32.0) -> None:
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
    ax.set(xlim=REAL_LIMITS, ylim=IMAG_LIMITS, xlabel=r"Re($E$)", ylabel=r"Im($E$)")


def plot_combined_spectrum(output: Path, spectra: list[ComplexSpectrum], config: ComplexScalingConfig) -> Path:
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
    output: Path, spectra: list[ComplexSpectrum], config: ComplexScalingConfig
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
    top.plot(energy, transmission, "k", lw=1.2, label="Hermitian QM Transmission profile")
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
    titles = ("Transmission profile around first resonance peak", "Around the second resonance peak")
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))
    banner(fig, "Breit-Wigner profiles (detailed zoom)")
    for resonance_number, (ax, (track, peak_energy, half_window), title) in enumerate(
        zip(axes, windows, titles), start=1
    ):
        energy = np.linspace(peak_energy - half_window, peak_energy + half_window, 1800)
        exact = np.interp(energy, part3_energy, part3_transmission)
        ax.plot(energy, exact, "k", lw=1.2, label="Hermitian method")
        peak_value = float(np.interp(peak_energy, part3_energy, part3_transmission))
        ax.scatter(
            [peak_energy], [peak_value], color="red", s=24, zorder=6,
            label=rf"$\theta=0.00$, $E_{resonance_number}={peak_energy:.6f}$",
        )
        for slot, theta in enumerate(angles[1:], start=1):
            pole = nearest_track_value(track, theta)
            profile = breit_wigner(energy, pole)
            ax.plot(
                energy,
                profile,
                linestyle=":",
                color=BW_COLORS[(slot - 1) % len(BW_COLORS)],
                lw=1.5,
                label=rf"$\theta={theta:.2f}$, $E_{resonance_number}={format_pole(pole, 6)}$",
            )
        ax.set(
            xlabel=r"Re($E$)", ylabel=r"$|T(E)|^2$", ylim=(-0.04, 1.05), title=title
        )
        ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    path = output / "part6_breit_wigner_zoom.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def continuum_diagnostics(
    spectra: list[ComplexSpectrum], tracks: list[ResonanceTrack]
) -> list[dict]:
    result: list[dict] = []
    poles = np.asarray([track.pole for track in tracks], dtype=np.complex128)
    for spectrum in spectra:
        values = plot_window(spectrum.eigenvalues, (0.08, 3.0), (-2.0, 0.05))
        if poles.size:
            keep = np.min(np.abs(values[:, None] - poles[None, :]), axis=1) > 0.025
            values = values[keep]
        ray_distance = distance_from_continuum_ray(values, spectrum.theta)
        values = values[ray_distance < 0.035]
        if values.size:
            slope = float(np.dot(values.real, values.imag) / np.dot(values.real, values.real))
            fitted_angle = float(np.arctan(-slope))
        else:
            slope = float("nan")
            fitted_angle = float("nan")
        result.append({
            "theta": spectrum.theta,
            "expected_continuum_angle": 2.0 * spectrum.theta,
            "fitted_continuum_angle": fitted_angle,
            "angle_error": fitted_angle - 2.0 * spectrum.theta,
            "fitted_slope_imag_over_real": slope,
            "expected_slope": -float(np.tan(2.0 * spectrum.theta)),
            "points_used": int(values.size),
        })
    return result


def comparison_diagnostics(
    tracks: list[ResonanceTrack], peaks: dict[str, float],
    part3_energy: np.ndarray, part3_transmission: np.ndarray,
) -> list[dict]:
    comparisons: list[dict] = []
    for index, track in enumerate(tracks[:2], start=1):
        peak = peaks[f"E{index}"]
        half_width = -track.pole.imag
        window = max(8.0 * half_width, 0.00045 if index == 1 else 0.12)
        energy = np.linspace(peak - window, peak + window, 2200)
        exact = np.interp(energy, part3_energy, part3_transmission)
        approximation = breit_wigner(energy, track.pole)
        comparisons.append({
            "label": track.label,
            "part3_peak_energy": peak,
            "complex_pole_real": track.pole.real,
            "peak_minus_pole_real": peak - track.pole.real,
            "Gamma_minus_2Im": track.width,
            "peak_offset_in_units_of_Gamma": (peak - track.pole.real) / track.width,
            "breit_wigner_rmse_in_comparison_window": float(
                np.sqrt(np.mean((exact - approximation) ** 2))
            ),
            "comparison_half_window": window,
        })
    return comparisons


def bound_state_diagnostics(spectra: list[ComplexSpectrum]) -> dict:
    """Audit invariance of the negative bound state under complex rotation."""
    values = np.asarray([
        spectrum.eigenvalues[
            int(np.argmin(np.abs(spectrum.eigenvalues - REFERENCE_BOUND_ENERGY)))
        ]
        for spectrum in spectra
    ], dtype=np.complex128)
    real_spread = float(np.ptp(values.real))
    maximum_imaginary = float(np.max(np.abs(values.imag)))
    maximum_reference_error = float(np.max(np.abs(values - REFERENCE_BOUND_ENERGY)))
    return {
        "reference_energy": REFERENCE_BOUND_ENERGY,
        "theta_values": [spectrum.theta for spectrum in spectra],
        "tracked_values": [
            {"real": value.real, "imaginary": value.imag} for value in values
        ],
        "real_part_spread": real_spread,
        "maximum_absolute_imaginary_part": maximum_imaginary,
        "maximum_absolute_error_from_reference": maximum_reference_error,
        "acceptance": {
            "real_part_spread_below_1e-8": real_spread < 1.0e-8,
            "maximum_abs_imag_below_1e-8": maximum_imaginary < 1.0e-8,
            "maximum_reference_error_below_1e-6": maximum_reference_error < 1.0e-6,
        },
    }


def reference_pole_diagnostics(tracks: list[ResonanceTrack]) -> dict:
    comparisons = []
    for index, (track, reference) in enumerate(zip(tracks, REFERENCE_POLES), start=1):
        error = abs(track.pole - reference)
        comparisons.append({
            "label": f"E{index}",
            "reference": {"real": reference.real, "imaginary": reference.imag},
            "computed": {"real": track.pole.real, "imaginary": track.pole.imag},
            "absolute_complex_error": float(error),
        })
    count_matches = len(tracks) == len(REFERENCE_POLES)
    maximum_error = max((item["absolute_complex_error"] for item in comparisons), default=float("inf"))
    return {
        "reference_angle": 0.25,
        "expected_pole_count": len(REFERENCE_POLES),
        "computed_pole_count": len(tracks),
        "acceptance_count_equals_five": count_matches,
        "comparisons": comparisons,
        "maximum_absolute_complex_error": maximum_error,
        "acceptance_maximum_error_below_1e-5": (
            count_matches and maximum_error < 1.0e-5
        ),
    }


def breit_wigner_identity_diagnostics(tracks: list[ResonanceTrack]) -> dict:
    """Check unit height and half height one half-width from every pole."""
    checks = []
    for track in tracks:
        half_width = -track.pole.imag
        values = breit_wigner(
            np.asarray([track.pole.real, track.pole.real + half_width]), track.pole
        )
        checks.append({
            "label": track.label,
            "center_value": float(values[0]),
            "positive_half_width_value": float(values[1]),
            "center_error_from_one": float(abs(values[0] - 1.0)),
            "half_width_error_from_half": float(abs(values[1] - 0.5)),
        })
    maximum_center_error = max(item["center_error_from_one"] for item in checks)
    maximum_half_error = max(item["half_width_error_from_half"] for item in checks)
    return {
        "checks": checks,
        "maximum_center_error_from_one": maximum_center_error,
        "maximum_half_width_error_from_half": maximum_half_error,
        "acceptance_errors_below_1e-12": (
            maximum_center_error < 1.0e-12 and maximum_half_error < 1.0e-12
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    output: Path,
    profile: str,
    config: ComplexScalingConfig,
    artifact_paths: list[Path],
) -> Path:
    """Describe only artifacts produced by the current invocation.

    This keeps ``--only`` runs honest when their output directory already
    contains figures or tables from an earlier, broader invocation.
    """
    files = []
    unique_paths = sorted({path.resolve() for path in artifact_paths}, key=lambda path: path.name)
    for path in unique_paths:
        if path.parent != output.resolve():
            raise ValueError(f"manifest artifact lies outside output directory: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"manifest artifact was not generated: {path}")
        files.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "part": 6,
        "profile": profile,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "files": files,
    }
    path = output / "part6_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 6 non-Hermitian complex scaling and Breit-Wigner resonances"
    )
    parser.add_argument("--quick", action="store_true", help="J=400 validation profile")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only", choices=("all", "spectra", "breit-wigner"), default="all",
        help="limit figure generation; spectra are always solved and audited",
    )
    parser.add_argument(
        "--angles", type=parse_angles, default=DEFAULT_ANGLES,
        help="comma-separated rotation angles (default: 0,0.05,...,0.25)",
    )
    parser.add_argument("--basis-size", type=int, help="override profile J")
    parser.add_argument("--box-length", type=float, help="override profile L")
    parser.add_argument("--quadrature-points", type=int, help="override profile N")
    args = parser.parse_args()

    started = time.perf_counter()
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
    spectra, angle_runtimes = solve_family(angles, config)
    tracks = identify_resonance_tracks(spectra)
    if len(tracks) != len(REFERENCE_POLES):
        raise RuntimeError(
            f"expected exactly {len(REFERENCE_POLES)} resonance poles, found {len(tracks)}"
        )
    print(
        "identified poles: "
        + "; ".join(
            f"{track.label}={format_pole(track.pole, 8)} ({track.status})" for track in tracks
        )
    )

    peaks = read_part3_peaks()

    generated_figures: list[str] = []
    if args.only in ("all", "spectra"):
        generated_figures.append(plot_combined_spectrum(output, spectra, config).name)
        generated_figures.append(plot_individual_spectra(output, spectra, config).name)

    part3_energy: np.ndarray | None = None
    part3_transmission: np.ndarray | None = None
    part3_source = "not loaded (--only spectra)"
    comparisons: list[dict] = []
    if args.only in ("all", "breit-wigner"):
        part3_energy, part3_transmission, part3_source = load_part3_transmission()
        generated_figures.append(
            plot_all_breit_wigner(
                output, spectra, tracks, part3_energy, part3_transmission, peaks
            ).name
        )
        zoom_path = plot_breit_wigner_zoom(
            output, angles, tracks, part3_energy, part3_transmission, peaks
        )
        generated_figures.append(zoom_path.name)
        comparisons = comparison_diagnostics(
            tracks, peaks, part3_energy, part3_transmission
        )

    total_runtime = time.perf_counter() - started
    bound_audit = bound_state_diagnostics(spectra)
    pole_audit = reference_pole_diagnostics(tracks)
    breit_wigner_audit = breit_wigner_identity_diagnostics(tracks)
    bound_acceptance = bound_audit["acceptance"]
    automatic_acceptance = {
        "bound_state_invariance": all(bound_acceptance.values()),
        "five_reference_poles": pole_audit["acceptance_maximum_error_below_1e-5"],
        "breit_wigner_center_and_half_width": (
            breit_wigner_audit["acceptance_errors_below_1e-12"]
        ),
    }
    automatic_acceptance["all_passed"] = all(automatic_acceptance.values())
    diagnostics = {
        "method": {
            "name": "analytic uniform complex-coordinate scaling in a sine box basis",
            "hamiltonian": "H(theta)=exp(-2i theta) T + V(x exp(i theta))",
            "potential": "V(z)=(z^2/2-0.8) exp(-z^2/10)",
            "matrix_structure": "complex symmetric, not Hermitian",
            "breit_wigner": "(-Im E_pole)^2 / ((E-Re E_pole)^2+(-Im E_pole)^2)",
            "width_convention": "E_pole=E_r-i Gamma/2",
        },
        "profile": "quick" if args.quick else "teacher_reference",
        "config": asdict(config),
        "angles": angles,
        "plot_window": {"real": REAL_LIMITS, "imaginary": IMAG_LIMITS},
        "part3_source": part3_source,
        "part3_peaks": peaks,
        "resonance_selection": {
            "reference_angle": spectra[-1].theta,
            "continuum_clearance_threshold": 0.04,
            "matching_tolerance": 0.012,
            "exposure_condition": "2 theta >= abs(arg(E_pole)) + 0.05",
            "stable_requires_at_least_angles": 2,
            "note": "a deepest pole exposed only at theta=0.25 is retained as provisional",
        },
        "resonances": [
            {
                "label": track.label,
                "status": track.status,
                "pole": {"real": track.pole.real, "imaginary": track.pole.imag},
                "Gamma": track.width,
                "tracked_thetas": track.tracked_thetas.tolist(),
                "tracked_values": [
                    {"real": value.real, "imaginary": value.imag}
                    for value in track.tracked_values
                ],
                "max_track_deviation": (
                    float(np.max(track.distances_from_reference))
                    if track.distances_from_reference.size else None
                ),
                "continuum_clearance_at_max_theta": track.continuum_clearance,
            }
            for track in tracks
        ],
        "validation": {
            "theta_zero_max_abs_imaginary_eigenvalue": float(
                np.max(np.abs(spectra[0].eigenvalues.imag))
            ),
            "maximum_complex_symmetry_relative_error": float(
                max(item.complex_symmetry_error for item in spectra)
            ),
            "maximum_odd_cosine_moment_absolute_value": float(
                max(item.odd_cosine_moment_max for item in spectra)
            ),
            "maximum_cross_parity_coupling_upper_bound": float(
                max(item.cross_parity_coupling_upper_bound for item in spectra)
            ),
            "parity_block_note": (
                "even analytic potential gives zero odd cosine moments; J is solved as "
                "independent odd-index and even-index sine blocks"
            ),
            "bound_state_invariance": bound_audit,
            "theta_0p25_reference_poles": pole_audit,
            "breit_wigner_identities": breit_wigner_audit,
            "automatic_acceptance": automatic_acceptance,
            "continuum_rotation": continuum_diagnostics(spectra, tracks),
            "part3_breit_wigner_comparison": comparisons,
        },
        "generated_figures": generated_figures,
        "runtime": {**angle_runtimes, "total_seconds": total_runtime},
    }
    (output / "part6_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    artifact_paths = [
        output / "part6_diagnostics.json",
        *(output / name for name in generated_figures),
    ]
    write_manifest(output, diagnostics["profile"], config, artifact_paths)
    print(f"Part 6 outputs: {output}")
    print(f"total runtime: {total_runtime:.2f} s")


if __name__ == "__main__":
    main()
