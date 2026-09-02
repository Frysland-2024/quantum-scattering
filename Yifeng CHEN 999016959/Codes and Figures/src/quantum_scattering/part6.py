"""Complex scaling and Breit--Wigner analysis for project Part 6.

The implementation follows Lecture 10 literally: the Hamiltonian is represented
in the sine-box basis and globally complex scaled,

    H(theta) = exp(-2 i theta) T + V(x exp(i theta)).

Because the assigned potential is even, odd and even sine-basis indices form two
independent complex-symmetric blocks.  Diagonalising the two J/2 blocks is both
mathematically exact (up to the quadrature) and much less expensive than treating
one dense J by J matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.linalg import eigvals

from .potential import double_barrier_potential
from .stationary import (
    FINITE_DIFFERENCE_DX,
    FINITE_DIFFERENCE_X_MAX,
    FINITE_DIFFERENCE_X_MIN,
    solve_resonance_peaks,
    transmission_spectrum,
)


THETA_VALUES: tuple[float, ...] = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
TEACHER_POLE_REFERENCES: tuple[complex, ...] = (
    0.620970951 - 0.000058267j,
    1.327197073 - 0.015447319j,
    1.784582869 - 0.173750710j,
    2.124421892 - 0.564794982j,
    2.455486272 - 1.111531572j,
)

_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
_MARKERS = ("o", "s", "^", "D", "v", "*")
_BW_COLORS = ("red", "limegreen", "dodgerblue", "darkorange", "magenta")
_ORANGE = "#e66b00"


@dataclass(frozen=True)
class ComplexSpectrum:
    """Complex eigenvalues at one scaling angle."""

    theta: float
    length: float
    basis_size: int
    grid_points: int
    eigenvalues: NDArray[np.complex128]
    parity: tuple[str, ...]
    maximum_odd_moment: float
    cross_parity_coupling_bound: float
    hamiltonian_symmetry_error: float


@dataclass(frozen=True)
class ResonancePole:
    """A resonance exposed above the rotated continuum at theta_max."""

    number: int
    energy: complex
    gamma: float
    first_exposed_theta: float
    supporting_angles: tuple[float, ...]
    maximum_angle_spread: float


def complex_scaled_potential(
    x: NDArray[np.float64], theta: float
) -> NDArray[np.complex128]:
    """Evaluate V(x exp(i theta)) for the assigned entire potential."""

    z = np.asarray(x, dtype=float) * np.exp(1j * float(theta))
    return np.asarray((0.5 * z**2 - 0.8) * np.exp(-0.1 * z**2), dtype=np.complex128)


def _complex_cosine_moments(
    *,
    theta: float,
    length: float,
    basis_size: int,
    grid_points: int,
) -> tuple[NDArray[np.complex128], float]:
    """Return trapezoidal cosine moments through a complex DCT-I."""

    if grid_points <= 2 * basis_size:
        raise ValueError("grid_points must exceed twice the basis size")
    x = np.linspace(-length / 2.0, length / 2.0, grid_points)
    values = complex_scaled_potential(x, theta)
    spacing = length / (grid_points - 1)
    moments = np.asarray(0.5 * spacing * dct(values, type=1), dtype=np.complex128)
    moments = moments[: 2 * basis_size + 1]
    maximum_odd = float(np.max(np.abs(moments[1::2])))
    # V(x exp(i theta)) remains even.  The exact odd moments vanish; setting
    # their roundoff remnants to zero makes the parity separation explicit.
    moments[1::2] = 0.0
    return moments, maximum_odd


def _complex_hamiltonian_block(
    basis_indices: NDArray[np.int64],
    moments: NDArray[np.complex128],
    *,
    theta: float,
    length: float,
) -> NDArray[np.complex128]:
    differences = np.abs(basis_indices[:, None] - basis_indices[None, :])
    sums = basis_indices[:, None] + basis_indices[None, :]
    hamiltonian = np.asarray((moments[differences] - moments[sums]) / length)
    kinetic = 0.5 * (np.pi * basis_indices / length) ** 2 * np.exp(-2j * theta)
    hamiltonian[np.diag_indices_from(hamiltonian)] += kinetic
    return hamiltonian


def solve_complex_scaled_spectrum(
    theta: float,
    *,
    length: float = 200.0,
    basis_size: int = 2000,
    grid_points: int = 10000,
) -> ComplexSpectrum:
    """Diagonalise the Lecture-10 complex-scaled sine-box Hamiltonian.

    The returned array contains all J eigenvalues.  Evenness of the potential
    permits two J/2 dense diagonalizations rather than one J dense solve.
    """

    if not 0.0 <= theta < np.pi / 4.0:
        raise ValueError("theta must satisfy 0 <= theta < pi/4")
    if length <= 0 or basis_size < 4 or grid_points < 3:
        raise ValueError("length, basis_size, and grid_points must be positive")

    moments, maximum_odd = _complex_cosine_moments(
        theta=theta,
        length=length,
        basis_size=basis_size,
        grid_points=grid_points,
    )
    all_values: list[NDArray[np.complex128]] = []
    all_parity: list[str] = []
    maximum_symmetry_error = 0.0
    indices = np.arange(1, basis_size + 1, dtype=np.int64)
    for selector, parity_label in ((indices % 2 == 1, "even"), (indices % 2 == 0, "odd")):
        block_indices = indices[selector]
        hamiltonian = _complex_hamiltonian_block(
            block_indices,
            moments,
            theta=theta,
            length=length,
        )
        maximum_symmetry_error = max(
            maximum_symmetry_error,
            float(np.max(np.abs(hamiltonian - hamiltonian.T))),
        )
        values = np.asarray(
            eigvals(hamiltonian, overwrite_a=True, check_finite=False),
            dtype=np.complex128,
        )
        if theta == 0.0:
            values = values.real.astype(np.complex128)
        all_values.append(values)
        all_parity.extend([parity_label] * values.size)

    eigenvalues = np.concatenate(all_values)
    parity = np.asarray(all_parity, dtype=object)
    order = np.lexsort((eigenvalues.imag, eigenvalues.real))
    eigenvalues = eigenvalues[order]
    parity = parity[order]
    return ComplexSpectrum(
        theta=float(theta),
        length=float(length),
        basis_size=int(basis_size),
        grid_points=int(grid_points),
        eigenvalues=eigenvalues,
        parity=tuple(str(value) for value in parity),
        maximum_odd_moment=maximum_odd,
        cross_parity_coupling_bound=2.0 * maximum_odd / length,
        hamiltonian_symmetry_error=maximum_symmetry_error,
    )


def breit_wigner(energy: NDArray[np.float64], pole: complex) -> NDArray[np.float64]:
    r"""Return (Gamma/2)^2 / ((E-E_r)^2 + (Gamma/2)^2)."""

    values = np.asarray(energy, dtype=float)
    half_gamma = -float(np.imag(pole))
    if half_gamma <= 0:
        raise ValueError("a resonance pole must have a negative imaginary part")
    return half_gamma**2 / ((values - float(np.real(pole))) ** 2 + half_gamma**2)


def identify_resonance_poles(
    spectra: Iterable[ComplexSpectrum],
    *,
    exposure_margin: float = 0.05,
    matching_tolerance: float = 2.0e-3,
) -> tuple[ResonancePole, ...]:
    """Identify poles separated from the continuum ray at the largest theta.

    Multiplication by exp(2 i theta) rotates the free continuum back to the
    real axis.  Resonances exposed by complex scaling then have a positive
    rotated imaginary part.  Candidates are separated at the largest angle;
    their supporting angles obey 2 theta >= |arg(E)| + 0.05, the geometric
    exposure condition used in the Lecture-10 interpretation.
    """

    ordered = sorted(spectra, key=lambda item: item.theta)
    nonzero = [item for item in ordered if item.theta > 0]
    if not nonzero:
        raise ValueError("at least one non-zero theta spectrum is required")
    final = nonzero[-1]
    values = final.eigenvalues
    rotated_imag = np.imag(values * np.exp(2j * final.theta))
    mask = (
        (values.real > 0.0)
        & (values.real <= 3.0)
        & (values.imag < 0.0)
        & (rotated_imag > exposure_margin)
    )
    candidates = values[mask]
    candidates = candidates[np.argsort(candidates.real)]

    poles: list[ResonancePole] = []
    for number, candidate in enumerate(candidates, start=1):
        support: list[tuple[float, complex]] = []
        for spectrum in nonzero:
            index = int(np.argmin(np.abs(spectrum.eigenvalues - candidate)))
            match = complex(spectrum.eigenvalues[index])
            is_exposed = 2.0 * spectrum.theta >= abs(float(np.angle(match))) + 0.05
            if abs(match - candidate) <= matching_tolerance and is_exposed:
                support.append((spectrum.theta, match))
        if not support:
            support = [(final.theta, complex(candidate))]
        spread = max(abs(value - candidate) for _, value in support)
        poles.append(
            ResonancePole(
                number=number,
                energy=complex(candidate),
                gamma=-2.0 * float(candidate.imag),
                first_exposed_theta=float(support[0][0]),
                supporting_angles=tuple(float(theta) for theta, _ in support),
                maximum_angle_spread=float(spread),
            )
        )
    return tuple(poles)


def _profile_parameters(profile: str) -> tuple[int, float, int]:
    if profile == "quick":
        return 400, 200.0, 4001
    if profile in {"reference", "full"}:
        # Both report-producing modes retain the exact parameters shown by the
        # teacher.  Quick exists only for inexpensive development checks.
        return 2000, 200.0, 10000
    raise ValueError("profile must be quick, reference, or full")


def _display_values(spectrum: ComplexSpectrum) -> NDArray[np.complex128]:
    values = spectrum.eigenvalues
    return values[
        (values.real >= -0.35)
        & (values.real <= 3.0)
        & (values.imag >= -2.0)
        & (values.imag <= 0.1)
    ]


def _plot_spectrum(
    ax: plt.Axes,
    spectrum: ComplexSpectrum,
    *,
    color: str,
    marker: str,
    label: str | None = None,
    markersize: float = 5.5,
) -> None:
    values = _display_values(spectrum)
    kwargs: dict[str, object] = {
        "linestyle": "none",
        "marker": marker,
        "markersize": markersize,
        "markeredgewidth": 1.1,
        "color": color,
        "label": label,
    }
    if marker != "*":
        kwargs["markerfacecolor"] = "none"
    ax.plot(values.real, values.imag, **kwargs)


def _format_complex_axis(ax: plt.Axes, *, lower: float = -2.0) -> None:
    ax.axhline(0.0, color="black", lw=0.8)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlim(-0.35, 3.0)
    ax.set_ylim(lower, 0.1)
    ax.set_xlabel(r"Re$(E)$")
    ax.set_ylabel(r"Im$(E)$")
    ax.grid(False)


def _outer_heading(fig: plt.Figure, text: str) -> None:
    fig.text(0.01, 0.995, text, ha="left", va="top", fontsize=18, fontweight="bold", color=_ORANGE)


def _save(fig: plt.Figure, path: Path, *, dpi: int = 150) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _fwhm(energy: NDArray[np.float64], transmission: NDArray[np.float64], peak: float) -> float:
    order = np.argsort(energy)
    x = energy[order]
    y = transmission[order]
    peak_index = int(np.argmin(np.abs(x - peak)))
    half = 0.5 * float(y[peak_index])
    left_candidates = np.flatnonzero(y[:peak_index] <= half)
    right_candidates = np.flatnonzero(y[peak_index + 1 :] <= half)
    if left_candidates.size == 0 or right_candidates.size == 0:
        return float("nan")
    left = int(left_candidates[-1])
    right = int(peak_index + 1 + right_candidates[0])
    left_cross = float(np.interp(half, [y[left], y[left + 1]], [x[left], x[left + 1]]))
    right_cross = float(np.interp(half, [y[right], y[right - 1]], [x[right], x[right - 1]]))
    return right_cross - left_cross


def generate_part6(
    root: Path | str = Path("output"),
    *,
    profile: str = "reference",
) -> list[Path]:
    """Generate the four teacher-style Part-6 figures and audit files."""

    basis_size, length, grid_points = _profile_parameters(profile)
    output = Path(root) / "part6_complex_scaling"
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    spectra = [
        solve_complex_scaled_spectrum(
            theta,
            length=length,
            basis_size=basis_size,
            grid_points=grid_points,
        )
        for theta in THETA_VALUES
    ]
    poles = identify_resonance_poles(spectra)
    if len(poles) != 5:
        raise RuntimeError(f"expected five exposed resonance poles, found {len(poles)}")

    # 1. All complex spectra in one panel.
    fig, ax = plt.subplots(figsize=(14.8, 7.4))
    for spectrum, color, marker in zip(spectra, _COLORS, _MARKERS, strict=True):
        _plot_spectrum(
            ax,
            spectrum,
            color=color,
            marker=marker,
            label=rf"$\theta={spectrum.theta:.2f}$",
        )
    _format_complex_axis(ax)
    ax.set_title(rf"Complex-scaling spectra for multiple $\theta$, $J={basis_size}$", fontsize=15)
    ax.legend(loc="lower left", frameon=True)
    _outer_heading(fig, "complex eigenvalues (plotted in one picture for various values of theta)")
    fig.subplots_adjust(top=0.88)
    created.append(_save(fig, output / "complex_eigenvalues_all.png"))

    # 2. One panel per theta.
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=True, sharey=True)
    for ax, spectrum, color, marker in zip(axes.flat, spectra, _COLORS, _MARKERS, strict=True):
        _plot_spectrum(ax, spectrum, color=color, marker=marker, markersize=4.5)
        _format_complex_axis(ax)
        ax.set_title(rf"$\theta={spectrum.theta:.2f}$ rad", fontsize=11)
    fig.suptitle(rf"Complex-scaling individual spectra, $J={basis_size}$", y=0.94, fontsize=15)
    _outer_heading(fig, "complex eigenvalues (separately for each theta)")
    fig.subplots_adjust(top=0.86, hspace=0.22, wspace=0.22)
    created.append(_save(fig, output / "complex_eigenvalues_by_theta.png"))

    # Part-3 Hermitian transmission is the independent comparison curve.
    part3_peaks = solve_resonance_peaks()
    first_grid = np.linspace(0.62045, 0.62150, 1601)
    second_grid = np.linspace(1.20, 1.48, 1601)
    overview_grid = np.linspace(0.002, 3.0, 1601)
    energy = np.unique(
        np.concatenate(
            (
                overview_grid,
                first_grid,
                second_grid,
                [part3_peaks.first.energy, part3_peaks.second.energy],
                [pole.energy.real for pole in poles],
            )
        )
    )
    hermitian = transmission_spectrum(
        energy,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
        dx=FINITE_DIFFERENCE_DX,
    )
    transmission = np.asarray(hermitian.transmission, dtype=float)
    bw_profiles = [breit_wigner(energy, pole.energy) for pole in poles]

    # 3. Stacked Breit-Wigner comparison and complex spectra.
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(14.8, 8.4),
        gridspec_kw={"height_ratios": (1.0, 1.0), "hspace": 0.12},
    )
    top.plot(energy, np.clip(transmission, 0.0, 1.02), color="black", lw=1.2, label="Hermitian QM Transmission profile")
    for pole, values, color in zip(poles, bw_profiles, _BW_COLORS, strict=True):
        top.plot(
            energy,
            values,
            color=color,
            ls=":",
            lw=1.5,
            label=rf"Breit-Wigner at $E={pole.energy.real:.5f}{pole.energy.imag:+.5f}i$",
        )
    for peak in (part3_peaks.first, part3_peaks.second):
        top.scatter([peak.energy], [min(peak.transmission, 1.0)], color="red", s=18, zorder=4)
        top.annotate(
            f"E={peak.energy:.6f}\nT={peak.transmission:.6f}",
            (peak.energy, min(peak.transmission, 1.0)),
            xytext=(7, -2),
            textcoords="offset points",
            color="red",
            fontsize=9,
            va="top",
        )
    top.set_xlim(0.0, 3.0)
    top.set_ylim(-0.05, 1.05)
    top.set_ylabel(r"$|T(E)|^2$")
    top.set_title("Breit-Wigner Transmission Approximation based on non-Hermitian Complex-scaling")
    top.grid(False)
    top.legend(loc="lower left", fontsize=7)

    for spectrum, color, marker in zip(spectra, _COLORS, _MARKERS, strict=True):
        _plot_spectrum(
            bottom,
            spectrum,
            color=color,
            marker=marker,
            label=rf"$\theta={spectrum.theta:.2f}$",
            markersize=3.5,
        )
    bottom.scatter(
        [pole.energy.real for pole in poles],
        [pole.energy.imag for pole in poles],
        color="red",
        s=18,
        zorder=5,
        label="Breit-Wigner selected",
    )
    _format_complex_axis(bottom, lower=-1.75)
    bottom.legend(loc="lower left", fontsize=7)
    _outer_heading(fig, "Breit-Wigner profiles (all)")
    fig.subplots_adjust(top=0.90)
    created.append(_save(fig, output / "breit_wigner_profiles_all.png"))

    # 4. Teacher-style detailed zooms for the two physical transmission peaks.
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.4))
    zoom_specs = (
        (axes[0], first_grid[0], first_grid[-1], poles[0], part3_peaks.first, "Transmission profile around first resonance peak"),
        (axes[1], second_grid[0], second_grid[-1], poles[1], part3_peaks.second, "Around the second resonance peak"),
    )
    for resonance_number, (ax, lower, upper, pole, peak, title) in enumerate(zoom_specs, start=1):
        mask = (energy >= lower) & (energy <= upper)
        ax.plot(energy[mask], np.clip(transmission[mask], 0.0, 1.02), color="black", lw=1.4, label="Hermitian method")
        ax.scatter(
            [peak.energy],
            [min(peak.transmission, 1.0)],
            color="red",
            s=28,
            zorder=5,
            label=rf"$\theta=0.00$, $E_{resonance_number}={peak.energy:.6f}$",
        )
        for spectrum, color in zip(spectra[1:], _BW_COLORS, strict=True):
            index = int(np.argmin(np.abs(spectrum.eigenvalues - pole.energy)))
            theta_pole = complex(spectrum.eigenvalues[index])
            ax.plot(
                energy[mask],
                breit_wigner(energy[mask], theta_pole),
                color=color,
                ls=":",
                lw=1.5,
                label=rf"$\theta={spectrum.theta:.2f}$, $E_{resonance_number}={theta_pole.real:.6f}{theta_pole.imag:+.6f}i$",
            )
        ax.set_xlim(lower, upper)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r"Re$(E)$")
        ax.set_ylabel(r"$|T(E)|^2$")
        ax.set_title(title)
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=7)
    _outer_heading(fig, "Breit-Wigner profiles (detailed zoom)")
    fig.subplots_adjust(top=0.88, wspace=0.22)
    created.append(_save(fig, output / "breit_wigner_profiles_zoom.png"))

    continuum_checks: list[dict[str, object]] = []
    for spectrum in spectra:
        values = _display_values(spectrum)
        mask = (values.real >= 2.5) & (values.real <= 3.0)
        high = values[mask]
        if spectrum.theta == 0.0 or high.size < 3:
            fitted_slope = 0.0
            expected_slope = -float(np.tan(2.0 * spectrum.theta))
            slope_error = abs(fitted_slope - expected_slope)
        else:
            fitted_slope = float(np.dot(high.real, high.imag) / np.dot(high.real, high.real))
            expected_slope = -float(np.tan(2.0 * spectrum.theta))
            slope_error = abs(fitted_slope - expected_slope)
        continuum_checks.append(
            {
                "theta": spectrum.theta,
                "expected_slope": expected_slope,
                "fitted_high_energy_slope": fitted_slope,
                "absolute_slope_error": slope_error,
                "displayed_eigenvalue_count": int(values.size),
            }
        )

    bound_states = []
    for spectrum in spectra:
        negative = spectrum.eigenvalues[spectrum.eigenvalues.real < 0]
        bound_states.append(complex(negative[np.argmax(negative.real)]))
    bound_reference = bound_states[0]
    first_mask = (energy >= first_grid[0]) & (energy <= first_grid[-1])
    second_mask = (energy >= second_grid[0]) & (energy <= second_grid[-1])
    first_bw = breit_wigner(energy[first_mask], poles[0].energy)
    second_bw = breit_wigner(energy[second_mask], poles[1].energy)
    diagnostics = {
        "method": "Lecture-10 sine-box basis with global complex scaling and parity-block diagonalization",
        "complex_scaled_hamiltonian": "H(theta)=exp(-2 i theta) T + V(x exp(i theta))",
        "basis": "sqrt(2/L) sin(j*pi*(x+L/2)/L), j=1..J",
        "potential_matrix": "V_ij=(C_|i-j|-C_i+j)/L from a complex trapezoidal DCT-I",
        "profile": profile,
        "parameters": {"J": basis_size, "L": length, "N": grid_points, "theta": list(THETA_VALUES)},
        "parity_block_sizes": [int((basis_size + 1) // 2), int(basis_size // 2)],
        "spectra": [
            {
                "theta": spectrum.theta,
                "eigenvalue_count": int(spectrum.eigenvalues.size),
                "maximum_odd_cosine_moment": spectrum.maximum_odd_moment,
                "cross_parity_coupling_bound": spectrum.cross_parity_coupling_bound,
                "hamiltonian_symmetry_error": spectrum.hamiltonian_symmetry_error,
            }
            for spectrum in spectra
        ],
        "continuum_rotation_checks": continuum_checks,
        "bound_state": {
            "values": [[value.real, value.imag] for value in bound_states],
            "maximum_theta_spread": float(max(abs(value - bound_reference) for value in bound_states)),
        },
        "resonance_poles": [
            {
                "number": pole.number,
                "real": pole.energy.real,
                "imag": pole.energy.imag,
                "gamma": pole.gamma,
                "first_exposed_theta": pole.first_exposed_theta,
                "supporting_angles": list(pole.supporting_angles),
                "maximum_angle_spread": pole.maximum_angle_spread,
                "teacher_reference": [reference.real, reference.imag],
                "absolute_reference_error": abs(pole.energy - reference),
                "multi_angle_stable": len(pole.supporting_angles) >= 2,
            }
            for pole, reference in zip(poles, TEACHER_POLE_REFERENCES, strict=True)
        ],
        "part3_cross_check": {
            "method": "three-point backward finite difference",
            "dx": FINITE_DIFFERENCE_DX,
            "first_peak_energy": part3_peaks.first.energy,
            "second_peak_energy": part3_peaks.second.energy,
            "pole_minus_part3_peak": [
                poles[0].energy.real - part3_peaks.first.energy,
                poles[1].energy.real - part3_peaks.second.energy,
            ],
            "first_numerical_fwhm": _fwhm(energy[first_mask], transmission[first_mask], part3_peaks.first.energy),
            "second_numerical_fwhm": _fwhm(energy[second_mask], transmission[second_mask], part3_peaks.second.energy),
            "first_breit_wigner_gamma": poles[0].gamma,
            "second_breit_wigner_gamma": poles[1].gamma,
            "first_zoom_rmse": float(np.sqrt(np.mean((transmission[first_mask] - first_bw) ** 2))),
            "second_zoom_rmse": float(np.sqrt(np.mean((transmission[second_mask] - second_bw) ** 2))),
            "maximum_unitarity_error": float(np.max(hermitian.unitarity_error)),
        },
        "breit_wigner_formula_checks": [
            {
                "resonance": pole.number,
                "at_center": float(breit_wigner(np.array([pole.energy.real]), pole.energy)[0]),
                "at_minus_gamma_over_2": float(
                    breit_wigner(np.array([pole.energy.real - pole.gamma / 2.0]), pole.energy)[0]
                ),
                "at_plus_gamma_over_2": float(
                    breit_wigner(np.array([pole.energy.real + pole.gamma / 2.0]), pole.energy)[0]
                ),
            }
            for pole in poles
        ],
        "selection_note": "Candidates are separated above the back-rotated continuum at theta=0.25; angle support requires 2 theta >= |arg(E)| + 0.05. Teacher references are used only for audit errors.",
        "fifth_pole_note": "The broad fifth pole is first exposed only at theta=0.25, so the requested angle set provides single-angle rather than multi-angle stability evidence.",
    }
    diagnostics_path = output / "part6_diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    created.append(diagnostics_path)

    manifest_path = output / "part6_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "profile": profile,
                "parameters": {"J": basis_size, "L": length, "N": grid_points},
                "theta": list(THETA_VALUES),
                "static_figure_count": 4,
                "resonance_count": len(poles),
                "artifacts": [path.name for path in created],
                "diagnostics": diagnostics_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    created.append(manifest_path)
    return created


__all__ = [
    "ComplexSpectrum",
    "ResonancePole",
    "TEACHER_POLE_REFERENCES",
    "THETA_VALUES",
    "breit_wigner",
    "complex_scaled_potential",
    "generate_part6",
    "identify_resonance_poles",
    "solve_complex_scaled_spectrum",
]
