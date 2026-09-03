"""Resonance wavefunctions before and after complex-coordinate scaling.

The second positive-energy resonance is selected from the theta=0.1 spectrum.
Its even-parity solution is propagated from the origin.  Beyond |x|=15 the
potential is negligible, so the state is continued with its outgoing Siegert
form.  The same continuation on x exp(i theta) decays after complex scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .complex_scaling import solve_complex_scaled_spectrum


WAVEFUNCTION_THETA = 0.10
ASYMPTOTIC_MATCH = 15.0
SECOND_RESONANCE_WINDOW = (1.15, 1.50, -0.08, 0.0)
_ORANGE = "#e66b00"


@dataclass(frozen=True)
class ResonanceWavefunction:
    x: NDArray[np.float64]
    energy: complex
    unscaled: NDArray[np.complex128]
    scaled: NDArray[np.complex128]


def _profile_parameters(profile: str) -> tuple[int, float, int]:
    if profile == "quick":
        return 400, 200.0, 4001
    if profile in {"reference", "full"}:
        return 2000, 200.0, 10000
    raise ValueError("profile must be quick, reference, or full")


def _potential(z: NDArray[np.complex128] | complex) -> NDArray[np.complex128]:
    coordinate = np.asarray(z, dtype=np.complex128)
    return np.asarray(
        (0.5 * coordinate**2 - 0.8) * np.exp(-0.1 * coordinate**2),
        dtype=np.complex128,
    )


def _second_resonance(profile: str) -> complex:
    basis_size, length, grid_points = _profile_parameters(profile)
    spectrum = solve_complex_scaled_spectrum(
        WAVEFUNCTION_THETA,
        length=length,
        basis_size=basis_size,
        grid_points=grid_points,
    )
    values = np.asarray(spectrum.eigenvalues, dtype=np.complex128)
    r0, r1, i0, i1 = SECOND_RESONANCE_WINDOW
    candidates = values[
        (values.real >= r0)
        & (values.real <= r1)
        & (values.imag >= i0)
        & (values.imag < i1)
    ]
    if candidates.size != 1:
        raise RuntimeError(
            "the theta=0.1 spectrum must contain exactly one pole in the "
            f"second-resonance window; found {candidates.size}"
        )
    return complex(candidates[0])


def _outgoing_momentum(energy: complex) -> complex:
    momentum = complex(np.sqrt(2.0 * energy))
    if momentum.real < 0.0:
        momentum = -momentum
    if not (momentum.real > 0.0 and momentum.imag < 0.0):
        raise ValueError("the selected pole does not define an outgoing resonance momentum")
    return momentum


def _integrate_positive_branch(energy: complex, theta: float):
    rotation = np.exp(1j * theta)

    def equation(x: float, y: NDArray[np.complex128]) -> NDArray[np.complex128]:
        psi, derivative = y
        curvature = (
            2.0
            * np.exp(2j * theta)
            * (_potential(x * rotation) - energy)
            * psi
        )
        return np.asarray((derivative, curvature), dtype=np.complex128)

    result = solve_ivp(
        equation,
        (0.0, ASYMPTOTIC_MATCH),
        np.asarray((1.0 + 0.0j, 0.0 + 0.0j), dtype=np.complex128),
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.04,
        dense_output=True,
    )
    if not result.success or result.sol is None:
        raise RuntimeError(f"wavefunction integration failed: {result.message}")
    return result


def _prepare_resonance_state(energy: complex):
    momentum = _outgoing_momentum(energy)
    unscaled_solution = _integrate_positive_branch(energy, 0.0)
    boundary_value, boundary_derivative = unscaled_solution.sol(ASYMPTOTIC_MATCH)
    residual = abs(boundary_derivative - 1j * momentum * boundary_value)
    residual_scale = max(abs(momentum * boundary_value), np.finfo(float).tiny)
    if residual / residual_scale > 1.0e-5:
        raise RuntimeError("the selected energy fails the outgoing-boundary check")
    outgoing_amplitude = complex(boundary_value) * np.exp(
        -1j * momentum * ASYMPTOTIC_MATCH
    )
    scaled_solution = _integrate_positive_branch(energy, WAVEFUNCTION_THETA)
    return momentum, outgoing_amplitude, unscaled_solution, scaled_solution


def _evaluate_prepared_branch(
    x: NDArray[np.float64],
    *,
    theta: float,
    momentum: complex,
    outgoing_amplitude: complex,
    interior_solution,
) -> NDArray[np.complex128]:
    radius = np.abs(np.asarray(x, dtype=float))
    rotation = np.exp(1j * theta)
    values = np.empty(radius.shape, dtype=np.complex128)
    interior = radius <= ASYMPTOTIC_MATCH
    values[interior] = interior_solution.sol(radius[interior])[0]
    values[~interior] = outgoing_amplitude * np.exp(
        1j * momentum * radius[~interior] * rotation
    )
    return values


def _evaluate_wavefunction(
    x: NDArray[np.float64],
    energy: complex,
    prepared,
) -> ResonanceWavefunction:
    momentum, outgoing_amplitude, unscaled_solution, scaled_solution = prepared
    coordinate = np.asarray(x, dtype=float)
    return ResonanceWavefunction(
        x=coordinate,
        energy=energy,
        unscaled=_evaluate_prepared_branch(
            coordinate,
            theta=0.0,
            momentum=momentum,
            outgoing_amplitude=outgoing_amplitude,
            interior_solution=unscaled_solution,
        ),
        scaled=_evaluate_prepared_branch(
            coordinate,
            theta=WAVEFUNCTION_THETA,
            momentum=momentum,
            outgoing_amplitude=outgoing_amplitude,
            interior_solution=scaled_solution,
        ),
    )


def _outer_heading(fig: plt.Figure, text: str) -> None:
    fig.text(
        0.01,
        0.995,
        text,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color=_ORANGE,
    )


def _save_wavefunction_figure(
    result: ResonanceWavefunction,
    path: Path,
    *,
    heading: str,
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(14.8, 7.6), sharex=True)
    _outer_heading(fig, heading)
    fig.suptitle(
        rf"Resonance state at $E_{{res}}={result.energy.real:.6f}{result.energy.imag:+.6f}i$",
        fontsize=14,
        y=0.94,
    )

    axes[0].plot(result.x, result.unscaled.real, color="red", lw=1.2, label=r"$\psi(x)$")
    axes[0].plot(
        result.x,
        result.scaled.real,
        color="black",
        lw=0.95,
        label=rf"$\psi_\theta(x)$ ($\theta={WAVEFUNCTION_THETA:.1f}$)",
    )
    axes[1].plot(result.x, result.unscaled.imag, color="blue", lw=1.2, label=r"$\psi(x)$")
    axes[1].plot(
        result.x,
        result.scaled.imag,
        color="black",
        lw=0.95,
        label=rf"$\psi_\theta(x)$ ($\theta={WAVEFUNCTION_THETA:.1f}$)",
    )

    axes[0].set_ylabel(r"Re $\psi(x)$")
    axes[1].set_ylabel(r"Im $\psi(x)$")
    axes[1].set_xlabel(r"$x$")
    y_limit = 1.3 if float(np.max(np.abs(result.x))) <= 200.0 else 22.0
    for axis in axes:
        axis.axhline(0.0, color="black", lw=0.7)
        axis.set_ylim(-y_limit, y_limit)
        axis.grid(False)
        axis.legend(loc="upper right", fontsize=8.5)

    fig.subplots_adjust(top=0.88, hspace=0.08)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_resonance_wavefunctions(
    root: Path | str = Path("output"),
    *,
    profile: str = "reference",
) -> list[Path]:
    """Create the two Part-6 wavefunction figures supplied after the first four plots."""
    output = Path(root) / "part6_complex_scaling"
    output.mkdir(parents=True, exist_ok=True)

    energy = _second_resonance(profile)
    prepared = _prepare_resonance_state(energy)
    near_points = 8001 if profile == "quick" else 12001
    far_points = 12001 if profile == "quick" else 24001
    near = _evaluate_wavefunction(
        np.linspace(-200.0, 200.0, near_points), energy, prepared
    )
    far = _evaluate_wavefunction(
        np.linspace(-500.0, 500.0, far_points), energy, prepared
    )
    return [
        _save_wavefunction_figure(
            near,
            output / "resonance_wavefunction_near.png",
            heading="wavefunction - plot 1",
        ),
        _save_wavefunction_figure(
            far,
            output / "resonance_wavefunction_far.png",
            heading="wavefunction - plot 2",
        ),
    ]


__all__ = [
    "ResonanceWavefunction",
    "WAVEFUNCTION_THETA",
    "generate_resonance_wavefunctions",
]
