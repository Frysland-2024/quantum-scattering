from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from quantum_scattering.complex_scaling import ComplexScalingConfig, solve_complex_spectrum
from quantum_scattering.core import ensure_dir


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "part6"
SCALING_ANGLE = 0.10
MATCH_POINT = 15.0
SECOND_RESONANCE_WINDOW = (1.15, 1.50, -0.08, 0.0)
ORANGE = "#ef6c00"


def configuration(quick: bool) -> ComplexScalingConfig:
    if quick:
        return ComplexScalingConfig(400, 200.0, 2401)
    return ComplexScalingConfig(2000, 200.0, 10000)


def model_potential(z: np.ndarray | complex) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    return (0.5 * z**2 - 0.8) * np.exp(-0.1 * z**2)


def find_second_resonance(config: ComplexScalingConfig) -> complex:
    """Select the isolated pole around the second Part-3 transmission peak."""
    spectrum = solve_complex_spectrum(SCALING_ANGLE, config)
    values = np.asarray(spectrum.eigenvalues, dtype=np.complex128)
    real_min, real_max, imag_min, imag_max = SECOND_RESONANCE_WINDOW
    candidates = values[
        (values.real >= real_min)
        & (values.real <= real_max)
        & (values.imag >= imag_min)
        & (values.imag < imag_max)
    ]
    if candidates.size != 1:
        raise RuntimeError(
            "expected one second-resonance pole in "
            f"{SECOND_RESONANCE_WINDOW}, found {candidates.size}"
        )
    return complex(candidates[0])


def outgoing_momentum(energy: complex) -> complex:
    momentum = complex(np.sqrt(2.0 * energy))
    if momentum.real < 0.0:
        momentum = -momentum
    if not (momentum.real > 0.0 and momentum.imag < 0.0):
        raise ValueError("the selected pole does not define an outgoing resonance momentum")
    return momentum


def integrate_even_state(
    energy: complex,
    theta: float,
    match_point: float = MATCH_POINT,
):
    """Integrate the even resonance state from x=0 to the asymptotic region."""
    phase = np.exp(1j * theta)

    def rhs(x: float, state: np.ndarray) -> np.ndarray:
        psi, derivative = state
        z = x * phase
        second_derivative = (
            2.0 * np.exp(2j * theta) * (model_potential(z) - energy) * psi
        )
        return np.asarray((derivative, second_derivative), dtype=np.complex128)

    solution = solve_ivp(
        rhs,
        (0.0, match_point),
        np.asarray((1.0 + 0.0j, 0.0 + 0.0j)),
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
        max_step=0.04,
        dense_output=True,
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"resonance-state integration failed: {solution.message}")
    return solution


def prepare_resonance_state(energy: complex):
    """Prepare the unscaled and rotated branches once for both figures."""
    momentum = outgoing_momentum(energy)
    unscaled_solution = integrate_even_state(energy, 0.0)
    psi_at_match, derivative_at_match = unscaled_solution.sol(MATCH_POINT)
    outgoing_residual = abs(derivative_at_match - 1j * momentum * psi_at_match)
    outgoing_scale = max(abs(momentum * psi_at_match), np.finfo(float).tiny)
    if outgoing_residual / outgoing_scale > 1.0e-5:
        raise RuntimeError("the selected energy does not satisfy the outgoing boundary condition")
    outgoing_amplitude = complex(psi_at_match) * np.exp(-1j * momentum * MATCH_POINT)
    scaled_solution = integrate_even_state(energy, SCALING_ANGLE)
    return momentum, outgoing_amplitude, unscaled_solution, scaled_solution


def evaluate_even_resonance(
    x: np.ndarray,
    *,
    theta: float,
    momentum: complex,
    outgoing_amplitude: complex,
    interior_solution,
) -> np.ndarray:
    """Evaluate one prepared even branch inside and outside the matching point."""
    coordinate = np.asarray(x, dtype=float)
    radius = np.abs(coordinate)
    rotation = np.exp(1j * theta)
    values = np.empty(radius.shape, dtype=np.complex128)
    interior = radius <= MATCH_POINT
    values[interior] = interior_solution.sol(radius[interior])[0]
    values[~interior] = outgoing_amplitude * np.exp(
        1j * momentum * radius[~interior] * rotation
    )
    return values


def calculate_wavefunctions(x: np.ndarray, prepared) -> tuple[np.ndarray, np.ndarray]:
    momentum, outgoing_amplitude, unscaled_solution, scaled_solution = prepared
    psi = evaluate_even_resonance(
        x,
        theta=0.0,
        momentum=momentum,
        outgoing_amplitude=outgoing_amplitude,
        interior_solution=unscaled_solution,
    )
    psi_theta = evaluate_even_resonance(
        x,
        theta=SCALING_ANGLE,
        momentum=momentum,
        outgoing_amplitude=outgoing_amplitude,
        interior_solution=scaled_solution,
    )
    return psi, psi_theta


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


def plot_wavefunction(
    output: Path,
    energy: complex,
    prepared,
    *,
    x_limit: float,
    points: int,
    number: int,
) -> Path:
    x = np.linspace(-x_limit, x_limit, points)
    psi, psi_theta = calculate_wavefunctions(x, prepared)

    fig, axes = plt.subplots(2, 1, figsize=(13.8, 7.3), sharex=True)
    banner(fig, f"wavefunction - plot {number}")
    fig.suptitle(
        rf"Resonance state at $E_{{res}}={energy.real:.6f}{energy.imag:+.6f}i$",
        fontsize=15,
        y=0.94,
    )

    axes[0].plot(x, psi.real, color="red", lw=1.25, label=r"$\psi(x)$")
    axes[0].plot(
        x,
        psi_theta.real,
        color="black",
        lw=1.0,
        label=rf"$\psi_\theta(x)$ ($\theta={SCALING_ANGLE:.1f}$)",
    )
    axes[1].plot(x, psi.imag, color="blue", lw=1.25, label=r"$\psi(x)$")
    axes[1].plot(
        x,
        psi_theta.imag,
        color="black",
        lw=1.0,
        label=rf"$\psi_\theta(x)$ ($\theta={SCALING_ANGLE:.1f}$)",
    )

    y_limit = 1.3 if x_limit <= 200.0 else 22.0
    for ax, ylabel in zip(axes, (r"Re $\psi(x)$", r"Im $\psi(x)$")):
        ax.axhline(0.0, color="black", lw=0.75)
        ax.set_ylabel(ylabel)
        ax.set_ylim(-y_limit, y_limit)
        ax.legend(loc="upper right", fontsize=9)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_xlim(-x_limit, x_limit)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    suffix = "near" if number == 1 else "far"
    path = output / f"part6_resonance_wavefunction_{suffix}.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def generate_wavefunction_figures(
    output: Path,
    *,
    quick: bool = False,
) -> list[Path]:
    output = ensure_dir(output.resolve())
    energy = find_second_resonance(configuration(quick))
    prepared = prepare_resonance_state(energy)
    print(
        "second resonance used for wavefunctions: "
        f"{energy.real:.9f}{energy.imag:+.9f}i"
    )
    return [
        plot_wavefunction(
            output,
            energy,
            prepared,
            x_limit=200.0,
            points=8001 if quick else 12001,
            number=1,
        ),
        plot_wavefunction(
            output,
            energy,
            prepared,
            x_limit=500.0,
            points=12001 if quick else 24001,
            number=2,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Part 6 resonance wavefunctions before and after complex scaling"
    )
    parser.add_argument("--quick", action="store_true", help="use the J=400 spectrum")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    paths = generate_wavefunction_figures(args.output_dir, quick=args.quick)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
