from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

import part3_stationary_scattering as part3
from quantum_scattering.box_basis import (
    reconstruct_states,
    select_localized_resonance,
    solve_box_basis,
)
from quantum_scattering.core import ensure_dir
from quantum_scattering.scattering import model_potential


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "part5"
BARRIER_HALF_WIDTH = float(np.sqrt(11.6))


@dataclass(frozen=True)
class BasisConfig:
    basis_size: int
    box_length: float
    quadrature_points: int
    n_states: int | None


def harmonic_potential(x: np.ndarray) -> np.ndarray:
    return 0.5 * x**2 - 0.8


def reference_peaks() -> dict[str, float]:
    """Use the Part 3 numerical scan directly; no metadata file is needed."""
    (e1, _), (e2, _) = part3.find_resonance_peaks()
    return {"E1": e1, "E2": e2}


def profiles(quick: bool) -> dict[str, BasisConfig]:
    if quick:
        return {
            "harmonic": BasisConfig(120, 60.0, 2000, 10),
            "short_range": BasisConfig(180, 120.0, 3000, 11),
            "all_states": BasisConfig(101, 120.0, 3000, None),
            "localized": BasisConfig(600, 200.0, 5000, 125),
        }
    return {
        "harmonic": BasisConfig(500, 150.0, 6000, 10),
        "short_range": BasisConfig(600, 200.0, 10000, 11),
        "all_states": BasisConfig(201, 200.0, 10000, None),
        "localized": BasisConfig(2000, 200.0, 10000, 125),
    }


def orient_states_for_plot(states: np.ndarray) -> np.ndarray:
    oriented = np.asarray(states, dtype=float).copy()
    if oriented.ndim == 1:
        oriented = oriented[:, None]
    for column in range(oriented.shape[1]):
        peak = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[peak, column] < 0.0:
            oriented[:, column] *= -1.0
    return oriented


def parameter_title(config: BasisConfig) -> str:
    return (
        f"num of basis J={config.basis_size}, length of box "
        f"L={config.box_length:g}, pts of x axis N={config.quadrature_points}"
    )


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def clean_old_outputs(output: Path) -> None:
    legacy_names = (
        "harmonic_oscillator_validation.png",
        "short_range_first11_combined.png",
        "short_range_first11_separate.png",
        "short_range_all_states.png",
        "three_localized_states.png",
        "part5_diagnostics.json",
        "part5_manifest.json",
    )
    for name in legacy_names:
        try:
            (output / name).unlink()
        except FileNotFoundError:
            pass


def harmonic_validation(output: Path, config: BasisConfig, kinetic_scheme: str = "teacher_grid") -> float:
    result = solve_box_basis(
        harmonic_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=10,
        kinetic_scheme=kinetic_scheme,
    )
    exact = np.arange(10, dtype=float) - 0.3
    error = result.energies[:10] - exact
    x = np.linspace(-config.box_length / 2.0, config.box_length / 2.0, config.quadrature_points)
    states = orient_states_for_plot(reconstruct_states(x, result.coefficients[:, :10], config.box_length))
    plot_mask = np.abs(x) <= 10.0
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x[plot_mask], harmonic_potential(x[plot_mask]), color="0.5", lw=1.2, label=r"$V(x)=0.5x^2-0.8$")
    for index in range(10):
        ax.plot(
            x[plot_mask],
            result.energies[index] + states[plot_mask, index],
            lw=1.0,
            label=rf"n={index}, $E_n$={result.energies[index]:.8f}",
        )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.0, 10.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(parameter_title(config))
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, output / "part5_harmonic_validation.png")
    return float(np.max(np.abs(error)))


def short_range_first_states(output: Path, config: BasisConfig) -> float:
    result = solve_box_basis(
        model_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=11,
    )
    x = np.linspace(-config.box_length / 2.0, config.box_length / 2.0, config.quadrature_points)
    states = orient_states_for_plot(reconstruct_states(x, result.coefficients[:, :11], config.box_length))
    center = int(np.argmin(np.abs(x)))
    if states[center, 0] < 0.0:
        states[:, 0] *= -1.0
    potential = model_potential(x)
    potential_label = r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$"

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(x, potential, color="0.5", lw=1.1, label=potential_label)
    for index in range(11):
        ax.plot(
            x,
            result.energies[index] + states[:, index],
            lw=0.9,
            label=rf"$\psi_{{{index}}}(x),\ E={result.energies[index]:.8f}$",
        )
    ax.set_xlim(-config.box_length / 2.0, config.box_length / 2.0)
    ax.set_ylim(-0.9, 1.9)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(parameter_title(config))
    ax.legend(fontsize=7, loc="upper right")
    save_figure(fig, output / "part5_short_range_first11_combined.png")

    fig = plt.figure(figsize=(14, 13.5))
    grid = fig.add_gridspec(6, 2, hspace=0.54, wspace=0.16)
    panel_indices = ((9, 10), (7, 8), (5, 6), (3, 4), (1, 2))
    for row, pair in enumerate(panel_indices):
        for column, index in enumerate(pair):
            ax = fig.add_subplot(grid[row, column])
            ax.plot(x, potential, color="0.6", lw=0.8, label=potential_label)
            color = "blue" if column == 0 else "green"
            ax.plot(
                x,
                result.energies[index] + states[:, index],
                color=color,
                lw=1.0,
                label=rf"$\psi_{{{index}}}(x),\ E={result.energies[index]:.8f}$",
            )
            ax.set_xlim(-config.box_length / 2.0, config.box_length / 2.0)
            ax.set_ylim(-0.9, 1.9)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(loc="upper right", fontsize=6)

    ax = fig.add_subplot(grid[5, :])
    ax.plot(x, potential, color="0.6", lw=0.8, label=potential_label)
    ax.plot(
        x,
        result.energies[0] + states[:, 0],
        color="green",
        lw=1.0,
        label=rf"$\psi_0(x),\ E={result.energies[0]:.8f}$",
    )
    ax.set_xlim(-config.box_length / 2.0, config.box_length / 2.0)
    ax.set_ylim(-0.9, 1.9)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=6)
    fig.suptitle(parameter_title(config), y=0.995)
    save_figure(fig, output / "part5_short_range_first11_panels.png")
    return float(result.energies[0])


def all_states_overlay(output: Path, config: BasisConfig) -> int:
    result = solve_box_basis(
        model_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=None,
    )
    x = np.linspace(-config.box_length / 2.0, config.box_length / 2.0, config.quadrature_points)
    states = orient_states_for_plot(reconstruct_states(x, result.coefficients, config.box_length))
    center = int(np.argmin(np.abs(x)))
    if states[center, 0] < 0.0:
        states[:, 0] *= -1.0
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(
        x,
        model_potential(x),
        color="black",
        lw=1.0,
        label=r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$",
    )
    for index in range(result.energies.size):
        ax.plot(x, result.energies[index] + states[:, index], alpha=0.82, lw=0.65)
    ax.set_xlim(-config.box_length / 2.0, config.box_length / 2.0)
    ax.set_ylim(-1.0, 5.6)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(parameter_title(config))
    ax.legend(loc="upper right")
    ax.text(30.0, -0.48, rf"Bound state $E_0={result.energies[0]:.6f}$", fontsize=11, color="black")
    save_figure(fig, output / "part5_box_spectrum_all_states.png")
    return int(result.energies.size)


def localization_metrics(result, config: BasisConfig, peaks: dict[str, float]) -> dict:
    basis_parity = (-1.0) ** (np.arange(1, config.basis_size + 1) + 1)
    parity = np.sum(result.coefficients**2 * basis_parity[:, None], axis=0)
    candidate_mask = (
        (np.abs(result.energies - peaks["E1"]) <= 0.08)
        | (np.abs(result.energies - peaks["E2"]) <= 0.08)
    )
    candidate_indices = np.unique(np.r_[0, np.flatnonzero(candidate_mask)])
    x_central = np.linspace(-12.0, 12.0, 2401)
    candidate_states = reconstruct_states(x_central, result.coefficients[:, candidate_indices], config.box_length)
    core = np.abs(x_central) <= BARRIER_HALF_WIDTH
    barrier_probability = np.zeros(result.energies.size)
    barrier_probability[candidate_indices] = np.trapezoid(candidate_states[core] ** 2, x_central[core], axis=0)
    return {
        "barrier_probability": barrier_probability,
        "parity_expectation": parity,
    }


def select_triplet(result, config: BasisConfig, peaks: dict[str, float]):
    metrics = localization_metrics(result, config, peaks)
    first_index, _ = select_localized_resonance(
        result.energies,
        metrics["barrier_probability"],
        metrics["parity_expectation"],
        peaks["E1"],
    )
    second_index, _ = select_localized_resonance(
        result.energies,
        metrics["barrier_probability"],
        metrics["parity_expectation"],
        peaks["E2"],
    )
    return first_index, second_index


def localized_states(output: Path, config: BasisConfig, peaks: dict[str, float]) -> tuple[int, float, int, float]:
    result = solve_box_basis(
        model_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=config.n_states,
    )
    first_index, second_index = select_triplet(result, config, peaks)
    selected = [0, first_index, second_index]
    x = np.linspace(-config.box_length / 2.0, config.box_length / 2.0, config.quadrature_points)
    selected_states = orient_states_for_plot(
        reconstruct_states(x, result.coefficients[:, selected], config.box_length)
    )
    center = int(np.argmin(np.abs(x)))
    if selected_states[center, 0] < 0.0:
        selected_states[:, 0] *= -1.0
    left_central = np.flatnonzero((x >= -3.0) & (x < 0.0))
    if left_central.size:
        first_pivot = int(left_central[np.argmax(np.abs(selected_states[left_central, 1]))])
        if selected_states[first_pivot, 1] < 0.0:
            selected_states[:, 1] *= -1.0
    if selected_states[center, 2] > 0.0:
        selected_states[:, 2] *= -1.0

    loc_mask = np.abs(x) <= 20.0
    colors = ("red", "blue", "green")
    potential_label = r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$"
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(x[loc_mask], model_potential(x[loc_mask]), color="black", lw=1.2, label=potential_label)
    for column, color in enumerate(colors):
        energy = float(result.energies[selected[column]])
        ax.plot(x[loc_mask], energy + selected_states[loc_mask, column], color=color, lw=1.4)
    inline_labels = (
        (0, -0.20, rf"Bound state $E_0={result.energies[selected[0]]:.6f}$"),
        (1, 0.72, rf"First Resonance $E_{{{selected[1]}}}={result.energies[selected[1]]:.6f}$"),
        (2, 1.43, rf"Second Resonance $E_{{{selected[2]}}}={result.energies[selected[2]]:.6f}$"),
    )
    for column, y_position, text in inline_labels:
        ax.text(6.0, y_position, text, color=colors[column], fontsize=10)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-1.0, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Bound and Resonance eigenfunctions, J={config.basis_size}, "
        f"L={config.box_length:g}, N={config.quadrature_points}"
    )
    ax.legend(loc="upper right")
    save_figure(fig, output / "part5_localized_states.png")
    return (
        first_index,
        float(result.energies[first_index]),
        second_index,
        float(result.energies[second_index]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 5 particle-in-a-box sine-basis matrix method")
    parser.add_argument("--quick", action="store_true", help="reduced basis/grid profile for checks")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only",
        choices=("all", "harmonic", "short-range", "spectrum", "localized"),
        default="all",
    )
    parser.add_argument("--localized-basis", type=int, help="override the localized-state basis size")
    parser.add_argument("--localized-grid", type=int, help="override its endpoint-inclusive quadrature grid")
    parser.add_argument(
        "--harmonic-kinetic",
        choices=("teacher_grid", "spectral"),
        default="teacher_grid",
        help="teacher_grid reproduces the lecture's printed harmonic energies",
    )
    args = parser.parse_args()

    output = ensure_dir(args.output_dir.resolve())
    clean_old_outputs(output)
    config = profiles(args.quick)
    if args.localized_basis is not None or args.localized_grid is not None:
        old = config["localized"]
        config["localized"] = BasisConfig(
            args.localized_basis or old.basis_size,
            old.box_length,
            args.localized_grid or old.quadrature_points,
            old.n_states,
        )

    peaks = reference_peaks()
    if args.only in ("all", "harmonic"):
        max_error = harmonic_validation(output, config["harmonic"], args.harmonic_kinetic)
        print(f"harmonic max error: {max_error:.3e}")
    if args.only in ("all", "short-range"):
        ground = short_range_first_states(output, config["short_range"])
        print(f"short-range ground energy: {ground:.9f}")
    if args.only in ("all", "spectrum"):
        state_count = all_states_overlay(output, config["all_states"])
        print(f"all-state spectrum: {state_count} states")
    if args.only in ("all", "localized"):
        first_i, first_e, second_i, second_e = localized_states(output, config["localized"], peaks)
        print(
            f"localized states: first n={first_i} E={first_e:.9f}; "
            f"second n={second_i} E={second_e:.9f}"
        )
    clean_old_outputs(output)
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
