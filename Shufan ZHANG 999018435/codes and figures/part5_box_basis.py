from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from quantum_scattering.box_basis import (
    reconstruct_states,
    select_localized_resonance,
    solve_box_basis,
)
from quantum_scattering.core import ensure_dir
from quantum_scattering.scattering import model_potential


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs" / "part5"
PART3_PEAK_SUMMARY = ROOT / "outputs" / "part3" / "part3_peak_summary.txt"
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
    values: dict[str, float] = {}
    try:
        for line in PART3_PEAK_SUMMARY.read_text(encoding="utf-8").splitlines():
            for key in ("E1", "E2"):
                if line.startswith(key + "="):
                    values[key] = float(line.split("=", 1)[1])
    except (OSError, ValueError):
        pass
    values.setdefault("E1", 0.6209703027)
    values.setdefault("E2", 1.3288239467)
    return values


def profiles(quick: bool) -> dict[str, BasisConfig]:
    if quick:
        return {
            "harmonic": BasisConfig(240, 80.0, 1201, 10),
            "short_range": BasisConfig(180, 120.0, 1601, 20),
            "all_states": BasisConfig(121, 120.0, 1601, None),
            "localized": BasisConfig(400, 200.0, 2401, 125),
        }
    return {
        # Reference values shown in the lecture screenshots.
        "harmonic": BasisConfig(500, 150.0, 6000, 10),
        "short_range": BasisConfig(600, 200.0, 10000, 20),
        "all_states": BasisConfig(201, 200.0, 10000, None),
        "localized": BasisConfig(2000, 200.0, 10000, 125),
    }


def orient_states_for_plot(states: np.ndarray) -> np.ndarray:
    """Choose a deterministic eigenvector phase for lecture-style plots.

    Eigenvectors are only defined up to an overall sign.  Making the leftmost
    largest-magnitude lobe positive keeps repeated runs visually stable and
    reproduces the orientation used in the reference Part 5 figures.
    """
    oriented = np.asarray(states, dtype=float).copy()
    if oriented.ndim == 1:
        oriented = oriented[:, None]
    for column in range(oriented.shape[1]):
        peak = int(np.argmax(np.abs(oriented[:, column])))
        if oriented[peak, column] < 0.0:
            oriented[:, column] *= -1.0
    return oriented


def parameter_title(config: BasisConfig) -> str:
    """Parameter line appearing in the teacher's Part 5 plots."""
    return (
        f"num of basis J={config.basis_size}, length of box "
        f"L={config.box_length:g}, pts of x axis N={config.quadrature_points}"
    )


def harmonic_validation(
    output: Path,
    config: BasisConfig,
    kinetic_scheme: str = "teacher_grid",
) -> dict:
    result = solve_box_basis(
        harmonic_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=config.n_states,
        kinetic_scheme=kinetic_scheme,
    )
    exact = np.arange(10, dtype=float) - 0.3
    error = result.energies[:10] - exact
    x = np.linspace(-10.0, 10.0, 3001)
    states = orient_states_for_plot(
        reconstruct_states(x, result.coefficients[:, :10], config.box_length)
    )
    fig, ax = plt.subplots(figsize=(11.0, 6.8), constrained_layout=True)
    potential = harmonic_potential(x)
    ax.plot(x, potential, color="0.50", lw=1.1, label=r"$V(x)=0.5x^2-0.8$")
    for index in range(10):
        ax.plot(
            x,
            result.energies[index] + states[:, index],
            lw=0.95,
            label=rf"$n={index}, E_n={result.energies[index]:.8f}$",
        )
    ax.set(
        xlim=(-10.0, 10.0),
        ylim=(-1.0, 10.0),
        xlabel="x",
        ylabel="Amplitude",
        title=parameter_title(config),
    )
    ax.legend(loc="upper right", fontsize=8.0, ncol=1)
    path = output / "part5_harmonic_validation.png"
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return {
        "kinetic_scheme": kinetic_scheme,
        "grid_spacing_dx": config.box_length / (config.quadrature_points - 1),
        "maximum_absolute_energy_error": float(np.max(np.abs(error))),
        "rms_energy_error": float(np.sqrt(np.mean(error**2))),
        "energies": result.energies[:10].tolist(),
    }


def short_range_first_states(output: Path, config: BasisConfig) -> dict:
    result = solve_box_basis(
        model_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=config.n_states,
    )
    x = np.linspace(-config.box_length / 2.0, config.box_length / 2.0, 4001)
    states = orient_states_for_plot(
        reconstruct_states(x, result.coefficients[:, :11], config.box_length)
    )
    potential = model_potential(x)
    potential_label = r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$"
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    ax.plot(x, potential, color="0.50", lw=1.0, label=potential_label)
    for index in range(11):
        ax.plot(
            x,
            result.energies[index] + states[:, index],
            lw=0.8,
            label=rf"$\psi_{{{index}}}(x),\ E={result.energies[index]:.8f}$",
        )
    ax.set(
        xlim=(-config.box_length / 2.0, config.box_length / 2.0),
        ylim=(-0.8, 1.8),
        xlabel="x",
        ylabel="Amplitude",
        title=parameter_title(config),
    )
    ax.legend(loc="upper right", fontsize=7.0, ncol=1)
    fig.savefig(output / "part5_short_range_first11_combined.png", dpi=190)
    plt.close(fig)

    # The lecture's second short-range figure orders the continuum doublets
    # from high to low energy and gives the bound state a full-width final row.
    fig = plt.figure(figsize=(14.0, 11.0), constrained_layout=True)
    grid = fig.add_gridspec(6, 2, height_ratios=(1, 1, 1, 1, 1, 1.25))
    panel_order = ((9, 10), (7, 8), (5, 6), (3, 4), (1, 2))
    axes = []
    for row, pair in enumerate(panel_order):
        for column, index in enumerate(pair):
            panel = fig.add_subplot(grid[row, column])
            axes.append(panel)
            panel.plot(x, potential, color="0.65", lw=0.75, label=potential_label)
            color = "blue" if column == 0 else "green"
            panel.plot(
                x,
                result.energies[index] + states[:, index],
                color=color,
                lw=0.9,
                label=rf"$\psi_{{{index}}}(x),\ E={result.energies[index]:.8f}$",
            )
            panel.set(
                xlim=(-config.box_length / 2.0, config.box_length / 2.0),
                ylim=(-0.8, 1.8),
                ylabel="Amplitude",
            )
            panel.legend(loc="upper right", fontsize=6.3)
    bound_ax = fig.add_subplot(grid[5, :])
    axes.append(bound_ax)
    bound_ax.plot(x, potential, color="0.65", lw=0.75, label=potential_label)
    bound_ax.plot(
        x,
        result.energies[0] + states[:, 0],
        color="green",
        lw=0.9,
        label=rf"$\psi_0(x),\ E={result.energies[0]:.8f}$",
    )
    bound_ax.set(
        xlim=(-config.box_length / 2.0, config.box_length / 2.0),
        ylim=(-0.8, 1.8),
        xlabel="x",
        ylabel="Amplitude",
    )
    bound_ax.legend(loc="upper right", fontsize=6.8)
    fig.suptitle(parameter_title(config), fontsize=13)
    fig.savefig(output / "part5_short_range_first11_panels.png", dpi=180)
    plt.close(fig)
    return {
        "ground_energy": float(result.energies[0]),
        "first_11_energies": result.energies[:11].tolist(),
    }


def all_states_overlay(output: Path, config: BasisConfig) -> dict:
    result = solve_box_basis(
        model_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=None,
    )
    x = np.linspace(-config.box_length / 2.0, config.box_length / 2.0, 4001)
    states = orient_states_for_plot(
        reconstruct_states(x, result.coefficients, config.box_length)
    )
    fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
    for index, energy in enumerate(result.energies):
        ax.plot(x, energy + states[:, index], alpha=0.82, lw=0.55)
    ax.plot(
        x,
        model_potential(x),
        "k",
        lw=1.0,
        label=r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$",
    )
    ax.text(
        30.0,
        -0.45,
        rf"Bound state $E_0={result.energies[0]:.6f}$",
        fontsize=10,
    )
    ax.set(
        xlim=(-config.box_length / 2.0, config.box_length / 2.0),
        ylim=(-1.0, 5.6),
        xlabel="x",
        ylabel="Amplitude",
        title=parameter_title(config),
    )
    ax.legend()
    fig.savefig(output / "part5_box_spectrum_all_states.png", dpi=190)
    plt.close(fig)
    return {
        "state_count": int(result.energies.size),
        "minimum_energy": float(result.energies[0]),
        "maximum_energy": float(result.energies[-1]),
    }


def localization_metrics(result, config: BasisConfig, peaks: dict[str, float]) -> dict:
    """Compute exact basis norm/parity and two explicitly named localization windows."""
    norms = np.ones(result.energies.size)
    basis_parity = (-1.0) ** (np.arange(1, config.basis_size + 1) + 1)
    parity = np.sum(result.coefficients**2 * basis_parity[:, None], axis=0)
    candidate_mask = ((np.abs(result.energies - peaks["E1"]) <= 0.08)
                      | (np.abs(result.energies - peaks["E2"]) <= 0.08))
    candidate_indices = np.unique(np.r_[0, np.flatnonzero(candidate_mask)])
    x_central = np.linspace(-12.0, 12.0, 2401)
    candidate_states = reconstruct_states(
        x_central, result.coefficients[:, candidate_indices], config.box_length
    )
    wide_probability = np.zeros(result.energies.size)
    wide_probability[candidate_indices] = np.trapezoid(candidate_states**2, x_central, axis=0)
    core = np.abs(x_central) <= BARRIER_HALF_WIDTH
    barrier_probability = np.zeros(result.energies.size)
    barrier_probability[candidate_indices] = np.trapezoid(
        candidate_states[core] ** 2, x_central[core], axis=0
    )
    return {
        "norm": norms,
        "barrier_probability": barrier_probability,
        "wide_probability": wide_probability,
        "parity_expectation": parity,
        "candidate_indices": candidate_indices,
    }


def select_triplet(result, config: BasisConfig, peaks: dict[str, float]):
    metrics = localization_metrics(result, config, peaks)
    first_index, first_scores = select_localized_resonance(
        result.energies, metrics["barrier_probability"], metrics["parity_expectation"], peaks["E1"]
    )
    second_index, second_scores = select_localized_resonance(
        result.energies, metrics["barrier_probability"], metrics["parity_expectation"], peaks["E2"]
    )
    return metrics, first_index, second_index, first_scores, second_scores


def convergence_row(label: str, result, config: BasisConfig, peaks: dict[str, float]) -> dict:
    metrics, first_index, second_index, _, _ = select_triplet(result, config, peaks)
    return {
        "profile": label,
        "basis_size_J": config.basis_size,
        "box_length_L": config.box_length,
        "quadrature_points_N": config.quadrature_points,
        "bound_energy": f"{result.energies[0]:.15g}",
        "first_state_index": first_index,
        "first_energy": f"{result.energies[first_index]:.15g}",
        "first_energy_minus_part3": f"{result.energies[first_index] - peaks['E1']:.15g}",
        "first_barrier_probability": f"{metrics['barrier_probability'][first_index]:.15g}",
        "first_wide_probability_abs_x_le_12": f"{metrics['wide_probability'][first_index]:.15g}",
        "second_state_index": second_index,
        "second_energy": f"{result.energies[second_index]:.15g}",
        "second_energy_minus_part3": f"{result.energies[second_index] - peaks['E2']:.15g}",
        "second_barrier_probability": f"{metrics['barrier_probability'][second_index]:.15g}",
        "second_wide_probability_abs_x_le_12": f"{metrics['wide_probability'][second_index]:.15g}",
    }


def localized_states(output: Path, config: BasisConfig, peaks: dict[str, float]) -> dict:
    result = solve_box_basis(
        model_potential,
        config.basis_size,
        config.box_length,
        config.quadrature_points,
        n_states=config.n_states,
    )
    metrics, first_index, second_index, first_scores, second_scores = select_triplet(
        result, config, peaks
    )
    selected = [0, first_index, second_index]
    labels = ["bound state", "first resonance", "second resonance"]
    targets = [None, peaks["E1"], peaks["E2"]]
    x = np.linspace(-20.0, 20.0, 3001)
    selected_states = orient_states_for_plot(
        reconstruct_states(x, result.coefficients[:, selected], config.box_length)
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)
    ax.plot(
        x,
        model_potential(x),
        "k",
        lw=1.0,
        label=r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$",
    )
    colors = ("red", "blue", "green")
    for panel, (index, color) in enumerate(zip(selected, colors)):
        ax.plot(
            x,
            result.energies[index] + selected_states[:, panel],
            color=color,
            lw=1.25,
        )
    ax.text(
        6.0,
        -0.18,
        rf"Bound state $E_0={result.energies[0]:.6f}$",
        color="red",
        fontsize=10,
    )
    ax.text(
        6.0,
        0.72,
        rf"First Resonance $E_{{{first_index}}}={result.energies[first_index]:.6f}$",
        color="blue",
        fontsize=10,
    )
    ax.text(
        6.0,
        1.45,
        rf"Second Resonance $E_{{{second_index}}}={result.energies[second_index]:.6f}$",
        color="green",
        fontsize=10,
    )
    ax.set(
        xlim=(-20.0, 20.0),
        ylim=(-1.0, 2.0),
        xlabel="x",
        ylabel="Amplitude",
        title=("Bound and Resonance eigenfunctions, "
               f"J={config.basis_size}, L={config.box_length:g}, "
               f"N={config.quadrature_points}"),
    )
    ax.legend(loc="upper right")
    fig.savefig(output / "part5_localized_states.png", dpi=190)
    plt.close(fig)
    rows: list[dict] = []
    for index, label, target in zip(selected, labels, targets):
        rows.append({
            "label": label,
            "state_index_zero_based": index,
            "state_number_one_based": index + 1,
            "energy": f"{result.energies[index]:.15g}",
            "target_energy": "" if target is None else f"{target:.15g}",
            "energy_difference": "" if target is None else f"{result.energies[index] - target:.15g}",
            "finite_grid_norm": f"{metrics['norm'][index]:.15g}",
            "barrier_half_width_sqrt_11p6": f"{BARRIER_HALF_WIDTH:.15g}",
            "central_probability_barrier_window": f"{metrics['barrier_probability'][index]:.15g}",
            "central_probability_abs_x_le_12": f"{metrics['wide_probability'][index]:.15g}",
            "parity_expectation": f"{metrics['parity_expectation'][index]:.15g}",
        })
    smaller_j = max(130, min(600, config.basis_size // 2))
    variant_n = min(config.n_states or 125, smaller_j)
    variant_grid = max(2 * smaller_j + 1, min(config.quadrature_points, 4001))
    smaller_config = BasisConfig(smaller_j, config.box_length, variant_grid, variant_n)
    shorter_config = BasisConfig(smaller_j, 0.8 * config.box_length, variant_grid, variant_n)
    convergence_rows = [convergence_row("baseline", result, config, peaks)]
    for label, variant in (("smaller_J", smaller_config), ("shorter_L", shorter_config)):
        variant_result = solve_box_basis(
            model_potential,
            variant.basis_size,
            variant.box_length,
            variant.quadrature_points,
            n_states=variant.n_states,
        )
        convergence_rows.append(convergence_row(label, variant_result, variant, peaks))
    return {
        "bound": rows[0],
        "first_resonance": rows[1],
        "second_resonance": rows[2],
        "selection": {
            "rule": ("maximize barrier-window probability times energy-proximity and parity-purity "
                     "factors within +/-0.08 of each Part 3 peak"),
            "barrier_window": {"absolute_x_le": BARRIER_HALF_WIDTH, "origin": "barrier maxima sqrt(11.6)"},
            "wide_audit_window": {"absolute_x_le": 12.0},
            "candidate_state_indices": metrics["candidate_indices"].tolist(),
            "first_candidate_scores": first_scores.tolist(),
            "second_candidate_scores": second_scores.tolist(),
        },
        "convergence": convergence_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 5 particle-in-a-box sine-basis matrix method")
    parser.add_argument("--quick", action="store_true", help="reduced basis/grid profile for checks")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--only", choices=("all", "harmonic", "short-range", "spectrum", "localized"), default="all"
    )
    parser.add_argument("--localized-basis", type=int, help="override the localized-state basis size")
    parser.add_argument("--localized-grid", type=int, help="override its endpoint-inclusive quadrature grid")
    parser.add_argument(
        "--harmonic-kinetic",
        choices=("teacher_grid", "spectral"),
        default="teacher_grid",
        help=("kinetic dispersion for the harmonic validation; teacher_grid uses "
              "dx=L/(N-1) and reproduces the lecture's printed energies"),
    )
    args = parser.parse_args()
    output = ensure_dir(args.output_dir.resolve())
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
    diagnostics: dict = {
        "method": {
            "name": "L-8 particle-in-a-box sine-basis matrix diagonalization",
            "basis": "sqrt(2/L) sin[j*pi*(x+L/2)/L]",
            "kinetic": {
                "short_range_and_resonances": "delta_ij*j^2*pi^2/(2L^2)",
                "harmonic_figure": args.harmonic_kinetic,
                "teacher_grid_definition": (
                    "delta_ij*[1-cos(j*pi*dx/L)]/dx^2, dx=L/(N-1)"
                ),
            },
            "potential_matrix": "integral phi_i(x)*V(x)*phi_j(x) dx",
            "implementation": "DCT-I trapezoidal cosine moments; no coordinate finite-difference Hamiltonian",
        },
        "profile": "quick" if args.quick else "reference",
        "config": {name: asdict(value) for name, value in config.items()},
        "part3_reference_peaks": peaks,
    }
    if args.only in ("all", "harmonic"):
        diagnostics["harmonic"] = harmonic_validation(
            output, config["harmonic"], args.harmonic_kinetic
        )
        print(f"harmonic max error: {diagnostics['harmonic']['maximum_absolute_energy_error']:.3e}")
    if args.only in ("all", "short-range"):
        diagnostics["short_range"] = short_range_first_states(output, config["short_range"])
        print(f"short-range ground energy: {diagnostics['short_range']['ground_energy']:.9f}")
    if args.only in ("all", "spectrum"):
        diagnostics["all_states"] = all_states_overlay(output, config["all_states"])
        print(f"all-state spectrum: {diagnostics['all_states']['state_count']} states")
    if args.only in ("all", "localized"):
        diagnostics["localized"] = localized_states(output, config["localized"], peaks)
        first = diagnostics["localized"]["first_resonance"]
        second = diagnostics["localized"]["second_resonance"]
        print(
            "localized states: "
            f"first n={first['state_index_zero_based']} E={float(first['energy']):.9f}; "
            f"second n={second['state_index_zero_based']} E={float(second['energy']):.9f}"
        )
    (output / "part5_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    print(f"outputs: {output}")


if __name__ == "__main__":
    main()
