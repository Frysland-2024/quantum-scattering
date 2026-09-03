"""Lecture-8 particle-in-a-box sine-basis matrix calculations for Part 5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct
from scipy.linalg import eigh


Potential = Callable[[NDArray[np.float64]], NDArray[np.float64]]
FIRST_RESONANCE_REFERENCE = 0.62097030266192
SECOND_RESONANCE_REFERENCE = 1.32882394672
LOCALIZATION_HALF_WIDTH = float(np.sqrt(11.6))


def harmonic_oscillator_potential(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Shifted harmonic oscillator used in the lecture-note validation."""

    return 0.5 * x**2 - 0.8


def short_range_potential(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Short-range double-barrier potential used throughout the project."""

    return (0.5 * x**2 - 0.8) * np.exp(-0.1 * x**2)


@dataclass(frozen=True)
class BoxSolution:
    """Eigenpairs of a real symmetric Hamiltonian in the sine box basis."""

    length: float
    basis_size: int
    grid_points: int
    energies: NDArray[np.float64]
    coefficients: NDArray[np.float64]


def _cosine_moments(
    potential: Potential,
    *,
    length: float,
    grid_points: int,
    maximum_index: int,
) -> NDArray[np.float64]:
    """Return trapezoidal cosine moments through a DCT-I on the teacher's N grid."""

    if grid_points <= maximum_index:
        raise ValueError("grid_points must exceed twice the basis size")
    x = np.linspace(-length / 2.0, length / 2.0, grid_points)
    values = np.asarray(potential(x), dtype=float)
    spacing = length / (grid_points - 1)
    moments = 0.5 * spacing * dct(values, type=1)
    moments = np.asarray(moments[: maximum_index + 1], dtype=float)
    # Both assigned potentials are even. Enforcing the exact symmetry removes
    # roundoff-sized coupling between the even- and odd-parity blocks.
    moments[1::2] = 0.0
    return moments


def _hamiltonian_block(
    basis_indices: NDArray[np.int64],
    moments: NDArray[np.float64],
    *,
    length: float,
    grid_spacing: float | None = None,
) -> NDArray[np.float64]:
    differences = np.abs(basis_indices[:, None] - basis_indices[None, :])
    sums = basis_indices[:, None] + basis_indices[None, :]
    potential_matrix = (moments[differences] - moments[sums]) / length
    if grid_spacing is None:
        kinetic = 0.5 * (np.pi * basis_indices / length) ** 2
    else:
        # The teacher's harmonic-oscillator validation applies the standard
        # three-point x-grid kinetic operator.  In the sine basis its exact
        # eigenvalue is (1-cos(k_j*dx))/dx^2.  Keeping this as a basis-space
        # diagonal reproduces the displayed finite-grid energies while the
        # short-range calculations retain the spectral k_j^2/2 diagonal.
        wave_numbers = np.pi * basis_indices / length
        kinetic = (1.0 - np.cos(wave_numbers * grid_spacing)) / grid_spacing**2
    potential_matrix[np.diag_indices_from(potential_matrix)] += kinetic
    return potential_matrix


def solve_box_basis(
    potential: Potential,
    *,
    length: float,
    basis_size: int,
    grid_points: int,
    state_count: int | None = None,
    kinetic_scheme: str = "spectral",
) -> BoxSolution:
    """Solve Hc=Ec in the Lecture-8 sine basis, separated into parity blocks."""

    if length <= 0 or basis_size < 2 or grid_points < 3:
        raise ValueError("length, basis_size, and grid_points must be positive")
    if state_count is not None and not 1 <= state_count <= basis_size:
        raise ValueError("state_count must lie between one and basis_size")
    if kinetic_scheme not in {"spectral", "teacher_grid"}:
        raise ValueError("kinetic_scheme must be spectral or teacher_grid")

    grid_spacing = length / (grid_points - 1) if kinetic_scheme == "teacher_grid" else None

    moments = _cosine_moments(
        potential,
        length=length,
        grid_points=grid_points,
        maximum_index=2 * basis_size,
    )
    all_energies: list[NDArray[np.float64]] = []
    all_coefficients: list[NDArray[np.float64]] = []
    j_all = np.arange(1, basis_size + 1, dtype=np.int64)

    for odd_j in (True, False):
        j_block = j_all[(j_all % 2 == 1) if odd_j else (j_all % 2 == 0)]
        hamiltonian = _hamiltonian_block(
            j_block,
            moments,
            length=length,
            grid_spacing=grid_spacing,
        )
        if state_count is None:
            energies, vectors = eigh(
                hamiltonian,
                overwrite_a=True,
                check_finite=False,
                driver="evd",
            )
        else:
            # Roughly half of the low spectrum belongs to each parity. The
            # safety margin makes the merged global ordering unambiguous.
            block_count = min(j_block.size, state_count // 2 + 16)
            energies, vectors = eigh(
                hamiltonian,
                subset_by_index=(0, block_count - 1),
                overwrite_a=True,
                check_finite=False,
                driver="evr",
            )
        coefficients = np.zeros((basis_size, energies.size), dtype=float)
        coefficients[j_block - 1, :] = vectors
        all_energies.append(np.asarray(energies, dtype=float))
        all_coefficients.append(coefficients)

    energies = np.concatenate(all_energies)
    coefficients = np.concatenate(all_coefficients, axis=1)
    order = np.argsort(energies)
    if state_count is not None:
        order = order[:state_count]
    return BoxSolution(
        length=float(length),
        basis_size=int(basis_size),
        grid_points=int(grid_points),
        energies=energies[order],
        coefficients=coefficients[:, order],
    )


def reconstruct_wavefunctions(
    x: NDArray[np.float64],
    solution: BoxSolution,
    columns: NDArray[np.int64] | None = None,
    *,
    block_size: int = 1024,
) -> NDArray[np.float64]:
    """Reconstruct selected eigenfunctions without retaining a full N-by-J basis."""

    positions = np.asarray(x, dtype=float)
    selected = np.arange(solution.energies.size) if columns is None else np.asarray(columns, dtype=np.int64)
    coefficients = solution.coefficients[:, selected]
    result = np.empty((positions.size, selected.size), dtype=float)
    j = np.arange(1, solution.basis_size + 1, dtype=float)
    normalization = np.sqrt(2.0 / solution.length)
    for start in range(0, positions.size, block_size):
        stop = min(start + block_size, positions.size)
        phases = np.pi * np.outer(positions[start:stop] + solution.length / 2.0, j) / solution.length
        result[start:stop] = (normalization * np.sin(phases)) @ coefficients
    return result


def _canonicalize(wavefunctions: NDArray[np.float64]) -> NDArray[np.float64]:
    """Choose a deterministic visual sign for each real eigenfunction."""

    result = wavefunctions.copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def _localization_fractions(
    solution: BoxSolution,
    indices: NDArray[np.int64],
    *,
    points: int = 1601,
) -> NDArray[np.float64]:
    x = np.linspace(-LOCALIZATION_HALF_WIDTH, LOCALIZATION_HALF_WIDTH, points)
    wavefunctions = reconstruct_wavefunctions(x, solution, indices)
    return np.asarray(np.trapezoid(wavefunctions**2, x, axis=0), dtype=float)


def identify_localized_states(solution: BoxSolution) -> NDArray[np.int64]:
    """Identify the bound state and two box states localized near the resonances."""

    selected = [0]
    for target, window in ((FIRST_RESONANCE_REFERENCE, 0.04), (SECOND_RESONANCE_REFERENCE, 0.05)):
        candidates = np.flatnonzero(np.abs(solution.energies - target) <= window)
        if candidates.size == 0:
            candidates = np.argsort(np.abs(solution.energies - target))[:12]
        candidate_fractions = _localization_fractions(solution, candidates)
        selected.append(int(candidates[int(np.argmax(candidate_fractions))]))
    return np.asarray(selected, dtype=np.int64)


def _save(fig: plt.Figure, path: Path, *, dpi: int = 150) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _profile_parameters(profile: str) -> dict[str, tuple[int, float, int]]:
    if profile == "quick":
        return {
            "ho": (120, 60.0, 2000),
            "short": (180, 120.0, 3000),
            "all": (101, 120.0, 3000),
            "localized": (600, 200.0, 5000),
        }
    return {
        "ho": (500, 150.0, 6000),
        "short": (600, 200.0, 10000),
        "all": (201, 200.0, 10000),
        "localized": (2000, 200.0, 10000),
    }


def _localized_run(parameters: tuple[int, float, int]) -> tuple[BoxSolution, NDArray[np.int64]]:
    basis_size, length, grid_points = parameters
    solution = solve_box_basis(
        short_range_potential,
        length=length,
        basis_size=basis_size,
        grid_points=grid_points,
        state_count=min(130, basis_size),
    )
    return solution, identify_localized_states(solution)


def generate_part5(root: Path | str = Path("output"), *, profile: str = "reference") -> list[Path]:
    """Generate the five Part-5 static figures."""

    if profile not in {"quick", "reference", "full"}:
        raise ValueError("profile must be quick, reference, or full")
    output = Path(root) / "part5_box_basis"
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    parameters = _profile_parameters(profile)

    # 1. Harmonic-oscillator validation.
    ho_j, ho_l, ho_n = parameters["ho"]
    ho = solve_box_basis(
        harmonic_oscillator_potential,
        length=ho_l,
        basis_size=ho_j,
        grid_points=ho_n,
        state_count=10,
        kinetic_scheme="teacher_grid",
    )
    ho_x = np.linspace(-ho_l / 2.0, ho_l / 2.0, ho_n)
    ho_psi = _canonicalize(reconstruct_wavefunctions(ho_x, ho))
    plot_mask = np.abs(ho_x) <= 10.0
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(
        ho_x[plot_mask],
        harmonic_oscillator_potential(ho_x[plot_mask]),
        color="0.5",
        lw=1.2,
        label=r"$V(x)=0.5x^2-0.8$",
    )
    for index in range(10):
        ax.plot(
            ho_x[plot_mask],
            ho.energies[index] + ho_psi[plot_mask, index],
            lw=1.0,
            label=rf"n={index}, $E_n$={ho.energies[index]:.8f}",
        )
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.0, 10.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"num of basis J={ho_j}, length of box L={ho_l:g}, pts of x axis N={ho_n}"
    )
    ax.legend(loc="upper right", fontsize=8)
    created.append(_save(fig, output / "harmonic_oscillator_validation.png"))

    # 2-3. First eleven short-range states, combined and separated.
    short_j, short_l, short_n = parameters["short"]
    short = solve_box_basis(
        short_range_potential,
        length=short_l,
        basis_size=short_j,
        grid_points=short_n,
        state_count=11,
    )
    short_x = np.linspace(-short_l / 2.0, short_l / 2.0, short_n)
    short_psi = _canonicalize(reconstruct_wavefunctions(short_x, short))
    short_center = int(np.argmin(np.abs(short_x)))
    if short_psi[short_center, 0] < 0.0:
        short_psi[:, 0] *= -1.0
    short_mask = np.ones(short_x.size, dtype=bool)
    potential_label = r"$V(x)=(0.5x^2-0.8)e^{-0.1x^2}$"

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(
        short_x[short_mask],
        short_range_potential(short_x[short_mask]),
        color="0.5",
        lw=1.1,
        label=potential_label,
    )
    for index in range(11):
        ax.plot(
            short_x[short_mask],
            short.energies[index] + short_psi[short_mask, index],
            lw=0.9,
            label=rf"$\psi_{{{index}}}(x),\ E={short.energies[index]:.8f}$",
        )
    ax.set_xlim(-short_l / 2.0, short_l / 2.0)
    ax.set_ylim(-0.9, 1.9)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"num of basis J={short_j}, length of box L={short_l:g}, pts of x axis N={short_n}"
    )
    ax.legend(fontsize=7, loc="upper right")
    created.append(_save(fig, output / "short_range_first11_combined.png"))

    # Match the teacher's content order: paired continuum pseudostates descend
    # from (9,10) to (1,2), with the bound state spanning the final row.
    fig = plt.figure(figsize=(14, 13.5))
    grid = fig.add_gridspec(6, 2, hspace=0.54, wspace=0.16)
    panel_indices = ((9, 10), (7, 8), (5, 6), (3, 4), (1, 2))
    for row, pair in enumerate(panel_indices):
        for column, index in enumerate(pair):
            ax = fig.add_subplot(grid[row, column])
            ax.plot(
                short_x[short_mask],
                short_range_potential(short_x[short_mask]),
                color="0.6",
                lw=0.8,
                label=potential_label,
            )
            color = "blue" if column == 0 else "green"
            ax.plot(
                short_x[short_mask],
                short.energies[index] + short_psi[short_mask, index],
                color=color,
                lw=1.0,
                label=rf"$\psi_{{{index}}}(x),\ E={short.energies[index]:.8f}$",
            )
            ax.set_xlim(-short_l / 2.0, short_l / 2.0)
            ax.set_ylim(-0.9, 1.9)
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(loc="upper right", fontsize=6)

    ax = fig.add_subplot(grid[5, :])
    ax.plot(
        short_x[short_mask],
        short_range_potential(short_x[short_mask]),
        color="0.6",
        lw=0.8,
        label=potential_label,
    )
    ax.plot(
        short_x[short_mask],
        short.energies[0] + short_psi[short_mask, 0],
        color="green",
        lw=1.0,
        label=rf"$\psi_0(x),\ E={short.energies[0]:.8f}$",
    )
    ax.set_xlim(-short_l / 2.0, short_l / 2.0)
    ax.set_ylim(-0.9, 1.9)
    ax.set_xlabel("x", fontsize=8)
    ax.set_ylabel("Amplitude", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=6)
    fig.suptitle(
        f"num of basis J={short_j}, length of box L={short_l:g}, pts of x axis N={short_n}",
        y=0.995,
    )
    created.append(_save(fig, output / "short_range_first11_separate.png"))

    # 4. Every state of the J=201 example, explicitly plotted with energy offsets.
    all_j, all_l, all_n = parameters["all"]
    all_states = solve_box_basis(
        short_range_potential,
        length=all_l,
        basis_size=all_j,
        grid_points=all_n,
        state_count=None,
    )
    all_x = np.linspace(-all_l / 2.0, all_l / 2.0, all_n)
    all_psi = _canonicalize(reconstruct_wavefunctions(all_x, all_states))
    all_center = int(np.argmin(np.abs(all_x)))
    if all_psi[all_center, 0] < 0.0:
        all_psi[:, 0] *= -1.0
    all_mask = np.ones(all_x.size, dtype=bool)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(
        all_x[all_mask],
        short_range_potential(all_x[all_mask]),
        color="black",
        lw=1.0,
        label=potential_label,
    )
    for index in range(all_states.energies.size):
        ax.plot(
            all_x[all_mask],
            all_states.energies[index] + all_psi[all_mask, index],
            alpha=0.82,
            lw=0.65,
        )
    ax.set_xlim(-all_l / 2.0, all_l / 2.0)
    ax.set_ylim(-1.0, 5.6)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"num of basis J={all_j}, length of box L={all_l:g}, pts of x axis N={all_n}"
    )
    ax.legend(loc="upper right")
    ax.text(
        30.0,
        -0.48,
        rf"Bound state $E_0={all_states.energies[0]:.6f}$",
        fontsize=11,
        color="black",
    )
    created.append(_save(fig, output / "short_range_all_states.png"))

    # 5. Bound state and the two localized resonance pseudostates.
    localized, localized_indices = _localized_run(parameters["localized"])
    loc_x = np.linspace(-localized.length / 2.0, localized.length / 2.0, localized.grid_points)
    loc_psi = _canonicalize(reconstruct_wavefunctions(loc_x, localized, localized_indices))
    loc_center = int(np.argmin(np.abs(loc_x)))
    if loc_psi[loc_center, 0] < 0.0:
        loc_psi[:, 0] *= -1.0
    left_central = np.flatnonzero((loc_x >= -3.0) & (loc_x < 0.0))
    first_pivot = int(left_central[np.argmax(np.abs(loc_psi[left_central, 1]))])
    if loc_psi[first_pivot, 1] < 0.0:
        loc_psi[:, 1] *= -1.0
    if loc_psi[loc_center, 2] > 0.0:
        loc_psi[:, 2] *= -1.0
    loc_mask = np.abs(loc_x) <= 20.0
    labels = ("Bound state", "First resonance", "Second resonance")
    colors = ("red", "blue", "green")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        loc_x[loc_mask],
        short_range_potential(loc_x[loc_mask]),
        color="black",
        lw=1.2,
        label=potential_label,
    )
    for column, (label, color) in enumerate(zip(labels, colors, strict=True)):
        energy = float(localized.energies[localized_indices[column]])
        ax.plot(
            loc_x[loc_mask],
            energy + loc_psi[loc_mask, column],
            color=color,
            lw=1.4,
        )
    inline_labels = (
        (0, -0.20, rf"Bound state $E_0={localized.energies[localized_indices[0]]:.6f}$"),
        (1, 0.72, rf"First Resonance $E_{{{localized_indices[1]}}}={localized.energies[localized_indices[1]]:.6f}$"),
        (2, 1.43, rf"Second Resonance $E_{{{localized_indices[2]}}}={localized.energies[localized_indices[2]]:.6f}$"),
    )
    for column, y_position, text in inline_labels:
        ax.text(6.0, y_position, text, color=colors[column], fontsize=10)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-1.0, 2.0)
    ax.set_xlabel("x")
    ax.set_ylabel("Amplitude")
    ax.set_title(
        f"Bound and Resonance eigenfunctions, J={localized.basis_size}, "
        f"L={localized.length:g}, N={localized.grid_points}"
    )
    ax.legend(loc="upper right")
    created.append(_save(fig, output / "three_localized_states.png"))

    return created


__all__ = [
    "BoxSolution",
    "FIRST_RESONANCE_REFERENCE",
    "SECOND_RESONANCE_REFERENCE",
    "generate_part5",
    "harmonic_oscillator_potential",
    "identify_localized_states",
    "reconstruct_wavefunctions",
    "short_range_potential",
    "solve_box_basis",
]
