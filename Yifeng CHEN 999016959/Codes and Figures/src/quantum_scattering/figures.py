"""Generate the reference-style PNG and GIF project figures."""

from __future__ import annotations

import csv

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .delta import convergence_table, regularized_delta
from .gaussian import (
    free_gaussian,
    gaussian_initial_state,
    packet_momentum_amplitude,
)
from .potential import double_barrier_potential
from .box_basis import generate_part5
from .complex_scaling import generate_part6
from .stationary import (
    FINITE_DIFFERENCE_DX,
    FINITE_DIFFERENCE_X_MAX,
    FINITE_DIFFERENCE_X_MIN,
    raw_continuum_state,
    solve_resonance_peaks,
    transmission_spectrum,
)
from .wavepacket import (
    PacketCase,
    basis_for_case,
    evaluate_packet,
    evaluate_packet_on_grid,
    packet_coefficients,
    reference_cases,
)


def _prepare_output(root: Path, part: str) -> Path:
    directory = root / part
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_complex_packet(ax: plt.Axes, x: np.ndarray, psi: np.ndarray) -> None:
    ax.plot(x, psi.real, color="red", lw=1.2, label=r"Re $\psi$")
    ax.plot(x, psi.imag, color="blue", ls="--", lw=1.1, label=r"Im $\psi$")
    ax.plot(x, np.abs(psi) ** 2, color="black", lw=1.3, label=r"$|\psi|^2$")


def generate_part1(root: Path, *, profile: str = "reference", include_gifs: bool = True) -> list[Path]:
    """Generate Gaussian initial-state and free-propagation figures."""

    output = _prepare_output(root, "part1_free_gaussian")
    created: list[Path] = []
    x = np.linspace(-10.0, 10.0, 2401)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, a in zip(axes, (0.1, 1.0, 5.0), strict=True):
        psi = gaussian_initial_state(x, a=a, momentum=1.0)
        _plot_complex_packet(ax, x, psi)
        ax.set_title(f"a={a:.1f}, p=1.0")
        ax.set_xlabel("Position")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1, 3)
        ax.legend(fontsize=8)
    fig.suptitle("Initial Gaussians - width is changing", fontsize=18, fontweight="bold")
    created.append(_save(fig, output / "initial_width_scan.png"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, momentum in zip(axes, (0.1, 1.0, 5.0), strict=True):
        psi = gaussian_initial_state(x, a=1.0, momentum=momentum)
        _plot_complex_packet(ax, x, psi)
        ax.set_title(f"a=1.0, p={momentum:.1f}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1, 3)
        ax.legend(fontsize=8)
    fig.suptitle("Initial Gaussians - momentum is changing", fontsize=18, fontweight="bold")
    created.append(_save(fig, output / "initial_momentum_scan.png"))

    snapshot_x = np.linspace(-10.0, 80.0, 5001)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, sharey=True)
    for ax, time in zip(axes, (0.0, 4.0, 8.0), strict=True):
        psi = free_gaussian(snapshot_x, time, a=1.0, momentum=5.0)
        _plot_complex_packet(ax, snapshot_x, psi)
        ax.set_title(f"a=1.0, p=5.0, t={time:.1f}")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Position")
    fig.suptitle("Free wavepacket propagation - snapshots", fontsize=17, fontweight="bold")
    created.append(_save(fig, output / "free_snapshots.png"))

    if include_gifs:
        animation_x = np.linspace(-50.0, 300.0, 5001)
        frames = 31 if profile == "quick" else 61 if profile == "reference" else 101
        times = np.linspace(0.0, 25.0, frames)
        fig, ax = plt.subplots(figsize=(9, 5))
        initial = free_gaussian(animation_x, times[0], a=1.0, momentum=4.0)
        real_line, = ax.plot(animation_x, initial.real, "r-", lw=1.0, label=r"Re $\psi_t$")
        imag_line, = ax.plot(animation_x, initial.imag, "b-", lw=1.0, label=r"Im $\psi_t$")
        density_line, = ax.plot(animation_x, np.abs(initial) ** 2, "k-", lw=1.3, label=r"$|\psi_t|^2$")
        ax.set_xlim(-50, 300)
        ax.set_ylim(-1, 1)
        ax.set_xlabel("Position")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

        def update(time: float) -> tuple[plt.Artist, ...]:
            psi = free_gaussian(animation_x, float(time), a=1.0, momentum=4.0)
            real_line.set_ydata(psi.real)
            imag_line.set_ydata(psi.imag)
            density_line.set_ydata(np.abs(psi) ** 2)
            ax.set_title(f"Free particle evolution, a=1, p=4, t={time:.1f}")
            return real_line, imag_line, density_line

        animation = FuncAnimation(fig, update, frames=times, interval=60, blit=False)
        gif_path = output / "free_propagation.gif"
        animation.save(gif_path, writer=PillowWriter(fps=15), dpi=100)
        plt.close(fig)
        created.append(gif_path)
    return created


def generate_part2(root: Path) -> list[Path]:
    """Generate the regularized-delta plots and convergence table."""

    output = _prepare_output(root, "part2_regularized_delta")
    created: list[Path] = []
    p = np.linspace(-2.0, 2.0, 40001)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for ax, cutoff in zip(axes, (10.0, 100.0, 1000.0), strict=True):
        ax.plot(p, regularized_delta(p, cutoff), color="black", lw=0.8)
        ax.set_title(f"L={cutoff:.0e}")
        ax.set_xlabel("Momentum")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(-2, 2)
        ax.legend([r"$\delta_L(p)=\sin(pL)/(\pi p)$"], fontsize=8)
    fig.suptitle("Regularized delta function", fontsize=17, fontweight="bold")
    created.append(_save(fig, output / "regularized_delta.png"))

    rows = convergence_table()
    csv_path = output / "delta_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["L", "integral", "phi(0)", "absolute_difference", "grid_step"])
        for row in rows:
            writer.writerow(
                [
                    f"{row.cutoff:.0f}",
                    f"{row.integral:.12g}",
                    f"{row.target:.12g}",
                    f"{row.absolute_error:.12g}",
                    f"{row.grid_step:.12g}",
                ]
            )
    created.append(csv_path)

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.axis("off")
    table_text = [
        [
            f"{row.cutoff:.0f}",
            f"{row.integral:.8f}",
            f"{row.target:.8f}",
            f"{row.absolute_error:.8f}",
        ]
        for row in rows
    ]
    table = ax.table(
        cellText=table_text,
        colLabels=["L", r"$\int_{-10}^{10}\phi(p)\delta_L(p)\,dp$", r"$\phi(0)$", "Absolute difference"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)
    ax.set_title("Example of a test calculation", fontsize=16, fontweight="bold", pad=16)
    created.append(_save(fig, output / "delta_test_table.png"))
    return created


def _continuum_figure(energies: Iterable[float], title: str, *, dx: float) -> plt.Figure:
    energy_values = tuple(float(value) for value in energies)
    fig, axes = plt.subplots(len(energy_values), 1, figsize=(11, 8), sharex=True)
    if len(energy_values) == 1:
        axes = np.array([axes])
    for ax, energy in zip(axes, energy_values, strict=True):
        state = raw_continuum_state(
            energy,
            x_min=FINITE_DIFFERENCE_X_MIN,
            x_max=FINITE_DIFFERENCE_X_MAX,
            dx=dx,
        )
        potential = double_barrier_potential(state.x)
        ax.plot(state.x, state.raw_wavefunction.real, color="tab:red", lw=0.9, label=r"Re $\tilde\psi$")
        ax.plot(state.x, state.raw_wavefunction.imag, color="tab:blue", lw=0.9, label=r"Im $\tilde\psi$")
        ax.plot(state.x, potential, color="0.5", lw=1.0, label=r"$V(x)$")
        ax.set_title(f"E = {energy:.8f}", fontsize=10)
        ax.set_ylabel(r"$\tilde\psi(x)$")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("x")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()
    return fig


def generate_part3(root: Path) -> list[Path]:
    """Generate transmission and raw-continuum figures."""

    output = _prepare_output(root, "part3_stationary_scattering")
    created: list[Path] = []
    peaks = solve_resonance_peaks()
    first_energy = peaks.first.energy
    first_probability = peaks.first.transmission
    second_energy = peaks.second.energy
    second_probability = peaks.second.transmission
    coarse = np.linspace(0.1, 3.0, 1501)
    first_dense = np.unique(np.concatenate((np.linspace(0.6200, 0.6220, 1001), [first_energy])))
    second_dense = np.unique(np.concatenate((np.linspace(1.20, 1.55, 1001), [second_energy])))
    overview_energy = np.unique(np.concatenate((coarse, first_dense, second_dense, [first_energy, second_energy])))
    overview = transmission_spectrum(
        overview_energy,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
        dx=FINITE_DIFFERENCE_DX,
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(overview.energy, overview.transmission, color="black", lw=1.3)
    ax.scatter([first_energy, second_energy], [first_probability, second_probability], color="red", s=25, zorder=3)
    ax.annotate(f"E={first_energy:.4f}", (first_energy, first_probability), xytext=(8, -18), textcoords="offset points", color="red")
    ax.annotate(f"E={second_energy:.4f}", (second_energy, second_probability), xytext=(8, -18), textcoords="offset points", color="red")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Transmission probability")
    ax.set_title("Transmission profile")
    ax.grid(alpha=0.3, ls="--")
    created.append(_save(fig, output / "transmission_profile.png"))

    first_result = transmission_spectrum(
        first_dense,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
        dx=FINITE_DIFFERENCE_DX,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(first_result.energy, first_result.transmission, color="black", lw=1.3)
    ax.scatter([first_energy], [first_probability], color="red", s=25)
    ax.annotate(f"E={first_energy:.4f}", (first_energy, first_probability), xytext=(22, -18), textcoords="offset points", color="red")
    ax.set_xlim(0.6200, 0.6220)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Energy")
    ax.set_ylabel(r"$|T|^2$")
    ax.set_title(f"Transmission zoom near peak 1, E={first_energy:.6f}")
    ax.grid(alpha=0.3, ls="--")
    created.append(_save(fig, output / "first_transmission_peak.png"))

    second_result = transmission_spectrum(
        second_dense,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
        dx=FINITE_DIFFERENCE_DX,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(second_result.energy, second_result.transmission, color="black", lw=1.3)
    ax.scatter([second_energy], [second_probability], color="red", s=25)
    ax.annotate(f"E={second_energy:.4f}", (second_energy, second_probability), xytext=(22, -18), textcoords="offset points", color="red")
    ax.set_xlim(1.20, 1.55)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Energy")
    ax.set_ylabel(r"$|T|^2$")
    ax.set_title(f"Transmission zoom near peak 2, E={second_energy:.6f}")
    ax.grid(alpha=0.3, ls="--")
    created.append(_save(fig, output / "second_transmission_peak.png"))

    created.append(
        _save(
            _continuum_figure(
                (0.4, first_energy, 0.75),
                "Continuum wavefunctions around the first peak",
                dx=FINITE_DIFFERENCE_DX,
            ),
            output / "continuum_first_peak.png",
        )
    )
    created.append(
        _save(
            _continuum_figure(
                (1.0, second_energy, 1.6),
                "Continuum wavefunctions around the second peak",
                dx=FINITE_DIFFERENCE_DX,
            ),
            output / "continuum_second_peak.png",
        )
    )

    return created


def _case_spectrum(
    case: PacketCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = 801
    energies = np.linspace(case.spectrum_limits[0], case.spectrum_limits[1], count)
    marker_energies = np.array([case.energy_min, case.energy_center, case.energy_max])
    if case.slug == "blocked":
        energies = np.unique(np.concatenate((energies, np.linspace(0.6205, 0.6212, 501))))
    elif case.slug == "first_resonance":
        energies = np.unique(
            np.concatenate((energies, np.linspace(case.energy_center - 7.0e-5, case.energy_center + 7.0e-5, 1201)))
        )
    elif case.slug == "second_resonance":
        energies = np.unique(
            np.concatenate((energies, np.linspace(case.energy_center - 0.002, case.energy_center + 0.002, 501)))
        )
    energies = np.unique(np.concatenate((energies, marker_energies)))
    result = transmission_spectrum(
        energies,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
        dx=FINITE_DIFFERENCE_DX,
    )
    marker_result = transmission_spectrum(
        marker_energies,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
        dx=FINITE_DIFFERENCE_DX,
    )
    return result.energy, result.transmission, marker_result.transmission


_STANDARD_PACKET_SLUGS = frozenset(
    {"blocked", "partially_blocked", "pass", "second_resonance"}
)


def _standard_packet_overview(
    case: PacketCase,
    x: np.ndarray,
    packets: list[np.ndarray],
    spectrum_energy: np.ndarray,
    spectrum_probability: np.ndarray,
    marker_probability: np.ndarray,
) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(4, 2, width_ratios=(0.9, 1.25), hspace=0.45, wspace=0.25)
    spectrum_ax = fig.add_subplot(grid[:2, 0])
    momentum_ax = fig.add_subplot(grid[2:, 0])
    packet_axes = [fig.add_subplot(grid[index, 1]) for index in range(4)]

    spectrum_ax.plot(spectrum_energy, spectrum_probability, color="black", lw=1.3)
    marker_colors = ("blue", "red", "green")
    marker_labels = (r"$E_{min}$", r"$E_0$", r"$E_{max}$")
    marker_energies = (case.energy_min, case.energy_center, case.energy_max)
    for value, color, label in zip(marker_energies, marker_colors, marker_labels, strict=True):
        spectrum_ax.axvline(value, color=color, ls="--", lw=1.2, label=f"{label} = {value:.6f}")
    spectrum_ax.set_xlim(*case.spectrum_limits)
    spectrum_ax.set_ylim(-0.05, 1.05)
    spectrum_ax.set_xlabel("Energy")
    spectrum_ax.set_ylabel(r"$|T(E)|^2$")
    spectrum_ax.set_title(r"Transmission Spectrum $T(E)$")
    spectrum_ax.legend(loc="upper right", fontsize=8)
    spectrum_ax.grid(alpha=0.25)

    p_padding = 0.12 * (case.momentum_max - case.momentum_min)
    p_plot = np.linspace(case.momentum_min - p_padding, case.momentum_max + p_padding, 1001)
    phi = packet_momentum_amplitude(
        p_plot,
        momentum=case.momentum_center,
        sigma=case.momentum_sigma,
        center=0.0,
    )
    momentum_ax.plot(p_plot, np.abs(phi) ** 2, color="black", lw=1.4, label=r"$|\Phi(p)|^2$")
    marker_momenta = (case.momentum_min, case.momentum_center, case.momentum_max)
    for momentum, energy, probability, color in zip(
        marker_momenta, marker_energies, marker_probability, marker_colors, strict=True
    ):
        value = float(
            np.abs(
                packet_momentum_amplitude(
                    np.array([momentum]),
                    momentum=case.momentum_center,
                    sigma=case.momentum_sigma,
                    center=0.0,
                )[0]
            )
            ** 2
        )
        momentum_ax.scatter([momentum], [value], color=color, s=32, zorder=3)
        momentum_ax.plot([], [], color=color, marker="o", ls="", label=f"p, E={energy:.6f}, |T|^2={probability:.4f}")
    momentum_ax.set_xlabel("Momentum")
    momentum_ax.set_ylabel(r"$|\Phi(p)|^2$")
    momentum_ax.set_title(
        rf"$a_0={case.momentum_sigma:.6f},\ p_0={case.momentum_center:.4f},\ \sigma_p={case.reported_sigma:.4f}$"
    )
    momentum_ax.legend(loc="upper right", fontsize=7)
    momentum_ax.grid(alpha=0.25)

    potential = double_barrier_potential(x)
    for ax, time, psi in zip(packet_axes, case.snapshot_times, packets, strict=True):
        ax.plot(x, psi.real, color="red", lw=0.8, label=r"Re $\Psi$")
        ax.plot(x, psi.imag, color="blue", ls="--", lw=0.8, label=r"Im $\Psi$")
        ax.plot(x, np.abs(psi) ** 2, color="black", lw=1.1, label=r"$|\Psi|^2$")
        ax.plot(x, potential, color="0.6", lw=1.0, label=r"$V(x)$")
        ax.set_xlim(*case.x_limits)
        ax.set_ylim(*case.y_limits)
        ax.set_ylabel("amplitude")
        ax.set_title(f"t = {time:.1f}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=7)
    packet_axes[-1].set_xlabel("x")
    fig.suptitle(case.label, fontsize=17, fontweight="bold")
    return fig


def _first_resonance_overview(
    case: PacketCase,
    x: np.ndarray,
    packets: list[np.ndarray],
    spectrum_energy: np.ndarray,
    spectrum_probability: np.ndarray,
    marker_probability: np.ndarray,
) -> plt.Figure:
    """Plot the three-snapshot first-resonance case in the report's visual language."""

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(4, 2, width_ratios=(0.9, 1.25), hspace=0.45, wspace=0.25)
    spectrum_ax = fig.add_subplot(grid[:2, 0])
    momentum_ax = fig.add_subplot(grid[2:, 0])
    packet_grid = grid[:, 1].subgridspec(len(case.snapshot_times), 1, hspace=0.45)
    packet_axes = [fig.add_subplot(packet_grid[index, 0]) for index in range(len(case.snapshot_times))]

    spectrum_ax.plot(spectrum_energy, spectrum_probability, color="black", lw=1.3)
    marker_colors = ("blue", "red", "green")
    marker_labels = (r"$E_{min}$", r"Resonance $E_0$", r"$E_{max}$")
    marker_energies = (case.energy_min, case.energy_center, case.energy_max)
    for value, color, label in zip(marker_energies, marker_colors, marker_labels, strict=True):
        spectrum_ax.axvline(value, color=color, ls="--", lw=1.2, label=f"{label} = {value:.6f}")
    spectrum_ax.set_xlim(*case.spectrum_limits)
    spectrum_ax.set_ylim(-0.05, 1.05)
    spectrum_ax.set_xlabel("Energy")
    spectrum_ax.set_ylabel(r"$|T(E)|^2$")
    spectrum_ax.set_title(r"Transmission Spectrum $T(E)$")
    spectrum_ax.legend(loc="upper right", fontsize=8)
    spectrum_ax.grid(alpha=0.25)

    p_display_half_width = 5.0e-5
    p_plot = np.linspace(
        case.momentum_center - p_display_half_width,
        case.momentum_center + p_display_half_width,
        1001,
    )
    phi = packet_momentum_amplitude(
        p_plot,
        momentum=case.momentum_center,
        sigma=case.momentum_sigma,
        center=0.0,
    )
    momentum_ax.plot(p_plot, np.abs(phi) ** 2, color="black", lw=1.4, label=r"$|\Phi(p)|^2$")
    marker_momenta = (case.momentum_min, case.momentum_center, case.momentum_max)
    momentum_labels = (r"$p_{min}$", r"$p_0$", r"$p_{max}$")
    for momentum, energy, probability, color, momentum_label in zip(
        marker_momenta,
        marker_energies,
        marker_probability,
        marker_colors,
        momentum_labels,
        strict=True,
    ):
        value = float(
            np.abs(
                packet_momentum_amplitude(
                    np.array([momentum]),
                    momentum=case.momentum_center,
                    sigma=case.momentum_sigma,
                    center=0.0,
                )[0]
            )
            ** 2
        )
        momentum_ax.scatter([momentum], [value], color=color, s=32, zorder=3)
        momentum_ax.plot(
            [],
            [],
            color=color,
            marker="o",
            ls="",
            label=rf"{momentum_label}, $E={energy:.6f},\ |T|^2={probability:.4f}$",
        )
    momentum_ax.set_xlabel("Momentum")
    momentum_ax.set_ylabel(r"$|\Phi(p)|^2$")
    momentum_ax.set_title(
        rf"$a_0={case.momentum_sigma:.8f},\ p_0={case.momentum_center:.6f},\ \sigma_p={case.reported_sigma:.8f}$"
    )
    momentum_ax.legend(loc="upper right", fontsize=7)
    momentum_ax.grid(alpha=0.25)

    potential = double_barrier_potential(x)
    for ax, time, psi in zip(packet_axes, case.snapshot_times, packets, strict=True):
        ax.plot(x, psi.real, color="red", lw=0.8, label=r"Re $\Psi$")
        ax.plot(x, psi.imag, color="blue", ls="--", lw=0.8, label=r"Im $\Psi$")
        ax.plot(x, np.abs(psi) ** 2, color="green", lw=1.1, label=r"$|\Psi|^2$")
        ax.plot(x, potential, color="0.6", lw=1.0, label=r"$V(x)$")
        ax.set_xlim(*case.x_limits)
        ax.set_ylim(*case.y_limits)
        ax.set_ylabel("amplitude")
        ax.set_title(f"t = {time:.1f}")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right", fontsize=7)
    packet_axes[-1].set_xlabel("x")
    fig.suptitle(
        rf"$E_0={case.energy_center:.6f},\ a_0={case.momentum_sigma:.8f}$",
        fontsize=14,
    )
    return fig


def _packet_overview(
    case: PacketCase,
    x: np.ndarray,
    packets: list[np.ndarray],
    spectrum_energy: np.ndarray,
    spectrum_probability: np.ndarray,
    marker_probability: np.ndarray,
) -> plt.Figure:
    if case.slug in _STANDARD_PACKET_SLUGS:
        return _standard_packet_overview(
            case,
            x,
            packets,
            spectrum_energy,
            spectrum_probability,
            marker_probability,
        )
    return _first_resonance_overview(
        case,
        x,
        packets,
        spectrum_energy,
        spectrum_probability,
        marker_probability,
    )


def _save_packet_gif(
    path: Path,
    case: PacketCase,
    x: np.ndarray,
    state_matrix: np.ndarray | None,
    basis,
    coefficients: np.ndarray,
    times: np.ndarray,
    *,
    fps: int = 15,
    y_limits: tuple[float, float] | None = None,
    block_size: int = 1024,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    potential = double_barrier_potential(x)

    def packet_at(time: float) -> np.ndarray:
        if state_matrix is None:
            return evaluate_packet_on_grid(
                x,
                basis,
                coefficients,
                time,
                block_size=block_size,
            )
        return evaluate_packet(state_matrix, basis, coefficients, time)

    psi = packet_at(float(times[0]))
    primary_line, = ax.plot(x, psi.real, color="red", lw=0.9, label=r"Re $\Psi$")
    secondary_line, = ax.plot(x, psi.imag, color="blue", ls="--", lw=0.9, label=r"Im $\Psi$")
    density_color = "green" if case.slug == "first_resonance" else "black"
    density_line, = ax.plot(
        x,
        np.abs(psi) ** 2,
        color=density_color,
        lw=1.1,
        label=r"$|\Psi|^2$",
    )
    potential_line, = ax.plot(x, potential, color="0.6", lw=1.0, label=r"$V(x)$")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_ylim(*(case.y_limits if y_limits is None else y_limits))
    ax.set_xlabel("x")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0, edgecolor="black")
    ax.grid(alpha=0.25)

    if case.slug == "first_resonance":
        parameter_title = (
            rf"$E_0={case.energy_center:.6f},\ p_0={case.momentum_center:.4f},\ "
            rf"a_0={case.momentum_sigma:.8f}$"
        )
    else:
        parameter_title = (
            rf"$E_{{min}}={case.energy_min:.4f},\ E_0={case.energy_center:.4f},\ E_{{max}}={case.energy_max:.4f}$"
            + "\n"
            + rf"$p_0={case.momentum_center:.4f},\ a_0={case.momentum_sigma:.4f}$"
        )
    ax.set_title(parameter_title)
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )

    def update(time: float) -> tuple[plt.Artist, ...]:
        current = packet_at(float(time))
        primary_line.set_ydata(current.real)
        secondary_line.set_ydata(current.imag)
        density_line.set_ydata(np.abs(current) ** 2)
        time_text.set_text(rf"$t={time:.1f}$")
        return primary_line, secondary_line, density_line, potential_line, time_text

    animation = FuncAnimation(fig, update, frames=times, interval=60, blit=False)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def generate_part4(
    root: Path,
    *,
    profile: str = "reference",
    include_gifs: bool = True,
) -> list[Path]:
    """Generate the spectral scattering-wavepacket PNGs and GIFs."""

    output = _prepare_output(root, "part4_wavepacket_scattering")
    created: list[Path] = []
    quick = profile == "quick"
    normal_frames = 25 if quick else 51 if profile == "reference" else 81
    resonance_frames = 13 if quick else 40 if profile == "reference" else 60
    second_frames = 13 if quick else 40 if profile == "reference" else 60
    peaks = solve_resonance_peaks()

    for case in reference_cases(peaks.first.energy, peaks.second.energy):
        basis = basis_for_case(case, quick=quick)
        coefficients = packet_coefficients(case, basis)
        if case.slug == "first_resonance":
            points = 6001 if quick else 12001 if profile == "reference" else 20001
        elif case.slug == "second_resonance":
            points = 12001 if quick else 40001
        else:
            points = 2401 if quick else 4001
        x = np.linspace(case.x_limits[0], case.x_limits[1], points)
        matrix = None if case.slug == "first_resonance" else basis.state_matrix(x, dtype=np.dtype(np.complex64))
        if matrix is None:
            packets = [
                evaluate_packet_on_grid(x, basis, coefficients, time, block_size=1024)
                for time in case.snapshot_times
            ]
        else:
            packets = [evaluate_packet(matrix, basis, coefficients, time) for time in case.snapshot_times]
        spectrum_energy, spectrum_probability, marker_probability = _case_spectrum(case)
        overview = _packet_overview(
            case,
            x,
            packets,
            spectrum_energy,
            spectrum_probability,
            marker_probability,
        )
        png_path = output / f"{case.slug}.png"
        created.append(_save(overview, png_path, dpi=130))

        if include_gifs:
            if case.slug == "first_resonance":
                full_times = np.linspace(case.snapshot_times[0], case.snapshot_times[-1], resonance_frames)
                full_path = output / "first_resonance_full.gif"
                _save_packet_gif(
                    full_path,
                    case,
                    x,
                    None,
                    basis,
                    coefficients,
                    full_times,
                    fps=10,
                    y_limits=(-0.05, 0.05),
                    block_size=1024,
                )
                created.append(full_path)

                zoom_x = np.linspace(-60.0, 60.0, 1601 if quick else 2401)
                zoom_matrix = basis.state_matrix(zoom_x, dtype=np.dtype(np.complex64))
                zoom_times = full_times
                zoom_path = output / "first_resonance_zoom.gif"
                _save_packet_gif(
                    zoom_path,
                    case,
                    zoom_x,
                    zoom_matrix,
                    basis,
                    coefficients,
                    zoom_times,
                    fps=10,
                    y_limits=(-0.125, 0.125),
                )
                created.append(zoom_path)
            elif case.slug == "second_resonance":
                full_times = np.linspace(-20.0, 10000.0, second_frames)
                full_path = output / "second_resonance_full.gif"
                _save_packet_gif(full_path, case, x, matrix, basis, coefficients, full_times, fps=12)
                created.append(full_path)

                zoom_x = np.linspace(-60.0, 60.0, 2401)
                zoom_matrix = basis.state_matrix(zoom_x, dtype=np.dtype(np.complex64))
                zoom_times = full_times
                zoom_path = output / "second_resonance_zoom.gif"
                _save_packet_gif(zoom_path, case, zoom_x, zoom_matrix, basis, coefficients, zoom_times, fps=12)
                created.append(zoom_path)
            else:
                animation_times = np.linspace(case.snapshot_times[0], case.snapshot_times[-1], normal_frames)
                gif_path = output / f"{case.slug}.gif"
                _save_packet_gif(gif_path, case, x, matrix, basis, coefficients, animation_times)
                created.append(gif_path)
        del matrix
    return created


def generate_all(
    root: Path | str = Path("output"),
    *,
    profile: str = "reference",
    include_gifs: bool = True,
) -> list[Path]:
    """Generate every project figure."""

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    created.extend(generate_part1(root_path, profile=profile, include_gifs=include_gifs))
    created.extend(generate_part2(root_path))
    created.extend(generate_part3(root_path))
    created.extend(generate_part4(root_path, profile=profile, include_gifs=include_gifs))
    created.extend(generate_part5(root_path, profile=profile))
    created.extend(generate_part6(root_path, profile=profile))
    return created
