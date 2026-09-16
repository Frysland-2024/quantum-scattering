from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import part3_stationary_scattering as part3
from quantum_scattering.core import ensure_dir, gaussian_p
from quantum_scattering.scattering import (
    appendix_v_scan,
    left_incident_state_matrix,
    model_potential,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "part4"
MASS = 1.0
HBAR = 1.0
PART4_BASIS_DX = 0.01


@dataclass(frozen=True)
class Scenario:
    name: str
    Emin: float
    E0: float
    Emax: float
    a0: float
    collision_time: float
    snapshots: tuple[float, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    spectrum_xlim: tuple[float, float]
    global_points: int
    animation_points: int
    animation_times: tuple[float, float]
    basis_dx: float = PART4_BASIS_DX
    local_xlim: tuple[float, float] | None = None
    local_ylim: tuple[float, float] | None = None
    local_points: int = 2401
    local_times: tuple[float, float] | None = None

    @property
    def p0(self) -> float:
        return float(np.sqrt(2.0 * MASS * self.E0))

    @property
    def pmin(self) -> float:
        return float(np.sqrt(2.0 * MASS * self.Emin))

    @property
    def pmax(self) -> float:
        return float(np.sqrt(2.0 * MASS * self.Emax))

    @property
    def alpha(self) -> float:
        return (self.a0 / HBAR) ** 2

    @property
    def sigma_p_amplitude(self) -> float:
        return float(np.sqrt(2.0) * self.a0)

    @property
    def x0(self) -> float:
        return -(self.p0 / MASS) * self.collision_time


def part3_peaks() -> dict[str, float]:
    """Use the Part 3 numerical scan directly; no metadata file is needed."""
    (e1, _), (e2, _) = part3.find_resonance_peaks()
    return {"E1": e1, "E2": e2}


@lru_cache(maxsize=1)
def scenario_catalog() -> dict[str, Scenario]:
    peaks = part3_peaks()
    first_peak = peaks["E1"]
    second_peak = peaks["E2"]
    return {
        "blocked": Scenario(
            "blocked", 0.8, 1.0, 1.2, 0.049497, 25.0,
            (-20.0, 10.0, 25.0, 50.0), (-200.0, 200.0), (-1.0, 2.0),
            (0.72, 1.28), 1500, 1200, (-25.0, 70.0),
        ),
        "partially_blocked": Scenario(
            "partially_blocked", 1.248814, 1.328814, 1.408814,
            0.015556, 30.0, (-20.0, 10.0, 35.0, 80.0),
            (-200.0, 200.0), (-1.0, 2.0),
            (1.216814, 1.440814), 1500, 1200, (-30.0, 105.0),
        ),
        "pass": Scenario(
            "pass", 2.5, 2.7, 2.9, 0.02687, 5.0,
            (-20.0, 5.0, 20.0, 50.0), (-200.0, 200.0), (-1.0, 2.0),
            (2.42, 2.98), 1500, 1200, (-25.0, 65.0),
        ),
        "first_resonance": Scenario(
            "first_resonance", first_peak - 7.5e-6, first_peak, first_peak + 7.5e-6,
            2.12e-6, 450000.0, (-20.0, 450000.0, 1000000.0),
            (-1.0e6, 1.0e6), (-0.02, 0.02), (0.62090, 0.62102),
            9001, 9001, (-20.0, 1.0e6), basis_dx=0.0025,
            local_xlim=(-60.0, 60.0), local_ylim=(-0.125, 0.125),
            local_points=2401, local_times=(350000.0, 800000.0),
        ),
        "second_resonance": Scenario(
            "second_resonance", second_peak - 0.001, second_peak, second_peak + 0.001,
            0.000177, 5500.0, (-20.0, 2000.0, 4500.0, 10000.0),
            (-20000.0, 20000.0), (-0.10, 0.10), (1.30, 1.36),
            50001, 50001, (-100.0, 11000.0),
            local_xlim=(-60.0, 60.0), local_ylim=(-0.11, 0.11),
            local_points=2401, local_times=(2600.0, 4300.0),
        ),
    }


SCENARIO_NAMES = (
    "blocked", "partially_blocked", "pass", "first_resonance", "second_resonance"
)


def quadrature_weights(p: np.ndarray) -> np.ndarray:
    if p.ndim != 1 or p.size < 3 or np.any(np.diff(p) <= 0):
        raise ValueError("p must be a strictly increasing one-dimensional grid")
    w = np.empty_like(p)
    w[1:-1] = 0.5 * (p[2:] - p[:-2])
    w[0] = 0.5 * (p[1] - p[0])
    w[-1] = 0.5 * (p[-1] - p[-2])
    return w


def setup(scenario: Scenario, x: np.ndarray, n_p: int):
    if n_p % 2 == 0:
        n_p += 1
    p = np.linspace(
        scenario.p0 - 6.5 * scenario.a0,
        scenario.p0 + 6.5 * scenario.a0,
        n_p,
    )
    weights = quadrature_weights(p)
    states, _, _ = left_incident_state_matrix(
        p, x, dx=scenario.basis_dx
    )
    if x.size >= 10000:
        states = states.astype(np.complex64)
    coeff = gaussian_p(p, scenario.alpha, scenario.p0, scenario.x0)
    coeff = coeff / np.sqrt(float(np.sum(np.abs(coeff) ** 2 * weights)))
    energy = p**2 / (2.0 * MASS)
    return p, energy, states, coeff, weights


def evolve(
    states: np.ndarray,
    energy: np.ndarray,
    coeff: np.ndarray,
    weights: np.ndarray,
    time: float,
) -> np.ndarray:
    phase = np.exp(-1j * energy * time / HBAR)
    return (weights * coeff * phase) @ states


def plot_wave(ax, x: np.ndarray, psi: np.ndarray, time: float, scenario: Scenario) -> None:
    ax.plot(x, psi.real, color="red", lw=0.8, label=r"Re $\Psi$")
    ax.plot(x, psi.imag, color="blue", ls="--", lw=0.8, label=r"Im $\Psi$")
    density_color = "green" if scenario.name == "first_resonance" else "black"
    ax.plot(x, np.abs(psi) ** 2, color=density_color, lw=1.0, label=r"$|\Psi|^2$")
    ax.plot(x, model_potential(x), color="0.65", lw=0.8, label=r"$V(x)$")
    ax.axhline(0.0, color="0.4", lw=0.5)
    ax.set_xlim(*scenario.xlim)
    ax.set_ylim(*scenario.ylim)
    ax.set_title(rf"$t={time:.1f}$")
    ax.set_ylabel("amplitude")
    ax.grid(alpha=0.2)


def spectrum_for(scenario: Scenario, n: int = 801):
    energies = np.linspace(*scenario.spectrum_xlim, n)
    scan = appendix_v_scan(energies, dx=scenario.basis_dx)
    return energies, np.abs(scan.T) ** 2


def transmission_probability(energy: float, dx: float) -> float:
    scan = appendix_v_scan(np.array([energy]), dx=dx)
    return float(np.abs(scan.T[0]) ** 2)


def parameter_title(scenario: Scenario) -> str:
    if scenario.name == "first_resonance":
        return (
            rf"$a_0={scenario.a0:.8f},\ p_0={scenario.p0:.6f},\ "
            rf"\sigma_p={scenario.sigma_p_amplitude:.8f}$"
        )
    return (
        rf"$a_0={scenario.a0:.6f},\ p_0={scenario.p0:.4f},\ "
        rf"\sigma_p={scenario.sigma_p_amplitude:.4f}$"
    )


def animation_parameter_title(scenario: Scenario) -> str:
    if scenario.name == "first_resonance":
        return (
            rf"$E_0={scenario.E0:.6f},\ p_0={scenario.p0:.4f},\ "
            rf"a_0={scenario.a0:.8f}$"
        )
    return (
        rf"$E_{{min}}={scenario.Emin:.4f},\ E_0={scenario.E0:.4f},\ "
        rf"E_{{max}}={scenario.Emax:.4f}$\n"
        rf"$p_0={scenario.p0:.4f},\ a_0={scenario.a0:.4f}$"
    )


def summary(scenario: Scenario, n_p: int) -> Path:
    ensure_dir(OUT)
    x = np.linspace(*scenario.xlim, scenario.global_points)
    p, energy, states, coeff, weights = setup(scenario, x, n_p)
    scan_points = 1000 if "resonance" in scenario.name else 900
    e_scan, t_scan = spectrum_for(scenario, n=scan_points)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.2])
    ax_t = fig.add_subplot(grid[:2, 0])
    ax_p = fig.add_subplot(grid[2:, 0])
    packet_grid = grid[:, 1].subgridspec(len(scenario.snapshots), 1, hspace=0.45)
    wave_axes = [fig.add_subplot(packet_grid[row, 0]) for row in range(len(scenario.snapshots))]

    ax_t.plot(e_scan, t_scan, color="black")
    display_digits = 6
    markers = (
        (scenario.Emin, "blue", r"$E_{min}$"),
        (scenario.E0, "red", r"Resonance $E_0$"),
        (scenario.Emax, "green", r"$E_{max}$"),
    )
    for value, color, symbol in markers:
        ax_t.axvline(value, color=color, ls="--", label=rf"{symbol} = {value:.{display_digits}f}")
    ax_t.set(
        xlabel="Energy",
        ylabel=r"$|T(E)|^2$",
        ylim=(-0.02, 1.05),
        title=r"Transmission Spectrum $T(E)$",
    )
    ax_t.grid(alpha=0.2)
    ax_t.legend(fontsize=8)

    momentum_density = np.abs(coeff) ** 2
    ax_p.plot(p, momentum_density, color="black", label=r"$|\Phi(p)|^2$")
    momentum_markers = (
        (scenario.pmin, scenario.Emin, "blue", r"p_{min}"),
        (scenario.p0, scenario.E0, "red", r"p_0"),
        (scenario.pmax, scenario.Emax, "green", r"p_{max}"),
    )
    for momentum, marker_energy, color, symbol in momentum_markers:
        height = float(np.interp(momentum, p, momentum_density))
        marker_t = transmission_probability(marker_energy, scenario.basis_dx)
        ax_p.scatter(
            [momentum],
            [height],
            color=color,
            s=35,
            label=(
                rf"${symbol}$, E = {marker_energy:.{display_digits}f}, "
                rf"$|T|^2$ = {marker_t:.4f}"
            ),
        )
    ax_p.set(xlabel="Momentum", ylabel=r"$|\Phi(p)|^2$", title=parameter_title(scenario))
    if scenario.name == "first_resonance":
        ax_p.set_xlim(scenario.p0 - 5.0e-5, scenario.p0 + 5.0e-5)
    elif scenario.name == "second_resonance":
        ax_p.set_xlim(scenario.p0 - 3.0e-3, scenario.p0 + 3.0e-3)
    ax_p.grid(alpha=0.2)
    ax_p.legend(fontsize=7)

    for row, (ax, time) in enumerate(zip(wave_axes, scenario.snapshots)):
        psi = evolve(states, energy, coeff, weights, time)
        plot_wave(ax, x, psi, time, scenario)
        if row == 0:
            ax.legend(loc="upper right", fontsize=7)
    wave_axes[-1].set_xlabel("x")

    if scenario.name == "first_resonance":
        fig.suptitle(rf"$E_0={scenario.E0:.6f},\ a_0={scenario.a0:.8f}$", fontsize=14)
    fig.subplots_adjust(
        left=0.06, right=0.98, bottom=0.07, top=0.92, wspace=0.25, hspace=0.45
    )
    path = OUT / f"part4_{scenario.name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def animation(scenario: Scenario, n_p: int, quick: bool, *, local: bool = False) -> Path:
    ensure_dir(OUT)
    if local:
        if scenario.local_xlim is None or scenario.local_times is None:
            raise ValueError(f"{scenario.name} has no local animation specification")
        xlim = scenario.local_xlim
        ylim = scenario.local_ylim or scenario.ylim
        points = scenario.local_points
        timespec = scenario.local_times
    else:
        xlim = scenario.xlim
        ylim = scenario.ylim
        points = scenario.animation_points
        timespec = scenario.animation_times

    x = np.linspace(*xlim, points)
    p, energy, states, coeff, weights = setup(scenario, x, n_p)
    resonance = "resonance" in scenario.name
    n_frames = 40 if resonance else 65 if quick else 110
    times = np.linspace(*timespec, n_frames)
    psi0 = evolve(states, energy, coeff, weights, float(times[0]))

    fig, ax = plt.subplots(figsize=(12, 5))
    line_real, = ax.plot(x, psi0.real, color="red", lw=0.8, label=r"Re $\Psi$")
    line_imag, = ax.plot(x, psi0.imag, color="blue", ls="--", lw=0.8, label=r"Im $\Psi$")
    density_color = "green" if scenario.name == "first_resonance" else "black"
    line_density, = ax.plot(
        x, np.abs(psi0) ** 2, color=density_color, lw=1.0, label=r"$|\Psi|^2$"
    )
    potential_line, = ax.plot(x, model_potential(x), color="0.65", lw=0.8, label=r"$V(x)$")
    ax.axhline(0.0, color="0.4", lw=0.5)
    ax.set(xlim=xlim, ylim=ylim, xlabel="x", ylabel="amplitude")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8, framealpha=1.0, edgecolor="black")
    ax.set_title(animation_parameter_title(scenario))
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, ha="left", va="top", fontsize=12)

    def update(frame: int):
        psi = evolve(states, energy, coeff, weights, float(times[frame]))
        line_real.set_ydata(psi.real)
        line_imag.set_ydata(psi.imag)
        line_density.set_ydata(np.abs(psi) ** 2)
        time_text.set_text(rf"$t={times[frame]:.1f}$")
        return line_real, line_imag, line_density, potential_line, time_text

    movie = FuncAnimation(fig, update, frames=n_frames, interval=75, blit=False)
    suffix = "_local" if local else ""
    path = OUT / f"part4_{scenario.name}{suffix}.gif"
    movie.save(path, writer=PillowWriter(fps=15), dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 4 continuum wave-packet scattering")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES + ("all",), default="all")
    parser.add_argument("--quick", action="store_true", help="fewer momentum points and GIF frames")
    parser.add_argument("--skip-gifs", action="store_true", help="generate static figures only")
    args = parser.parse_args()

    scenarios = scenario_catalog()
    selected = list(SCENARIO_NAMES) if args.scenario == "all" else [args.scenario]
    n_p = 181 if args.quick else 301
    for name in selected:
        scenario = scenarios[name]
        print(summary(scenario, n_p))
        if not args.skip_gifs:
            animation(scenario, n_p, args.quick)
            if scenario.local_xlim is not None:
                animation(scenario, n_p, args.quick, local=True)


if __name__ == "__main__":
    main()
