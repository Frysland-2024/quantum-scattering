"""Three-point backward propagation for stationary one-dimensional scattering."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .potential import double_barrier_potential

Potential = Callable[[ArrayLike], NDArray[np.float64]]

FINITE_DIFFERENCE_DX = 0.0025
FINITE_DIFFERENCE_X_MIN = -50.0
FINITE_DIFFERENCE_X_MAX = 50.0


def uniform_grid(x_min: float, x_max: float, dx: float) -> NDArray[np.float64]:
    """Construct an inclusive, exactly uniform grid."""

    if dx <= 0 or x_max <= x_min:
        raise ValueError("require dx > 0 and x_max > x_min")
    requested_intervals = (x_max - x_min) / dx
    intervals = int(round(requested_intervals))
    tolerance = 1.0e-10 * max(1.0, abs(requested_intervals))
    if not np.isclose(requested_intervals, intervals, rtol=0.0, atol=tolerance):
        raise ValueError("dx must divide the requested interval exactly")
    if intervals < 2:
        raise ValueError("grid needs at least three points")
    return np.linspace(x_min, x_max, intervals + 1)


@dataclass(frozen=True)
class SpectrumResult:
    energy: NDArray[np.float64]
    transmission_amplitude: NDArray[np.complex128]
    reflection_amplitude: NDArray[np.complex128]

    @property
    def transmission(self) -> NDArray[np.float64]:
        return np.abs(self.transmission_amplitude) ** 2

    @property
    def reflection(self) -> NDArray[np.float64]:
        return np.abs(self.reflection_amplitude) ** 2




@dataclass(frozen=True)
class ContinuumState:
    x: NDArray[np.float64]
    energy: float
    raw_wavefunction: NDArray[np.complex128]


@dataclass(frozen=True)
class ResonancePeak:
    """One resonance obtained from the finite-difference transmission curve."""

    energy: float
    transmission: float


@dataclass(frozen=True)
class ResonancePeaks:
    """The two finite-difference resonance solutions used in Parts 3 and 4."""

    first: ResonancePeak
    second: ResonancePeak


def _fit_left_coefficients(
    psi0: NDArray[np.complex128],
    psi1: NDArray[np.complex128],
    k: NDArray[np.float64],
    x0: float,
    x1: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    forward0 = np.exp(1j * k * x0)
    forward1 = np.exp(1j * k * x1)
    backward0 = np.exp(-1j * k * x0)
    backward1 = np.exp(-1j * k * x1)
    determinant = forward0 * backward1 - forward1 * backward0
    incoming = (psi0 * backward1 - psi1 * backward0) / determinant
    reflected = (forward0 * psi1 - forward1 * psi0) / determinant
    return incoming, reflected


def transmission_spectrum(
    energy: ArrayLike,
    *,
    x_min: float = -50.0,
    x_max: float = 50.0,
    dx: float = FINITE_DIFFERENCE_DX,
    mass: float = 1.0,
    hbar: float = 1.0,
    potential: Potential = double_barrier_potential,
) -> SpectrumResult:
    r"""Propagate a unit transmitted wave backwards and extract T and R.

    This deliberately implements the lecture's second-order three-point
    recurrence rather than replacing it with a higher-order black-box ODE
    solver.
    """

    energies = np.atleast_1d(np.asarray(energy, dtype=float))
    if np.any(energies <= 0) or mass <= 0 or hbar <= 0:
        raise ValueError("energies, mass, and hbar must be positive")
    x = uniform_grid(x_min, x_max, dx)
    spacing = float(x[1] - x[0])
    potential_values = np.asarray(potential(x), dtype=float)
    k = np.sqrt(2.0 * mass * energies) / hbar

    psi_next = np.exp(1j * k * x[-1])
    psi = np.exp(1j * k * x[-2])
    factor = 2.0 * mass * spacing**2 / hbar**2
    for index in range(x.size - 2, 0, -1):
        psi_previous = (2.0 + factor * (potential_values[index] - energies)) * psi - psi_next
        psi_next, psi = psi, psi_previous

    incoming, reflected = _fit_left_coefficients(psi, psi_next, k, x[0], x[1])
    transmission_amplitude = 1.0 / incoming
    reflection_amplitude = reflected / incoming
    return SpectrumResult(
        energy=energies,
        transmission_amplitude=transmission_amplitude,
        reflection_amplitude=reflection_amplitude,
    )


def raw_continuum_state(
    energy: float,
    *,
    x_min: float = -50.0,
    x_max: float = 50.0,
    dx: float = FINITE_DIFFERENCE_DX,
    mass: float = 1.0,
    hbar: float = 1.0,
    potential: Potential = double_barrier_potential,
) -> ContinuumState:
    """Return the raw continuum state with transmitted amplitude set to one."""

    if energy <= 0:
        raise ValueError("energy must be positive")
    x = uniform_grid(x_min, x_max, dx)
    spacing = float(x[1] - x[0])
    potential_values = np.asarray(potential(x), dtype=float)
    k = np.sqrt(2.0 * mass * energy) / hbar
    psi = np.empty(x.size, dtype=np.complex128)
    psi[-1] = np.exp(1j * k * x[-1])
    psi[-2] = np.exp(1j * k * x[-2])
    factor = 2.0 * mass * spacing**2 / hbar**2
    for index in range(x.size - 2, 0, -1):
        psi[index - 1] = (
            (2.0 + factor * (potential_values[index] - energy)) * psi[index]
            - psi[index + 1]
        )
    return ContinuumState(
        x=x,
        energy=float(energy),
        raw_wavefunction=psi,
    )


def refine_transmission_peak(
    lower: float,
    upper: float,
    *,
    dx: float = FINITE_DIFFERENCE_DX,
    iterations: int = 4,
    samples: int = 401,
    x_min: float = -50.0,
    x_max: float = 50.0,
) -> tuple[float, float]:
    """Grid-refine a resonance by locating its zero-reflection energy."""

    if not (0 < lower < upper) or samples < 11:
        raise ValueError("invalid peak bracket or sample count")
    lo, hi = float(lower), float(upper)
    best_energy = (lo + hi) / 2.0
    best_probability = 0.0
    for _ in range(iterations):
        energies = np.linspace(lo, hi, samples)
        result = transmission_spectrum(
            energies, x_min=x_min, x_max=x_max, dx=dx
        )
        # For this real potential, a transmission resonance is equivalently a
        # zero of R.  Minimizing R is more stable than maximizing T because tiny
        # unitarity roundoff can otherwise make T slightly larger than one.
        index = int(np.argmin(result.reflection))
        best_energy = float(energies[index])
        best_probability = float(result.transmission[index])
        step = float(energies[1] - energies[0])
        lo = max(lower, best_energy - 2.0 * step)
        hi = min(upper, best_energy + 2.0 * step)
    return best_energy, best_probability


@lru_cache(maxsize=1)
def solve_resonance_peaks() -> ResonancePeaks:
    """Compute the two Part 3 resonances once on the fine finite-difference grid."""

    first_energy, first_transmission = refine_transmission_peak(
        0.6205,
        0.6212,
        dx=FINITE_DIFFERENCE_DX,
        iterations=5,
        samples=201,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
    )
    second_energy, second_transmission = refine_transmission_peak(
        1.30,
        1.36,
        dx=FINITE_DIFFERENCE_DX,
        iterations=5,
        samples=201,
        x_min=FINITE_DIFFERENCE_X_MIN,
        x_max=FINITE_DIFFERENCE_X_MAX,
    )
    return ResonancePeaks(
        first=ResonancePeak(first_energy, first_transmission),
        second=ResonancePeak(second_energy, second_transmission),
    )
