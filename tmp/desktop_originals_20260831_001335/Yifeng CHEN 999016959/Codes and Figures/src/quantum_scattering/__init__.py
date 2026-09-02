"""One-dimensional quantum-scattering reference implementation."""

from .gaussian import free_gaussian, gaussian_initial_state, momentum_gaussian
from .potential import double_barrier_potential
from .stationary import (
    FINITE_DIFFERENCE_DX,
    raw_continuum_state,
    solve_resonance_peaks,
    transmission_spectrum,
)

__all__ = [
    "double_barrier_potential",
    "FINITE_DIFFERENCE_DX",
    "free_gaussian",
    "gaussian_initial_state",
    "momentum_gaussian",
    "raw_continuum_state",
    "solve_resonance_peaks",
    "transmission_spectrum",
]

__version__ = "0.1.0"
