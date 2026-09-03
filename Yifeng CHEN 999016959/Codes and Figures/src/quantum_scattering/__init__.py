"""One-dimensional quantum-scattering reference implementation."""

from .gaussian import free_gaussian, gaussian_initial_state, momentum_gaussian
from .potential import double_barrier_potential
from .box_basis import generate_part5, solve_box_basis
from .complex_scaling import (
    breit_wigner,
    generate_part6,
    identify_resonance_poles,
    solve_complex_scaled_spectrum,
)
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
    "generate_part5",
    "generate_part6",
    "breit_wigner",
    "identify_resonance_poles",
    "raw_continuum_state",
    "solve_resonance_peaks",
    "solve_box_basis",
    "solve_complex_scaled_spectrum",
    "transmission_spectrum",
]

__version__ = "0.1.0"
