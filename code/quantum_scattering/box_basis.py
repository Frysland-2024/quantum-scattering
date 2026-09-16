from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.fft import dct
from scipy.linalg import eigh


ArrayLikePotential = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BoxBasisResult:
    energies: np.ndarray
    coefficients: np.ndarray
    box_length: float
    basis_size: int
    quadrature_points: int
    kinetic_scheme: str


def sine_basis(x: np.ndarray, indices: np.ndarray, box_length: float) -> np.ndarray:
    r"""Particle-in-a-box basis on [-L/2,L/2].

    phi_j(x)=sqrt(2/L) sin[j*pi*(x+L/2)/L], j=1,2,... .
    This is equivalent, up to a state-dependent sign, to the lecture's
    sin[j*pi*(x-L/2)/L] convention.
    """
    x = np.asarray(x, dtype=float)
    j = np.asarray(indices, dtype=float)
    if box_length <= 0.0:
        raise ValueError("box_length must be positive")
    if x.ndim != 1 or j.ndim != 1 or np.any(j < 1):
        raise ValueError("x and positive basis indices must be one-dimensional")
    return np.sqrt(2.0 / box_length) * np.sin(
        np.pi * (x[:, None] + box_length / 2.0) * j[None, :] / box_length
    )


def kinetic_energies(basis_size: int, box_length: float,
                     mass: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    if basis_size < 1 or box_length <= 0.0 or mass <= 0.0:
        raise ValueError("basis_size, box_length and mass must be positive")
    j = np.arange(1, basis_size + 1, dtype=float)
    return hbar**2 * (np.pi * j / box_length) ** 2 / (2.0 * mass)


def teacher_grid_kinetic_energies(
    basis_size: int,
    box_length: float,
    quadrature_points: int,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> np.ndarray:
    r"""Kinetic dispersion used by the lecture's harmonic reference plot.

    The three-point second derivative on the same endpoint-inclusive grid used
    for the potential quadrature has sine-mode eigenvalues

        T_j = hbar^2 [1-cos(k_j dx)] / (m dx^2),
        k_j = j*pi/L,  dx = L/(N-1).

    This tends to the spectral ``hbar^2 k_j^2/(2m)`` value as ``dx -> 0`` and
    reproduces the numerical energies printed in the teacher's Part 5 figure.
    """
    if quadrature_points < 3:
        raise ValueError("quadrature_points must be at least 3")
    if basis_size < 1 or box_length <= 0.0 or mass <= 0.0:
        raise ValueError("basis_size, box_length and mass must be positive")
    dx = box_length / (quadrature_points - 1)
    j = np.arange(1, basis_size + 1, dtype=float)
    wave_number = np.pi * j / box_length
    return hbar**2 * (1.0 - np.cos(wave_number * dx)) / (mass * dx**2)


def cosine_moments(potential: ArrayLikePotential, box_length: float,
                   quadrature_points: int, max_order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return C_k=int_0^L V(y-L/2) cos(k*pi*y/L)dy.

    A type-I DCT of values on an endpoint-inclusive uniform grid is exactly the
    trapezoidal-rule cosine transform.  It reduces the potential-matrix work
    from O(N J^2) to O(N log N + J^2).
    """
    if quadrature_points < 3:
        raise ValueError("quadrature_points must be at least 3")
    if max_order > quadrature_points - 1:
        raise ValueError(
            "quadrature_points must satisfy N-1 >= max_order to avoid DCT aliasing"
        )
    y = np.linspace(0.0, box_length, quadrature_points)
    x = y - box_length / 2.0
    values = np.asarray(potential(x), dtype=float)
    if values.shape != x.shape or not np.all(np.isfinite(values)):
        raise ValueError("potential(x) must return one finite value per x point")
    dx = box_length / (quadrature_points - 1)
    moments = 0.5 * dx * dct(values, type=1)
    return moments[:max_order + 1], x, values


def build_hamiltonian(potential: ArrayLikePotential, basis_size: int,
                      box_length: float, quadrature_points: int,
                      mass: float = 1.0, hbar: float = 1.0,
                      kinetic_scheme: str = "spectral") -> np.ndarray:
    r"""Build H_ij in the strict particle-in-a-box sine basis.

    H_ij = delta_ij*j^2*pi^2/(2 L^2) + integral phi_i V phi_j dx.
    Product-to-sum gives V_ij=(C_|i-j|-C_i+j)/L.
    """
    if basis_size < 1:
        raise ValueError("basis_size must be positive")
    moments, _, _ = cosine_moments(
        potential, box_length, quadrature_points, 2 * basis_size
    )
    indices = np.arange(1, basis_size + 1, dtype=int)
    difference = np.abs(indices[:, None] - indices[None, :])
    total = indices[:, None] + indices[None, :]
    hamiltonian = (moments[difference] - moments[total]) / box_length
    diagonal = np.diag_indices_from(hamiltonian)
    if kinetic_scheme == "spectral":
        kinetic = kinetic_energies(
            basis_size, box_length, mass=mass, hbar=hbar
        )
    elif kinetic_scheme == "teacher_grid":
        kinetic = teacher_grid_kinetic_energies(
            basis_size,
            box_length,
            quadrature_points,
            mass=mass,
            hbar=hbar,
        )
    else:
        raise ValueError("kinetic_scheme must be 'spectral' or 'teacher_grid'")
    hamiltonian[diagonal] += kinetic
    return 0.5 * (hamiltonian + hamiltonian.T)


def solve_box_basis(potential: ArrayLikePotential, basis_size: int,
                    box_length: float, quadrature_points: int,
                    n_states: int | None = None, mass: float = 1.0,
                    hbar: float = 1.0,
                    kinetic_scheme: str = "spectral") -> BoxBasisResult:
    hamiltonian = build_hamiltonian(
        potential,
        basis_size,
        box_length,
        quadrature_points,
        mass=mass,
        hbar=hbar,
        kinetic_scheme=kinetic_scheme,
    )
    if n_states is None:
        energies, coefficients = eigh(
            hamiltonian, overwrite_a=True, check_finite=False, driver="evd"
        )
    else:
        if not 1 <= n_states <= basis_size:
            raise ValueError("n_states must lie between 1 and basis_size")
        energies, coefficients = eigh(
            hamiltonian,
            subset_by_index=(0, n_states - 1),
            overwrite_a=True,
            check_finite=False,
            driver="evr",
        )
    return BoxBasisResult(
        energies=energies,
        coefficients=coefficients,
        box_length=float(box_length),
        basis_size=int(basis_size),
        quadrature_points=int(quadrature_points),
        kinetic_scheme=kinetic_scheme,
    )


def reconstruct_states(x: np.ndarray, coefficients: np.ndarray, box_length: float,
                       chunk_size: int = 256) -> np.ndarray:
    """Reconstruct columns psi_n(x)=sum_j c_jn phi_j(x) in basis chunks."""
    x = np.asarray(x, dtype=float)
    coeff = np.asarray(coefficients, dtype=float)
    if coeff.ndim == 1:
        coeff = coeff[:, None]
    if coeff.ndim != 2 or x.ndim != 1:
        raise ValueError("coefficients must be J-by-M and x must be one-dimensional")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    reconstructed = np.zeros((x.size, coeff.shape[1]), dtype=float)
    for start in range(0, coeff.shape[0], chunk_size):
        stop = min(start + chunk_size, coeff.shape[0])
        indices = np.arange(start + 1, stop + 1)
        reconstructed += sine_basis(x, indices, box_length) @ coeff[start:stop]
    return reconstructed


def select_localized_resonance(energies: np.ndarray, central_probability: np.ndarray,
                               parity_expectation: np.ndarray, target_energy: float,
                               window: float = 0.08) -> tuple[int, np.ndarray]:
    """Select a positive-energy box state using proximity, localization and parity."""
    energies = np.asarray(energies, dtype=float)
    central_probability = np.asarray(central_probability, dtype=float)
    parity_expectation = np.asarray(parity_expectation, dtype=float)
    candidates = np.flatnonzero((energies > 0.0) & (np.abs(energies - target_energy) <= window))
    if candidates.size == 0:
        raise ValueError(f"no positive box states within {window:g} of E={target_energy:g}")
    spacings = np.diff(energies)
    local_spacing = float(np.median(spacings[np.maximum(candidates[0] - 2, 0):
                                                   min(candidates[-1] + 2, spacings.size)]))
    local_spacing = max(local_spacing, 1.0e-12)
    proximity = 1.0 / (1.0 + (np.abs(energies[candidates] - target_energy)
                              / (3.0 * local_spacing)) ** 2)
    parity_purity = 0.5 + 0.5 * np.abs(parity_expectation[candidates])
    scores = central_probability[candidates] * proximity * parity_purity
    return int(candidates[int(np.argmax(scores))]), scores
