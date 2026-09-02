"""Independent numerical oracle for the Part 5 particle-in-a-box basis.

This file intentionally lives outside both student code trees.  It uses closed
form matrix elements for the two potentials in L-8, so implementation results
can be checked without sharing their quadrature or basis-building code.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np


def basis_matrix(x: np.ndarray, length: float, size: int) -> np.ndarray:
    """Return phi_j(x)=sqrt(2/L) sin(j*pi*(x+L/2)/L), j=1..J."""
    x = np.asarray(x, dtype=float)
    j = np.arange(1, size + 1, dtype=float)
    return np.sqrt(2.0 / length) * np.sin(
        np.pi * np.outer(x + 0.5 * length, j) / length
    )


def kinetic_diagonal(length: float, size: int) -> np.ndarray:
    j = np.arange(1, size + 1, dtype=float)
    return 0.5 * (np.pi * j / length) ** 2


def harmonic_hamiltonian(length: float, size: int) -> np.ndarray:
    """Exact finite-box Hamiltonian for V=x^2/2-0.8."""
    j = np.arange(1, size + 1, dtype=float)
    n = j[:, None]
    m = j[None, :]
    delta = n - m
    sigma = n + m

    x2 = np.empty((size, size), dtype=float)
    diagonal = length**2 * (1.0 / 12.0 - 1.0 / (2.0 * np.pi**2 * j**2))
    same_parity = ((n.astype(int) - m.astype(int)) % 2) == 0
    off = np.zeros_like(x2)
    nonzero = delta != 0.0
    mask = same_parity & nonzero
    off[mask] = (2.0 * length**2 / np.pi**2) * (
        1.0 / delta[mask] ** 2 - 1.0 / sigma[mask] ** 2
    )
    x2[:] = off
    np.fill_diagonal(x2, diagonal)

    h = 0.5 * x2
    h[np.diag_indices(size)] += kinetic_diagonal(length, size) - 0.8
    return h


def double_barrier_potential(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (0.5 * x**2 - 0.8) * np.exp(-0.1 * x**2)


def double_barrier_hamiltonian(length: float, size: int) -> np.ndarray:
    """Closed-form full-line matrix for the localized Gaussian potential.

    For the requested L>=60, replacing [-L/2,L/2] by the real line has an
    exponentially negligible tail (exp(-L^2/40)).
    """
    indices = np.arange(0, 2 * size + 1, dtype=float)
    k = np.pi * indices / length
    gaussian_transform = math.sqrt(math.pi / 0.1) * np.exp(-k**2 / 0.4)
    cosine_transform = gaussian_transform * (1.7 - 12.5 * k**2)
    shifted_transform = np.cos(0.5 * np.pi * indices) * cosine_transform

    j = np.arange(1, size + 1, dtype=int)
    diff = np.abs(j[:, None] - j[None, :])
    summ = j[:, None] + j[None, :]
    vmat = (shifted_transform[diff] - shifted_transform[summ]) / length
    h = vmat
    h[np.diag_indices(size)] += kinetic_diagonal(length, size)
    return h


def quadrature_potential_matrix(
    potential, length: float, size: int, order: int = 1024
) -> np.ndarray:
    """Independent Gauss-Legendre reference for small-J matrix checks."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x = 0.5 * length * nodes
    w = 0.5 * length * weights
    phi = basis_matrix(x, length, size)
    return phi.T @ ((w * potential(x))[:, None] * phi)


def reconstruct_probability(
    coefficients: np.ndarray,
    length: float,
    half_width: float = math.sqrt(11.6),
    points: int = 4001,
) -> tuple[float, float, float]:
    """Return central probability, total norm, and parity expectation."""
    x_inner = np.linspace(-half_width, half_width, points)
    phi_inner = basis_matrix(x_inner, length, coefficients.size)
    psi_inner = phi_inner @ coefficients
    p_inner = float(np.trapezoid(np.abs(psi_inner) ** 2, x_inner))

    # Coefficient normalization is exact for the full box; report it explicitly.
    norm = float(np.vdot(coefficients, coefficients).real)
    parity_eigenvalues = np.where(
        np.arange(1, coefficients.size + 1) % 2 == 1, 1.0, -1.0
    )
    parity = float(np.sum(np.abs(coefficients) ** 2 * parity_eigenvalues))
    return p_inner, norm, parity


def most_localized_in_window(
    energies: np.ndarray,
    vectors: np.ndarray,
    length: float,
    low: float,
    high: float,
    points: int = 2001,
) -> dict:
    candidates = np.flatnonzero((energies >= low) & (energies <= high))
    if candidates.size == 0:
        raise ValueError(f"No eigenvalue in [{low}, {high}]")
    x = np.linspace(-math.sqrt(11.6), math.sqrt(11.6), points)
    phi = basis_matrix(x, length, vectors.shape[0])
    psi = phi @ vectors[:, candidates]
    scores = np.trapezoid(np.abs(psi) ** 2, x, axis=0)
    best_local = int(np.argmax(scores))
    index = int(candidates[best_local])
    score, norm, parity = reconstruct_probability(
        vectors[:, index], length, points=points
    )
    return {
        "index_zero_based": index,
        "energy": float(energies[index]),
        "central_probability": score,
        "coefficient_norm": norm,
        "parity_expectation": parity,
        "candidate_count": int(candidates.size),
    }


@dataclass
class ReferenceResult:
    length: float
    size: int
    build_seconds: float
    diagonalization_seconds: float
    hermiticity_max_abs: float
    bound_energy: float
    negative_state_count: int
    resonance_1: dict
    resonance_2: dict


def double_barrier_reference(length: float, size: int) -> ReferenceResult:
    start = time.perf_counter()
    h = double_barrier_hamiltonian(length, size)
    build_seconds = time.perf_counter() - start
    hermiticity = float(np.max(np.abs(h - h.T)))
    start = time.perf_counter()
    energies, vectors = np.linalg.eigh(h)
    diagonalization_seconds = time.perf_counter() - start
    return ReferenceResult(
        length=length,
        size=size,
        build_seconds=build_seconds,
        diagonalization_seconds=diagonalization_seconds,
        hermiticity_max_abs=hermiticity,
        bound_energy=float(energies[0]),
        negative_state_count=int(np.sum(energies < 0.0)),
        resonance_1=most_localized_in_window(
            energies, vectors, length, 0.45, 0.85
        ),
        resonance_2=most_localized_in_window(
            energies, vectors, length, 1.05, 1.60
        ),
    )


def harmonic_reference(length: float, size: int) -> dict:
    start = time.perf_counter()
    h = harmonic_hamiltonian(length, size)
    build_seconds = time.perf_counter() - start
    start = time.perf_counter()
    energies, vectors = np.linalg.eigh(h)
    diagonalization_seconds = time.perf_counter() - start
    exact = np.arange(10, dtype=float) - 0.3
    numerical = energies[:10]
    errors = numerical - exact
    return {
        "length": length,
        "size": size,
        "build_seconds": build_seconds,
        "diagonalization_seconds": diagonalization_seconds,
        "hermiticity_max_abs": float(np.max(np.abs(h - h.T))),
        "orthogonality_max_abs": float(
            np.max(np.abs(vectors[:, :10].T @ vectors[:, :10] - np.eye(10)))
        ),
        "numerical": numerical.tolist(),
        "exact": exact.tolist(),
        "signed_error": errors.tolist(),
        "max_abs_error": float(np.max(np.abs(errors))),
    }


def small_matrix_check(length: float = 30.0, size: int = 16) -> dict:
    identity = np.eye(size)
    h_ho = harmonic_hamiltonian(length, size)
    v_ho_analytic = h_ho - np.diag(kinetic_diagonal(length, size))
    v_ho_quad = quadrature_potential_matrix(
        lambda x: 0.5 * x**2 - 0.8, length, size, order=512
    )

    h_db = double_barrier_hamiltonian(length, size)
    v_db_analytic = h_db - np.diag(kinetic_diagonal(length, size))
    v_db_quad = quadrature_potential_matrix(
        double_barrier_potential, length, size, order=768
    )
    basis_overlap = quadrature_potential_matrix(
        lambda x: np.ones_like(x), length, size, order=512
    )
    return {
        "length": length,
        "size": size,
        "basis_overlap_max_abs_error": float(np.max(np.abs(basis_overlap - identity))),
        "harmonic_v_matrix_max_abs_error": float(
            np.max(np.abs(v_ho_analytic - v_ho_quad))
        ),
        "double_barrier_v_matrix_max_abs_error": float(
            np.max(np.abs(v_db_analytic - v_db_quad))
        ),
        "double_barrier_tail_bound_scale": float(math.exp(-0.1 * (0.5 * length) ** 2)),
    }


def resource_estimate(length: float = 200.0, size: int = 2000, points: int = 10000) -> dict:
    del length
    bytes_float = 8
    basis_bytes = points * size * bytes_float
    dense_bytes = size * size * bytes_float
    return {
        "basis_shape": [points, size],
        "basis_mib": basis_bytes / 2**20,
        "one_dense_matrix_mib": dense_bytes / 2**20,
        "basis_plus_weighted_copy_plus_h_plus_eigenvectors_mib": (
            2 * basis_bytes + 2 * dense_bytes
        ) / 2**20,
        "dense_quadrature_multiply_flops_approx": 2 * points * size**2,
        "dense_eigh_cubic_scale_j_cubed": size**3,
    }


def convergence_scan() -> dict:
    """Separate box-length and cutoff effects at approximately fixed k_max."""
    rows = []
    for length, size in ((60.0, 180), (80.0, 240), (100.0, 300), (150.0, 450), (200.0, 600)):
        result = asdict(double_barrier_reference(length, size))
        rows.append(
            {
                "length": length,
                "size": size,
                "k_max": math.pi * size / length,
                "bound_energy": result["bound_energy"],
                "negative_state_count": result["negative_state_count"],
                "resonance_1_energy": result["resonance_1"]["energy"],
                "resonance_1_central_probability": result["resonance_1"]["central_probability"],
                "resonance_2_energy": result["resonance_2"]["energy"],
                "resonance_2_central_probability": result["resonance_2"]["central_probability"],
            }
        )
    return {
        "fixed_k_max_rows": rows,
        "part3_reference_energies": [0.62097030266192, 1.32882394672],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("quick", "teacher", "matrix", "risk", "convergence"),
        default="quick",
    )
    args = parser.parse_args()

    if args.mode == "matrix":
        payload = {"matrix_check": small_matrix_check()}
    elif args.mode == "risk":
        payload = {"resource_estimate": resource_estimate()}
    elif args.mode == "convergence":
        payload = {"convergence": convergence_scan()}
    elif args.mode == "teacher":
        payload = {
            "matrix_check": small_matrix_check(),
            "harmonic": harmonic_reference(150.0, 500),
            "double_barrier": asdict(double_barrier_reference(200.0, 600)),
            "resource_estimate": resource_estimate(),
        }
    else:
        payload = {
            "matrix_check": small_matrix_check(),
            "harmonic": harmonic_reference(30.0, 120),
            "double_barrier": asdict(double_barrier_reference(80.0, 240)),
            "resource_estimate": resource_estimate(),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
