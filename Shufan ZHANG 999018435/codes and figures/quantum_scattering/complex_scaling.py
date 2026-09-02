from __future__ import annotations

"""Complex-coordinate scaling in the Part 5 particle-in-a-box basis.

For an analytic one-dimensional potential and ``m = hbar = 1``, the rotated
Hamiltonian is

    H(theta) = exp(-2 i theta) T + V(x exp(i theta)).

The module deliberately uses the same normalized sine basis as ``box_basis``.
Potential matrix elements are evaluated with endpoint-inclusive trapezoidal
quadrature accelerated by a type-I discrete cosine transform.  The resulting
matrix is complex symmetric (``H.T == H``), rather than Hermitian.
"""

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from scipy.fft import dct
from scipy.linalg import eigvals, eigvalsh


ComplexPotential = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ComplexScalingConfig:
    basis_size: int = 2000
    box_length: float = 200.0
    quadrature_points: int = 10000
    mass: float = 1.0
    hbar: float = 1.0

    def validate(self) -> None:
        if self.basis_size < 2:
            raise ValueError("basis_size must be at least 2")
        if self.box_length <= 0.0 or self.mass <= 0.0 or self.hbar <= 0.0:
            raise ValueError("box_length, mass and hbar must be positive")
        if self.quadrature_points - 1 < 2 * self.basis_size:
            raise ValueError(
                "quadrature_points must satisfy N-1 >= 2*basis_size "
                "for the cosine moments"
            )


@dataclass(frozen=True)
class ComplexSpectrum:
    theta: float
    eigenvalues: np.ndarray
    complex_symmetry_error: float
    odd_cosine_moment_max: float
    cross_parity_coupling_upper_bound: float


@dataclass(frozen=True)
class ResonanceTrack:
    label: str
    pole: complex
    reference_at_max_theta: complex
    status: str
    tracked_thetas: np.ndarray
    tracked_values: np.ndarray
    distances_from_reference: np.ndarray
    eligible_theta_count: int
    continuum_clearance: float

    @property
    def width(self) -> float:
        """Breit-Wigner full width Gamma for a pole E_r - i Gamma/2."""
        return -2.0 * float(self.pole.imag)


def rotated_model_potential(x: np.ndarray, theta: float) -> np.ndarray:
    """Analytic continuation of V(x)=(x^2/2-0.8) exp(-x^2/10)."""
    z = np.asarray(x, dtype=float) * np.exp(1j * float(theta))
    return (0.5 * z**2 - 0.8) * np.exp(-0.1 * z**2)


def complex_cosine_moments(
    potential: ComplexPotential,
    box_length: float,
    quadrature_points: int,
    max_order: int,
) -> np.ndarray:
    r"""Return C_k=int_0^L V(y-L/2) cos(k*pi*y/L)dy.

    A DCT-I of an endpoint-inclusive grid equals twice the trapezoidal cosine
    sum.  SciPy applies the transform independently to the real and imaginary
    parts, so the same expression is valid for the rotated complex potential.
    """
    if quadrature_points < 3:
        raise ValueError("quadrature_points must be at least 3")
    if max_order > quadrature_points - 1:
        raise ValueError("quadrature grid is too small for the requested moments")
    y = np.linspace(0.0, box_length, quadrature_points)
    x = y - box_length / 2.0
    values = np.asarray(potential(x), dtype=np.complex128)
    if values.shape != x.shape or not np.all(np.isfinite(values)):
        raise ValueError("potential(x) must return one finite value per x point")
    dx = box_length / (quadrature_points - 1)
    moments = 0.5 * dx * dct(values, type=1)
    return moments[: max_order + 1]


def build_complex_scaled_hamiltonian(
    theta: float,
    config: ComplexScalingConfig,
    potential_factory: Callable[[np.ndarray, float], np.ndarray] = rotated_model_potential,
) -> np.ndarray:
    """Build the complex-symmetric sine-basis Hamiltonian at one angle."""
    config.validate()
    theta = float(theta)
    if theta < 0.0 or theta >= np.pi / 4.0:
        raise ValueError("theta must satisfy 0 <= theta < pi/4 for this potential")
    moments = complex_cosine_moments(
        lambda x: potential_factory(x, theta),
        config.box_length,
        config.quadrature_points,
        2 * config.basis_size,
    )
    indices = np.arange(1, config.basis_size + 1, dtype=int)
    difference = np.abs(indices[:, None] - indices[None, :])
    total = indices[:, None] + indices[None, :]
    hamiltonian = (moments[difference] - moments[total]) / config.box_length
    kinetic = (
        config.hbar**2
        * (np.pi * indices / config.box_length) ** 2
        / (2.0 * config.mass)
    )
    hamiltonian[np.diag_indices_from(hamiltonian)] += kinetic * np.exp(-2j * theta)
    # Roundoff in the indexed construction is already symmetric.  Averaging
    # makes that mathematical invariant explicit for downstream diagnostics.
    return 0.5 * (hamiltonian + hamiltonian.T)


def _rotated_moments(
    theta: float,
    config: ComplexScalingConfig,
    potential_factory: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    config.validate()
    theta = float(theta)
    if theta < 0.0 or theta >= np.pi / 4.0:
        raise ValueError("theta must satisfy 0 <= theta < pi/4 for this potential")
    return complex_cosine_moments(
        lambda x: potential_factory(x, theta),
        config.box_length,
        config.quadrature_points,
        2 * config.basis_size,
    )


def _parity_block(
    indices: np.ndarray,
    moments: np.ndarray,
    theta: float,
    config: ComplexScalingConfig,
) -> np.ndarray:
    """Build one of the two exact parity blocks (odd or even sine indices)."""
    difference = np.abs(indices[:, None] - indices[None, :])
    total = indices[:, None] + indices[None, :]
    block = (moments[difference] - moments[total]) / config.box_length
    kinetic = (
        config.hbar**2
        * (np.pi * indices / config.box_length) ** 2
        / (2.0 * config.mass)
    )
    block[np.diag_indices_from(block)] += kinetic * np.exp(-2j * theta)
    return 0.5 * (block + block.T)


def solve_complex_spectrum(
    theta: float,
    config: ComplexScalingConfig,
    potential_factory: Callable[[np.ndarray, float], np.ndarray] = rotated_model_potential,
) -> ComplexSpectrum:
    """Diagonalize the two exact parity blocks and merge their spectra.

    The analytically continued potential remains even.  Consequently sine
    indices of opposite parity never couple: both ``|i-j|`` and ``i+j`` are
    odd for a cross-parity element, while every odd cosine moment of an even
    function vanishes.  Splitting J=2000 into two 1000-dimensional problems
    gives exactly the same eigenvalues with substantially lower runtime and
    memory than a dense 2000-dimensional solve.
    """
    config.validate()
    theta = float(theta)
    moments = _rotated_moments(theta, config, potential_factory)
    odd_moment_max = float(np.max(np.abs(moments[1::2])))
    # Each cross-parity matrix element is (C_odd-C_odd)/L.
    coupling_bound = 2.0 * odd_moment_max / config.box_length
    all_indices = np.arange(1, config.basis_size + 1, dtype=int)
    blocks = (
        _parity_block(all_indices[::2], moments, theta, config),
        _parity_block(all_indices[1::2], moments, theta, config),
    )
    errors: list[float] = []
    block_values: list[np.ndarray] = []
    for block in blocks:
        denominator = max(float(np.linalg.norm(block)), np.finfo(float).tiny)
        errors.append(float(np.linalg.norm(block - block.T) / denominator))
        if np.isclose(theta, 0.0, atol=1.0e-15):
            # At theta=0 the block is real Hermitian; using eigvalsh both
            # enforces that limit and removes numerical imaginary noise.
            values = eigvalsh(
                block.real, overwrite_a=True, check_finite=False, driver="evd"
            ).astype(np.complex128)
        else:
            values = eigvals(block, overwrite_a=True, check_finite=False)
        block_values.append(values)
    values = np.concatenate(block_values)
    order = np.lexsort((values.imag, values.real))
    return ComplexSpectrum(
        theta,
        values[order],
        max(errors),
        odd_moment_max,
        coupling_bound,
    )


def solve_angle_family(
    angles: Iterable[float], config: ComplexScalingConfig
) -> list[ComplexSpectrum]:
    """Solve a monotonically increasing, duplicate-free angle family."""
    theta = np.asarray(tuple(float(value) for value in angles), dtype=float)
    if theta.ndim != 1 or theta.size < 2 or np.any(np.diff(theta) <= 0.0):
        raise ValueError("angles must be a strictly increasing one-dimensional sequence")
    return [solve_complex_spectrum(value, config) for value in theta]


def plot_window(
    eigenvalues: np.ndarray,
    real_limits: tuple[float, float] = (-0.35, 3.0),
    imag_limits: tuple[float, float] = (-2.0, 0.10),
) -> np.ndarray:
    """Select finite eigenvalues inside a requested complex-energy rectangle."""
    values = np.asarray(eigenvalues, dtype=np.complex128)
    mask = (
        np.isfinite(values.real)
        & np.isfinite(values.imag)
        & (values.real >= real_limits[0])
        & (values.real <= real_limits[1])
        & (values.imag >= imag_limits[0])
        & (values.imag <= imag_limits[1])
    )
    return values[mask]


def distance_from_continuum_ray(values: np.ndarray, theta: float) -> np.ndarray:
    """Perpendicular distance from Im(E)=-tan(2 theta) Re(E)."""
    values = np.asarray(values, dtype=np.complex128)
    slope = np.tan(2.0 * float(theta))
    return np.abs(values.imag + slope * values.real) / np.sqrt(1.0 + slope**2)


def identify_resonance_tracks(
    spectra: list[ComplexSpectrum],
    *,
    real_limits: tuple[float, float] = (0.0, 3.0),
    imag_limits: tuple[float, float] = (-2.0, -1.0e-10),
    continuum_clearance: float = 0.04,
    match_tolerance: float = 0.012,
    exposure_margin: float = 0.05,
    minimum_stable_angles: int = 2,
) -> list[ResonanceTrack]:
    r"""Identify theta-stationary poles exposed above the rotated continuum.

    Candidate poles are the maximum-angle eigenvalues separated from the
    continuum ray.  A pole can only be assessed at angles satisfying
    ``2 theta > |arg(E)|`` (the Aguilar-Balslev-Combes exposure condition).
    The nearest eigenvalue at each eligible angle is retained when it lies
    within ``match_tolerance``.  A final, deeply rotated pole may therefore be
    reported as ``provisional`` when only one supplied angle exposes it.
    """
    if len(spectra) < 2:
        raise ValueError("at least two spectra are required")
    spectra = sorted(spectra, key=lambda item: item.theta)
    reference_spectrum = spectra[-1]
    reference_values = plot_window(
        reference_spectrum.eigenvalues, real_limits, imag_limits
    )
    clearances = distance_from_continuum_ray(reference_values, reference_spectrum.theta)
    candidates = reference_values[clearances >= continuum_clearance]
    candidate_clearances = clearances[clearances >= continuum_clearance]
    order = np.argsort(candidates.real)
    candidates = candidates[order]
    candidate_clearances = candidate_clearances[order]
    tracks: list[ResonanceTrack] = []
    for number, (reference, clearance) in enumerate(
        zip(candidates, candidate_clearances), start=1
    ):
        pole_angle = abs(float(np.angle(reference)))
        eligible = [
            item for item in spectra
            if item.theta > 0.0 and 2.0 * item.theta >= pole_angle + exposure_margin
        ]
        tracked_thetas: list[float] = []
        tracked_values: list[complex] = []
        distances: list[float] = []
        for item in eligible:
            values = plot_window(item.eigenvalues, real_limits, imag_limits)
            if values.size == 0:
                continue
            nearest = values[int(np.argmin(np.abs(values - reference)))]
            distance = float(abs(nearest - reference))
            if distance <= match_tolerance:
                tracked_thetas.append(item.theta)
                tracked_values.append(complex(nearest))
                distances.append(distance)
        # Report the most strongly exposed (maximum-theta) value, matching the
        # lecture figures.  The lower-angle matches are an independent
        # stabilization audit and are not averaged into the quoted pole.
        pole = complex(reference)
        status = "stable" if len(tracked_values) >= minimum_stable_angles else "provisional"
        tracks.append(
            ResonanceTrack(
                label=f"E{number}",
                pole=pole,
                reference_at_max_theta=complex(reference),
                status=status,
                tracked_thetas=np.asarray(tracked_thetas, dtype=float),
                tracked_values=np.asarray(tracked_values, dtype=np.complex128),
                distances_from_reference=np.asarray(distances, dtype=float),
                eligible_theta_count=len(eligible),
                continuum_clearance=float(clearance),
            )
        )
    return tracks


def breit_wigner(energy: np.ndarray | float, pole: complex) -> np.ndarray:
    r"""Unit-height Breit-Wigner profile for E_pole=E_r-i Gamma/2."""
    energy = np.asarray(energy, dtype=float)
    half_width = -float(complex(pole).imag)
    if half_width <= 0.0:
        raise ValueError("a resonance pole must have a negative imaginary part")
    return half_width**2 / ((energy - float(complex(pole).real)) ** 2 + half_width**2)


def nearest_track_value(track: ResonanceTrack, theta: float) -> complex:
    """Return the tracked pole nearest a requested positive rotation angle."""
    if track.tracked_thetas.size == 0:
        return track.pole
    index = int(np.argmin(np.abs(track.tracked_thetas - float(theta))))
    return complex(track.tracked_values[index])
