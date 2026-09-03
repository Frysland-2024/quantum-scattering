"""Spectral wavepacket construction from the scattering continuum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gaussian import packet_momentum_amplitude
from .potential import double_barrier_potential
from .stationary import uniform_grid


def trapezoid_weights(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Quadrature weights for a strictly increasing one-dimensional grid."""

    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("x must be a strictly increasing one-dimensional grid")
    weights = np.empty_like(x)
    weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    weights[0] = (x[1] - x[0]) / 2.0
    weights[-1] = (x[-1] - x[-2]) / 2.0
    return weights


@dataclass(frozen=True)
class PacketCase:
    slug: str
    label: str
    energy_min: float
    energy_center: float
    energy_max: float
    momentum_sigma: float
    center: float
    snapshot_times: tuple[float, ...]
    x_limits: tuple[float, float]
    y_limits: tuple[float, float]
    spectrum_limits: tuple[float, float]
    basis_dx: float
    momentum_points: int

    @property
    def momentum_min(self) -> float:
        return float(np.sqrt(2.0 * self.energy_min))

    @property
    def momentum_center(self) -> float:
        return float(np.sqrt(2.0 * self.energy_center))

    @property
    def momentum_max(self) -> float:
        return float(np.sqrt(2.0 * self.energy_max))

    @property
    def reported_sigma(self) -> float:
        return float(np.sqrt(2.0) * self.momentum_sigma)


def reference_cases(first_peak_energy: float, second_peak_energy: float) -> tuple[PacketCase, ...]:
    """Build the Part 4 cases using Part 3's finite-difference resonance peaks."""

    if not np.isfinite(first_peak_energy) or first_peak_energy <= 0:
        raise ValueError("first_peak_energy must be a positive finite value")
    if not np.isfinite(second_peak_energy) or second_peak_energy <= 0:
        raise ValueError("second_peak_energy must be a positive finite value")
    partial_half_width = 0.08
    resonance_half_width = 0.001
    first_half_width = 7.5e-6
    return (
        PacketCase(
            slug="blocked",
            label="Blocked",
            energy_min=0.8,
            energy_center=1.0,
            energy_max=1.2,
            momentum_sigma=0.049497,
            center=-30.0,
            snapshot_times=(-20.0, 10.0, 25.0, 50.0),
            x_limits=(-200.0, 200.0),
            y_limits=(-1.0, 2.0),
            spectrum_limits=(0.6, 1.25),
            basis_dx=0.02,
            momentum_points=321,
        ),
        PacketCase(
            slug="partially_blocked",
            label="Partially blocked",
            energy_min=second_peak_energy - partial_half_width,
            energy_center=second_peak_energy,
            energy_max=second_peak_energy + partial_half_width,
            momentum_sigma=0.015556,
            center=-30.0,
            snapshot_times=(-20.0, 10.0, 35.0, 80.0),
            x_limits=(-200.0, 200.0),
            y_limits=(-1.0, 2.0),
            spectrum_limits=(1.0, 1.7),
            basis_dx=0.02,
            momentum_points=401,
        ),
        PacketCase(
            slug="pass",
            label="Pass",
            energy_min=2.5,
            energy_center=2.7,
            energy_max=2.9,
            momentum_sigma=0.026870,
            center=-10.0,
            snapshot_times=(-20.0, 5.0, 20.0, 50.0),
            x_limits=(-200.0, 200.0),
            y_limits=(-1.0, 2.0),
            spectrum_limits=(1.7, 3.0),
            basis_dx=0.02,
            momentum_points=321,
        ),
        PacketCase(
            slug="first_resonance",
            label="First resonance",
            energy_min=first_peak_energy - first_half_width,
            energy_center=first_peak_energy,
            energy_max=first_peak_energy + first_half_width,
            momentum_sigma=2.12e-6,
            center=-500000.0,
            snapshot_times=(-20.0, 450000.0, 1000000.0),
            x_limits=(-1000000.0, 1000000.0),
            y_limits=(-0.02, 0.02),
            spectrum_limits=(first_peak_energy - 7.0e-5, first_peak_energy + 7.0e-5),
            basis_dx=0.0025,
            momentum_points=501,
        ),
        PacketCase(
            slug="second_resonance",
            label="Second resonance",
            energy_min=second_peak_energy - resonance_half_width,
            energy_center=second_peak_energy,
            energy_max=second_peak_energy + resonance_half_width,
            momentum_sigma=0.000177,
            center=-9000.0,
            snapshot_times=(-20.0, 2000.0, 4500.0, 10000.0),
            x_limits=(-20000.0, 20000.0),
            y_limits=(-0.1, 0.1),
            spectrum_limits=(1.30, 1.36),
            basis_dx=0.02,
            momentum_points=501,
        ),
    )


@dataclass
class ScatteringBasis:
    """Incoming-normalized positive-momentum scattering basis."""

    p: NDArray[np.float64]
    energy: NDArray[np.float64]
    x_local: NDArray[np.float64]
    states_local: NDArray[np.complex128]
    transmission: NDArray[np.complex128]
    reflection: NDArray[np.complex128]

    @classmethod
    def build(
        cls,
        p: ArrayLike,
        *,
        x_min: float = -20.0,
        x_max: float = 20.0,
        dx: float = 0.02,
    ) -> "ScatteringBasis":
        momenta = np.asarray(p, dtype=float)
        if momenta.ndim != 1 or np.any(momenta <= 0) or np.any(np.diff(momenta) <= 0):
            raise ValueError("p must be a strictly increasing positive grid")
        x = uniform_grid(x_min, x_max, dx)
        spacing = float(x[1] - x[0])
        energy = momenta**2 / 2.0
        potential = double_barrier_potential(x)
        raw = np.empty((x.size, momenta.size), dtype=np.complex128)
        raw[-1] = np.exp(1j * momenta * x[-1])
        raw[-2] = np.exp(1j * momenta * x[-2])
        factor = 2.0 * spacing**2
        for index in range(x.size - 2, 0, -1):
            raw[index - 1] = (
                (2.0 + factor * (potential[index] - energy)) * raw[index]
                - raw[index + 1]
            )

        forward0 = np.exp(1j * momenta * x[0])
        forward1 = np.exp(1j * momenta * x[1])
        backward0 = np.exp(-1j * momenta * x[0])
        backward1 = np.exp(-1j * momenta * x[1])
        determinant = forward0 * backward1 - forward1 * backward0
        incoming = (raw[0] * backward1 - raw[1] * backward0) / determinant
        reflected_raw = (forward0 * raw[1] - forward1 * raw[0]) / determinant
        transmission = 1.0 / incoming
        reflection = reflected_raw / incoming
        normalized = raw / (incoming[np.newaxis, :] * np.sqrt(2.0 * np.pi))
        return cls(
            p=momenta,
            energy=energy,
            x_local=x,
            states_local=normalized,
            transmission=transmission,
            reflection=reflection,
        )

    def state_matrix(
        self,
        x: ArrayLike,
        *,
        dtype: np.dtype = np.dtype(np.complex64),
        block_size: int = 1024,
    ) -> NDArray[np.complexfloating]:
        """Evaluate every scattering state on arbitrary plotting positions."""

        positions = np.asarray(x, dtype=float)
        if positions.ndim != 1:
            raise ValueError("x must be one-dimensional")
        result = np.empty((positions.size, self.p.size), dtype=dtype)
        left = positions < self.x_local[0]
        right = positions > self.x_local[-1]
        middle = ~(left | right)
        normalization = np.sqrt(2.0 * np.pi)

        def fill_asymptotic(indices: NDArray[np.int64], side: str) -> None:
            for start in range(0, indices.size, block_size):
                selected = indices[start : start + block_size]
                phase = np.exp(1j * np.outer(positions[selected], self.p))
                if side == "left":
                    values = (phase + self.reflection[np.newaxis, :] / phase) / normalization
                else:
                    values = phase * self.transmission[np.newaxis, :] / normalization
                result[selected] = values.astype(dtype, copy=False)

        fill_asymptotic(np.flatnonzero(left), "left")
        fill_asymptotic(np.flatnonzero(right), "right")
        middle_indices = np.flatnonzero(middle)
        if middle_indices.size:
            middle_x = positions[middle_indices]
            interpolated = np.empty((middle_x.size, self.p.size), dtype=np.complex128)
            for column in range(self.p.size):
                state = self.states_local[:, column]
                interpolated[:, column] = np.interp(middle_x, self.x_local, state.real) + 1j * np.interp(
                    middle_x, self.x_local, state.imag
                )
            result[middle_indices] = interpolated.astype(dtype, copy=False)
        return result


def basis_for_case(case: PacketCase, *, quick: bool = False) -> ScatteringBasis:
    """Construct the momentum grid and scattering basis for a reference case."""

    points = case.momentum_points
    if quick:
        points = max(121, (points // 2) | 1)
    lower = max(1.0e-8, case.momentum_center - 6.0 * case.momentum_sigma)
    upper = case.momentum_center + 6.0 * case.momentum_sigma
    momenta = np.linspace(lower, upper, points)
    return ScatteringBasis.build(momenta, dx=case.basis_dx)


def packet_coefficients(case: PacketCase, basis: ScatteringBasis) -> NDArray[np.complex128]:
    """Return normalized momentum coefficients including the position shift."""

    amplitude = packet_momentum_amplitude(
        basis.p,
        momentum=case.momentum_center,
        sigma=case.momentum_sigma,
        center=case.center,
    )
    norm = float(np.trapezoid(np.abs(amplitude) ** 2, basis.p))
    return amplitude / np.sqrt(norm)


def evaluate_packet(
    basis_matrix: NDArray[np.complexfloating],
    basis: ScatteringBasis,
    coefficients: NDArray[np.complex128],
    time: float,
) -> NDArray[np.complex128]:
    """Evaluate integral dp Phi(p) psi_p(x) exp(-i E t)."""

    weights = trapezoid_weights(basis.p)
    timed = weights * coefficients * np.exp(-1j * basis.energy * time)
    return np.asarray(basis_matrix @ timed, dtype=np.complex128)


def evaluate_packet_on_grid(
    x: ArrayLike,
    basis: ScatteringBasis,
    coefficients: NDArray[np.complex128],
    time: float,
    *,
    block_size: int = 1024,
    matrix_dtype: np.dtype = np.dtype(np.complex64),
) -> NDArray[np.complex128]:
    """Evaluate a packet in position blocks without allocating a full x-by-p matrix."""

    positions = np.asarray(x, dtype=float)
    if positions.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    result = np.empty(positions.size, dtype=np.complex128)
    for start in range(0, positions.size, block_size):
        stop = min(start + block_size, positions.size)
        matrix = basis.state_matrix(positions[start:stop], dtype=matrix_dtype, block_size=block_size)
        result[start:stop] = evaluate_packet(matrix, basis, coefficients, time)
    return result
