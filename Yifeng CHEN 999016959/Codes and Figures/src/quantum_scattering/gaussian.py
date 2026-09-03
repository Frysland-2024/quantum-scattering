"""Normalized Gaussian packets in position and momentum space."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaussian_initial_state(
    x: ArrayLike,
    a: float = 1.0,
    momentum: float = 1.0,
    center: float = 0.0,
    hbar: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Return the normalized lecture-note initial state.

    .. math::
       \psi_0(x)=(2a/\pi)^{1/4}e^{-a(x-x_0)^2}e^{ip_0(x-x_0)/\hbar}.
    """

    if a <= 0 or hbar <= 0:
        raise ValueError("a and hbar must be positive")
    positions = np.asarray(x, dtype=float)
    shifted = positions - center
    normalization = (2.0 * a / np.pi) ** 0.25
    return normalization * np.exp(-a * shifted**2 + 1j * momentum * shifted / hbar)


def momentum_gaussian(
    p: ArrayLike,
    a: float = 1.0,
    momentum: float = 1.0,
    center: float = 0.0,
    hbar: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Fourier coefficient of :func:`gaussian_initial_state`.

    The Fourier convention is
    ``psi(x) = integral dp phi(p) exp(i p x / hbar) / sqrt(2 pi hbar)``.
    """

    if a <= 0 or hbar <= 0:
        raise ValueError("a and hbar must be positive")
    momenta = np.asarray(p, dtype=float)
    normalization = (1.0 / (2.0 * np.pi * hbar**2 * a)) ** 0.25
    envelope = np.exp(-((momenta - momentum) ** 2) / (4.0 * hbar**2 * a))
    translation = np.exp(-1j * momenta * center / hbar)
    return normalization * envelope * translation


def free_gaussian(
    x: ArrayLike,
    t: float,
    a: float = 1.0,
    momentum: float = 1.0,
    center: float = 0.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Analytic free-particle evolution of the lecture Gaussian."""

    if a <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("a, mass, and hbar must be positive")
    positions = np.asarray(x, dtype=float)
    q = 1.0 + 2j * hbar * a * t / mass
    classical_center = center + momentum * t / mass
    normalization = (2.0 * a / np.pi) ** 0.25 / np.sqrt(q)
    spatial = np.exp(-a * (positions - classical_center) ** 2 / q)
    phase = np.exp(
        1j * momentum * (positions - center) / hbar
        - 1j * momentum**2 * t / (2.0 * mass * hbar)
    )
    return normalization * spatial * phase


def packet_momentum_amplitude(
    p: ArrayLike,
    momentum: float,
    sigma: float,
    center: float,
    hbar: float = 1.0,
) -> NDArray[np.complex128]:
    r"""Gaussian coefficient used in the scattering-wavepacket figures.

    Its probability density is
    ``exp(-(p-p0)^2/(2 sigma^2)) / (sqrt(2 pi) sigma)``.
    The reference figures call this width ``a0`` and separately print
    ``sigma_p = sqrt(2) a0``.
    """

    if sigma <= 0 or hbar <= 0:
        raise ValueError("sigma and hbar must be positive")
    momenta = np.asarray(p, dtype=float)
    normalization = (1.0 / (2.0 * np.pi * sigma**2)) ** 0.25
    envelope = np.exp(-((momenta - momentum) ** 2) / (4.0 * sigma**2))
    return normalization * envelope * np.exp(-1j * momenta * center / hbar)

