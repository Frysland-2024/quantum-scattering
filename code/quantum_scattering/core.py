from __future__ import annotations
from pathlib import Path
import numpy as np

HBAR = 1.0
MASS = 1.0


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def gaussian_x(x: np.ndarray, alpha: float, p0: float, x0: float = 0.0,
               hbar: float = HBAR) -> np.ndarray:
    r"""Lecture Gaussian (blackboard notation).

    psi(0,x) = (2 alpha/pi)^(1/4) exp[-alpha (x-x0)^2]
               exp[+ i p0 (x-x0)/hbar].

    This convention gives
        Delta x = 1/(2 sqrt(alpha)),
        Delta p = hbar sqrt(alpha).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    x = np.asarray(x, dtype=float)
    y = x - x0
    return (2.0 * alpha / np.pi) ** 0.25 * np.exp(-alpha * y**2) * np.exp(1j * p0 * y / hbar)


def gaussian_p(p: np.ndarray, alpha: float, p0: float, x0: float = 0.0,
               hbar: float = HBAR) -> np.ndarray:
    r"""Fourier coefficient for gaussian_x, with the lecture convention.

    phi(p) = 1/sqrt(2 pi hbar) int dx psi(x) exp(-i p x/hbar)
           = (1/(2 pi hbar^2 alpha))^(1/4)
             exp[-(p-p0)^2/(4 hbar^2 alpha)] exp[-i p x0/hbar].

    Therefore |phi(p)|^2 has standard deviation Delta p = hbar sqrt(alpha).
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    p = np.asarray(p, dtype=float)
    pref = (1.0 / (2.0 * np.pi * hbar**2 * alpha)) ** 0.25
    return pref * np.exp(-(p - p0) ** 2 / (4.0 * hbar**2 * alpha)) * np.exp(-1j * p * x0 / hbar)


def free_propagate_fft(psi0: np.ndarray, dx: float, t: float,
                       mass: float = MASS, hbar: float = HBAR) -> np.ndarray:
    """Numerical implementation of the lecture Fourier propagation integral."""
    n = psi0.size
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    p = hbar * k
    return np.fft.ifft(np.fft.fft(psi0) * np.exp(-1j * p**2 * t / (2.0 * mass * hbar)))


def regularized_delta(p: np.ndarray | float, L: float) -> np.ndarray | float:
    r"""delta_L(p) = sin(p L)/(pi p), with delta_L(0)=L/pi."""
    p_arr = np.asarray(p, dtype=float)
    out = np.empty_like(p_arr)
    zero = np.isclose(p_arr, 0.0)
    out[zero] = L / np.pi
    out[~zero] = np.sin(p_arr[~zero] * L) / (np.pi * p_arr[~zero])
    return float(out) if np.isscalar(p) else out
