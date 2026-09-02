"""Potential used throughout the lecture project."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def double_barrier_potential(x: ArrayLike) -> NDArray[np.float64]:
    """Return V(x) = (x^2/2 - 0.8) exp(-0.1 x^2)."""

    values = np.asarray(x, dtype=float)
    return (0.5 * values**2 - 0.8) * np.exp(-0.1 * values**2)


def potential_landmarks() -> dict[str, float]:
    """Analytic landmarks useful for tests and report text."""

    zero = float(np.sqrt(1.6))
    maximum_x = float(np.sqrt(11.6))
    maximum_v = float(double_barrier_potential(maximum_x))
    return {
        "center_value": -0.8,
        "zero_abs_x": zero,
        "maximum_abs_x": maximum_x,
        "maximum_value": maximum_v,
    }

