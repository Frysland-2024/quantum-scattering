"""Finite-cutoff representation of the Dirac delta distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gaussian import momentum_gaussian


def regularized_delta(p: ArrayLike, cutoff: float) -> NDArray[np.float64]:
    r"""Return delta_L(p) = sin(L p) / (pi p), including p=0."""

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    momenta = np.asarray(p, dtype=float)
    return (cutoff / np.pi) * np.sinc(cutoff * momenta / np.pi)


def lecture_test_function(p: ArrayLike) -> NDArray[np.float64]:
    """The Part 1 momentum Gaussian with a=1 and p0=5 used in the table."""

    return np.real(momentum_gaussian(p, a=1.0, momentum=5.0))


@dataclass(frozen=True)
class DeltaConvergenceRow:
    cutoff: float
    integral: float
    target: float
    absolute_error: float
    grid_step: float


def delta_convolution(
    cutoff: float,
    integration_bound: float = 10.0,
    points_per_period: int = 80,
) -> DeltaConvergenceRow:
    """Numerically reproduce the oscillatory finite-window test.

    The reference table is consistent with integrating on [-10, 10].  The
    spacing is chosen from the shortest sinc oscillation instead of a fixed
    grid, which prevents aliasing when L is large.
    """

    if integration_bound <= 0 or points_per_period < 8:
        raise ValueError("integration_bound must be positive and resolution >= 8")
    period = 2.0 * np.pi / cutoff
    step = period / points_per_period
    count = int(np.ceil(2.0 * integration_bound / step)) + 1
    grid = np.linspace(-integration_bound, integration_bound, count)
    values = lecture_test_function(grid) * regularized_delta(grid, cutoff)
    integral = float(np.trapezoid(values, grid))
    target = float(lecture_test_function(np.array([0.0]))[0])
    return DeltaConvergenceRow(
        cutoff=float(cutoff),
        integral=integral,
        target=target,
        absolute_error=abs(integral - target),
        grid_step=float(grid[1] - grid[0]),
    )


def convergence_table(cutoffs: tuple[float, ...] = (10.0, 100.0, 1000.0)) -> list[DeltaConvergenceRow]:
    """Return the three rows shown in the teacher's example."""

    return [delta_convolution(value) for value in cutoffs]

