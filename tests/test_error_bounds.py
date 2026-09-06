"""Stable gap evaluation and scientifically interpretable centroid bounds."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust import solve
from ssnalclust.graph import graph_from_weights
from ssnalclust.solvers import _diagnostics


def test_gap_survives_primal_dual_cancellation():
    x = np.array([[-1e8], [1e8]])
    u = np.array([[1e-4], [1e-4]])
    b, _ = graph_from_weights(None, 2)
    z = np.array([[-1e8]])
    info = _diagnostics(x, u, z, b, np.array([1e8]), "l2", np.ones((2, 1)))
    assert info["objective"] - info["dual_objective"] == 0.0
    assert info["gap"] > 0
    assert_allclose(info["gap"], 1e-8, rtol=5e-4)


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
@pytest.mark.parametrize("mass", [[1.0, 1.0], [0.1, 3.0]])
def test_bound_contains_error_against_weighted_analytic_solution(solver, mass):
    x = np.array([[0.0, 1.0], [3.0, 5.0]])
    gamma = 0.6
    displacement = x[1] - x[0]
    length = np.linalg.norm(displacement)
    direction = displacement / length
    first = x[0] + gamma / mass[0] * direction
    second = x[1] - gamma / mass[1] * direction
    if gamma * (1 / mass[0] + 1 / mass[1]) >= length:
        first = second = np.average(x, axis=0, weights=mass)
    truth = np.vstack([first, second])
    result = solve(x, sample_weight=mass, gamma=gamma, solver=solver, max_iter=2, tol=1e-12)
    error = np.linalg.norm(result.centers - truth)
    assert error <= result.center_error_bound + 1e-7
    assert result.strong_convexity == min(mass)
    assert_allclose(result.center_error_bound, np.sqrt(2 * result.gap / min(mass)))


def test_zero_strength_bound_is_zero():
    result = solve([[1.0, 3.0], [2.0, 4.0]], gamma=0)
    assert result.center_error_bound == 0


def test_error_bound_does_not_square_tiny_masses_before_dividing():
    result = solve([[0.0], [1.0]], gamma=1e-300, sample_weight=[1e-300, 1e-300], x0=[[0.8], [0.8]])
    # Small objective/gradient does not imply small centroid error for this
    # almost-flat fidelity. The absolute-error diagnostic must retain that.
    assert result.gap > 0
    assert result.center_error_bound >= np.linalg.norm(result.centers - 0.5)
