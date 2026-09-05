"""Numerical regressions found by independent whole-library review."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ssnalclust.generalized import solve_generalized
from ssnalclust.missing import solve_missing
from ssnalclust.prox import project_dual
from ssnalclust.solvers import solve


def test_zero_fusion_preserves_observations_exactly():
    X = np.random.default_rng(5).normal(size=(5, 3))
    assert_array_equal(solve(X, gamma=0).centers, X)


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_certificate_describes_returned_centers_after_offset_rounding(solver):
    X = np.array([[1e16], [1e16 + 4]])
    result = solve(X, gamma=0.5, solver=solver, tol=1e-9, max_iter=100)
    differences = result.centers[0] - result.centers[1]
    actual_objective = 0.5 * np.sum((result.centers - X) ** 2) + 0.5 * np.linalg.norm(differences)
    assert_allclose(result.objective, actual_objective, rtol=1e-14, atol=0)
    gradient = result.centers - X
    adjoint = np.vstack((result.dual, -result.dual))
    stationarity = np.linalg.norm(gradient + adjoint) / (
        1 + np.linalg.norm(gradient) + np.linalg.norm(adjoint)
    )
    assert result.kkt_residual >= stationarity - 1e-14
    if stationarity > 1e-9:
        assert not result.converged


@pytest.mark.parametrize("scale,radius", [(1e200, 1.0), (1e-200, 1e-201), (1e308, 1.0)])
def test_l2_projection_handles_finite_extreme_scales(scale, radius):
    # Four equal coordinates have norm 2*scale, including when that norm
    # itself lies above the representable floating-point range.
    actual = project_dual([[scale, -scale, scale, -scale]], [radius], "l2")
    assert_allclose(
        actual, [[radius / 2, -radius / 2, radius / 2, -radius / 2]], rtol=1e-14, atol=0
    )


def test_linf_dual_projection_avoids_large_threshold_cancellation():
    actual = project_dual([[1e200, -1e200]], [1.0], "linf")
    assert_allclose(actual, [[0.5, -0.5]], rtol=1e-14, atol=0)


def test_missing_identifiability_uses_effective_positive_radii():
    weights = [[0, 1e-200], [1e-200, 0]]
    with pytest.raises(ValueError, match="unidentifiable"):
        solve_missing([[3.0], [np.nan]], weights, gamma=1e-200)


@pytest.mark.parametrize("count", [1e4, 1e6])
def test_poisson_proximal_root_handles_large_but_ordinary_counts(count):
    gamma = 0.1
    result = solve_generalized(
        [[count], [2 * count]], loss="poisson", gamma=gamma, tol=1e-6, max_iter=1000
    )
    # With distinct fitted values, scalar stationarity is exp(u0)=x0+gamma
    # and exp(u1)=x1-gamma. These data are far from the fusion threshold.
    expected = np.log([[count + gamma], [2 * count - gamma]])
    assert result.converged
    assert_allclose(result.centers, expected, atol=1e-8, rtol=0)


def test_logistic_initialization_handles_proportions_near_one():
    X = np.array([[1.0], [np.nextafter(1.0, 0.0)]])
    result = solve_generalized(X, loss="logistic", gamma=1.0, tol=1e-9)
    expected = np.log(X.mean()) - np.log((1 - X).mean())
    assert np.isfinite(result.centers).all()
    assert result.converged
    assert_allclose(result.centers, expected, atol=1e-8, rtol=0)
