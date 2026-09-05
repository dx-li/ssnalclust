"""Missing-data solutions against an independent convex modeling oracle."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust.missing import solve_missing
from ssnalclust.solvers import solve


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_missing_matches_cvxpy_objective(penalty):
    cp = pytest.importorskip("cvxpy")
    X = np.array([[0.0, np.nan], [1.0, 2.0], [np.nan, 3.0], [3.0, 1.0]])
    mask = np.isfinite(X)
    weights = np.ones((4, 4)) - np.eye(4)
    gamma = 0.35
    result = solve_missing(X, weights, gamma, penalty, tol=1e-8, max_iter=50000)
    U = cp.Variable(X.shape)
    order = {"l1": 1, "l2": 2, "linf": "inf"}[penalty]
    objective = 0.5 * cp.sum_squares(cp.multiply(mask, U - np.nan_to_num(X)))
    objective += gamma * sum(cp.norm(U[i] - U[j], order) for i in range(4) for j in range(i + 1, 4))
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    assert problem.status == "optimal"
    assert result.converged
    assert result.kkt_residual <= 1e-8
    assert_allclose(result.objective, problem.value, atol=2e-7)


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_fully_observed_matches_complete_data_solver(penalty):
    X = np.random.default_rng(23).normal(size=(5, 2))
    weights = np.ones((5, 5)) - np.eye(5)
    missing = solve_missing(X, weights, gamma=0.2, penalty=penalty, tol=1e-8)
    complete = solve(
        X, weights, gamma=0.2, penalty=penalty, solver="admm", tol=1e-8, max_iter=10000
    )
    assert missing.converged
    assert_allclose(missing.centers, complete.centers, atol=1e-6)


def test_unobserved_values_have_no_effect():
    observed = np.array([[True, False], [True, True], [False, True]])
    X = np.array([[0.0, np.nan], [1.0, 2.0], [np.inf, 3.0]])
    alternative = np.where(observed, X, -1e15)
    weights = np.ones((3, 3)) - np.eye(3)
    first = solve_missing(X, weights, observed=observed, tol=1e-8)
    second = solve_missing(alternative, weights, observed=observed, tol=1e-8)
    assert_allclose(first.centers, second.centers, atol=0, rtol=0)
    assert first.objective == second.objective
    assert first.history == second.history


def test_partial_observations_recover_common_centroid():
    X = [[2, np.nan], [np.nan, 4]]
    result = solve_missing(X, [[0, 1], [1, 0]])
    assert result.converged
    assert_allclose(result.centers, [[2, 4], [2, 4]])
    assert result.objective == 0


@pytest.mark.parametrize(
    "X,weights,gamma",
    [
        ([[np.nan], [np.nan]], [[0, 1], [1, 0]], 1),
        ([[0], [np.nan]], [[0, 0], [0, 0]], 1),
        ([[0], [np.nan]], [[0, 1], [1, 0]], 0),
    ],
)
def test_unidentifiable_components_rejected(X, weights, gamma):
    with pytest.raises(ValueError, match="unidentifiable"):
        solve_missing(X, weights, gamma=gamma)


def test_mask_validation_and_explicit_graph():
    with pytest.raises(ValueError, match="explicit"):
        solve_missing([[0], [1]], None)
    with pytest.raises(ValueError, match="finite"):
        solve_missing([[np.nan]], [[0]], observed=[[True]])
    for mask in [[[1]], [[True, False]]]:
        with pytest.raises(ValueError, match="boolean"):
            solve_missing([[0]], [[0]], observed=mask)


def test_iteration_limit_is_reported():
    result = solve_missing(
        [[0, np.nan], [1, 2], [3, 4]], np.ones((3, 3)) - np.eye(3), max_iter=1, tol=1e-14
    )
    assert result.n_iter == 1
    assert not result.converged
    assert len(result.history) == 2


def test_zero_gamma_observed_recovery():
    X = np.array([[0.0, 1], [3, 4]])
    result = solve_missing(X, [[0, 1], [1, 0]], gamma=0)
    assert result.converged
    assert_allclose(result.centers, X)
    assert result.objective == 0


def test_twenty_thousand_isolated_components():
    from scipy import sparse

    X = np.arange(40000, dtype=float).reshape(20000, 2)
    result = solve_missing(X, sparse.csr_matrix((len(X), len(X))))
    assert result.converged
    assert result.n_iter == 0
    assert_allclose(result.centers, X)
