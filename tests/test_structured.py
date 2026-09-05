"""Independent optimality tests for structured convex clustering objectives."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from ssnalclust.solvers import solve
from ssnalclust.structured import solve_biclustering, solve_sparse


def _oracle(
    X,
    row_weights,
    column_weights=None,
    feature_weights=None,
    gamma_row=1.0,
    gamma_col=1.0,
    alpha=1.0,
):
    cp = pytest.importorskip("cvxpy")
    U = cp.Variable(X.shape)
    objective = 0.5 * cp.sum_squares(U - X)
    for i in range(X.shape[0]):
        for j in range(i + 1, X.shape[0]):
            if row_weights[i, j] > 0:
                objective += gamma_row * row_weights[i, j] * cp.norm(U[i] - U[j], 2)
    if column_weights is not None:
        for i in range(X.shape[1]):
            for j in range(i + 1, X.shape[1]):
                if column_weights[i, j] > 0:
                    objective += gamma_col * column_weights[i, j] * cp.norm(U[:, i] - U[:, j], 2)
    if feature_weights is not None:
        for j in range(X.shape[1]):
            if feature_weights[j] > 0:
                objective += alpha * feature_weights[j] * cp.norm(U[:, j], 2)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-8, tol_feas=1e-8, tol_gap_rel=1e-8)
    assert problem.status == "optimal"
    return U.value, problem.value


@pytest.mark.parametrize("strength", [0.05, 0.4, 5.0])
def test_biclustering_matches_independent_convex_objective(strength):
    X = np.random.default_rng(91).normal(size=(5, 3))
    row = np.array(
        [[0, 2, 0, 0, 0], [2, 0, 0.5, 0, 0], [0, 0.5, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0.0]]
    )
    column = np.array([[0, 1, 0.25], [1, 0, 0], [0.25, 0, 0]])
    result = solve_biclustering(
        X, sparse.csr_matrix(row), column, strength, 0.7 * strength, tol=1e-8, max_iter=20000
    )
    expected, objective = _oracle(X, row, column, gamma_row=strength, gamma_col=0.7 * strength)
    assert result.converged
    assert result.kkt_residual <= 1e-8
    assert_allclose(result.centers, expected, atol=3e-5)
    assert_allclose(result.objective, objective, atol=3e-7)
    assert result.duals["row"].shape == (3, 3)
    assert result.duals["column"].shape == (2, 5)
    assert result.offset is None


@pytest.mark.parametrize("alpha", [0.05, 0.5, 5.0])
def test_sparse_matches_independent_centered_objective(alpha):
    X = np.random.default_rng(31).normal(size=(5, 3)) + [10, -3, 0.1]
    row = np.ones((5, 5)) - np.eye(5)
    feature = np.array([0.0, 0.5, 2.0])
    result = solve_sparse(
        X, row, gamma=0.1, alpha=alpha, feature_weights=feature, tol=1e-8, max_iter=20000
    )
    expected, objective = _oracle(
        X - X.mean(axis=0), row, feature_weights=feature, gamma_row=0.1, alpha=alpha
    )
    assert result.converged
    assert_allclose(result.centers, expected, atol=3e-5)
    assert_allclose(result.objective, objective, atol=3e-7)
    assert_allclose(result.offset, X.mean(axis=0))
    assert_allclose(result.feature_norms, np.linalg.norm(result.centers, axis=0))
    assert result.duals["feature"].shape == (3, 5)
    assert_allclose(result.centers.mean(axis=0), 0, atol=1e-14)


def test_biclustering_transpose_equivariance():
    X = np.random.default_rng(1).normal(size=(6, 4))
    row = np.ones((6, 6)) - np.eye(6)
    col = np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1)
    direct = solve_biclustering(X, row, col, 0.2, 0.7, tol=1e-8)
    transposed = solve_biclustering(X.T, col, row, 0.7, 0.2, tol=1e-8)
    assert direct.converged and transposed.converged
    assert_allclose(direct.centers, transposed.centers.T, atol=1e-12)
    assert_allclose(direct.objective, transposed.objective, atol=1e-12)


def test_biclustering_column_strength_zero_reduces_to_clustering():
    X = np.random.default_rng(22).normal(size=(5, 3))
    reference = solve(X, gamma=0.2, tol=1e-8)
    result = solve_biclustering(X, gamma_row=0.2, gamma_col=0, tol=1e-8)
    assert result.converged and reference.converged
    assert_allclose(result.centers, reference.centers, atol=1e-6)
    assert_allclose(result.objective, reference.objective, atol=1e-7)
    assert_array_equal(result.duals["column"], np.zeros((3, 5)))


def test_sparse_alpha_zero_reduces_to_centered_clustering():
    X = np.random.default_rng(22).normal(size=(5, 3)) + 30
    reference = solve(X - X.mean(axis=0), gamma=0.2, tol=1e-8)
    result = solve_sparse(X, gamma=0.2, alpha=0, tol=1e-8)
    assert result.converged and reference.converged
    assert_allclose(result.centers, reference.centers, atol=1e-6)
    assert_allclose(result.objective, reference.objective, atol=1e-7)


def test_sparse_without_edges_has_exact_group_shrinkage_solution():
    X = np.array([[10.0, 4.0, 7.0], [12.0, 5.0, 7.0], [14.0, 6.0, 7.0]])
    centered = X - X.mean(axis=0)
    feature = np.array([1.0, 3.0, 0.0])
    norms = np.linalg.norm(centered, axis=0)
    expected = np.zeros_like(centered)
    active = norms > 0
    expected[:, active] = centered[:, active] * np.maximum(0, 1 - feature[active] / norms[active])
    result = solve_sparse(X, sparse.csr_matrix((3, 3)), feature_weights=feature, tol=1e-9)
    assert result.converged
    assert_allclose(result.centers, expected, atol=1e-8)
    assert result.feature_norms[1] < 1e-8
    assert result.feature_norms[2] == 0


def test_zero_graphs_and_zero_strengths():
    X = np.arange(12.0).reshape(4, 3)
    for result in [
        solve_biclustering(X, gamma_row=0, gamma_col=0),
        solve_biclustering(X, np.zeros((4, 4)), np.zeros((3, 3))),
    ]:
        assert result.converged and result.n_iter == 0
        assert result.objective == result.kkt_residual == 0
        assert_array_equal(result.centers, X)
    sparse_result = solve_sparse(X, gamma=0, alpha=0)
    assert sparse_result.converged and sparse_result.n_iter == 0
    assert_array_equal(sparse_result.centers, X - X.mean(axis=0))
    assert_array_equal(X, np.arange(12.0).reshape(4, 3))


def test_singleton_axes_and_constant_features():
    assert_allclose(solve_biclustering([[4.0]]).centers, [[4.0]])
    result = solve_sparse([[4.0, 8.0, 9.0]])
    assert result.converged
    assert_array_equal(result.centers, [[0, 0, 0]])
    assert_array_equal(result.feature_norms, [0, 0, 0])


def test_returned_biclustering_duals_satisfy_independent_kkt_equations():
    X = np.random.default_rng(29).normal(size=(4, 3))
    result = solve_biclustering(X, gamma_row=0.2, gamma_col=0.15, tol=1e-9)
    gradient = result.centers - X
    for strength, name, count, transpose in [(0.2, "row", 4, False), (0.15, "column", 3, True)]:
        view = result.centers.T if transpose else result.centers
        correction = np.zeros_like(view)
        edges = [(i, j) for i in range(count) for j in range(i + 1, count)]
        for (i, j), dual in zip(edges, result.duals[name]):
            difference = view[i] - view[j]
            assert np.linalg.norm(dual) <= strength + 1e-14
            assert_allclose(dual @ difference, strength * np.linalg.norm(difference), atol=1e-7)
            correction[i] += dual
            correction[j] -= dual
        gradient += correction.T if transpose else correction
    assert_allclose(gradient, 0, atol=1e-7)


def test_returned_sparse_duals_satisfy_independent_kkt_equations():
    X = np.random.default_rng(29).normal(size=(4, 3)) + 5
    result = solve_sparse(X, gamma=0.2, alpha=0.15, tol=1e-9)
    gradient = result.centers - (X - X.mean(axis=0))
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    for (i, j), dual in zip(edges, result.duals["row"]):
        difference = result.centers[i] - result.centers[j]
        assert np.linalg.norm(dual) <= 0.2 + 1e-14
        assert_allclose(dual @ difference, 0.2 * np.linalg.norm(difference), atol=1e-7)
        gradient[i] += dual
        gradient[j] -= dual
    for j, dual in enumerate(result.duals["feature"]):
        assert np.linalg.norm(dual) <= 0.15 + 1e-14
        assert_allclose(dual @ result.centers[:, j], 0.15 * result.feature_norms[j], atol=1e-7)
        gradient[:, j] += dual
    assert_allclose(gradient, 0, atol=1e-7)


@pytest.mark.parametrize("solver", [solve_sparse, solve_biclustering])
def test_exhaustion_is_not_success(solver):
    result = solver(np.random.default_rng(2).normal(size=(4, 3)), max_iter=1, tol=1e-12)
    assert not result.converged
    assert result.n_iter == 1
    assert len(result.history) == 2
    assert result.history[-1]["kkt_residual"] == result.kkt_residual


@pytest.mark.parametrize("solver", [solve_sparse, solve_biclustering])
@pytest.mark.parametrize("X", [[], [1, 2], [[np.inf]], [[np.nan]], [[1j]], np.empty((2, 0))])
def test_invalid_data(solver, X):
    with pytest.raises(ValueError, match="X"):
        solver(X)


@pytest.mark.parametrize("solver", [solve_sparse, solve_biclustering])
@pytest.mark.parametrize(
    "kwargs", [{"tol": 0}, {"tol": np.nan}, {"max_iter": True}, {"max_iter": 0}, {"max_iter": 1.5}]
)
def test_invalid_controls(solver, kwargs):
    with pytest.raises(ValueError):
        solver([[1.0, 2.0], [3.0, 4.0]], **kwargs)


@pytest.mark.parametrize("name", ["gamma", "alpha"])
@pytest.mark.parametrize("value", [-1, np.inf, np.nan, True, 1j])
def test_invalid_sparse_strength(name, value):
    with pytest.raises(ValueError, match=name):
        solve_sparse([[1.0, 2.0], [3.0, 4.0]], **{name: value})


@pytest.mark.parametrize("name", ["gamma_row", "gamma_col"])
@pytest.mark.parametrize("value", [-1, np.inf, np.nan, True, 1j])
def test_invalid_biclustering_strength(name, value):
    with pytest.raises(ValueError, match=name):
        solve_biclustering([[1.0, 2.0], [3.0, 4.0]], **{name: value})


@pytest.mark.parametrize("feature", [[1], [-1, 1], [1, np.nan], [1, np.inf], [1j, 1], [[1, 1]]])
def test_invalid_feature_weights(feature):
    with pytest.raises(ValueError, match="feature_weights"):
        solve_sparse([[1.0, 2.0], [3.0, 4.0]], feature_weights=feature)


def test_invalid_graphs():
    with pytest.raises(ValueError, match="symmetric"):
        solve_biclustering([[1.0, 2.0], [3.0, 4.0]], column_weights=[[0, 1], [0, 0]])
    with pytest.raises(ValueError, match="shape"):
        solve_sparse([[1.0, 2.0], [3.0, 4.0]], weights=np.zeros((3, 3)))
