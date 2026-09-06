"""Independent optimality tests for structured convex clustering objectives."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from ssnalclust.solvers import solve
from ssnalclust.structured import _diagnostics, _Term, solve_biclustering, solve_sparse


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


def _independent_certificate(X, result, row, column=None, feature=None, gr=1.0, gc=1.0, alpha=1.0):
    """Reconstruct adjoints and the dual objective without package operators."""
    adjoint = np.zeros_like(X)
    for name, weights, strength, transpose in [
        ("row", row, gr, False),
        ("column", column, gc, True),
    ]:
        if weights is None:
            continue
        correction = adjoint.T if transpose else adjoint
        edge = 0
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if weights[i, j] > 0:
                    z = result.duals[name][edge]
                    assert np.linalg.norm(z) <= strength * weights[i, j] + 1e-12
                    correction[i] += z
                    correction[j] -= z
                    edge += 1
    if feature is not None:
        assert np.all(np.linalg.norm(result.duals["feature"], axis=1) <= alpha * feature + 1e-12)
        adjoint += result.duals["feature"].T
    dual_value = np.sum(X * adjoint) - 0.5 * np.sum(adjoint**2)
    assert_allclose(result.dual_objective, dual_value, atol=1e-12, rtol=1e-12)
    assert_allclose(result.gap, result.objective - dual_value, atol=2e-12, rtol=1e-5)
    assert result.gap >= 0
    assert_allclose(
        result.relative_gap, result.gap / (1 + abs(result.objective) + abs(result.dual_objective))
    )
    for record in result.history:
        assert record["gap"] >= 0
        assert_allclose(
            record["relative_gap"],
            record["gap"] / (1 + abs(record["objective"]) + abs(record["dual_objective"])),
        )


def _dual_oracle(X, row, column=None, feature=None, gr=1.0, gc=1.0, alpha=1.0):
    """Independently optimize the constrained dual, not a primal surrogate."""
    cp = pytest.importorskip("cvxpy")
    entries = [[cp.Constant(0.0) for _ in range(X.shape[1])] for _ in range(X.shape[0])]
    constraints = []
    for weights, strength, transpose in [(row, gr, False), (column, gc, True)]:
        if weights is None:
            continue
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if weights[i, j] > 0:
                    z = cp.Variable(X.shape[0] if transpose else X.shape[1])
                    constraints.append(cp.norm(z, 2) <= strength * weights[i, j])
                    for k in range(z.size):
                        if transpose:
                            entries[k][i] += z[k]
                            entries[k][j] -= z[k]
                        else:
                            entries[i][k] += z[k]
                            entries[j][k] -= z[k]
    if feature is not None:
        for j, weight in enumerate(feature):
            z = cp.Variable(X.shape[0])
            constraints.append(cp.norm(z, 2) <= alpha * weight)
            for i in range(X.shape[0]):
                entries[i][j] += z[i]
    adjoint = cp.bmat(entries)
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.multiply(X, adjoint)) - 0.5 * cp.sum_squares(adjoint)), constraints
    )
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    assert problem.status == "optimal"
    return problem.value


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
    assert result.relative_gap <= 1e-8
    assert_allclose(result.centers, expected, atol=3e-5)
    assert_allclose(result.objective, objective, atol=3e-7)
    assert result.duals["row"].shape == (3, 3)
    assert result.duals["column"].shape == (2, 5)
    assert result.offset is None
    _independent_certificate(X, result, row, column, gr=strength, gc=0.7 * strength)


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
    assert result.relative_gap <= 1e-8
    assert_allclose(result.centers, expected, atol=3e-5)
    assert_allclose(result.objective, objective, atol=3e-7)
    assert_allclose(result.offset, X.mean(axis=0))
    assert_allclose(result.feature_norms, np.linalg.norm(result.centers, axis=0))
    assert result.duals["feature"].shape == (3, 5)
    assert_allclose(result.centers.mean(axis=0), 0, atol=1e-14)
    _independent_certificate(X - X.mean(axis=0), result, row, feature=feature, gr=0.1, alpha=alpha)


@pytest.mark.parametrize("model", ["biclustering", "sparse"])
@pytest.mark.parametrize("strength", [0.1, 3.0])
def test_certificates_bracket_independently_solved_dual(model, strength):
    X = np.random.default_rng(803).normal(size=(4, 3))
    row = np.diag([1.0, 0.2, 2.0], 1) + np.diag([1.0, 0.2, 2.0], -1)
    if model == "biclustering":
        column = np.array([[0.0, 0.7, 0.0], [0.7, 0.0, 1.2], [0.0, 1.2, 0.0]])
        result = solve_biclustering(X, row, column, strength, 0.4, tol=1e-9)
        dual_optimum = _dual_oracle(X, row, column, gr=strength, gc=0.4)
    else:
        feature = np.array([0.0, 0.4, 1.0])
        result = solve_sparse(X, row, gamma=strength, alpha=0.6, feature_weights=feature, tol=1e-9)
        dual_optimum = _dual_oracle(
            X - X.mean(axis=0), row, feature=feature, gr=strength, alpha=0.6
        )
    assert result.converged
    assert result.relative_gap <= 1e-9
    assert result.dual_objective <= dual_optimum + 2e-8
    assert dual_optimum <= result.objective + 2e-8
    assert_allclose(result.dual_objective, dual_optimum, atol=2e-8)


@pytest.mark.parametrize("model", ["biclustering", "sparse"])
def test_small_normalized_kkt_alone_cannot_certify_large_objective_error(model):
    if model == "biclustering":
        result = solve_biclustering([[0.0, 1e8]], gamma_row=0.0, gamma_col=1.0)
    else:
        result = solve_sparse([[-1e8], [1e8]], gamma=0.0, alpha=1.0)
    initial = result.history[0]
    # Historical KKT-only stopping accepted this untouched initial point.
    assert initial["kkt_residual"] < 1e-6
    assert initial["relative_gap"] > 0.9
    assert result.n_iter > 0
    assert result.converged
    assert result.relative_gap <= 1e-6
    assert result.kkt_residual <= 1e-6


def test_gap_decomposition_retains_error_lost_in_objective_subtraction():
    X = np.array([[1e8]])
    U = np.array([[1e8 - 1 + 1e-4]])
    term = _Term("feature", np.ones(1), lambda u: u.T, lambda z: z.T, 1.0)
    status = _diagnostics(X, U, {"feature": np.ones((1, 1))}, [term])
    exact_residual_gap = 0.5 * float(U[0, 0] - X[0, 0] + 1.0) ** 2
    assert exact_residual_gap > 0
    assert status["gap"] == exact_residual_gap
    assert status["objective"] - status["dual_objective"] == 0.0


def test_infeasible_dual_cannot_be_reported_as_a_lower_bound():
    term = _Term("feature", np.ones(1), lambda u: u.T, lambda z: z.T, 1.0)
    with pytest.raises(ValueError, match="feasible"):
        _diagnostics(np.zeros((1, 1)), np.zeros((1, 1)), {"feature": np.array([[2.0]])}, [term])


def test_dual_evaluation_ignores_common_biclustering_offset():
    X = np.array([[0.0, 2.0], [4.0, 1.0], [3.0, -2.0]])
    direct = solve_biclustering(X, gamma_row=0.3, gamma_col=0.4, tol=1e-9)
    shifted = solve_biclustering(X + 1e7, gamma_row=0.3, gamma_col=0.4, tol=1e-8)
    assert direct.converged and shifted.converged
    assert_allclose(shifted.objective, direct.objective, atol=2e-8)
    assert_allclose(shifted.dual_objective, direct.dual_objective, atol=2e-8)


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
