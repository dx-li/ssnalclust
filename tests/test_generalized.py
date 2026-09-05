"""Independent tests for generalized-loss convex clustering."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.special import expit

from ssnalclust.generalized import (
    _certificate_dual,
    _fidelity,
    _generalized_diagnostics,
    _log_mean_kl,
    _prox_fidelity,
    solve_generalized,
)


@pytest.mark.parametrize("loss", ["huber", "logistic", "poisson"])
@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
@pytest.mark.parametrize("gamma", [0.15, 2.0])
def test_generalized_against_independent_cvxpy(loss, penalty, gamma):
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(382)
    if loss == "huber":
        x = rng.normal(size=(5, 2))
        x[-1] += 6
    elif loss == "logistic":
        x = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    else:
        x = np.array([[0.0, 1.0], [2.0, 0.0], [1.0, 3.0], [4.0, 2.0], [6.0, 1.0]])
    w = np.zeros((5, 5))
    edges = [(0, 1), (0, 3), (1, 2), (2, 3), (3, 4)]
    for i, j in edges:
        w[i, j] = w[j, i] = rng.uniform(0.6, 1.4)
    u = cp.Variable(x.shape)
    if loss == "huber":
        objective = 0.5 * cp.sum(cp.huber(u - x, 0.7))
    elif loss == "logistic":
        objective = cp.sum(cp.logistic(u) - cp.multiply(x, u))
    else:
        objective = cp.sum(cp.exp(u) - cp.multiply(x, u))
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    objective += gamma * sum(w[i, j] * cp.norm(u[i] - u[j], order) for i, j in edges)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    assert problem.status == cp.OPTIMAL
    result = solve_generalized(x, w, gamma, loss, 0.7, penalty, tol=2e-7, max_iter=20000)
    assert result.converged
    assert result.kkt_residual <= 2e-7
    assert result.relative_gap <= 2e-7
    assert result.gap >= 0
    assert result.dual_objective <= problem.value + 3e-7
    assert_allclose(result.gap, result.objective - result.dual_objective, atol=1e-12, rtol=1e-5)
    assert_allclose(result.objective, problem.value, atol=3e-5, rtol=2e-6)
    # Huber centers need not be unique; the two likelihoods are strictly convex.
    if loss != "huber":
        assert_allclose(result.centers, u.value, atol=3e-4, rtol=3e-4)
    expected_means = (
        result.centers
        if loss == "huber"
        else (expit(result.centers) if loss == "logistic" else np.exp(result.centers))
    )
    assert_allclose(result.fitted_means, expected_means)
    assert len(result.history) == result.n_iter + 1
    assert np.isfinite(result.dual).all()


@pytest.mark.parametrize("loss", ["huber", "logistic", "poisson"])
@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_loss_specific_dual_against_independent_cvxpy(loss, penalty):
    cp = pytest.importorskip("cvxpy")
    if loss == "huber":
        x = np.array([[-1.0, 0.2], [0.7, 3.0], [2.0, -2.0]])
    elif loss == "logistic":
        x = np.array([[0.0, 0.2], [1.0, 0.7], [0.4, 1.0]])
    else:
        x = np.array([[0.0, 2.0], [3.0, 0.7], [1.0, 4.0]])
    # Explicit independent incidence, avoiding package graph construction.
    incidence = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    z = cp.Variable((3, 2))
    divergence = incidence.T @ z
    dual_order = {"l1": np.inf, "l2": 2, "linf": 1}[penalty]
    constraints = [cp.norm(z[e], dual_order) <= 0.3 for e in range(3)]
    if loss == "huber":
        constraints.append(cp.abs(divergence) <= 0.8)
        objective = cp.sum(cp.multiply(x, divergence)) - 0.5 * cp.sum_squares(divergence)
    elif loss == "logistic":
        q = x - divergence
        constraints += [q >= 0, q <= 1]
        objective = cp.sum(cp.entr(q) + cp.entr(1 - q))
    else:
        q = x - divergence
        constraints.append(q >= 0)
        objective = cp.sum(q + cp.entr(q))
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    assert problem.status == cp.OPTIMAL
    result = solve_generalized(
        x, gamma=0.3, loss=loss, huber_delta=0.8, penalty=penalty, tol=1e-8, max_iter=20000
    )
    assert result.converged
    assert result.relative_gap <= 1e-8
    assert result.dual_objective <= problem.value + 1e-7
    assert result.objective >= problem.value - 1e-7
    assert_allclose(result.dual_objective, problem.value, atol=3e-7)
    actual_s = incidence.T @ result.dual
    assert np.all(np.linalg.norm(result.dual, ord=dual_order, axis=1) <= 0.3 + 1e-14)
    if loss == "huber":
        assert np.all(np.abs(actual_s) <= 0.8)
        direct_dual = np.sum(x * actual_s) - 0.5 * np.sum(actual_s**2)
    else:
        actual_q = x - actual_s
        assert np.all(actual_s <= x)
        if loss == "logistic":
            assert np.all(actual_s >= x - 1)

        def xlogx(v):
            active = v > 0
            out = np.zeros_like(v)
            out[active] = v[active] * np.log(v[active])
            return out

        direct_dual = (
            -np.sum(xlogx(actual_q) + xlogx(1 - actual_q))
            if loss == "logistic"
            else np.sum(actual_q - xlogx(actual_q))
        )
    assert_allclose(result.dual_objective, direct_dual, atol=1e-12)
    assert 0 <= result.dual_scale <= 1


@pytest.mark.parametrize("loss", ["logistic", "poisson"])
def test_boundary_wrong_sign_forces_valid_zero_dual(loss):
    x = np.array([[0.0], [1.0]])
    b = sparse.csr_matrix([[1.0, -1.0]])
    repaired, divergence, scale = _certificate_dual(
        x, b, np.array([[0.1]]), np.ones(1), loss, 1.0, "l2"
    )
    assert scale == 0
    assert_allclose(repaired, 0)
    assert_allclose(divergence, 0)
    # Correct signs do not require the weak zero certificate.
    _, divergence, scale = _certificate_dual(x, b, np.array([[-0.1]]), np.ones(1), loss, 1.0, "l2")
    assert scale == 1
    assert np.all(divergence <= x)


@pytest.mark.parametrize("loss", ["huber", "logistic", "poisson"])
def test_global_rescaling_enforces_fidelity_domain_on_returned_arrays(loss):
    x = np.array([[0.2], [0.8]])
    b = sparse.csr_matrix([[1.0, -1.0]])
    repaired, divergence, scale = _certificate_dual(
        x, b, np.array([[0.9]]), np.ones(1), loss, 0.3, "l2"
    )
    assert 0 < scale < 1
    assert_allclose(divergence, b.T @ repaired)
    if loss == "huber":
        assert np.all(np.abs(divergence) <= 0.3)
    elif loss == "logistic":
        assert np.all((divergence <= x) & (divergence >= x - 1))
    else:
        assert np.all(divergence <= x)


@pytest.mark.parametrize("loss", ["logistic", "poisson"])
def test_small_likelihood_gradient_does_not_imply_small_gap(loss):
    x = np.full((2, 1), 1e-8)
    u = np.full_like(x, -1e8)
    b = sparse.csr_matrix([[1.0, -1.0]])
    status, _, _ = _generalized_diagnostics(x, u, np.zeros((1, 1)), b, np.ones(1), loss, 1.0, "l2")
    assert status["kkt_residual"] < 1e-6
    assert status["relative_gap"] > 0.5
    assert status["gap"] > 1.9


def test_huber_requires_gap_when_initial_edge_residual_is_small():
    result = solve_generalized([[-1e8], [1e8]], loss="huber", gamma=0.5, tol=1e-6)
    assert result.history[0]["kkt_residual"] < 1e-6
    assert result.history[0]["relative_gap"] > 0.9
    assert result.converged and result.n_iter > 0
    assert result.relative_gap <= 1e-6


def test_kl_gap_matches_high_precision_when_large_terms_cancel():
    q = 1e6
    mean = q + 1e-7
    with localcontext() as context:
        context.prec = 80
        dq, dm = Decimal.from_float(q), Decimal.from_float(mean)
        expected = float(dq * (dq / dm).ln() - dq + dm)
    actual = _log_mean_kl(np.array([q]), np.array([mean]), np.log([mean]))[0]
    assert actual > 0
    assert_allclose(actual, expected, rtol=1e-14)


def test_kl_gap_stays_finite_after_exponential_underflow():
    q = np.array([1e-300])
    actual = _log_mean_kl(q, np.zeros(1), np.array([-1000.0]))
    assert_allclose(actual, q * (np.log(q) + 999.0), rtol=1e-14, atol=0.0)


def test_logistic_tail_and_kl_preserve_subnormal_probabilities():
    u = np.array([[-710.0]])
    x = np.array([[1e-310]])
    _, gradient, means = _fidelity(u, x, "logistic", 1.0)
    expected_mean = np.exp(-710.0)
    assert means[0, 0] > 0
    assert_allclose(means[0, 0], expected_mean, rtol=1e-14, atol=0.0)
    assert_allclose(gradient, expected_mean - x, rtol=1e-14, atol=0.0)
    # Also repair a caller's prematurely underflowed sigmoid argument.
    actual = _log_mean_kl(x, np.zeros_like(x), u)
    expected = expected_mean + x * (np.log(x) - u - 1)
    assert np.all(actual > 0)
    assert_allclose(actual, expected, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("loss", ["huber", "logistic", "poisson"])
def test_gamma_zero_interior_analytic_solution(loss):
    x = np.array([[0.2, 0.5], [0.7, 0.8]])
    result = solve_generalized(x, gamma=0, loss=loss, tol=1e-10)
    expected = (
        x if loss == "huber" else (np.log(x) - np.log1p(-x) if loss == "logistic" else np.log(x))
    )
    assert result.converged
    assert result.n_iter == 0
    assert_allclose(result.centers, expected, atol=1e-14)
    assert_allclose(result.fitted_means, x)


@pytest.mark.parametrize("loss", ["huber", "logistic", "poisson"])
@pytest.mark.parametrize("step", [0.01, 0.5, 10.0])
def test_fidelity_prox_scalar_optimality(loss, step):
    y = np.array([[-40.0, -3.0, 0.0, 4.0, 40.0]])
    x = np.array([[0.0, 0.1, 0.5, 1.0, 0.9]])
    result = _prox_fidelity(y, x, step, loss, 0.7)
    gradient = (
        np.clip(result - x, -0.7, 0.7)
        if loss == "huber"
        else (expit(result) - x if loss == "logistic" else np.exp(result) - x)
    )
    assert_allclose(result - y + step * gradient, 0.0, atol=2e-10)


def test_poisson_prox_brackets_large_positive_natural_parameter():
    y = np.array([[1000.0, -1000.0, 1e100]])
    x = np.array([[0.0, 1.0, 2.0]])
    result = _prox_fidelity(y, x, 1.0, "poisson", 1.0)
    assert np.isfinite(result).all()
    assert_allclose((result - y + np.exp(result) - x) / (1 + np.abs(y)), 0.0, atol=2e-13)


@pytest.mark.parametrize("count", [1e4, 1e6, 1e10])
def test_large_poisson_counts_return_actual_kkt_instead_of_root_failure(count):
    x = np.array([[count], [2 * count]])
    result = solve_generalized(x, loss="poisson", gamma=0.1, max_iter=100, tol=1e-6)
    assert np.isfinite(result.centers).all()
    assert np.isfinite(result.objective)
    # The single oriented edge is U[0]-U[1]. Recompute stationarity without
    # the proximal helper or a relaxed high-count stopping threshold.
    gradient = np.exp(result.centers) - x
    bt = np.vstack([result.dual[0], -result.dual[0]])
    stationarity = np.linalg.norm(gradient + bt) / (
        1 + np.linalg.norm(gradient) + np.linalg.norm(bt)
    )
    assert result.kkt_residual >= stationarity - 1e-15
    assert result.converged == (result.kkt_residual <= 1e-6)
    # Both points remain unfused; KKT gives means x[0]+gamma, x[1]-gamma.
    expected = np.log(x + np.array([[0.1], [-0.1]]))
    assert_allclose(result.centers, expected, atol=1e-11, rtol=1e-12)


def test_logistic_initial_mean_rounding_near_one_remains_finite():
    x = np.array([[1.0], [np.nextafter(1.0, 0.0)]])
    result = solve_generalized(x, loss="logistic", gamma=0.1)
    assert np.isfinite(result.centers).all()
    failure_mean = (1 - x).sum() / 2
    expected = np.log(x.mean()) - np.log(failure_mean)
    assert_allclose(result.centers, expected)
    _, gradient, _ = _fidelity(result.centers, x, "logistic", 1.0)
    assert gradient[0, 0] < 0 < gradient[1, 0]
    assert_allclose(gradient.sum(), 0.0, atol=1e-30)


@pytest.mark.parametrize("loss", ["logistic", "poisson"])
def test_subnormal_observations_have_finite_initial_natural_parameters(loss):
    x = np.array([[0.0], [np.nextafter(0.0, 1.0)]])
    result = solve_generalized(x, loss=loss, gamma=1.0)
    assert np.isfinite(result.centers).all()
    assert_allclose(result.centers, np.log(x[1, 0]) - np.log(2.0))


@pytest.mark.parametrize("loss", ["logistic", "poisson"])
def test_many_isolated_vertices_keep_each_unpenalized_solution(loss):
    x = np.random.default_rng(934).uniform(0.1, 0.9, size=(10000, 2))
    weights = sparse.csr_matrix((len(x), len(x)))
    result = solve_generalized(x, weights=weights, loss=loss, tol=1e-10)
    expected = np.log(x) - np.log1p(-x) if loss == "logistic" else np.log(x)
    assert result.converged
    assert result.n_iter == 0
    assert_allclose(result.centers, expected, atol=1e-14)


@pytest.mark.parametrize(
    "loss,x",
    [("logistic", [[0.0], [0.0]]), ("logistic", [[1.0], [1.0]]), ("poisson", [[0.0], [0.0]])],
)
def test_unattained_likelihood_infimum_is_rejected(loss, x):
    with pytest.raises(ValueError, match="no finite optimum"):
        solve_generalized(x, loss=loss)


@pytest.mark.parametrize("loss,x", [("logistic", [[0.0], [1.0]]), ("poisson", [[0.0], [2.0]])])
def test_gamma_zero_and_disconnected_boundary_detection(loss, x):
    with pytest.raises(ValueError, match="no finite optimum"):
        solve_generalized(x, gamma=0, loss=loss)
    with pytest.raises(ValueError, match="no finite optimum"):
        solve_generalized(x, np.zeros((2, 2)), loss=loss)
    # The same boundary observations have a finite solution when coupled.
    result = solve_generalized(x, gamma=2, loss=loss, tol=1e-7)
    assert result.converged
    assert np.isfinite(result.centers).all()


@pytest.mark.parametrize(
    "loss,x,expected",
    [
        ("logistic", [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [2 / 3, 2 / 3]),
        ("poisson", [[0.0, 1.0], [2.0, 3.0], [4.0, 2.0]], [2.0, 2.0]),
    ],
)
def test_strong_fusion_recovers_feature_means(loss, x, expected):
    result = solve_generalized(x, gamma=10, loss=loss, tol=1e-8)
    assert result.converged
    assert_allclose(result.fitted_means, np.tile(expected, (3, 1)), atol=2e-7)


def test_huber_translation_and_nonunique_fused_solution():
    x = np.array([[-10.0], [10.0]])
    result = solve_generalized(x, gamma=10, huber_delta=1, tol=1e-8)
    shifted = solve_generalized(x + 100.0, gamma=10, huber_delta=1, tol=1e-8)
    assert result.converged and shifted.converged
    assert_allclose(result.objective, 19.0, atol=1e-6)
    assert_allclose(shifted.centers, result.centers + 100.0, atol=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gamma=-1),
        dict(gamma=np.nan),
        dict(huber_delta=0),
        dict(tol=0),
        dict(max_iter=True),
        dict(max_iter=0),
        dict(loss="unknown"),
        dict(penalty="bad"),
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        solve_generalized([[0.2], [0.7]], **kwargs)


@pytest.mark.parametrize(
    "loss,x",
    [
        ("huber", [[np.nan]]),
        ("huber", [[1j]]),
        ("huber", []),
        ("logistic", [[-1.0]]),
        ("logistic", [[2.0]]),
        ("poisson", [[-1.0]]),
    ],
)
def test_invalid_observations(loss, x):
    with pytest.raises(ValueError):
        solve_generalized(x, loss=loss)


def test_max_iteration_failure_is_honest():
    result = solve_generalized(
        [[0.0], [1.0], [0.0]], loss="logistic", gamma=0.1, max_iter=1, tol=1e-12
    )
    assert not result.converged
    assert result.n_iter == 1
    assert result.kkt_residual > 1e-12
