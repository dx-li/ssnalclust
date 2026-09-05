"""Solver correctness, checked independently of shared proximal helpers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from ssnalclust.solvers import solve

COMBINATIONS = [("ssnal", "l2")] + [
    (s, p) for s in ("admm", "ama", "fama") for p in ("l1", "l2", "linf")
]


def assert_certificate(result, tol):
    assert result.converged, result.message
    assert result.relative_gap <= tol
    assert result.kkt_residual <= tol
    assert result.dual_objective <= result.objective + 1e-9
    assert_allclose(result.gap, result.objective - result.dual_objective, atol=1e-9)
    assert len(result.history) == result.n_iter + 1


@pytest.mark.parametrize("solver,penalty", COMBINATIONS)
@pytest.mark.parametrize("gamma", [0.08, 0.8, 8.0])
def test_random_weighted_graph_against_cvxpy(solver, penalty, gamma):
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(97)
    x = rng.normal(size=(6, 3))
    weights = np.zeros((6, 6))
    edges = [(0, 1), (0, 3), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)]
    for i, j in edges:
        weights[i, j] = weights[j, i] = rng.uniform(0.2, 1.4)
    mass = rng.uniform(0.5, 2.0, size=6)
    u = cp.Variable(x.shape)
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    objective = 0.5 * cp.sum(cp.multiply(mass[:, None], cp.square(u - x)))
    objective += gamma * sum(weights[i, j] * cp.norm(u[i] - u[j], order) for i, j in edges)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
    assert problem.status == cp.OPTIMAL
    result = solve(
        x,
        sparse.csr_matrix(weights),
        gamma,
        penalty,
        solver,
        sample_weight=mass,
        sigma=0.3,
        tol=2e-7,
        max_iter=20000,
    )
    assert_certificate(result, 2e-7)
    assert_allclose(result.objective, problem.value, atol=2e-5, rtol=1e-6)
    assert_allclose(result.centers, u.value, atol=4e-5, rtol=2e-5)
    # An independently formed convex optimum lies between the certificates.
    assert result.dual_objective <= problem.value + 2e-7
    assert result.objective >= problem.value - 2e-7


@pytest.mark.parametrize("solver,penalty", COMBINATIONS)
@pytest.mark.parametrize("gamma", [0.1, 10.0])
def test_two_point_weighted_closed_form(solver, penalty, gamma):
    # Only one feature: every supported norm reduces to absolute value.
    x = np.array([[-2.0], [4.0]])
    mass = np.array([2.0, 5.0])
    radius = gamma * 1.7
    difference = max(6.0 - radius * (1 / mass[0] + 1 / mass[1]), 0.0)
    mean = np.average(x[:, 0], weights=mass)
    expected = np.array(
        [mean - mass[1] / mass.sum() * difference, mean + mass[0] / mass.sum() * difference]
    )
    result = solve(
        x,
        [[0.0, 1.7], [1.7, 0.0]],
        gamma,
        penalty,
        solver,
        sample_weight=mass,
        sigma=3.0,
        tol=1e-8,
        max_iter=20000,
    )
    assert_certificate(result, 1e-8)
    assert_allclose(result.centers[:, 0], expected, atol=3e-7)


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
@pytest.mark.parametrize("gamma", [0.2, 4.0])
def test_two_point_euclidean_vector_closed_form(solver, gamma):
    x = np.array([[1.0, -2.0, 3.0], [4.0, 2.0, 3.0]])
    mass = np.array([2.0, 3.0])
    delta = x[0] - x[1]
    expected_delta = (
        max(0.0, 1.0 - gamma * (1 / mass[0] + 1 / mass[1]) / np.linalg.norm(delta)) * delta
    )
    mean = np.average(x, weights=mass, axis=0)
    expected = np.vstack(
        [mean + mass[1] / mass.sum() * expected_delta, mean - mass[0] / mass.sum() * expected_delta]
    )
    result = solve(x, gamma=gamma, solver=solver, sample_weight=mass, tol=1e-8, max_iter=20000)
    assert_certificate(result, 1e-8)
    assert_allclose(result.centers, expected, atol=3e-7)


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_euclidean_orthogonal_equivariance(solver):
    rng = np.random.default_rng(812)
    x = rng.normal(size=(5, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    result = solve(x, gamma=0.2, solver=solver, tol=1e-8, max_iter=20000)
    rotated = solve(x @ rotation, gamma=0.2, solver=solver, tol=1e-8, max_iter=20000)
    assert_certificate(result, 1e-8)
    assert_certificate(rotated, 1e-8)
    assert_allclose(rotated.centers, result.centers @ rotation, atol=3e-7)


@pytest.mark.parametrize("solver,penalty", COMBINATIONS)
def test_warm_start_reprojects_and_does_not_mutate_inputs(solver, penalty):
    x = np.array([[0.0, -2.0], [3.0, 1.0], [5.0, 2.0]])
    initial = x + 0.5
    dual = np.full((3, 2), 100.0)
    saved_x, saved_initial, saved_dual = x.copy(), initial.copy(), dual.copy()
    warm = solve(
        x,
        gamma=0.2,
        penalty=penalty,
        solver=solver,
        x0=initial,
        dual0=dual,
        tol=1e-8,
        max_iter=20000,
    )
    cold = solve(x, gamma=0.2, penalty=penalty, solver=solver, tol=1e-8, max_iter=20000)
    assert_certificate(warm, 1e-8)
    assert_certificate(cold, 1e-8)
    assert_allclose(warm.centers, cold.centers, atol=3e-7)
    assert_allclose(x, saved_x)
    assert_allclose(initial, saved_initial)
    assert_allclose(dual, saved_dual)


@pytest.mark.parametrize("solver,penalty", COMBINATIONS)
def test_disconnected_translation_and_component_means(solver, penalty):
    x = np.array([[0.0, 1.0], [2.0, 3.0], [8.0, 2.0], [9.0, 4.0], [20.0, -3.0]])
    w = np.zeros((5, 5))
    w[0, 1] = w[1, 0] = w[2, 3] = w[3, 2] = 1.0
    result = solve(x, w, 5.0, penalty, solver, tol=1e-8, max_iter=20000)
    shifted = solve(x + [4.0, -7.0], w, 5.0, penalty, solver, tol=1e-8, max_iter=20000)
    assert_certificate(result, 1e-8)
    assert_certificate(shifted, 1e-8)
    expected = x.copy()
    expected[:2] = x[:2].mean(axis=0)
    expected[2:4] = x[2:4].mean(axis=0)
    assert_allclose(result.centers, expected, atol=3e-7)
    assert_allclose(shifted.centers, result.centers + [4.0, -7.0], atol=3e-7)


@pytest.mark.parametrize("solver,penalty", COMBINATIONS)
def test_zero_gamma_duplicates_and_edgeless(solver, penalty):
    x = np.array([[2.0, 1.0], [2.0, 1.0], [-1.0, 3.0]])
    for weights, gamma in [(None, 0.0), (np.zeros((3, 3)), 4.0)]:
        result = solve(x, weights, gamma, penalty, solver)
        assert_certificate(result, 1e-6)
        assert_allclose(result.centers, x)
        assert result.n_iter == 0
    duplicate = solve(x[:2], gamma=3.0, penalty=penalty, solver=solver)
    assert_certificate(duplicate, 1e-6)
    assert_allclose(duplicate.centers, x[:2])


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_iteration_exhaustion_is_not_success(solver):
    x = np.random.default_rng(131).normal(size=(12, 4))
    result = solve(x, gamma=0.25, solver=solver, max_iter=1, tol=1e-12, inner_max_iter=1)
    assert not result.converged
    assert result.n_iter == 1
    assert "iterations" in result.message.lower()
    assert max(result.kkt_residual, result.relative_gap) > 1e-12


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gamma=-1),
        dict(gamma=np.nan),
        dict(tol=0),
        dict(sigma=0),
        dict(max_iter=0),
        dict(max_iter=True),
        dict(inner_max_iter=1.5),
        dict(penalty="bad"),
        dict(solver="bad"),
        dict(penalty="l1", solver="ssnal"),
        dict(sample_weight=[1.0, 0.0]),
        dict(sample_weight=[1.0]),
        dict(x0=[[1.0, 2.0]]),
        dict(dual0=np.zeros((4, 1))),
    ],
)
def test_invalid_solver_parameters(kwargs):
    with pytest.raises(ValueError):
        solve([[0.0], [2.0]], **kwargs)


@pytest.mark.parametrize("x", [[], [1.0, 2.0], [[np.inf]], [[1j]], np.empty((2, 0))])
def test_invalid_data(x):
    with pytest.raises(ValueError):
        solve(x)
