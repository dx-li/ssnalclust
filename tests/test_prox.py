"""Independent proximal-map and generalized derivative checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust.graph import graph_from_weights
from ssnalclust.prox import project_dual, prox_norm
from ssnalclust.solvers import _newton_operator, _reduced


@pytest.mark.parametrize("penalty,dual_order", [("l1", np.inf), ("l2", 2), ("linf", 1)])
def test_projection_and_prox_against_independent_convex_oracle(penalty, dual_order):
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(491)
    values = np.vstack([np.zeros(4), rng.normal(size=(4, 4))])
    radii = np.array([0.0, 0.0, 0.2, 1.0, 20.0])
    actual = project_dual(values, radii, penalty)
    for index, (value, radius) in enumerate(zip(values, radii)):
        variable = cp.Variable(value.shape)
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(variable - value)),
            [cp.norm(variable, dual_order) <= radius],
        )
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
        assert problem.status == cp.OPTIMAL
        assert_allclose(actual[index], variable.value, atol=2e-6)
    assert_allclose(project_dual(actual, radii, penalty), actual, atol=1e-14)
    assert np.all(np.linalg.norm(actual, ord=dual_order, axis=1) <= radii + 1e-14)
    assert_allclose(prox_norm(values, radii, penalty) + actual, values)


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_zero_radius_zero_vectors_and_empty_edges(penalty):
    values = np.array([[0.0, 0.0], [3.0, -4.0]])
    assert_allclose(project_dual(values, np.zeros(2), penalty), 0)
    assert_allclose(prox_norm(values, np.zeros(2), penalty), values)
    assert prox_norm(np.empty((0, 2)), np.empty(0), penalty).shape == (0, 2)


@pytest.mark.parametrize("sigma", [0.2, 1.0, 5.0])
def test_reduced_gradient_and_hessian_finite_differences(sigma):
    rng = np.random.default_rng(18)
    b, _ = graph_from_weights(None, 5)
    x, u = rng.normal(size=(2, 5, 3))
    z = rng.normal(size=(10, 3))
    radii = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 20.0, 20.0, 20.0, 20.0, 20.0])
    mass = np.array([0.5, 1.0, 2.0, 3.0, 5.0])[:, None]
    direction = rng.normal(size=u.shape)
    step = 1e-6
    _, gradient = _reduced(x, u, z, b, radii, sigma, mass)
    plus, gradient_plus = _reduced(x, u + step * direction, z, b, radii, sigma, mass)
    minus, gradient_minus = _reduced(x, u - step * direction, z, b, radii, sigma, mass)
    assert_allclose((plus - minus) / (2 * step), np.sum(gradient * direction), rtol=1e-7, atol=1e-7)
    hessian, _ = _newton_operator(u, z, b, radii, sigma, mass)
    product = (hessian @ direction.ravel()).reshape(u.shape)
    assert_allclose(product, (gradient_plus - gradient_minus) / (2 * step), rtol=2e-7, atol=2e-7)
    other = rng.normal(size=u.size)
    assert_allclose(other @ (hessian @ direction.ravel()), direction.ravel() @ (hessian @ other))
    assert np.sum(direction * product) >= np.sum(mass * direction**2) - 1e-10


def test_historical_interior_hessian_and_sigma_scaling_regressions():
    b, _ = graph_from_weights(None, 2)
    x = np.zeros((2, 1))
    z = np.zeros((1, 1))
    mass = np.ones((2, 1))
    value, gradient = _reduced(x, np.array([[2.0], [0.0]]), z, b, np.ones(1), 2.0, mass)
    assert_allclose(value, 3.75)
    assert_allclose(gradient[:, 0], [3.0, -1.0])
    operator, _ = _newton_operator(x, z, b, np.ones(1), 1.0, mass)
    assert_allclose(operator @ np.array([1.0, -1.0]), [3.0, -3.0])


@pytest.mark.parametrize(
    "values,radii",
    [
        ([[1.0, 2.0]], [-1.0]),
        ([[np.nan, 0.0]], [1.0]),
        ([[1.0, 0.0]], [np.inf]),
        ([1.0, 2.0], [1.0]),
        ([[1.0, 2.0]], [1.0, 2.0]),
    ],
)
def test_invalid_projection_inputs(values, radii):
    with pytest.raises(ValueError):
        project_dual(values, radii)
