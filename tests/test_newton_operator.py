"""Independent dense and differential checks of the SSNAL Newton operator."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from ssnalclust.solvers import _newton_operator, _reduced


def explicit_hessian(u, z, b, radii, sigma, mass):
    """Assemble edge blocks directly, without the matrix-free implementation."""
    n, p = u.shape
    hessian = np.diag(np.broadcast_to(mass, (n, p)).ravel())
    for edge, row in enumerate(b.toarray()):
        q = sigma * (row @ u) + z[edge]
        norm = math.hypot(*q)
        if radii[edge] == 0:
            block = np.zeros((p, p))
        elif norm <= radii[edge]:
            # Equality selects the derivative approached from inside the ball.
            block = np.eye(p)
        else:
            direction = q / norm
            block = radii[edge] / norm * (np.eye(p) - np.outer(direction, direction))
        hessian += sigma * np.kron(np.outer(row, row), block)
    return hessian


@pytest.mark.parametrize("sigma", [0.07, 1.0, 19.0])
@pytest.mark.parametrize("mass_kind", ["scalar", "sample", "entry"])
def test_dense_action_and_jacobi_with_all_edge_regimes(sigma, mass_kind):
    # Nonunit incidence entries also check that Jacobi uses squared entries.
    b = sparse.csr_matrix(
        [[1, -1, 0, 0], [0, 1, -1, 0], [1, 0, -1, 0], [0, 0, 2, -2], [1, 0, 0, -1]]
    )
    u = np.zeros((4, 2))
    z = np.array([[0.2, 0.1], [6.0, 8.0], [3.0, 4.0], [0.0, 0.0], [1.0, -2.0]])
    radii = np.array([1.0, 2.0, 5.0, 0.0, 0.0])
    mass = {
        "scalar": 1.7,
        "sample": np.array([[0.3], [1.0], [2.1], [4.0]]),
        "entry": np.arange(1, 9).reshape(4, 2) / 3,
    }[mass_kind]
    expected = explicit_hessian(u, z, b, radii, sigma, mass)
    operator, preconditioner = _newton_operator(u, z, b, radii, sigma, mass)
    actual = np.column_stack([operator @ basis for basis in np.eye(u.size)])
    assert_allclose(actual, expected, rtol=3e-14, atol=3e-14)
    assert_allclose(actual, actual.T, atol=3e-14)
    assert np.linalg.eigvalsh(actual).min() >= np.min(mass) - 1e-12
    vector = np.arange(1, u.size + 1, dtype=float)
    assert_allclose(preconditioner @ vector, vector / np.diag(expected), rtol=3e-14)
    assert_allclose(operator @ vector[:, None], expected @ vector[:, None], atol=1e-12)


@pytest.mark.parametrize("sigma", [0.13, 1.0, 8.0])
@pytest.mark.parametrize("p", [1, 3, 7])
def test_action_matches_reduced_gradient_finite_difference(sigma, p):
    rng = np.random.default_rng(811 + p)
    b = sparse.csr_matrix([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]])
    x = rng.normal(size=(4, p))
    u = rng.normal(size=x.shape)
    q = rng.normal(size=(3, p))
    q /= np.linalg.norm(q, axis=1)[:, None]
    q *= np.array([0.25, 3.0, 2.0])[:, None]
    radii = np.array([1.0, 0.8, 0.0])
    z = q - sigma * (b @ u)
    mass = rng.uniform(0.3, 2.0, size=(4, 1))
    direction = rng.normal(size=u.shape)
    operator, _ = _newton_operator(u, z, b, radii, sigma, mass)
    step = 2e-6 / max(1.0, sigma)
    plus = _reduced(x, u + step * direction, z, b, radii, sigma, mass)[1]
    minus = _reduced(x, u - step * direction, z, b, radii, sigma, mass)[1]
    numerical = (plus - minus) / (2 * step)
    assert_allclose(
        (operator @ direction.ravel()).reshape(u.shape), numerical, rtol=2e-7, atol=2e-8
    )


@pytest.mark.parametrize("p", [1, 4])
def test_no_edges_reduces_to_mass_operator(p):
    u = np.zeros((3, p))
    mass = np.array([[0.2], [2.0], [7.0]])
    operator, preconditioner = _newton_operator(
        u, np.empty((0, p)), sparse.csr_matrix((0, 3)), np.empty(0), 11.0, mass
    )
    direction = np.arange(1, u.size + 1).reshape(u.shape)
    assert_allclose(operator @ direction.ravel(), (mass * direction).ravel())
    assert_allclose(preconditioner @ direction.ravel(), (direction / mass).ravel())


@pytest.mark.parametrize("radius", [0.0, 2.0])
def test_origin_and_positive_boundary_jacobian_selection(radius):
    u = np.zeros((2, 2))
    b = sparse.csr_matrix([[1.0, -1.0]])
    z = np.array([[radius, 0.0]])
    operator, _ = _newton_operator(u, z, b, np.array([radius]), 3.0, 1.0)
    direction = np.array([1.0, 0.0, -1.0, 0.0])
    # At positive radius equality, identity is an allowed generalized Jacobian.
    # A zero-radius ball is constant even at its origin, so its derivative is zero.
    assert_allclose(operator @ direction, direction * (7.0 if radius else 1.0))


@pytest.mark.parametrize("multiplier", [1e-200, 1e200])
def test_projection_jacobian_invariant_to_joint_radius_input_scaling(multiplier):
    u = np.zeros((3, 3))
    b = sparse.csr_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    z = np.array([[3.0, -4.0, 2.0], [0.1, 0.2, 0.1]])
    radii = np.array([0.7, 1.0])
    mass = np.array([[0.3], [2.0], [1.1]])
    expected = explicit_hessian(u, z, b, radii, 2.3, mass)
    operator, preconditioner = _newton_operator(u, multiplier * z, b, multiplier * radii, 2.3, mass)
    for vector in np.eye(u.size):
        assert_allclose(operator @ vector, expected @ vector, atol=3e-15, rtol=3e-14)
    vector = np.arange(1, u.size + 1, dtype=float)
    assert_allclose(preconditioner @ vector, vector / np.diag(expected), rtol=3e-14)


@pytest.mark.parametrize("magnitude", [np.finfo(float).max, np.nextafter(0.0, 1.0)])
def test_outside_ball_at_extreme_finite_magnitudes(magnitude):
    # The true norm of (maxfloat,maxfloat) exceeds representable float range;
    # the norm of (minsubnormal,minsubnormal) must not be rounded before comparing.
    # Both have radius/norm = 1/sqrt(2) and the same exact projection derivative.
    u = np.zeros((2, 2))
    b = sparse.csr_matrix([[1.0, -1.0]])
    z = np.array([[magnitude, magnitude]])
    operator, preconditioner = _newton_operator(u, z, b, np.array([magnitude]), 1.0, 1.0)
    block = np.array([[0.5, -0.5], [-0.5, 0.5]]) / np.sqrt(2.0)
    expected = np.eye(4) + np.kron(np.array([[1.0, -1.0], [-1.0, 1.0]]), block)
    actual = np.column_stack([operator @ vector for vector in np.eye(4)])
    assert_allclose(actual, expected, atol=1e-15)
    assert_allclose(preconditioner @ np.ones(4), 1 / np.diag(expected), atol=1e-15)


def test_huge_radius_tiny_input_remains_inside_without_overflow_warning():
    u = np.zeros((2, 2))
    b = sparse.csr_matrix([[1.0, -1.0]])
    z = np.full((1, 2), np.nextafter(0.0, 1.0))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        operator, _ = _newton_operator(u, z, b, np.array([np.finfo(float).max]), 1.0, 1.0)
        actual = operator @ np.array([1.0, 0.0, -1.0, 0.0])
    assert_allclose(actual, [3.0, 0.0, -3.0, 0.0])
