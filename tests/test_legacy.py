"""Compatibility tests for the original column-oriented fitting API."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from ssnalclust.graph import graph_from_weights
from ssnalclust.legacy import SSNAL
from ssnalclust.solvers import solve


def make_legacy(A, weights, gamma):
    with pytest.warns(DeprecationWarning, match="samples in rows"):
        return SSNAL(A, weights, gamma)


def test_original_fit_tuple_and_column_layout():
    A = np.array([[0.0, 3.0, 8.0], [1.0, -1.0, 2.0]])
    weights = sparse.csr_matrix([[0, 1, 0], [1, 0, 2], [0, 2, 0]])
    legacy = make_legacy(A, weights, 0.5)
    X, U, Z, status = legacy.fit(300, eps=1e-8)
    reference = solve(A.T, weights, gamma=0.5, max_iter=300, tol=1e-8)
    B, _ = graph_from_weights(weights, A.shape[1])
    assert status == 0
    assert X.shape == (2, 3)
    assert U.shape == Z.shape == (2, 2)
    assert_allclose(X.T, reference.centers)
    assert_allclose(Z.T, reference.dual)
    assert_array_equal(U.T, B @ X.T)
    assert legacy.result_.relative_gap <= 1e-8
    assert legacy.result_.kkt_residual <= 1e-8


def test_mutable_gamma_and_warm_start():
    A = np.array([[0.0, 4.0]])
    legacy = make_legacy(A, [[0, 1], [1, 0]], 0)
    X, U, Z, status = legacy.fit(100)
    assert status == 0
    assert_array_equal(X, A)
    legacy.gamma = 0.5
    assert_array_equal(legacy.weights, [0.5])
    assert_array_equal(legacy.pre_weights, [1])
    X, U, Z, status = legacy.fit(100, X0=X, U0=U, Z0=Z, eps=1e-9)
    assert status == 0
    assert_allclose(X, [[0.5, 3.5]], atol=1e-8)
    assert_allclose(U, [[-3]], atol=1e-8)
    assert_allclose(Z, [[-0.5]], atol=1e-8)


def test_auxiliary_warm_start_is_eliminated():
    legacy = make_legacy([[0.0, 4.0]], None, 0.5)
    first = legacy.fit(100, U0=np.zeros((1, 1)))
    second = legacy.fit(100, U0=np.full((1, 1), 1000))
    for left, right in zip(first, second):
        assert_array_equal(left, right)


def test_iteration_exhaustion_reports_failure():
    legacy = make_legacy([[0.0, 4.0, 10.0]], None, 2)
    *_, status = legacy.fit(1, eps=1e-12)
    assert status == 1
    assert not legacy.result_.converged


def test_edgeless_graph_return_shapes():
    legacy = make_legacy([[1.0, 2.0], [3.0, 4.0]], np.zeros((2, 2)), 10)
    X, U, Z, status = legacy.fit(10, U0=np.empty((2, 0)))
    assert status == 0
    assert_array_equal(X, legacy.A)
    assert U.shape == Z.shape == (2, 0)


@pytest.mark.parametrize("name", ["X0", "U0", "Z0"])
@pytest.mark.parametrize("value", [np.zeros((3, 4)), [[np.nan]], [[1j]]])
def test_invalid_warm_starts(name, value):
    legacy = make_legacy([[0.0, 4.0]], None, 1)
    with pytest.raises(ValueError, match=name):
        legacy.fit(10, **{name: value})


@pytest.mark.parametrize("gamma", [-1, np.inf, np.nan, True, 1j])
def test_invalid_gamma_setter(gamma):
    legacy = make_legacy([[0.0, 4.0]], None, 1)
    with pytest.raises(ValueError, match="gamma"):
        legacy.gamma = gamma
    assert legacy.gamma == 1


@pytest.mark.parametrize("A", [[], [1, 2], [[np.nan]], [[1j]], np.empty((0, 2))])
def test_invalid_data(A):
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="A"):
        SSNAL(A, None, 1)
