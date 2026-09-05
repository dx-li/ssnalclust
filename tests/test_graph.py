"""Graph construction tests with independent incidence and distance oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from ssnalclust.graph import graph_from_weights, k_neighbors_graph


def test_complete_graph_incidence_and_weights():
    B, w = graph_from_weights(None, 4)
    assert sparse.isspmatrix_csr(B)
    assert B.shape == (6, 4)
    assert B.nnz == 12
    assert_array_equal(w, np.ones(6))
    X = np.arange(12).reshape(4, 3)
    assert_array_equal(B @ X, [X[i] - X[j] for i in range(4) for j in range(i + 1, 4)])


@pytest.mark.parametrize(
    "convert",
    [
        np.asarray,
        sparse.csr_matrix,
        sparse.coo_matrix,
        sparse.lil_matrix,
        sparse.dok_matrix,
        sparse.csr_array,
    ],
)
def test_disconnected_graph_and_order(convert):
    W = np.zeros((5, 5))
    W[0, 2] = W[2, 0] = 2.5
    W[1, 3] = W[3, 1] = 0.25
    B, w = graph_from_weights(convert(W), 5)
    assert_array_equal(B.toarray(), [[1, 0, -1, 0, 0], [0, 1, 0, -1, 0]])
    assert_array_equal(w, [2.5, 0.25])


@pytest.mark.parametrize("n", [1, 4])
def test_edgeless_graph(n):
    B, w = graph_from_weights(sparse.csr_matrix((n, n)), n)
    assert B.shape == (0, n)
    assert w.shape == (0,)
    assert_array_equal(B.T @ np.empty((0, 2)), np.zeros((n, 2)))
    if n == 1:
        assert graph_from_weights(None, n)[0].shape == (0, 1)


@pytest.mark.parametrize(
    "weights",
    [
        [[0, -1], [-1, 0]],
        [[0, np.nan], [np.nan, 0]],
        [[0, np.inf], [np.inf, 0]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 1j], [1j, 0]],
        np.ones((3, 3)),
    ],
)
@pytest.mark.parametrize("convert", [np.asarray, sparse.csr_matrix])
def test_invalid_weights(weights, convert):
    with pytest.raises(ValueError):
        graph_from_weights(convert(weights), 2)


@pytest.mark.parametrize("n", [0, -1, 2.5, True, "2"])
def test_invalid_sample_count(n):
    with pytest.raises(ValueError, match="positive integer"):
        graph_from_weights(None, n)


def test_sparse_duplicate_entries_and_explicit_zeros():
    W = sparse.coo_matrix(([1.0, 2.0, 3.0, 0.0], ([0, 0, 1, 0], [1, 1, 0, 0])), shape=(2, 2))
    B, w = graph_from_weights(W, 2)
    assert_array_equal(w, [3])
    assert B.shape == (1, 2)
    assert W.nnz == 4  # input is not normalized in place


def test_sparse_negative_duplicates_cannot_cancel():
    W = sparse.coo_matrix(([2.0, -1.0, 1.0], ([0, 0, 1], [1, 1, 0])), shape=(2, 2))
    with pytest.raises(ValueError, match="nonnegative"):
        graph_from_weights(W, 2)


def test_large_sparse_graph_keeps_only_edges():
    n = 100_000
    W = sparse.coo_matrix(([2.0, 2.0], ([0, n - 1], [n - 1, 0])), shape=(n, n))
    B, w = graph_from_weights(W, n)
    assert B.shape == (1, n)
    assert B.nnz == 2
    assert_array_equal(w, [2])


def test_knn_matches_independent_pairwise_union():
    X = np.random.default_rng(41).normal(size=(20, 3))
    k, bandwidth = 3, 0.8
    distances2 = np.sum((X[:, None] - X[None, :]) ** 2, axis=2)
    np.fill_diagonal(distances2, np.inf)
    order = np.argsort(distances2, axis=1)[:, :k]
    expected = np.zeros((20, 20))
    rows = np.arange(20)[:, None]
    expected[rows, order] = np.exp(-distances2[rows, order] / (2 * bandwidth**2))
    expected = np.maximum(expected, expected.T)
    W = k_neighbors_graph(X, k, bandwidth)
    assert sparse.isspmatrix_csr(W)
    assert_allclose(W.toarray(), expected, rtol=1e-14, atol=0)
    graph_from_weights(W, len(X))  # valid solver input


@pytest.mark.parametrize("k", [1, 3, 9])
def test_duplicate_observations_exclude_only_self(k):
    W = k_neighbors_graph(np.zeros((10, 2)), k)
    assert_array_equal(W.diagonal(), np.zeros(10))
    assert_array_equal(W.data, np.ones(W.nnz))
    assert (W != W.T).nnz == 0
    assert np.all(np.diff(W.indptr) >= k)
    assert W.nnz <= 2 * 10 * k


def test_full_neighbor_graph_gaussian_weights():
    W = k_neighbors_graph([[0], [2], [5]], 2, bandwidth=2)
    expected = np.exp(-np.array([[0, 4, 25], [4, 0, 9], [25, 9, 0]]) / 8)
    np.fill_diagonal(expected, 0)
    assert_allclose(W.toarray(), expected)


@pytest.mark.parametrize("k", [0, -1, 3, 1.5, True])
def test_invalid_neighbors(k):
    with pytest.raises(ValueError, match="n_neighbors"):
        k_neighbors_graph([[0], [1], [2]], k)


@pytest.mark.parametrize("bandwidth", [0, -1, np.inf, np.nan, True, "1"])
def test_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        k_neighbors_graph([[0], [1]], 1, bandwidth)


@pytest.mark.parametrize(
    "X",
    [[], [1, 2], [[np.nan], [0]], [[np.inf], [0]], [[1j], [0]], np.empty((2, 0)), sparse.eye(2)],
)
def test_invalid_observations(X):
    with pytest.raises(ValueError, match="X"):
        k_neighbors_graph(X, 1)


def test_tiny_bandwidth_underflow_remains_valid():
    W = k_neighbors_graph([[0], [1]], 1, bandwidth=1e-300)
    assert W.nnz == 0
    graph_from_weights(W, 2)
