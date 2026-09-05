"""Structural and numerical correctness of additional graph builders."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import cdist

from ssnalclust.graph import k_neighbors_graph
from ssnalclust.graph_extra import (
    connected_k_neighbors_graph,
    minimum_spanning_tree_graph,
    self_tuning_graph,
)


def test_mst_known_length():
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    weights = minimum_spanning_tree_graph(X)
    edges = sparse.triu(weights).tocoo()
    assert len(edges.data) == 3
    assert_allclose(np.linalg.norm(X[edges.row] - X[edges.col], axis=1).sum(), 3)
    assert_allclose(edges.data, np.exp(-0.5))
    assert connected_components(weights, directed=False)[0] == 1


def test_mst_matches_independent_scipy_tree():
    X = np.random.default_rng(17).normal(size=(20, 3))
    distances = cdist(X, X)
    expected = minimum_spanning_tree(distances).sum()
    result = sparse.triu(minimum_spanning_tree_graph(X, bandwidth=2)).tocoo()
    assert len(result.data) == len(X) - 1
    assert_allclose(distances[result.row, result.col].sum(), expected, rtol=1e-13)


def test_mst_preserves_zero_distance_edges():
    X = [[0], [0], [1], [1]]
    weights = minimum_spanning_tree_graph(X)
    assert weights.nnz == 6
    assert weights[0, 1] == 1
    assert weights[2, 3] == 1
    assert connected_components(weights, directed=False)[0] == 1


def test_connected_graph_bridges_knn_components():
    X = [[0], [0.1], [4], [4.1]]
    base = k_neighbors_graph(X, n_neighbors=1, bandwidth=2)
    assert connected_components(base, directed=False)[0] == 2
    graph = connected_k_neighbors_graph(X, n_neighbors=1, bandwidth=2)
    assert connected_components(graph, directed=False)[0] == 1
    assert_allclose(graph.maximum(base).toarray(), graph.toarray())
    assert graph.nnz == 6


@pytest.mark.parametrize("builder", [minimum_spanning_tree_graph, connected_k_neighbors_graph])
def test_required_edge_underflow_rejected(builder):
    with pytest.raises(ValueError, match="underflowed"):
        builder([[0], [1000]], bandwidth=0.01)


@pytest.mark.parametrize(
    "builder", [minimum_spanning_tree_graph, connected_k_neighbors_graph, self_tuning_graph]
)
def test_singleton_and_identical_data(builder):
    singleton = builder([[4, 5]])
    assert singleton.shape == (1, 1)
    assert singleton.nnz == 0
    graph = builder(np.ones((8, 2)))
    assert graph.nnz > 0
    assert np.all(graph.data == 1)
    assert np.all(graph.diagonal() == 0)
    assert (graph != graph.T).nnz == 0
    assert connected_components(graph, directed=False)[0] == 1


def test_self_tuning_known_scales_and_duplicate_fallback():
    graph = self_tuning_graph([[0], [1], [3]], n_neighbors=2, scale_neighbors=1)
    # Nearest distances are 1, 1 and 2.
    expected = [
        [0, np.exp(-1), np.exp(-4.5)],
        [np.exp(-1), 0, np.exp(-2)],
        [np.exp(-4.5), np.exp(-2), 0],
    ]
    assert_allclose(graph.toarray(), expected)
    duplicates = self_tuning_graph([[0], [0], [2]], n_neighbors=2, scale_neighbors=1)
    # All scales become 2, including zero-neighbor-distance fallbacks.
    assert_allclose(
        duplicates.toarray(), [[0, 1, np.exp(-1)], [1, 0, np.exp(-1)], [np.exp(-1), np.exp(-1), 0]]
    )


def test_local_scaling_invariant_to_global_scaling():
    X = np.random.default_rng(42).normal(size=(25, 4))
    X[1] = X[0]
    reference = self_tuning_graph(X, n_neighbors=5, scale_neighbors=3)
    for scale in [0.01, 3, 1e4]:
        transformed = self_tuning_graph(scale * X, n_neighbors=5, scale_neighbors=3)
        assert_allclose(reference.toarray(), transformed.toarray(), atol=1e-14)


@pytest.mark.parametrize("builder", [connected_k_neighbors_graph, self_tuning_graph])
@pytest.mark.parametrize("neighbors", [0, -1, 1.5, True])
def test_invalid_neighbor_count(builder, neighbors):
    with pytest.raises(ValueError, match="n_neighbors"):
        builder([[0], [1]], n_neighbors=neighbors)


@pytest.mark.parametrize(
    "builder", [minimum_spanning_tree_graph, connected_k_neighbors_graph, self_tuning_graph]
)
@pytest.mark.parametrize("X", [[], [1, 2], [[np.nan]], [[np.inf]], [[1j]]])
def test_invalid_data(builder, X):
    with pytest.raises(ValueError):
        builder(X)


@pytest.mark.parametrize("bandwidth", [0, -1, np.nan, True])
def test_invalid_bandwidth(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        minimum_spanning_tree_graph([[0], [1]], bandwidth=bandwidth)


def test_invalid_scale_neighbors():
    with pytest.raises(ValueError, match="scale_neighbors"):
        self_tuning_graph([[0], [1]], scale_neighbors=0)
