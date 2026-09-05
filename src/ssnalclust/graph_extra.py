"""Connected and locally scaled similarity graphs for convex clustering."""

from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from .graph import k_neighbors_graph


def _data(X):
    if sparse.issparse(X):
        raise ValueError("X must be a finite real dense array")
    try:
        values = np.asarray(X)
        if np.iscomplexobj(values):
            raise ValueError("X must be real")
        values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a finite real dense array") from exc
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("X must have finite nonempty shape (n_samples, n_features)")
    return values


def _bandwidth(bandwidth):
    if (
        isinstance(bandwidth, (bool, np.bool_))
        or not isinstance(bandwidth, Real)
        or not np.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise ValueError("bandwidth must be finite and positive")


def _neighbors(value, name, n_samples):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return min(value, n_samples - 1)


def minimum_spanning_tree_graph(X, bandwidth=1.0):
    """Gaussian-weighted exact Euclidean minimum spanning tree.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite real observations.
    bandwidth : float, default=1.0
        Edge weights are exp(-distance**2 / (2 * bandwidth**2)).

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_samples, n_samples)
        Symmetric adjacency with exactly n_samples - 1 undirected edges.
        A singleton returns an empty (1, 1) adjacency.

    Notes
    -----
    Uses a dense Euclidean distance matrix and Prim's algorithm: O(n_samples²)
    memory, O(n_samples² * n_features) distance work and O(n_samples²) tree
    work. It is intended for moderate data sizes. Zero-length edges between
    duplicate observations are preserved with similarity weight one. Ties are
    broken deterministically by input order. If a required Gaussian weight
    underflows to zero, raises ValueError; increase bandwidth or rescale X.
    """
    X = _data(X)
    _bandwidth(bandwidth)
    n = len(X)
    if n == 1:
        return sparse.csr_matrix((1, 1))
    distances = cdist(X, X, metric="euclidean")
    if not np.isfinite(distances).all():
        raise ValueError("Euclidean distances overflowed; rescale X")
    # Do not encode Euclidean zero as an absent sparse edge: duplicates must
    # connect at zero cost when finding the tree, before Gaussian weighting.
    active = np.zeros(n, dtype=bool)
    active[0] = True
    nearest = distances[0].copy()
    nearest[0] = np.inf
    parent = np.zeros(n, dtype=np.intp)
    rows, cols, lengths = [], [], []
    for _ in range(n - 1):
        vertex = int(np.argmin(nearest))
        rows.append(int(parent[vertex]))
        cols.append(vertex)
        lengths.append(nearest[vertex])
        active[vertex] = True
        improve = (~active) & (distances[vertex] < nearest)
        nearest[improve] = distances[vertex, improve]
        parent[improve] = vertex
        nearest[active] = np.inf
    with np.errstate(over="ignore", under="ignore"):
        values = np.exp(-0.5 * (np.asarray(lengths) / bandwidth) ** 2)
    if np.any(values == 0):
        raise ValueError(
            "MST Gaussian weights underflowed to zero; increase bandwidth or rescale X"
        )
    graph = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    return (graph + graph.T).tocsr()


def connected_k_neighbors_graph(X, n_neighbors=10, bandwidth=1.0):
    """Union a Gaussian nearest-neighbor graph with its Euclidean MST.

    The returned symmetric CSR adjacency is connected, including for duplicate
    observations. n_neighbors must be positive and is clipped to n_samples-1;
    a singleton returns an empty adjacency. The MST adds at most n_samples-1
    edges, each using the same Gaussian bandwidth as the nearest-neighbor
    graph. This construction uses O(n_samples²) memory for the exact MST.
    Required MST edge weights that underflow to zero raise ValueError instead
    of returning a disconnected graph.
    """
    X = _data(X)
    k = _neighbors(n_neighbors, "n_neighbors", len(X))
    tree = minimum_spanning_tree_graph(X, bandwidth=bandwidth)
    if len(X) == 1:
        return tree
    neighbors = k_neighbors_graph(X, n_neighbors=k, bandwidth=bandwidth)
    return neighbors.maximum(tree).tocsr()


def self_tuning_graph(X, n_neighbors=10, scale_neighbors=None):
    """Build a nearest-neighbor graph using local Gaussian distance scales.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite real observations.
    n_neighbors : int, default=10
        Number of neighbors defining edges, clipped to n_samples - 1.
    scale_neighbors : int, optional
        Neighbor rank defining each local scale s_i. Defaults to n_neighbors;
        also clipped to n_samples - 1.

    Returns
    -------
    scipy.sparse.csr_matrix of shape (n_samples, n_samples)
        Symmetric union graph with weights exp(-d_ij² / (s_i * s_j)). A
        singleton has no edges. Connectivity is not guaranteed.

    Notes
    -----
    The scale s_i is the distance to the requested nonself neighbor. If that
    distance is zero because of duplicates, use the nearest strictly distinct
    observation. If every observation is identical, use scale one, giving all
    retained edges weight one. These conventions preserve invariance to a
    common positive rescaling of all features. No dense pairwise matrix is
    built; query storage is O(n_samples * max(n_neighbors, scale_neighbors)).
    Very small similarities can underflow to zero and omit edges.
    """
    X = _data(X)
    n = len(X)
    k = _neighbors(n_neighbors, "n_neighbors", n)
    scale_k = _neighbors(
        n_neighbors if scale_neighbors is None else scale_neighbors, "scale_neighbors", n
    )
    if n == 1:
        return sparse.csr_matrix((1, 1))
    query_k = max(k, scale_k)
    distances, neighbors = KDTree(X).query(X, k=query_k + 1)
    if not np.isfinite(distances).all():
        raise ValueError("Neighbor distances overflowed; rescale X")
    nonself = neighbors != np.arange(n)[:, None]
    selected = nonself & (np.cumsum(nonself, axis=1) <= query_k)
    distances = distances[selected].reshape(n, query_k)
    neighbors = neighbors[selected].reshape(n, query_k)
    scales = distances[:, scale_k - 1].copy()
    zero = scales == 0
    if np.any(zero):
        distinct = np.unique(X, axis=0)
        if len(distinct) == 1:
            scales[zero] = 1.0
        else:
            fallback, _ = KDTree(distinct).query(X[zero], k=2)
            scales[zero] = fallback[:, 1]
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("Local distances overflowed or underflowed; rescale X")
    rows = np.repeat(np.arange(n), k)
    cols = neighbors[:, :k].ravel()
    lengths = distances[:, :k].ravel()
    with np.errstate(over="ignore", under="ignore"):
        values = np.exp(-(lengths / scales[rows]) * (lengths / scales[cols]))
    directed = sparse.csr_matrix((values, (rows, cols)), shape=(n, n))
    graph = directed.maximum(directed.T).tocsr()
    graph.eliminate_zeros()
    return graph
