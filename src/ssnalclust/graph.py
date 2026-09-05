"""Sparse weighted graphs for convex clustering.

Rows of an incidence matrix encode ``X[i] - X[j]`` for ``i < j``.
Zero weights represent absent edges; disconnected graphs are supported.
"""

from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.spatial import KDTree


def graph_from_weights(weights, n_samples):
    """Construct an oriented incidence matrix and aligned edge weights.

    Parameters
    ----------
    weights : array-like or sparse matrix of shape (n_samples, n_samples), or None
        Symmetric, finite, nonnegative adjacency with zero diagonal. ``None``
        selects a complete graph with unit weights. Sparse input stays sparse.
    n_samples : int
        Positive number of vertices, including any isolated vertices.

    Returns
    -------
    incidence : scipy.sparse.csr_matrix of shape (n_edges, n_samples)
        Each row has +1 at vertex i and -1 at vertex j, where i < j.
        Edges are ordered lexicographically by (i, j).
    edge_weights : ndarray of shape (n_edges,)
        Strictly positive weights in the corresponding edge order.
    """
    if isinstance(n_samples, (bool, np.bool_)) or not isinstance(n_samples, Integral):
        raise ValueError("n_samples must be a positive integer")
    if n_samples < 1:
        raise ValueError("n_samples must be a positive integer")

    if weights is None:
        i, j = np.triu_indices(n_samples, k=1)
        values = np.ones(i.size, dtype=float)
    else:
        if sparse.issparse(weights):
            # COO supports every sparse format, including DOK and LIL, and
            # exposes duplicate entries so negatives cannot cancel unseen.
            adjacency = weights.tocoo(copy=True)
            if np.iscomplexobj(adjacency.data):
                raise ValueError("weights must be real")
            if not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any():
                raise ValueError("weights must be finite and nonnegative")
            adjacency = adjacency.astype(float).tocsr()
            adjacency.sum_duplicates()
        else:
            if np.iscomplexobj(weights):
                raise ValueError("weights must be real")
            try:
                array = np.asarray(weights, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("weights must be a numeric adjacency matrix") from exc
            if array.ndim != 2:
                raise ValueError("weights must have shape (n_samples, n_samples)")
            adjacency = sparse.csr_matrix(array)
        if adjacency.shape != (n_samples, n_samples):
            raise ValueError("weights must have shape (n_samples, n_samples)")
        if not np.isfinite(adjacency.data).all() or (adjacency.data < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        if np.any(adjacency.diagonal() != 0):
            raise ValueError("weights must have a zero diagonal")
        if (adjacency != adjacency.T).nnz:
            raise ValueError("weights must be symmetric")
        upper = sparse.triu(adjacency, k=1, format="coo")
        upper.eliminate_zeros()
        order = np.lexsort((upper.col, upper.row))
        i, j, values = upper.row[order], upper.col[order], upper.data[order]

    n_edges = len(values)
    indices = np.column_stack((i, j)).ravel()
    data = np.tile([1.0, -1.0], n_edges)
    indptr = np.arange(0, 2 * n_edges + 1, 2)
    incidence = sparse.csr_matrix((data, indices, indptr), shape=(n_edges, n_samples))
    return incidence, np.asarray(values, dtype=float)


def k_neighbors_graph(X, n_neighbors=10, bandwidth=1.0):
    """Build a symmetric Gaussian-weighted union of nearest-neighbor graphs.

    An edge exists when either endpoint selects the other among its nearest
    neighbors. Its weight is ``exp(-distance**2 / (2 * bandwidth**2))``.
    Self edges are excluded even when observations coincide. Distance ties
    may be resolved arbitrarily. Very small weights may underflow to zero.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite, real, dense observations.
    n_neighbors : int, default=10
        Number of neighbors per observation; must be in [1, n_samples - 1].
    bandwidth : float, default=1.0
        Positive, finite Gaussian bandwidth in the units of X.

    Returns
    -------
    weights : scipy.sparse.csr_matrix of shape (n_samples, n_samples)
        Symmetric adjacency with zero diagonal. Storage is O(n_samples *
        n_neighbors); no dense pairwise distance matrix is constructed.
    """
    if sparse.issparse(X) or np.iscomplexobj(X):
        raise ValueError("X must be a finite, real, dense two-dimensional array")
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a numeric two-dimensional array") from exc
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a finite, nonempty two-dimensional array")
    n_samples = X.shape[0]
    if (
        isinstance(n_neighbors, (bool, np.bool_))
        or not isinstance(n_neighbors, Integral)
        or not 1 <= n_neighbors < n_samples
    ):
        raise ValueError("n_neighbors must be an integer in [1, n_samples - 1]")
    if (
        isinstance(bandwidth, (bool, np.bool_))
        or not isinstance(bandwidth, Real)
        or not np.isfinite(bandwidth)
        or bandwidth <= 0
    ):
        raise ValueError("bandwidth must be positive and finite")

    distances, neighbors = KDTree(X).query(X, k=n_neighbors + 1)
    if not np.isfinite(distances).all():
        raise ValueError("Neighbor distances overflowed; rescale X")
    # A tied query need not contain its own index. Select the first k actual
    # nonself indices instead of assuming that the first result is self.
    nonself = neighbors != np.arange(n_samples)[:, None]
    selected = nonself & (np.cumsum(nonself, axis=1) <= n_neighbors)
    cols = neighbors[selected]
    rows = np.repeat(np.arange(n_samples), n_neighbors)
    with np.errstate(over="ignore", under="ignore"):
        values = np.exp(-0.5 * (distances[selected] / bandwidth) ** 2)
    directed = sparse.csr_matrix((values, (rows, cols)), shape=(n_samples, n_samples))
    adjacency = directed.maximum(directed.T).tocsr()
    adjacency.eliminate_zeros()
    adjacency.sort_indices()
    return adjacency
