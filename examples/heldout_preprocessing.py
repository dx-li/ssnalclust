"""Training-only preprocessing for entry-holdout experiments (not public API).

Graph construction is a modeling choice: missing standardized entries are set
to zero only for graph distances. They remain NaN in the masked-loss input.
This modest-data example computes a dense pairwise distance matrix.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


@dataclass
class TrainingData:
    training_values: np.ndarray
    observed: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    graph: sparse.csr_matrix
    bandwidth: float
    graph_inputs: np.ndarray
    components: int


def prepare_training(X, observed, neighbors=10):
    """Erase hidden values, standardize training entries, and construct a graph.

    Every feature needs training observations in every positive-weight graph
    component. Population standard deviations use training entries only;
    constant training features have scale one. No hidden targets are retained.
    """
    raw = np.asarray(X)
    mask = np.asarray(observed)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 1:
        raise ValueError("X must have at least two samples and one feature")
    if raw.dtype.kind not in "iuf":
        raise ValueError("X must contain real numeric values")
    if mask.dtype.kind != "b" or mask.shape != raw.shape:
        raise ValueError("observed must be a boolean array with X's shape")
    if isinstance(neighbors, (bool, np.bool_)) or not isinstance(neighbors, Integral):
        raise ValueError("neighbors must be an integer")
    if not 1 <= neighbors < len(raw):
        raise ValueError("neighbors must be between 1 and n_samples - 1")
    mask = mask.copy()
    # Copy only training entries, before reductions or any distance operations.
    values = np.full(raw.shape, np.nan, dtype=float)
    values[mask] = raw[mask]
    if not np.isfinite(values[mask]).all():
        raise ValueError("Observed training entries must be finite")
    if not mask.any(axis=0).all():
        raise ValueError("Every feature needs a training observation")
    means = np.empty(raw.shape[1])
    scales = np.empty(raw.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        for feature in range(raw.shape[1]):
            selected = values[mask[:, feature], feature]
            means[feature] = selected.mean()
            scales[feature] = selected.std(ddof=0)
        scales[scales == 0] = 1.0
        values = (values - means) / scales
    if (
        not np.isfinite(means).all()
        or not np.isfinite(scales).all()
        or not np.isfinite(values[mask]).all()
    ):
        raise ValueError("Training statistics exceed supported floating-point range")
    graph_inputs = np.where(mask, values, 0.0)
    distances = cdist(graph_inputs, graph_inputs, metric="euclidean")
    ranking = distances.copy()
    np.fill_diagonal(ranking, np.inf)
    columns = np.argsort(ranking, axis=1, kind="stable")[:, :neighbors].ravel()
    rows = np.repeat(np.arange(len(raw)), neighbors)
    directed = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(raw), len(raw)))
    edges = sparse.triu(directed.maximum(directed.T), 1, format="coo")
    lengths = distances[edges.row, edges.col]
    positive = lengths[lengths > 0]
    if not len(positive) or not np.isfinite(lengths).all():
        raise ValueError("Graph needs finite distances and a positive retained-edge distance")
    bandwidth = float(np.median(positive))
    weights = np.exp(-0.5 * (lengths / bandwidth) ** 2)
    upper = sparse.csr_matrix((weights, (edges.row, edges.col)), shape=directed.shape)
    graph = (upper + upper.T).tocsr()
    graph.eliminate_zeros()
    count, labels = connected_components(graph, directed=False)
    coverage = np.zeros((count, raw.shape[1]), dtype=bool)
    np.logical_or.at(coverage, labels, mask)
    if not coverage.all():
        raise ValueError("Every graph component needs a training observation for every feature")
    return TrainingData(values, mask, means, scales, graph, bandwidth, graph_inputs, int(count))
