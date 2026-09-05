"""Inspect numerical solution paths without assuming hierarchical fusion."""

import numpy as np

from .estimator import _centroid_labels
from .solvers import _positive


def summarize_path(results, cluster_tol=1e-4):
    """Return per-point clusters, certificates, and merge/split transitions.

    Parameters
    ----------
    results : iterable of optimization results
        Results for the same observations, in path evaluation order. Each
        needs centers, objective, converged, n_iter, and kkt_residual fields.
    cluster_tol : float, default=1e-4
        Euclidean threshold used to label transitive centroid components.

    Returns
    -------
    summaries : list of dict
        Each dictionary contains labels, n_clusters, objective, converged,
        n_iter, kkt_residual, merges, and splits. A merge records a current
        label and the previous labels it contains; a split records a previous
        label and its current labels. These describe thresholded partitions,
        not mathematically exact fusion times or an assumed dendrogram.
        The first point has empty transitions. Unconverged points remain
        explicitly marked and should not be used for scientific path claims.
    """
    _positive(cluster_tol, "cluster_tol", allow_zero=True)
    summaries = []
    previous = None
    shape = None
    for result in results:
        centers = np.asarray(result.centers)
        if centers.ndim != 2 or min(centers.shape) < 1 or not np.isfinite(centers).all():
            raise ValueError("each result must contain nonempty finite 2D centers")
        if shape is not None and centers.shape != shape:
            raise ValueError("all results must describe the same samples and features")
        shape = centers.shape
        labels = _centroid_labels(centers, cluster_tol)
        merges, splits = [], []
        if previous is not None:
            pairs = np.unique(np.column_stack((previous, labels)), axis=0)
            # Group only the observed label pairs, rather than scanning every
            # observation once per cluster.
            old_to_new, new_to_old = {}, {}
            for old, new in pairs:
                old_to_new.setdefault(int(old), []).append(int(new))
                new_to_old.setdefault(int(new), []).append(int(old))
            splits = [(old, current) for old, current in old_to_new.items() if len(current) > 1]
            merges = [(new, old) for new, old in new_to_old.items() if len(old) > 1]
        summaries.append(
            dict(
                labels=labels,
                n_clusters=int(labels.max()) + 1,
                objective=result.objective,
                converged=result.converged,
                n_iter=result.n_iter,
                kkt_residual=result.kkt_residual,
                merges=merges,
                splits=splits,
            )
        )
        previous = labels
    return summaries
