"""Inspect numerical solution paths without assuming hierarchical fusion."""

import numpy as np

from .estimator import _centroid_labels
from .solvers import _positive


def summarize_path(results, cluster_tol=1e-4):
    """Return a materialized list of :func:`iter_path_summaries` dictionaries.

    Labels and transitions for every point remain in the returned list.
    Consume iter_path_summaries directly to avoid full-path retention.
    """
    return list(iter_path_summaries(results, cluster_tol=cluster_tol))


def iter_path_summaries(results, cluster_tol=1e-4):
    """Yield clusters, numerical certificates, and merge/split transitions.

    Parameters
    ----------
    results : iterable of optimization results
        Results for the same observations in path order. Each must provide
        centers, objective, converged, n_iter, and kkt_residual.
    cluster_tol : float, default=1e-4
        Nonnegative Euclidean threshold for transitive centroid components.
        Validated immediately; individual results are validated as consumed.

    Yields
    ------
    dict
        labels, n_clusters, objective, converged, n_iter, kkt_residual,
        merges, splits, dual_objective, gap, relative_gap, center_error_bound.
        The last four are None when absent from the input result; bounds are
        never inferred for models that do not provide them. A merge pairs a
        current label with its previous labels; a split pairs a previous label
        with its current labels. The first point has empty transitions.

    Notes
    -----
    This describes thresholded partitions, not exact fusion events or an
    assumed hierarchy. Unconverged points remain explicitly marked.
    The iterator keeps a private copy of only the previous partition, not
    earlier fits or summaries. Mutating yielded labels cannot change future
    transitions. The consumer must discard summaries to avoid accumulating
    their arrays. This iterator borrows its input iterable: closing it does
    not close an upstream solver generator owned by the caller.
    """
    _positive(cluster_tol, "cluster_tol", allow_zero=True)
    return _iter_path_summaries(results, cluster_tol)


def _iter_path_summaries(results, cluster_tol):
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
            del pairs, old_to_new, new_to_old
        summary = dict(
            labels=labels,
            n_clusters=int(labels.max()) + 1,
            objective=result.objective,
            converged=result.converged,
            n_iter=result.n_iter,
            kkt_residual=result.kkt_residual,
            merges=merges,
            splits=splits,
        )
        for field in ("dual_objective", "gap", "relative_gap", "center_error_bound"):
            summary[field] = getattr(result, field, None)
        # Preserve transitions even if the consumer changes its label array.
        previous = labels.copy()
        del result, centers, labels, merges, splits
        yield summary
        # Release the preceding output before upstream allocates another fit.
        del summary
