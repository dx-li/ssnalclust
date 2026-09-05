"""Scikit-learn interfaces for alternative convex clustering objectives."""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import validate_data

from .estimator import _centroid_labels
from .solvers import _positive


def _publish(estimator, result):
    _positive(estimator.cluster_tol, "cluster_tol", allow_zero=True)
    estimator.result_ = result
    estimator.centers_ = result.centers
    estimator.labels_ = _centroid_labels(result.centers, estimator.cluster_tol)
    estimator.n_clusters_ = int(estimator.labels_.max()) + 1
    sums = np.zeros((estimator.n_clusters_, result.centers.shape[1]))
    np.add.at(sums, estimator.labels_, result.centers)
    counts = np.bincount(estimator.labels_, minlength=estimator.n_clusters_)
    estimator.cluster_centers_ = sums / counts[:, None]
    estimator.n_iter_ = result.n_iter
    estimator.objective_ = result.objective
    estimator.converged_ = result.converged
    if not result.converged:
        warnings.warn(
            f"Optimization did not converge (KKT residual {result.kkt_residual:.3g})",
            ConvergenceWarning,
            stacklevel=3,
        )
    return estimator


class MissingConvexClustering(ClusterMixin, BaseEstimator):
    """Masked quadratic convex clustering with an explicit fixed graph.

    Parameters mirror :func:`solve_missing`. ``weights`` must be provided at
    fit time through the constructor; building weights from partially observed
    features is deliberately left to the user. NaN and infinite entries are
    missing by default. ``cluster_tol`` determines transitive centroid labels.
    Learned attributes follow :class:`ConvexClustering` except no dual gap is
    reported. Missing-coordinate centroids need not be unique.
    """

    def __init__(
        self, weights=None, gamma=1.0, penalty="l2", tol=1e-6, max_iter=10000, cluster_tol=1e-4
    ):
        self.weights = weights
        self.gamma = gamma
        self.penalty = penalty
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_tol = cluster_tol

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(self, X, y=None, observed=None):
        """Fit using the optional boolean observed-entry mask."""
        from .missing import solve_missing

        X = validate_data(self, X, dtype=float, ensure_all_finite=False)
        result = solve_missing(
            X, self.weights, self.gamma, self.penalty, self.tol, self.max_iter, observed
        )
        return _publish(self, result)


class GeneralizedConvexClustering(ClusterMixin, BaseEstimator):
    """Huber, Bernoulli-logistic or Poisson graph convex clustering.

    Parameters mirror :func:`solve_generalized`. ``weights=None`` selects a
    complete graph. ``centers_`` and labels use natural-parameter space;
    ``fitted_means_`` transforms these into observation-space fitted means.
    ``cluster_tol`` is a transitive Euclidean threshold in parameter space.
    """

    def __init__(
        self,
        weights=None,
        gamma=1.0,
        loss="huber",
        huber_delta=1.0,
        penalty="l2",
        tol=1e-6,
        max_iter=10000,
        cluster_tol=1e-4,
    ):
        self.weights = weights
        self.gamma = gamma
        self.loss = loss
        self.huber_delta = huber_delta
        self.penalty = penalty
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_tol = cluster_tol

    def fit(self, X, y=None):
        """Fit centroids in the loss's natural parameterization."""
        from .generalized import solve_generalized

        X = validate_data(self, X, dtype=float)
        result = solve_generalized(
            X,
            weights=self.weights,
            gamma=self.gamma,
            loss=self.loss,
            huber_delta=self.huber_delta,
            penalty=self.penalty,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.fitted_means_ = result.fitted_means
        return _publish(self, result)


class SparseConvexClustering(ClusterMixin, BaseEstimator):
    """Feature-group sparse convex clustering on column-centered data.

    Parameters mirror :func:`solve_sparse`. ``weights=None`` selects a complete
    graph. ``centers_`` are centered fitted centroids; add ``offset_`` to restore
    feature locations. ``feature_norms_`` measures each fitted feature's group
    norm. ``cluster_tol`` governs transitive centroid labels.
    """

    def __init__(
        self,
        weights=None,
        gamma=1.0,
        alpha=1.0,
        feature_weights=None,
        tol=1e-6,
        max_iter=10000,
        cluster_tol=1e-4,
    ):
        self.weights = weights
        self.gamma = gamma
        self.alpha = alpha
        self.feature_weights = feature_weights
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_tol = cluster_tol

    def fit(self, X, y=None):
        """Center features and jointly fit fusion and feature selection."""
        from .structured import solve_sparse

        X = validate_data(self, X, dtype=float)
        result = solve_sparse(
            X,
            weights=self.weights,
            gamma=self.gamma,
            alpha=self.alpha,
            feature_weights=self.feature_weights,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        self.offset_ = result.offset
        self.feature_norms_ = result.feature_norms
        return _publish(self, result)


class ConvexBiclustering(BaseEstimator):
    """Simultaneously fuse matrix rows and columns with convex penalties.

    Parameters mirror :func:`solve_biclustering`. None axis weights select
    complete unit graphs. ``centers_`` is the fitted matrix; ``row_labels_``
    and ``column_labels_`` threshold fitted row and column distances using
    ``cluster_tol``. Biclustering is transductive and has no ``predict``.
    """

    def __init__(
        self,
        row_weights=None,
        column_weights=None,
        gamma_row=1.0,
        gamma_col=1.0,
        tol=1e-6,
        max_iter=10000,
        cluster_tol=1e-4,
    ):
        self.row_weights = row_weights
        self.column_weights = column_weights
        self.gamma_row = gamma_row
        self.gamma_col = gamma_col
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_tol = cluster_tol

    def fit(self, X, y=None):
        """Fit a matrix with separate row and column fusion strengths."""
        from .structured import solve_biclustering

        X = validate_data(self, X, dtype=float)
        result = solve_biclustering(
            X,
            row_weights=self.row_weights,
            column_weights=self.column_weights,
            gamma_row=self.gamma_row,
            gamma_col=self.gamma_col,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        _publish(self, result)
        self.row_labels_ = self.labels_.copy()
        self.column_labels_ = _centroid_labels(result.centers.T, self.cluster_tol)
        self.n_row_clusters_ = self.n_clusters_
        self.n_column_clusters_ = int(self.column_labels_.max()) + 1
        return self
