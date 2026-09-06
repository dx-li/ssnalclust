"""Scikit-learn estimator and regularization paths for convex clustering."""

import warnings
from numbers import Integral, Real

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array, validate_data

from .graph import k_neighbors_graph
from .solvers import solve


def _centroid_labels(centers, tolerance):
    """Connected components under Euclidean centroid distance <= tolerance."""
    # Exact duplicates form cliques. Querying the same clique once per sample
    # becomes quadratic even though the graph itself is never materialized.
    unique, inverse = np.unique(centers, axis=0, return_inverse=True)
    if len(unique) < len(centers):
        expanded = _centroid_labels(unique, tolerance)[inverse]
        _, first = np.unique(expanded, return_index=True)
        relabel = np.empty(len(first), dtype=np.intp)
        relabel[np.argsort(first)] = np.arange(len(first))
        return relabel[expanded]
    # Traverse the implicit radius graph without materializing its potentially
    # quadratic number of edges (a fully fused solution forms a clique).
    tree = cKDTree(centers)
    labels = np.full(len(centers), -1, dtype=np.intp)
    remaining = len(centers)
    label = 0
    for seed in range(len(centers)):
        if labels[seed] != -1:
            continue
        labels[seed] = label
        remaining -= 1
        pending = [seed]
        while pending and remaining:
            node = pending.pop()
            neighbors = np.asarray(tree.query_ball_point(centers[node], tolerance), dtype=np.intp)
            new = neighbors[labels[neighbors] == -1]
            labels[new] = label
            remaining -= len(new)
            pending.extend(new.tolist())
        label += 1
    return labels


class ConvexClustering(ClusterMixin, BaseEstimator):
    """Cluster observations by convex fusion of their fitted centroids.

    Minimize ``0.5 * ||U - X||_F**2 + gamma * sum(w_ij * ||U_i-U_j||)``.
    The input layout is ``(n_samples, n_features)``. Features are not scaled
    internally; preprocessing can be supplied with a scikit-learn Pipeline.

    Parameters
    ----------
    gamma : float, default=1.0
        Nonnegative fusion strength.
    solver : {'ssnal', 'admm', 'ama', 'fama'}, default='ssnal'
        Optimization algorithm.
    penalty : {'l2', 'l1', 'linf'}, default='l2'
        Norm applied to each weighted centroid difference.
    weights : array-like or sparse matrix of shape (n_samples, n_samples), optional
        Symmetric, nonnegative adjacency. Zero entries omit edges. If None,
        build a symmetric Gaussian weighted nearest-neighbor graph from X.
    n_neighbors : int, default=10
        Number of neighbors when weights is None, clipped to n_samples - 1.
    bandwidth : float, default=1.0
        Gaussian bandwidth when weights is None.
    tol : float, default=1e-6
        Optimization convergence tolerance.
    max_iter : int, default=200
        Maximum outer solver iterations.
    cluster_tol : float, default=1e-4
        Euclidean distance threshold for connecting fitted centroids. Labels
        are the transitive connected components of *all* centroid pairs within
        this threshold, including pairs without an edge in the weight graph.
        Consequently, two centroids in one cluster can be farther apart than
        cluster_tol. This affects labels only, not the optimization solution.

    Attributes
    ----------
    centers_ : ndarray of shape (n_samples, n_features)
        Optimized centroid for every observation.
    labels_ : ndarray of shape (n_samples,)
        Consecutive integer cluster labels.
    cluster_centers_ : ndarray of shape (n_clusters_, n_features)
        Mean of fitted centroids in each thresholded cluster.
    n_clusters_ : int
        Number of thresholded clusters.
    result_ : SolverResult
        Complete optimization result, including convergence diagnostics.
    n_iter_ : int
        Number of solver outer iterations.
    objective_ : float
        Primal objective value.
    dual_gap_ : float
        Primal-dual objective gap.
    center_error_bound_ : float
        Numerical Frobenius centroid-error bound from the absolute duality gap
        and minimum fidelity mass, using strong convexity.
    converged_ : bool
        Whether the solver satisfied its convergence criteria.
    weights_ : scipy.sparse matrix or ndarray
        Weight adjacency passed to the solver.
    n_features_in_ : int
        Number of features seen in fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Input feature names, when all input names are strings.

    Notes
    -----
    Convex clustering is transductive: fitting jointly determines the centroids
    of the supplied observations. No out-of-sample ``predict`` is defined.
    Numerical labels depend on cluster_tol as well as optimization accuracy.
    """

    def __init__(
        self,
        gamma=1.0,
        solver="ssnal",
        penalty="l2",
        weights=None,
        n_neighbors=10,
        bandwidth=1.0,
        tol=1e-6,
        max_iter=200,
        cluster_tol=1e-4,
    ):
        self.gamma = gamma
        self.solver = solver
        self.penalty = penalty
        self.weights = weights
        self.n_neighbors = n_neighbors
        self.bandwidth = bandwidth
        self.tol = tol
        self.max_iter = max_iter
        self.cluster_tol = cluster_tol

    def fit(self, X, y=None, sample_weight=None):
        """Fit the centroid optimization and assign fusion labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Finite real observations.
        y : ignored
            Accepted for scikit-learn compatibility.
        sample_weight : array-like of shape (n_samples,), optional
            Strictly positive fidelity weights. The data-fitting term becomes
            ``0.5 * sum(sample_weight[i] * ||U_i-X_i||**2)``.

        Returns
        -------
        self : ConvexClustering
            Fitted estimator.
        """
        X = validate_data(self, X, dtype=np.float64, ensure_min_samples=1)
        if (
            not isinstance(self.cluster_tol, Real)
            or isinstance(self.cluster_tol, bool)
            or not np.isfinite(self.cluster_tol)
            or self.cluster_tol < 0
        ):
            raise ValueError("cluster_tol must be a finite nonnegative number")
        if self.weights is None:
            if (
                not isinstance(self.n_neighbors, Integral)
                or isinstance(self.n_neighbors, bool)
                or self.n_neighbors < 1
            ):
                raise ValueError("n_neighbors must be a positive integer")
            if (
                not isinstance(self.bandwidth, Real)
                or not np.isfinite(self.bandwidth)
                or self.bandwidth <= 0
            ):
                raise ValueError("bandwidth must be finite and positive")
            weights = (
                sparse.csr_matrix((1, 1))
                if len(X) == 1
                else k_neighbors_graph(
                    X, n_neighbors=min(self.n_neighbors, len(X) - 1), bandwidth=self.bandwidth
                )
            )
        else:
            weights = self.weights
        result = solve(
            X,
            weights=weights,
            gamma=self.gamma,
            penalty=self.penalty,
            solver=self.solver,
            tol=self.tol,
            max_iter=self.max_iter,
            sample_weight=sample_weight,
        )
        self.weights_ = weights.copy() if hasattr(weights, "copy") else np.array(weights)
        self.result_ = result
        self.centers_ = result.centers
        self.labels_ = _centroid_labels(self.centers_, self.cluster_tol)
        self.n_clusters_ = int(self.labels_.max()) + 1
        self.cluster_centers_ = np.zeros((self.n_clusters_, self.centers_.shape[1]))
        np.add.at(self.cluster_centers_, self.labels_, self.centers_)
        self.cluster_centers_ /= np.bincount(self.labels_)[:, None]
        self.n_iter_ = result.n_iter
        self.objective_ = result.objective
        self.dual_gap_ = result.gap
        self.center_error_bound_ = result.center_error_bound
        self.converged_ = result.converged
        if not self.converged_:
            warnings.warn(
                f"{self.solver}: {result.message} "
                f"(KKT residual {result.kkt_residual:.3g}, relative gap {result.relative_gap:.3g}).",
                ConvergenceWarning,
                stacklevel=2,
            )
        return self


def convex_clustering_path(X, gammas, weights=None, **solver_options):
    """Solve a sequence of fusion strengths with primal and dual warm starts.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite observations.
    gammas : iterable of nonnegative finite floats
        Strengths in the desired evaluation order; sorting is not required.
    weights : array-like or sparse matrix, optional
        Fixed graph adjacency used for every strength. None uses the complete
        graph with unit weights, matching ``solve``. To match the estimator's
        default graph, explicitly supply ``k_neighbors_graph(X, ...)``.
    **solver_options
        Additional keyword arguments to ``solve``, such as solver, penalty,
        tol, and max_iter. Initial x0 and dual0 may be provided for the first
        point. Later points use the preceding optimization result.

    Returns
    -------
    results : list of SolverResult
        Solutions in input gamma order. Inspect each result's converged flag;
        this low-level utility does not emit convergence warnings.
    """
    X = check_array(X, dtype=np.float64, ensure_min_samples=1)
    try:
        values = list(gammas)
    except TypeError as exc:
        raise ValueError("gammas must be an iterable of nonnegative numbers") from exc
    if any(
        not isinstance(gamma, Real)
        or isinstance(gamma, bool)
        or not np.isfinite(gamma)
        or gamma < 0
        for gamma in values
    ):
        raise ValueError("gammas must contain finite nonnegative numbers")
    options = dict(solver_options)
    from .problem import ConvexClusteringProblem

    problem = ConvexClusteringProblem(X, weights, options.pop("sample_weight", None))
    return problem.path(values, **options)


def iter_convex_clustering_path(X, gammas, weights=None, **solver_options):
    """Stream fixed-graph path results with bounded retained solver state.

    Options match convex_clustering_path. Unlike its list-returning sibling,
    strengths are validated as consumed. Graph construction and an ADMM
    factorization are shared across path points. Set store_history=False to
    omit per-iteration histories; consume and release results to avoid storing
    the full centroid trajectory in memory.
    """
    from .problem import ConvexClusteringProblem

    options = dict(solver_options)
    problem = ConvexClusteringProblem(X, weights, options.pop("sample_weight", None))
    return problem.iter_path(gammas, **options)
