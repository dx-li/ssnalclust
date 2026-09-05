"""Convex biclustering and feature-sparse convex clustering via PDHG.

The primal-dual hybrid gradient method uses exact proximal updates and a
conservative operator norm bound. Its stopping test checks stationarity and
every penalty's subgradient inclusion on the returned primal and dual arrays.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from .graph import graph_from_weights
from .prox import penalty_value, project_dual


@dataclass
class StructuredResult:
    """Solution, feasible dual variables, and measured KKT diagnostics.

    ``duals`` maps penalty names to arrays with groups in rows: ``row`` has
    shape (n_row_edges, n_features), ``column`` has shape (n_column_edges,
    n_samples), and ``feature`` has shape (n_features, n_samples). Only the
    penalties belonging to the requested model appear. No duality gap is
    reported; convergence requires ``kkt_residual <= tol``.

    For sparse clustering, ``centers`` are on centered data, ``offset`` is the
    removed feature mean, and ``feature_norms`` contains the column norms of
    these centers. ``centers + offset`` restores the original coordinates.
    For biclustering, ``offset`` and ``feature_norms`` are None.
    """

    centers: np.ndarray
    objective: float
    kkt_residual: float
    n_iter: int
    converged: bool
    history: list
    duals: dict
    offset: np.ndarray | None = None
    feature_norms: np.ndarray | None = None


def _data(X):
    if sparse.issparse(X) or np.iscomplexobj(X):
        raise ValueError("X must be a real, finite dense two-dimensional array")
    try:
        X = np.array(X, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a numeric two-dimensional array") from exc
    if X.ndim != 2 or min(X.shape) < 1 or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite two-dimensional array")
    return X


def _scalar(value, name, positive=False):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        adjective = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {adjective}")


def _radii(strength, weights, name):
    _scalar(strength, name)
    with np.errstate(over="ignore"):
        radii = strength * weights
    if not np.isfinite(radii).all():
        raise ValueError(f"{name} times weights must be finite")
    return radii


@dataclass
class _Term:
    name: str
    radii: np.ndarray
    forward: Callable[[np.ndarray], np.ndarray]
    adjoint: Callable[[np.ndarray], np.ndarray]
    norm_bound_squared: float


def _graph_term(name, B, radii, transpose=False):
    # ||B||^2 = lambda_max(B.T B) <= 2 * maximum vertex degree.
    bound = 2 * float(np.max(np.asarray(B.power(2).sum(axis=0))))
    if transpose:
        return _Term(name, radii, lambda u: B @ u.T, lambda z: (B.T @ z).T, bound)
    return _Term(name, radii, lambda u: B @ u, lambda z: B.T @ z, bound)


def _diagnostics(X, U, duals, terms):
    adjoint = np.zeros_like(U)
    objective = 0.5 * np.sum((U - X) ** 2)
    residual = 0.0
    for term in terms:
        Z = duals[term.name]
        differences = term.forward(U)
        adjoint += term.adjoint(Z)
        objective += penalty_value(differences, term.radii, "l2")
        proximal_residual = Z - project_dual(Z + differences, term.radii)
        residual = max(
            residual,
            np.linalg.norm(proximal_residual)
            / (1 + np.linalg.norm(differences) + np.linalg.norm(Z)),
        )
    stationarity = np.linalg.norm(U - X + adjoint) / (
        1 + np.linalg.norm(U - X) + np.linalg.norm(adjoint)
    )
    return float(objective), float(max(stationarity, residual))


def _pdhg(X, terms, tol, max_iter):
    _scalar(tol, "tol", positive=True)
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    U = X.copy()
    extrapolated = U.copy()
    duals = {term.name: np.zeros_like(term.forward(U)) for term in terms}
    active = [term for term in terms if np.any(term.radii)]
    objective, residual = _diagnostics(X, U, duals, terms)
    history = [dict(iteration=0, objective=objective, kkt_residual=residual)]
    bound = sum(term.norm_bound_squared for term in active)
    # tau * sigma * ||K||^2 < 1 for the vertically stacked operator K.
    step = 0.99 / np.sqrt(bound) if bound else 1.0
    for iteration in range(1, max_iter + 1):
        if residual <= tol:
            break
        adjoint = np.zeros_like(U)
        for term in active:
            Z = project_dual(duals[term.name] + step * term.forward(extrapolated), term.radii)
            duals[term.name] = Z
            adjoint += term.adjoint(Z)
        previous = U
        # prox_{tau f}(V), f(U) = 0.5 ||U-X||_F^2.
        U = (U - step * adjoint + step * X) / (1 + step)
        extrapolated = 2 * U - previous
        objective, residual = _diagnostics(X, U, duals, terms)
        if not np.isfinite(objective) or not np.isfinite(residual):
            raise ValueError("Numerical overflow in structured solve; rescale X and weights")
        history.append(dict(iteration=iteration, objective=objective, kkt_residual=residual))
    return StructuredResult(
        U, objective, residual, len(history) - 1, residual <= tol, history, duals
    )


def solve_biclustering(
    X, row_weights=None, column_weights=None, gamma_row=1.0, gamma_col=1.0, tol=1e-6, max_iter=10000
):
    r"""Solve convex row-and-column fusion without centering the input.

    Minimize over U with shape (n_samples, n_features)::

        0.5 * ||U-X||_F^2
        + gamma_row * sum_e row_weights[e] * ||(B_row U)[e, :]||_2
        + gamma_col * sum_e column_weights[e] * ||(U B_col.T)[:, e]||_2

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite dense data.
    row_weights, column_weights : array-like or sparse matrices, optional
        Symmetric, nonnegative adjacency with zero diagonal on samples and
        features respectively. None selects a complete graph with unit weights.
    gamma_row, gamma_col : float, default=1
        Finite nonnegative row and column fusion strengths.
    tol : float, default=1e-6
        Maximum normalized KKT residual required for convergence.
    max_iter : int, default=10000
        Maximum PDHG iterations. Check ``result.converged`` after solving.

    Returns
    -------
    StructuredResult
        Centers in the input coordinates, objective, KKT diagnostics and duals.
    """
    X = _data(X)
    Br, wr = graph_from_weights(row_weights, X.shape[0])
    Bc, wc = graph_from_weights(column_weights, X.shape[1])
    terms = [
        _graph_term("row", Br, _radii(gamma_row, wr, "gamma_row")),
        _graph_term("column", Bc, _radii(gamma_col, wc, "gamma_col"), transpose=True),
    ]
    return _pdhg(X, terms, tol, max_iter)


def solve_sparse(
    X, weights=None, gamma=1.0, alpha=1.0, feature_weights=None, tol=1e-6, max_iter=10000
):
    r"""Solve feature-sparse convex clustering after centering each feature.

    Let ``offset = mean(X, axis=0)`` and ``C = X - offset``. Minimize::

        0.5 * ||U-C||_F^2
        + gamma * sum_e weights[e] * ||(B U)[e, :]||_2
        + alpha * sum_j feature_weights[j] * ||U[:, j]||_2

    The feature penalty can shrink whole centered columns to zero. Returned
    centers are U, not U + offset; feature norms must be interpreted in those
    centered coordinates. This is explicitly an objective on centered data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite dense observations; features are centered internally.
    weights : array-like or sparse matrix, optional
        Symmetric nonnegative sample adjacency, zero diagonal. None selects
        the complete graph with unit weights.
    gamma, alpha : float, default=1
        Finite nonnegative fusion and feature sparsity strengths respectively.
    feature_weights : array-like of shape (n_features,), optional
        Finite nonnegative feature penalties, defaulting to ones.
    tol : float, default=1e-6
        Maximum normalized KKT residual required for convergence.
    max_iter : int, default=10000
        Maximum PDHG iterations. Check ``result.converged`` after solving.

    Returns
    -------
    StructuredResult
        Centered centers, column ``feature_norms``, removed ``offset``, and
        diagnostics for the objective on centered data.
    """
    X = _data(X)
    with np.errstate(over="ignore", invalid="ignore"):
        offset = X.mean(axis=0)
        X -= offset
    if not np.isfinite(X).all():
        raise ValueError("Centering overflowed; rescale X")
    B, weights = graph_from_weights(weights, X.shape[0])
    if feature_weights is None:
        feature_weights = np.ones(X.shape[1])
    if np.iscomplexobj(feature_weights):
        raise ValueError("feature_weights must be real")
    feature_weights = np.asarray(feature_weights, dtype=float)
    if (
        feature_weights.shape != (X.shape[1],)
        or not np.isfinite(feature_weights).all()
        or (feature_weights < 0).any()
    ):
        raise ValueError("feature_weights must have one finite nonnegative value per feature")
    terms = [
        _graph_term("row", B, _radii(gamma, weights, "gamma")),
        _Term(
            "feature", _radii(alpha, feature_weights, "alpha"), lambda u: u.T, lambda z: z.T, 1.0
        ),
    ]
    result = _pdhg(X, terms, tol, max_iter)
    result.offset = offset
    result.feature_norms = np.linalg.norm(result.centers, axis=0)
    return result
