"""Convex clustering with missing entries and a masked squared-error loss."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy.sparse.csgraph import connected_components

from .graph import graph_from_weights
from .prox import project_dual, prox_norm
from .solvers import _positive


@dataclass
class MissingResult:
    """Masked-loss solution with true normalized KKT diagnostics.

    The masked quadratic is not strictly convex on missing coordinates, so
    fitted centroids need not be unique. The objective gap uses the equivalent
    observed-range box problem, not the unconstrained masked Fenchel dual.
    """

    centers: np.ndarray
    dual: np.ndarray
    objective: float
    kkt_residual: float
    n_iter: int
    converged: bool
    history: list
    dual_objective: float = float("nan")
    gap: float = float("nan")
    relative_gap: float = float("nan")
    certificate_model: str = "observed_range_box"


def solve_missing(X, weights, gamma=1.0, penalty="l2", tol=1e-6, max_iter=10000, observed=None):
    """Minimize a masked fidelity loss plus graph fusion using PDHG.

    The objective is ``0.5 * sum(observed * (U-X)**2)`` plus
    ``gamma * sum(w_ij * norm(U_i-U_j, penalty))``. No ridge term or imputed
    fidelity penalty is added. Ignored entries have no effect on the result.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Observations, with nonfinite entries considered missing by default.
    weights : array-like or sparse matrix of shape (n_samples, n_samples)
        Explicit nonnegative symmetric adjacency. Constructing a graph from
        missing data is a modeling choice and is left to the caller.
    gamma : float, default=1.0
        Nonnegative fusion strength.
    penalty : {'l2', 'l1', 'linf'}, default='l2'
        Norm of each centroid difference.
    tol : float, default=1e-6
        Maximum normalized KKT residual and relative objective gap.
    max_iter : int, default=10000
        Maximum primal-dual hybrid gradient iterations.
    observed : boolean array of the same shape as X, optional
        Observed-entry mask. All entries marked True must be finite. False
        entries may contain arbitrary values, including NaN or infinity.

    Returns
    -------
    MissingResult
        Fitted centroids, feasible edge duals, objective, box dual bound and
        gap, original KKT diagnostics, and history (including iteration zero).

    Notes
    -----
    Each feature must be observed somewhere in every connected component of
    the positive fusion graph, otherwise its location is unidentifiable and
    ValueError is raised. With gamma=0 every vertex is its own component, so
    every entry must be observed. This condition ensures coercivity, but does
    not ensure a unique solution; compare objective values and KKT residuals
    when assessing solutions to partially observed problems. Coordinatewise
    clipping to each component/feature observed range preserves the optimum.
    The solver selects a solution in this box and uses its dual bound; this
    is distinct from the unconstrained masked-loss Fenchel dual.
    """
    try:
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise ValueError("X must be real")
        raw = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a numeric two-dimensional array") from exc
    if raw.ndim != 2 or not all(raw.shape):
        raise ValueError("X must have nonempty shape (n_samples, n_features)")
    if observed is None:
        mask = np.isfinite(raw)
    else:
        mask = np.asarray(observed)
        if mask.shape != raw.shape or mask.dtype.kind != "b":
            raise ValueError("observed must be a boolean array with the same shape as X")
        if not np.isfinite(raw[mask]).all():
            raise ValueError("observed entries of X must be finite")
    if weights is None:
        raise ValueError("weights must be an explicit graph adjacency")
    _positive(gamma, "gamma", allow_zero=True)
    _positive(tol, "tol")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if penalty not in ("l2", "l1", "linf"):
        raise ValueError("penalty must be 'l2', 'l1', or 'linf'")
    b, edge_weights = graph_from_weights(weights, len(raw))
    radii = gamma * edge_weights
    if not np.isfinite(radii).all():
        raise ValueError("gamma times weights must be finite")
    if gamma == 0:
        components = np.arange(len(raw))
    else:
        components = connected_components(b[radii > 0].T @ b[radii > 0], directed=False)[1]
    n_components = int(components.max()) + 1
    counts = np.zeros((n_components, raw.shape[1]), dtype=np.intp)
    np.add.at(counts, components, mask)
    if np.any(counts == 0):
        raise ValueError(
            "unidentifiable feature: each graph component must have an observation of every feature"
        )
    lower = np.full(counts.shape, np.inf)
    upper = np.full(counts.shape, -np.inf)
    np.minimum.at(lower, components, np.where(mask, raw, np.inf))
    np.maximum.at(upper, components, np.where(mask, raw, -np.inf))
    offset = lower[components]
    widths = (upper - lower)[components]
    if not np.isfinite(widths).all():
        raise ValueError("observed ranges overflow; rescale X")
    x = np.zeros_like(raw)
    np.subtract(raw, offset, out=x, where=mask)
    means = np.zeros_like(lower)
    # Divide before summation to avoid overflowing a finite component mean.
    np.add.at(means, components, x / counts[components])
    u = np.where(mask, x, means[components])
    z = np.zeros((len(radii), raw.shape[1]))

    def diagnostics(iteration):
        # Certify the representable centers actually returned to the caller.
        restored = np.clip(u + offset, offset, upper[components])
        differences = b @ restored
        bt = b.T @ z
        gradient = np.zeros_like(raw)
        np.subtract(restored, raw, out=gradient, where=mask)
        stationarity = np.linalg.norm(gradient + bt) / (
            1 + np.linalg.norm(gradient) + np.linalg.norm(bt)
        )
        residual = differences - prox_norm(differences + z, radii, penalty)
        subgradient = np.linalg.norm(residual) / (
            1 + np.linalg.norm(differences) + np.linalg.norm(z)
        )
        order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
        group_penalties = radii * np.linalg.norm(differences, ord=order, axis=1)
        objective = 0.5 * np.sum(gradient**2) + np.sum(group_penalties)
        # Express the observed scalar minimizer as X + displacement without
        # rounding that sum. This keeps small dual corrections at large X.
        lower_delta = np.zeros_like(raw)
        upper_delta = np.zeros_like(raw)
        np.subtract(offset, raw, out=lower_delta, where=mask)
        np.subtract(upper[components], raw, out=upper_delta, where=mask)
        displacement = np.clip(-bt, lower_delta, upper_delta)
        delta = gradient - displacement
        missing_target = np.where(bt >= 0, offset, upper[components])
        data_slack = np.where(
            mask,
            0.5 * delta**2 + delta * (displacement + bt),
            bt * (restored - missing_target),
        )
        pairings = np.einsum("ij,ij->i", z, differences)
        edge_slack = group_penalties - pairings
        roundoff = 64 * np.finfo(float).eps
        data_scale = np.where(
            mask,
            0.5 * delta**2 + np.abs(delta * (displacement + bt)),
            np.abs(bt) * (np.abs(restored) + np.abs(missing_target)),
        )
        if np.any(data_slack < -roundoff * data_scale) or np.any(
            edge_slack < -roundoff * (group_penalties + np.abs(pairings))
        ):
            raise FloatingPointError("Negative masked Fenchel gap exceeds floating-point roundoff")
        gap = float(np.maximum(data_slack, 0).sum() + np.maximum(edge_slack, 0).sum())
        reference = np.where(mask, raw, missing_target)
        dual_objective = float(
            np.sum(np.where(mask, 0.5 * displacement**2 + bt * displacement, 0))
            + np.sum(z * (b @ reference))
        )
        relative_gap = gap / (1 + abs(objective) + abs(dual_objective))
        if not np.isfinite([objective, dual_objective, gap, relative_gap]).all():
            raise ValueError("objective diagnostics overflow; rescale X and weights")
        return {
            "iteration": iteration,
            "objective": float(objective),
            "dual_objective": dual_objective,
            "gap": gap,
            "relative_gap": relative_gap,
            "kkt_residual": float(max(stationarity, subgradient)),
        }

    history = [diagnostics(0)]
    extrapolated = u.copy()
    degree = np.asarray(abs(b).sum(axis=0)).ravel()
    # ||B||² <= 2 * max_degree, so tau * sigma * ||B||² < 1.
    step = 0.99 / np.sqrt(max(1.0, 2 * degree.max()))
    for iteration in range(1, max_iter + 1):
        if max(history[-1]["kkt_residual"], history[-1]["relative_gap"]) <= tol:
            break
        z = project_dual(z + step * (b @ extrapolated), radii, penalty)
        previous = u
        u = (u - step * (b.T @ z) + step * x) / (1 + step * mask)
        u = np.clip(u, 0, widths)
        extrapolated = 2 * u - previous
        history.append(diagnostics(iteration))
    last = history[-1]
    return MissingResult(
        np.clip(u + offset, offset, upper[components]),
        z,
        last["objective"],
        last["kkt_residual"],
        last["iteration"],
        max(last["kkt_residual"], last["relative_gap"]) <= tol,
        history,
        last["dual_objective"],
        last["gap"],
        last["relative_gap"],
    )
