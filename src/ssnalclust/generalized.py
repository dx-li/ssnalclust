"""Convex clustering with Huber, Bernoulli, and Poisson fidelity.

The primal-dual hybrid gradient algorithm follows Chambolle and Pock (2011),
https://doi.org/10.1007/s10851-010-0251-1. Each fidelity
proximal step is solved directly; no quadratic-loss dual gap is reused.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import expit

from .graph import graph_from_weights
from .prox import penalty_value, project_dual, prox_norm


@dataclass
class GeneralizedResult:
    """Natural parameters and an evaluated KKT certificate.

    ``centers`` and ``fitted_means`` coincide for Huber. For logistic and
    Poisson loss, fitted means are sigmoid(centers) and exp(centers).
    No quadratic-fidelity duality gap is claimed for these models.
    """

    centers: np.ndarray
    fitted_means: np.ndarray
    objective: float
    kkt_residual: float
    n_iter: int
    converged: bool
    history: list
    dual: np.ndarray


def _fidelity(u, x, loss, delta):
    if loss == "huber":
        residual = u - x
        absolute = np.abs(residual)
        quadratic = np.minimum(absolute, delta)
        value = np.sum(0.5 * quadratic**2 + delta * (absolute - quadratic))
        gradient = np.clip(residual, -delta, delta)
        means = u.copy()
    elif loss == "logistic":
        # This avoids cancellation in log(1+exp(u))-x*u for binary x.
        value = np.sum(
            np.logaddexp(0.0, -np.abs(u)) + (1 - x) * np.maximum(u, 0.0) + x * np.maximum(-u, 0.0)
        )
        means = expit(u)
        # Keep the small failure probability when sigmoid(u) rounds to one.
        gradient = np.where(u >= 0, (1 - x) - expit(-u), means - x)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            means = np.exp(u)
            value = np.sum(means - x * u)
        gradient = means - x
    return float(value), gradient, means


def _prox_fidelity(y, x, step, loss, delta):
    """Separable fidelity prox, with safeguarded monotone scalar roots."""
    if loss == "huber":
        residual = y - x
        return y - step * np.clip(residual / (1 + step), -delta, delta)
    if loss == "logistic":
        low = y - step * (1 - x)
        high = y + step * x
    else:
        # The root lies between y and the unpenalized optimum log(x).
        # At x=0 there is no finite unpenalized optimum; expand a lower
        # bracket instead. These brackets change no model parameters.
        logs = np.zeros_like(x)
        np.log(x, out=logs, where=x > 0)
        low = np.where(x > 0, np.minimum(y, logs), y - 1.0)
        high = np.where(x > 0, np.maximum(y, logs), y)
        # If the root is positive, exp(root) <= x + max(y,0)/step.
        # Form that upper bound in log space even when y itself is huge.
        log_x = np.where(x > 0, logs, -np.inf)
        log_y = np.full_like(y, -np.inf)
        np.log(y, out=log_y, where=y > 0)
        positive_bound = np.maximum(0.0, np.logaddexp(log_x, log_y - np.log(step)))
        high = np.minimum(high, positive_bound)
        low = np.minimum(low, high - 1.0)
        for _ in range(64):
            with np.errstate(over="ignore"):
                below = low - y + step * (np.exp(low) - x) > 0
            if not np.any(below):
                break
            low[below] -= np.maximum(1.0, high[below] - low[below])
        else:
            raise FloatingPointError("Could not bracket Poisson proximal root; rescale X")
    current = np.minimum(np.maximum(y, low), high)
    for _ in range(100):
        if loss == "logistic":
            means = expit(current)
            gradient = np.where(current >= 0, (1 - x) - expit(-current), means - x)
            derivative = 1 + step * means * expit(-current)
        else:
            with np.errstate(over="ignore"):
                means = np.exp(current)
            gradient = means - x
            derivative = 1 + step * means
        residual = current - y + step * gradient
        with np.errstate(invalid="ignore", divide="ignore"):
            correction = residual / derivative
        # An exponential's absolute evaluation error grows with its value.
        # Requiring only a fixed absolute equation residual can therefore
        # reject an already machine-accurate root for large counts. Accept
        # a Newton correction below a few ulps of the natural parameter.
        # The outer KKT check remains unchanged and may still report that
        # the requested global accuracy is unattainable at this data scale.
        accurate = (np.abs(residual) <= 2e-13 * (1 + np.abs(current) + np.abs(y))) | (
            np.isfinite(correction)
            & (np.abs(correction) <= 4 * np.finfo(float).eps * (1 + np.abs(current)))
        )
        if np.all(accurate):
            return current
        high = np.where(residual > 0, current, high)
        low = np.where(residual < 0, current, low)
        with np.errstate(invalid="ignore", divide="ignore"):
            newton = current - correction
        midpoint = low + 0.5 * (high - low)
        margin = 0.01 * (high - low)
        valid = np.isfinite(newton) & (newton > low + margin) & (newton < high - margin)
        next_value = np.where(valid, newton, midpoint)
        # Roots already accurate should not be displaced by bracket logic.
        current = np.where(accurate, current, next_value)
    raise FloatingPointError("Fidelity proximal root did not converge; rescale X")


def _initial_centers(x, b, radii, loss):
    if loss == "huber":
        return x.copy()
    active = b[radii > 0]
    count, labels = connected_components(active.T @ active, directed=False)
    # Accumulate by component in one pass; scanning labels separately for
    # every component becomes quadratic when many vertices are isolated.
    log_sums = np.full((count, x.shape[1]), -np.inf)
    with np.errstate(divide="ignore"):
        np.logaddexp.at(log_sums, labels, np.log(x))
    if loss == "logistic":
        log_failure_sums = np.full_like(log_sums, -np.inf)
        with np.errstate(divide="ignore"):
            np.logaddexp.at(log_failure_sums, labels, np.log1p(-x))
        if np.any(np.isneginf(log_sums) | np.isneginf(log_failure_sums)):
            raise ValueError(
                "logistic loss has no finite optimum: a connected "
                "component has an all-zero or all-one feature"
            )
        # The component-size factors cancel. Computing the two log sums
        # separately retains failure probabilities when mean(x) rounds to1.
        component_centers = log_sums - log_failure_sums
    else:
        if np.any(np.isneginf(log_sums)):
            raise ValueError(
                "poisson loss has no finite optimum: a connected component has an all-zero feature"
            )
        component_centers = log_sums - np.log(np.bincount(labels))[:, None]
    return component_centers[labels]


def solve_generalized(
    X,
    weights=None,
    gamma=1.0,
    loss="huber",
    huber_delta=1.0,
    penalty="l2",
    tol=1e-6,
    max_iter=10000,
):
    """Minimize separable convex fidelity plus weighted centroid fusion.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Huber accepts real finite data. Logistic accepts values in [0, 1]
        (binary observations or Bernoulli proportions); Poisson accepts
        nonnegative counts, including noninteger nonnegative observations.
    weights : symmetric nonnegative adjacency matrix, optional
        Dense or sparse; zero diagonal. None means complete unit graph.
    gamma : float, default=1
        Nonnegative fusion strength applied to natural parameters.
    loss : {'huber', 'logistic', 'poisson'}, default='huber'
        Elementwise fidelity: Huber_delta(U-X), log(1+exp(U))-X*U,
        or exp(U)-X*U. Huber is .5*r**2 inside delta and
        delta*abs(r)-.5*delta**2 outside. Poisson omits log-factorial constants.
    huber_delta : float, default=1
        Strictly positive Huber transition threshold.
    penalty : {'l1', 'l2', 'linf'}, default='l2'
        Norm of each edge's natural-parameter difference.
    tol : float, default=1e-6
        Required maximum normalized stationarity and edge KKT residual.
    max_iter : int, default=10000
        Maximum PDHG iterations; exhaustion returns converged=False.

    Notes
    -----
    These losses are not globally strongly convex; Huber solutions can be
    nonunique. Logistic and Poisson have finite unique natural-parameter
    solutions under the validated component conditions. A component whose
    feature is entirely Bernoulli zero/one or Poisson zero has an unattained
    likelihood infimum and is rejected. At gamma=0 each vertex is a component,
    so boundary observations are rejected, rather than clipped to manufacture
    a finite estimate. Weights remain fixed throughout optimization.

    PDHG uses equal primal and dual steps satisfying
    step**2 * ||B||**2 < 1, with a graph-degree upper bound. Every returned
    history record evaluates the actual objective and KKT equations.
    """
    if np.iscomplexobj(X):
        raise ValueError("X must be real")
    x = np.asarray(X, dtype=float)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("X must be a nonempty finite 2D array")
    for value, name, zero_allowed in [
        (gamma, "gamma", True),
        (tol, "tol", False),
        (huber_delta, "huber_delta", False),
    ]:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
            or (value == 0 and not zero_allowed)
        ):
            raise ValueError(
                f"{name} must be finite and {'nonnegative' if zero_allowed else 'positive'}"
            )
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, Integral) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if loss not in ("huber", "logistic", "poisson"):
        raise ValueError("loss must be 'huber', 'logistic', or 'poisson'")
    if penalty not in ("l1", "l2", "linf"):
        raise ValueError("penalty must be 'l1', 'l2', or 'linf'")
    if loss == "logistic" and np.any((x < 0) | (x > 1)):
        raise ValueError("logistic observations must lie in [0, 1]")
    if loss == "poisson" and np.any(x < 0):
        raise ValueError("poisson observations must be nonnegative")
    b, edge_weights = graph_from_weights(weights, len(x))
    with np.errstate(over="ignore"):
        radii = gamma * edge_weights
    if not np.isfinite(radii).all():
        raise ValueError("gamma times weights must be finite")
    u = _initial_centers(x, b, radii, loss)
    extrapolated = u.copy()
    dual = np.zeros((b.shape[0], x.shape[1]))
    degree = np.asarray(b.power(2).sum(axis=0)).ravel()
    step = 0.99 / np.sqrt(max(1.0, 2 * degree.max()))
    history = []
    for iteration in range(max_iter + 1):
        value, gradient, means = _fidelity(u, x, loss, huber_delta)
        bt = b.T @ dual
        differences = b @ u
        stationarity = np.linalg.norm(gradient + bt) / (
            1 + np.linalg.norm(gradient) + np.linalg.norm(bt)
        )
        edge_residual = differences - prox_norm(differences + dual, radii, penalty)
        edge_error = np.linalg.norm(edge_residual) / (
            1 + np.linalg.norm(differences) + np.linalg.norm(dual)
        )
        kkt = float(max(stationarity, edge_error))
        objective = value + penalty_value(differences, radii, penalty)
        if not np.isfinite(objective) or not np.isfinite(kkt):
            raise FloatingPointError("Objective or residual overflowed; rescale X")
        history.append(dict(iteration=iteration, objective=objective, kkt_residual=kkt))
        if kkt <= tol or iteration == max_iter:
            break
        dual = project_dual(dual + step * (b @ extrapolated), radii, penalty)
        next_u = _prox_fidelity(u - step * (b.T @ dual), x, step, loss, huber_delta)
        extrapolated = 2 * next_u - u
        u = next_u
    return GeneralizedResult(u, means, objective, kkt, iteration, kkt <= tol, history, dual)
