"""Convex clustering with Huber, Bernoulli, and Poisson fidelity.

The primal-dual hybrid gradient algorithm follows Chambolle and Pock (2011),
https://doi.org/10.1007/s10851-010-0251-1. Each fidelity
proximal step is solved directly; no quadratic-loss dual gap is reused.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.special import kl_div, xlogy

from .graph import graph_from_weights
from .prox import project_dual, prox_norm


@dataclass
class GeneralizedResult:
    """Natural parameters with loss-specific primal-dual and KKT certificates.

    ``centers`` and ``fitted_means`` coincide for Huber. For logistic and
    Poisson loss, fitted means are sigmoid(centers) and exp(centers).
    ``dual`` is feasible for both the edge balls and fidelity conjugate.
    ``dual_scale`` records global shrinkage of the internal PDHG multiplier
    to obtain that certificate. A zero scale gives a valid but often weak
    lower bound. Convergence requires both relative gap and KKT <= tol.
    """

    centers: np.ndarray
    fitted_means: np.ndarray
    objective: float
    kkt_residual: float
    n_iter: int
    converged: bool
    history: list
    dual: np.ndarray
    dual_objective: float = float("nan")
    gap: float = float("nan")
    relative_gap: float = float("nan")
    dual_scale: float = float("nan")


def _sigmoid_pair(u):
    """Success and failure probabilities, preserving representable tails."""
    exponential = np.exp(-np.abs(u))
    small = exponential / (1 + exponential)
    large = 1 / (1 + exponential)
    return np.where(u >= 0, large, small), np.where(u >= 0, small, large)


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
        means, failures = _sigmoid_pair(u)
        # Keep the small failure probability when sigmoid(u) rounds to one.
        gradient = np.where(u >= 0, (1 - x) - failures, means - x)
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            means = np.exp(u)
            value = np.sum(means - x * u)
        gradient = means - x
    return float(value), gradient, means


def _certificate_dual(x, b, dual, radii, loss, delta, penalty):
    """Repair the fidelity-conjugate domain by globally shrinking edge duals."""
    edge_feasible = project_dual(dual, radii, penalty)
    divergence = b.T @ edge_feasible
    scale = 1.0
    if loss == "huber":
        maximum = np.max(np.abs(divergence), initial=0.0)
        if maximum > delta:
            scale = delta / maximum
    else:
        positive = divergence > 0
        if np.any(positive):
            with np.errstate(over="ignore"):
                scale = min(scale, float(np.min(x[positive] / divergence[positive])))
        if loss == "logistic":
            negative = divergence < 0
            if np.any(negative):
                with np.errstate(over="ignore"):
                    scale = min(scale, float(np.min((1 - x[negative]) / -divergence[negative])))
    if scale < 1:
        scale = float(np.nextafter(scale, 0.0))

    def in_domain(a):
        if loss == "huber":
            return np.all(np.abs(a) <= delta)
        if loss == "logistic":
            # Comparing divergence directly catches violations that would
            # disappear if computing x-a rounded back onto 0 or 1.
            return np.all((a <= x) & (a >= x - 1))
        return np.all(a <= x)

    # Recompute from the actual arrays to check rounding of graph reductions.
    # Boundary wrong signs may survive every positive rescaling: zero is a
    # valid fallback, not a reason to pretend an infeasible bound is finite.
    for _ in range(5):
        candidate = scale * edge_feasible
        divergence = b.T @ candidate
        if in_domain(divergence):
            return candidate, divergence, scale
        scale = float(np.nextafter(0.99 * scale, 0.0))
    return np.zeros_like(edge_feasible), np.zeros_like(x), 0.0


def _log_mean_kl(q, mean, log_mean):
    """KL(q,exp(log_mean)), retaining finite values after exp underflow."""
    # A caller's sigmoid may underflow before exp(log_mean) does. Recover
    # any still-representable mean before using the true-underflow formula.
    mean = np.array(mean, copy=True)
    zero_mean = mean == 0
    mean[zero_mean] = np.exp(log_mean[zero_mean])
    values = kl_div(q, mean)
    # Some special-function implementations evaluate KL as three large
    # cancelling terms near q=mean. Use its convergent local series instead:
    # KL = mean * sum_{k>=2} (-1)^k t^k/[k(k-1)], t=(q-mean)/mean.
    close = (mean > 0) & (np.abs(q - mean) <= 0.001 * mean)
    t = (q[close] - mean[close]) / mean[close]
    polynomial = np.full_like(t, 1 / 56)
    for coefficient in (-1 / 42, 1 / 30, -1 / 20, 1 / 12, -1 / 6, 0.5):
        polynomial = coefficient + t * polynomial
    values[close] = mean[close] * t**2 * polynomial
    underflow = (mean == 0) & (q > 0)
    if np.any(underflow):
        values[underflow] = q[underflow] * (np.log(q[underflow]) - log_mean[underflow] - 1)
    return values


def _generalized_diagnostics(x, u, dual, b, radii, loss, delta, penalty):
    value, gradient, means = _fidelity(u, x, loss, delta)
    feasible, divergence, scale = _certificate_dual(x, b, dual, radii, loss, delta, penalty)
    differences = b @ u
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    group_penalties = radii * np.linalg.norm(differences, ord=order, axis=1)
    pairings = np.einsum("ij,ij->i", feasible, differences)
    slacks = group_penalties - pairings
    roundoff = 64 * np.finfo(float).eps * (group_penalties + np.abs(pairings))
    if np.any(slacks < -roundoff):
        raise FloatingPointError("Negative edge Fenchel gap exceeds floating-point roundoff")
    if loss == "huber":
        residual = u - x
        clipped = np.clip(residual, -delta, delta)
        data_gaps = 0.5 * (clipped + divergence) ** 2 + np.maximum(
            np.abs(residual) - delta, 0.0
        ) * (delta + np.sign(residual) * divergence)
        # Apply the graph adjoint identity before multiplying large offsets.
        dual_objective = np.einsum("ij,ij->", b @ x, feasible) - 0.5 * np.sum(divergence**2)
    elif loss == "logistic":
        q = x - divergence
        dual_objective = -np.sum(xlogy(q, q) + xlogy(1 - q, 1 - q))
        _, failures = _sigmoid_pair(u)
        data_gaps = _log_mean_kl(q, means, -np.logaddexp(0.0, -u)) + _log_mean_kl(
            1 - q, failures, -np.logaddexp(0.0, u)
        )
    else:
        q = x - divergence
        dual_objective = np.sum(q - xlogy(q, q))
        data_gaps = _log_mean_kl(q, means, u)
    gap = float(np.sum(data_gaps) + np.sum(np.maximum(slacks, 0.0)))
    objective = float(value + np.sum(group_penalties))
    stationarity = np.linalg.norm(gradient + divergence) / (
        1 + np.linalg.norm(gradient) + np.linalg.norm(divergence)
    )
    edge_residual = differences - prox_norm(differences + feasible, radii, penalty)
    edge_error = np.linalg.norm(edge_residual) / (
        1 + np.linalg.norm(differences) + np.linalg.norm(feasible)
    )
    status = dict(
        objective=objective,
        dual_objective=float(dual_objective),
        gap=gap,
        relative_gap=float(gap / (1 + abs(objective) + abs(dual_objective))),
        kkt_residual=float(max(stationarity, edge_error)),
        dual_scale=scale,
    )
    if not all(np.isfinite(number) for number in status.values()) or gap < 0:
        raise FloatingPointError("Objective or certificate overflowed; rescale X")
    return status, feasible, means


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
            means, failures = _sigmoid_pair(current)
            gradient = np.where(current >= 0, (1 - x) - failures, means - x)
            derivative = 1 + step * means * failures
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
        Required relative primal-dual gap and normalized KKT residual.
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
    history record evaluates the actual objective, gap and KKT equations.
    Edge multipliers are globally scaled to satisfy the fidelity-conjugate
    domain before certification. Boundary observations can force that scale
    to zero and give a weak bound even when the raw iterate's KKT is small;
    this is reported as nonconvergence when the requested gap is unmet.
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
        status, feasible, means = _generalized_diagnostics(
            x, u, dual, b, radii, loss, huber_delta, penalty
        )
        history.append(dict(iteration=iteration, **status))
        converged = max(status["relative_gap"], status["kkt_residual"]) <= tol
        if converged or iteration == max_iter:
            break
        dual = project_dual(dual + step * (b @ extrapolated), radii, penalty)
        next_u = _prox_fidelity(u - step * (b.T @ dual), x, step, loss, huber_delta)
        extrapolated = 2 * next_u - u
        u = next_u
    return GeneralizedResult(
        centers=u,
        fitted_means=means,
        **status,
        n_iter=iteration,
        converged=converged,
        history=history,
        dual=feasible,
    )
