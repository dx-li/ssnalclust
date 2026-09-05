"""Row-wise norm proximal maps and projections onto their dual balls."""

import numpy as np


def project_dual(values, radii, penalty="l2"):
    """Project rows onto balls dual to ``penalty`` with given radii.

    The projection radius does not depend on a proximal step size: the
    conjugate of a weighted norm is an indicator function.
    """
    values = np.asarray(values, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if values.ndim != 2 or radii.shape != (values.shape[0],):
        raise ValueError("values must be 2D and radii must have one entry per row")
    if not np.isfinite(values).all() or not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError("values and radii must be finite; radii must be nonnegative")
    if penalty == "l1":
        return np.clip(values, -radii[:, None], radii[:, None])
    if penalty == "l2":
        # Normalize before computing the norm: both squared norms and the
        # true norm may exceed float64, and radius/norm can underflow.
        maximum = np.max(np.abs(values), axis=1, initial=0.0)
        scaled = np.zeros_like(values)
        np.divide(values, maximum[:, None], out=scaled, where=maximum[:, None] > 0)
        norms = np.linalg.norm(scaled, axis=1)
        threshold = np.full_like(norms, np.inf)
        np.divide(radii, norms, out=threshold, where=norms > 0)
        outside = maximum > threshold
        result = values.copy()
        result[outside] = scaled[outside] / norms[outside, None] * radii[outside, None]
        return result
    if penalty == "linf":
        # Water filling in descending order avoids subtracting a huge
        # threshold from huge, nearly equal coordinates to obtain tiny ones.
        result = values.copy()
        with np.errstate(over="ignore"):
            outside = np.sum(np.abs(values), axis=1) > radii
        v = np.abs(values[outside])
        if len(v):
            ordered = np.sort(v, axis=1)[:, ::-1]
            budget = radii[outside].copy()
            active = np.ones(len(v), dtype=int)
            for k in range(1, values.shape[1]):
                with np.errstate(over="ignore"):
                    cost = k * (ordered[:, k - 1] - ordered[:, k])
                include = (active == k) & (cost < budget)
                budget[include] -= cost[include]
                active[include] = k + 1
            level = ordered[np.arange(len(v)), active - 1]
            above = v >= level[:, None]
            projected = np.where(above, v - level[:, None] + (budget / active)[:, None], 0.0)
            result[outside] = np.sign(values[outside]) * projected
        return result
    raise ValueError("penalty must be 'l1', 'l2', or 'linf'")


def prox_norm(values, radii, penalty="l2"):
    """Proximal map of the sum of row norms, via Moreau decomposition."""
    return np.asarray(values) - project_dual(values, radii, penalty)


def penalty_value(values, radii, penalty):
    """Evaluate a weighted sum of row norms."""
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    return float(radii @ np.linalg.norm(values, ord=order, axis=1))
