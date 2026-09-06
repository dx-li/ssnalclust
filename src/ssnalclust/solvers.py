"""Sparse convex clustering solvers with primal-dual certificates.

All public arrays use samples in rows. The incidence operator B has one row
per undirected edge. We minimize .5 sum_i m_i ||U_i-X_i||² + p(BU), with
p(V) = gamma sum_e w_e ||V_e||_q. The dual is
<X, B.T Z> - .5 sum_i ||(B.T Z)_i||²/m_i over dual norm balls.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from .prox import penalty_value, project_dual, prox_norm


@dataclass
class SolverResult:
    """Solution and independently evaluated optimality diagnostics.

    ``gap`` is the absolute primal-dual gap; ``relative_gap`` divides by
    ``1 + abs(primal) + abs(dual)``. ``kkt_residual`` is the maximum normalized
    stationarity and proximal residual. Success requires both relative gap
    and KKT residual <= requested tolerance. ``history`` records each outer
    iteration (the initial point has iteration zero).
    """

    centers: np.ndarray
    dual: np.ndarray
    objective: float
    dual_objective: float
    gap: float
    relative_gap: float
    kkt_residual: float
    n_iter: int
    converged: bool
    history: list
    message: str
    strong_convexity: float = 1.0

    @property
    def center_error_bound(self):
        """Frobenius centroid error bound from strong convexity and the gap.

        In exact arithmetic ||U-U*||_F <= sqrt(2*gap/min(sample_weight)).
        This is a numerical evaluation of that bound, not interval arithmetic.
        A small relative gap can coexist with a large absolute error bound.
        """
        return float(np.sqrt(2.0) * np.sqrt(self.gap) / np.sqrt(self.strong_convexity))


def _positive(value, name, allow_zero=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


def _diagnostics(x, u, z, b, radii, penalty, mass):
    # A feasible pair is essential: p(BU), never p(an unconverged split V).
    z = project_dual(z, radii, penalty)
    bt = b.T @ z
    differences = b @ u
    fusion = penalty_value(differences, radii, penalty)
    primal = 0.5 * np.sum(mass * (u - x) ** 2) + fusion
    dual = np.sum(x * bt) - 0.5 * np.sum(bt**2 / mass)
    if not np.isfinite(primal) or not np.isfinite(dual):
        raise FloatingPointError("Objective overflowed; center or rescale X and weights")
    stationarity_vector = mass * (u - x) + bt
    # Fenchel gaps avoid cancellation between large primal/dual objectives.
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    edge_norms = np.linalg.norm(differences, ord=order, axis=1)
    slack = radii * edge_norms - np.sum(z * differences, axis=1)
    scaled_stationarity = np.sqrt(mass) * (u - x) + bt / np.sqrt(mass)
    gap = float(0.5 * np.sum(scaled_stationarity**2) + np.maximum(slack, 0).sum())
    stationarity = np.linalg.norm(stationarity_vector) / (
        1 + np.linalg.norm(mass * (u - x)) + np.linalg.norm(bt)
    )
    residual = differences - prox_norm(differences + z, radii, penalty)
    subgradient = np.linalg.norm(residual) / (1 + np.linalg.norm(differences) + np.linalg.norm(z))
    return dict(
        objective=float(primal),
        dual_objective=float(dual),
        gap=gap,
        relative_gap=float(gap / (1 + abs(primal) + abs(dual))),
        kkt_residual=float(max(stationarity, subgradient)),
    )


def _reduced(x, u, z, b, radii, sigma, mass):
    q = b @ u + z / sigma
    v = prox_norm(q, radii / sigma)
    projected = project_dual(sigma * q, radii)
    value = (
        0.5 * np.sum(mass * (u - x) ** 2)
        + penalty_value(v, radii, "l2")
        + 0.5 * np.sum(projected**2) / sigma
        - 0.5 * np.sum(z**2) / sigma
    )
    grad = mass * (u - x) + b.T @ projected
    return float(value), grad


def _newton_operator(u, z, b, radii, sigma, mass):
    """Generalized Hessian M + sigma B.T J_projection B, matrix free."""
    q = sigma * (b @ u) + z
    if not np.isfinite(q).all():
        raise ValueError("Newton linearization overflow; rescale data or sigma")
    # Normalize before taking a norm: q may be finite even when squaring it
    # overflows or underflows. Compare the radius in the same scaled units.
    magnitude = np.max(np.abs(q), axis=1)
    unit = np.zeros_like(q)
    np.divide(q, magnitude[:, None], out=unit, where=magnitude[:, None] > 0)
    lengths = np.sqrt(np.einsum("ij,ij->i", unit, unit))
    scaled_radii = np.full_like(lengths, np.inf)
    with np.errstate(over="ignore"):
        np.divide(radii, magnitude, out=scaled_radii, where=magnitude > 0)
    outside = lengths > scaled_radii
    scale = np.ones_like(lengths)
    np.divide(scaled_radii, lengths, out=scale, where=outside)
    np.divide(unit, lengths[:, None], out=unit, where=lengths[:, None] > 0)
    # Zero interior directions makes the rank-one correction vanish there.
    # At positive-radius equality we retain the identity Jacobian selection.
    unit[~outside] = 0
    # A radius-zero ball is a constant map, even at its origin.
    scale[radii == 0] = 0

    def matvec(direction):
        direction = direction.reshape(u.shape)
        bd = b @ direction
        radial = np.einsum("ij,ij->i", unit, bd)
        transformed = bd - unit * radial[:, None]
        transformed *= scale[:, None]
        return (mass * direction + sigma * (b.T @ transformed)).ravel()

    diagonal_edges = scale[:, None] * (1 - unit**2)
    diagonal = mass + sigma * (b.power(2).T @ diagonal_edges)
    operator = LinearOperator((u.size, u.size), matvec=matvec, dtype=float)
    preconditioner = LinearOperator(
        operator.shape, matvec=lambda v: v / diagonal.ravel(), dtype=float
    )
    return operator, preconditioner


def _ssn_step(x, u, z, b, radii, sigma, mass, target, inner_max_iter):
    for _ in range(inner_max_iter):
        value, grad = _reduced(x, u, z, b, radii, sigma, mass)
        norm = np.linalg.norm(grad)
        if norm <= target:
            return u, True
        hessian, preconditioner = _newton_operator(u, z, b, radii, sigma, mass)
        direction, info = cg(
            hessian,
            -grad.ravel(),
            M=preconditioner,
            rtol=0.0,
            atol=min(0.01, norm**1.5),
            maxiter=max(50, min(u.size, 1000)),
        )
        direction = direction.reshape(u.shape)
        slope = np.sum(grad * direction)
        if info < 0 or not np.isfinite(direction).all() or slope >= 0:
            direction = -grad
            slope = -(norm**2)
        step = 1.0
        for _ in range(60):
            candidate = u + step * direction
            next_value, next_grad = _reduced(x, candidate, z, b, radii, sigma, mass)
            # At floating-point precision, accept only an improving residual.
            slack = 10 * np.finfo(float).eps * (1 + abs(value))
            if next_value <= value + 1e-4 * step * slope or (
                abs(next_value - value) <= slack and np.linalg.norm(next_grad) < norm
            ):
                u = candidate
                break
            step *= 0.5
        else:
            return u, False
    return u, np.linalg.norm(_reduced(x, u, z, b, radii, sigma, mass)[1]) <= target


def solve(
    X,
    weights=None,
    gamma=1.0,
    penalty="l2",
    solver="ssnal",
    tol=1e-6,
    max_iter=1000,
    *,
    sigma=1.0,
    inner_max_iter=100,
    x0=None,
    dual0=None,
    sample_weight=None,
    store_history=True,
    check_every=1,
):
    """Solve weighted sum-of-norms convex clustering.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite real observations, with samples in rows.
    weights : dense or sparse matrix, optional
        Symmetric nonnegative adjacency, zero diagonal. None means a complete
        graph with unit weights. Prefer sparse weights for large problems.
    gamma : float, default=1
        Nonnegative fusion strength.
    penalty : {'l1', 'l2', 'linf'}, default='l2'
        Norm of each edge difference.
    solver : {'ssnal', 'admm', 'ama', 'fama'}, default='ssnal'
        SSNAL supports l2; the other methods support all three norms. FAMA is
        accelerated dual projected gradient (FISTA acceleration of AMA).
    tol : float, default=1e-6
        Required relative primal-dual gap and normalized KKT residual.
    max_iter : int, default=1000
        Maximum outer iterations. An unconverged result is returned on exhaustion.
    sigma : float, default=1
        Initial augmented-Lagrangian penalty for SSNAL; fixed ADMM penalty.
    inner_max_iter : int, default=100
        Maximum semismooth Newton iterations per SSNAL outer iteration.
    x0, dual0 : arrays, optional
        Warm starts in sample-row and edge-row layout, respectively. The dual
        start is projected onto the current feasible set.
    sample_weight : array of shape (n_samples,), optional
        Strictly positive fidelity weights; defaults to ones.
    store_history : bool, default=True
        Record every iteration. False returns an empty history and retains
        only the final diagnostics; useful for long streamed paths.
    check_every : int, default=1
        Evaluate convergence every this many iterations, and always on the
        final iterate. Values above one reduce certificate evaluation cost
        for first-order methods but can delay stopping. History contains only
        checked iterations. Every returned result is checked regardless.

    Returns
    -------
    SolverResult
        Centers, feasible dual variables, iteration history and certificates.
    """
    from .problem import ConvexClusteringProblem

    problem = ConvexClusteringProblem(X, weights, sample_weight)
    return problem.solve(
        gamma=gamma,
        penalty=penalty,
        solver=solver,
        tol=tol,
        max_iter=max_iter,
        sigma=sigma,
        inner_max_iter=inner_max_iter,
        x0=x0,
        dual0=dual0,
        store_history=store_history,
        check_every=check_every,
    )


def _solve_prepared(
    problem,
    gamma=1.0,
    penalty="l2",
    solver="ssnal",
    tol=1e-6,
    max_iter=1000,
    *,
    sigma=1.0,
    inner_max_iter=100,
    x0=None,
    dual0=None,
    store_history=True,
    check_every=1,
):
    x, original_x = problem._x, problem._original_x
    offset, mass = problem._offset, problem._mass
    b, edge_weights = problem._b, problem._edge_weights
    if not isinstance(store_history, (bool, np.bool_)):
        raise ValueError("store_history must be boolean")
    _positive(gamma, "gamma", allow_zero=True)
    _positive(tol, "tol")
    _positive(sigma, "sigma")
    for value, name in [
        (max_iter, "max_iter"),
        (inner_max_iter, "inner_max_iter"),
        (check_every, "check_every"),
    ]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if penalty not in ("l1", "l2", "linf"):
        raise ValueError("penalty must be 'l1', 'l2', or 'linf'")
    if solver not in ("ssnal", "admm", "ama", "fama"):
        raise ValueError("solver must be 'ssnal', 'admm', 'ama', or 'fama'")
    if solver == "ssnal" and penalty != "l2":
        raise ValueError("SSNAL supports penalty='l2'; use ADMM, AMA or FAMA for other norms")
    radii = gamma * edge_weights
    if not np.isfinite(radii).all():
        raise ValueError("gamma times weights must be finite")
    if np.iscomplexobj(x0) or np.iscomplexobj(dual0):
        raise ValueError("warm starts must be real")
    u = x.copy() if x0 is None else np.array(x0, dtype=float, copy=True) - offset
    z = (
        np.zeros((b.shape[0], x.shape[1]))
        if dual0 is None
        else np.array(dual0, dtype=float, copy=True)
    )
    if u.shape != x.shape or not np.isfinite(u).all():
        raise ValueError("x0 must be finite with the shape of X")
    if z.shape != (b.shape[0], x.shape[1]) or not np.isfinite(z).all():
        raise ValueError("dual0 must be finite with shape (n_edges, n_features)")
    z = project_dual(z, radii, penalty)
    if not np.any(radii):
        u, z = x.copy(), np.zeros_like(z)
    history = []
    status = _diagnostics(x, u, z, b, radii, penalty, mass)
    if store_history:
        history.append(dict(iteration=0, **status))
    if solver == "admm" and np.any(radii):
        linear_solve = problem._linear_solve(sigma)
        v = b @ u
    if solver in ("ama", "fama") and np.any(radii):
        # Gershgorin bound for ||B M^-1 B.T||; unweighted incidence.
        lipschitz = problem._lipschitz
        step_size = 1 / lipschitz
        extrapolated = z.copy()
        momentum = 1.0
    inner_ok = True
    completed_iterations = 0
    for iteration in range(1, max_iter + 1):
        if max(status["relative_gap"], status["kkt_residual"]) <= tol:
            break
        if solver == "ssnal":
            # Inexact ALM: do not solve early subproblems to final accuracy.
            # The absolute cap is summable (paper criterion A'); the KKT-based
            # term tightens the solve as the current outer iterate improves.
            stationarity_scale = 1 + np.linalg.norm(mass * (u - x)) + np.linalg.norm(b.T @ z)
            requested = 0.2 * max(status["kkt_residual"], tol) * stationarity_scale
            target = max(1e-13, min(0.1 / iteration**1.1, requested) / max(1.0, np.sqrt(sigma)))
            u, inner_ok = _ssn_step(x, u, z, b, radii, sigma, mass, target, inner_max_iter)
            z = project_dual(z + sigma * (b @ u), radii, penalty)
            if iteration % 5 == 0:
                sigma = min(sigma * 2, 1e6)
        elif solver == "admm":
            u = linear_solve(mass * x + b.T @ (sigma * v - z))
            bu = b @ u
            v = prox_norm(bu + z / sigma, radii / sigma, penalty)
            z = project_dual(z + sigma * bu, radii, penalty)
        else:
            old_z = z
            primal_at_extrapolation = x - (b.T @ extrapolated) / mass
            z = project_dual(
                extrapolated + step_size * (b @ primal_at_extrapolation), radii, penalty
            )
            u = x - (b.T @ z) / mass
            if solver == "fama":
                next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
                extrapolated = z + (momentum - 1) / next_momentum * (z - old_z)
                momentum = next_momentum
            else:
                extrapolated = z
        completed_iterations = iteration
        if iteration % check_every == 0 or iteration == max_iter:
            status = _diagnostics(x, u, z, b, radii, penalty, mass)
            if store_history:
                history.append(dict(iteration=iteration, **status))
    solved_in_centered_coordinates = max(status["relative_gap"], status["kkt_residual"]) <= tol
    returned_centers = u + offset if np.any(radii) else original_x.copy()
    # Certificates must describe the actually returned, representable array.
    # Restoring a large offset may round away a small optimal displacement.
    status = _diagnostics(x, returned_centers - offset, z, b, radii, penalty, mass)
    if store_history:
        history[-1].update(status)
    converged = bool(max(status["relative_gap"], status["kkt_residual"]) <= tol)
    message = "Converged" if converged else "Maximum iterations reached"
    if solved_in_centered_coordinates and not converged:
        message = "Returned centers cannot meet tolerance at input scale; center or rescale X"
    if not converged and not inner_ok:
        message += "; final SSNAL inner solve did not meet its tolerance"
    return SolverResult(
        returned_centers,
        z,
        **status,
        n_iter=completed_iterations,
        converged=converged,
        history=history,
        message=message,
        strong_convexity=float(mass.min()),
    )
