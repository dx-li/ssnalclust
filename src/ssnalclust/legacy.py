"""Compatibility with the original column-oriented SSNAL fitting interface."""

import warnings
from numbers import Real

import numpy as np
from scipy import sparse

from .graph import graph_from_weights
from .solvers import solve


def _matrix(value, name, shape=None):
    if sparse.issparse(value) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a real, finite dense matrix")
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real, finite dense matrix") from exc
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a real, finite dense matrix")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return array


class SSNAL:
    """Deprecated adapter for ``SSNAL(A, weights_matrix, gamma).fit(...)``.

    ``A`` has shape (n_features, n_samples), as in the original implementation.
    New applications should use :class:`ssnalclust.ConvexClustering` or
    :func:`ssnalclust.solve`, which put samples in rows.

    The constructor, mutable ``gamma`` parameter, and ``fit`` return tuple are
    retained. Internal routines from the original implementation are not part
    of this compatibility interface. The corrected solver's full diagnostics
    are available as ``result_`` after fitting.
    """

    def __init__(self, A, weights_matrix, gamma):
        warnings.warn(
            "SSNAL's column-oriented interface is deprecated; use ConvexClustering "
            "or solve with samples in rows (A.T).",
            DeprecationWarning,
            stacklevel=2,
        )
        self.A = _matrix(A, "A")
        if min(self.A.shape) < 1:
            raise ValueError("A must be nonempty")
        self.d, self.n = self.A.shape
        self._incidence, self.pre_weights = graph_from_weights(weights_matrix, self.n)
        self.E = self._incidence.shape[0]
        self.weights_matrix = (
            None
            if weights_matrix is None
            else weights_matrix.copy()
            if sparse.issparse(weights_matrix)
            else np.array(weights_matrix, dtype=float, copy=True)
        )
        self.gamma = gamma

    @property
    def gamma(self):
        """Nonnegative fusion strength, changeable between fits."""
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("gamma must be finite and nonnegative")
        with np.errstate(over="ignore"):
            scaled = value * self.pre_weights
        if not np.isfinite(scaled).all():
            raise ValueError("gamma times weights must be finite")
        self.weights = scaled
        self._gamma = value

    def fit(self, max_iter, sigma0=1, eps=1e-6, X0=None, U0=None, Z0=None):
        """Return ``(X, U, Z, status)`` using the legacy column layout.

        ``X`` has shape (n_features, n_samples), while ``U`` and ``Z`` have
        shape (n_features, n_edges). Edges follow lexicographic (i, j) order
        with i < j. ``U`` equals the differences X[:, i] - X[:, j], so the
        returned primal variables are feasible. ``status`` is 0 on convergence
        and 1 on iteration exhaustion.

        ``X0`` and ``Z0`` are warm starts in these same layouts. ``U0`` is
        accepted and validated for compatibility but does not affect the solve:
        the corrected method eliminates that auxiliary variable analytically.
        ``sigma0`` sets the initial augmented-Lagrangian penalty and ``eps``
        requires both the relative duality gap and KKT residual to be small.
        """
        x0 = None if X0 is None else _matrix(X0, "X0", (self.d, self.n)).T
        z0 = None if Z0 is None else _matrix(Z0, "Z0", (self.d, self.E)).T
        if U0 is not None:
            _matrix(U0, "U0", (self.d, self.E))
        self.result_ = solve(
            self.A.T,
            self.weights_matrix,
            gamma=self.gamma,
            solver="ssnal",
            penalty="l2",
            max_iter=max_iter,
            sigma=sigma0,
            tol=eps,
            x0=x0,
            dual0=z0,
        )
        return (
            self.result_.centers.T.copy(),
            (self._incidence @ self.result_.centers).T,
            self.result_.dual.T.copy(),
            0 if self.result_.converged else 1,
        )
