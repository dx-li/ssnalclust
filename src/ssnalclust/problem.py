"""Reusable fixed-data optimization problems and streamed regularization paths."""

from numbers import Real

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from .graph import graph_from_weights


class ConvexClusteringProblem:
    """Prepare a fixed squared-loss clustering problem for repeated solves.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite real observations. Input data are copied, so later caller
        mutations cannot change the prepared problem.
    weights : dense or sparse adjacency, optional
        Symmetric nonnegative edge weights with zero diagonal. None selects
        a complete unit graph. Sparse adjacency stays sparse.
    sample_weight : array-like of shape (n_samples,), optional
        Strictly positive fidelity masses, copied at construction.

    Notes
    -----
    Graph validation, incidence construction, data centering, and the AMA
    step bound are shared by all solves. ADMM reuses one sparse factorization
    at the most recently requested sigma. Gamma and the fusion norm do not
    change that factorization. The cache is bounded to one factorization.

    A problem owns mutable solver caches and should not be solved concurrently
    from multiple threads; use separate instances for concurrent work. Returned
    results own their arrays and are independent of other results and inputs.
    """

    def __init__(self, X, weights=None, sample_weight=None):
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise ValueError("X must be real")
        x = np.array(raw, dtype=float, copy=True)
        if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError("X must be a nonempty finite 2D array")
        self._b, self._edge_weights = graph_from_weights(weights, len(x))
        self._mass = np.ones((len(x), 1))
        if sample_weight is not None:
            masses = np.asarray(sample_weight)
            if np.iscomplexobj(masses):
                raise ValueError("sample_weight must be real")
            masses = np.array(masses, dtype=float, copy=True)
            if masses.shape != (len(x),) or not np.isfinite(masses).all() or (masses <= 0).any():
                raise ValueError(
                    "sample_weight must have one finite strictly positive value "
                    "per sample; zero is unsupported"
                )
            self._mass = masses[:, None]
        self._original_x = x
        self._offset = np.average(x, axis=0, weights=self._mass[:, 0])
        self._x = x - self._offset
        degree = np.asarray(self._b.power(2).sum(axis=0)).ravel()
        self._lipschitz = 2 * float(np.max(degree / self._mass[:, 0]))
        self._factorization = None
        self._factor_sigma = None

    @property
    def n_samples(self):
        """Number of samples in the fixed problem."""
        return self._x.shape[0]

    @property
    def n_features(self):
        """Number of features in the fixed problem."""
        return self._x.shape[1]

    @property
    def n_edges(self):
        """Number of positive-weight undirected edges."""
        return self._b.shape[0]

    def _linear_solve(self, sigma):
        if self._factorization is None or self._factor_sigma != sigma:
            matrix = sparse.diags(self._mass[:, 0]) + sigma * (self._b.T @ self._b)
            self._factorization = factorized(matrix.tocsc())
            self._factor_sigma = sigma
        return self._factorization

    def solve(
        self,
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
        """Solve with fixed data/weights; options match :func:`solve`.

        Cold starts are the default. To warm-start explicitly pass x0 and
        dual0, or use iter_path/path. store_history=False retains only the
        final diagnostics in the result fields and returns an empty history.
        """
        from .solvers import _solve_prepared

        return _solve_prepared(
            self,
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

    def iter_path(self, gammas, **solver_options):
        """Yield warm-started solutions in order, reusing graph and factors.

        Only the previous primal and dual solution is retained internally.
        Each yielded array is independent: mutating a yielded result does not
        alter the next warm start. Use store_history=False and consume each
        result immediately to bound memory independently of path length.
        A generator of gammas is consumed lazily; invalid later strengths
        raise when reached. Arbitrary increasing/decreasing orders are valid.
        """
        options = dict(solver_options)
        try:
            strengths = iter(gammas)
        except TypeError as exc:
            raise ValueError("gammas must be an iterable of nonnegative numbers") from exc
        for gamma in strengths:
            if (
                not isinstance(gamma, Real)
                or isinstance(gamma, (bool, np.bool_))
                or not np.isfinite(gamma)
                or gamma < 0
            ):
                raise ValueError("gammas must contain finite nonnegative numbers")
            result = self.solve(gamma=gamma, **options)
            # Snapshot before yielding: a caller may mutate/reuse result arrays.
            options.update(x0=result.centers.copy(), dual0=result.dual.copy())
            yield result

    def path(self, gammas, **solver_options):
        """Return a list of warm-started results, retaining the entire path."""
        return list(self.iter_path(gammas, **solver_options))
