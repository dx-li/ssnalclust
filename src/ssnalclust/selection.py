"""Reproducible entry holdout selection for missing-data convex clustering."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.sparse.csgraph import connected_components

from .graph import graph_from_weights
from .missing import MissingResult, solve_missing


@dataclass
class SelectionResult:
    """Holdout scores, masks, candidate fits, and the selected full-data fit."""

    best_gamma: float
    gammas: np.ndarray
    validation_errors: np.ndarray
    results: list
    training_mask: np.ndarray
    validation_mask: np.ndarray
    best_result: MissingResult


def _finite_heldout_mse(predictions, targets):
    """Compute finite MSE without overflowing a representable mean of squares."""
    predictions = np.asarray(predictions, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("held-out MSE requires finite predictions and targets")
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = predictions - targets
    if not np.isfinite(residuals).all():
        raise ValueError("held-out residuals overflow; rescale the data")
    scale = float(np.max(np.abs(residuals)))
    if scale == 0:
        return 0.0
    # Normalize before either squaring or summing; intermediate sums of the
    # original squares can overflow even when their mean is representable.
    with np.errstate(over="ignore", invalid="ignore"):
        rmse = scale * np.sqrt(np.mean((residuals / scale) ** 2))
        mse = float(rmse * rmse)
    if not np.isfinite(mse):
        raise ValueError("held-out MSE is not representable as a finite float; rescale the data")
    return mse


def select_gamma(
    X, gammas, weights, validation_fraction=0.1, random_state=None, **missing_solver_options
):
    """Choose fusion strength by held-out entry MSE and refit observed data.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Finite values are observed; nonfinite values are missing.
    gammas : iterable of positive finite floats
        Candidate strengths in evaluation order. Zero is excluded because a
        held-out entry has no identifiable estimate without graph fusion.
    weights : array-like or sparse matrix of shape (n_samples, n_samples)
        Explicit fixed graph. Use exogenous weights: a graph built from all
        values of X, including held-out entries, leaks validation information.
        No preprocessing or graph estimation is performed internally.
    validation_fraction : float, default=0.1
        Fraction of observed entries to hold out, between zero and one. The
        actual count is bounded so each component/feature keeps at least one
        training observation. At least one validation entry is required.
    random_state : int, numpy.random.Generator or None, default=None
        Seed or generator for the reproducible entry split.
    **missing_solver_options
        Options such as penalty, tol and max_iter for solve_missing. The
        observed mask is controlled by this function and cannot be overridden.

    Returns
    -------
    SelectionResult
        Candidate strengths and MSEs, training/validation masks, candidate
        fits, and best_result refitted on every originally observed entry.
        Ties select the first candidate. Every returned fit has converged.

    Notes
    -----
    This is a single holdout split, not an unbiased final performance estimate.
    Entries are split jointly once, then the same training data are used at
    every strength. Unidentifiable data and nonconverged fits raise errors.
    Nonfinite held-out predictions, overflowing residuals, or an unrepresentable
    MSE raise ValueError before selection and refitting; candidates are never
    silently discarded. Missing-coordinate optima need not be unique: scores
    evaluate solve_missing's deterministic initialization and observed-range
    box convention, not a uniquely determined imputation.
    """
    try:
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise ValueError("X must be real")
        x = raw.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("X must be a real numeric array") from exc
    if x.ndim != 2 or not all(x.shape):
        raise ValueError("X must have nonempty shape (n_samples, n_features)")
    if weights is None:
        raise ValueError("weights must be an explicit fixed graph")
    if (
        not isinstance(validation_fraction, Real)
        or isinstance(validation_fraction, bool)
        or not np.isfinite(validation_fraction)
        or not 0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be strictly between zero and one")
    try:
        strengths = list(gammas)
    except TypeError as exc:
        raise ValueError("gammas must be a nonempty iterable of positive finite numbers") from exc
    if not strengths or any(
        not isinstance(g, Real) or isinstance(g, bool) or not np.isfinite(g) or g <= 0
        for g in strengths
    ):
        raise ValueError("gammas must be a nonempty iterable of positive finite numbers")
    if "observed" in missing_solver_options:
        raise ValueError("observed is controlled by the holdout split")
    b, _ = graph_from_weights(weights, len(x))
    components = connected_components(b.T @ b, directed=False)[1]
    observed = np.isfinite(x)
    rng = np.random.default_rng(random_state)
    # Randomly order observed entries, then reserve the first entry in each
    # component/feature group. Group reductions avoid scanning all rows for
    # each component, which is quadratic for graphs with many components.
    entries = rng.permutation(np.flatnonzero(observed))
    n_features = x.shape[1]
    n_groups = (int(components.max()) + 1) * n_features
    groups = components[entries // n_features] * n_features + entries % n_features
    first = np.full(n_groups, len(entries), dtype=np.intp)
    np.minimum.at(first, groups, np.arange(len(entries)))
    if np.any(first == len(entries)):
        raise ValueError("unidentifiable feature in a graph component")
    eligible = np.ones(len(entries), dtype=bool)
    eligible[first] = False
    candidates = entries[eligible]
    if not len(candidates):
        raise ValueError(
            "no identifiable holdout split: every observation is required for training"
        )
    n_validation = min(len(candidates), max(1, int(validation_fraction * observed.sum())))
    selected = rng.choice(candidates, size=n_validation, replace=False)
    validation = np.zeros_like(observed)
    validation.flat[selected] = True
    training = observed & ~validation
    # Physically erase validation values as well as passing the training mask.
    # This keeps validation targets out of the optimization call entirely.
    training_x = np.where(training, x, np.nan)
    results, errors = [], []
    for gamma in strengths:
        result = solve_missing(
            training_x, weights, gamma=gamma, observed=training, **missing_solver_options
        )
        if not result.converged:
            raise RuntimeError(
                f"holdout fit did not converge for gamma={gamma}; increase max_iter or relax tol"
            )
        results.append(result)
        errors.append(_finite_heldout_mse(result.centers[validation], x[validation]))
    best = int(np.argmin(errors))
    best_result = solve_missing(
        x, weights, gamma=strengths[best], observed=observed, **missing_solver_options
    )
    if not best_result.converged:
        raise RuntimeError(
            "selected full-data refit did not converge; increase max_iter or relax tol"
        )
    return SelectionResult(
        float(strengths[best]),
        np.asarray(strengths, dtype=float),
        np.asarray(errors),
        results,
        training,
        validation,
        best_result,
    )
