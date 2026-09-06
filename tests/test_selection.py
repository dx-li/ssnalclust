"""Holdout selection tests including leakage and independent score checks."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import ssnalclust.selection as selection_module
from ssnalclust.missing import solve_missing
from ssnalclust.selection import select_gamma


@pytest.fixture
def data():
    return np.array([[0.0, 0.1], [0.2, -0.1], [2.0, 1.0], [2.1, 1.2]])


def test_reproducible_fixed_split_and_full_refit(data):
    weights = np.ones((4, 4)) - np.eye(4)
    first = select_gamma(data, [0.1, 0.5], weights, random_state=12, tol=1e-8)
    second = select_gamma(data, [0.1, 0.5], weights, random_state=12, tol=1e-8)
    assert_array_equal(first.training_mask, second.training_mask)
    assert_array_equal(first.validation_mask, second.validation_mask)
    assert_array_equal(first.training_mask | first.validation_mask, np.isfinite(data))
    assert not np.any(first.training_mask & first.validation_mask)
    assert_allclose(first.validation_errors, second.validation_errors, atol=0, rtol=0)
    full = solve_missing(data, weights, gamma=first.best_gamma, tol=1e-8)
    assert_allclose(first.best_result.centers, full.centers, atol=0, rtol=0)
    assert first.best_result.converged


def test_validation_values_erased_from_candidate_calls(data, monkeypatch):
    calls = []
    original = selection_module.solve_missing

    def spy(X, weights, **options):
        calls.append((np.array(X), options["observed"].copy()))
        return original(X, weights, **options)

    monkeypatch.setattr(selection_module, "solve_missing", spy)
    result = select_gamma(data, [0.1, 0.5], np.ones((4, 4)) - np.eye(4), random_state=8)
    assert len(calls) == 3
    for X, mask in calls[:-1]:
        assert np.isnan(X[result.validation_mask]).all()
        assert_array_equal(mask, result.training_mask)
        assert_allclose(X[mask], data[mask])
    assert_allclose(calls[-1][0], data)
    assert_array_equal(calls[-1][1], np.isfinite(data))


def test_scores_and_choice_match_independent_oracle(data):
    cp = pytest.importorskip("cvxpy")
    weights = np.ones((4, 4)) - np.eye(4)
    result = select_gamma(
        data,
        [0.1, 0.5, 1.0],
        weights,
        random_state=4,
        validation_fraction=0.25,
        tol=1e-8,
        max_iter=50000,
    )
    scores = []
    for gamma in result.gammas:
        U = cp.Variable(data.shape)
        objective = 0.5 * cp.sum_squares(cp.multiply(result.training_mask, U - data))
        objective += gamma * sum(cp.norm(U[i] - U[j], 2) for i in range(4) for j in range(i + 1, 4))
        problem = cp.Problem(cp.Minimize(objective))
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_feas=1e-11, tol_gap_rel=1e-11)
        assert problem.status == "optimal"
        scores.append(
            np.mean((U.value[result.validation_mask] - data[result.validation_mask]) ** 2)
        )
    assert_allclose(result.validation_errors, scores, atol=3e-5)
    assert result.best_gamma == result.gammas[np.argmin(scores)]


def test_retains_training_observation_per_component_feature():
    X = [[0, 1], [np.nan, 2], [3, 4], [5, np.nan]]
    weights = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]
    result = select_gamma(X, [0.2], weights, validation_fraction=0.99, random_state=2)
    assert result.validation_mask.sum() == 2
    assert np.all(result.training_mask[:2].sum(axis=0) >= 1)
    assert np.all(result.training_mask[2:].sum(axis=0) >= 1)


def test_nonconverged_candidates_rejected(data):
    with pytest.raises(RuntimeError, match="holdout fit did not converge"):
        select_gamma(
            data, [0.2], np.ones((4, 4)) - np.eye(4), random_state=1, max_iter=1, tol=1e-14
        )


@pytest.mark.parametrize("gammas", [[], [0], [-1], [np.nan], [np.inf], ["a"], 0.5])
def test_invalid_gammas(data, gammas):
    with pytest.raises(ValueError, match="gammas"):
        select_gamma(data, gammas, np.ones((4, 4)) - np.eye(4))


@pytest.mark.parametrize("fraction", [0, 1, -1, np.nan, "a"])
def test_invalid_validation_fraction(data, fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        select_gamma(data, [1], np.ones((4, 4)) - np.eye(4), validation_fraction=fraction)


def test_unidentifiable_or_no_holdout():
    with pytest.raises(ValueError, match="unidentifiable"):
        select_gamma([[np.nan], [np.nan]], [1], [[0, 1], [1, 0]])
    with pytest.raises(ValueError, match="no identifiable holdout"):
        select_gamma([[0], [1]], [1], [[0, 0], [0, 0]])


def test_requires_fixed_graph_and_controls_mask(data):
    with pytest.raises(ValueError, match="explicit fixed graph"):
        select_gamma(data, [1], None)
    with pytest.raises(ValueError, match="observed"):
        select_gamma(data, [1], np.ones((4, 4)) - np.eye(4), observed=np.isfinite(data))


def test_twenty_thousand_isolated_components_reject_without_holdout():
    from scipy import sparse

    X = np.arange(40000, dtype=float).reshape(20000, 2)
    with pytest.raises(ValueError, match="no identifiable holdout"):
        select_gamma(X, [1], sparse.csr_matrix((len(X), len(X))), random_state=0)


def test_large_representable_mean_squared_error_avoids_sum_overflow():
    # Each square is finite but their ordinary sum exceeds float64 range.
    score = selection_module._finite_heldout_mse(np.full(4, 1e154), np.zeros(4))
    assert np.isfinite(score)
    assert_allclose(score, 1e308, rtol=3e-16)
    # One unrepresentable individual square can still have a finite mean.
    score = selection_module._finite_heldout_mse([2e154, 0, 0, 0], np.zeros(4))
    assert_allclose(score, 1e308, rtol=3e-16)


def test_scaled_mse_matches_decimal_reference():
    from decimal import Decimal, localcontext

    predictions = np.array([1.2e154, -0.8e154, 0.3e154, -1e153])
    targets = np.array([0.1e154, 0.2e154, -0.1e154, 0.0])
    with localcontext() as context:
        context.prec = 80
        reference = sum(
            (Decimal.from_float(float(p)) - Decimal.from_float(float(t))) ** 2
            for p, t in zip(predictions, targets)
        ) / Decimal(len(predictions))
    assert_allclose(
        selection_module._finite_heldout_mse(predictions, targets),
        float(reference),
        rtol=1e-15,
    )


@pytest.mark.parametrize(
    "predictions,targets,message",
    [
        ([np.nan], [0.0], "finite predictions"),
        ([np.inf], [0.0], "finite predictions"),
        ([0.0], [-np.inf], "finite predictions"),
        ([1e308], [-1e308], "residuals overflow"),
        ([2e154], [0.0], "not representable"),
    ],
)
def test_invalid_heldout_mse_is_rejected(predictions, targets, message):
    with pytest.raises(ValueError, match=message):
        selection_module._finite_heldout_mse(predictions, targets)


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, 2e154])
def test_invalid_score_aborts_before_refit(data, monkeypatch, invalid_prediction):
    from types import SimpleNamespace

    calls = []

    def fake_solve(X, weights, *, observed, **options):
        calls.append(observed.copy())
        centers = np.zeros_like(X)
        centers[~observed] = invalid_prediction
        return SimpleNamespace(centers=centers, converged=True)

    monkeypatch.setattr(selection_module, "solve_missing", fake_solve)
    with pytest.raises(ValueError, match="held-out"):
        select_gamma(data, [0.1, 0.2], np.ones((4, 4)) - np.eye(4), random_state=3)
    assert len(calls) == 1  # Neither a later candidate nor the full refit ran.
    assert not calls[0].all()


def test_finite_score_ties_keep_first_gamma(data, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake_solve(X, weights, *, gamma, observed, **options):
        calls.append((gamma, observed.copy()))
        return SimpleNamespace(centers=np.zeros_like(X), converged=True)

    monkeypatch.setattr(selection_module, "solve_missing", fake_solve)
    result = select_gamma(data, [0.5, 0.1], np.ones((4, 4)) - np.eye(4), random_state=3)
    assert result.best_gamma == 0.5
    assert result.validation_errors[0] == result.validation_errors[1]
    assert len(calls) == 3
    assert calls[-1][0] == 0.5 and calls[-1][1].all()
