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
