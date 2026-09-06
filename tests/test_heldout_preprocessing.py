"""Training-only feature statistics and graph construction must not leak targets."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


@pytest.fixture
def helper():
    path = Path(__file__).resolve().parents[1] / "examples" / "heldout_preprocessing.py"
    spec = importlib.util.spec_from_file_location("heldout_preprocessing", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data():
    return np.array([[1.0, 5.0], [3.0, 9.0], [100.0, 5.0], [7.0, 5.0]]), np.array(
        [[True, True], [True, False], [False, True], [True, True]]
    )


@pytest.mark.parametrize("hidden", [np.nan, np.inf, -np.inf, 1e300])
def test_hidden_values_cannot_reach_statistics_or_graph(helper, monkeypatch, hidden):
    X, mask = data()
    baseline = helper.prepare_training(X, mask, neighbors=2)
    X[~mask] = hidden
    distance = helper.cdist
    calls = []

    def spy(a, b, **kwargs):
        calls.append(a.copy())
        assert_array_equal(a[~mask], 0.0)
        assert np.isfinite(a).all()
        return distance(a, b, **kwargs)

    monkeypatch.setattr(helper, "cdist", spy)
    actual = helper.prepare_training(X, mask, neighbors=2)
    assert len(calls) == 1
    for name in ("training_values", "means", "scales", "graph_inputs", "observed"):
        assert_array_equal(getattr(actual, name), getattr(baseline, name))
    assert_array_equal(actual.graph.toarray(), baseline.graph.toarray())
    assert actual.bandwidth == baseline.bandwidth
    assert np.isnan(actual.training_values[~mask]).all()
    mask[:] = False
    assert actual.observed.any()


def test_manual_training_statistics_and_constant_feature(helper):
    X, mask = data()
    result = helper.prepare_training(X, mask, neighbors=2)
    assert_allclose(result.means, [11 / 3, 5])
    assert_allclose(
        result.scales, [np.sqrt(((1 - 11 / 3) ** 2 + (3 - 11 / 3) ** 2 + (7 - 11 / 3) ** 2) / 3), 1]
    )
    assert_allclose(
        result.training_values[mask[:, 0], 0], (np.array([1, 3, 7]) - 11 / 3) / result.scales[0]
    )


def test_component_without_feature_training_data_rejected(helper):
    X = np.array([[-10.0, 1.0], [-9.0, 2.0], [9.0, np.nan], [10.0, np.nan]])
    with pytest.raises(ValueError, match="Every graph component"):
        helper.prepare_training(X, np.isfinite(X), neighbors=1)


def test_union_knn_breaks_equal_distance_ties_by_row_index(helper):
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = helper.prepare_training(X, np.ones(X.shape, dtype=bool), neighbors=1)
    expected = np.zeros((4, 4))
    for a, b in ((0, 1), (0, 2), (1, 3)):
        expected[a, b] = expected[b, a] = np.exp(-0.5)
    assert_allclose(result.graph.toarray(), expected)
    assert result.bandwidth == 2


@pytest.mark.parametrize("kind", ["nonfinite", "complex", "mask", "feature", "constant"])
def test_invalid_training_inputs(helper, kind):
    X, mask = data()
    if kind == "nonfinite":
        X[0, 0] = np.nan
    elif kind == "complex":
        X = X.astype(complex)
    elif kind == "mask":
        mask = mask.astype(int)
    elif kind == "feature":
        mask[:, 0] = False
    else:
        X[:] = 1
    with pytest.raises(ValueError):
        helper.prepare_training(X, mask, neighbors=2)


def test_hidden_variants_produce_identical_missing_fits(helper):
    from ssnalclust import solve_missing

    X, mask = data()
    first = helper.prepare_training(X, mask, neighbors=2)
    X[~mask] = np.inf
    second = helper.prepare_training(X, mask, neighbors=2)
    fits = [
        solve_missing(
            p.training_values, p.graph, observed=p.observed, gamma=0.1, tol=1e-6, max_iter=10000
        )
        for p in (first, second)
    ]
    assert all(fit.converged for fit in fits)
    assert_array_equal(fits[0].centers, fits[1].centers)
    for name in ("gap", "kkt_residual", "objective", "relative_gap"):
        assert getattr(fits[0], name) == getattr(fits[1], name)
