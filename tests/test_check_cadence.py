import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust import solve
from ssnalclust.estimator import _centroid_labels


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_cadence_preserves_accuracy(solver):
    x = [[0.0, 1.0], [0.2, 0.5], [4.0, 5.0]]
    daily = solve(x, gamma=0.5, solver=solver, tol=1e-8, max_iter=5000)
    spaced = solve(x, gamma=0.5, solver=solver, tol=1e-8, max_iter=5000, check_every=7)
    assert daily.converged and spaced.converged
    assert_allclose(daily.centers, spaced.centers, atol=1e-6)
    assert spaced.history[-1]["iteration"] == spaced.n_iter
    assert all(record["iteration"] % 7 == 0 for record in spaced.history)


def test_max_iter_always_checked_even_when_not_cadence_boundary():
    x = np.random.default_rng(2).normal(size=(20, 4))
    result = solve(x, gamma=0.2, solver="admm", tol=1e-12, max_iter=3, check_every=10)
    reference = solve(x, gamma=0.2, solver="admm", tol=1e-12, max_iter=3)
    assert result.n_iter == 3
    assert [row["iteration"] for row in result.history] == [0, 3]
    assert result.converged == reference.converged
    assert_allclose(result.gap, reference.gap)
    assert_allclose(result.kkt_residual, reference.kkt_residual)
    assert_allclose(result.centers, reference.centers)


@pytest.mark.parametrize("value", [0, -1, 1.2, True, "10"])
def test_invalid_cadence(value):
    with pytest.raises(ValueError, match="check_every"):
        solve([[0.0], [1.0]], check_every=value)


def test_duplicate_centroids_only_query_distinct_locations(monkeypatch):
    import ssnalclust.estimator as module

    tree = module.cKDTree
    queries = []

    class CountTree:
        def __init__(self, x):
            self.tree = tree(x)

        def query_ball_point(self, x, tolerance):
            queries.append(1)
            return self.tree.query_ball_point(x, tolerance)

    monkeypatch.setattr(module, "cKDTree", CountTree)
    centers = np.repeat([[4.0, 0.0], [1.0, 0.0], [8.0, 0.0], [2.0, 0.0]], 5000, axis=0)
    labels = _centroid_labels(centers, 1e-4)
    assert np.array_equal(labels, np.repeat(np.arange(4), 5000))
    assert len(queries) <= 4


def test_duplicate_expansion_preserves_transitive_radius_components():
    centers = np.array([[2.0], [0.0], [0.9], [1.8], [0.0], [2.0], [10.0]])
    labels = _centroid_labels(centers, 1.0)
    assert np.array_equal(labels, [0, 0, 0, 0, 0, 0, 1])
