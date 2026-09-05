"""Estimator interoperability and regularization-path correctness."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssnalclust.estimator import ConvexClustering, convex_clustering_path
from ssnalclust.solvers import solve


def test_clone_and_parameters():
    model = ConvexClustering(gamma=0.25, n_neighbors=3, cluster_tol=1e-3)
    copied = clone(model)
    assert copied.get_params() == model.get_params()
    assert copied.set_params(gamma=2) is copied
    assert copied.gamma == 2
    assert model.gamma == 0.25


def test_fit_predict_and_pipeline():
    X = np.array([[0.0, 0], [0.01, 0], [5, 5], [5.01, 5]])
    model = ConvexClustering(gamma=1, n_neighbors=1, tol=1e-8)
    labels = model.fit_predict(X)
    assert_array_equal(labels, [0, 0, 1, 1])
    assert model.n_clusters_ == 2
    assert model.n_features_in_ == 2
    assert model.centers_.shape == X.shape
    assert_allclose(model.cluster_centers_, [[0.005, 0], [5.005, 5]], atol=1e-7)
    pipeline = make_pipeline(StandardScaler(), clone(model))
    assert_array_equal(pipeline.fit_predict(X), labels)


def test_zero_gamma_and_singleton():
    X = np.array([[0.0], [1.0], [2.0]])
    model = ConvexClustering(gamma=0).fit(X)
    assert_allclose(model.centers_, X)
    assert_array_equal(model.labels_, [0, 1, 2])
    model.fit([[3, 4]])
    assert_array_equal(model.labels_, [0])
    assert_allclose(model.centers_, [[3, 4]])
    assert model.n_features_in_ == 2
    assert model.converged_


def test_labels_connect_all_pairs_and_are_transitive():
    X = np.array([[0.0], [0.09], [0.18], [2.0]])
    model = ConvexClustering(weights=np.zeros((4, 4)), cluster_tol=0.1).fit(X)
    assert_array_equal(model.labels_, [0, 0, 0, 1])
    assert_allclose(model.cluster_centers_, [[0.09], [2]])


def test_exact_duplicate_labels_with_zero_tolerance():
    model = ConvexClustering(gamma=0, cluster_tol=0).fit([[1], [1], [2]])
    assert_array_equal(model.labels_, [0, 0, 1])


@pytest.mark.parametrize("X", [[[np.nan]], [[np.inf]], [1, 2, 3], []])
def test_invalid_data(X):
    with pytest.raises(ValueError):
        ConvexClustering().fit(X)


@pytest.mark.parametrize(
    "params",
    [
        {"cluster_tol": -1},
        {"cluster_tol": np.nan},
        {"cluster_tol": "bad"},
        {"n_neighbors": 0},
        {"n_neighbors": 1.5},
        {"bandwidth": 0},
    ],
)
def test_invalid_estimator_parameters(params):
    with pytest.raises(ValueError):
        ConvexClustering(**params).fit([[0], [1]])


def test_convergence_warning():
    X = np.random.default_rng(42).normal(size=(12, 3))
    with pytest.warns(ConvergenceWarning):
        model = ConvexClustering(solver="admm", max_iter=1, tol=1e-14).fit(X)
    assert not model.converged_
    assert model.n_iter_ == 1


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_regularization_path_matches_independent_solutions(solver):
    X = np.array([[0.0, 0.0], [0.2, 0.1], [2, 1]])
    strengths = [0, 0.2, 1, 0.1]
    results = convex_clustering_path(X, strengths, solver=solver, tol=1e-7, max_iter=10000)
    for gamma, result in zip(strengths, results):
        expected = solve(X, gamma=gamma, solver=solver, tol=1e-7, max_iter=10000)
        assert result.converged
        assert_allclose(result.centers, expected.centers, atol=2e-5)
        assert_allclose(result.objective, expected.objective, atol=2e-6)


@pytest.mark.parametrize("gammas", [[-1], [np.nan], [np.inf], ["x"], 1.0])
def test_path_rejects_invalid_strengths(gammas):
    with pytest.raises(ValueError, match="gammas"):
        convex_clustering_path([[0], [1]], gammas)


def test_empty_path_and_generator():
    assert convex_clustering_path([[0]], []) == []
    results = convex_clustering_path([[0], [1]], (g for g in [0, 1]))
    assert len(results) == 2


def test_weighted_estimator_fuses_to_weighted_mean():
    X = [[0.0], [2.0]]
    model = ConvexClustering(gamma=10, weights=[[0, 1], [1, 0]], tol=1e-9)
    assert model.fit(X, sample_weight=[1, 3]) is model
    assert_allclose(model.centers_, [[1.5], [1.5]], atol=1e-7)
    assert_array_equal(model.labels_, [0, 0])


def test_weighted_fit_predict():
    model = ConvexClustering(gamma=10, weights=[[0, 1], [1, 0]])
    labels = model.fit_predict([[0.0], [2.0]], sample_weight=[1, 3])
    assert_array_equal(labels, [0, 0])
    assert_allclose(model.cluster_centers_, [[1.5]], atol=1e-5)


@pytest.mark.parametrize("sample_weight", [[0, 1], [-1, 1], [1], [1, np.nan]])
def test_invalid_sample_weights(sample_weight):
    with pytest.raises(ValueError):
        ConvexClustering().fit([[0], [1]], sample_weight=sample_weight)


def test_sklearn_estimator_checks():
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(
        ConvexClustering(),
        expected_failed_checks={
            "check_sample_weight_equivalence_on_dense_data": "Zero fidelity weights are unsupported; duplicating observations "
            "also changes the graph and fusion penalty in transductive clustering.",
        },
        on_skip=None,
    )


def test_twenty_thousand_isolated_clusters():
    from scipy import sparse

    X = np.arange(20000, dtype=float)[:, None]
    model = ConvexClustering(weights=sparse.csr_matrix((len(X), len(X)))).fit(X)
    assert model.n_clusters_ == len(X)
    assert_array_equal(model.labels_, np.arange(len(X)))
    assert_allclose(model.cluster_centers_, X)
