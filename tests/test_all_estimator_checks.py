"""Protocol conformance across models, with explicit scientific exceptions.

The missing-data check fixture supplies an exogenous complete adjacency for
scikit-learn's changing dataset sizes. Production MissingConvexClustering
continues to require a caller-supplied graph; direct tests below use that API.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted

from ssnalclust import (
    ConvexBiclustering,
    GeneralizedConvexClustering,
    MissingConvexClustering,
    SparseConvexClustering,
)


class MissingCompleteGraphFixture(MissingConvexClustering):
    """Adapt only the exogenous graph size to generic check input datasets.

    This graph depends solely on sample count, never on observed values.
    Temporarily providing it exercises the production fit/publish method;
    restoring the constructor parameter preserves clone/parameter checks.
    """

    def fit(self, X, y=None, observed=None):
        original_weights = self.weights
        # Older SciPy sparse arrays can raise while being coerced by NumPy.
        # Leave sparse-input rejection to the production validator.
        shape = X.shape if sparse.issparse(X) else np.asarray(X).shape
        if len(shape) == 2:
            self.weights = np.ones((shape[0], shape[0])) - np.eye(shape[0])
        try:
            return super().fit(X, y=y, observed=observed)
        finally:
            self.weights = original_weights


@pytest.mark.parametrize("estimator", [GeneralizedConvexClustering(), SparseConvexClustering()])
def test_default_estimator_checks_with_documented_score_exception(estimator):
    check_estimator(
        estimator,
        expected_failed_checks={
            "check_clustering": "Default gamma=1 on the complete graph fuses the standardized check dataset. "
            "A fusion-strength model does not promise three groups at this default. "
            "The full check is also run at gamma=.04 with no exclusion.",
        },
        on_skip=None,
    )


@pytest.mark.parametrize(
    "estimator",
    [
        GeneralizedConvexClustering(gamma=0.04),
        SparseConvexClustering(gamma=0.04),
        MissingCompleteGraphFixture(gamma=0.04),
        ConvexBiclustering(),
    ],
)
def test_full_model_estimator_checks(estimator):
    check_estimator(estimator, on_skip=None)


MODEL_NAMES = ["huber", "sparse", "missing", "biclustering"]


def _estimator(name, n_samples, **options):
    weights = np.ones((n_samples, n_samples)) - np.eye(n_samples)
    if name == "huber":
        return GeneralizedConvexClustering(weights=weights, **options)
    if name == "sparse":
        return SparseConvexClustering(weights=weights, **options)
    if name == "missing":
        return MissingConvexClustering(weights=weights, **options)
    return ConvexBiclustering(row_weights=weights, **options)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_singleton_consistent_input_shape(name):
    model = _estimator(name, 1).fit([[3.0, 4.0]])
    assert model.converged_
    assert model.centers_.shape == (1, 2)
    assert model.n_features_in_ == 2
    assert_array_equal(model.labels_, [0])
    if name == "sparse":
        assert_allclose(model.centers_ + model.offset_, [[3, 4]])
    elif name == "biclustering":
        assert model.n_row_clusters_ == 1
        assert model.column_labels_.shape == (2,)
    else:
        assert_allclose(model.centers_, [[3, 4]])


def test_singleton_missing_coordinate_is_unidentifiable():
    with pytest.raises(ValueError, match="unidentifiable"):
        MissingConvexClustering(weights=[[0]]).fit([[np.nan, 4]])


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_clone_fitted_estimator_drops_learned_state_and_copies_weights(name):
    X = [[0.0, 1.0], [0.2, 1.1], [2.0, 0.0], [2.2, 0.1]]
    fitted = _estimator(name, len(X)).fit(X)
    copied = clone(fitted)
    with pytest.raises(NotFittedError):
        check_is_fitted(copied)
    assert not hasattr(copied, "centers_")
    graph_parameter = "row_weights" if name == "biclustering" else "weights"
    original_weights = getattr(fitted, graph_parameter)
    copied_weights = getattr(copied, graph_parameter)
    assert copied_weights is not original_weights
    assert_array_equal(copied_weights, original_weights)
    copied.fit(X)
    assert_allclose(copied.centers_, fitted.centers_)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_feature_names_and_refit_metadata(name):
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame([[0, 1], [0.1, 1.1], [2, 0], [2.1, 0.1]], columns=["height", "width"])
    model = _estimator(name, len(X)).fit(X)
    assert_array_equal(model.feature_names_in_, ["height", "width"])
    model.fit(X.to_numpy())
    assert not hasattr(model, "feature_names_in_")
    assert model.n_features_in_ == 2


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_readonly_observations_and_graph_are_not_mutated(name):
    X = np.array([[0, 1], [0.1, 1.1], [2, 0], [2.1, 0.1]], dtype=float)
    original = X.copy()
    model = _estimator(name, len(X))
    graph_parameter = "row_weights" if name == "biclustering" else "weights"
    weights = getattr(model, graph_parameter)
    original_weights = weights.copy()
    X.flags.writeable = False
    weights.flags.writeable = False
    model.fit(X)
    assert_array_equal(X, original)
    assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_warning_preserves_returned_diagnostics(name):
    X = np.random.default_rng(49).normal(size=(8, 3))
    model = _estimator(name, len(X), max_iter=1, tol=1e-14)
    with pytest.warns(ConvergenceWarning):
        model.fit(X)
    assert not model.converged_
    assert not model.result_.converged
    assert model.n_iter_ == model.result_.n_iter == 1
    assert model.objective_ == model.result_.objective
    assert_allclose(model.centers_, model.result_.centers)
    assert np.isfinite(model.result_.kkt_residual)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_invalid_parameters_do_not_mutate_input_arrays(name):
    X = np.array([[0, 1], [0.1, 1.1], [2, 0], [2.1, 0.1]], dtype=float)
    original = X.copy()
    model = _estimator(name, len(X), max_iter=0)
    graph_parameter = "row_weights" if name == "biclustering" else "weights"
    original_weights = getattr(model, graph_parameter).copy()
    with pytest.raises(ValueError):
        model.fit(X)
    assert_array_equal(X, original)
    assert_array_equal(getattr(model, graph_parameter), original_weights)
    assert not hasattr(model, "centers_")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_invalid_cluster_threshold_rejected_before_optimization(name, monkeypatch):
    import ssnalclust.generalized as generalized
    import ssnalclust.missing as missing
    import ssnalclust.structured as structured

    X = np.array([[0, 1], [0.1, 1.1], [2, 0], [2.1, 0.1]], dtype=float)
    model = _estimator(name, len(X)).fit(X)
    learned = {key: value for key, value in vars(model).items() if key.endswith("_")}
    model.set_params(cluster_tol=-1)

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("Invalid cluster_tol reached the optimizer")

    monkeypatch.setattr(generalized, "solve_generalized", unexpected_solve)
    monkeypatch.setattr(missing, "solve_missing", unexpected_solve)
    monkeypatch.setattr(structured, "solve_sparse", unexpected_solve)
    monkeypatch.setattr(structured, "solve_biclustering", unexpected_solve)
    with pytest.raises(ValueError, match="cluster_tol"):
        model.fit(np.ones((4, 3)))
    for key, value in learned.items():
        assert getattr(model, key) is value
