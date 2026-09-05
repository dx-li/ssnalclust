"""Public estimator integration for the alternative objective families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssnalclust.model_estimators import (
    ConvexBiclustering,
    GeneralizedConvexClustering,
    MissingConvexClustering,
    SparseConvexClustering,
)


@pytest.mark.parametrize(
    "model",
    [
        GeneralizedConvexClustering(),
        SparseConvexClustering(),
        ConvexBiclustering(),
        MissingConvexClustering(),
    ],
)
def test_constructor_clone(model):
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("loss", ["huber", "logistic", "poisson"])
def test_generalized_fitted_means(loss):
    x = np.array([[0.2, 0.3], [0.4, 0.5], [0.7, 0.9]])
    fit = GeneralizedConvexClustering(loss=loss, gamma=0.3).fit(x)
    assert fit.converged_
    assert fit.labels_.shape == (3,)
    assert fit.n_features_in_ == 2
    means = (
        fit.centers_
        if loss == "huber"
        else 1 / (1 + np.exp(-fit.centers_))
        if loss == "logistic"
        else np.exp(fit.centers_)
    )
    assert_allclose(means, fit.fitted_means_)


def test_sparse_pipeline():
    x = [[1.0, 0.0], [2.0, 0.0], [8.0, 0.0], [9.0, 0.0]]
    pipeline = make_pipeline(StandardScaler(), SparseConvexClustering(gamma=0.01, alpha=0.1))
    labels = pipeline.fit_predict(x)
    assert labels.shape == (4,)
    assert pipeline[-1].converged_
    assert pipeline[-1].feature_norms_[1] == 0


def test_biclustering_transpose_labels():
    x = np.array([[0.0, 1.0, 8.0], [0.1, 1.1, 8.1], [5.0, 6.0, 13.0]])
    fit = ConvexBiclustering(gamma_row=0.3, gamma_col=0.1).fit(x)
    transposed = ConvexBiclustering(gamma_row=0.1, gamma_col=0.3).fit(x.T)
    assert fit.converged_ and transposed.converged_
    assert_allclose(fit.centers_, transposed.centers_.T, atol=1e-5)
    assert np.array_equal(fit.row_labels_, transposed.column_labels_)


def test_missing_estimator_mask():
    x = np.array([[0.0, np.nan], [1.0, 1.0], [np.nan, 2.0]])
    weights = np.ones((3, 3)) - np.eye(3)
    fit = MissingConvexClustering(weights=weights).fit(x)
    assert fit.converged_
    assert np.isfinite(fit.centers_).all()
    assert fit.__sklearn_tags__().input_tags.allow_nan


@pytest.mark.parametrize(
    "model",
    [
        GeneralizedConvexClustering(max_iter=1),
        SparseConvexClustering(max_iter=1),
        ConvexBiclustering(max_iter=1),
    ],
)
def test_nonconvergence_warning(model):
    with pytest.warns(ConvergenceWarning):
        model.fit([[0.0, 1.0], [5.0, 8.0], [9.0, 10.0]])
    assert not model.converged_
