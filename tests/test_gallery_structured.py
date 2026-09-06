"""Structured gallery coordinate, graph, and certificate contracts."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.distance import cdist


@pytest.fixture
def gallery(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(
        "gallery_structured", examples / "gallery_structured.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loader_does_not_read_cultivar_column(gallery, tmp_path):
    features = np.arange(26, dtype=float).reshape(2, 13)
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    for path, names in [(first, ["unknown", "cultivar"]), (second, ["999", "NaN"])]:
        path.write_text(
            "\n".join(name + "," + ",".join(map(str, row)) for name, row in zip(names, features))
        )
    a, rows = gallery.load_features(first)
    b, _ = gallery.load_features(second)
    assert_array_equal(a, features)
    assert_array_equal(a, b)
    assert_array_equal(rows, [0, 1])


def test_column_graph_uses_feature_profiles_without_restandardization(gallery):
    X = np.array([[0.0, 2.0, 5.0], [1.0, 3.0, 8.0], [4.0, 7.0, 9.0]])
    graph, bandwidth, k = gallery.column_graph(X, neighbors=1)
    distance = cdist(X.T, X.T)
    # Consecutive profiles are nearest; the union includes both edges.
    assert k == 1
    assert_allclose(bandwidth, np.median([distance[0, 1], distance[1, 2]]))
    assert graph.nnz == 4
    assert_allclose(graph[0, 1], np.exp(-0.5 * (distance[0, 1] / bandwidth) ** 2))
    assert_array_equal(graph.toarray(), graph.T.toarray())
    assert not graph.diagonal().any()


def test_fits_preserve_coordinates_mapping_and_certificates(gallery):
    rng = np.random.default_rng(29)
    X = rng.normal(size=(14, 13)) * np.arange(1, 14) + np.arange(13) * 100
    original = X.copy()
    report, arrays = gallery.fit_models(X, smoke=True)
    assert_array_equal(X, original)
    assert_allclose(arrays["standardized"].mean(axis=0), 0, atol=1e-13)
    assert_allclose(arrays["standardized"].std(axis=0), 1, atol=1e-13)
    for point in report["sparse_path"]:
        index = point["index"]
        centered = arrays[f"sparse_{index}_centered"]
        restored = arrays[f"sparse_{index}_standardized_fitted"]
        assert_allclose(restored, centered + arrays[f"sparse_{index}_offset"], atol=0, rtol=0)
        assert_allclose(
            arrays[f"sparse_{index}_original_units_fitted"],
            restored * arrays["feature_scales"] + arrays["feature_means"],
            atol=0,
            rtol=0,
        )
        assert_allclose(arrays["sparse_feature_norms"][index], np.linalg.norm(centered, axis=0))
        assert point["active_features"] == np.count_nonzero(
            arrays["sparse_feature_norms"][index] > report["feature_zero_tol"]
        )
    for point in [*report["sparse_path"], report["biclustering"]]:
        assert point["converged"]
        assert point["gap"] >= 0
        assert max(point["relative_gap"], point["kkt_residual"]) <= report["tol"]
    rows, cols = arrays["row_order"], arrays["column_order"]
    assert_array_equal(np.sort(rows), np.arange(len(X)))
    assert_array_equal(np.sort(cols), np.arange(X.shape[1]))
    assert np.all(np.diff(arrays["biclustering_row_labels"][rows]) >= 0)
    assert np.all(np.diff(arrays["biclustering_column_labels"][cols]) >= 0)
    ordered = arrays["biclustering_standardized_fitted"][np.ix_(rows, cols)]
    restored_order = ordered[np.ix_(np.argsort(rows), np.argsort(cols))]
    assert_array_equal(restored_order, arrays["biclustering_standardized_fitted"])
    assert_allclose(
        arrays["biclustering_original_units_fitted"],
        arrays["biclustering_standardized_fitted"] * arrays["feature_scales"]
        + arrays["feature_means"],
    )
    json.dumps(report, allow_nan=False)


def test_unconverged_fit_fails_loudly(gallery):
    from types import SimpleNamespace

    result = SimpleNamespace(
        converged=False,
        n_iter=20000,
        objective=1.0,
        dual_objective=0.0,
        gap=1.0,
        relative_gap=0.5,
        kkt_residual=0.1,
    )
    with pytest.raises(RuntimeError, match="failed convergence"):
        gallery._diagnostics(result, 0.1)
