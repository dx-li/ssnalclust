"""Real-data gallery uses disjoint covariates and saves auditable raw-unit scores."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse

from ssnalclust import solve_missing


@pytest.fixture
def gallery(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(
        "gallery_missing", examples / "gallery_missing.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_values_cannot_change_graph(gallery):
    values = gallery.load_features(smoke=True)
    before = gallery.graph_from_covariates(values, 5)
    values[::2, 4] = np.nan
    values[1::2, 4] = 1e100
    after = gallery.graph_from_covariates(values, 5)
    assert_array_equal(before.graph.toarray(), after.graph.toarray())
    assert_array_equal(before.graph_inputs, after.graph_inputs)
    assert_array_equal(before.means, after.means)


def test_saved_masked_fit_scores_and_target_erasure(gallery, tmp_path, monkeypatch):
    # Plot rendering is checked by the gallery run; this test targets numerical artifacts.
    monkeypatch.setattr(gallery, "plot", lambda *args: None)
    report = gallery.run(tmp_path, smoke=True)
    assert report == json.loads((tmp_path / "gallery_missing.json").read_text())
    with np.load(tmp_path / "gallery_missing.npz", allow_pickle=False) as arrays:
        x = arrays["raw_target"]
        mask = arrays["training_mask"]
        hidden = arrays["validation_mask"]
        assert_array_equal(mask, ~hidden)
        assert report["validation_count"] == 2
        assert_allclose(report["training_mean"], x[mask].mean())
        assert_allclose(report["baseline_mse"], np.mean((x[hidden] - x[mask].mean()) ** 2))
        graph = sparse.csr_matrix(
            (arrays["graph_data"], arrays["graph_indices"], arrays["graph_indptr"]),
            shape=(len(x), len(x)),
        )
        for index, point in enumerate(report["candidates"]):
            centers = arrays[f"candidate_{index}_centers"]
            assert_allclose(point["mse"], np.mean((centers[hidden] - x[hidden]) ** 2))
            assert point["converged"]
        # A changed target at hidden entries must not alter the fitted training problem.
        changed = x.copy()
        changed[hidden] = 1e100
        result = solve_missing(
            changed, graph, gamma=report["gammas"][0], observed=mask, tol=1e-6, max_iter=20000
        )
        assert result.converged
        assert_allclose(result.centers, arrays["candidate_0_centers"], rtol=0, atol=0)
        assert_allclose(result.dual, arrays["candidate_0_dual"], rtol=0, atol=0)


def test_failure_is_saved_without_claiming_selection(gallery, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("nonconverged candidate")

    monkeypatch.setattr(gallery, "select_gamma", fail)
    with pytest.raises(RuntimeError, match="nonconverged"):
        gallery.run(tmp_path, smoke=True)
    report = json.loads((tmp_path / "gallery_missing.json").read_text())
    assert report["status"] == "error"
    assert "selected_gamma" not in report
    assert (tmp_path / "gallery_missing.npz").exists()
