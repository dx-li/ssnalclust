"""Label-blind study selection and checkpoint contracts without network access."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse


@pytest.fixture
def study():
    spec = importlib.util.spec_from_file_location(
        "digits_study", Path(__file__).resolve().parents[1] / "examples" / "digits_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_features_and_truth_are_separate(study, tmp_path):
    rows = np.zeros((90, 65), dtype=int)
    rows[:, :64] = 16
    rows[:, 64] = np.arange(90) % 10
    path = tmp_path / "digits.csv"
    np.savetxt(path, rows, delimiter=",", fmt="%d")
    X, truth = study.load_digits_csv(path, smoke=True)
    assert X.shape == (80, 64)
    assert_allclose(X, 1)
    assert_array_equal(truth, rows[:80, 64])
    rows[0, 0] = 17
    np.savetxt(path, rows, delimiter=",", fmt="%d")
    with pytest.raises(ValueError, match="Pixels"):
        study.load_digits_csv(path)


def test_graph_union_bandwidth_and_row_order_ties(study):
    X = np.array([[0.0], [1.0], [-1.0], [4.0]])
    graph, bandwidth = study.make_graph(X, 1)
    assert bandwidth == 1
    expected = np.zeros((4, 4))
    for i, j, distance in [(0, 1, 1), (0, 2, 1), (1, 3, 3)]:
        expected[i, j] = expected[j, i] = np.exp(-0.5 * distance**2)
    assert_allclose(graph.toarray(), expected)
    duplicate_graph, _ = study.make_graph(np.array([[0.0], [0.0], [2.0]]), 1)
    assert duplicate_graph[0, 1] == 1


def selection_inputs():
    X = np.array([[0.0], [0.1], [10.0], [10.1], [20.0]])
    good = np.array([0, 0, 1, 1, 2])
    bad = np.array([0, 1, 0, 1, 2])
    points = [dict(grid_index=i, gamma=float(i + 1), converged=True) for i in range(3)]
    return X, points, [good, bad, good.copy()]


def test_selection_is_label_blind_allows_singletons_and_breaks_ties_first(study, monkeypatch):
    X, points, partitions = selection_inputs()
    original_metric = study.adjusted_rand_score
    monkeypatch.setattr(
        study, "adjusted_rand_score", lambda *a: pytest.fail("Truth used in selection")
    )
    selected = study.select_by_silhouette(X, points, partitions, expected_points=3)
    assert selected["status"] == "valid"
    assert selected["selected_grid_index"] == 0
    assert selected["silhouette_scores"][0] > selected["silhouette_scores"][1]
    monkeypatch.setattr(study, "adjusted_rand_score", original_metric)
    truth = np.array([0, 0, 1, 1, 2])
    first = study.evaluate_truth(truth, partitions, selected)
    second = study.evaluate_truth(truth[[0, 2, 1, 3, 4]], partitions, selected)
    assert first["selected_ari"] != second["selected_ari"]
    again = study.select_by_silhouette(X, points, partitions, expected_points=3)
    assert again["selected_grid_index"] == selected["selected_grid_index"]
    assert again["silhouette_scores"] == selected["silhouette_scores"]


@pytest.mark.parametrize("missing", [False, True])
def test_failed_or_missing_point_prevents_selection(study, missing):
    X, points, partitions = selection_inputs()
    if missing:
        points, partitions = points[:-1], partitions[:-1]
    else:
        points[-1]["converged"] = False
    selected = study.select_by_silhouette(X, points, partitions, expected_points=3)
    assert selected["status"] == "incomplete"
    assert selected["selected_gamma"] is None


def test_no_eligible_partition_is_unavailable(study):
    X, points, _ = selection_inputs()
    selected = study.select_by_silhouette(
        X, points[:2], [np.zeros(5), np.arange(5)], expected_points=2
    )
    assert selected["status"] == "unavailable"
    assert selected["silhouette_scores"] == [None, None]


def test_partition_comparison_ignores_names(study):
    assert_array_equal(study.canonical_partition([9, 4, 9, 8]), [0, 1, 0, 2])
    assert_array_equal(study.canonical_partition([1, 0, 1, 3]), [0, 1, 0, 2])


def test_path_retains_failure_and_warms_from_last_returned_solution(study):
    X = np.array([[0.0], [1.0]])
    graph = sparse.csr_matrix([[0, 1], [1, 0]])
    calls = []

    class Problem:
        def __init__(self, X, weights):
            assert weights is graph

        def solve(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 2:
                raise RuntimeError("deliberate failure")
            return SimpleNamespace(
                centers=X * 0.5,
                dual=np.zeros((1, 1)),
                converged=False,
                n_iter=1,
                objective=1.0,
                dual_objective=0.0,
                gap=1.0,
                relative_gap=0.5,
                kkt_residual=0.5,
                center_error_bound=2**0.5,
                message="budget exhausted",
            )

    points = list(study.fit_path(X, graph, [0.1, 0.2, 0.3], problem_factory=Problem))
    assert len(points) == 3
    assert points[1][0]["status"] == "error"
    assert not points[0][0]["converged"]
    assert calls[0]["x0"] is None
    assert_array_equal(calls[2]["x0"], points[0][1])


def test_unsupported_rss_is_explicit_null(study, monkeypatch):
    monkeypatch.setattr(study.sys, "platform", "win32")
    assert study._peak_rss() is None


def test_small_study_checkpoints_arrays_and_crosschecks(study, tmp_path):
    data = np.zeros((12, 65), dtype=int)
    data[:, :3] = np.random.default_rng(42).integers(0, 17, size=(12, 3))
    data[:, 64] = np.arange(12) % 3
    source = tmp_path / "digits.csv"
    np.savetxt(source, data, delimiter=",", fmt="%d")
    output, arrays_path = tmp_path / "report.json", tmp_path / "arrays.npz"
    report = study.run_study(source, output, arrays_path, neighbors=[2], smoke=True)
    saved = json.loads(output.read_text())
    assert saved == report
    assert report["status"] == "completed"
    assert report["checkpoint_count"] >= 8
    graph = report["graphs"][0]
    assert [p["grid_index"] for p in graph["ssnal"]] == [0, 4, 8]
    assert [p["grid_index"] for p in graph["admm"]] == [0, 4, 8]
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert_allclose(arrays["scaled_features"], data[:, :64] / 16)
        for point in graph["admm"]:
            index = point["grid_index"]
            assert point["status"] == "completed"
            distance = np.linalg.norm(
                arrays[f"k2_admm_{index}_centers"] - arrays[f"k2_ssnal_{index}_centers"]
            )
            assert_allclose(point["ssnal_center_distance"], distance)
            assert point["within_combined_center_bound"]
            assert isinstance(point["same_partition_as_ssnal"], bool)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_nonfinite_silhouette_is_ineligible_and_json_safe(study, monkeypatch, nonfinite):
    X, points, partitions = selection_inputs()
    returned = iter([nonfinite, 0.25, nonfinite])
    monkeypatch.setattr(study, "silhouette_score", lambda *a, **kw: next(returned))
    result = study.select_by_silhouette(X, points, partitions, expected_points=3)
    assert result["selected_position"] == 1
    assert result["silhouette_scores"] == [None, 0.25, None]
    json.dumps(result, allow_nan=False)


def test_interrupted_checkpoint_preserves_json_referenced_arrays(study, monkeypatch, tmp_path):
    output, array_path = tmp_path / "report.json", tmp_path / "arrays.npz"
    study._checkpoint({"keys": ["first"]}, {"first": np.ones(2)}, output, array_path)
    original_replace = study.os.replace

    def interrupt_json(source, destination):
        if destination == output:
            raise OSError("simulated interruption after array replacement")
        original_replace(source, destination)

    monkeypatch.setattr(study.os, "replace", interrupt_json)
    with pytest.raises(OSError, match="simulated"):
        study._checkpoint(
            {"keys": ["first", "second"]},
            {"first": np.ones(2), "second": np.zeros(3)},
            output,
            array_path,
        )
    saved = json.loads(output.read_text())
    with np.load(array_path, allow_pickle=False) as arrays:
        assert all(key in arrays for key in saved["keys"])
        assert_array_equal(arrays["first"], np.ones(2))


def test_study_refuses_to_overwrite_prior_checkpoints(study, tmp_path):
    output = tmp_path / "report.json"
    output.write_text('{"previous":true}')
    with pytest.raises(FileExistsError, match="fresh output paths"):
        study.run_study(tmp_path / "missing.csv", output, tmp_path / "arrays.npz")
    assert json.loads(output.read_text()) == {"previous": True}
