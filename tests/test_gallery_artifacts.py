"""Integrity and scientific coverage of saved full gallery runs, without refitting.

These checks validate provenance, saved stopping diagnostics and array/report
contracts; they do not claim an independent optimization oracle or model validity.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit

DOCS = Path(__file__).resolve().parents[1] / "docs"
RESULTS = DOCS / "gallery_results"
MANIFEST = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))


def _report(group):
    paths = MANIFEST["groups"][group]["reports"]
    assert len(paths) == 1
    return json.loads((DOCS / paths[0]).read_text(encoding="utf-8"))


def _arrays(group):
    paths = MANIFEST["groups"][group]["arrays"]
    assert len(paths) == 1
    return np.load(DOCS / paths[0], allow_pickle=False)


def _converged(point, tol):
    assert point["converged"] is True
    fields = ["objective", "dual_objective", "gap", "relative_gap", "kkt_residual"]
    assert np.isfinite([point[key] for key in fields]).all()
    assert point["gap"] >= 0
    assert 0 <= point["relative_gap"] <= tol
    assert 0 <= point["kkt_residual"] <= tol
    assert isinstance(point["n_iter"], int) and point["n_iter"] >= 0
    assert_allclose(
        point["relative_gap"],
        point["gap"] / (1 + abs(point["objective"]) + abs(point["dual_objective"])),
        rtol=1e-12,
        atol=1e-16,
    )


@pytest.mark.parametrize("artifact", MANIFEST["artifacts"], ids=lambda item: item["path"])
def test_saved_artifact_matches_manifest(artifact):
    path = DOCS / artifact["path"]
    assert path.resolve().is_relative_to(DOCS.resolve())
    content = path.read_bytes()
    assert len(content) == artifact["bytes"]
    assert hashlib.sha256(content).hexdigest() == artifact["sha256"]
    if path.suffix == ".png":
        assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_manifest_covers_all_four_methods_and_ten_figures():
    assert MANIFEST["schema_version"] == 1
    assert set(MANIFEST["groups"]) == {"solvers", "missing", "structured", "generalized"}
    paths = [item["path"] for item in MANIFEST["artifacts"]]
    assert len(paths) == len(set(paths)) == 18
    expected = {
        "gallery_solvers.png",
        "gallery_convergence.png",
        "gallery_paths.png",
        "gallery_graphs.png",
        "gallery_missing.png",
        "gallery_sparse.png",
        "gallery_biclustering.png",
        "gallery_huber.png",
        "gallery_logistic.png",
        "gallery_poisson.png",
    }
    assert {Path(p).name for p in paths if p.endswith(".png")} == expected
    discovered = {
        p.relative_to(DOCS).as_posix()
        for group in MANIFEST["groups"]
        for p in (RESULTS / group).iterdir()
        if p.suffix in {".json", ".npz"}
    }
    assert {p for p in paths if not p.endswith(".png")} == discovered
    for group, records in MANIFEST["groups"].items():
        assert {p for values in records.values() for p in values} == {
            item["path"] for item in MANIFEST["artifacts"] if item["group"] == group
        }


@pytest.mark.parametrize("group", ["solvers", "missing", "structured", "generalized"])
def test_npz_arrays_are_portable_without_object_pickle(group):
    with _arrays(group) as arrays:
        assert len(arrays.files) > 0
        for name in arrays.files:
            value = arrays[name]
            assert value.dtype.kind != "O", name
            if value.dtype.kind in "biufc":
                assert np.isfinite(value).all(), name


def test_solver_gallery_covers_algorithms_norms_paths_and_graphs():
    report = _report("solvers")
    assert report["smoke"] is False and report["status"] == "completed"
    assert (report["samples"], report["features"]) == (178, 13)
    assert {(p["solver"], p["penalty"]) for p in report["fits"]} == {
        ("ssnal", "l2"),
        ("admm", "l2"),
        ("ama", "l2"),
        ("fama", "l2"),
        ("admm", "l1"),
        ("admm", "linf"),
    }
    assert len(report["paths"]) >= 4
    assert {g["name"] for g in report["graph_variants"]} == {
        "Gaussian kNN",
        "Minimum spanning tree",
        "Connected kNN",
        "Local scales",
    }
    for point in [*report["fits"], *report["paths"]]:
        _converged(point, report["tol"])
    with _arrays("solvers") as arrays:
        assert arrays["X"].shape == (178, 13)
        for point in report["fits"]:
            prefix = point["name"]
            assert arrays[prefix + "_centers"].shape == (178, 13)
            assert arrays[prefix + "_dual"].shape == (report["edges"], 13)
            assert len(np.unique(arrays[prefix + "_labels"])) == point["n_clusters"]


def test_missing_gallery_scores_masks_and_selected_refit():
    report = _report("missing")
    assert report["smoke"] is False and report["status"] == "completed"
    assert report["samples"] == 178
    assert len(report["candidates"]) == len(report["gammas"]) == 4
    for point in [*report["candidates"], report["full_refit"]]:
        _converged(point, report["tol"])
        assert point["certificate_model"] == "observed_range_box"
    with _arrays("missing") as arrays:
        train, validation = arrays["training_mask"], arrays["validation_mask"]
        assert train.dtype.kind == validation.dtype.kind == "b"
        assert not np.any(train & validation)
        assert np.all(train | validation)
        assert int(validation.sum()) == report["validation_count"]
        scores = []
        for i, point in enumerate(report["candidates"]):
            centers = arrays[f"candidate_{i}_centers"]
            score = float(np.mean((centers[validation] - arrays["raw_target"][validation]) ** 2))
            assert_allclose(point["mse"], score, rtol=1e-12)
            scores.append(score)
        assert report["selected_index"] == int(np.argmin(scores))
        assert report["selected_gamma"] == report["gammas"][report["selected_index"]]
        assert arrays["full_centers"].shape == arrays["raw_target"].shape == (178, 1)


def test_structured_gallery_sparse_and_biclustering_coordinate_contracts():
    report = _report("structured")
    assert report["smoke"] is False
    assert (report["samples"], report["features"]) == (178, 13)
    assert len(report["sparse_path"]) == 6
    for point in [*report["sparse_path"], report["biclustering"]]:
        _converged(point, report["tol"])
    with _arrays("structured") as arrays:
        assert arrays["original"].shape == (178, 13)
        for point in report["sparse_path"]:
            index = point["index"]
            centered = arrays[f"sparse_{index}_centered"]
            restored = centered + arrays[f"sparse_{index}_offset"]
            assert_allclose(restored, arrays[f"sparse_{index}_standardized_fitted"])
            assert_allclose(
                arrays[f"sparse_{index}_original_units_fitted"],
                restored * arrays["feature_scales"] + arrays["feature_means"],
            )
            assert_allclose(np.linalg.norm(centered, axis=0), arrays["sparse_feature_norms"][index])
        assert_array_equal(np.sort(arrays["row_order"]), np.arange(178))
        assert_array_equal(np.sort(arrays["column_order"]), np.arange(13))
        assert (
            len(np.unique(arrays["biclustering_row_labels"]))
            == report["biclustering"]["row_clusters"]
        )
        assert (
            len(np.unique(arrays["biclustering_column_labels"]))
            == report["biclustering"]["column_clusters"]
        )


def test_generalized_gallery_covers_losses_and_correct_mean_links():
    report = _report("generalized")
    assert report["smoke"] is False and report["status"] == "completed"
    assert set(report["cases"]) == {"huber", "logistic", "poisson"}
    expected_shapes = {"huber": (178, 13), "logistic": (232, 16), "poisson": (90, 1)}
    with _arrays("generalized") as arrays:
        for loss, case in report["cases"].items():
            assert case["loss"] == loss
            _converged(case["diagnostics"], report["tol"])
            assert (case["samples"], case["features"]) == expected_shapes[loss]
            for name in case["arrays"].values():
                assert name in arrays
            centers = arrays[case["arrays"]["centers"]]
            means = arrays[case["arrays"]["fitted_means"]]
            assert centers.shape == means.shape == expected_shapes[loss]
            expected_means = (
                expit(centers)
                if loss == "logistic"
                else np.exp(centers)
                if loss == "poisson"
                else centers
            )
            assert_allclose(means, expected_means, rtol=1e-12, atol=1e-12)
        _converged(report["cases"]["huber"]["squared_comparison"], report["tol"])
