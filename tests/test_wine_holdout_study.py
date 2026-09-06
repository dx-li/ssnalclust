"""Selection gates and data separation for the fixed Wine experiment."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def study():
    path = Path(__file__).resolve().parents[1] / "examples/wine_holdout_study.py"
    spec = importlib.util.spec_from_file_location("wine_study_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def reports():
    return [
        dict(
            seed=seed,
            status="completed",
            validation_count=count,
            baseline_mse=2.0,
            points=[
                dict(index=0, factor=0.1, converged=True, score_mse=left),
                dict(index=1, factor=1.0, converged=True, score_mse=right),
            ],
        )
        for seed, count, left, right in [(1, 2, 0.1, 0.3), (2, 6, 0.9, 0.3)]
    ]


def test_pooled_score_weights_entries_and_does_not_mutate_inputs(study, reports):
    before = copy.deepcopy(reports)
    chosen = study.select_factor(reports, [0.1, 1.0], [1, 2])
    assert reports == before
    assert chosen["status"] == "valid"
    np.testing.assert_allclose(chosen["aggregate_mse"], [0.7, 0.3])
    assert chosen["selected_factor"] == 1.0
    assert chosen["scored_entries"] == 8


@pytest.mark.parametrize(
    "failure", ["timeout", "missing", "duplicate", "wrong_seed", "unconverged", "nan",
                "infinity", "negative", "wrong_factor", "wrong_index", "no_validation"]
)
def test_no_selection_from_incomplete_or_malformed_evidence(study, reports, failure):
    if failure == "timeout":
        reports[0]["status"] = "timeout"
    elif failure == "missing":
        reports.pop()
    elif failure == "duplicate":
        reports[1]["seed"] = reports[0]["seed"]
    elif failure == "wrong_seed":
        reports[1]["seed"] = 3
    elif failure == "no_validation":
        reports[0]["validation_count"] = 0
    else:
        point = reports[0]["points"][0]
        if failure == "unconverged":
            point["converged"] = False
        elif failure == "wrong_factor":
            point["factor"] = 0.2
        elif failure == "wrong_index":
            point["index"] = 1
        else:
            point["score_mse"] = {"nan": np.nan, "infinity": np.inf, "negative": -0.1}[failure]
    chosen = study.select_factor(reports, [0.1, 1.0], [1, 2])
    assert chosen["status"] == "incomplete"
    assert chosen["selected_factor"] is None


def test_exact_tie_keeps_prespecified_first_candidate(study, reports):
    for report in reports:
        for point in report["points"]:
            point["score_mse"] = 0.5
    assert study.select_factor(reports, [0.1, 1.0], [1, 2])["selected_factor"] == 0.1


def test_masks_are_value_independent_and_have_fixed_column_counts(study):
    first = study.make_training_mask((178, 13), 1729)
    second = study.make_training_mask((178, 13), 1729)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal((~first).sum(axis=0), np.full(13, 17))
    assert not np.array_equal(first, study.make_training_mask((178, 13), 1730))


def test_feature_loader_cannot_include_cultivar_column(study, tmp_path):
    data = np.arange(42.0).reshape(3, 14)
    original = tmp_path / "original.data"
    changed = tmp_path / "changed.data"
    np.savetxt(original, data, delimiter=",")
    data[:, 0] = [np.nan, np.inf, -999]
    np.savetxt(changed, data, delimiter=",")
    np.testing.assert_array_equal(study.load_features(original), study.load_features(changed))
    np.testing.assert_array_equal(study.load_features(changed), data[:, 1:])
