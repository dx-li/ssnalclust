"""Replay captured unmodified cvxclustr results without requiring R in CI.

The fixture includes failures of the common accuracy target. Every completed
candidate is recertified, while only certified external candidates serve as
centroid regression references for freshly computed Python fits.
"""

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust import solve

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "docs" / "external_reference_results.jsonl"
COMMON_TOLERANCE = 1e-8


def _load_auditor():
    spec = importlib.util.spec_from_file_location(
        "external_reference_regression_auditor", ROOT / "examples" / "external_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.certify_candidate


@pytest.fixture(scope="module")
def capture():
    records = [json.loads(line) for line in SNAPSHOT.read_text().splitlines() if line.strip()]
    metadata = [record for record in records if record["kind"] == "metadata"]
    assert len(metadata) == 1, "This regression fixture must describe one complete reference run"
    problems = {
        record["case"]: (np.asarray(record["X"]), np.asarray(record["weights"]))
        for record in records
        if record["kind"] == "problem"
    }
    return records, metadata[0], problems, _load_auditor()


def _key(record):
    return record["case"], record["penalty"], record["gamma"]


def test_capture_has_provenance_and_complete_grid(capture):
    records, metadata, problems, _ = capture
    assert len(metadata["reference_revision"]) == 40
    assert metadata["reference_source_sha256"]
    assert all(len(value) == 64 for value in metadata["reference_source_sha256"].values())
    assert "src/cvxclustr.c" in metadata["reference_source_sha256"]
    assert metadata["versions"]["ssnalclust"]
    assert set(problems) == {
        "two_point",
        "random_complete",
        "random_disconnected",
        "iris_standardized_knn10",
    }
    expected = set(itertools.product(problems, ("l1", "l2"), metadata["gamma_grid"]))
    local = [record for record in records if record["kind"] == "local"]
    references = [record for record in records if record["kind"] == "reference"]
    assert {_key(record) for record in local} == expected
    assert len(local) == len(expected)
    expected_references = {
        (case, penalty, gamma, solver)
        for case, penalty, gamma in expected
        for solver in ("cvxclustr_ama", "cvxclustr_fama")
    }
    assert {(*_key(record), record["solver"]) for record in references} == expected_references
    assert len(references) == len(expected_references)
    for record in local + references:
        X, weights = problems[record["case"]]
        assert record["data_sha256"] == hashlib.sha256(X.astype("<f8").tobytes()).hexdigest()
        assert record["graph_sha256"] == hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()
        assert (record["n"], record["p"]) == X.shape
        if record["kind"] == "reference":
            assert record["status"] in {"completed", "timeout", "error"}
            if record["status"] == "timeout":
                assert record["timeout"] > 0
            elif record["status"] == "error":
                assert record["error"]


def test_recompute_all_saved_candidate_certificates(capture):
    records, _, problems, certify = capture
    count = 0
    for record in records:
        if record["kind"] != "local" and not (
            record["kind"] == "reference" and record["status"] == "completed"
        ):
            continue
        X, weights = problems[record["case"]]
        recalculated = certify(
            X, weights, record["gamma"], record["penalty"], record["centers"], record["dual"]
        )
        for field, value in recalculated.items():
            assert np.isfinite(value), (record["case"], record["solver"], field)
            if field == "center_error_bound":
                # A square root amplifies relative changes in a gap at the
                # floating-point floor. Compare the bound in objective units
                # across platforms, and verify its formula on each platform.
                assert value >= 0
                assert_allclose(
                    value, np.sqrt(2 * recalculated["gap"]), rtol=8 * np.finfo(float).eps, atol=0
                )
                assert_allclose(
                    0.5 * value**2,
                    record["certificate"]["gap"],
                    rtol=5e-10,
                    atol=2e-12,
                    err_msg=f"{_key(record)} {record['solver']}: squared bound",
                )
                assert_allclose(
                    record["certificate"][field],
                    np.sqrt(2 * record["certificate"]["gap"]),
                    rtol=8 * np.finfo(float).eps,
                    atol=0,
                )
                continue
            assert_allclose(
                value,
                record["certificate"][field],
                rtol=5e-10,
                atol=2e-12,
                err_msg=f"{_key(record)} {record['solver']}: {field}",
            )
        if record["kind"] == "reference":
            accurate = (
                max(recalculated["relative_gap"], recalculated["kkt_residual"]) <= COMMON_TOLERANCE
            )
            assert accurate == record["common_accuracy_pass"]
            assert record["metadata"]["version"]
            assert record["metadata"]["R"]
        count += 1
    assert count >= 32


def test_failed_reference_accuracy_is_preserved_not_promoted(capture):
    records, _, problems, certify = capture
    completed = [
        record
        for record in records
        if record["kind"] == "reference" and record["status"] == "completed"
    ]
    inaccurate = [record for record in completed if not record["common_accuracy_pass"]]
    assert inaccurate, "Retain the recorded unsuccessful common-accuracy comparisons"
    for record in inaccurate:
        X, weights = problems[record["case"]]
        checked = certify(
            X, weights, record["gamma"], record["penalty"], record["centers"], record["dual"]
        )
        assert max(checked["relative_gap"], checked["kkt_residual"]) > COMMON_TOLERANCE
    # Native stopping fields are retained separately from common certificates;
    # reaching the native iteration cap does not invalidate a good common fit.
    assert any(
        record["common_accuracy_pass"] and record["metadata"]["iterations"] == 100000
        for record in completed
    )


def test_fresh_small_fits_agree_with_certified_external_center_bounds(capture):
    records, _, problems, certify = capture
    selected = [
        record
        for record in records
        if record["kind"] == "reference"
        and record["status"] == "completed"
        and record["common_accuracy_pass"]
        and record["n"] <= 12
        and record["gamma"] in (0.1, 1.0)
    ]
    assert {record["penalty"] for record in selected} == {"l1", "l2"}
    assert {record["case"] for record in selected} == {
        "two_point",
        "random_complete",
        "random_disconnected",
    }
    assert {record["solver"] for record in selected} == {"cvxclustr_ama", "cvxclustr_fama"}
    local_cache = {}
    for record in selected:
        X, weights = problems[record["case"]]
        key = _key(record)
        if key not in local_cache:
            # Use an independently implemented local algorithm for each norm,
            # not the saved local arrays or native reference objective fields.
            fit = solve(
                X,
                weights,
                gamma=record["gamma"],
                penalty=record["penalty"],
                solver="ssnal" if record["penalty"] == "l2" else "admm",
                tol=1e-9,
                max_iter=50000,
                store_history=False,
            )
            assert fit.converged, (key, fit.message)
            certificate = certify(
                X, weights, record["gamma"], record["penalty"], fit.centers, fit.dual
            )
            assert max(certificate["relative_gap"], certificate["kkt_residual"]) <= 1e-9
            local_cache[key] = fit.centers, certificate
        centers, certificate = local_cache[key]
        external_centers = np.asarray(record["centers"])
        external = certify(
            X, weights, record["gamma"], record["penalty"], external_centers, record["dual"]
        )
        distance = np.linalg.norm(centers - external_centers)
        combined_bound = certificate["center_error_bound"] + external["center_error_bound"]
        # Account only for evaluating norms and gaps in finite precision.
        rounding = (
            128
            * np.finfo(float).eps
            * (1 + np.linalg.norm(X) + np.linalg.norm(centers) + np.linalg.norm(external_centers))
        )
        assert distance <= combined_bound + rounding, (key, distance, combined_bound)
        # Each feasible lower bound must bracket the common optimum below
        # either candidate's primal objective, independently of center distance.
        assert certificate["dual_objective"] <= external["objective"] + rounding
        assert external["dual_objective"] <= certificate["objective"] + rounding


@pytest.mark.parametrize("solver", ["cvxclustr_ama", "cvxclustr_fama"])
def test_captured_weighted_l1_stopping_failure_is_not_a_solution_reference(solver):
    """A negative native gap must not override a positive true objective gap."""
    records = [
        json.loads(line)
        for line in (ROOT / "docs" / "external_reference_stress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    problem = next(
        record
        for record in records
        if record["kind"] == "problem" and record["case"] == "weighted_stop_stress"
    )
    reference = next(
        record
        for record in records
        if record["kind"] == "reference"
        and record["solver"] == solver
        and record["penalty"] == "l1"
        and record["gamma"] == 0.1
    )
    assert reference["status"] == "completed"
    assert not reference["common_accuracy_pass"]
    native_gap = reference["metadata"]["native_primal"] - reference["metadata"]["native_dual"]
    assert native_gap < 0
    X, weights = np.asarray(problem["X"]), np.asarray(problem["weights"])
    assert reference["data_sha256"] == hashlib.sha256(X.astype("<f8").tobytes()).hexdigest()
    assert reference["graph_sha256"] == hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()
    certify = _load_auditor()
    external_centers = np.asarray(reference["centers"])
    external = certify(X, weights, 0.1, "l1", external_centers, reference["dual"])
    assert external["gap"] > 0.01
    assert external["kkt_residual"] > 0.01
    assert_allclose(external["gap"], reference["certificate"]["gap"], atol=1e-12)

    fit = solve(
        X,
        weights,
        gamma=0.1,
        penalty="l1",
        solver="admm",
        tol=1e-10,
        max_iter=50000,
        store_history=False,
    )
    assert fit.converged, fit.message
    local = certify(X, weights, 0.1, "l1", fit.centers, fit.dual)
    assert max(local["relative_gap"], local["kkt_residual"]) <= 1e-10
    distance = np.linalg.norm(fit.centers - external_centers)
    # The candidates materially disagree; the bad reference has a correspondingly
    # broad valid error bound and must never be promoted to a precise oracle.
    assert distance > max(0.1, 100 * local["center_error_bound"])
    rounding = 128 * np.finfo(float).eps * (1 + np.linalg.norm(X))
    assert distance <= local["center_error_bound"] + external["center_error_bound"] + rounding
    assert local["objective"] < external["objective"] - 0.01
