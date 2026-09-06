"""Frozen sampling and truth separation in the experimental workflow."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def study(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("stability_study")


def test_sampling_is_reproducible_with_distinct_audit_bank(study):
    tuning = study.sample_pairs(120, 108101, 5)
    repeated = study.sample_pairs(120, 108101, 5)
    audit = study.sample_pairs(120, 208101, 5)
    for pair, again, other in zip(tuning, repeated, audit):
        for ids, same, different in zip(pair, again, other):
            np.testing.assert_array_equal(ids, same)
            assert len(ids) == 96 and len(np.unique(ids)) == 96
            assert np.all(np.diff(ids) > 0)
            assert not np.array_equal(ids, different)
        assert len(np.intersect1d(*pair)) >= 72


def test_generator_preserves_balanced_mixture_and_gaussian_control(study):
    for scenario in study.SCENARIOS:
        x, truth = study.generate_data(scenario, 8101)
        assert x.shape == (120, 2) and np.isfinite(x).all()
        np.testing.assert_array_equal(x, study.generate_data(scenario, 8101)[0])
        assert np.unique(truth, return_counts=True)[1].tolist() == (
            [120] if scenario == "single_gaussian" else [40, 40, 40]
        )


@pytest.mark.parametrize("unconverged", [False, True])
def test_choice_frozen_before_audit_and_truth_cannot_change_it(
    study, monkeypatch, tmp_path, unconverged
):
    original_generator = study.generate_data
    choices = []
    for altered_truth in [False, True]:
        folder = tmp_path / str(altered_truth)
        folder.mkdir()
        phases = []

        def generate(*args):
            x, truth = original_generator(*args)
            return x, np.zeros_like(truth) if altered_truth else truth

        def fit(x, ids, factors, name, report, arrays, save, neighbors):
            if name.startswith("audit") or name == "full":
                decision = json.loads((folder / "selection.json").read_text())
                assert decision["selected_factor"] == (None if unconverged else 1.0)
            phases.append(name)
            labels = [
                np.arange(len(ids)) if f == 0 else ids % 3 if f == 1 else np.zeros(len(ids), int)
                for f in factors
            ]
            record = dict(
                name=name,
                status="failed" if unconverged and name == "tuning_0_0" else "completed",
                points=[dict(factor=f) for f in factors],
            )
            report["samples_fitted"].append(record)
            return record, labels

        monkeypatch.setattr(study, "generate_data", generate)
        monkeypatch.setattr(study, "_fit_sample", fit)
        code = study.run_worker(
            SimpleNamespace(smoke=True, scenario="separated", seed=8101, output_dir=folder)
        )
        assert code == int(unconverged)
        choices.append(json.loads((folder / "selection.json").read_text()))
        report = json.loads((folder / "report.json").read_text())
        assert report["selection_audit_complete"] is not unconverged
        assert any(p.startswith("audit") for p in phases) is not unconverged
    assert choices[0] == choices[1]


@pytest.fixture(scope="module")
def smoke_artifacts(tmp_path_factory):
    import subprocess
    import sys

    folder = tmp_path_factory.mktemp("stability") / "study"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "examples/stability_study.py"),
            "--smoke",
            "--output-dir",
            str(folder),
        ],
        cwd=root,
        check=True,
        timeout=90,
        capture_output=True,
    )
    return folder


def test_smoke_saved_arrays_pass_independent_audit(study, smoke_artifacts):
    audit = importlib.import_module("audit_stability")
    result = audit.audit_study(smoke_artifacts)
    assert result["status"] == "passed" and result["checks"] > 250


@pytest.mark.parametrize("tamper", ["centers", "decision", "comparison", "incomplete"])
def test_audit_rejects_corruption_even_after_hash_refresh(study, smoke_artifacts, tmp_path, tamper):
    import shutil

    audit = importlib.import_module("audit_stability")
    folder = tmp_path / "changed"
    shutil.copytree(smoke_artifacts, folder)
    manifest_path = folder / "study.json"
    manifest = json.loads(manifest_path.read_text())
    process = manifest["processes"][0]
    report_path, arrays_path = folder / process["checkpoint"], folder / process["arrays"]
    report = json.loads(report_path.read_text())
    if tamper == "centers":
        with np.load(arrays_path) as saved:
            values = {key: saved[key].copy() for key in saved.files}
        key = report["samples_fitted"][0]["points"][0]["arrays"]["centers"]
        values[key][0, 0] += 0.1
        np.savez_compressed(arrays_path, **values)
    elif tamper == "decision":
        decision_path = folder / process["selection"]
        decision = json.loads(decision_path.read_text())
        decision["selected_factor"] = 1.0 if decision["selected_factor"] != 1.0 else 30.0
        decision_path.write_text(json.dumps(decision))
        report["selection"] = decision
        report["selection_sha256"] = audit.digest(decision_path)
    elif tamper == "comparison":
        report["tuning"][0]["comparisons"][0]["raw_disagreement"] = 0.5
    else:
        manifest["status"] = "running"
    report_path.write_text(json.dumps(report))
    for field in ["checkpoint", "arrays", "selection", "log"]:
        process[field + "_sha256"] = audit.digest(folder / process[field])
    manifest_path.write_text(json.dumps(manifest))
    result = audit.audit_study(folder)
    assert result["status"] == "failed" and result["failures"]
