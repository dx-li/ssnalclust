"""Independent memory-evidence audit without running clustering fits."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def auditor():
    path = Path(__file__).resolve().parents[1] / "examples" / "plot_streaming_memory.py"
    spec = importlib.util.spec_from_file_location("streaming_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pair():
    # One tiny logical result, with the same arrays/certificate in both modes.
    prepared = dict(
        data_sha256="a" * 64,
        graph_sha256="b" * 64,
        gammas=[0.1],
        source_sha256={"solver.py": "c" * 64},
        harness_sha256="d" * 64,
        samples=2,
        features=1,
        seed=7301,
        neighbors=1,
        solver="ssnal",
        tol=1e-6,
        max_iter=300,
        store_history=False,
        requested_thread_environment={"OMP_NUM_THREADS": "1"},
        versions={"numpy": "fixture"},
        platform="fixture",
    )
    point = dict(
        event="point",
        index=0,
        gamma=0.1,
        converged=True,
        n_iter=4,
        objective=0.09,
        dual_objective=0.09,
        gap=0.0,
        relative_gap=0.0,
        kkt_residual=0.0,
        center_error_bound=0.0,
        centers_sha256="e" * 64,
        dual_sha256="f" * 64,
        displacement_frobenius=0.1,
        relative_displacement=0.2,
        direct_fused_edge_fraction=0.0,
        center_array_bytes=16,
        dual_array_bytes=8,
    )
    return [
        (
            dict(
                mode=mode,
                length=1,
                process_status="completed",
                checkpoint_events=[copy.deepcopy(point)],
            ),
            copy.deepcopy(prepared),
            dict(
                returned_points=1,
                all_converged=True,
                cumulative_generated_center_dual_bytes=24,
                consumer_retained_center_dual_bytes=24 if mode == "materialized" else 0,
            ),
        )
        for mode in ("stream", "materialized")
    ]


def test_accepts_complete_exact_pair(auditor, pair):
    assert auditor.validate_pairs(pair)["passed"]


@pytest.mark.parametrize("mutation", ["hash", "point", "timeout", "retention", "convergence"])
def test_rejects_false_parity_claims(auditor, pair, mutation):
    process, _, finished = pair[0]
    if mutation == "hash":
        process["checkpoint_events"][0]["centers_sha256"] = "0" * 64
    elif mutation == "point":
        process["checkpoint_events"] = []
    elif mutation == "timeout":
        process["process_status"] = "timeout"
    elif mutation == "retention":
        finished["consumer_retained_center_dual_bytes"] = 24
    else:
        finished["all_converged"] = False
    assert not auditor.validate_pairs(pair)["passed"]


def test_identical_unconverged_results_are_parity_not_success(auditor, pair):
    for process, _, finished in pair:
        process["checkpoint_events"][0]["converged"] = False
        finished["all_converged"] = False
    assert auditor.validate_pairs(pair)["passed"]


def test_different_source_invalidates_controlled_pair(auditor, pair):
    pair[0][1]["source_sha256"] = {"solver.py": "different"}
    assert not auditor.validate_pairs(pair)["passed"]


def test_different_data_at_second_length_invalidates_scaling(auditor, pair):
    second = copy.deepcopy(pair)
    for process, prepared, finished in second:
        process["length"] = 2
        prepared["gammas"] = [0.1, 0.2]
        prepared["data_sha256"] = "0" * 64
        point = copy.deepcopy(process["checkpoint_events"][0])
        point.update(index=1, gamma=0.2)
        process["checkpoint_events"].append(point)
        finished["returned_points"] = 2
        finished["cumulative_generated_center_dual_bytes"] = 48
        if process["mode"] == "materialized":
            finished["consumer_retained_center_dual_bytes"] = 48
    assert auditor.validate_pairs(second)["passed"]
    audit = auditor.validate_pairs(pair + second)
    assert not audit["passed"]
    assert any("scaling configuration differs: data_sha256" in error for error in audit["errors"])


@pytest.mark.parametrize("original_exists", [False, True])
def test_relocated_bundle_takes_priority_over_original(auditor, tmp_path, original_exists):
    report = tmp_path / "relocated" / "results.jsonl"
    bundled = report.parent / (report.name + ".workers")
    bundled.mkdir(parents=True)
    original = tmp_path / "old-checkout" / "stream-1.jsonl"
    if original_exists:
        original.parent.mkdir()
        original.write_text("Wrong other-checkout contents", encoding="utf-8")
    raw = b'{"event":"prepared","samples":2}\n'
    (bundled / original.name).write_bytes(raw)
    record = dict(
        event="process_finished",
        mode="stream",
        length=1,
        checkpoint=str(original),
        checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
    )
    report.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert auditor.read_results(report)[0][1]["samples"] == 2


def test_trailing_started_process_cannot_be_ignored(auditor, tmp_path):
    report = tmp_path / "results.jsonl"
    worker = tmp_path / "stream-1.jsonl"
    worker.write_bytes(b"{}\n")
    completed = dict(
        event="process_finished",
        mode="stream",
        length=1,
        checkpoint=str(worker),
        checkpoint_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
    )
    started = dict(event="process_started", mode="materialized", length=1)
    report.write_text(json.dumps(completed) + "\n" + json.dumps(started) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no finished record"):
        auditor.read_results(report)
