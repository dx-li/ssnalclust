"""Scientific benchmark evidence, independently of timing comparisons."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse


@pytest.fixture
def harness():
    spec = importlib.util.spec_from_file_location(
        "scalability_evidence", Path(__file__).resolve().parents[1] / "examples" / "scalability.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_graph_evidence_weighted_disconnected_graph(harness):
    graph = sparse.csr_matrix(
        ([1.0, 1.0, 3.0, 3.0, 2.0, 2.0], ([0, 1, 1, 2, 3, 4], [1, 0, 2, 1, 4, 3])), shape=(6, 6)
    )
    summary = harness._graph_evidence(graph)
    assert summary == dict(
        connected_components=3,
        isolated_vertices=1,
        positive_edge_weight_min=1.0,
        positive_edge_weight_median=2.0,
        positive_edge_weight_max=3.0,
        weighted_degree_min=0.0,
        weighted_degree_median=2.0,
        weighted_degree_max=4.0,
    )


def test_empty_graph_has_explicit_absent_edge_statistics(harness):
    summary = harness._graph_evidence(sparse.csr_matrix((3, 3)))
    assert summary["connected_components"] == summary["isolated_vertices"] == 3
    assert summary["positive_edge_weight_min"] is None
    assert summary["positive_edge_weight_median"] is None
    assert summary["positive_edge_weight_max"] is None
    assert summary["weighted_degree_min"] == summary["weighted_degree_max"] == 0
    json.dumps(summary, allow_nan=False)


def test_direct_edge_fusion_does_not_use_transitive_labels(harness):
    centers = np.array([[0.0], [0.09], [0.18]])
    graph = sparse.csr_matrix(np.ones((3, 3)) - np.eye(3))
    evidence = harness._solution_evidence(
        centers, graph, 2.0, SimpleNamespace(centers=centers, gap=0.25), 0.1
    )
    assert evidence["direct_fused_edge_fraction"] == 2 / 3
    assert evidence["gamma"] == 2
    assert evidence["gap"] == 0.25
    assert evidence["centroid_displacement_frobenius"] == 0
    assert evidence["relative_centroid_displacement"] == 0
    assert_allclose(evidence["zero_dual_initial_gap"], 2 * (0.09 + 0.18 + 0.09))


def test_relative_displacement_and_initial_gap(harness):
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    U = np.array([[0.6, 0.8], [2.4, 3.2]])
    graph = sparse.csr_matrix([[0.0, 2.0], [2.0, 0.0]])
    evidence = harness._solution_evidence(X, graph, 0.5, SimpleNamespace(centers=U, gap=0.0), 1e-4)
    assert_allclose(evidence["centroid_displacement_frobenius"], np.sqrt(2))
    assert_allclose(evidence["centered_input_norm"], np.sqrt(12.5))
    assert_allclose(evidence["relative_centroid_displacement"], 0.4)
    assert evidence["zero_dual_initial_gap"] == 5
    assert evidence["direct_fused_edge_fraction"] == 0


@pytest.mark.parametrize(
    "shift,relative,status",
    [
        (0.0, 0.0, "zero_baseline_no_displacement"),
        (1.0, None, "undefined_zero_baseline"),
    ],
)
def test_zero_baseline_and_no_edges_produce_finite_json(harness, shift, relative, status):
    X = np.ones((3, 2))
    evidence = harness._solution_evidence(
        X, sparse.csr_matrix((3, 3)), 1.0, SimpleNamespace(centers=X + shift, gap=0.0), 1e-4
    )
    assert evidence["relative_centroid_displacement"] == relative
    assert evidence["relative_displacement_normalization"] == status
    assert evidence["direct_fused_edge_fraction"] is None
    assert evidence["zero_dual_initial_gap"] == 0
    json.dumps(evidence, allow_nan=False)


def test_sparse_graph_evidence_never_densifies_adjacency(harness, monkeypatch):
    graph = sparse.diags(
        [np.ones(9999), np.ones(9999)], [-1, 1], shape=(10000, 10000), format="csr"
    )
    X = np.arange(10000, dtype=float)[:, None]

    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark evidence densified adjacency")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    assert harness._graph_evidence(graph)["connected_components"] == 1
    evidence = harness._solution_evidence(X, graph, 1.0, SimpleNamespace(centers=X, gap=0.0), 0.5)
    assert evidence["zero_dual_initial_gap"] == 9999
    assert evidence["direct_fused_edge_fraction"] == 0


def test_completed_path_emits_individual_certificates_and_separate_evidence_time(harness):
    config = dict(
        samples=8,
        features=2,
        seed=1729,
        data="gaussian",
        neighbors=2,
        bandwidth=1.0,
        scenario="path",
        gamma=0.5,
        path_gammas=[0.0, 0.1],
        solver="ssnal",
        tol=1e-6,
        max_iter=300,
        check_every=1,
        cluster_tol=1e-4,
    )
    record = harness._run(config, timeout=30, threads=1)
    assert record["status"] == "completed", record
    assert record["all_converged"]
    completed = [event for event in record["events"] if event["event"] == "solution_complete"]
    assert [event["gamma"] for event in completed] == [0.0, 0.1]
    assert [result["gamma"] for result in record["solutions"]] == [0.0, 0.1]
    assert record["evidence_timing"] == "excluded_from_pipeline_included_in_worker"
    assert_allclose(
        record["evidence_seconds"],
        record["stage_seconds"]["graph_evidence"] + record["stage_seconds"]["solution_evidence"],
    )
    assert_allclose(
        record["graph_plus_pipeline_seconds"],
        record["stage_seconds"]["graph"] + record["pipeline_seconds"],
    )
    assert record["solutions"][0]["zero_dual_initial_gap"] == 0


def test_timeout_preserves_completed_prefix_without_full_path_claim(harness, monkeypatch):
    event = dict(
        event="solution_complete",
        gamma=0.1,
        objective=2.0,
        gap=1e-8,
        relative_gap=1e-9,
        kkt_residual=1e-9,
        converged=True,
        iterations=3,
    )

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output=json.dumps(event) + "\n")

    monkeypatch.setattr(harness.subprocess, "run", timeout)
    record = harness._run({}, timeout=1, threads=1)
    assert record["status"] == "timeout"
    assert record["events"] == [event]
    assert not record["peak_rss_complete"]
    assert "all_converged" not in record
