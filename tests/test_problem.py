"""Fixed-problem reuse, streaming independence, and cache correctness."""

import json
import weakref

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from ssnalclust import convex_clustering_path, solve
from ssnalclust.estimator import iter_convex_clustering_path
from ssnalclust.problem import ConvexClusteringProblem


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_prepared_matches_single_solves(solver):
    x = np.array([[0.0, 1.0], [0.2, 1.2], [2.0, 3.0]])
    masses = [1.0, 2.0, 3.0]
    problem = ConvexClusteringProblem(x, sample_weight=masses)
    assert (problem.n_samples, problem.n_features, problem.n_edges) == (3, 2, 3)
    for gamma, result in zip(
        [0.1, 2.0, 0.05], problem.path([0.1, 2.0, 0.05], solver=solver, max_iter=5000, tol=1e-8)
    ):
        direct = solve(x, gamma=gamma, solver=solver, sample_weight=masses, max_iter=5000, tol=1e-8)
        assert result.converged and direct.converged
        assert_allclose(result.centers, direct.centers, atol=2e-6)
        assert_allclose(result.objective, direct.objective, atol=1e-7)


def test_path_builds_graph_once_and_reuses_factorization(monkeypatch):
    import ssnalclust.problem as module

    graph_calls, factor_calls = [], []
    original_graph, original_factor = module.graph_from_weights, module.factorized

    def graph(*args, **kwargs):
        graph_calls.append(1)
        return original_graph(*args, **kwargs)

    def factor(*args, **kwargs):
        factor_calls.append(1)
        return original_factor(*args, **kwargs)

    monkeypatch.setattr(module, "graph_from_weights", graph)
    monkeypatch.setattr(module, "factorized", factor)
    results = convex_clustering_path([[0.0], [2.0], [4.0]], [0.1, 0.5, 2.0], solver="admm")
    assert all(r.converged for r in results)
    assert len(graph_calls) == len(factor_calls) == 1


def test_cache_sigma_changes_and_norm_does_not(monkeypatch):
    import ssnalclust.problem as module

    factor = module.factorized
    calls = []

    def record(matrix):
        calls.append(matrix.copy())
        return factor(matrix)

    monkeypatch.setattr(module, "factorized", record)
    problem = ConvexClusteringProblem([[0.0, 1.0], [3.0, 5.0]])
    for sigma, penalty in [(1.0, "l2"), (1.0, "l1"), (2.0, "linf"), (1.0, "l2")]:
        result = problem.solve(solver="admm", sigma=sigma, penalty=penalty)
        direct = solve([[0.0, 1.0], [3.0, 5.0]], solver="admm", sigma=sigma, penalty=penalty)
        assert result.converged and direct.converged
        assert_allclose(result.centers, direct.centers)
    # Three prepared factorizations (one-entry bounded cache) + four direct ones.
    assert len(calls) == 7


def test_inputs_are_snapshots():
    x = np.array([[0.0], [2.0]])
    w = np.array([[0.0, 1.0], [1.0, 0.0]])
    m = np.ones(2)
    problem = ConvexClusteringProblem(x, w, m)
    x[:] = 100
    w[:] = 0
    m[:] = 10
    result = problem.solve(gamma=0.5)
    assert_allclose(result.centers, [[0.5], [1.5]], atol=1e-6)
    result.centers[:] = -200
    assert_allclose(problem.solve(gamma=0.5).centers, [[0.5], [1.5]], atol=1e-6)


def test_stream_is_lazy_and_result_mutation_cannot_change_next_start():
    strengths_consumed = []

    def strengths():
        for gamma in [0.1, 1.0, 0.2]:
            strengths_consumed.append(gamma)
            yield gamma

    stream = iter_convex_clustering_path([[0.0], [2.0]], strengths(), store_history=False)
    assert not strengths_consumed
    first = next(stream)
    assert strengths_consumed == [0.1]
    first.centers[:] = np.nan
    first.dual[:] = np.nan
    array_ref = weakref.ref(first.centers)
    del first
    second = next(stream)
    assert second.converged and second.history == []
    assert array_ref() is None
    assert_allclose(second.centers, [[1.0], [1.0]], atol=1e-5)
    assert len(list(stream)) == 1


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_no_history_does_not_change_final_diagnostics(solver):
    x = [[0.0, 1.0], [2.0, 4.0]]
    stored = solve(x, solver=solver, gamma=0.2)
    streamed = solve(x, solver=solver, gamma=0.2, store_history=False)
    assert streamed.history == []
    assert_array_equal(stored.centers, streamed.centers)
    assert stored.n_iter == streamed.n_iter > 0
    assert stored.objective == streamed.objective
    assert stored.gap == streamed.gap
    assert stored.kkt_residual == streamed.kkt_residual
    assert stored.converged == streamed.converged
    json.dumps(streamed.n_iter)


def test_stream_rejects_bad_strength_when_consumed():
    stream = ConvexClusteringProblem([[0.0], [2.0]]).iter_path([0, -1])
    assert next(stream).converged
    with pytest.raises(ValueError, match="gammas"):
        next(stream)
    with pytest.raises(ValueError, match="store_history"):
        solve([[0.0], [1.0]], store_history="no")
