from types import SimpleNamespace

import numpy as np
import pytest

from ssnalclust import convex_clustering_path
from ssnalclust.path import summarize_path


def test_exact_small_path():
    path = convex_clustering_path([[0.0], [2.0]], [0, 0.5, 1, 2], tol=1e-8)
    summaries = summarize_path(path)
    assert [s["n_clusters"] for s in summaries] == [2, 2, 1, 1]
    assert summaries[2]["merges"] == [(0, [0, 1])]
    assert all(s["converged"] for s in summaries)
    assert all(not s["splits"] for s in summaries)


def test_splits_and_failure_are_preserved():
    results = [
        SimpleNamespace(
            centers=np.array(x), objective=0, converged=converged, n_iter=1, kkt_residual=0
        )
        for x, converged in [([[0.0], [0.0], [1.0]], True), ([[0.0], [1.0], [1.0]], False)]
    ]
    summaries = summarize_path(results)
    assert summaries[1]["splits"] == [(0, [0, 1])]
    assert summaries[1]["merges"] == [(1, [0, 1])]
    assert not summaries[1]["converged"]


def test_shape_validation_and_empty_path():
    assert summarize_path([]) == []
    results = convex_clustering_path([[0.0], [1.0]], [0, 1])
    results[1].centers = np.ones((3, 1))
    with pytest.raises(ValueError, match="same samples"):
        summarize_path(results)
