"""Lazy path diagnostics preserve transitions without retaining past results."""

import gc
import weakref

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ssnalclust import convex_clustering_path, iter_path_summaries, summarize_path


class Result:
    """Minimal weak-referenceable result supporting the original summary API."""

    def __init__(self, centers, converged=True):
        self.centers = np.asarray(centers, dtype=float)
        self.objective = 1.0
        self.converged = converged
        self.n_iter = 2
        self.kkt_residual = 0.0 if converged else 0.1


def test_real_solver_path_matches_materialized_summaries():
    path = convex_clustering_path([[0.0], [2.0]], [0, 0.5, 1, 2], tol=1e-8)
    streamed = list(iter_path_summaries(iter(path)))
    materialized = summarize_path(path)
    assert len(streamed) == len(materialized) == 4
    for actual, expected, result in zip(streamed, materialized, path):
        assert actual.keys() == expected.keys()
        assert_array_equal(actual["labels"], expected["labels"])
        assert {key: value for key, value in actual.items() if key != "labels"} == {
            key: value for key, value in expected.items() if key != "labels"
        }
        for key in ["dual_objective", "gap", "relative_gap", "center_error_bound"]:
            assert actual[key] == getattr(result, key)
    assert [point["n_clusters"] for point in streamed] == [2, 2, 1, 1]


def test_optional_certificates_are_none_for_legacy_result():
    summary = next(iter_path_summaries([Result([[0], [1]])]))
    required = {
        "labels",
        "n_clusters",
        "objective",
        "converged",
        "n_iter",
        "kkt_residual",
        "merges",
        "splits",
    }
    assert required <= summary.keys()
    for key in ["dual_objective", "gap", "relative_gap", "center_error_bound"]:
        assert summary[key] is None


@pytest.mark.parametrize("bad_centers", [[[np.nan], [1.0]], [0.0, 1.0], [[0], [1], [2]]])
def test_laziness_and_invalid_later_point_errors_only_when_reached(bad_centers):
    reached = []

    def source():
        reached.append(0)
        yield Result([[0], [1]])
        reached.append(1)
        yield Result(bad_centers)
        reached.append(2)
        yield Result([[0], [0]])

    upstream = source()
    summaries = iter_path_summaries(upstream)
    assert reached == []
    first = next(summaries)
    assert first["n_clusters"] == 2 and reached == [0]
    with pytest.raises(ValueError):
        next(summaries)
    assert reached == [0, 1]
    # Even an error in the wrapper does not close its borrowed source.
    assert next(upstream).centers.shape == (2, 1)
    assert reached == [0, 1, 2]
    upstream.close()


def test_simultaneous_merge_split_and_unconverged_point_remain_visible():
    source = [Result([[0], [0], [1]]), Result([[0], [1], [1]], converged=False)]
    summaries = list(iter_path_summaries(source))
    assert summaries[0]["merges"] == summaries[0]["splits"] == []
    assert summaries[1]["merges"] == [(1, [0, 1])]
    assert summaries[1]["splits"] == [(0, [0, 1])]
    assert summaries[1]["converged"] is False
    assert summaries[1]["kkt_residual"] == 0.1


def test_consumer_label_mutation_cannot_change_future_transitions():
    summaries = iter_path_summaries(
        [Result([[0], [0], [1]]), Result([[0], [1], [1]]), Result([[0], [0], [0]])]
    )
    first = next(summaries)
    first["labels"][:] = 987
    second = next(summaries)
    assert second["merges"] == [(1, [0, 1])]
    assert second["splits"] == [(0, [0, 1])]
    second["labels"][:] = -10
    third = next(summaries)
    assert third["merges"] == [(0, [0, 1])]
    assert third["splits"] == []


def test_prior_input_objects_are_released_before_upstream_advances():
    class Source:
        def __init__(self):
            self.references = []
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            gc.collect()
            assert all(reference() is None for reference in self.references)
            if self.count == 3:
                raise StopIteration
            result = Result([[0], [self.count + 1]])
            self.references = [weakref.ref(result), weakref.ref(result.centers)]
            self.count += 1
            return result

    source = Source()
    iterator = iter_path_summaries(source)
    summaries = []
    for _ in range(3):
        summaries.append(next(iterator))
        gc.collect()
        # Inputs must already be gone while the wrapper is suspended at yield.
        assert all(reference() is None for reference in source.references)
    with pytest.raises(StopIteration):
        next(iterator)
    assert len(summaries) == 3
    assert all(reference() is None for reference in source.references)


def test_discarded_summary_labels_are_not_held_by_wrapper():
    class Source:
        previous_labels = None
        count = 0

        def __iter__(self):
            return self

        def __next__(self):
            gc.collect()
            if self.previous_labels is not None:
                assert self.previous_labels() is None
            if self.count == 3:
                raise StopIteration
            self.count += 1
            return Result([[0], [1]])

    source = Source()
    summaries = iter_path_summaries(source)
    for _ in range(3):
        summary = next(summaries)
        source.previous_labels = weakref.ref(summary["labels"])
        del summary
    with pytest.raises(StopIteration):
        next(summaries)


def test_closing_wrapper_does_not_close_borrowed_upstream():
    closed = []

    def source():
        try:
            for value in [1, 2, 3]:
                yield Result([[0], [value]])
        finally:
            closed.append(True)

    upstream = source()
    summaries = iter_path_summaries(upstream)
    next(summaries)
    summaries.close()
    assert not closed
    assert_array_equal(next(upstream).centers, [[0], [2]])
    upstream.close()
    assert closed == [True]


def test_empty_stream():
    assert list(iter_path_summaries(iter([]))) == []


@pytest.mark.parametrize("cluster_tol", [-1, np.nan, np.inf, True, "invalid"])
def test_static_tolerance_errors_are_eager_without_consuming_source(cluster_tol):
    reached = []

    def source():
        reached.append(True)
        yield Result([[0], [1]])

    with pytest.raises(ValueError, match="cluster_tol"):
        iter_path_summaries(source(), cluster_tol=cluster_tol)
    assert reached == []
