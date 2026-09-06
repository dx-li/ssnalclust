"""Independent pair-indicator oracle for experimental stability scoring."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def stability(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    return importlib.import_module("partition_stability")


def partitions(n):
    def extend(prefix):
        if len(prefix) == n:
            yield np.array(prefix)
        else:
            for label in range(max(prefix) + 2):
                yield from extend(prefix + [label])

    yield from extend([0])


def test_exhaustive_five_observation_pair_oracle(stability):
    rows, cols = np.triu_indices(5, 1)
    all_partitions = list(partitions(5))
    assert len(all_partitions) == 52
    for a in all_partitions:
        for b in all_partitions:
            left, right = a[rows] == a[cols], b[rows] == b[cols]
            result = stability.compare_partitions(a, b, min_pairs=1)
            assert result["raw_disagreement"] == np.mean(left != right)
            assert result["same_a"] == left.sum()
            assert result["same_b"] == right.sum()
            assert result["both_same"] == (left & right).sum()
            eligible = left.any() and not left.all() and right.any() and not right.all()
            assert result["eligible"] == eligible
            if eligible:
                expected = -np.corrcoef(left.astype(float), right.astype(float))[0, 1]
                assert result["corrected_disagreement"] == pytest.approx(expected, abs=5e-15)
            else:
                assert result["corrected_disagreement"] is None


def test_relabel_permute_swap_and_support(stability):
    a = np.array([0, 0, 0, 1, 1, 2])
    b = np.array([0, 0, 1, 1, 1, 2])
    expected = stability.compare_partitions(a, b, min_pairs=1)
    perm = [4, 1, 0, 5, 3, 2]
    assert stability.compare_partitions(-3 * a[perm] - 9, b[perm] + 14, 1) == expected
    swapped = stability.compare_partitions(b, a, 1)
    assert swapped["corrected_disagreement"] == expected["corrected_disagreement"]
    unsupported = stability.compare_partitions(a, b, 10)
    assert unsupported["corrected_disagreement"] == expected["corrected_disagreement"]
    assert not unsupported["eligible"]
    assert unsupported["reason"] == "insufficient_pair_support"


@pytest.mark.parametrize("labels", [np.arange(10), np.zeros(10, dtype=int)])
def test_raw_score_trivial_endpoints_are_not_perfect_corrected_stability(stability, labels):
    report = stability.compare_partitions(labels, labels)
    assert report["raw_disagreement"] == 0
    assert report["corrected_disagreement"] is None
    assert not report["eligible"]


def test_large_counts_use_python_integer_products(stability):
    a = np.repeat(np.arange(2), 50000)
    b = np.tile(np.repeat(np.arange(2), 25000), 2)
    result = stability.compare_partitions(a, b)
    total = 100000 * 99999 // 2
    same = 2 * (50000 * 49999 // 2)
    both = 4 * (25000 * 24999 // 2)
    expected = -(total * both - same * same) / (same * (total - same))
    assert result["total_pairs"] == total
    assert result["corrected_disagreement"] == pytest.approx(expected)
    assert stability._choose_two(np.int64(10**10)) == 10**10 * (10**10 - 1) // 2


@pytest.mark.parametrize("bad", [[True, 1], [0.0, 1.0], [np.nan, 1], [[0, 1]], [0], []])
def test_invalid_labels(stability, bad):
    with pytest.raises(ValueError):
        stability.compare_partitions(bad, [0, 1])


@pytest.mark.parametrize("minimum", [True, 0, -1, 1.5, np.inf])
def test_invalid_support(stability, minimum):
    with pytest.raises(ValueError):
        stability.compare_partitions([0, 1], [0, 1], minimum)


def test_selection_gates_and_exact_tie(stability):
    good = dict(eligible=True, corrected_disagreement=-0.7)
    unsupported = dict(eligible=False, corrected_disagreement=-1.0)
    result = stability.select_stable_factor(
        [[unsupported, good, good], [good, good, good]], [0, 1, 3]
    )
    assert result["selected_index"] == 1
    assert result["mean_scores"] == [None, -0.7, -0.7]
    none = stability.select_stable_factor([[unsupported]], [1])
    assert none["selected_factor"] is None
    assert none["no_selection_reason"]


@pytest.mark.parametrize("score", [np.nan, np.inf, 1.1, True, "0", None])
def test_eligible_invalid_scores_fail_closed(stability, score):
    with pytest.raises(ValueError):
        stability.select_stable_factor([[dict(eligible=True, corrected_disagreement=score)]], [1])


@pytest.mark.parametrize("factors", [[], [1, 1], [-1], [True], [np.nan]])
def test_invalid_factor_grids(stability, factors):
    with pytest.raises(ValueError):
        stability.select_stable_factor([], factors)
