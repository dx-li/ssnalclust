"""Experimental co-membership instability for aligned shared observations.

Haslbeck and Wulff (2020), Equation6, define corrected disagreement as the
negative correlation of pairwise co-membership indicators:
https://doi.org/10.1007/s00180-020-00981-5
Their correction relies on simplifying assumptions; it is not a universal
bias correction or a guarantee of scientifically meaningful cluster selection.
The minimum-pair support rule here is an additional study convention.
"""

import math
from collections import Counter
from numbers import Integral, Real

import numpy as np


def _choose_two(count):
    """Count unordered pairs using Python integers, including NumPy inputs."""
    count = int(count)
    return count * (count - 1) // 2


def _labels(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer label array")
    # A mixed Python sequence must not silently turn boolean labels into ints.
    if not isinstance(values, np.ndarray) and any(
        isinstance(value, (bool, np.bool_)) for value in values
    ):
        raise ValueError(f"{name} must contain integer labels, not booleans")
    return array


def compare_partitions(labels_a, labels_b, min_pairs=10):
    """Compare two partitions on the same prealigned observations.

    Labels must be finite integer arrays; negative names are allowed. Storage
    grows with observations and observed contingency cells, never all pairs or
    a dense cluster-by-cluster table. Same-cluster and different-cluster pair
    counts must each reach min_pairs in both partitions for eligibility.
    Corrected disagreement is still reported for insufficient positive support;
    it is None when either co-membership indicator has zero variance.
    """
    if (
        isinstance(min_pairs, (bool, np.bool_))
        or not isinstance(min_pairs, Integral)
        or min_pairs < 1
    ):
        raise ValueError("min_pairs must be a positive integer")
    a, b = _labels(labels_a, "labels_a"), _labels(labels_b, "labels_b")
    if len(a) < 2 or len(a) != len(b):
        raise ValueError("Partitions need equal lengths with at least two shared observations")
    counts_a, counts_b, contingency = Counter(), Counter(), Counter()
    for left, right in zip(a, b):
        left, right = int(left), int(right)
        counts_a[left] += 1
        counts_b[right] += 1
        contingency[left, right] += 1
    total = _choose_two(len(a))
    same_a = sum(_choose_two(count) for count in counts_a.values())
    same_b = sum(_choose_two(count) for count in counts_b.values())
    both = sum(_choose_two(count) for count in contingency.values())
    margins = (same_a, total - same_a, same_b, total - same_b)
    corrected = None
    if all(margins):
        numerator = total * both - same_a * same_b
        variance_product = math.prod(margins)
        # The ratio is a squared correlation in [0,1]. Python's integer ratio
        # division avoids converting potentially huge count products to float.
        magnitude = math.sqrt((numerator * numerator) / variance_product)
        corrected = -magnitude if numerator >= 0 else magnitude
    eligible = all(count >= min_pairs for count in margins)
    return dict(
        n_shared=len(a),
        total_pairs=total,
        same_a=same_a,
        same_b=same_b,
        both_same=both,
        raw_disagreement=(same_a + same_b - 2 * both) / total,
        corrected_disagreement=corrected,
        eligible=eligible,
        reason=None
        if eligible
        else "degenerate_partition"
        if not all(margins)
        else "insufficient_pair_support",
    )


def select_stable_factor(pair_reports, factors):
    """Minimize mean corrected disagreement, requiring support in every pair.

    Reports have orientation [pair][factor] and contain comparator dictionaries.
    Invalid eligibility flags or scores raise, even for an ineligible factor.
    An ineligible comparison disqualifies its entire factor; it is never
    omitted from an average. Equal means select the first supplied factor.
    Only scalar comparison reports enter this function, never fits or truth.
    """
    try:
        factors = list(factors)
        pairs = list(pair_reports)
    except TypeError as error:
        raise ValueError("factors and pair_reports must be nonempty iterables") from error
    if (
        not factors
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
            for value in factors
        )
        or len(set(factors)) != len(factors)
    ):
        raise ValueError("factors must be unique finite nonnegative numbers")
    if not pairs:
        raise ValueError("pair_reports must contain at least one comparison pair")
    columns = [[] for _ in factors]
    eligible = [True for _ in factors]
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != len(factors):
            raise ValueError("Each pair must contain exactly one report per factor")
        for index, report in enumerate(pair):
            if (
                not isinstance(report, dict)
                or "eligible" not in report
                or "corrected_disagreement" not in report
            ):
                raise ValueError(
                    "Each comparison report needs eligibility and corrected disagreement"
                )
            flag, score = report["eligible"], report["corrected_disagreement"]
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError("Comparison eligibility must be boolean")
            if score is not None and (
                isinstance(score, (bool, np.bool_))
                or not isinstance(score, Real)
                or not math.isfinite(score)
                or not -1 <= score <= 1
            ):
                raise ValueError("Corrected disagreement must be finite in [-1,1] or None")
            if flag and score is None:
                raise ValueError("Eligible comparisons must have a finite corrected disagreement")
            eligible[index] = eligible[index] and bool(flag)
            columns[index].append(score)
    means = [
        math.fsum(scores) / len(pairs) if valid else None
        for scores, valid in zip(columns, eligible)
    ]
    candidates = [index for index, valid in enumerate(eligible) if valid]
    selected = min(candidates, key=lambda index: means[index]) if candidates else None
    return dict(
        selected_index=selected,
        selected_factor=float(factors[selected]) if selected is not None else None,
        mean_scores=means,
        eligible=eligible,
        no_selection_reason=None if selected is not None else "no_factor_eligible_for_every_pair",
    )
