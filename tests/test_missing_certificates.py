"""Independent certificates for the equivalent bounded missing-data model."""

from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust.missing import solve_missing


def _bounds_and_incidence(x, weights, gamma):
    """Independent graph traversal, component extrema, and dense incidence."""
    observed = np.isfinite(x)
    lower = np.empty_like(x)
    upper = np.empty_like(x)
    unseen = set(range(len(x)))
    while unseen:
        stack = [min(unseen)]
        component = []
        while stack:
            node = stack.pop()
            if node not in unseen:
                continue
            unseen.remove(node)
            component.append(node)
            if gamma > 0:
                stack.extend(np.flatnonzero(gamma * weights[node] > 0))
        for feature in range(x.shape[1]):
            values = x[component, feature]
            values = values[np.isfinite(values)]
            assert len(values), "test data must identify each component feature"
            lower[component, feature] = values.min()
            upper[component, feature] = values.max()
    edges = [(i, j) for i in range(len(x)) for j in range(i + 1, len(x)) if weights[i, j] > 0]
    b = np.zeros((len(edges), len(x)))
    radii = np.empty(len(edges))
    for e, (i, j) in enumerate(edges):
        b[e, i], b[e, j] = 1.0, -1.0
        radii[e] = gamma * weights[i, j]
    return observed, lower, upper, b, radii


def _objective(u, x, observed, b, radii, penalty):
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    return 0.5 * np.sum((u[observed] - x[observed]) ** 2) + radii @ np.linalg.norm(
        b @ u, ord=order, axis=1
    )


def _box_dual(x, observed, lower, upper, b, z):
    a = b.T @ z
    t = np.where(a >= 0, lower, upper)
    t[observed] = np.clip(x[observed] - a[observed], lower[observed], upper[observed])
    # Build this scalar sum directly, independently of solver gap helpers.
    terms = a * t
    terms[observed] += 0.5 * (t[observed] - x[observed]) ** 2
    return float(terms.sum()), t, a


def _problem():
    x = np.array(
        [
            [-3.0, 10.0, np.nan],
            [np.nan, 16.0, np.nan],
            [1.0, np.nan, 2.0],
            [40.0, -8.0, 0.0],
            [43.0, -5.0, 9.0],
            [-100.0, 2.0, 6.0],
        ]
    )
    weights = np.zeros((6, 6))
    for i, j, weight in [(0, 1, 1.4), (1, 2, 0.7), (3, 4, 2.0)]:
        weights[i, j] = weights[j, i] = weight
    return x, weights


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_componentwise_clipping_decreases_the_original_objective(penalty):
    x, weights = _problem()
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 0.8)
    u = np.random.default_rng(496).normal(size=x.shape) * 100
    clipped = np.clip(u, lower, upper)
    assert np.all(np.abs(b @ clipped) <= np.abs(b @ u) + 1e-12)
    assert np.all(np.abs(clipped[observed] - x[observed]) <= np.abs(u[observed] - x[observed]))
    assert _objective(clipped, x, observed, b, radii, penalty) <= _objective(
        u, x, observed, b, radii, penalty
    )
    assert_allclose(lower[0], [-3.0, 10.0, 2.0])
    assert_allclose(upper[0], [1.0, 16.0, 2.0])
    assert_allclose(lower[3], [40.0, -8.0, 0.0])
    assert_allclose(upper[3], [43.0, -5.0, 9.0])
    assert_allclose(lower[5], upper[5])


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
@pytest.mark.parametrize("gamma", [0.2, 3.0])
def test_original_boxed_primal_and_extended_dual_oracles_agree(penalty, gamma):
    cp = pytest.importorskip("cvxpy")
    x, weights = _problem()
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, gamma)
    filled = np.where(observed, x, 0.0)
    order = {"l1": 1, "l2": 2, "linf": np.inf}[penalty]
    dual_order = {"l1": np.inf, "l2": 2, "linf": 1}[penalty]
    u = cp.Variable(x.shape)
    objective = 0.5 * cp.sum_squares(cp.multiply(observed, u - filled))
    objective += sum(radii[e] * cp.norm((b @ u)[e], order) for e in range(len(b)))
    original = cp.Problem(cp.Minimize(objective))
    boxed = cp.Problem(cp.Minimize(objective), [u >= lower, u <= upper])
    options = dict(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    original.solve(**options)
    boxed.solve(**options)
    assert original.status == boxed.status == cp.OPTIMAL
    assert_allclose(original.value, boxed.value, atol=2e-6, rtol=1e-8)

    # Independent extended dual with explicit nonnegative lower/upper bound
    # multipliers. Zero divergence is imposed only after adding box normals.
    z = cp.Variable((len(b), x.shape[1]))
    lower_multiplier = cp.Variable(x.shape, nonneg=True)
    upper_multiplier = cp.Variable(x.shape, nonneg=True)
    a = b.T @ z - lower_multiplier + upper_multiplier
    dual_objective = cp.sum(cp.multiply(filled, a)) - 0.5 * cp.sum_squares(cp.multiply(observed, a))
    dual_objective += cp.sum(
        cp.multiply(lower_multiplier, lower) - cp.multiply(upper_multiplier, upper)
    )
    constraints = [cp.multiply(~observed, a) == 0]
    constraints += [cp.norm(z[e], dual_order) <= radii[e] for e in range(len(b))]
    dual = cp.Problem(cp.Maximize(dual_objective), constraints)
    dual.solve(**options)
    assert dual.status == cp.OPTIMAL
    assert_allclose(dual.value, original.value, atol=3e-6, rtol=2e-8)

    result = solve_missing(x, weights, gamma=gamma, penalty=penalty, tol=2e-7, max_iter=20000)
    assert result.certificate_model == "observed_range_box"
    assert result.converged
    assert result.kkt_residual <= 2e-7
    assert result.relative_gap <= 2e-7
    assert np.all(result.centers >= lower) and np.all(result.centers <= upper)
    assert np.all(np.linalg.norm(result.dual, ord=dual_order, axis=1) <= radii + 1e-12)
    independent_dual, _, _ = _box_dual(x, observed, lower, upper, b, result.dual)
    assert_allclose(result.dual_objective, independent_dual, atol=3e-11, rtol=1e-12)
    assert result.dual_objective <= original.value + 3e-6
    assert result.objective >= original.value - 3e-6
    assert_allclose(result.gap, result.objective - result.dual_objective, atol=2e-10, rtol=1e-5)
    for record in result.history:
        assert record["gap"] >= 0
        assert_allclose(
            record["relative_gap"],
            record["gap"] / (1 + abs(record["objective"]) + abs(record["dual_objective"])),
        )


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_box_certificate_does_not_claim_unconstrained_dual_feasibility(penalty):
    x = np.array([[0.0], [np.nan], [4.0]])
    weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 0.5)
    z = np.array([[0.2], [-0.1]])
    dual_value, _, a = _box_dual(x, observed, lower, upper, b, z)
    assert np.any(a[~observed] != 0)  # ordinary masked-loss dual would be -inf
    optimum = np.array([[0.5], [1.0], [3.5]])
    value = _objective(optimum, x, observed, b, radii, penalty)
    assert_allclose(value, 1.75)
    assert np.isfinite(dual_value)
    assert dual_value <= value


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_nonunique_missing_centers_share_the_same_certified_optimum(penalty):
    x = np.array([[0.0], [np.nan], [4.0]])
    weights = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 0.5)
    z = np.full((2, 1), -0.5)
    dual_value, _, _ = _box_dual(x, observed, lower, upper, b, z)
    for middle in [0.5, 1.0, 2.0, 3.5]:
        u = np.array([[0.5], [middle], [3.5]])
        assert_allclose(_objective(u, x, observed, b, radii, penalty), dual_value, atol=1e-14)
    result = solve_missing(x, weights, gamma=0.5, penalty=penalty, tol=1e-9)
    assert result.converged
    assert 0.5 - 1e-7 <= result.centers[1, 0] <= 3.5 + 1e-7
    assert_allclose(result.objective, 1.75, atol=1e-8)
    assert result.relative_gap <= 1e-9


def test_collapsed_box_gap_can_vanish_without_original_multiplier_stationarity():
    x = np.array([[3.0], [np.nan]])
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 1.0)
    z = np.array([[0.4]])
    boxed_dual, _, a = _box_dual(x, observed, lower, upper, b, z)
    u = np.full((2, 1), 3.0)
    assert_allclose(_objective(u, x, observed, b, radii, "l2") - boxed_dual, 0.0)
    assert np.linalg.norm(a) > 0.5
    # This does not justify reporting the supplied z as an original KKT
    # multiplier. The solver retains the original KKT check as a separate gate.
    result = solve_missing(x, weights)
    assert result.converged
    assert result.kkt_residual == 0


def test_extended_dual_sign_with_fixed_nonzero_box_multipliers():
    # Fix multipliers instead of optimizing them: sign symmetry of optimized
    # edge variables must not mask an incorrect conjugate convention.
    x = np.array([[2.0], [np.nan], [5.0]])
    mask = np.isfinite(x)
    b = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    z = np.array([[0.5], [-0.25]])
    lower, upper = np.full_like(x, 2.0), np.full_like(x, 5.0)
    lower_multiplier = np.array([[0.125], [0.25], [0.375]])
    upper_multiplier = np.array([[0.375], [1.0], [0.25]])
    c = b.T @ z - lower_multiplier + upper_multiplier
    assert c[1, 0] == 0  # finite infimum at the missing coordinate
    minimizer = np.array([[2.0 - c[0, 0]], [10.0], [5.0 - c[2, 0]]])

    def lagrangian(u):
        # V=0 minimizes the split edge terms since z lies in their unit balls.
        return (
            0.5 * np.sum((u[mask] - x[mask]) ** 2)
            + np.sum(z * (b @ u))
            + np.sum(lower_multiplier * (lower - u))
            + np.sum(upper_multiplier * (u - upper))
        )

    bound_terms = np.sum(lower_multiplier * lower - upper_multiplier * upper)
    dual = np.sum(x[mask] * c[mask] - 0.5 * c[mask] ** 2) + bound_terms
    assert_allclose(dual, lagrangian(minimizer), atol=1e-14)
    assert_allclose(dual, -4.7890625, atol=1e-14)
    perturbation = np.array([[0.25], [-123.0], [0.5]])
    assert_allclose(
        lagrangian(minimizer + perturbation) - dual,
        0.5 * np.sum(perturbation[mask] ** 2),
        atol=1e-14,
    )
    wrong_sign = np.sum(-x[mask] * c[mask] - 0.5 * c[mask] ** 2) + bound_terms
    assert abs(wrong_sign - lagrangian(minimizer)) > 1.0


def test_stable_box_gap_decomposition_matches_direct_gap():
    x, weights = _problem()
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 0.8)
    rng = np.random.default_rng(111)
    z = rng.normal(size=(len(b), x.shape[1]))
    z *= radii[:, None] / np.maximum(np.linalg.norm(z, axis=1), radii)[:, None]
    u = rng.uniform(lower, upper)
    dual_value, t, a = _box_dual(x, observed, lower, upper, b, z)
    difference = u - t
    observed_terms = 0.5 * difference[observed] ** 2 + difference[observed] * (
        t[observed] - x[observed] + a[observed]
    )
    missing_terms = a[~observed] * difference[~observed]
    edge_terms = radii * np.linalg.norm(b @ u, axis=1) - np.einsum("ij,ij->i", z, b @ u)
    assert np.all(observed_terms >= -1e-14)
    assert np.all(missing_terms >= -1e-14)
    assert np.all(edge_terms >= -1e-14)
    decomposed = observed_terms.sum() + missing_terms.sum() + edge_terms.sum()
    assert_allclose(decomposed, _objective(u, x, observed, b, radii, "l2") - dual_value, atol=1e-11)


def _decimal_box_dual(x, observed, lower, upper, b, z):
    """High-precision scalar minimization including exact edge accumulation."""
    with localcontext() as context:
        context.prec = 70
        value = Decimal(0)
        for i in range(len(x)):
            for j in range(x.shape[1]):
                a = sum(
                    (
                        Decimal.from_float(float(b[e, i])) * Decimal.from_float(float(z[e, j]))
                        for e in range(len(b))
                    ),
                    Decimal(0),
                )
                low, high = Decimal.from_float(lower[i, j]), Decimal.from_float(upper[i, j])
                if observed[i, j]:
                    data = Decimal.from_float(x[i, j])
                    target = min(max(data - a, low), high)
                    value += (target - data) ** 2 / 2 + a * target
                else:
                    value += a * (low if a >= 0 else high)
        return float(value)


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_returned_certificates_under_large_independent_component_translations(penalty):
    x, weights = _problem()
    shifted = x.copy()
    shifted[:3] += [1e9, -1e9, 5e8]
    shifted[3:5] += [-3e8, 2e9, 7e8]
    shifted[5] += [4e9, -3e9, 8e8]
    observed, lower, upper, b, radii = _bounds_and_incidence(shifted, weights, 0.4)
    result = solve_missing(shifted, weights, gamma=0.4, penalty=penalty, tol=5e-6, max_iter=20000)
    assert result.converged
    actual_objective = _objective(result.centers, shifted, observed, b, radii, penalty)
    actual_dual = _decimal_box_dual(shifted, observed, lower, upper, b, result.dual)
    assert_allclose(result.objective, actual_objective, atol=1e-9, rtol=1e-12)
    assert_allclose(result.dual_objective, actual_dual, atol=2e-9, rtol=1e-12)
    assert_allclose(result.gap, actual_objective - actual_dual, atol=2e-9, rtol=1e-5)
    gradient = np.zeros_like(shifted)
    gradient[observed] = result.centers[observed] - shifted[observed]
    a = b.T @ result.dual
    stationarity = np.linalg.norm(gradient + a) / (1 + np.linalg.norm(gradient) + np.linalg.norm(a))
    assert result.kkt_residual >= stationarity - 1e-14


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_underflowed_fusion_link_does_not_join_component_bounds(penalty):
    x = np.array([[0.0, np.nan], [2.0, 4.0], [100.0, 3.0], [102.0, np.nan]])
    weights = np.zeros((4, 4))
    weights[0, 1] = weights[1, 0] = weights[2, 3] = weights[3, 2] = 1.0
    weights[1, 2] = weights[2, 1] = np.nextafter(0.0, 1.0)
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 0.5)
    assert radii[1] == 0
    result = solve_missing(x, weights, gamma=0.5, penalty=penalty, tol=1e-8)
    assert result.converged
    assert_allclose(result.dual[1], 0.0)
    assert_allclose(result.centers[:2, 1], 4.0)
    assert_allclose(result.centers[2:, 1], 3.0)
    assert_allclose(
        result.objective, _objective(result.centers, x, observed, b, radii, penalty), atol=1e-12
    )
    dual_value, _, _ = _box_dual(x, observed, lower, upper, b, result.dual)
    assert_allclose(result.dual_objective, dual_value, atol=1e-11)


def test_gamma_zero_links_leave_independent_singleton_boxes():
    x = np.array([[-1e9, 3.0], [1e9, -4.0]])
    result = solve_missing(x, [[0.0, 1.0], [1.0, 0.0]], gamma=0.0, tol=1e-12)
    assert result.converged and result.n_iter == 0
    assert_allclose(result.centers, x)
    assert result.objective == result.dual_objective == result.gap == 0.0


@pytest.mark.parametrize("penalty", ["l1", "l2", "linf"])
def test_finite_ignored_placeholders_do_not_change_bounds_or_certificate(penalty):
    x, weights = _problem()
    observed = np.isfinite(x)
    filled = np.where(observed, x, 1e100)
    reference = solve_missing(x, weights, gamma=0.4, penalty=penalty, tol=1e-8)
    result = solve_missing(filled, weights, gamma=0.4, penalty=penalty, tol=1e-8, observed=observed)
    assert result.converged and reference.converged
    assert_allclose(result.centers, reference.centers, atol=0.0)
    for field in ("objective", "dual_objective", "gap", "relative_gap", "kkt_residual"):
        assert getattr(result, field) == getattr(reference, field)


def test_small_initial_kkt_cannot_skip_the_gap_gate():
    result = solve_missing(
        [[-1e8], [np.nan], [1e8]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        gamma=0.5,
        max_iter=1,
        tol=1e-6,
    )
    assert result.history[0]["kkt_residual"] < 1e-6
    assert result.history[0]["relative_gap"] > 0.9
    assert result.n_iter == 1
    assert not result.converged


def test_extreme_range_reports_kkt_of_returned_values_not_rounded_centered_data():
    x = np.array([[-1e16], [1.0]])
    result = solve_missing(x, [[0.0, 1.0], [1.0, 0.0]], gamma=8.0, max_iter=100, tol=1e-8)
    gradient = result.centers - x
    a = np.vstack([result.dual[0], -result.dual[0]])
    stationarity = np.linalg.norm(gradient + a) / (1 + np.linalg.norm(gradient) + np.linalg.norm(a))
    differences = result.centers[:1] - result.centers[1:]
    trial = differences + result.dual
    proximal = np.sign(trial) * np.maximum(np.abs(trial) - 8.0, 0.0)
    subgradient = np.linalg.norm(differences - proximal) / (
        1 + np.linalg.norm(differences) + np.linalg.norm(result.dual)
    )
    assert_allclose(result.kkt_residual, max(stationarity, subgradient), atol=1e-14)
    assert result.converged == (max(result.kkt_residual, result.relative_gap) <= 1e-8)


def test_box_can_select_among_original_optima_without_preserving_all_of_them():
    x = np.array([[0.0, 0.0], [np.nan, 10.0]])
    weights = np.array([[0.0, 1.0], [1.0, 0.0]])
    observed, lower, upper, b, radii = _bounds_and_incidence(x, weights, 0.1)
    outside = np.array([[0.0, 0.1], [5.0, 9.9]])
    clipped = np.clip(outside, lower, upper)
    z = np.array([[0.0, -0.1]])
    dual_value, _, _ = _box_dual(x, observed, lower, upper, b, z)
    assert outside[1, 0] > upper[1, 0]
    assert_allclose(_objective(outside, x, observed, b, radii, "linf"), 0.99)
    assert_allclose(_objective(clipped, x, observed, b, radii, "linf"), dual_value, atol=1e-14)
    assert_allclose(dual_value, 0.99)
