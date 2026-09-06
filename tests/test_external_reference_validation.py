"""Independent certificate audit for candidates from external implementations.

These tests use analytic identities, not the production solver as an oracle.
No external R installation or reference package is needed.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]


def _load_reference_module():
    spec = importlib.util.spec_from_file_location(
        "external_reference", ROOT / "examples" / "external_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def certify():
    return _load_reference_module().certify_candidate


@pytest.mark.parametrize(
    "penalty,centers,dual,objective",
    [
        ("l2", [[0.6, 0.8], [2.4, 3.2]], [[-0.6, -0.8]], 4.0),
        ("l1", [[1.0, 1.0], [2.0, 3.0]], [[-1.0, -1.0]], 5.0),
    ],
)
def test_analytic_two_point_optimum(certify, penalty, centers, dual, objective):
    result = certify([[0, 0], [3, 4]], [[0, 2], [2, 0]], 0.5, penalty, centers, dual)
    assert_allclose(result["objective"], objective, atol=1e-14)
    assert_allclose(result["dual_objective"], objective, atol=1e-14)
    assert result["gap"] <= 1e-13
    assert result["relative_gap"] <= 1e-13
    assert result["kkt_residual"] <= 1e-13
    assert result["dual_violation"] <= 1e-14
    assert result["center_error_bound"] <= 1e-6


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_fully_fused_analytic_optimum(certify, penalty):
    result = certify(
        [[0, 0], [3, 4]], [[0, 1], [1, 0]], 10, penalty, [[1.5, 2], [1.5, 2]], [[-1.5, -2]]
    )
    assert_allclose(result["objective"], 6.25)
    assert_allclose(result["dual_objective"], 6.25)
    assert result["gap"] <= 1e-13
    assert result["kkt_residual"] <= 1e-13


def test_arbitrary_feasible_dual_certificate(certify):
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    result = certify(X, [[0, 2], [2, 0]], 0.5, "l2", X, [[-0.1, -0.2]])
    # Primal = 5; dual = 3*.1 + 4*.2 - (.1²+.2²) = 1.05.
    assert_allclose(result["objective"], 5)
    assert_allclose(result["dual_objective"], 1.05)
    assert_allclose(result["gap"], 3.95)
    assert_allclose(result["relative_gap"], 3.95 / 7.05)
    assert_allclose(result["center_error_bound"], np.sqrt(2 * 3.95))
    assert result["kkt_residual"] > 0
    assert result["dual_violation"] == 0


@pytest.mark.parametrize(
    "penalty,optimal,dual",
    [
        ("l2", [[0.6, 0.8], [2.4, 3.2]], [[-0.6, -0.8]]),
        ("l1", [[1.0, 1.0], [2.0, 3.0]], [[-1.0, -1.0]]),
    ],
)
def test_error_bound_is_tight_for_translated_candidate(certify, penalty, optimal, dual):
    optimal = np.array(optimal)
    centers = optimal + [2.0, -1.0]
    result = certify([[0, 0], [3, 4]], [[0, 2], [2, 0]], 0.5, penalty, centers, dual)
    actual_error = np.linalg.norm(centers - optimal)
    assert_allclose(result["gap"], 5.0, atol=1e-13)
    assert_allclose(result["center_error_bound"], actual_error, atol=1e-13)
    assert result["kkt_residual"] > 0


@pytest.mark.parametrize(
    "penalty,bad_dual",
    [
        ("l2", [[-1.1, 0]]),
        ("l2", [[-0.8, -0.8]]),
        ("l1", [[-1.1, 0]]),
        ("l1", [[0, np.nan]]),
        ("l2", [[0, np.inf]]),
    ],
)
def test_materially_infeasible_or_nonfinite_dual_rejected(certify, penalty, bad_dual):
    message = "dual" if np.isfinite(bad_dual).all() else "finite"
    with pytest.raises(ValueError, match=message):
        certify([[0, 0], [3, 4]], [[0, 1], [1, 0]], 1, penalty, [[0, 0], [3, 4]], bad_dual)


def test_l1_fusion_uses_linf_dual_ball(certify):
    # Both coordinates may independently attain radius one for l1 fusion.
    # An accidental l2-ball feasibility test would incorrectly reject this.
    result = certify([[0, 0], [3, 4]], [[0, 1], [1, 0]], 1, "l1", [[1, 1], [2, 3]], [[-1, -1]])
    assert result["dual_violation"] == 0
    assert result["gap"] <= 1e-13


def test_dual_orientation_is_not_silently_reversed(certify):
    centers = [[0.6, 0.8], [2.4, 3.2]]
    result = certify([[0, 0], [3, 4]], [[0, 2], [2, 0]], 0.5, "l2", centers, [[0.6, 0.8]])
    # The opposite-sign feasible dual has value -5-1, not the optimum 4.
    assert_allclose(result["objective"], 4)
    assert_allclose(result["dual_objective"], -6)
    assert_allclose(result["gap"], 10)
    assert result["kkt_residual"] > 0


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_row_permutation_reorders_and_reorients_edge_duals(certify, penalty):
    X = np.array([[0.0, 1.0], [2.0, -1.0], [0.5, 3.0]])
    U = np.array([[0.1, 0.8], [1.8, -0.7], [0.4, 2.8]])
    W = np.array([[0, 1, 2], [1, 0, 0.7], [2, 0.7, 0]])
    Z = np.array([[0.2, -0.1], [-0.15, 0.05], [0.1, 0.1]])
    permutation = [2, 0, 1]
    # Old edges are 01,02,12; new edges denote old 20,21,01.
    permuted_dual = np.array([-Z[1], -Z[2], Z[0]])
    original = certify(X, W, 0.6, penalty, U, Z)
    permuted = certify(
        X[permutation],
        W[np.ix_(permutation, permutation)],
        0.6,
        penalty,
        U[permutation],
        permuted_dual,
    )
    for key in [
        "objective",
        "dual_objective",
        "gap",
        "relative_gap",
        "kkt_residual",
        "dual_violation",
        "center_error_bound",
    ]:
        assert_allclose(permuted[key], original[key], atol=1e-13)


def test_feature_and_fusion_scaling(certify):
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    W = np.array([[0, 2], [2, 0]])
    U = np.array([[0.9, 0.5], [2.4, 3.2]])
    Z = np.array([[-0.6, -0.8]])
    baseline = certify(X, W, 0.5, "l2", U, Z)
    scaled = certify(4 * X, W, 2, "l2", 4 * U, 4 * Z)
    for key in ["objective", "dual_objective", "gap"]:
        assert_allclose(scaled[key], 16 * baseline[key], atol=1e-12)
    assert_allclose(scaled["center_error_bound"], 4 * baseline["center_error_bound"])
    reweighted = certify(X, 4 * W, 0.125, "l2", U, Z)
    for key in baseline:
        assert_allclose(reweighted[key], baseline[key], atol=1e-13)


def test_common_translation_preserves_certificate(certify):
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    U = np.array([[0.9, 0.5], [2.4, 3.2]])
    Z = np.array([[-0.6, -0.8]])
    W = [[0, 2], [2, 0]]
    baseline = certify(X, W, 0.5, "l2", U, Z)
    translated = certify(X + [10, -7], W, 0.5, "l2", U + [10, -7], Z)
    for key in baseline:
        assert_allclose(translated[key], baseline[key], atol=1e-12)


def test_zero_gamma_error_bound(certify):
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    U = X + [[0.2, -0.1], [0.4, 0.3]]
    result = certify(X, [[0, 2], [2, 0]], 0, "l2", U, [[0, 0]])
    assert_allclose(result["gap"], 0.5 * np.sum((U - X) ** 2))
    assert_allclose(result["center_error_bound"], np.linalg.norm(U - X))


def test_unsupported_norm_rejected(certify):
    with pytest.raises(ValueError, match="penalty"):
        certify([[0], [1]], [[0, 1], [1, 0]], 1, "linf", [[0], [1]], [[0]])


def test_certificate_does_not_use_production_diagnostics_or_proximals(monkeypatch):
    import ssnalclust.prox as prox
    import ssnalclust.solvers as solvers

    def forbidden(*args, **kwargs):
        raise AssertionError("External candidate audit called production optimization code")

    for name in ["project_dual", "prox_norm", "penalty_value"]:
        monkeypatch.setattr(prox, name, forbidden)
    monkeypatch.setattr(solvers, "_diagnostics", forbidden)
    monkeypatch.setattr(solvers, "solve", forbidden)
    independent = _load_reference_module().certify_candidate
    result = independent(
        [[0, 0], [3, 4]], [[0, 2], [2, 0]], 0.5, "l2", [[0.6, 0.8], [2.4, 3.2]], [[-0.6, -0.8]]
    )
    assert result["gap"] <= 1e-13
