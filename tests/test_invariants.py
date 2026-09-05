"""Numerical and model invariants beyond small optimization oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ssnalclust import solve


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_large_translation_preserves_certificate(solver):
    x = np.array([[0.0, 1.0], [2.0, 0.0], [4.0, 8.0]])
    first = solve(x, gamma=0.4, solver=solver, tol=1e-8, max_iter=5000)
    shifted = solve(x + 1e10, gamma=0.4, solver=solver, tol=1e-8, max_iter=5000)
    assert first.converged
    # At this offset float64 cannot represent 1e-8 optimality.
    assert not shifted.converged
    assert "input scale" in shifted.message
    assert_allclose(shifted.centers - 1e10, first.centers, atol=2e-6)
    assert_allclose(shifted.objective, first.objective, atol=2e-6)
    assert_allclose(shifted.dual_objective, first.dual_objective, atol=1e-7)


@pytest.mark.parametrize(
    "name,value", [("sample_weight", [1 + 1j, 2]), ("x0", [[1j], [1]]), ("dual0", [[1j]])]
)
def test_complex_auxiliary_inputs_rejected(name, value):
    with pytest.raises(ValueError, match="real"):
        solve([[0.0], [1.0]], **{name: value})


@pytest.mark.parametrize("solver", ["ssnal", "admm", "ama", "fama"])
def test_weighted_mean_preservation(solver):
    x = np.random.default_rng(89).normal(size=(9, 3))
    mass = np.arange(1, 10)
    result = solve(x, gamma=0.8, sample_weight=mass, solver=solver, tol=1e-8, max_iter=10000)
    assert result.converged
    assert_allclose(
        np.average(result.centers, axis=0, weights=mass),
        np.average(x, axis=0, weights=mass),
        atol=1e-7,
    )


def test_diagnostics_are_json_serializable_scalars():
    import json

    result = solve([[0.0], [2.0]], gamma=5.0)
    json.dumps(
        {
            name: getattr(result, name)
            for name in [
                "objective",
                "dual_objective",
                "gap",
                "relative_gap",
                "kkt_residual",
                "converged",
                "n_iter",
            ]
        }
    )
    assert type(result.converged) is bool
