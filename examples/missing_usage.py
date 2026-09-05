"""Fit missing entries and select gamma using an exogenous chain graph.

Run with: python examples/missing_usage.py
"""

import numpy as np
from scipy import sparse

from ssnalclust.missing import solve_missing
from ssnalclust.selection import select_gamma

rng = np.random.default_rng(17)
# The graph represents known sequence adjacency, independent of feature data.
X = np.repeat([[0.0, 1.0], [3.0, -1.0]], 6, axis=0)
X += rng.normal(scale=0.2, size=X.shape)
X[[1, 7], [0, 1]] = np.nan
weights = sparse.diags([np.ones(11), np.ones(11)], [-1, 1], shape=(12, 12), format="csr")
fit = solve_missing(X, weights, gamma=0.3)
assert fit.converged
print(f"Masked fit: objective={fit.objective:.6f}, KKT={fit.kkt_residual:.2e}")
selection = select_gamma(X, [0.1, 0.3, 1.0], weights, random_state=42, tol=1e-6)
print(f"Selected gamma: {selection.best_gamma}")
print(f"Holdout MSE: {selection.validation_errors}")
print("Refitted centroids:")
print(selection.best_result.centers)
