"""Reuse one graph/factorization and stream a path without retaining histories."""

import numpy as np

from ssnalclust import ConvexClusteringProblem, k_neighbors_graph

rng = np.random.default_rng(16)
X = np.vstack([rng.normal(-1, 0.15, (20, 2)), rng.normal(1, 0.15, (20, 2))])
problem = ConvexClusteringProblem(X, weights=k_neighbors_graph(X, n_neighbors=8))
print(f"Prepared {problem.n_samples} samples and {problem.n_edges} edges")
for gamma, result in zip(
    np.geomspace(0.01, 1.0, 8),
    problem.iter_path(
        np.geomspace(0.01, 1.0, 8), solver="admm", max_iter=5000, tol=1e-7, store_history=False
    ),
):
    assert result.converged, result.message
    assert result.history == []
    print(
        f"gamma={gamma:.4f}, objective={result.objective:.6f}, "
        f"gap={result.relative_gap:.2e}, iterations={result.n_iter}"
    )
