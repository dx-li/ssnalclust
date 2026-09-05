"""Run with: python examples/basic_usage.py (after pip install -e .)."""

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ssnalclust import ConvexClustering, convex_clustering_path, k_neighbors_graph, solve

rng = np.random.default_rng(10)
X = np.vstack([rng.normal(-2, 0.15, (15, 2)), rng.normal(2, 0.15, (15, 2))])
model = make_pipeline(StandardScaler(), ConvexClustering(gamma=1.0, n_neighbors=8))
labels = model.fit_predict(X)
fit = model[-1]
assert fit.converged_
print(f"Found {fit.n_clusters_} clusters; relative gap={fit.result_.relative_gap:.2e}")

weights = k_neighbors_graph(X, n_neighbors=8)
path = convex_clustering_path(X, [0.0, 0.01, 0.1, 1.0], weights=weights, tol=1e-7)
assert all(result.converged for result in path)
for gamma, result in zip([0.0, 0.01, 0.1, 1.0], path):
    print(f"gamma={gamma:g}: objective={result.objective:.6f}, iterations={result.n_iter}")

# Coordinate-wise fusion with a first-order solver.
l1 = solve(X, weights=weights, gamma=0.1, penalty="l1", solver="admm", max_iter=5000)
assert l1.converged
print(f"l1 ADMM KKT residual={l1.kkt_residual:.2e}")
