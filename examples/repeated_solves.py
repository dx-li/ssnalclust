"""Reuse a prepared graph and stream cluster summaries and certificates."""

from contextlib import closing

import numpy as np

from ssnalclust import ConvexClusteringProblem, iter_path_summaries, k_neighbors_graph

rng = np.random.default_rng(16)
X = np.vstack([rng.normal(-1, 0.15, (20, 2)), rng.normal(1, 0.15, (20, 2))])
problem = ConvexClusteringProblem(X, weights=k_neighbors_graph(X, n_neighbors=8))
print(f"Prepared {problem.n_samples} samples and {problem.n_edges} edges")
gammas = np.geomspace(0.01, 1.0, 8)
# The caller owns the solver generator; closing the summarizer alone does not
# close it. Both contexts also release state if consumption stops early.
with closing(
    problem.iter_path(gammas, solver="admm", max_iter=5000, tol=1e-7, store_history=False)
) as results:
    with closing(iter_path_summaries(results)) as summaries:
        for gamma, point in zip(gammas, summaries):
            assert point["converged"], point
            print(
                f"gamma={gamma:.4f}, clusters={point['n_clusters']}, "
                f"objective={point['objective']:.6f}, "
                f"relative_gap={point['relative_gap']:.2e}, "
                f"center_error_bound={point['center_error_bound']:.2e}"
            )
            # Keep only printed scalars; collecting summaries would retain
            # one label array (and merge/split transitions) for every point.
            del point
