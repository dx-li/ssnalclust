"""Run with: python examples/structured_usage.py."""

import numpy as np

from ssnalclust.structured import solve_biclustering, solve_sparse


def main():
    rng = np.random.default_rng(42)
    # Two row groups and two column groups, perturbed by measurement noise.
    blocks = np.repeat(np.repeat([[0.0, 4.0], [5.0, 1.0]], 4, axis=0), 3, axis=1)
    X = blocks + rng.normal(scale=0.1, size=blocks.shape)
    biclustered = solve_biclustering(X, gamma_row=0.2, gamma_col=0.2)
    assert biclustered.converged
    print("Biclustering centers:")
    print(np.round(biclustered.centers, 2))
    print("Biclustering KKT residual:", biclustered.kkt_residual)

    # Add nearly constant features. Sparse clustering centers each column;
    # its feature penalty can remove these small centered fluctuations.
    sparse_data = np.column_stack((X[:, :3], 10 + rng.normal(scale=0.01, size=(8, 2))))
    selected = solve_sparse(sparse_data, gamma=0.1, alpha=0.2)
    assert selected.converged
    print("Centered feature norms:", np.round(selected.feature_norms, 4))
    print("Removed feature means:", np.round(selected.offset, 2))
    print("Sparse clustering KKT residual:", selected.kkt_residual)
    # Recover center coordinates for plotting against the original input.
    original_coordinate_centers = selected.centers + selected.offset
    assert original_coordinate_centers.shape == sparse_data.shape


if __name__ == "__main__":
    main()
