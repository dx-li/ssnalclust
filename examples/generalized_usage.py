"""Robust, binary, and count clustering with natural-parameter fusion.

Run from the repository root after installation:
    python examples/generalized_usage.py
"""

import numpy as np

from ssnalclust.generalized import solve_generalized


def main():
    examples = {
        "huber": np.array([[0.0, 0.2], [0.1, 0.0], [3.0, 3.1], [3.2, 3.0], [12.0, -5.0]]),
        "logistic": np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
        "poisson": np.array([[0.0, 1.0], [1.0, 0.0], [5.0, 4.0], [6.0, 5.0]]),
    }
    for loss, data in examples.items():
        result = solve_generalized(data, loss=loss, gamma=0.25, huber_delta=0.5)
        if not result.converged:
            raise RuntimeError(f"{loss} did not converge: {result.kkt_residual}")
        print(f"{loss}: objective={result.objective:.6f}, KKT={result.kkt_residual:.2e}")
        print("Fitted means:\n", np.round(result.fitted_means, 3))


if __name__ == "__main__":
    main()
