"""Reproducible solver comparison, emitting JSON; no timing assertions.

Run: python examples/benchmark.py --samples 100 500 --features 3 20 --gammas .05 .5 5
Memory is Python-traced allocation peak; it excludes some native allocations.
"""

import argparse
import json
import platform
import time
import tracemalloc

import numpy as np
import scipy

from ssnalclust import k_neighbors_graph, solve

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--samples", type=int, nargs="+", default=[100, 500])
parser.add_argument("--features", type=int, nargs="+", default=[3])
parser.add_argument("--gammas", type=float, nargs="+", default=[0.5])
parser.add_argument("--max-iter", type=int, default=2000)
args = parser.parse_args()
for n in args.samples:
    for dimension in args.features:
        rng = np.random.default_rng(1729)
        X = rng.normal(size=(n, dimension))
        start = time.perf_counter()
        weights = k_neighbors_graph(X, min(10, n - 1))
        graph_seconds = time.perf_counter() - start
        for gamma in args.gammas:
            for method in ["ssnal", "admm", "ama", "fama"]:
                tracemalloc.start()
                start = time.perf_counter()
                result = solve(
                    X, weights=weights, gamma=gamma, solver=method, tol=1e-6, max_iter=args.max_iter
                )
                seconds = time.perf_counter() - start
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                print(
                    json.dumps(
                        dict(
                            samples=n,
                            features=dimension,
                            gamma=gamma,
                            edges=weights.nnz // 2,
                            solver=method,
                            graph_seconds=graph_seconds,
                            solve_seconds=seconds,
                            traced_peak_bytes=peak,
                            iterations=result.n_iter,
                            relative_gap=result.relative_gap,
                            kkt_residual=result.kkt_residual,
                            converged=result.converged,
                            python=platform.python_version(),
                            platform=platform.platform(),
                            numpy=np.__version__,
                            scipy=scipy.__version__,
                        )
                    ),
                    flush=True,
                )
