# ssnalclust

Convex clustering for scientific Python: sparse optimization, independently
checked solutions, and a scikit-learn estimator interface.

The original single-file SSNAL implementation contained mathematical errors.
This package replaces it with tested solvers. The [algorithm audit](docs/algorithm_audit.md)
records the errors and derivations. This is an initial development release,
not a claim to match the original MATLAB implementation's large-scale speed.

## Installation

From this repository (Python 3.10+):

```bash
python -m pip install .
# Development, including independent CVXPY correctness tests:
python -m pip install -e '.[dev]'
python -m pytest -q
```

Runtime dependencies are NumPy, SciPy, and scikit-learn. CVXPY is used only in
tests; PyLops is no longer required. A PyPI release has not been published.

## Quick start

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from ssnalclust import ConvexClustering

X = [[0., 0.], [0.1, 0.2], [4., 4.], [4.2, 4.1]]
model = make_pipeline(StandardScaler(), ConvexClustering(gamma=0.5))
labels = model.fit_predict(X)
centroids = model[-1].centers_
print(labels, model[-1].result_.relative_gap)
```

Samples are **rows** and features are columns. The estimator builds a symmetric
Gaussian k-nearest-neighbor graph, with weight
`exp(-distance**2 / (2 * bandwidth**2))`, taking the union of directed neighbors.
Scaling is your choice: a pipeline makes it explicit. Supply `weights=` for a
custom sparse symmetric adjacency. Zero weights mean absent edges, and
self-edges are rejected. Additional builders include `self_tuning_graph` for
local bandwidths, `minimum_spanning_tree_graph`, and
`connected_k_neighbors_graph` (kNN plus MST). Exact MST construction uses
quadratic distance storage, so use it only when that memory cost is acceptable.
Connected builders reject required edges whose weights underflow to zero.

`fit` returns the estimator, `fit_predict` returns labels, and `get_params`,
`set_params`, and cloning follow scikit-learn conventions. There is no
out-of-sample `predict`: convex clustering estimates all input centroids jointly.
`sample_weight` in `fit` means strictly positive fidelity mass on a **fixed**
graph; duplicating points changes the graph problem and is not equivalent.

## Objective and solvers

For data A, centroids U, fidelity masses m, and edge weights w:

```
min_U  0.5 sum_i m_i ||U_i - A_i||_2²
       + gamma sum_{i<j} w_ij ||U_i - U_j||_q
```

The low-level `solve` function defaults to a **complete unit-weight graph**.
For large datasets, pass an explicit sparse graph. The estimator defaults to
kNN to avoid quadratic graph storage.

| Solver | Edge norms | Method |
| --- | --- | --- |
| `ssnal` | `l2` | Augmented Lagrangian with semismooth Newton-CG and matrix-free generalized Hessians |
| `admm` | `l1`, `l2`, `linf` | Split ADMM with a reused sparse factorization |
| `ama` | `l1`, `l2`, `linf` | Dual projected gradient with a safe graph-based step |
| `fama` | `l1`, `l2`, `linf` | FISTA acceleration of dual projected gradient |

```python
from ssnalclust import solve, k_neighbors_graph, convex_clustering_path

W = k_neighbors_graph(X, n_neighbors=2)
result = solve(X, weights=W, gamma=0.2, solver="admm", penalty="l1",
               tol=1e-7, max_iter=5000)
assert result.converged, result.message
path = convex_clustering_path(X, [0, 0.1, 0.5, 1], weights=W)
```

Paths reuse primal and dual iterates and keep the graph fixed. Every path
point has its own convergence diagnostics. `summarize_path(path)` reports
cluster counts, certificates, and numerical merge/split transitions. Arbitrary weighted convex
clustering paths can split; this package does not force a dendrogram or
irreversibly compress fused clusters.

## Accuracy and interpretation

`SolverResult` reports the feasible primal objective, feasible dual objective,
absolute and relative duality gaps, normalized KKT residual, convergence flag,
iteration count, and diagnostic history. Success requires **both** the relative
gap and KKT residual to meet `tol`. Exhausting `max_iter` returns an unconverged
result; the estimator also emits `ConvergenceWarning`. Inspect these results,
especially for first-order methods at tight tolerances. Certificates describe
the returned floating-point centroids. If a large coordinate offset makes the
requested accuracy unrepresentable, the result is marked unconverged with a
message suggesting centering or rescaling.

`centers_` contains optimization centroids. `labels_` connects every pair of
centroids within `cluster_tol`, then takes transitive components. A component's
diameter can exceed that tolerance. `cluster_centers_` averages the fitted
centroids in each label group. Numerical labels may change near a fusion event;
optimization tolerance and label tolerance serve different purposes.

Positive fidelity masses imply a unique centroid solution. Disconnected
weight graphs solve independently, and isolated observations remain unchanged.
The edge dual need not be unique. `gamma=0` returns the observations exactly
(up to floating-point representation).

## Additional models

The package also includes separately formulated solvers for missing entries,
feature-sparse clustering, biclustering, and robust/generalized fidelities.
See [model definitions and examples](docs/models.md). These use primal-dual
hybrid gradient (PDHG) with actual KKT checks. They do **not** reuse the
complete-data squared-loss duality certificate or assert uniqueness for
models that lack strict convexity.

## Compatibility with the original file

```python
from ssnalclust import SSNAL
legacy = SSNAL(A, weights_matrix, gamma)  # A has observations in columns
X, U, Z, status = legacy.fit(max_iter=200)
```

The deprecated adapter preserves constructor, mutable `gamma`, `fit` warm
starts, column layout, and 0/1 termination status. Its `result_` exposes the new
diagnostics. Historical internal optimization helpers are intentionally not
part of the compatibility API. `U0` is validated but the eliminated split
variable does not need an initial value.

## Validation and performance

Tests include analytic two-point solutions, independent CVXPY comparisons,
finite differences of the reduced gradient and generalized Hessian,
translation/rotation and weighted-mean invariants, sparse/disconnected graphs,
zero regularization, nonconvergence, and estimator integration. CVXPY is an
independent test oracle, never a runtime fallback.

```bash
python examples/basic_usage.py
python examples/benchmark.py --samples 100 500 --features 3
```

The benchmark reports achieved accuracy alongside timings. Recorded local
results are in [benchmark_results.jsonl](docs/benchmark_results.jsonl); they
are not portable performance guarantees. Sparse ADMM factorizations can fill
in; high-dimensional nearest-neighbor queries and dense graphs can be costly.
See [scope and development plan](docs/scope.md) for remaining work.

## References

- Sun, Toh, and Yuan (2021), [Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm](https://www.jmlr.org/papers/volume22/18-694/18-694.pdf).
- Chi, Molstad, Gao, and Chi (2025), [The Why and How of Convex Clustering](https://arxiv.org/html/2507.09077v2).
- Chi and Lange (2015), *Splitting Methods for Convex Clustering* (ADMM and AMA).

See [CITATION.cff](CITATION.cff). Algorithmic references deserve citation in
scientific work using the corresponding methods.
