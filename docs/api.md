# Public API reference

All public names below are importable from `ssnalclust`. Arrays use samples in
rows unless the deprecated `SSNAL` adapter is explicitly used. Function and
class docstrings describe individual parameters; this page maps the interfaces,
returned data, and decisions that affect scientific interpretation. Start with
the [user guide](user_guide.md) for a worked workflow.

## Shared conventions

- `X` has shape `(n_samples, n_features)` and uses dense real observations.
  Missing-data interfaces accept an observed-entry mask; other interfaces
  require finite observations.
- A sample adjacency has shape `(n_samples, n_samples)`, is symmetric and
  nonnegative, and has a zero diagonal. Zero entries omit edges. Dense and
  SciPy sparse adjacencies are accepted. Sample ordering must match `X`.
- `gamma`, `alpha`, `gamma_row`, and `gamma_col` are nonnegative regularization
  strengths. `tol` is positive and `max_iter` is a positive iteration budget.
  Likelihood and missing-entry models can impose stronger conditions for
  existence of a finite identifiable fit.
- Estimator `fit` returns the estimator and accepts an ignored `y` where
  applicable. No clustering estimator defines out-of-sample `predict`.
- Low-level optimization returns diagnostics even when it exhausts the
  iteration budget. Check `converged`. Estimators also issue
  `sklearn.exceptions.ConvergenceWarning` when convergence is not achieved.

## Complete squared-loss clustering

### `ConvexClustering`

[Source and full docstring](../src/ssnalclust/estimator.py)

```text
ConvexClustering(gamma=1.0, solver="ssnal", penalty="l2", weights=None,
                n_neighbors=10, bandwidth=1.0, tol=1e-6, max_iter=200,
                cluster_tol=1e-4)
fit(X, y=None, sample_weight=None)
fit_predict(X, y=None, **fit_parameters)
```

With `weights=None`, fitting constructs a symmetric Gaussian kNN graph.
`n_neighbors` is clipped to `n_samples - 1`; a singleton is supported.
`sample_weight` gives one **strictly positive fidelity mass** per sample.

| Learned attribute | Meaning |
| --- | --- |
| `centers_` | Optimized centroid for each observation, shape `(n_samples, n_features)` |
| `labels_` | Integer labels from transitive Euclidean centroid thresholding |
| `cluster_centers_` | Mean optimized centroid within each label group |
| `n_clusters_` | Number of numerical label groups |
| `weights_` | Adjacency used for the fit |
| `result_` | Complete `SolverResult` |
| `objective_`, `dual_gap_`, `converged_`, `n_iter_` | Selected result diagnostics |
| `n_features_in_`, `feature_names_in_` | Scikit-learn input metadata; names appear when applicable |

The labeling threshold operates on every centroid pair, including pairs absent
from the graph. Transitive groups can have diameter greater than `cluster_tol`.
Labels are numerical summaries, not exact symbolic fusion events.

### `solve`

[Source and full docstring](../src/ssnalclust/solvers.py)

```text
solve(X, weights=None, gamma=1.0, penalty="l2", solver="ssnal", tol=1e-6,
      max_iter=1000, *, sigma=1.0, inner_max_iter=100, x0=None, dual0=None,
      sample_weight=None, store_history=True, check_every=1) -> SolverResult
```

`weights=None` means a **complete unit-weight graph**, unlike the estimator.
The objective is the summed weighted squared fidelity plus one weighted norm
per undirected edge. No sample-count normalization is implicit.

| Method | Supported `penalty` | Main operations |
| --- | --- | --- |
| `ssnal` | `l2` | Augmented Lagrangian, semismooth Newton, matrix-free CG |
| `admm` | `l1`, `l2`, `linf` | Split updates with sparse factorization |
| `ama` | `l1`, `l2`, `linf` | Dual projected gradient |
| `fama` | `l1`, `l2`, `linf` | Accelerated dual projected gradient |

`sigma` initializes the SSNAL augmented-Lagrangian penalty or fixes the ADMM
penalty. `inner_max_iter` caps SSNAL's inner Newton iterations. `x0` is in
sample-row layout; `dual0` has shape `(n_edges, n_features)` in the edge order
returned by `graph_from_weights`. The dual warm start is projected onto the
current feasible set. `store_history=False` suppresses iteration records while
retaining final result diagnostics.

### `SolverResult`

| Field | Meaning |
| --- | --- |
| `centers` | Returned centroids in the input coordinates |
| `dual` | Feasible dual rows aligned with incidence edges |
| `objective` | Primal objective evaluated at the returned centroids |
| `dual_objective` | Feasible dual objective |
| `gap` | Absolute primal-dual gap |
| `relative_gap` | `gap / (1 + abs(objective) + abs(dual_objective))` |
| `kkt_residual` | Maximum normalized stationarity and proximal residual |
| `converged` | Both relative gap and KKT residual satisfy `tol` |
| `n_iter` | Completed outer iterations |
| `history` | Per-iteration dictionaries including iteration zero, or an empty list when disabled |
| `message` | Convergence, iteration exhaustion, or a numerical limitation explanation |

Certificates describe returned floating-point arrays. A unique complete-data
centroid solution does not imply a unique edge dual.

## Prepared problems and paths

[Prepared-problem source](../src/ssnalclust/problem.py) ·
[Path-summary source](../src/ssnalclust/path.py)

```text
ConvexClusteringProblem(X, weights=None, sample_weight=None)
problem.solve(gamma=1.0, penalty="l2", solver="ssnal", tol=1e-6,
              max_iter=1000, *, sigma=1.0, inner_max_iter=100,
              x0=None, dual0=None, store_history=True, check_every=1)
problem.iter_path(gammas, **solver_options)
problem.path(gammas, **solver_options)

convex_clustering_path(X, gammas, weights=None, **solver_options)
iter_convex_clustering_path(X, gammas, weights=None, **solver_options)
summarize_path(results, cluster_tol=1e-4)
```

A prepared problem snapshots data, adjacency, and fidelity masses, shares graph
preparation across solves, and caches one ADMM factorization at the latest
`sigma`. Its `n_samples`, `n_features`, and `n_edges` properties describe the
fixed problem. Separate `.solve` calls start cold unless warm starts are given.
Use separate instances for concurrent work because solver caches are mutable.

`.path` and `convex_clustering_path` return a list of `SolverResult` objects.
`.iter_path` and `iter_convex_clustering_path` yield results one at a time;
combine them with `store_history=False` and immediate consumption for bounded
path storage. All path interfaces preserve the fixed graph and use preceding
centroids and duals as warm starts. Nonnegative finite strengths are evaluated
in input order; arbitrary increasing or decreasing sequences are valid. A
stream validates each strength when it reaches it.

`summarize_path` returns a list of dictionaries with `labels`, `n_clusters`,
`objective`, `converged`, `n_iter`, `kkt_residual`, `merges`, and `splits`.
Transitions refer to the thresholded partitions at consecutive supplied
points. Unconverged points remain marked. No hierarchical path assumption or
exact fusion-time calculation is imposed.

## Graph construction

[Core graph source](../src/ssnalclust/graph.py) ·
[Additional graph source](../src/ssnalclust/graph_extra.py)

| Function | Parameters beyond `X` | Behavior |
| --- | --- | --- |
| `k_neighbors_graph` | `n_neighbors=10, bandwidth=1.0` | Symmetric union of Gaussian weighted kNN edges; requires `1 <= n_neighbors < n_samples` |
| `connected_k_neighbors_graph` | `n_neighbors=10, bandwidth=1.0` | kNN plus exact Euclidean MST; clips positive neighbor count; singleton supported |
| `minimum_spanning_tree_graph` | `bandwidth=1.0` | Gaussian weights on the exact Euclidean MST; preserves duplicate zero-distance edges |
| `self_tuning_graph` | `n_neighbors=10, scale_neighbors=None` | Local-scale weights; clips positive neighbor counts; singleton supported |

Every builder returns a symmetric CSR adjacency with zero diagonal. Gaussian
weights use `exp(-d² / (2 * bandwidth²))`; self-tuning uses
`exp(-d² / (s_i * s_j))`. The default `scale_neighbors` equals `n_neighbors`.
The nearest distinct observation supplies a fallback for zero local scales;
all-identical data produce unit weights on retained edges.

MST construction uses quadratic distance storage. Required MST similarities
that underflow to zero raise `ValueError`. Ordinary kNN and self-tuning can
omit underflowed weights and do not guarantee connectivity.

```text
graph_from_weights(weights, n_samples) -> (incidence, edge_weights)
```

This validates an adjacency and returns a CSR incidence operator of shape
`(n_edges, n_samples)` plus a positive weight vector. Each row is `+1` at `i`
and `-1` at `j` for `i < j`; edges are lexicographically ordered by `(i, j)`.
`weights=None` constructs the complete unit graph.

## Missing entries and holdout selection

[Missing solver source](../src/ssnalclust/missing.py) ·
[Selection source](../src/ssnalclust/selection.py)

```text
solve_missing(X, weights, gamma=1.0, penalty="l2", tol=1e-6,
              max_iter=10000, observed=None) -> MissingResult
MissingConvexClustering(weights=None, gamma=1.0, penalty="l2", tol=1e-6,
                       max_iter=10000, cluster_tol=1e-4)
estimator.fit(X, y=None, observed=None)

select_gamma(X, gammas, weights, validation_fraction=0.1, random_state=None,
             **missing_solver_options) -> SelectionResult
```

Missing fitting requires an explicit graph. By default, finite entries are
observed and NaN/infinite entries are missing. An explicit boolean mask must
match `X`; entries marked observed must be finite. Ignored entries have no
influence on the fit, including initialization. Each feature must be observed
somewhere in every positive-fusion component; gamma zero therefore requires
fully observed data.

`MissingResult` contains `centers`, `dual`, `objective`, `kkt_residual`,
`n_iter`, `converged`, and `history`. It does not contain a complete-data
duality gap. Missing-coordinate solutions need not be unique.

`select_gamma` evaluates positive strengths using a reproducible entry holdout.
The fixed graph must not leak validation targets. It reserves sufficient
training observations for component/feature identifiability, uses one mask for
all candidates, and refits the selected strength on all originally observed
entries. It raises `RuntimeError` on a nonconverged candidate or final refit.
It controls `observed` internally; callers cannot override that mask.

| `SelectionResult` field | Meaning |
| --- | --- |
| `gammas`, `validation_errors` | Candidate strengths and aligned held-out MSEs |
| `best_gamma` | First minimum-MSE candidate |
| `results` | Candidate fits on the training mask |
| `training_mask`, `validation_mask` | Boolean entry masks used for selection |
| `best_result` | Selected-gamma fit using all originally observed entries |

## Robust and generalized clustering

[Source and full docstring](../src/ssnalclust/generalized.py)

```text
solve_generalized(X, weights=None, gamma=1.0, loss="huber", huber_delta=1.0,
                  penalty="l2", tol=1e-6, max_iter=10000) -> GeneralizedResult
GeneralizedConvexClustering(weights=None, gamma=1.0, loss="huber",
                           huber_delta=1.0, penalty="l2", tol=1e-6,
                           max_iter=10000, cluster_tol=1e-4)
```

Supported losses are `huber`, `logistic`, and `poisson`; fusion norms are
`l1`, `l2`, and `linf`. Huber accepts finite real values, logistic accepts
values in `[0, 1]`, and Poisson accepts nonnegative values. `weights=None`
selects a complete graph. Component configurations without a finite
likelihood optimum are rejected; see the [model definitions](models.md).

`GeneralizedResult` contains `centers`, `fitted_means`, `dual`, `objective`,
`kkt_residual`, `n_iter`, `converged`, and `history`. For logistic and Poisson,
`centers` are natural parameters; `fitted_means` applies sigmoid or exponential.
All three losses also expose `dual_objective`, `gap`, `relative_gap`, and
`dual_scale`, with a feasible loss-specific dual and both gap/KKT stopping.
Huber centroids and fitted means coincide. The estimator exposes
`fitted_means_`; its labels threshold natural-parameter centroids. Their certificate uses the actual fidelity conjugate, not a substituted
quadratic objective.

`SolverResult.center_error_bound` (squared-loss models) evaluates the
strong-convexity error bound from the absolute gap and minimum sample mass.
For long first-order solves, `check_every` evaluates convergence periodically
and always on return. It defaults to one, and histories contain checked
iterations only.

## Sparse clustering and biclustering

[Source and full docstrings](../src/ssnalclust/structured.py) ·
[Estimator wrappers](../src/ssnalclust/model_estimators.py)

```text
solve_sparse(X, weights=None, gamma=1.0, alpha=1.0, feature_weights=None,
             tol=1e-6, max_iter=10000) -> StructuredResult
SparseConvexClustering(weights=None, gamma=1.0, alpha=1.0,
                      feature_weights=None, tol=1e-6, max_iter=10000,
                      cluster_tol=1e-4)

solve_biclustering(X, row_weights=None, column_weights=None,
                  gamma_row=1.0, gamma_col=1.0, tol=1e-6,
                  max_iter=10000) -> StructuredResult
ConvexBiclustering(row_weights=None, column_weights=None, gamma_row=1.0,
                  gamma_col=1.0, tol=1e-6, max_iter=10000, cluster_tol=1e-4)
```

Sparse clustering column-centers `X`, then combines row fusion with a weighted
sum of fitted column Euclidean norms. `feature_weights` is a nonnegative
vector of length `n_features`, defaulting to ones. `alpha=0` removes feature
sparsity. `centers + offset` restores the original feature locations.

Biclustering jointly fuses rows and columns using distinct graphs. The column
adjacency has shape `(n_features, n_features)`; `None` on either axis means a
complete unit graph for that axis. Zero strength disables that axis's fusion.
These models currently use Euclidean group penalties.

`StructuredResult` contains `centers`, `objective`, `dual_objective`, `gap`,
`relative_gap`, `kkt_residual`, `n_iter`, `converged`, `history`, and `duals`.
Convergence requires both its model-specific relative gap and KKT residual.
The `duals` dictionary contains penalty blocks: `row` has shape
`(n_row_edges, n_features)`, `column` has shape `(n_column_edges, n_samples)`,
and `feature` has shape `(n_features, n_samples)` where applicable.

Sparse results additionally provide `offset` and `feature_norms`; these are
`None` for biclustering. `SparseConvexClustering` exposes `offset_` and
`feature_norms_`. `ConvexBiclustering` exposes `row_labels_`, `column_labels_`,
`n_row_clusters_`, and `n_column_clusters_` instead of one row-only partition.

## Legacy adapter

[Legacy source](../src/ssnalclust/legacy.py)

```text
SSNAL(A, weights_matrix, gamma)
```

This deprecated adapter uses the historical **observations-in-columns** layout
and returns the historical fit tuple. New code should use `ConvexClustering`
or `solve`; the adapter's `result_` exposes modern diagnostics. Historical
private optimization helpers are not public compatibility promises.
