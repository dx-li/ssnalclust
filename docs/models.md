# Models beyond complete-data squared loss

Every model uses samples in rows, finite nonnegative symmetric graph weights,
and an explicit objective. The alternative solvers use primal-dual hybrid
gradient (PDHG), with steps chosen from a bound on the stacked incidence
operators. They report the actual primal objective and normalized KKT
residuals. Structured quadratic models additionally report a primal-dual gap;
generalized losses report a loss-specific conjugate gap after repairing dual
feasibility. Both checks must pass for success. Missing-data results currently
report KKT residuals only, without a dual-gap claim.

Use the low-level functions for full diagnostic access, or the corresponding
scikit-learn estimators for `fit`, `fit_predict` where applicable, and cloning.
The estimators emit `ConvergenceWarning` on iteration exhaustion. Functions
return a result with `converged=False`; callers must inspect it.

## Missing entries

`solve_missing(X, weights, observed=None, ...)` minimizes

```
0.5 sum_{(i,j) observed} (U_ij - X_ij)² + gamma sum_e w_e ||(BU)_e||_q.
```

Default missing entries are NaN or infinity. An explicit boolean mask can
ignore any entries, even finite ones. Ignored values do not enter either the
loss or initialization. Supply the graph explicitly: a meaningful graph
cannot be inferred universally from partially observed data.

Each feature needs an observation in each positive-fusion connected component.
Unidentifiable components are rejected. At gamma zero this requires every
entry to be observed. This policy ensures a bounded location, but missing
centroids can still be nonunique. No hidden ridge penalty is added. Fitting
selects centroids inside the component-wise observed range; coordinatewise
clipping preserves the original optimum. The equivalent box problem provides
a valid lower bound and objective gap, described in
[missing-data certificates](missing_certificates.md). Convergence requires
both this relative gap and the original unconstrained KKT residual.

```python
import numpy as np
from ssnalclust import MissingConvexClustering

X = np.array([[0., np.nan], [.2, .1], [1., .9]])
W = np.ones((3, 3)) - np.eye(3)
fit = MissingConvexClustering(weights=W, gamma=.1).fit(X)
assert fit.converged_
print(fit.centers_)
```

`select_gamma` holds out observed entries, preserves training identifiability,
fits every candidate on exactly the same mask, scores held-out MSE, and refits
the selected gamma on all observed entries. It requires positive candidate
gammas and an explicit fixed graph. Use an exogenous graph, or construct one
without held-out feature values. A graph built from the full target data can
leak information; the selector cannot infer its provenance. No internal
preprocessing is performed. See [missing_usage.py](../examples/missing_usage.py).

## Feature-sparse clustering

`solve_sparse` first column-centers X to A and solves

```
0.5 ||U-A||_F² + gamma sum_e w_e ||(BU)_e||_2
                + alpha sum_j feature_weights_j ||U[:,j]||_2.
```

The feature-group penalty can set an entire fitted feature column to zero.
Centering is part of this model, not implicit standardization. The result's
`centers` are in centered coordinates, `offset` restores feature locations,
and `feature_norms` reports group magnitudes. Zero alpha reduces to ordinary
convex clustering of the centered data. The strictly convex fidelity gives a
unique solution.

```python
from ssnalclust import SparseConvexClustering
fit = SparseConvexClustering(gamma=.1, alpha=.5).fit(X_complete)
original_coordinates = fit.centers_ + fit.offset_
print(fit.feature_norms_)
```

See [structured_usage.py](../examples/structured_usage.py) for a runnable example.

## Biclustering

`solve_biclustering` solves

```
0.5 ||U-X||_F²
+ gamma_row sum_e row_weights_e ||(B_row U)[e,:]||_2
+ gamma_col sum_f column_weights_f ||(U B_col.T)[:,f]||_2.
```

Both axis graphs are independent; None selects complete unit graphs. This is
a jointly convex matrix problem, not alternating k-means. Transposing the
matrix and exchanging the two graphs/strengths transposes the optimum.
Disabling column fusion reduces to ordinary row clustering.

`ConvexBiclustering.fit` exposes `centers_`, `row_labels_`, `column_labels_`,
`n_row_clusters_`, and `n_column_clusters_`. Numerical row and column labels
use Euclidean centroid thresholds independently.

## Robust and generalized fidelities

`solve_generalized` adds graph fusion to a separable entrywise fidelity.

| loss | Fidelity at natural parameter u and observation x | Fitted mean |
| --- | --- | --- |
| `huber` | Huber_delta(u-x), quadratic near zero and linear beyond delta | u |
| `logistic` | log(1+exp(u)) - x*u | sigmoid(u) |
| `poisson` | exp(u) - x*u, omitting log-factorial constants | exp(u) |

Fusion penalizes differences of **natural parameters**, not fitted means.
For logistic loss, data lie in [0,1]; for Poisson, data are nonnegative.
Logistic and Poisson component checks reject unattained likelihood infima:
an entirely zero/one Bernoulli feature, or entirely zero Poisson feature,
inside a positive-fusion component. At gamma zero, components are individual
observations, so boundary observations have no finite optimum and are rejected.
The solver does not clip these cases into a different model.

Huber loss is not strictly convex everywhere; optimal centroids can be
nonunique. Logistic/Poisson objectives have unique finite solutions under
the implemented component conditions, but are not globally strongly convex.

```python
from ssnalclust import GeneralizedConvexClustering
fit = GeneralizedConvexClustering(loss="poisson", gamma=.2).fit(count_matrix)
natural_parameters = fit.centers_
expected_counts = fit.fitted_means_
```

See [generalized_usage.py](../examples/generalized_usage.py) for runnable
Huber, binary, and count examples.

## Numerical certificates

For any stacked block K with norm penalty p and feasible dual Z, the solvers
check the stationarity equation `gradient_f(U) + K.T Z = 0` and the proximal
inclusion `KU - prox_p(KU + Z) = 0`. With several blocks, every inclusion is
checked. Each norm-ball dual is feasible by construction. Residuals are
normalized to be scale-aware, and `tol` applies to their maximum. These are
numerical first-order conditions for the specified convex objective; they
are not universal bounds on centroid error for nonstrongly convex losses.
See [structured certificates](structured_certificates.md) and
[generalized certificates](generalized_certificates.md), and
[missing-data certificates](missing_certificates.md) for the additional
feasible duals and stable gap calculations. Global scaling can yield a weak
likelihood dual bound; a small KKT residual alone then does not imply success.

## References

The models are motivated by the [Chi et al. survey](https://arxiv.org/html/2507.09077v2),
particularly its missing-data, high-dimensional, and extension sections.
The PDHG update follows Chambolle and Pock (2011), *A First-Order Primal-Dual
Algorithm for Convex Problems with Applications to Imaging*,
[preprint](https://www.cmap.polytechnique.fr/preprint/repository/685.pdf).
The implementation is independently derived and checked against CVXPY; it is
not a reproduction claim for every named research implementation in the survey.
