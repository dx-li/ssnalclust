# Estimator compatibility and its scientific boundaries

The package uses scikit-learn's estimator conventions for parameter inspection,
cloning, input validation, learned attributes, and pipeline integration. That
protocol does not turn a transductive graph clustering model into an inductive
predictor or make every generic model-selection procedure appropriate.

The reproducible tests are
[`test_estimator.py`](../tests/test_estimator.py),
[`test_model_estimators.py`](../tests/test_model_estimators.py), and
[`test_all_estimator_checks.py`](../tests/test_all_estimator_checks.py).
The audit below was run with Python 3.12.5, scikit-learn 1.9.0, NumPy 2.5.2,
SciPy 1.18.1, and pandas 3.0.5. Check names and counts can change with
scikit-learn versions; the test suite runs the applicable checks at runtime.
The same 35-test all-model conformance suite also passes with Python 3.10.19,
NumPy 1.24.0, SciPy 1.12.0, scikit-learn 1.6.0, and pandas 1.5.3.

## Full generic checks

| Estimator/configuration | Audit result | Interpretation |
| --- | --- | --- |
| `ConvexClustering()` | Generic checks with the documented sample-weight equivalence exception | Positive fidelity masses weight a fixed graph; sample duplication changes the graph problem and zero masses are unsupported |
| `GeneralizedConvexClustering()` with default Huber loss | All protocol checks; default synthetic recovery score exception | Complete-graph gamma 1 fuses the generic check's standardized blobs |
| `GeneralizedConvexClustering(gamma=0.04)` | Full 46-check collection, no model-specific exclusions | Exercises the same protocol including the generic clustering recovery assertion |
| `SparseConvexClustering()` | All protocol checks; default synthetic recovery score exception | Complete-graph gamma 1 fuses the generic check's standardized blobs |
| `SparseConvexClustering(gamma=0.04)` | Full 46-check collection, no model-specific exclusions | Covers all clustering assertions at a smaller fusion strength |
| `MissingConvexClustering`, exogenous graph fixture, gamma 0.04 | Full 45-check collection, no model-specific exclusions | Fixture supplies a complete graph matching each generated dataset's sample count |
| `ConvexBiclustering()` | Full 41-check collection, no model-specific exclusions | Tested as `BaseEstimator`; it describes row and column partitions, rather than one `ClusterMixin` target |

The array-API check is skipped by scikit-learn when `SCIPY_ARRAY_API` is not
configured. This audit does not claim array-API/backend portability. The
pandas-specific direct tests execute in CI because pandas is included in the
`test` and `dev` extras (minimum supported test version 1.5.3). Pandas remains
optional at runtime. A manual test run without those extras explicitly skips
pandas-specific cases when pandas is absent.

### Why the default recovery assertion has an exception

Scikit-learn's `check_clustering` generates 50 standardized observations in
three groups and requires adjusted Rand index above 0.4 at the estimator's
parameters. For the generalized and feature-sparse defaults, a complete graph
with gamma 1 fuses those observations into one group. That is the intended
regularized solution, not evidence of an incorrect fit or label protocol.
These estimators choose a fusion strength rather than a requested number of
clusters.

Only that named check is marked as an expected failure for the defaults.
The **entire** check collection is also run at gamma 0.04 without exclusions,
so list input, repeated fitting, label integer types, consecutive labels, and
all other assertions in `check_clustering` are still exercised. The fixture
strength is a test configuration, not a general gamma recommendation or a
scientific performance benchmark.

### Why missing data needs a graph fixture

The production missing-data estimator deliberately requires an explicit graph.
A fixed adjacency belongs to a particular ordered observation set. Generic
estimator checks vary dataset size, so one fixed constructor adjacency cannot
match every test dataset.

`MissingCompleteGraphFixture` is confined to the tests. During each fit it
supplies an exogenous complete adjacency determined **only by sample count**,
then restores the original constructor parameter. It delegates all validation,
optimization, and result publication to the production estimator. For sparse input, the fixture reads its shape without NumPy coercion, so the
production validator retains responsibility for gracefully rejecting unsupported
sparse observations, including older SciPy sparse-array formats. It does not
infer weights from feature values, modify optimizer behavior, bypass missing
entry validation, or supply a public automatic-graph mode.

Separate tests exercise actual `MissingConvexClustering(weights=W)` directly,
including cloning, explicit masks, singleton identifiability, read-only inputs,
feature names, and nonconvergence. The generic fixture audit should not be
read as a claim that `MissingConvexClustering().fit(X)` accepts an omitted
graph.

## What the interfaces support

`get_params`, `set_params`, and `sklearn.base.clone` use constructor parameters.
Cloning a fitted estimator drops learned state and copies array-valued graph
parameters. Fitted centroids are not transferred into the clone as a warm
start. Use low-level solver warm-start arguments or path APIs for that purpose.

DataFrames with string column names populate `feature_names_in_` and
`n_features_in_`. Refitting on an ordinary array updates metadata and removes
stale feature names. Returned centroids and labels are arrays in input row
order; pandas row indices are not embedded in those arrays. Preserve row IDs
in the surrounding analysis when joining results back to observations.

Read-only observation and adjacency arrays are supported; fitting does not
mutate them. Singleton complete observations are supported across the model
families. A singleton missing entry is rejected as unidentifiable. With a
single row, biclustering can still fuse its columns; feature-sparse clustering
centers that row to zero and retains its original location in `offset_`.

A nonconverged fit emits `ConvergenceWarning` while preserving the actual
`result_`, `centers_`, `objective_`, `n_iter_`, and `converged_`. Warning handling
must not replace inspection of the diagnostics. Low-level functions return
their result without the estimator warning layer.

## What generic integration does not imply

No clustering estimator defines out-of-sample `predict`. `fit_predict` returns
a partition of the observations supplied to that fit; it does not establish
how a new observation would be assigned without refitting.

A fixed graph is indexed by observation order. Reordering rows requires the
same permutation on both adjacency axes. Subsetting rows requires a suitably
defined training graph, and changing feature count in biclustering requires
a compatible column graph. A generic train/test splitter does not automatically
perform those graph operations.

Pipelines can express preprocessing followed by clustering, but preprocessing
choices still define the model. Complete-data standardization changes graph
distances and fidelity units. Logistic/Poisson fitting expects valid response
values and natural-parameter interpretation; arbitrary standardized values do
not represent the same likelihood. Missing-entry validation must keep held-out
values out of preprocessing and graph construction.

Do not infer that the presence of `get_params` makes default `GridSearchCV`
scoring or cross-validation scientifically appropriate. A usable score,
correct graph construction for each split, and a validation target consistent
with transductive clustering are still required. `select_gamma` implements a
specific entry-reconstruction holdout procedure with explicit masks and graph
requirements; it is not an out-of-sample clustering classifier.

## Regression findings tracked during the audit

The audit identified that model-wrapper `cluster_tol` validation occurred only
after optimization. A failed refit could then update generalized or sparse
model-specific attributes before raising, leaving inconsistent learned state.
A strict regression test checks that an invalid threshold is rejected before
calling the optimizer and before changing existing learned attributes. The
wrappers now reject that parameter before optimization, and all four
regression cases pass. This defect was fixed rather than classified as a
scientific exception.

The broader numerical and statistical claims are tested separately against
analytic solutions, convex optimization oracles, and invariants. Passing
estimator checks is evidence about software protocol conformance; it is not
proof of numerical correctness, statistical recovery, or general performance.
