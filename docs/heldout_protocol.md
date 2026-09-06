# Wine held-out-entry selection protocol

This protocol is fixed before fitting the full Wine data. It follows the
entrywise prediction idea in Chi, Allen and Baraniuk, *Convex Biclustering*,
[§5.1](https://arxiv.org/pdf/1408.0856v4), with explicitly specified one-way
clustering, preprocessing and graph choices. It does not reproduce their
COBRA algorithm, add bias correction, or estimate a unique imputation when
the masked objective has multiple minimizers.

## Data, masks and fitting inputs

Use all 178 rows and 13 features of the original UCI `wine.data`, in file order.
Separate the first-column cultivar label before any preprocessing. Source
files, hashes and licensing are in the [data attribution](path:../examples/data/README.md).
Do not use labels in splitting, preprocessing, graph construction, candidate
selection or the decision to retain an outcome.

Use three prespecified independent holdout draws, seeds 1729, 1730 and 1731.
For each seed, NumPy's default generator independently permutes row indices
for each feature in original column order and holds out the first
`floor(0.1*n_samples)` entries. Thus each full-data draw holds out 17 entries
per feature (221 total) and retains 161 training entries per feature. Draws
may overlap; these are repeated tuning splits, not disjoint cross-validation
folds or independent test estimates. No outcome-dependent mask replacement.

Physically erase the held-out entries before calculating any statistics.
Compute each feature's mean and population standard deviation from its
training entries only. A training-constant feature uses scale 1. Reject a
feature with no training observations. Standardize observed training entries;
keep held-out entries missing in the optimization matrix. For graph construction
only, fill missing standardized entries with zero (the training feature mean).

Build a symmetric union 10-neighbor graph using exact Euclidean distances,
stable row-index tie handling, and median positive undirected retained-edge
distance as bandwidth. Set weights to `exp(-0.5*(distance/bandwidth)**2)`.
Reject undefined bandwidth and any positive-edge component lacking a training
observation of a feature. No graph repair or extra fidelity observations.
Every candidate within a draw shares this training-derived preprocessing and
graph. Retain masks, statistics, graph hashes and fitting inputs for inspection.

## Candidate fitting and scoring

Candidate factors are `[0.1, 0.3, 1, 3, 10]`, in that order. Within each draw,
`gamma = factor * training_graph_bandwidth / 10`. Factors, rather than absolute
gammas, are compared across draws; each draw's gamma is explicitly recorded.
Use `solve_missing` with l2 fusion, tolerance 1e-6, maximum 30,000 PDHG
iterations and its existing cold-start, observed-range-box convention. No
warm start, new ridge term, centroid debiasing or stopping-rule change.

Score mean squared error on held-out entries in that draw's training-derived
standardized units. This weights Wine's heterogeneous features comparably;
it is not MSE in the original chemical units. Report a training-feature-mean
baseline, whose standardized prediction is zero. Aggregate total squared
errors divided by the total scored entries across all three draws; repeated
entries count each time they are withheld. Do not select a factor unless
all 15 prescribed fits converge and all scores are finite. Ties select the
first factor. Retain every candidate outcome, including failures.

The masked loss need not determine unique missing coordinates. Its certificate
bounds objective error and checks the original KKT conditions, not prediction
error. The reconstruction scores evaluate the deterministic bounded solver
convention and may depend on that convention. Do not interpret the smallest
tuning score as unbiased performance or evidence of cultivar recovery.

## Full-data refit and posthoc evaluation

After fixing the selected factor, recompute feature scaling and the graph
using all original feature entries. Fit all five full-data strengths using
SSNAL warm starts, tolerance 1e-6, maximum 300 outer iterations,
`check_every=1`, `store_history=False`. Labels connect all centroid pairs at
distance at most 1e-4 transitively. Preserve all point certificates, centers,
partitions and posthoc ARIs. The reported selected full-data result must
converge. Its gamma uses the full-data bandwidth and the saved selected factor.

If selection is incomplete, preserve that status and still report the fixed
full-data path as descriptive evidence with no selected point. Ground-truth
labels must never replace an incomplete selection or select another factor.
No thresholding to force three groups. No second graph or extended grid added
in response to revealed labels.

## Execution and retained evidence

Run one fresh worker per holdout draw, then one full-data path worker,
sequentially. Request OMP, OpenBLAS, MKL, vecLib, NumExpr and BLIS thread counts
of one before numerical imports. Each worker has a 900-second deadline.
Preserve process status, timeouts, completed-point checkpoints, versions,
source hashes, requested threads, native peak RSS where available and timing
scope. Do not rerun a timeout with a longer deadline as though it satisfied
the original protocol.

Use JSON metadata and NPZ arrays for masks, scaling, graph CSR arrays, centers
and edge duals. Retain held-out targets separately from fitting arrays so the
saved artifacts can audit leakage and recompute objectives, box certificates,
KKT diagnostics, reconstruction scores and posthoc labels. A smoke mode uses
the first 24 rows, one draw and factors 0.1 and 10; it is excluded from full
scientific evidence and must not be used to tune this full protocol.
