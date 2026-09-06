# Frozen Optical Digits study protocol

This protocol is fixed before full-data solver execution and inspection of
label-agreement results. It extends the survey-motivated validation agenda
with real image structure. It is not a replication of a particular experiment
in the survey and is not a registered clinical or prediction study.

## Data and geometry

Use the original UCI `optdigits.tes` cohort: 1,797 rows, 64 pixel features and
a digit-label column. Keep row order and every feature; divide pixels by 16.
The input SHA-256 is
`6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8`.
[Attribution and provenance](path:../examples/data/README.md) accompany the source
files. The original training cohort is not used.

Fit the specified complete-data squared-fidelity, Euclidean-fusion objective
with unit sample masses. No centering, variance scaling, PCA, feature
selection or supervised representation is added. Build separate symmetric
union kNN graphs for `k=10` and `k=20`. Neighbor distance ties use original
row indices deterministically. Set bandwidth to the median **positive**
Euclidean distance of retained undirected neighbor edges. Zero-distance
edges remain in the graph with unit weight; reject a graph with no positive
retained distance rather than silently changing the bandwidth rule. Use
`exp(-distance**2 / (2 * bandwidth**2))` weights, a zero diagonal, and charge
one penalty per undirected edge.

Report edge counts, weights, degrees, connected components and isolates.
An exact pairwise Euclidean distance cache is reused for graph construction
and silhouette scoring; its quadratic storage is about 26 MB at this cohort
size. This example is not a large-scale nearest-neighbor memory benchmark.

## Fixed strengths and numerical checks

For each graph, set gamma to `median_distance / k` times
`[0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]` in that order. Solve a warm-started
SSNAL path with tolerance `1e-6`, at most 300 outer iterations per point,
diagnostic cadence one and numerical label tolerance `1e-4`. Iteration
histories are disabled; returned centers and per-point summaries are retained. Every strength
has its own convergence flag, absolute/relative gap, KKT residual and global
Frobenius centroid-error bound.

Cross-check the predetermined grid positions 0, 4 and 8 with cold ADMM solves
on exactly the same data and graph, tolerance `1e-6` and at most 10,000
iterations. Each ADMM check uses a fresh prepared problem, including its
factorization; neither iterates nor factorizations are reused between checks.
Record objective differences and feasible certificate intervals,
actual centroid distances versus summed error bounds, and partition equality
allowing relabeling. These compare two implementations in this library;
they are not a new independent external optimizer or proof beyond the
underlying numerical certificates.

Run each graph study in a fresh process, sequentially, with one requested
numerical-library thread and a 900-second process deadline. Keep checkpoint
records and completed arrays if a worker fails or times out. Do not treat
those records as a completed study. Record source/data hashes, versions,
platform, stage/solver timing and native process peak RSS where supported.
Separate evidence, silhouette and checkpoint overhead from solve timing.
This is a materialized path; memory includes its retained solutions.

## Selection before evaluation

Choose one gamma separately for each graph using exact mean Euclidean
silhouette on the **original scaled input features**, not fitted centroids.
Candidate partitions must have `2 <= n_clusters < n_samples` and a finite
score. Singleton groups within an otherwise eligible partition remain
eligible under the standard silhouette definition. Ties choose the earliest
fixed-grid index. No known class count is imposed.

The selection function must not receive digit labels. If any SSNAL path point
is unconverged or absent, selection is incomplete; do not present a best
converged subset as the completed procedure. If every point is numerically
complete but none has a defined silhouette, report selection unavailable.
Do not expand the grid after inspecting labels or force a ten-cluster answer.
ADMM nonconvergence makes its cross-check incomplete, without changing an
otherwise valid frozen silhouette choice.

Only after the selection index is fixed, compute adjusted Rand agreement
with digit labels. Preserve post-hoc agreement at every grid point, not just
the selected one. Do not select a graph or revise preprocessing from that
agreement. Centroid images, cluster sizes and label membership arrays may
illustrate the selected partitions; label-derived annotations are evaluation,
not tuning inputs.

This is transductive clustering of the UCI test-file cohort: every feature
vector participates in the graph and selection criterion. Agreement with
digit identities is not held-out prediction accuracy or proof that the
clusters are the only scientifically valid partition. Silhouette itself
favors particular geometric cluster shapes and sizes. Report these limits,
all failures and the fixed grid alongside numerical accuracy and runtime.

## Smoke tests

A small first-80-row, three-strength run may validate execution and artifact
formats in CI. Its selection and label agreement are excluded from the
full-data scientific evidence. Smoke success does not establish the full
study's numerical or statistical outcome.
