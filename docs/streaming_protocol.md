# Streaming-path memory protocol

This protocol fixes a path-retention comparison before the full measurements.
It addresses the open memory question in the [high-dimensional study](high_dimensional.md).
It does not evaluate model selection or establish a portable memory bound.

## Fixed problem and solver

Generate 512 observations in 32 features from four balanced Gaussian groups
using NumPy's default random generator with seed 7301. Generate four means
as standard normals multiplied by 2, then add independent normal noise of
standard deviation 0.35 to their observations. Retain original row order.
Build a symmetric union 10-neighbor graph from exact Euclidean distances,
using row index to break distance ties. Set the bandwidth to the median
positive undirected retained-edge distance and weights to
`exp(-0.5 * (distance / bandwidth)**2)`. The report must preserve feature and
graph hashes, edge count, components, degree and weight summaries.

For each length 8, 64 and 256, use `geomspace(0.005, 0.5, length)` in increasing
order. Fit squared-loss l2 convex clustering with SSNAL, unit fidelity masses,
`tol=1e-6`, `max_iter=300`, `check_every=1` and `store_history=False`.
Each path uses the same prepared-problem warm-start semantics.

Compare the actual public `ConvexClusteringProblem.path` method, which returns
a retained list, against `iter_path`, whose consumer records scalar summaries
and hashes then discards each result. The materialized measurement must retain
the returned list through final memory observation. The streaming consumer
must not accumulate centroids, duals, labels, histories or result objects.
Scalar records and the fixed gamma grid may grow with length; their cost is
part of the measured workflow and is not claimed to be constant.

## Execution and interpretation

Run six fresh worker processes sequentially, each with a 120-second deadline.
Set OMP, OpenBLAS, MKL, vecLib, NumExpr and BLIS thread environment variables
to one before numerical imports. Record process wall time, worker timing
scope, environment versions, source hashes and requested thread settings.

Observe native process peak RSS after graph/problem preparation and at final
completion (and after available point summaries). RSS is a high-water mark,
not current live memory; allocator retention, imports and temporary arrays
contribute. Report actual returned center/dual storage in bytes separately.
No absolute RSS threshold is a correctness test. Sparse factorization is not
used by this SSNAL comparison, and the results do not characterize ADMM fill-in.

Preserve every point's convergence, gap, KKT residual, objective and center
error bound. Compare center and dual hashes and scalar results between the
two modes at equal length, including all unconverged outcomes. Timing or
memory differences do not excuse mismatched numerical results. Include
centroid displacement and direct edge fusion to distinguish meaningful
optimization from an effectively zero penalty.

A timed-out worker is a timeout, not a successful or merely unconverged path.
The streaming worker can checkpoint completed scalar point records. The public
materialized method cannot expose intermediate returned points; its timeout
record must state that limitation. Do not relaunch a timed-out full run with
more time while presenting it as the original protocol.

The first measurement is descriptive, with one run per setting. This compares
path-length scaling for a fixed modest graph; it does not establish peak-memory
behavior for all sample sizes, feature dimensions or graph families. The
40-sample, 3-feature, length-3/5 smoke mode tests the harness only and is excluded
from full evidence.
