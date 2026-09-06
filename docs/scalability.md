# Reproducible scalability measurements

`examples/scalability.py` benchmarks full estimator fits, regularization paths,
or centroid labeling in a fresh Python worker for every scenario. Results are
JSONL; the checked-in observations are in `scalability_results.jsonl`. These
measurements characterize this implementation on one machine. They do not
establish parity with the paper's MATLAB implementation or a speed advantage
over another library.

## Reproduce

From an installed checkout:

```bash
python examples/scalability.py --samples 1000 5000 10000 --features 3 \
  --solver ssnal --max-iter 60 --timeout 40 --threads 1 \
  --output docs/my_scalability_results.jsonl
python examples/scalability.py --samples 1000 5000 10000 --features 3 \
  --solver admm --max-iter 60 --timeout 40 --threads 1 \
  --output docs/my_scalability_results.jsonl
python examples/scalability.py --samples 1000 --scenario path \
  --solver admm --path-gammas .1 .5 1 --max-iter 2000 --check-every 10 --timeout 40
python examples/scalability.py --samples 1000 5000 10000 --scenario labels \
  --data duplicates --timeout 40
```

`--repeat` repeats a scenario using the same seed, each in a new process.
`--output` appends records and never overwrites earlier measurements. The
default input is independent standard Gaussian data with seed 1729. Optional
`blobs` uses five separated groups with noise; `duplicates` uses five exactly
repeated points. Graphs are symmetric Gaussian union kNN graphs (k=10,
bandwidth=1 by default). Each fit receives the explicit graph and includes
validation, optimization, labeling and cluster-mean publication. Graph creation
is timed separately and included in `graph_plus_pipeline_seconds`.

## What is measured

The harness supports macOS and Linux. It uses `resource.getrusage`'s native
process peak resident set size (RSS), converting the platform's units to
bytes. RSS includes the Python interpreter, imported packages, sparse
factorizations, native numerical allocations and benchmark overhead. It is
not an isolated allocation measurement for the solver. Fresh processes avoid
carrying a previous scenario's high-water mark into the next one.

The parent enforces a wall-clock deadline including imports and setup. A
`timeout` record means that the particular worker was killed and reaped; it is
not a completed solve with `converged=False`. Successful worker termination
is recorded as `completed` even if the solver exhausted its iteration limit.
Inspect `solutions`, `all_converged`, relative gap and KKT residual together.
An `error` is recorded separately with its exception and traceback.

Completed workers report `peak_rss_bytes` and `peak_rss_complete=True`.
Heartbeat samples provide a lower bound, `peak_rss_observed_bytes`, if the
worker times out; `peak_rss_complete=False` explicitly marks that limitation.
Partial stage timers count only calls that finished before the latest sample.

`process_wall_seconds` includes interpreter startup, imports and shutdown.
`pipeline_seconds` excludes graph construction; `worker_seconds` includes
imports and dataset construction inside the worker. Stage timers are nested:
Newton time is part of solve time, CG time is part of Newton time, and solve
time is part of fit time. **Do not sum nested stage timings.** The harness
wraps the actual functions and counts Newton operator constructions, reduced
objective/gradient evaluations and CG callback iterations. Instrumentation has
some overhead. The earliest records predate the additional CG counters; absent
fields mean unmeasured, not zero.

Every record contains versions, platform, source-file SHA-256 hashes and the
requested numerical thread environment. Newer records also identify the
harness hash. Environment variables request a single thread by default before
numerical packages are imported. They are not a measurement of effective
thread counts. The worker is process-isolated, but other activity on the host,
thermal state and filesystem caches are not controlled.

The [high-dimensional study](high_dimensional.md) extends this protocol with
explicit graph-weight, centroid-movement and direct-edge-fusion diagnostics.
It includes a near-unregularized control to show why fixed bandwidth can
produce misleadingly easy problems as feature count grows.

## Initial observations

The first grid ran on macOS 26.5.1 arm64, Python 3.12.5, NumPy 2.5.2,
SciPy 1.18.1 and scikit-learn 1.9.0, with one numerical thread requested.
Inputs had three features, gamma=0.5, k=10, tolerance=1e-6 and at most 60
iterations. These are single observations, not statistical timing estimates.

| Method | Samples | Graph + pipeline (s) | Native peak RSS (MB) | Result |
| --- | ---: | ---: | ---: | --- |
| SSNAL | 1,000 | 5.37 | 119.4 | Converged, 44 outer iterations |
| SSNAL | 5,000 | 38.03 | 139.1 | Converged, 49 outer iterations |
| SSNAL | 10,000 | — | — | Worker timeout at 40 seconds |
| ADMM | 1,000 | 0.18 | 119.0 | Unconverged after 60 iterations; KKT 2.99e-4 |
| ADMM | 5,000 | 1.14 | 166.8 | Unconverged after 60 iterations; KKT 4.37e-4 |
| ADMM | 10,000 | 3.30 | 312.3 | Unconverged after 60 iterations; KKT 5.09e-4 |

ADMM's short runs did not achieve the requested accuracy, so this table does
**not** show an accuracy-matched speed advantage. The SSNAL timeout is neither
an iteration-exhaustion result nor a completed memory measurement.

For the first SSNAL 5,000-sample run, graph construction took 0.020 seconds,
labeling 0.030 seconds and solving 37.98 seconds. Inner Newton solves accounted
for 37.62 seconds. A subsequent instrumented 1,000-sample baseline counted 44
outer iterations, 224 Newton/CG solves, 8,388 CG iterations and 498 reduced
objective/gradient evaluations. CG consumed 4.77 of 5.58 solve seconds. This
supports prioritizing inner-solve work on this input family; it does not imply
graph construction or labeling is negligible on every dataset.

The source hashes distinguish measurements made before and after subsequent
changes. A performance comparison should hold data, requested tolerance,
thread settings and achieved convergence fixed, and should use repeated runs
before claiming a stable improvement.

## Adaptive Newton tolerance comparison

The baseline forced a tight absolute inner tolerance from the first outer
iteration. The revised rule scales the inner tolerance with the current
stationarity scale and KKT residual, retaining a summable outer-iteration cap.
The following runs used identical inputs and requested tolerance. Baseline
`solvers.py` SHA-256 begins `760b4aa6dde8`; revised SHA-256 begins
`a5f349dad39c`. Full hashes are recorded with each measurement.

| Samples | Baseline solve (s) | Revised solve (s) | Baseline / revised Newton solves | Baseline / revised CG iterations | Outer iterations, both |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 5.58 | 2.27 | 224 / 126 | 8,388 / 3,146 | 44 |
| 5,000 | 38.74 | 19.61 | 280 / 170 | 13,231 / 6,293 | 49 |

Both versions converged. At 5,000 samples the final KKT residual was about
8.51e-7 and relative gap about 1.74e-8 for both; objective values agreed to
better than 1e-9 absolute. At 1,000 samples the KKT residual was about 9.84e-7
for both. These single-run observations support the reduction in redundant
CG work; repeated timings on more graph and feature families remain necessary
before presenting a general speedup claim.

The revised 10,000-sample run completed in 37.08 solve seconds (37.18 seconds
including graph construction and estimator publication), using 165.1 MB native
peak RSS. It took 49 outer iterations, 170 Newton solves and 6,013 CG
iterations, with KKT residual 9.12e-7 and relative gap 2.42e-8. The graph had
59,622 undirected edges. The baseline's 40-second worker timeout supplies
only a lower bound on its process runtime, so no precise 10,000-sample speedup
ratio can be inferred.

## Other observed costs

A dedicated labeling probe used five exactly repeated locations. It returned
five clusters at every size. Labeling took 0.0115, 0.1834 and 0.7329 seconds
at 1,000, 5,000 and 10,000 observations respectively. The fourfold increase
from 5,000 to 10,000 is consistent with repeated dense-neighborhood queries
in the baseline traversal, even though the full radius graph is never stored.
Graph generation on these duplicate inputs took 0.0030, 0.0376 and 0.1366
seconds. This contrasts with the Gaussian fit measurements; favorable label
timings there do not establish scalability for fused or duplicated data.

An ADMM path at 1,000 samples, gammas `[0.1, 0.5, 1.0]`, tolerance 1e-6 and
at most 2,000 iterations per point took 10.34 seconds. Instrumentation observed
one incidence construction and one factorization across three prepared
solves, confirming that the path reused those structures. The first two
points converged after 147 and 1,942 iterations. The third exhausted 2,000
iterations at KKT residual 3.25e-6, so the entire path cannot be called
converged. Diagnostics consumed 4.76 seconds over 4,095 evaluations; the one
factorization took 0.0085 seconds. Reuse is useful infrastructure but is not
the dominant runtime saving on this particular small sparse graph.

## Final focused checks

The final labeling implementation collapses exactly duplicate rows before
traversing neighborhoods. With `estimator.py` hash prefix `6c50ca56d765`, the
same five-location probes took 0.000774, 0.002808 and 0.005909 seconds at
1,000, 5,000 and 10,000 samples. All still returned five clusters. The harness
counts recursive labeling calls but records their outer elapsed time only,
avoiding double-counting the deduplicated recursive traversal. This verifies
the specific exact-duplicate improvement; near-duplicates or other dense
radius neighborhoods can have different costs. Duplicate-point kNN graph
construction remained about 0.137 seconds at 10,000 samples.

The ADMM path was repeated with `check_every=10` instead of 1. With final
`solvers.py` hash prefix `21f0e012e097`, it took 6.02 path seconds; diagnostics
accounted for 0.49 seconds over 416 calls. Incidence construction and
factorization still occurred once each. The first two path points converged
after 150 and 1,950 iterations; the third remained unconverged after 2,000
iterations at KKT residual 3.25e-6. Checking less often can delay detection of
convergence, as these iteration counts illustrate. Final diagnostics are
always evaluated, including at iteration exhaustion.

The source also changed to a numerically stable gap evaluation between these
path runs, so the timings compare the recorded revisions, not a perfectly
isolated single-parameter experiment. The approximately tenfold reduction in
diagnostic evaluations is directly observed. No claim is made that the whole
path converged or that this cadence is best for every problem. New records
include the squared-loss `center_error_bound` in addition to gap and KKT
residual; earlier records lack that field.

A final SSNAL fit on 5,000 Gaussian samples using this final source hash and
`check_every=1` completed in 18.94 solve seconds (18.99 seconds including graph
creation and estimator publication), with 141.0 MB native peak RSS. It again
took 49 outer iterations, 170 Newton solves and 6,293 CG iterations. The final
KKT residual was 8.51e-7, relative gap 1.74e-8 and weighted Frobenius center
error bound 0.01386. This verifies the final revision on that scenario; the
earlier 10,000-sample measurement used the preceding adaptive-tolerance
revision and must not be relabeled as a measurement of the final hash.

## Subsequent Newton-operator measurements

The [Newton performance study](newton_performance.md) compares a pinned
baseline with the subsequent operator change. Its records have separate
source hashes and must not be conflated with the measurements above.
