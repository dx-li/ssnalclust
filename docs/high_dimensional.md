# High-dimensional fits and materialized paths

The earlier 50,000-observation Newton measurement used three features. This
study varies feature count and includes three-point regularization paths. It
measures computational behavior on specified Gaussian graph problems, not
statistical recovery or parity with another implementation. Independent
Gaussian observations have no planted cluster labels.

## Recorded results

Raw observations are retained in [high_dimensional_results.jsonl](high_dimensional_results.jsonl).
The 20 runs used macOS arm64, Python 3.12.5, NumPy 2.5.2, SciPy 1.18.1
and scikit-learn 1.9.0. They share one harness hash and the same package-source
hashes. The measured harness is preserved at
[commit 58b0353](https://github.com/dx-li/ssnalclust/blob/58b035331ffe8e6a38f14e6bdef03d3cd55f901e/examples/scalability.py).
Every completed solution passed both requested stopping conditions.
All 20 workers completed within their deadlines. No errors, timeouts or
unconverged points were discarded.

For fits, gamma is shown as a multiple of `sqrt(p)`. Times include graph
construction and the fit pipeline, excluding separately timed evidence
summaries and interpreter startup. Peak RSS includes the whole worker.
The first row uses bandwidth one; every other row uses `sqrt(p)`.

| n × p | Gamma factor | Solver | Graph + pipeline (s) | Peak RSS (MB) | Clusters | Direct fused edges | Result |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1000 × 128 | 0.10 | SSNAL | 0.12 | 242.8 | 1000 | 0.00% | Converged |
| 1000 × 32 | 0.10 | SSNAL | 0.18 | 187.5 | 1000 | 0.00% | Converged |
| 1000 × 128 | 0.10 | SSNAL | 0.75 | 270.4 | 1000 | 0.00% | Converged |
| 1000 × 512 | 0.10 | SSNAL | 2.69 | 567.1 | 1000 | 0.00% | Converged |
| 1000 × 32 | 0.10 | ADMM | 2.36 | 218.3 | 1000 | 0.00% | Converged |
| 1000 × 128 | 0.10 | ADMM | 11.55 | 270.3 | 1000 | 0.00% | Converged |
| 1000 × 512 | 0.10 | ADMM | 50.52 | 600.3 | 1000 | 0.00% | Converged |
| 5000 × 128 | 0.10 | SSNAL | 4.80 | 653.6 | 5000 | 0.00% | Converged |
| 1000 × 128 | 0.20 | SSNAL | 3.37 | 278.7 | 1000 | 0.00% | Converged |
| 1000 × 128 | 0.20 | ADMM | 7.32 | 267.7 | 1000 | 0.00% | Converged |
| 1000 × 128 | 0.30 | SSNAL | 3.79 | 265.7 | 5 | 99.53% | Converged |
| 1000 × 128 | 0.30 | ADMM | 5.96 | 270.5 | 5 | 99.53% | Converged |
| 1000 × 128 | 0.25 | SSNAL | 8.55 | 281.6 | 65 | 92.52% | Converged |
| 1000 × 128 | 0.25 | ADMM | 8.67 | 270.6 | 65 | 92.52% | Converged |
| 5000 × 128 | 0.25 | SSNAL | 43.05 | 709.3 | 113 | 97.48% | Converged |
| 5000 × 128 | 0.25 | ADMM | 117.94 | 944.3 | 113 | 97.48% | Converged |

The unfused scaled-bandwidth fits at gamma factor 0.1 move centroids by
about 46–48% of the centered input Frobenius norm at 1,000 observations,
yet retain 1,000 numerical clusters. At 128 features, factor 0.2 also fuses
no edges; factors 0.25 and 0.3 yield 65 and 5 labels respectively at 1,000
observations. These are numerical partitions of a single Gaussian cloud,
not recovery of that many underlying populations. The midpoint is already
near complete collapse; this is not a dense survey of fusion transitions.

At 5,000 observations and gamma factor 0.25, both methods return 113
labels and 97.48% directly fused edges. These records establish matching
counts, not identical label memberships: full label arrays are not retained.
SSNAL takes 38.92 seconds in the
solve stage; ADMM takes 113.90 seconds. ADMM completes the full process at
119.04 seconds, leaving less than one second before the deadline. That
observed completion does not establish a reliable 120-second budget on
other runs or machines.

The bandwidth-one control has median positive edge weight `5.79e-42`,
initial gap `4.10e-28`, zero SSNAL iterations, and relative centroid
movement `1.49e-17`. Its fast return is deliberately excluded from any
interpretation of meaningful high-dimensional fusion performance.

All four materialized paths below reach cluster counts `1000 → 1000 → 1`;
direct fused-edge fractions are `0 → 0 → 1`. Each uses the three strengths
listed in the protocol. Accuracy columns report the worst value across
all three points.

| Features | Solver | Graph + pipeline (s) | Peak RSS (MB) | Max relative gap | Max KKT | Max centroid bound |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | SSNAL | 0.52 | 204.6 | 2.16e-07 | 3.97e-07 | 0.118 |
| 128 | SSNAL | 1.68 | 304.4 | 9.57e-08 | 2.25e-07 | 0.156 |
| 512 | SSNAL | 6.46 | 664.0 | 8.73e-08 | 6.22e-07 | 0.298 |
| 128 | ADMM | 91.88 | 262.5 | 1.71e-07 | 9.91e-07 | 0.209 |

The fully fused endpoint absolute gaps are approximately 0.00693, 0.01223,
and 0.04454 for SSNAL at 32, 128, and 512 features; the ADMM endpoint at
128 features has gap 0.02184. The reported centroid bounds are global
Frobenius bounds, not per-entry errors. Meeting a normalized `1e-6` stopping
criterion does **not** establish `1e-6` absolute centroid precision.

Matched completed SSNAL/ADMM solutions have objective differences smaller
than their summed absolute gaps. This is a consistency check using the
library certificates, not a new independent optimization oracle. SSNAL is
faster in the recorded unfused fits, but the 1,000-observation midpoint
runtimes are similar. Data geometry, fusion regime, graph structure and
achieved accuracy matter; these single-host observations do not establish
a universal speed or memory advantage. Broader real-data graphs, repeated
runs, additional implementations and streaming-path memory remain open.

## Prevent a trivial high-dimensional comparison

The graph uses Gaussian union kNN weights
`exp(-distance**2 / (2 * bandwidth**2))`, with ten neighbors. Holding bandwidth
at one while increasing the number of independent unit-variance features can
make nearly every retained weight negligible. Convergence near the original
observations then says little about the cost of meaningful fusion.

The main grid therefore sets bandwidth to `sqrt(n_features)` and fit gamma
to `0.1 * sqrt(n_features)`. Path strengths are `[0.01, 0.1, 1]` times that
square root. This convention avoids gross weight collapse; it does **not**
equalize conditioning, fusion fraction, or problem difficulty across sizes.
Actual graph weights, weighted degrees, components and centroid movement are
recorded. A bandwidth-one case is retained as a negative control.

## Protocol and reproduction

Each scenario runs in a separate process with a 120-second deadline including
startup and imports. SSNAL receives at most 100 outer iterations, ADMM at most
2,000 iterations. These iteration units are different; both use tolerance
`1e-6`, diagnostic cadence one, label tolerance `1e-4`, seed 1729, and one
requested numerical-library thread. Single observations do not estimate
runtime variability. Other host activity and effective native thread counts
are not controlled.

After inspecting the unfused fit regime, additional fits use 128
features and gamma factors `0.2` and `0.3`, with both methods and the same
data, graph, accuracy target and deadline. A subsequent midpoint at factor
`0.25` is evaluated with both methods at 1,000 and 5,000 observations. This is an explicit diagnostic
follow-up to seek partial fusion, not a preregistered grid. All those results
are retained.

The path API retains all solutions and histories. Its peak memory describes
a **materialized path**, not the streaming interface. Every returned point
must be checked separately. A `solution_complete` event from a timed-out
worker establishes only that individual solve's return, not completion of
the whole pipeline, labeling, or the path.

From an installed checkout, use a new output filename to keep reproduced
observations separate from the checked-in evidence:

```python
import math
import subprocess
import sys

cases = [("ssnal", "fit", 1000, 128, 1.0)]  # Near-unregularized control
for solver in ["ssnal", "admm"]:
    cases += [(solver, "fit", 1000, p, math.sqrt(p)) for p in [32, 128, 512]]
cases += [("ssnal", "path", 1000, p, math.sqrt(p)) for p in [32, 128, 512]]
cases += [("admm", "path", 1000, 128, math.sqrt(128))]
cases += [("ssnal", "fit", 5000, 128, math.sqrt(128))]
for solver, scenario, n, p, bandwidth in cases:
    subprocess.run([
        sys.executable, "examples/scalability.py", "--samples", str(n),
        "--features", str(p), "--solver", solver, "--scenario", scenario,
        "--data", "gaussian", "--bandwidth", str(bandwidth),
        "--gamma", str(0.1 * math.sqrt(p)), "--path-gammas",
        *[str(c * math.sqrt(p)) for c in [0.01, 0.1, 1.0]],
        "--neighbors", "10", "--tol", "1e-6", "--max-iter",
        "100" if solver == "ssnal" else "2000", "--timeout", "120",
        "--threads", "1", "--seed", "1729",
        "--output", "build/high_dimensional_reproduction.jsonl",
    ], check=True)

# Follow-up strengths to probe the transition regime.
for factor in [0.2, 0.3]:
    for solver in ["ssnal", "admm"]:
        subprocess.run([
            sys.executable, "examples/scalability.py", "--samples", "1000",
            "--features", "128", "--solver", solver, "--scenario", "fit",
            "--data", "gaussian", "--bandwidth", str(math.sqrt(128)),
            "--gamma", str(factor * math.sqrt(128)), "--neighbors", "10",
            "--tol", "1e-6", "--max-iter", "100" if solver == "ssnal" else "2000",
            "--timeout", "120", "--threads", "1", "--seed", "1729",
            "--output", "build/high_dimensional_reproduction.jsonl",
        ], check=True)
for n in [1000, 5000]:
    for solver in ["ssnal", "admm"]:
        subprocess.run([
            sys.executable, "examples/scalability.py", "--samples", str(n),
            "--features", "128", "--solver", solver, "--scenario", "fit",
            "--data", "gaussian", "--bandwidth", str(math.sqrt(128)),
            "--gamma", str(0.25 * math.sqrt(128)), "--neighbors", "10",
            "--tol", "1e-6", "--max-iter", "100" if solver == "ssnal" else "2000",
            "--timeout", "120", "--threads", "1", "--seed", "1729",
            "--output", "build/high_dimensional_reproduction.jsonl",
        ], check=True)
```

The parent records numerical nonconvergence, worker errors and timeouts in
JSON; a successful launcher exit is not proof of successful optimization.
See [the scalability protocol](scalability.md) for timing boundaries, nested
stage counters, source hashes and native peak-RSS semantics.

## Additional evidence fields

Graph summaries report positive weight and weighted-degree minimum, median
and maximum, connected-component count, and isolated vertices. Solution
summaries include the absolute gap and the initial feasible primal-dual gap
at centroids equal to the input and zero dual. For this objective the latter
is `gamma * sum_edges(weight * input_edge_distance)`.

Centroid movement is `||U-X||_F`, also divided by `||X-mean(X)||_F`. For a
zero denominator the normalized movement is zero when the solution is
unchanged, otherwise null with an explicit normalization status. The direct
fused-edge fraction counts positive graph edges whose centroid distance is
at most the labeling tolerance. It does not count all endpoints that merely
share a transitive numerical label. A graph without edges has a null
fraction. Edge-distance calculations use bounded batches.

These additional summaries run outside the optimization timer. Their stage
and combined evidence times are recorded separately; total process time and
peak RSS still include their overhead.
