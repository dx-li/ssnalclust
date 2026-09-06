# Measured Newton operator optimization

This experiment isolates a change to the matrix-free generalized Hessian
used by SSNAL's conjugate-gradient inner solve. It compares an immutable
baseline checkout with the candidate source using the same synthetic inputs,
graph settings, stopping tolerance and instrumented benchmark harness.

## Baseline and measurement protocol

The baseline is commit `48c73749189974538f05dd90a239978d9351b6f1`, checked out
detached into `/tmp/ssnalclust-newton-baseline-48c7374`. It contains the earlier
adaptive inner tolerance and stable gap evaluation. Its `solvers.py` SHA-256
is `7d8b17106ea406a50bd211685dc3ed26216c1826cdc7307f7126d5ae53fb5a93`.

The shared virtual environment has an editable installation of the main
checkout. To prevent that installation from contaminating the baseline,
`PYTHONPATH` explicitly points to the detached checkout's `src` directory.
The imported package's `__file__` was checked before launching workers.
Each record also includes source-file hashes, including the solver source.

The full estimator scenarios use 1,000, 5,000 and 10,000 standard Gaussian
observations with three features and seed 1729. Graphs are union kNN with
10 neighbors and bandwidth 1. Gamma is 0.5, tolerance 1e-6, `check_every=1`,
and the outer iteration limit is 60. A fresh process handles each scenario
with a 90-second deadline and one numerical thread requested before imports.
Timeouts are preserved as timeouts rather than restarted or labeled
unconverged completions. These runs use the instrumentation and native RSS
definitions in [scalability.md](scalability.md).

The baseline command, from its detached checkout, is:

```bash
PYTHONPATH=/tmp/ssnalclust-newton-baseline-48c7374/src \
  /absolute/path/to/shared/.venv/bin/python examples/scalability.py \
  --samples 1000 5000 10000 --features 3 --solver ssnal \
  --gamma .5 --neighbors 10 --bandwidth 1 --tol 1e-6 \
  --max-iter 60 --check-every 1 --seed 1729 --threads 1 --timeout 90 \
  --output /absolute/path/to/main/docs/newton_performance_results.jsonl
```

The candidate uses the same command and shared interpreter with `PYTHONPATH`
changed to its own `src` directory. Raw records are appended to
`newton_performance_results.jsonl`; their hashes identify the version rather
than their position in the file. Timing differences are observations on this
host, not universal performance claims. Parent development tests can share
the host, but the benchmark workers themselves run sequentially; no competing
large benchmark was intentionally launched.

## Matched pipeline results

The candidate solver hash is
`e66555b1de384d0c54c2bd0af77f816fcf8b499d540d562cc087e2b7139470d1`.
All six full-pipeline workers completed within their deadlines and every fit
converged. The algebra and numerical-range tests are documented separately in
[newton_operator.md](newton_operator.md).

| Samples | Baseline solve seconds | Candidate solve seconds | Baseline / candidate CG seconds | Baseline / candidate CG iterations | Baseline / candidate Newton solves |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 2.153 | 1.091 | 1.673 / 0.616 | 3,146 / 3,135 | 126 / 126 |
| 5,000 | 18.571 | 7.916 | 15.712 / 5.160 | 6,293 / 6,289 | 170 / 170 |
| 10,000 | 34.904 | 15.318 | 29.469 / 9.894 | 6,013 / 6,037 | 170 / 171 |

The outer iteration counts were 44, 49 and 49 respectively for both versions.
The observed solve-time ratios are approximately 1.97, 2.35 and 2.28. These
ratios describe single runs on this host, not a guaranteed general speedup.
Arithmetic reordering slightly changed CG and, at 10,000 samples, Newton
counts; this is not a comparison of bit-identical iteration trajectories.

| Samples | Candidate objective | Candidate relative gap | Candidate KKT residual | Candidate center error bound | Baseline / candidate peak RSS, MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 794.5664900278 | 1.896e-8 | 9.841e-7 | 0.0077645 | 147.1 / 147.2 |
| 5,000 | 2756.6030825500 | 1.742e-8 | 8.507e-7 | 0.0138601 | 168.5 / 166.9 |
| 10,000 | 4561.5496900174 | 2.416e-8 | 9.116e-7 | 0.0209959 | 194.8 / 193.4 |

Baseline and candidate objectives differ by less than 6e-11 absolute on each
case; their reported gaps and KKT residuals are also closely aligned. These
results preserve achieved optimization accuracy while showing less time in
CG. Native peak RSS includes the interpreter and imports, so small differences
must not be interpreted as precise solver-allocation savings.

## Isolated operator measurements

`examples/newton_microbenchmark.py` constructs a 5,000-node circulant graph
connecting offsets 1 through 10 in both directions, giving 50,000 undirected
edges. It uses seed 317, sigma=2, fidelity masses sampled uniformly from
[0.5, 2], and radii alternating between twice and half the corresponding
projection-input norm. Exactly half the rows are outside the dual balls.
The active mask and operator are fixed during timing.

Every source/feature case runs in a fresh process with a 60-second deadline
and one numerical thread requested. After ten warm-up calls it records five
batches of 100 applications of the operator to a fixed random direction.
The reported number is median batch time divided by 100. These five batches
are repeated calls within a process, not independent host-level replicates.
Construction time and memory are recorded separately in the raw JSONL.

Before timing, each worker independently assembles a small dense generalized
Hessian from edgewise Jacobian blocks. Both the matrix action and inverse
Jacobi diagonal must match the production operator. The dense-oracle maximum
absolute errors were 3.55e-15 for three features and 8.88e-15 for twenty
features for both source versions. Large-operator output norms also matched
between versions: 3565.5426755 and 9986.8382684 respectively.

| Features | Baseline median ms/matvec | Candidate median ms/matvec | Baseline / candidate native peak RSS, MB |
| --- | ---: | ---: | ---: |
| 3 | 3.032 | 1.250 | 166.2 / 163.3 |
| 20 | 7.451 | 3.185 | 214.3 / 206.5 |

Reproduce using actual checkout paths:

```bash
python examples/newton_microbenchmark.py \
  --source /baseline/checkout/src /candidate/checkout/src \
  --samples 5000 --features 3 20 --sigma 2 --seed 317 \
  --calls-per-round 100 --rounds 5 --timeout 60 \
  --output docs/newton_performance_results.jsonl
```

The driver enforces each source path using `PYTHONPATH` and checks the imported
module location. Operator records have `kind="matvec_microbenchmark"`; the
other JSONL entries are the full pipeline records. A mixed 50% active mask on
one graph and repeated fixed-vector multiplication does not cover all Newton
subproblem regimes. The full pipeline experiments above provide complementary
evidence with naturally changing masks and CG directions.

## Candidate-only 50,000-sample extension

After the matched comparisons, the unchanged candidate source was run on
50,000 Gaussian observations with three features, using the same seed,
graph/gamma/tolerance settings and 60-iteration limit. This worker had a
180-second deadline. It completed in 112.70 process seconds and converged
after 50 outer iterations. No baseline run was attempted at this size, so
there is **no baseline speed ratio for 50,000 samples**.

| Measurement | Observed value |
| --- | ---: |
| Undirected edges | 292,386 |
| Graph construction | 0.196 s |
| Incidence preparation | 0.076 s |
| Solver | 110.748 s |
| CG within solver | 78.511 s |
| All Newton inner work within solver | 106.685 s |
| Centroid labeling | 0.380 s |
| Graph + complete estimator pipeline | 111.337 s |
| Native process peak RSS | 330.334 MB |
| Newton solves / CG iterations | 201 / 9,165 |
| Primal objective | 13925.515329285203 |
| Relative gap | 5.1542e-8 |
| KKT residual | 9.7318e-7 |
| Weighted Frobenius center error bound | 0.0535825 |

These measurements extend demonstrated feasibility beyond 10,000 samples for
one sparse, low-dimensional input family. They do not establish performance
for dense graphs, high feature counts, long paths, or all regularization
regimes. Inner Newton work remains the dominant measured cost, even after
the operator improvement. Nested timers must not be added together.
