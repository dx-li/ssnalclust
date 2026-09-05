# Recovery on nonconvex synthetic geometry

This study evaluates recovery separately from optimization correctness.
Convergence certificates concern the specified convex objective; they do not
guarantee recovery of a data generator's class labels. Moons and concentric
circles are deliberately challenging geometries for centroid-based methods.

## Fixed protocol

The dataset definitions use scikit-learn's independent generators:
[`make_moons`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
produces two interleaving half-circles, and
[`make_circles`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html)
produces concentric circles. The circle radius factor is fixed at 0.5. Noise
is independent Gaussian coordinate noise with standard deviation 0 or 0.05.
Coordinates are used directly, without data-dependent feature scaling.

The prespecified grid is:

| Setting | Values |
| --- | --- |
| Samples | 100, 300 |
| Seeds | 0, 1, 2 |
| Geometry | Moons, circles |
| Noise standard deviation | 0, 0.05 |
| Gaussian union kNN graph (k, bandwidth) | (5, 0.2), (10, 0.5) |
| Gamma | 0, 0.1, 0.3, 1, 3, 10, 30, 100 |
| Solver | SSNAL, tolerance 1e-6, at most 150 outer iterations per gamma |
| Numerical fusion-label distance threshold | 1e-3 |

Neither graph parameters nor gamma values are chosen using class labels.
The same grid is applied to both geometries, sizes and noise levels. Graphs
are constructed once per dataset/graph case and remain fixed along the
warm-started path. No gamma selector is applied: **this study does not estimate
unsupervised model-selection performance.**

At zero noise, changing the seed changes shuffling, not the set of geometric
locations. Those runs probe order and tie effects, not independent sampling
variability. With positive noise, seeds generate different perturbations.
Three seeds are still too few for broad statistical claims.

## Evaluation and comparator privileges

Each gamma reports
[`adjusted_rand_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
(ARI), cluster count, one-to-one label-matching accuracy, optimization gap,
KKT residual, center error bound, iterations and solve-plus-label runtime.
ARI is invariant to cluster numbering and accounts for chance agreement.
Matching accuracy pairs at most one predicted cluster to each of the two true
classes, maximizing the correctly matched observations; extra clusters are
unmatched. Thus fragmentation is not rewarded by relabeling every fragment
independently. Unconverged points retain their measurements but are excluded
from the converged oracle envelope.

The best ARI over the converged gamma grid is explicitly an **oracle
upper-envelope diagnostic**. It uses truth after fitting and is not an
achievable unsupervised selection score. Reporting a favorable point on a
path is different from identifying that point without labels. The fixed grid
can also miss useful gamma intervals.

[`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
uses 20 initializations and the dataset seed.
[`AgglomerativeClustering`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
uses Euclidean single linkage. Both receive **the known true cluster count,
k=2**, and are labeled `oracle_k` comparators. Convex clustering does not
receive k. Their metrics are therefore diagnostics under different parameter
privileges, not a fair tuning-budget contest or evidence of a universal
ranking. The fitting APIs were checked against current official scikit-learn
documentation using Context7.

## Reproduce and audit

```bash
python examples/recovery_study.py --smoke --self-test
python examples/recovery_study.py --output docs/recovery_study_results.jsonl
```

The smoke case has 40 noisy moon observations, one graph and three gamma
values. The full protocol uses 48 fresh worker processes and 384 requested
convex fits. Each worker has a 25-second deadline; a timeout kills and reaps
that specific worker and preserves any grid points already emitted. A
timeout is distinct from solver iteration exhaustion. Output is appended,
not overwritten. Process-isolated single-thread requests limit interference
between cases but do not control other activity on the host.

`fit_fixed_grid` accepts only observations and fixed algorithm settings; it
has no truth-label or desired-cluster-count argument. The `--self-test` runtime
audit wraps problem construction and solving with strict argument checks,
verifies that the exact prespecified gamma sequence is used, and confirms
that evaluating either original or permuted truth leaves fitted predictions
unchanged. Truth enters only the metric computation and the explicitly
oracle-k comparator construction. This is an auditable separation of fitting
from evaluation, not a claim to implement a general leakage detector.

Raw observations are in `recovery_study_results.jsonl`; the independent smoke
run is in `recovery_study_smoke.jsonl`. Records contain all settings, software
versions, source hashes, statuses and per-gamma results. Each path's graph
construction time is repeated in its grid-point records for context and
should be counted only once when totaling that path. Per-gamma `seconds`
includes solving and numerical labeling but excludes graph construction and
the one-time prepared-problem construction. Process time includes imports,
the full path, both comparators and output.

## Geometric limitation to inspect

Disconnected graph components are independent optimization subproblems, but
public numerical labels connect *all* sufficiently close fitted centroids.
Two concentric rings have the same mean in the symmetric noiseless design.
If each ring fuses completely, both components approach that same centroid;
centroid-distance labeling can merge the rings even when the weight graph
has no cross-ring edge. This follows from the objective and labeling rule,
not from a failed optimization solve. Intermediate gamma values can fragment
the rings instead. The experiment reports the entire fixed path to expose
such behavior instead of presenting only a favorable visualization.

## Observed results

All 48 workers completed without a timeout, and all 384 convex fits passed
the requested gap and KKT checks. The largest reported KKT residual was
9.97e-7 and largest relative gap 9.99e-7. All workers used the same solver
source SHA-256,
`7d8b17106ea406a50bd211685dc3ed26216c1826cdc7307f7126d5ae53fb5a93`.
The environment was macOS arm64, Python 3.12.5, NumPy 2.5.2, SciPy 1.18.1 and
scikit-learn 1.9.0. These measurements therefore establish optimization
success on the specified finite experiment, not successful label recovery.

The following values are mean ARI over three seeds. The convex columns are
**truth-informed best-over-gamma envelopes, separately for each fixed graph**.
The comparator columns use the **known true k=2**. These privileged summaries
must not be presented as unsupervised selection performance.

| Geometry | n | Noise | Convex oracle, kNN 5 / bandwidth .2 | Convex oracle, kNN 10 / bandwidth .5 | KMeans oracle-k | Single-linkage oracle-k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Moons | 100 | 0 | 1.000 | 1.000 | 0.223 | 1.000 |
| Moons | 100 | .05 | 1.000 | 1.000 | 0.216 | 1.000 |
| Moons | 300 | 0 | 1.000 | 1.000 | 0.247 | 1.000 |
| Moons | 300 | .05 | 0.906 | 1.000 | 0.243 | 1.000 |
| Circles | 100 | 0 | 0.559 | 0.000 | -0.010 | 1.000 |
| Circles | 100 | .05 | 0.564 | 0.027 | -0.010 | 1.000 |
| Circles | 300 | 0 | 0.115 | 0.501 | -0.003 | 1.000 |
| Circles | 300 | .05 | 1.000 | 1.000 | -0.003 | 1.000 |

The apparent exceptions are informative:

- For noisy moons with n=300 and kNN 5, oracle ARIs range from 0.807 to 1.000.
  Two seeds produce three graph components and the best reported partition
  still has three clusters. The denser graph has an oracle grid point with
  ARI 1 for every seed, but finding that gamma without truth is unresolved.
- For noiseless circles with n=100 and kNN 5, even the best ARI of 0.559 has
  **11 predicted clusters**, not two recovered rings. With kNN 10, the best
  ARI is zero. At n=300 the noiseless kNN-10 envelope of 0.501 corresponds to
  **151 clusters**. ARI values alone must not obscure fragmentation.
- For noisy circles with n=300, both graphs have two components and the
  oracle envelope is perfect. The two noisy ring means are approximately
  0.0075–0.0122 apart, exceeding the fixed labeling threshold of 0.001;
  noiseless ring means coincide. Recovery here is sensitive to centroid
  symmetry and numerical labeling. It is not evidence that noise generally
  improves convex clustering.
- Single linkage with oracle k succeeds on every generated case in this
  limited protocol. KMeans with oracle k struggles with these geometries.
  This favors neither a universal ranking nor a claim that convex clustering
  is the preferred approach whenever clusters are nonconvex.

Across the 384 convex fits, solve-plus-label time totaled 31.43 seconds;
the median per point was 0.052 seconds and the maximum 0.649 seconds. Worker
process time totaled 104.37 seconds, including imports, graph/problem setup,
paths and comparators. These are single-host observations without timing
replicates; they are not an accuracy-matched speed comparison across methods.

The protocol and all requested gamma values were fixed before the full run.
No additional grid was searched after observing these outcomes. Larger noise,
other ring separations, graph families, sample sizes and data-driven tuning
remain untested here. The raw per-gamma records retain cluster counts and
matching accuracy alongside ARI so that the reported limitations can be
checked directly.
