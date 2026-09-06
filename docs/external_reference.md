# Independent implementation comparison

This study executes the unmodified R/C package **cvxclustr 1.1.1**, the
implementation accompanying Chi and Lange's
[Splitting Methods for Convex Clustering](https://arxiv.org/abs/1304.0499).
It compares solutions to the same quadratic convex-clustering objective,
using independently recomputed certificates. This is an implementation
agreement study, not a ranking of runtime or clustering recovery.

The pinned reference is CRAN mirror commit
[`d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf`](https://github.com/cran/cvxclustr/tree/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf).
The package source stays in a separate temporary checkout and installation;
none is vendored into ssnalclust. Its own license is CC BY-NC-SA 4.0.
Our bridge and certificate evaluator are original project code. Reference
outputs and problem arrays are retained for independent inspection.

## Match the mathematical problem first

Both implementations minimize
`0.5*sum((U-X)**2) + gamma*sum(w_ij*norm(U_i-U_j))`, with each undirected edge
counted once and unit fidelity masses. No gamma rescaling is needed.
The bridge transposes data and centers, explicitly checks lexicographic edge
ordering, and maps the R multiplier with `Z=-Lambda.T`. It supplies a safe
fixed AMA step `nu=1/n` and starts every fit from zero multipliers; neither
implementation receives a previous path point as a warm start.

The [primary-source conventions audit](external_reference_conventions.md)
documents the exact mapping and reference limitations. In particular, its
weighted-l1 primal reporting omits weights, invalidating that reported gap
and stopping test. Its early-stop history also omits the terminating entry.
A recorded native objective or iteration count is therefore preserved as
reference metadata, not treated as a certificate or an exact update count.

`examples/external_reference.py:certify_candidate` independently computes
primal and dual objectives, a stable residual/Fenchel-slack gap, normalized
KKT conditions, edge-ball feasibility, and `sqrt(2*gap)` as a unit-mass
centroid error bound. It calls no production objective, proximal, graph or
solver diagnostic helpers. Materially infeasible duals and nonfinite
diagnostics are rejected. Numerical roundoff is tolerated at the scale of
the certificate terms; these are not interval-arithmetic guarantees.

## Fixed comparison protocol

The main run uses four datasets/graphs, two norms (l1 and l2), and the gamma
grid `[.01,.1,1,10]`. Each of the 32 problems is solved once by ssnalclust
(FAMA for l1, SSNAL for l2) and twice by cvxclustr (plain and accelerated AMA).
All graph weights and input arrays are shared exactly across implementations.

- Two-point, two-feature analytic geometry with a single weight 0.7.
- Twelve points in four dimensions from NumPy seed 20260905, with symmetric
  complete-graph weights drawn uniformly from 0.2 to 1.5.
- The same observations with a weighted graph consisting of two six-node
  chains. This exercises disconnected components and omitted edges.
- The 150 observations and four measurements from scikit-learn's
  [`load_iris`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html),
  standardized by the full-data feature means and population standard
  deviations, with Gaussian union kNN weights (k=10, bandwidth=1). Species
  targets never enter fitting, graph construction, gamma choice or evaluation.
  This is a real-data numerical comparison, not a species-recovery claim.

The local tolerance is 1e-8; the reference requests absolute gap 1e-10 and
at most 100,000 native iterations. Local fits also have a 100,000-iteration
budget. The same numeric tolerance would not mean the same stopping rule.
The common accuracy flag requires independently recomputed relative gap and
normalized KKT residual both at most 1e-8. Fits are retained regardless of
that flag. There is no post-hoc filtering of unfavorable reference outputs.

R runs in a fresh subprocess per fit, with a 30-second timeout and explicit
process errors. Local runs share one Python process and request one numerical
library thread; the R environment requests the same. Native solve time and
R process time are recorded separately. Startup, language, algorithm and
achieved-accuracy differences preclude treating these as fair speed ratios.

## Main observed results

All 32 local fits passed their convergence checks and the independently
recomputed common accuracy criterion. All 64 R calls completed without an
error or timeout. Every pair of returned centers lay within the sum of their
independent `sqrt(2*gap)` bounds, plus floating-point roundoff.

| Problem | Reference fits meeting common accuracy | Maximum center distance to local fit | Maximum reference KKT residual |
| --- | ---: | ---: | ---: |
| Two point | 16 / 16 | 4.94e-9 | 1.28e-16 |
| Random complete | 12 / 16 | 5.80e-6 | 3.30e-6 |
| Random disconnected | 12 / 16 | 8.61e-6 | 7.41e-6 |
| Standardized Iris | 13 / 16 | 3.89e-6 | 1.81e-7 |

All 32 l1 reference outputs meet common accuracy, but 24 exhaust the native
iteration budget: their faulty reported weighted gap need not approach zero
even when the actual solution is accurate. Of 32 l2 outputs, 21 meet common
accuracy. The remaining 11 have small objective gaps but do not meet the
stricter common KKT threshold. These facts illustrate why neither native
termination nor iteration exhaustion alone establishes solution accuracy.

The main raw file is [external_reference_results.jsonl](external_reference_results.jsonl).
It includes the actual inputs, graphs, returned centers and multipliers,
source hashes, versions, timings, native metadata and independently evaluated
diagnostics. The run used macOS arm64, R 4.4.1, Python 3.12.5, NumPy 2.5.2,
SciPy 1.18.1 and scikit-learn 1.9.0. The separately installed igraph binary
emitted a warning that it was built under R 4.4.3; the captured stderr retains
that warning. Each R record includes the loaded native library's MD5.

## Source-derived weighted-l1 stress check

After the main grid, a separate six-point, two-feature chain with weights
`[2,4,3,5,2]` probes the source audit's omitted-weight defect. This is a targeted
implementation regression, not another representative performance sample.
The same norm/gamma grid is used; all 16 reference calls complete.

At l1 gamma 0.1, plain reference AMA returns a native reported gap of about
**-0.322**, but its actual weighted gap is **1.558**, KKT residual **0.418**,
and distance to the certified local solution **1.286**. Accelerated reference
AMA also stops inaccurately (actual gap about 0.244). At gamma 1, both
reference gaps exceed 4.3. The reference dual remains feasible, so its broad
error bounds correctly describe the uncertainty; the common accuracy flag
rejects these outputs. The l2 control uses correctly weighted reporting.

The raw [stress records](external_reference_stress.jsonl) preserve all outputs,
including native negative gaps and incorrect early stops. This defect belongs
to the inspected historical reference revision; no claim is made about other
packages or later independent implementations.

## Reproduce

Use an existing R installation with a C compiler and its recommended Matrix
package. Install the reference in an isolated library, respecting its license:

```sh
git clone https://github.com/cran/cvxclustr.git /tmp/ssnalclust-cvxclustr-reference
git -C /tmp/ssnalclust-cvxclustr-reference checkout d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf
mkdir -p /tmp/ssnalclust-r-reference-library
Rscript -e 'install.packages("igraph", lib="/tmp/ssnalclust-r-reference-library", repos="https://cloud.r-project.org")'
R_LIBS=/tmp/ssnalclust-r-reference-library R CMD INSTALL --clean --library=/tmp/ssnalclust-r-reference-library /tmp/ssnalclust-cvxclustr-reference
python examples/external_reference.py --reference-source /tmp/ssnalclust-cvxclustr-reference --r-library /tmp/ssnalclust-r-reference-library --output /tmp/reference-full.jsonl
python examples/external_reference.py --stress --reference-source /tmp/ssnalclust-cvxclustr-reference --r-library /tmp/ssnalclust-r-reference-library --output /tmp/reference-stress.jsonl
```

`--smoke` runs the two-point gamma-0.1 case for both norms and both reference
engines. Output appends to the specified file; use a fresh filename for a new
run. The harness reads the clean checkout's revision and source hashes. The
installation command is what associates that checkout with the installed
package: the harness additionally records the package version and loaded
binary checksum, but cannot prove arbitrary external installations came from
a caller-provided checkout.

Normal CI needs no R installation. It recomputes certificates from saved
external outputs and repeats selected small local comparisons against them.
Independent analytic tests validate the certificate evaluator, including
edge orientation, scaling, dual infeasibility and independence from production
helpers. This preserves a regression check against executed external output
without downloading or compiling another package on every Python test run.

At the floating-point floor, tiny gaps can differ across numerical libraries
because summation and norm reductions round differently. Taking `sqrt(2*gap)`
amplifies the relative difference in those tiny values. The regression tests
therefore compare saved and recomputed centroid bounds in squared objective
units (`bound²/2`), using the same tolerance as the gap, while separately
checking that each bound equals `sqrt(2*gap)`. Objective, gap, KKT and common
accuracy checks remain unchanged. Comparisons between fresh local fits and
external centroids use freshly recomputed bounds with only a floating-point
rounding allowance; machine-floor bounds are not portable high-precision
measurements of centroid error.

## Integration validation

The full Python suite passes **646 tests** on the current Python 3.12 stack
and the installed wheel on Python 3.10 with the minimum dependency versions.
This phase adds 21 independent-certificate tests and six captured-output
regressions. The minimum plotting stack retains its upstream deprecation
warnings. Ruff lint/format, source/wheel builds, local documentation links,
and the final four-reference-fit smoke run pass. The source distribution
includes the R bridge and both raw reference snapshots.

## Remaining comparison work

One historical R implementation is useful evidence, not comprehensive external
validation. [PyClustrPath was separately inspected](external_reference_candidates.md)
but not executed because its stock imports require an unavailable CUDA
extension and Python-version-specific bytecode in this environment. It is not
counted as an executed reference. Accuracy-matched timing comparisons, larger
real datasets, an additional independent SSNAL implementation and broader
model-family reference comparisons remain open.
