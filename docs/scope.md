# Scope, evidence, and further development

The objective is a scientifically reliable, broadly useful Python convex
clustering library. SSNAL repair alone is insufficient: this development
release includes several solver families, alternative convex objectives,
estimator integration, graph construction, paths, and model selection.
Community adoption as the primary library will require public releases,
maintenance, independent users, and sustained performance work; local tests
cannot establish adoption.

## Implemented acceptance surface

| Requirement | Implemented behavior | Independent evidence |
| --- | --- | --- |
| Correct Euclidean SSNAL | ALM, semismooth Newton-CG, correct dual balls and generalized Hessian, bounded Armijo search | `test_solvers.py`, `test_prox.py`: derivative checks and CVXPY comparisons; historical derivation in `algorithm_audit.md` |
| Multiple classical algorithms | SSNAL, ADMM, AMA, accelerated AMA; l1/l2/linf where supported | All ten solver/norm combinations checked against independent convex solves |
| Scientific accuracy | Feasible primal/dual objectives and KKT; failure status; actual returned-array certificate | Analytic weighted two-point, invariance tests, extreme-scale and precision regressions |
| Sparse graph support | Sparse incidence, custom dense/sparse adjacency, kNN, MST, kNN+MST, local scales | `test_graph.py`, `test_graph_extra.py`: reference graph computations, duplicates, connectivity and 100k-node sparse graph |
| sklearn interface | Cloneable transductive clusterer, pipelines, fitted attributes; model-specific estimators | `test_estimator.py`, `test_model_estimators.py`; standard checks with documented fidelity-weight equivalence exception |
| Paths | Fixed graph, primal/dual warm starts, streamed results and summaries preserving splits | Cold/warm agreement, known fusions and splits, weak-reference lifetime tests, recorded stream/list parity |
| Weighted fidelity | Strictly positive sample masses on fixed graph | Weighted oracle, analytic formula, weighted-mean tests |
| Missing data | Masked squared loss, explicit graph, no hidden ridge, component identifiability | `test_missing.py`: masked oracle, ignored-entry invariance, nonunique solution interpretation |
| Feature selection | Centered feature-group sparse convex clustering | `test_structured.py`: independent oracle, alpha-zero reduction, feature elimination |
| Biclustering | Simultaneous row/column Euclidean fusion | Independent oracle, transpose equivariance, one-axis reductions |
| Robust/generalized fidelity | Huber, Bernoulli-logit, Poisson-log-intensity | `test_generalized.py`: loss-specific oracle, proximal and boundary tests |
| Gamma selection | Reproducible held-out-entry MSE and full-data refit | `test_selection.py`: masked-oracle scores, withheld-value erasure, identifiability, nonconvergence rejection |
| Distribution and examples | src-layout package, license/citation, wheel/sdist, CI, runnable scripts | Build/install checks and examples for every model family |
| Performance evidence | Graph/solver timing, process peak RSS, Newton profiling, achieved accuracy and long-path retention | Recorded scalability, Newton, high-dimensional and streaming studies; initial traced-allocation benchmark |

See `models.md` and public docstrings for exact objectives, supported options,
and limitations. Missing-data, structured and generalized results require
both original KKT conditions and model-specific relative gaps. Their feasible
lower bounds use an equivalent observed-range box, a stacked quadratic
conjugate, or the actual fidelity conjugate, respectively.

The [real-data method gallery](gallery.md) links runnable solver/model examples
to their visualizations and diagnostics. The [Digits](digits_study.md),
[Wine](wine_holdout.md) and [corrected stability](stability_study.md) studies
retain scientifically unsuccessful selections alongside numerical success.
Corrected stability currently lives in an example workflow, not the public API.

## Explicit limitations and future research engineering

The current implementation does not claim to reproduce the original paper's
200,000-sample runtime. SSNAL is matrix free in its Newton solve; ADMM uses a
sparse factorization that can fill in. kNN queries can deteriorate in high
feature dimensions. Exact MST graph construction is quadratic in memory.
The initial benchmark's allocation peaks use tracemalloc and omit some native
allocations. Subsequent [scalability](scalability.md),
[Newton](newton_performance.md), [high-dimensional](high_dimensional.md) and
[streaming-memory](streaming_memory.md) studies record native process peak RSS.
They include a converged 50,000-sample sparse case and long-path retention
measurements, not portable runtime guarantees. Broader graph structures,
hardware and model families still need performance characterization.

The following are future extensions, not claims of implemented functionality:

- CARP-style deliberately approximate paths, adaptive sieving, and safe graph
  compression with full-problem reactivation/certification. Arbitrary weighted
  paths must not be assumed to be agglomerative.
- Specialized GPU/distributed kernels and stochastic variants, with dedicated
  numerical and hardware benchmarking.
- Multi-view and tensor co-clustering, compositional and other domain-specific
  losses, and public custom-loss/operator protocols backed by concrete uses.
- A public partition-stability selector and justified information criteria.
  The public selector currently uses held-out entries with an explicitly
  supplied fixed graph; the completed example-level stability study does not
  establish general recovery or selection consistency.
- Broader real-dataset validation, model-selection sensitivity and independent
  evaluation beyond the existing galleries and recorded studies. Illustrative
  examples alone do not establish scientific recovery or recommended defaults.

Biconvex metric learning and nonconvex fusion penalties have different global
optimality guarantees and should not be silently presented as globally convex
models. They need a distinct API and validation plan if added.

## Release audit

Before merging or publishing a release, rerun the complete numerical tests,
execute all model-family examples, build the wheel and source distribution,
and test installation against supported dependency versions. Keep failed
benchmarks visible alongside converged runs. Review README/API claims against
actual tested combinations. A drafted PR and local checks do not establish
that remote CI passed or that a package was published.
