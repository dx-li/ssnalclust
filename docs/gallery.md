# Real-world method gallery

Start here to see what each implemented method does to measured data. Each
example includes the data preparation, runnable code, fitted visualization,
optimization diagnostics, and limitations of its interpretation. The bundled
datasets make the calculations reproducible without downloading data at runtime.

| Method or workflow | Real-data question | Example and visualization |
| --- | --- | --- |
| SSNAL | Which Wine samples fuse under squared-error L2 clustering? | [Fitted centroids and convergence](gallery_solvers.md#four-solvers-one-objective) |
| ADMM | Does a splitting solver reach the same solution? | [Same-data solver comparison](gallery_solvers.md#four-solvers-one-objective) |
| AMA | How does the simpler alternating method converge? | [Centroids and diagnostic history](gallery_solvers.md#four-solvers-one-objective) |
| FAMA | How does acceleration change AMA's progress? | [Centroids and diagnostic history](gallery_solvers.md#four-solvers-one-objective) |
| L1 and L-infinity fusion | How does fusion geometry change the Wine fit? | [Six fitted panels](gallery_solvers.md#change-the-fusion-norm) |
| Graph construction | Which Wine pairs are encouraged to fuse? | [kNN, tree, connected kNN, local scales](gallery_solvers.md#choose-the-graph) |
| Prepared and streamed paths | How does the fit change with fusion strength? | [Cluster counts and certificates](gallery_solvers.md#follow-a-regularization-path) |
| Missing-data clustering and entry-holdout selection | Can other measurements help reconstruct hidden magnesium values? | [Held-out predictions and error curve](gallery_missing.md) |
| Sparse convex clustering | How does a feature penalty shrink whole chemical profiles? | [Feature-norm path and fitted heatmap](gallery_structured.md#sparse-convex-clustering-suppress-whole-feature-profiles) |
| Convex biclustering | Can both sample and feature profiles be fused? | [Aligned input and fitted heatmaps](gallery_structured.md#convex-biclustering-fuse-both-axes) |
| Huber fidelity | How does bounded residual influence change the Wine fit? | [Fitted measurements and influence](gallery_generalized.md#huber-how-much-can-one-large-residual-influence-a-fitted-value) |
| Logistic fidelity | How can binary congressional votes be smoothed over a graph? | [Observed votes and fitted probabilities](gallery_generalized.md) |
| Poisson fidelity | How can daily bike-rental counts be smoothed over time? | [Counts, fitted means, and log intensities](gallery_generalized.md) |

The four complete-data solvers optimize the same objective when their inputs
and fusion norm match. Missing, sparse, biclustering, and generalized-loss models
use their dedicated PDHG solvers. Consult the [model matrix](models.md) before
choosing a solver for a different objective.

## Reproduce the figures

From a repository checkout or extracted source distribution, install the
example dependencies as described in [installation](installation.md), then run:

```bash
python -m pip install -e '.[examples]'
python examples/gallery_solvers.py --output-dir artifacts/solvers
python examples/gallery_missing.py --output-dir artifacts/missing
python examples/gallery_structured.py --output-dir artifacts/structured
python examples/gallery_generalized.py --output-dir artifacts/generalized
```

Use fresh output directories. Each command writes PNG figures and JSON/NPZ
results and exits with an error when a required fit does not converge. Add
`--smoke` for a smaller execution check; those runs do not reproduce the
full-data figures. The raw-count Poisson example may take substantially longer
than the other fits: its recorded run required 50,766 PDHG iterations.

The [dataset notes](path:../examples/data/README.md) give attribution, licenses,
raw-file hashes, and transformations. Source scripts live in `examples/` in the
checkout and source distribution. These are descriptive demonstrations with
illustrative settings, not a frozen benchmark or a parameter-selection study.
The missing-data example selects gamma on held-out entries; that tuning score
is not an independent test score. Wine missingness is simulated on real
measurements. The voting and bike examples use actual binary and count data.

## Inspect the saved full runs

| Example | Diagnostics and provenance | Numerical arrays |
| --- | --- | --- |
| Solvers, norms, graphs, paths | [JSON](gallery_results/solvers/solvers.json) | [NPZ](gallery_results/solvers/solvers.npz) |
| Missing measurements | [JSON](gallery_results/missing/gallery_missing.json) | [NPZ](gallery_results/missing/gallery_missing.npz) |
| Sparse and biclustering | [JSON](gallery_results/structured/gallery_structured.json) | [NPZ](gallery_results/structured/gallery_structured.npz) |
| Generalized fidelity | [JSON](gallery_results/generalized/diagnostics.json) | [NPZ](gallery_results/generalized/fits.npz) |

The [artifact manifest](gallery_results/manifest.json) records file hashes for
all ten figures and eight result files.

Load numerical arrays with `numpy.load(path, allow_pickle=False)`. The saved
records include data and script hashes, parameters, primal and dual objectives,
gaps, and KKT residuals. Dependency versions are also recorded by the solver,
structured, and generalized galleries. Each final full-run fit satisfies
both its relative-gap and KKT stopping checks at tolerance `1e-6`.

A small optimization gap establishes accuracy for the specified objective.
It does not validate the graph, choose the scientifically appropriate penalty,
or establish meaningful groups. Thresholded labels can change with numerical
tolerance, and two-dimensional projections can hide differences. The
[Wine holdout study](wine_holdout.md), [Digits study](digits_study.md), and
[stability study](stability_study.md) examine selection and its failure modes
more systematically than this gallery.

```{toctree}
:maxdepth: 1

Solvers, fusion norms, graphs and paths <gallery_solvers>
Missing measurements <gallery_missing>
Sparse clustering and biclustering <gallery_structured>
Huber, logistic and Poisson fidelity <gallery_generalized>
```
