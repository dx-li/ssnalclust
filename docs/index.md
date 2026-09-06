# Convex clustering with numerical certificates

**ssnalclust is under active scientific development.** This documentation
covers the development package version 0.1.0; no PyPI release has been made.
The [readiness ledger](readiness.md) separates verified capabilities from
remaining numerical, scientific and release work.

The library fits one centroid per observation and fuses centroids along a
weighted graph. It provides sparse SSNAL, ADMM, AMA and accelerated AMA
solvers, scikit-learn estimators, prepared and streamed paths, and several
alternative fidelity and fusion models. Samples are rows.

## Start with a fitted problem

```python
import numpy as np
from ssnalclust import ConvexClustering

X = np.array([[0., 0.], [.1, .1], [2., 2.], [2.1, 2.1]])
fit = ConvexClustering(gamma=.1, n_neighbors=3).fit(X)
print(fit.labels_)
print(fit.result_.relative_gap, fit.result_.kkt_residual)
```

Follow the [installation and documentation guide](installation.md), then the
[scientific user guide](user_guide.md). A small optimization gap concerns the
specified objective; it does not establish that chosen graphs or labels
answer a scientific question. The [recovery study](recovery_study.md) includes
cases where converged optimization still recovers clusters poorly.

```{toctree}
:maxdepth: 1
:caption: Use the library

Installation <installation>
User guide <user_guide>
API overview <api>
API docstrings <api_generated>
Model definitions <models>
Compatibility <api_compatibility>
```

```{toctree}
:maxdepth: 1
:caption: Understand the certificates

Algorithm audit <algorithm_audit>
Missing data <missing_certificates>
Structured models <structured_certificates>
Generalized losses <generalized_certificates>
Newton operator <newton_operator>
```

```{toctree}
:maxdepth: 1
:caption: Inspect the evidence

Recovery study <recovery_study>
Optical Digits protocol <digits_protocol>
External comparison <external_reference>
Reference conventions <external_reference_conventions>
Other implementations <external_reference_candidates>
Scalability <scalability>
Newton performance <newton_performance>
High-dimensional paths <high_dimensional>
Baseline validation <validation>
Reliability validation <validation_followup>
Missing-data validation <validation_missing_and_recovery>
```

```{toctree}
:maxdepth: 1
:caption: Develop and release

Project scope <scope>
Readiness ledger <readiness>
Release checklist <releasing>
Package installation checks <distribution_validation>
Platform validation <platform_validation>
```

The repository includes [contribution guidance](path:../CONTRIBUTING.md),
[a changelog](path:../CHANGELOG.md), and [citation metadata](../CITATION.cff).
