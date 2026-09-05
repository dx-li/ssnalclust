# Changelog

## Unreleased

- Certify missing-data objective gaps using an equivalent observed-range box,
  while retaining original KKT checks and avoiding artificial ridge penalties.
- Validate label tolerances before fitting alternative model estimators.
- Add broader estimator checks and reproducible nonconvex-geometry recovery
  experiments with explicit oracle tuning and known-cluster-count limitations.

- Reuse prepared data, graphs, and a bounded ADMM factorization cache across
  solves; stream paths without accumulating histories.
- Use inexact ALM tolerances that avoid oversolving early SSNAL subproblems.
- Add stable quadratic Fenchel gaps and a numerical centroid-error bound.
- Require primal-dual gap and KKT checks for structured and generalized
  models, using the actual fidelity conjugate and feasible dual variables.
- Make diagnostic cadence configurable while always checking returned results.
- Collapse exact duplicate centroids before label graph traversal.
- Add a scientific guide, executed path gallery, and process-isolated
  scalability evidence through 10,000 samples.

## 0.1.0 development baseline

Merged PR #2 repairs the original SSNAL prototype and introduces the initial
scientific Python package, model families, estimator interfaces, independent
numerical tests, examples and CI. No PyPI release has been published.
