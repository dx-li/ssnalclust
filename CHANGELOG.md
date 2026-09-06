# Changelog

## Unreleased

- Preserve original data and recorded artifact bytes on Windows checkouts,
  with provenance regressions and stable Python-source line endings.

- Compute held-out MSE with scaled residuals, rejecting invalid scores before
  selection or refitting, and clarify nonunique-imputation semantics.
  Add a frozen training-only Wine holdout study with saved edge duals,
  independent certificate audits, and a retained singleton-selection outcome.

- Release obsolete warm-start buffers and prior yielded results in streamed
  paths before computing the next point, preserving mutation-safe snapshots.
  Measure native memory for streamed versus retained 8–256-point paths, with
  exact numerical parity checks and portable checkpoint audits.

- Add a frozen Optical Digits protocol, offline data with attribution, atomic
  checkpoints, retained full-data fits and cross-checks, and figures showing
  graph sensitivity and a silhouette selection failure.

- Add Windows and macOS installed-wheel CI, scientific user workflows and
  isolated distribution checks, with retained environment and test evidence.

- Extend scalability evidence with graph-weight and centroid-movement
  diagnostics, direct edge fusion, and completed-point certificates that
  survive path timeouts; measure high-dimensional fits and paths.

- Add a strict Sphinx/MyST documentation build with generated public API
  reference, isolated wheel/source installation checks, release guidance,
  and scientific issue templates.

- Reduce temporary-array work in SSNAL Newton products and normalize edge
  vectors before evaluating their projection Jacobians, avoiding finite-input
  norm overflow and underflow.

- Compare with an executed, pinned R/C cvxclustr reference using independent
  certificates; preserve raw outputs, inaccurate stopping cases, and CI
  regressions without adding R as a runtime dependency.

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
