# Readiness beyond the initial library implementation

The initial PR is merged, but broad feature coverage and a green unit suite
are not enough to establish a leading scientific Python library. Development
continues against the following concrete gaps. This file is an evidence
ledger, not a declaration of completion.

## Numerical trust

- Every success flag must correspond to the actual returned arrays and the
  specified objective. Compare independently formulated primal and dual
  problems; normalized residuals alone can be misleading for scaled data.
- Complete-data and structured quadratic models should expose meaningful
  objective-gap certificates. Generalized likelihood models need feasible
  conjugate-domain duals. Missing-data fitting now uses a certificate for an
  equivalent observed-range box problem. Its dual is distinct from the
  unconstrained masked dual, which requires zero missing-entry divergence.
  See [the proof and independent checks](missing_certificates.md).
- Test model-specific boundary cases, data scaling, graph degeneracy,
  nonuniqueness and unattained optima. Preserve regressions for discovered
  failures, including failures absent from the initial oracle suite.

## Computational usefulness

- Measure graph construction, optimization, cluster extraction, and entire
  regularization paths separately. Use fresh processes and native peak RSS,
  not just Python allocations, and report timeouts distinctly.
- Compare algorithms at achieved accuracy. A fast unconverged ADMM run cannot
  establish superiority over a converged SSNAL run.
- Profile sparse graphs at 1k, 5k, 10k and beyond before extrapolating the
  paper's much larger experiments. Track dimensionality, graph parameters,
  numerical-library threads, source hashes, and machine details. A subsequent
  [Newton study](newton_performance.md) now records convergence at 50k and
  lower measured solve time at 5k/10k. The
  [high-dimensional study](high_dimensional.md) adds 32–512 features, materialized
  paths and partial-fusion cases with graph/solution diagnostics. Broader
  graph families, repeated measurements and real-data scales remain open.
- Reuse validated graphs and sparse factorizations across path points, and
  stream results for long paths. Further SSNAL work should follow measured
  Newton/CG costs; adaptive sieving needs full-problem verification and must
  allow previously removed edges to reactivate.

## Scientific usability

- Provide a scientific user guide explaining feature scaling, graph choices,
  regularization strength, tolerance interpretation, and model assumptions.
- Include executed graphical examples and recovery studies on informative
  nontrivial geometries. Avoid selecting gamma with ground-truth labels and
  reporting that as unsupervised model selection.
- Establish a reproducible paper-style benchmark/data protocol and a broader
  external reference comparison, not only small random CVXPY problems. The
  [executed cvxclustr study](external_reference.md) now covers shared synthetic
  and Iris problems, including an independently detected reference stopping
  defect. Additional implementations and larger real datasets remain open.
- Improve API coherence, error reporting, extension contracts and release
  documentation through actual workflows; avoid adding methods whose
  statistical meaning has not been defined.

## Distribution and maintenance

- Keep minimum/current dependency environments and supported Python versions
  in CI, execute examples, and build distributions.
- Build searchable documentation and validate installed wheel/source artifacts
  in fresh runtime environments. The [local documentation build](installation.md),
  [distribution harness](distribution_validation.md), [release checklist](releasing.md),
  changelog and issue templates now provide this foundation. Published versioned
  documentation, a release candidate, and an independently reviewed public
  release remain open.
- A PyPI release, third-party usage, API feedback and sustained maintenance
  are separate evidence from a local test suite. Do not label the project the
  community standard merely because an implementation milestone is finished.

Current work prioritizes stronger certificates, measured Newton performance,
prepared/streamed paths, recovery experiments, and estimator compatibility. Further gaps remain open
until their evidence is recorded and reviewed.
