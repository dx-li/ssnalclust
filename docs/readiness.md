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
  numerical-library threads, source hashes, and machine details.
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
  external reference comparison, not only small random CVXPY problems.
- Improve API coherence, error reporting, extension contracts and release
  documentation through actual workflows; avoid adding methods whose
  statistical meaning has not been defined.

## Distribution and maintenance

- Keep minimum/current dependency environments and supported Python versions
  in CI, execute examples, and build distributions.
- Establish versioned documentation, a deliberate release process, changelog,
  contribution guidelines and issue templates before a public package launch.
- A PyPI release, third-party usage, API feedback and sustained maintenance
  are separate evidence from a local test suite. Do not label the project the
  community standard merely because an implementation milestone is finished.

Current work prioritizes stronger certificates, measured Newton performance,
prepared/streamed paths, recovery experiments, and estimator compatibility. Further gaps remain open
until their evidence is recorded and reviewed.
