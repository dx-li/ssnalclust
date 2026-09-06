# Missing-data certificates and recovery validation

This phase builds on the reliability/scalability work in draft PR #3. It
adds a certificate for missing-data fitting, broadens estimator protocol
coverage, and records recovery failures as well as successes. It does not
establish the package as a community standard.

## Independent numerical evidence

`tests/test_missing_certificates.py` adds 31 checks across all three fusion
norms. Independently formulated CVXPY original primal, observed-range boxed
primal, and extended dual problems agree. The certificate remains a lower
bound for the original optimum even when its edge dual has nonzero divergence
at missing entries: it belongs to the equivalent box problem, not the ordinary
unconstrained masked Fenchel dual.

Additional checks cover nonunique optimizers inside and outside the box,
collapsed ranges, independent large component translations with a 70-digit
Decimal reference calculation, ignored finite placeholders, inactive edges
from underflowed radii, gamma zero, and objective-gap stopping when the initial
normalized KKT residual is already small. An extreme-range regression checks
KKT diagnostics against actual returned coordinates. These are numerical
checks, not interval-arithmetic guarantees. See the
[certificate proof](missing_certificates.md).

## API evidence

The [compatibility audit](api_compatibility.md) documents full estimator
checks at explicit fusion strengths, expected default-score exceptions, and
the test-only exogenous graph fixture needed for missing-data checks of varying
sample sizes. Direct production-interface tests cover feature names, cloning,
read-only inputs, singleton data, and warning diagnostics. Invalid label
tolerances are now rejected before fitting, preserving prior learned state.

## Scientific experiment

The [fixed recovery protocol](recovery_study.md) contains 48 scenarios and
384 converged convex fits on moons and circles, plus 96 known-k comparator
fits. Raw records preserve every requested gamma, graph connectivity, cluster
count, accuracy, objective certificates and runtime. Truth-informed gamma
envelopes and known-k comparators are explicitly labeled as oracle diagnostics.
Convergence does not guarantee recovery: the published table includes graph
fragmentation and concentric-centroid failures. CI executes the smoke protocol
and its runtime check that truth does not enter convex fitting.

## Integration checks

The complete suite passed 619 tests on Python 3.12 with current dependencies
and on Python 3.10 using the installed wheel with NumPy 1.24.0, SciPy 1.12.0,
scikit-learn 1.6.0, CVXPY 1.5.0, pandas 1.5.3 and Matplotlib 3.8.0. The minimum
stack emits upstream Matplotlib/Pyparsing deprecation warnings. No pandas
feature-name tests were skipped. Ruff lint/format checks, source and wheel
builds, the missing-data example, and recovery smoke/truth-separation audit
passed. Local documentation links resolve.

## Remaining evidence

External convex-clustering implementation comparisons, broader real-data
studies, model-selection validation, larger-scale performance, versioned docs,
and a deliberate public release remain open. The
[readiness ledger](readiness.md) tracks the broader objective.
