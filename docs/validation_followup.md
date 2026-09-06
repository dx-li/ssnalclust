# Reliability and scalability follow-up validation

After merging initial PR #2, this development pass expands the numerical
suite from 485 to **553 passing tests**.

- Python 3.12/current dependency environment: 553 passed.
- Clean installed wheel under Python 3.10, NumPy 1.24.0, SciPy 1.12.0,
  scikit-learn 1.6.0, CVXPY 1.5.0 and Matplotlib 3.8.0: 553 passed.
  The older plotting stack emits upstream Pyparsing deprecation warnings.
- Ruff lint and format checks passed; wheel and source distribution built.
- Basic, missing/selection, structured, generalized and repeated-solve scripts
  executed successfully. The path gallery solved all 52 points to the stated
  tolerance, rendered successfully, and was visually inspected.
- Local documentation links resolve. CI now installs plotting extras and
  executes the gallery as well as the existing examples.

## What the added tests establish

Independent convex dual optimizations validate structured and generalized
lower bounds. Tests cover feasible conjugate domains, weak global scaling,
objective cancellation, subnormal likelihood tails, and scaled cases where
KKT alone previously accepted a poor objective. Complete-data error bounds
retain weighted stationarity information even for very small fidelity masses.

Prepared-problem tests verify graph/factor reuse, bounded cache replacement,
input snapshots, warm/cold agreement, streaming laziness, independence from
caller mutations, and unchanged final diagnostics without histories. Cadence
tests verify final-iterate checks even between regular checkpoints. Duplicate
label tests count spatial queries and verify transitive components instead of
asserting fragile elapsed-time thresholds.

The guide's code snippets execute as tests. The gallery's synthetic recovery,
path endpoints, numerical certificates and rendered panels are checked. Its
illustrative snapshot uses known labels explicitly; it is not an evaluation
of unsupervised gamma selection.

## Performance evidence and its limits

See [scalability.md](scalability.md) and the accompanying raw JSONL records.
Fresh processes measure native RSS and separate graph, solve, label, Newton,
CG and diagnostic costs. Timeout records are retained separately from
iteration exhaustion and converged runs. Development source hashes identify
the measured versions.

The sparse 5,000-sample SSNAL case improved from 38.7 to 18.9 solve seconds
with essentially unchanged accuracy; the revised 10,000-sample case finished
in 37.1 seconds. Exact-duplicate labeling avoids repeated clique traversals.
Optional diagnostic cadence reduces ADMM path overhead, but the recorded
hardest path point still fails its tolerance within the iteration budget.
These results do not establish universal speed rankings or paper-scale
performance on 200,000 observations.

## Still open

Missing-data dual certificates, larger externally reproducible paper-style
experiments, broader dataset studies, robust large-scale path methods,
versioned documentation and release readiness remain open. See the
[readiness ledger](readiness.md). This validation is evidence of progress,
not completion of the broader library goal.
