# Executed platform validation

The scientific suite and installed-distribution checks now run on Windows
and macOS as well as Linux. Coverage is specific to the operating systems,
Python versions and dependency stacks below. Package metadata alone does not
establish that every combination works.

## CI coverage

| Runner | Python | Validation |
| --- | --- | --- |
| Ubuntu | 3.10, 3.12, 3.13 | Full scientific suite, examples, lint and distribution builds |
| Ubuntu, minimum dependencies | 3.10 | Full suite with minimum NumPy/SciPy/scikit-learn, CVXPY, pandas and Matplotlib pins; remaining tools resolve normally |
| Ubuntu, documentation/distributions | 3.12 | Strict documentation build from the source archive; isolated wheel and source-built wheel checks |
| Windows | 3.12 | Full suite against an installed wheel, user workflows, isolated wheel/source checks |
| macOS | 3.12 | Full suite against an installed wheel, user workflows, isolated wheel/source checks |

The Windows/macOS jobs build their own artifacts, install the wheel with
validation extras, and verify that the package import is inside site-packages
and outside the checkout. The full suite includes independent CVXPY oracle
and estimator-conformance tests. These jobs then execute basic, missing-data,
structured, generalized and repeated-solve examples, the recovery smoke and
self-test, and the clustering-path plotting example.

A separate [distribution harness](distribution_validation.md) creates fresh
runtime-only environments to check the wheel and a wheel rebuilt from the
source archive. Each installation passes dependency, import-provenance,
version and optional-dependency-absence checks before the 21 numerical smoke
contracts. The full scientific suite and these runtime-only checks serve
different purposes; neither substitutes for the other.

## Recorded first run

[CI run 34002829261](https://github.com/dx-li/ssnalclust/actions/runs/34002829261)
completed all seven jobs successfully at commit `9d80b22`. The preserved
[JSONL evidence](platform_validation_results.jsonl) contains the platform
metadata, test counts/skip reasons and complete distribution-harness reports,
including artifact hashes and numerical smoke certificates, from that run.
These are historical observations for the identified commit, not a
certification of a later release candidate.

| Actual platform | Python | Scientific tests | Fresh wheel checks | Source-built wheel checks |
| --- | --- | --- | --- | --- |
| Windows Server 2025, AMD64 | 3.12.10 | 682 passed, 1 skipped | 21 passed | 21 passed |
| macOS 26.6.2, arm64 | 3.12.10 | 683 passed, no skips | 21 passed | 21 passed |

Both used NumPy 2.5.2, SciPy 1.18.1, scikit-learn 1.9.0, CVXPY 1.9.2 and
pytest 9.1.1. The successful user-workflow steps include generating the
clustering-path figure. Runner labels such as `windows-latest` and
`macos-latest` can change their underlying images; consult each run's recorded
environment when reproducing a failure.

The single Windows skip is the Unix-native-RSS benchmark subprocess
integration test. The numerical graph and solution-evidence tests still run.
The package's solvers and model families are not skipped on Windows. The
scalability and Newton microbenchmark tools currently measure native memory
on Linux/macOS only; this run does not add Windows memory benchmarking.

## Evidence and remaining coverage

Each platform job uploads environment metadata, resolved dependencies, JUnit
results, the example figure and distribution validation JSON for 14 days.
Artifacts are retained even after a failed later step when files exist. The
matrix disables fail-fast so one operating-system failure does not cancel
its counterpart. Release evidence needs longer-term retention under the
[release process](releasing.md).

Windows/macOS coverage currently uses a current dependency stack on Python
3.12. Their minimum dependency combinations and other Python versions are
not covered by these jobs. Intel macOS and Windows ARM are also outside this
recorded matrix. Each job builds its own wheel/source archives; testing one
shared, exact release-candidate artifact on every platform remains part of
candidate validation. No package or documentation site has been published.
