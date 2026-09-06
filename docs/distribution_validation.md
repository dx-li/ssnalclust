# Validate the artifacts users install

Running tests from an editable checkout does not establish that a wheel or
source distribution contains a usable package. The distribution harness
installs the actual artifacts into fresh virtual environments and exercises
them from an unrelated working directory.

## Run the check

The invoking interpreter needs only its standard library, including `venv` and
`ensurepip`. Use a supported Python version with those modules available. Pip
requires access to the configured package index for runtime dependencies and,
for a source build, build requirements.

```sh
python scripts/check_distribution.py dist/ssnalclust-0.1.0-py3-none-any.whl \
  --sdist dist/ssnalclust-0.1.0.tar.gz \
  --report build/artifacts/distribution-validation.json
```

The wheel path is required. Omit `--sdist` to check only a wheel, or omit
`--report` for console status only. The harness does not build from the current
checkout, modify dependency files, or publish a package. It validates the
specific archives passed to it; rebuild those archives after source changes.

## Isolation and import checks

Each tested wheel gets a new virtual environment with system site-packages
disabled. Pip installs the wheel without requesting any extras, then runs
`pip check`. Numerical checks execute with Python's isolated `-I` option from
a separate temporary working directory. The harness verifies:

- `sys.prefix` identifies the new environment;
- `ssnalclust.__file__` is inside that environment's site-packages and outside
  the repository checkout;
- the installed package metadata version and `ssnalclust.__version__` match the
  wheel's metadata;
- all names in the installed package's public `__all__` exist;
- CVXPY, pytest, pandas, Matplotlib, and the `build` package are absent.

NumPy, SciPy, and scikit-learn, together with their runtime dependencies, are
installed through the wheel's declared requirements. The smoke program uses
NumPy assertions directly and does not import test helpers from the checkout.
The absence checks guard against accidentally relying on optional development
packages in runtime code.

## Scientific smoke checks

The harness checks convergence, finite objectives, centroid shapes, relative
gaps, and KKT residuals. Several checks also have closed-form expected values:

| Interface | Contract exercised |
| --- | --- |
| SSNAL, ADMM, AMA, FAMA | Two-observation Euclidean fusion solution and objective |
| ADMM with `l1` fusion | Coordinatewise two-observation solution and objective |
| `ConvexClustering` | Fused labels and cluster-center summary |
| `ConvexClusteringProblem` | Prepared dimensions and a solve on the fixed graph |
| Prepared/public streamed paths and public list path | Zero-gamma recovery, final common centroid, path summaries, disabled history |
| Missing-data model | Complementary observed entries recover a common vector with zero objective |
| Huber model | Large-delta case agrees with the squared-loss solution |
| Logistic/Poisson models | Constant-observation natural-parameter fits, means, and likelihood objectives |
| Feature-sparse model | Closed-form column shrinkage when row fusion is disabled |
| Biclustering | Disabling column fusion recovers ordinary row clustering |

This is an installed-artifact integration check, not a substitute for the
independent oracle, invariance, estimator-conformance, or external-reference
regression suites. Passing it shows that the public interfaces needed for
these examples are present and work with declared runtime dependencies.

## Source archive check

With `--sdist`, the harness checks that the source archive includes
`pyproject.toml`, `LICENSE`, and the package initializer. It then creates a
separate builder environment and asks pip to build a wheel from that archive
using isolated build requirements and `--no-deps`. This build uses the source
archive, not an editable checkout. A second fresh runtime environment installs
the resulting wheel and executes the same smoke program. The rebuilt wheel
must have the same package version as the supplied wheel.

Build requirements exist only in the build environment or pip's isolated
build environment. They are not added to the runtime smoke environment.
Temporary build directories and virtual environments are removed when the
harness finishes, whether it succeeds or fails.

## Read the report

The JSON report contains overall status, UTC start/end times, each artifact's
SHA-256 hash, wheel metadata, installed runtime versions, verified import
location, and each numerical check's objective and certificates. Source runs
also report source-archive hash and isolated-build completion. Recorded
temporary paths identify the environment used during execution; they no
longer exist after cleanup.

A nonzero exit can indicate invalid arguments or artifact paths, or failed
environment creation, installation, dependency validation, metadata checks,
import isolation, or numerical contracts. After CLI and artifact-path
validation succeeds, `--report` records failure status and the error before
an exception is propagated, provided the report destination is writable.
Argument errors and missing artifact paths occur before report creation and
do not produce that JSON report. An unavailable package index is an
installation failure, not evidence that numerical checks passed.
