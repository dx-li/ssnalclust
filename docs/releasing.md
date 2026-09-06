# Release process

**No public release has happened.** The repository currently uses `0.1.0`
as a development baseline; this is not evidence of a PyPI upload or a tagged
release. This document prepares a reviewable release. It does not create
tags, configure a publisher, or upload distributions.

## Prepare the candidate

1. Choose the release commit and resolve release-blocking numerical defects.
   Check [readiness](readiness.md), [API compatibility](api_compatibility.md),
   the open issues, and known unconverged benchmark cases. A package release
   does not close the broader scientific validation agenda.
2. Choose a candidate version, for example `0.1.0rc1`, and update all three
   current version declarations together: `[project].version` in
   `pyproject.toml`, `__version__` in `src/ssnalclust/__init__.py`, and
   `version` in `CITATION.cff`. Also update the development version stated on
   `docs/index.md` so the landing page agrees with installed metadata and the
   generated documentation. Recheck the actual files if version management
   changes. `0.1.0rc1` is a prerelease, distinct from `0.1.0`; use a new
   candidate number after changes. See the official
   [version specification](https://packaging.python.org/en/latest/specifications/version-specifiers/).
3. Turn the relevant `CHANGELOG.md` Unreleased entries into a candidate
   section. Retain the historical “0.1.0 development baseline” explanation;
   do not present it as a prior publication. Verify repository/issue links,
   the license, README installation instructions, algorithm citations, and
   citation authorship. Add a release date only when it is established.
4. Record the commit, Python/platform information, dependency versions,
   numerical-library thread settings, test commands and outputs. Historical
   evidence in `docs/validation*.md` and the benchmark JSONL files is useful
   context, but does not certify a changed candidate.

The [test workflow](../.github/workflows/tests.yml) covers Python 3.10, 3.12
and 3.13 on Linux, plus a minimum dependency job.
[Windows/macOS jobs](platform_validation.md) exercise Python 3.12 installed
wheels, the scientific suite, examples and fresh distribution installations.
A separate job builds documentation from the source archive and checks isolated wheel/source installations,
retaining HTML, distributions and the numerical validation report for 14
days. Archive release evidence separately when longer retention is needed.
Review the workflow at the candidate commit and retain its actual run links.
Do not imply additional operating-system/Python/dependency combinations
were tested merely because metadata does not exclude them.

## Build and validate the actual artifacts

Use a clean candidate checkout and a dedicated virtual environment. The
following commands build and inspect artifacts locally; none publishes them.
Use an empty output directory for each candidate so stale files cannot enter
the checks. From the checkout root:

```bash
python -m venv /tmp/ssnalclust-release-build
/tmp/ssnalclust-release-build/bin/python -m pip install --upgrade pip
/tmp/ssnalclust-release-build/bin/python -m pip install build twine
/tmp/ssnalclust-release-build/bin/python -m build --outdir /tmp/ssnalclust-candidate-dist
/tmp/ssnalclust-release-build/bin/python -m twine check --strict /tmp/ssnalclust-candidate-dist/*
```

Choose unused temporary directory names for repeated runs. On Windows, use
the virtual environment's `Scripts` executable paths. The standard build
command creates an sdist and builds a wheel from it, exercising the sdist's
build inputs. Twine checks distribution metadata/README rendering; it does
not establish numerical correctness. These commands follow the PyPA
[build guidance](https://packaging.python.org/en/latest/discussions/setup-py-deprecated/)
and [README validation guidance](https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/).

Inspect both archives. Confirm the package modules, license and distribution
metadata are present; confirm the sdist contains the tests, examples,
documentation, citation file, and other assets required by the validation
workflow. A wheel need not contain the repository's test/documentation tree.
Check the wheel's metadata version against its installed runtime version and
the citation. Save SHA-256 hashes alongside logs for the exact artifacts
tested.

Test each artifact in a separate fresh environment, with no editable install
and no checkout directory on `PYTHONPATH`. For a candidate named `0.1.0rc1`,
the wheel installation step is:

```bash
python -m venv /tmp/ssnalclust-release-wheel
/tmp/ssnalclust-release-wheel/bin/python -m pip install '/tmp/ssnalclust-candidate-dist/ssnalclust-0.1.0rc1-py3-none-any.whl[test,examples]'
```

Use the actual artifact filename/version. Repeat in another environment with
the `.tar.gz[test,examples]` artifact to test source installation. Install
the minimum supported dependency versions explicitly for the minimum-stack
run; obtain the pins from the candidate's CI rather than an old validation
report. Also test a current dependency stack. Capture the resolved packages
with `python -m pip freeze` using each environment's interpreter.

Run from a staging directory outside the source checkout. Unpack the sdist
there and run its tests/examples with the installed-artifact interpreter;
do not install that unpacked tree in editable mode. If tests or required
assets are missing, fix the packaging manifest and rebuild before continuing.
The imported `ssnalclust.__file__` must point into the fresh environment's
`site-packages`, and `importlib.metadata.version('ssnalclust')` must equal
`ssnalclust.__version__`. Follow PyPA's
[downstream artifact-testing guidance](https://packaging.python.org/en/latest/discussions/downstream-packaging/).

Retain evidence for:

- The full numerical and estimator suite, including independent oracle
  tests, from the installed wheel and sdist. Record any skips and why;
  silently missing CVXPY does not count as completing oracle validation.
- Basic, missing, structured, generalized and repeated-solve examples;
  documentation snippet tests; recovery smoke/self-test; and the rendered
  clustering-path gallery. Inspect the image as well as the process status.
- Lint/format checks and documentation links at the candidate commit.
- A warning-free HTML build from the unpacked candidate documentation using
  the installed candidate package. In a separate documentation environment,
  install the candidate wheel with its `[docs]` extra and run
  `python -m sphinx -b html -W --keep-going docs build/docs/html` from the
  unpacked sdist root. Verify that the imported package comes from that
  environment's `site-packages`, then inspect the rendered landing page,
  equations, tables, downloads and generated API pages.
- At least one representative path and externally referenced comparison at
  achieved accuracy when solver behavior changed. Preserve failed or
  budget-limited cases; do not select only successful benchmark rows.

## Candidate review and release notes

Circulate the candidate artifact hashes and evidence to intended testers.
Ask them to report installation environment, input shapes/graph construction,
solver options, attained gap/KKT residual and practical workflow problems
using the issue templates. Local artifact testing does not require an index
upload. A TestPyPI upload, if chosen later, is a separate publication action;
it has not occurred as part of this process.

Draft notes against the exact candidate commit. Include the model families
and supported solvers, public API changes, numerical fixes and certificate
interpretation, Python/dependency support, executed validation, reproducible
performance evidence, and known limits. Distinguish mathematical model
guarantees from floating-point diagnostics and observed benchmarks. Link
the changelog, citation, user guide, and issue tracker. Do not describe
unattained likelihood optima, unconverged budget-limited runs, or oracle-tuned
recovery as general guarantees.

For the final version, update all version declarations and candidate notes,
then rebuild and revalidate the final artifacts. A tested `rc1` wheel cannot
be renamed to make a tested final wheel. Keep the final artifact hashes and
validation logs with the release record. Creating a tag, publishing a GitHub
release, and uploading to a package index remain separate explicit release
actions; this checklist does not execute them.

Once publication actually occurs, update the unreleased notices in the
landing page, installation guide, this release guide and changelog to state
the published version and date accurately. Preserve the distinction between
historical development evidence and release validation. Until publication,
retain the notices that no PyPI release has happened.

## Trusted Publishing: future setup only

No publishing workflow or PyPI publisher is activated by this document.
When a public release is intended, consult PyPA's current
[GitHub Actions publishing guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
The documentation was checked through Context7 on 2026-09-05; resolve current
action versions when implementing a workflow, rather than copying old
example version pins.

Configure the PyPI project/pending publisher with project name `ssnalclust`,
GitHub owner `dx-li`, repository `ssnalclust`, the exact future publishing
workflow filename, and its GitHub environment name (for example `pypi`).
Confirm project-name availability and account ownership at that time.
TestPyPI uses a separate publisher and environment configuration. Use the
Trusted Publishing OIDC flow, granting `id-token: write` only to the future
publishing job. That job should consume the exact validated distribution
artifacts rather than rebuilding an untested copy. Define its release trigger
deliberately; ordinary test runs and pull requests must not upload packages.
This is a setup plan, not a claim that any project, publisher, environment,
tag or release already exists.
