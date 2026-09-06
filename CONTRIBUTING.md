# Contributing

This library is under active scientific and numerical development. See
[the readiness ledger](docs/readiness.md) for the gaps that matter most.

## Development

```bash
python -m pip install -e '.[dev,examples,docs]'
python -m pytest -q
ruff check src tests examples scripts docs/conf.py
ruff format --check src tests examples scripts docs/conf.py
python -m sphinx -b html -W --keep-going docs build/docs/html
python -m build
python scripts/check_distribution.py dist/ssnalclust-0.1.0-py3-none-any.whl \
  --sdist dist/ssnalclust-0.1.0.tar.gz \
  --report build/artifacts/distribution-validation.json
```

Keep runtime dependencies small; reference optimization packages belong in
optional test dependencies. Use samples in rows and explicit sparse graph
operators. Document objectives, input domains, supported solver/model pairs,
convergence criteria and nonuniqueness or nonattainment where relevant.

## Scientific changes

A solver change needs evidence beyond agreement with a second implementation
of the same update: use independently formulated convex optima/duals,
analytic solutions, derivative checks, or mathematical invariants appropriate
to the change. Keep regressions for discovered failures. A nonconverged run
must never acquire a successful flag through a looser, unrelated criterion.
Do not silently add a ridge, clip a data domain, compress a path irreversibly,
or replace a likelihood with squared loss.

For performance work, include reproducible data, source versions, achieved
accuracy, threads, native memory measurement, and separate timeouts from
iteration exhaustion. Compare at matched accuracy and avoid fragile test
assertions on elapsed seconds. Cite the algorithm and explain any deviations.

## API and examples

Follow NumPy-style docstrings and sklearn conventions where they have a clear
statistical meaning. In particular, joint convex clustering is transductive;
an inductive predict method would require a separately stated model. Execute
examples and inspect figures. Do not tune against known cluster labels and
present the result as unsupervised model selection.

Open a focused PR with the concrete problem, behavior change, validation and
remaining limits. Public release and adoption claims require separate review.

See the [release checklist](docs/releasing.md) for versioning, artifact
validation, and the separate public-release decision. Numerical bug reports
should include reproducible inputs and the achieved gap and KKT residual.
