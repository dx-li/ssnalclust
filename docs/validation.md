# Development validation

Validation performed on 2026-09-05 before opening the development PR.

## Numerical and integration suite

- Python 3.12.5, NumPy 2.5.2, SciPy 1.18.1: **485 tests passed**.
- Clean wheel installation, Python 3.10.19, NumPy 1.24.0, SciPy 1.12.0,
  scikit-learn 1.6.0, CVXPY 1.5.0: **485 tests passed**. The imported package
  was the installed wheel under the isolated environment's site-packages,
  not the editable source tree.
- Ruff checks and formatting checks passed.
- Source distribution and wheel built successfully. The source distribution
  includes examples, model documentation, and citation metadata.
- All four usage scripts executed successfully: basic, missing/selection,
  structured, and generalized models.

The numerical suite includes independent CVXPY optima/projections, analytic
solutions, derivative checks, invariant tests, and tests of the actual
returned numerical certificates. Standard sklearn checks run for the base
clusterer. There is one documented expected exception for sample-weight
replication equivalence: zero fidelity masses are unsupported, and duplicating
observations changes the transductive graph objective. The array API check is
not applicable to this NumPy/SciPy implementation. Alternative estimator
wrappers have explicit clone, pipeline, fitted-state, and warning tests.

## Benchmarks

`benchmark_results.jsonl` contains 48 runs: 100/500 observations, 3/20 features,
gamma 0.05/0.5/5, and four solvers. Each uses the same seeded Gaussian generator,
a union 10-neighbor Gaussian graph, tolerance 1e-6, and at most 2,000 iterations.

| Method | Runs meeting both convergence criteria |
| --- | --- |
| SSNAL | 12 / 12 |
| ADMM | 11 / 12 |
| AMA | 10 / 12 |
| Accelerated AMA | 11 / 12 |

Every run retains time, traced allocation peak, objective-gap/KKT diagnostics,
and convergence status. This is an initial synthetic benchmark, not a broad
ranking of algorithms or validation of the original paper's large-scale
runtime. Feature scale affects Gaussian weights and therefore problem
geometry; 3D and 20D runs do not represent identical difficulty. Tracemalloc
omits some native allocations. CPU scheduling also affects timings.

## Reproduction

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check src tests examples
ruff format --check src tests examples
python examples/basic_usage.py
python examples/missing_usage.py
python examples/structured_usage.py
python examples/generalized_usage.py
python examples/benchmark.py --samples 100 500 --features 3 20 --gammas .05 .5 5
python -m build
```

Remote CI status is a separate check on the PR. These local results do not
claim that the PR is merged or that a PyPI release has been published.
