---
name: Numerical or installation bug
about: Reproduce incorrect results, convergence, performance, or installation behavior
title: ""
labels: ""
assignees: ""
---

**What happened, and what did you expect?**
Include the affected public function/model and whether this is a numerical,
performance, API, or installation problem.

**Small reproducible example**
Include executable code with a fixed random seed, a small synthetic dataset,
graph weights/edge construction, and all solver options. For missing or
generalized models include the mask/loss and data domain. Avoid private data.

```python
# Minimal example
```

**Observed result**
Paste the complete traceback or returned `converged`, `n_iter`, `objective`,
`dual_objective`, `gap`, `relative_gap`, `kkt_residual`, and message fields
when available. Include input dimensions, feature scales, gamma, tolerances,
iteration budget and diagnostic cadence. For performance problems include
wall time, memory, graph size and numerical-library thread settings.

**Expected result or independent reference**
Give a closed form, independently specified objective, reference output, or
explanation. Include reference normalization and tolerance if comparing tools.
It is fine if you do not yet know the cause.

**Environment**
- ssnalclust version or commit; installation method (wheel, sdist, editable):
- Python, operating system/architecture:
- NumPy, SciPy, scikit-learn versions (and CVXPY if used):
- Relevant dependency installation output or `pip freeze`:
