# Additional external reference candidates

## PyClustrPath: inspected, not executed

The inspected revision is
[`21e7b2f83805338a3beac1b6083ac3855501d577`](https://github.com/D3IntOpt/PyClustrPath/tree/21e7b2f83805338a3beac1b6083ac3855501d577)
of D3IntOpt/PyClustrPath, resolved from the remote HEAD on 2026-09-05.
Context7 returned no indexed library entry, so the primary repository was
inspected directly. Only repository metadata, source text and a bytecode
header were read. **No upstream package import or numerical solver was run,
no dependencies were installed, and no upstream source was modified.**

The [project documentation](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/README.md)
advertises CPU and CUDA implementations of convex clustering, including
SSNAL, ADMM and fast AMA. Its stated reference environment is Ubuntu 22.04,
Python 3.10.14, PyTorch 2.2.2 and CUDA 12.1. It is a relevant candidate for
an independent implementation comparison, but the inspected checkout is not
yet a verified CPU/macOS reference.

### Feasibility findings

- [`utils.py`](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/utils.py#L8)
  imports `cusparse_extension` unconditionally. Its supplied
  [build script](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/IChol_CUDA/setup.py)
  uses `CUDAExtension`, a `.cu` source and the `cusparse` library. Selecting
  a CPU solver does not bypass this import. The short
  [requirements file](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/requirements.txt)
  is consequently not a complete CPU-only installation specification.
- The [solver package](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/solvers/__init__.py)
  eagerly imports SSNAL. That module explicitly loads
  [`ssncg.cpython-310.pyc`](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/solvers/ssnal.py#L11);
  no corresponding Python source is present in this revision. The file's
  magic bytes are `6f0d0d0a`, versus `cb0d0d0a` in our Python 3.12 environment.
  This is a concrete interpreter-version barrier even for a different solver
  selected through the eager package import.
- A CPU neighbor-search branch exists, but it tests the literal string
  `'cpu'`; the constructor default is a `torch.device('cpu')` object. A future
  invocation should explicitly use `device='cpu'` and verify that branch.
  The [global configuration](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/global_variables.py)
  also sets CUDA environment variables during import and requires its
  initialization routine before normal data processing.

These findings support deferring execution on the current macOS/Python 3.12
environment. They do not establish that the published Linux/CUDA configuration
fails. A later comparison needs a compatible isolated environment or an
explicitly documented upstream-supported CPU packaging fix; silently removing
imports would no longer test the unmodified reference.

### Objective and graph alignment

[`DataProcessor`](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/data_processor.py)
uses **features in rows**, computes the maximum pairwise distance D, and
normalizes observations to A/D. Its default graph is a symmetric union of
neighbor queries with weights `exp(-0.5 * distance_normalized**2)`.
Additional weight normalization is disabled initially; if enabled, it changes
the total weight to sqrt(number of features). Constant data need special care
because the inspected preprocessing divides by D without a zero-diameter
guard. Duplicate ties can also change edge selection; matching only k is
insufficient to establish identical graphs.

The [SSNAL objective](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/solvers/ssnal.py#L134)
is squared Frobenius fidelity with coefficient 1/2 plus a weighted sum of
Euclidean edge norms. Returned paths are multiplied by D. Algebraically, for
the **same realized weights**, its gamma corresponds to
`gamma_ssnalclust = D * gamma_pyclustrpath` in original data coordinates;
original-coordinate objectives are D² times normalized objectives. A
reference adapter should export the exact edge pairs and weights rather than
reconstruct them using this library's default bandwidth. It must also
transpose data/results, fix float precision and record all normalization
settings. The inspected path is an L2-fusion reference, not evidence for this
library's L1, L-infinity or extended loss models.

### Comparison limits to preserve

Preprocessing materializes dense pairwise distances, adjacency/Laplacian
matrices, and an n-by-edge-count incidence array before converting to sparse
storage. Therefore CPU scalability cannot be inferred from the final sparse
operators or compared with our sparse graph construction without measuring
the whole pipeline.

Reference convergence must be independently checked. The inspected
[AMA/fast-AMA](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/solvers/ama.py#L116)
and [ADMM](https://github.com/D3IntOpt/PyClustrPath/blob/21e7b2f83805338a3beac1b6083ac3855501d577/pyclustrpath/solvers/admm.py#L380)
set `solved` using `iter < max_iter` after zero-based `range(max_iter)` loops,
which does not distinguish iteration exhaustion. SSNAL's path array starts
at zero and fills only entries marked solved. A future run must retain
per-point status and independently evaluate the matched objective, primal
feasibility and optimality residuals; an output array alone is insufficient.

**Current evidence status:** source-level feasibility and normalization audit
only. No numerical agreement, runtime, CPU/macOS compatibility or relative
performance result is claimed for PyClustrPath.
