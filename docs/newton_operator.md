# Newton operator: algebra, numerical range and implementation

SSNAL applies conjugate gradients to a generalized Hessian of the reduced
augmented-Lagrangian objective. This operator remains matrix free:

```text
H[D] = M*D + sigma * B.T @ J[B@D].
```

Here `B` is the oriented incidence matrix, `M` contains positive fidelity
masses, and `J` is block diagonal over edges. Each block is a generalized
Jacobian of projection onto the Euclidean ball of radius `r=gamma*w`,
evaluated at `q=sigma*(B@U)+Z`.

For a positive radius, the selected block is the identity when `||q||<=r`.
Outside the ball it is `(r/||q||)*(I-e*e.T)`, where `e=q/||q||`. At positive
radius equality the implementation retains the identity selection. A
zero-radius ball is a constant map, so its block is zero even at the origin.
These choices preserve symmetry and positive semidefiniteness of `J`;
positive fidelity masses make `H` positive definite. The
[original algorithm audit](algorithm_audit.md) explains why retaining the
interior identity contribution is essential.

## Equivalent application without repeated edge gathers

Set `e=0` on interior edges and `a=1` there. On exterior edges use the unit
normal `e` and `a=r/||q||`. Override `a=0` for zero-radius edges. The same
block action is then

```text
J[v] = a * (v-e*dot(e,v)).
```

This permits one rowwise inner product and full-array operations per CG
application. Previously the implementation repeatedly gathered exterior rows
from both `e` and `B@D`, multiplied them, reduced them, and scattered the
result back. The new representation removes those repeated boolean gathers.
It changes the arithmetic order but not the generalized Hessian or its
Jacobi preconditioner:

```text
diag(H) = M + sigma * (B**2).T @ (a*(1-e**2)).
```

`B**2` here means entrywise squaring. This formula is also checked for
nonunit incidence entries, preventing a shortcut that would accidentally
assume all stored values have unit magnitude. The implementation uses
NumPy's rowwise Einstein contraction; current primary NumPy source was
checked through Context7. There is no dense Hessian assembly in production.

Changing floating-point reduction order may change CG stopping or the outer
iteration trajectory. Performance comparisons therefore report achieved
certificates and actual iteration counts, rather than assuming identical
trajectories or equating a microbenchmark with whole-solver performance.

## Normalize before computing a norm

Squaring finite `q` entries can overflow for large magnitudes or underflow
for tiny ones. An incorrect norm can choose the wrong projection branch or
Jacobian. The revised linearization first computes

```text
m = max(abs(q))
s = q/m
length = sqrt(dot(s,s))
scaled_radius = r/m.
```

For nonzero `q`, `length` stays between one and the square root of the feature
count. Compare `length>scaled_radius`, compute the exterior factor as
`scaled_radius/length`, and normalize `s` by `length`. This avoids forming
`m*length`, whose true value can itself exceed the floating-point maximum.
An overflowing `r/m` means the edge is interior and is represented by an
infinite scaled radius. Zero vectors have an explicit interior selection,
followed by the zero-radius override.

The tests cover both `(maxfloat,maxfloat)` and `(minsubnormal,minsubnormal)`
with the corresponding scalar radius. In both cases the correct exterior
Jacobian is `(I-e*e.T)/sqrt(2)` with `e=(1,1)/sqrt(2)`. Its entries are finite
even though a naive norm cannot reliably represent the calculation.

This improves the numerical range of the Newton linearization, not every
operation in the complete objective. If forming `q` itself produces a
nonfinite value, the operator raises a rescaling error. Extreme observations
can still overflow objective diagnostics or prevent the requested accuracy.
No broader arbitrary-magnitude guarantee is claimed.

## Independent validation

`tests/test_newton_operator.py` assembles small dense edge-block Hessians
independently of production helpers. Its 27 tests check action, symmetry,
the fidelity-mass lower bound, the diagonal preconditioner, scalar and
broadcast masses, nonunit incidence, edgeless graphs, zero radii, exact
boundaries and finite differences of the reduced gradient. Extreme tests
also check scale invariance under factors 1e-200 and 1e200 and explicitly
assembled maxfloat/subnormal blocks.

The existing full convex-clustering suite continues to check returned
solutions, independent primal and dual formulations, external reference
outputs, and convergence diagnostics. See
[measured Newton performance](newton_performance.md) for separately captured
pipeline and microbenchmark evidence.

## Integration and regression evidence

The complete suite passes 673 tests on the current Python 3.12 environment
and the installed wheel on Python 3.10 with minimum dependencies. The four
new scale/extreme-vector regressions were also run against the pinned
pre-change source: all four fail there and pass on the candidate. The
huge-radius/tiny-input interior control passes both versions. Thus the new
cases distinguish the numerical defect rather than merely exercising the
new implementation.

Ruff lint/format, package builds and local documentation links pass. The
performance driver was executed against both source trees, and all eleven
captured measurement records completed successfully. Minimum-version
plotting tests retain upstream Matplotlib/Pyparsing deprecation warnings.
