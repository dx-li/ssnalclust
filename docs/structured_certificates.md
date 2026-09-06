# Primal-dual certificates for structured clustering

`solve_biclustering` and `solve_sparse` now return `dual_objective`, `gap`, and
`relative_gap` in addition to KKT diagnostics. Both the relative gap and the
KKT residual must meet `tol` before `converged` is true. Every history entry
contains the same certificate fields, including iteration zero.

The following derivation applies directly to the objective implemented in
`structured.py`; it does not assume that its iterates are optimal.

## Common formulation

Write either structured objective as

```text
P(U) = 0.5 ||U-X||_F^2 + sum_t p_t(K_t U)
p_t(V) = sum_g r_tg ||V_g||_2.
```

For biclustering the two operators are row differences and column
differences. For sparse clustering they are row differences and transposition,
which organizes a column of centroids as a feature group. In sparse clustering,
`X` in these equations is the internally centered input; the returned `offset`
does not contribute to this objective.

Introduce auxiliary arrays `V_t = K_t U`, with multipliers `Z_t`. Their
Lagrangian is

```text
L(U,V,Z) = 0.5 ||U-X||_F^2
           + sum_t [p_t(V_t) + <Z_t, K_t U-V_t>].
```

Minimizing over `V_t` gives a finite value precisely when each multiplier
group satisfies `||Z_tg||_2 <= r_tg`. Set

```text
S = sum_t K_t* Z_t.
```

Minimization over `U` gives `U = X-S` and the dual objective

```text
D(Z) = <X,S> - 0.5 ||S||_F^2.
```

Every feasible collection of multiplier arrays gives a lower bound:
`D(Z) <= P(U*) <= P(U)` for every primal array `U`. No constraints on `U`
need to be repaired. The multiplier bounds are the same Euclidean balls
onto which each PDHG update projects.

## Evaluation and stopping

Subtracting two almost equal large objective values can erase the gap in
floating-point arithmetic. Expanding the square gives the equivalent form

```text
P(U)-D(Z) = 0.5 ||U-X+S||_F^2
           + sum_t,g [r_tg ||(K_t U)_g||_2 - <Z_tg,(K_t U)_g>].
```

The first term measures stationarity. Each remaining term is nonnegative
by the multiplier's norm bound and the Cauchy-Schwarz inequality, and
measures the corresponding subgradient inclusion. `gap` uses this
decomposition, rather than `max(P-D,0)`. The diagnostic rejects multipliers
outside their balls beyond floating-point tolerance and rejects materially
negative group terms; only a negative value consistent with local subtraction
roundoff is set to zero.

The dual's linear part is evaluated as
`sum_t <K_t X,Z_t>` using the adjoint identity. This avoids multiplying a
large common data offset by positive and negative entries of `S` whose
contributions would cancel only after rounding. This matters particularly
for biclustering, whose difference operators annihilate common offsets.

The normalization is

```text
relative_gap = gap / (1 + abs(objective) + abs(dual_objective)).
converged = relative_gap <= tol and kkt_residual <= tol.
```

These are numerical optimization certificates, not interval-arithmetic
proofs. At extreme data scales, floating-point precision still limits the
achievable tolerance. `objective - dual_objective` may differ slightly from
the more accurately computed `gap` when the objective values nearly cancel.

The quadratic fidelity is 1-strongly convex, so its unique minimizer also
satisfies `||U-U*||_F <= sqrt(2*gap)` in exact arithmetic. A small normalized
gap alone does not imply a small absolute centroid error when objective
values are large; inspect the absolute gap when absolute accuracy matters.

KKT alone was insufficient with the existing normalization. For example,
at an untouched initial array with differences of order `1e8`, a unit
fusion penalty gives a normalized edge residual near `1e-8`. That residual
passes a `1e-6` tolerance even while the initial dual is zero and the relative
primal-dual gap is nearly one. The new joint criterion forces the solver to
continue in this case.

## Validation

`tests/test_structured.py` includes:

- Independent primal CVXPY formulations for both models at weak, intermediate,
  and strong regularization, including disconnected graphs and zero feature
  weights.
- Separate CVXPY maximizations of the constrained dual, constructing
  multiplier contributions explicitly instead of using package operators.
- Independent reconstruction of returned multiplier feasibility, adjoints,
  dual objective, gap and history normalization.
- A regression in which both large-data initial states pass KKT alone but
  fail the relative gap.
- A residual gap of approximately `5e-9` that remains positive even when
  subtracting the reported objectives rounds to zero.
- Explicit rejection of infeasible multiplier diagnostics, common-offset
  invariance, degenerate graphs, and honest iteration exhaustion.

New result fields are appended after the prior fields with NaN defaults,
preserving prior positional construction of `StructuredResult`. Actual
solver returns populate all certificate fields with finite values.

## Why the other loss families need different certificates

The structured certificate follows from the conjugate of a complete-data
quadratic. The existing missing-data and generalized-loss models cannot
reuse it unchanged. With `S = B*Z`, direct conjugate calculations give the
following domains and objectives:

| Fidelity | Additional dual domain | Dual objective |
| --- | --- | --- |
| Masked quadratic | `S=0` at every missing coordinate | `<X,S>` over observed entries, minus `0.5*||S_observed||²` |
| Huber with threshold delta | Every entry `abs(S)<=delta` | `<X,S>-0.5*||S||²` |
| Bernoulli logistic | `q=X-S` lies in `[0,1]` entrywise | `-sum(q*log(q)+(1-q)*log(1-q))` |
| Poisson natural parameter | `q=X-S>=0` entrywise | `sum(q-q*log(q))` |

At the endpoints, `0*log(0)` means zero. All models also require the edge
multiplier ball constraints. Projecting each edge group onto its ball
does not ensure these additional nodewise conditions.

For Huber, shrinking the entire multiplier toward zero can enforce its
additional domain while preserving edge feasibility. With binary or zero
count observations, a wrong-sign divergence at a boundary coordinate may
remain infeasible under every positive global shrinkage factor. Missing
coordinates similarly require exact zero divergence, not merely a small
residual. Useful certificates for those cases therefore require a justified
dual-feasibility repair or a different bound. This audit does not claim that
the structured changes add missing-data or likelihood dual certificates.

The separate [generalized certificate implementation](generalized_certificates.md)
now provides globally scaled Huber and likelihood bounds, including its
explicit weak-bound behavior at boundary observations. Missing-data
divergence constraints remain a separate certification problem.
