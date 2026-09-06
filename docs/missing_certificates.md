# Missing-data certificates from an equivalent bounded problem

`solve_missing` reports `dual_objective`, `gap`, and `relative_gap`, with
`certificate_model="observed_range_box"`. This is a certificate for an
equivalent bounded formulation. It is not a claim that the returned edge
multiplier satisfies the domain of the unconstrained masked-loss conjugate.
Success requires both this relative gap and the original KKT residual to
meet `tol`.

## Why observed-range bounds preserve the optimum

The original objective is

```text
P(U) = 0.5 * sum_observed (U_ij-X_ij)^2
       + sum_e r_e * ||(BU)_e||_q,    q in {1,2,infinity}.
```

Form connected components using edges with positive represented penalty
`r_e=gamma*w_e`. Within each component and feature, let `L` and `H` be the
minimum and maximum observed values. At least one observation per component
and feature is required. Every node in that component receives the same
interval for that feature.

Clip each coordinate of an arbitrary `U` to its interval. For an observed
entry, its datum lies in the interval, so clipping cannot increase squared
error. Endpoints of a positive edge use the same coordinate intervals;
scalar clipping is nonexpansive, so every absolute coordinate difference
on that edge can only decrease. The l1, l2, and linfinity norms are monotone
in those absolute coordinates. Thus clipping cannot increase the fusion
penalty either.

Consequently, restricting `L<=U<=H` leaves the optimal objective value
unchanged. Every minimizer of the restricted problem is a minimizer of the
original problem. The full sets of minimizers need not coincide: a missing
coordinate can move outside the box without changing an inactive coordinate
of a linfinity edge norm. The box may select among nonunique original
minimizers; it does not add imputed observations or a ridge penalty.

This argument relies on common intervals along active edges and the
coordinatewise monotonicity of the supported norms. It does not automatically
extend to arbitrary rotated norms or node-specific intervals. Zero-penalty
edges do not join components. At gamma zero, every vertex is its own
component; all entries must then be observed. A floating-point product
`gamma*w_e` that underflows to zero is treated as a zero represented penalty.

## The bounded dual

Let each edge multiplier `Z_e` lie in the ball dual to the selected edge
norm, with radius `r_e`, and set `A=B.T@Z`. Introduce split edge differences
and minimize their Lagrangian over primal coordinates within the box.
The resulting dual lower bound is

```text
D_box(Z) = sum_observed min_{L<=t<=H} [0.5*(t-X)^2+A*t]
           + sum_missing min_{L<=t<=H} A*t.
```

The scalar minimizers are explicit:

```text
T_observed = clip(X-A,L,H)
T_missing  = L if A>=0, otherwise H.
```

Hence any edge-ball-feasible `Z` gives a finite lower bound, whether or not
`A` vanishes at missing entries. By weak duality and the equal-optimal-value
argument above,

```text
D_box(Z) <= min_{L<=U<=H} P(U) = min_U P(U) <= P(U_current).
```

For comparison, the ordinary unconstrained masked-loss conjugate is finite
only when `A=0` at missing coordinates. Using that ordinary conjugate with
approximately zero divergence would not establish a valid finite lower
bound. The observed-range box is the justification for the different
certificate here.

An independent extended dual can also introduce nonnegative lower-bound
and upper-bound multipliers `lambda` and `mu`. Set
`C=A-lambda+mu`, constrain `C_missing=0`, and maximize

```text
sum_observed [X*C-0.5*C^2] + <lambda,L> - <mu,H>,
```

subject to the edge balls and nonnegative box multipliers. The tests solve
this extended dual independently with CVXPY.

The positive sign on `X*C` follows from minimizing
`0.5*(U-X)^2+C*U` at `U=X-C`, which gives `X*C-0.5*C^2`.
For example, `X=2,C=1` yields `1.5`. A regression fixes nonzero lower and
upper multipliers and compares this formula against direct evaluation of
the full Lagrangian, including a missing coordinate with zero combined
coefficient.

## Stable gap and returned coordinates

For `U` inside the box, subtracting `D_box` from `P` decomposes into
nonnegative scalar data terms and edge Fenchel slacks. If `d=U-T`, the data
terms are

```text
observed: 0.5*d^2 + d*(T-X+A)
missing:  A*d.
```

The sign conditions for a scalar minimizer on an interval make these terms
nonnegative. Add `r_e*||(BU)_e||_q - <Z_e,(BU)_e>` for every edge to obtain
the gap. This avoids subtracting nearly equal complete objective values.

The PDHG primal update is the masked quadratic proximal map followed by
coordinate clipping. This is the exact proximal map of the masked
quadratic plus the box indicator, because the scalar unconstrained
minimizer is simply projected to its interval.

The optimizer works after componentwise translation for numerical reasons.
Diagnostics nevertheless evaluate the actual restored, representable
centers against the original observed data. Re-subtracting a large offset
from both arrays can erase low bits and report a different residual.
The tests explicitly check this distinction on a range spanning `-1e16`
to `1`.

For observed scalar dual minimizers, the implementation represents
`T=X+displacement` without rounding this sum prematurely, where
`displacement=clip(-A,L-X,H-X)`. The dual's linear term uses the graph
adjoint identity to cancel common offsets before multiplication. An
independent 70-digit Decimal calculation verifies this lower bound after
different graph components receive translations of order `1e9`.

The relative gap is
`gap/(1+abs(objective)+abs(dual_objective))`. These are floating-point
optimization certificates, not interval-arithmetic bounds. Extreme
ranges may prevent the solver from meeting the requested tolerance; finite
but unconverged output is preferable to certifying a different rounded
problem.

## Why the original KKT check remains separate

A zero boxed gap certifies the primal objective value, but its edge
multiplier need not itself be an original-problem KKT multiplier. For
example, if a component feature has a single observed value `3`, all its
box intervals collapse to `{3}`. Any feasible edge multiplier gives zero
boxed gap at constant `U=3`, even if its divergence is nonzero. That
multiplier is balanced by box normals in the bounded formulation.

At noncollapsed bounds, an extremal-set argument is helpful. At the lower
bound, sum stationarity over all nodes occupying that bound in a component
and feature. Observed gradients and outgoing penalty-coordinate flows are
nonpositive, whereas box KKT requires their nodewise sums to be
nonnegative. Thus all these normal contributions must vanish. The upper
bound is analogous. This uses the sign structure of the supported absolute
norms. In collapsed features, zero-initialized dual coordinates remain
zero under the implemented projections. Nevertheless, the solver retains
the original stationarity and edge subgradient residuals as explicit tests,
rather than relying solely on this reasoning or the boxed gap.

The masked loss is not strongly convex in all coordinates. A small gap
therefore bounds objective error without implying a unique missing-value
estimate or a universal centroid-distance error bound.

## Independent validation

`tests/test_missing_certificates.py` checks all three norms against original
and bounded CVXPY primal formulations and an independently constructed
extended dual. Cases include asymmetric component ranges, singleton
components, collapsed coordinates, missing-value nonuniqueness, and an
original linfinity optimum lying outside the box.

Further tests cover finite ignored placeholders, zero-gamma links,
underflowed fusion links between distinct components, independent large
translations, exact returned-coordinate residuals, and a budget-limited
solve whose initial normalized KKT is tiny but whose gap is large. The
tests explicitly distinguish a valid boxed lower bound from an infeasible
ordinary masked dual.
