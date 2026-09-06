# Certificates for generalized fidelity

`solve_generalized` certifies each Huber, logistic, or Poisson solution using
the corresponding convex conjugate. The result and every history record
include `dual_objective`, `gap`, `relative_gap`, and `dual_scale`. Both the
relative gap and normalized KKT residual must meet `tol` to set `converged`.

These fields are appended with defaults to the prior result dataclass;
existing positional construction remains possible. Solver-produced results
populate them with finite values.

## Dual domain and objective

All three models have the form

```text
P(U) = f(U;X) + p(BU),
p(V) = sum_e r_e ||V_e||,
r_e = gamma*w_e.
```

Let `Z` have one row per edge and let `S=B.T@Z`. Minimizing the Lagrangian
over the edge differences requires each `Z_e` to lie in the dual-norm ball
of radius `r_e`. Minimizing over `U` gives the dual `D(Z)=-f*(-S;X)`.
The conjugates below follow by scalar maximization over the natural parameter.

| Fidelity per entry | Additional dual domain | Dual contribution |
| --- | --- | --- |
| Standard Huber_delta(U-X) | `abs(S)<=delta` | `X*S - S²/2` |
| `log(1+exp(U))-X*U` | `q=X-S` in `[0,1]` | `-q*log(q)-(1-q)*log(1-q)` |
| `exp(U)-X*U` | `q=X-S>=0` | `q-q*log(q)` |

`0*log(0)` is defined as zero. Huber uses the `r²/2` convention in its
quadratic region. Poisson omits the data-only log-factorial constant in
both primal and dual values; consequently its objectives may be negative.

An edge-ball projection alone does not enforce the additional nodewise
domain. Reporting one of these formulas with infeasible nodewise arguments
would not give a valid lower bound.

## A globally scaled feasible multiplier

The internal PDHG multiplier is first projected onto its edge balls.
The certificate then uses `Z_certificate = alpha*Z`, with `0<=alpha<=1`.
Shrinking preserves every edge-ball constraint. With the unscaled
divergence `S`, the largest permissible scale is bounded as follows:

```text
Huber:
    alpha <= delta/max(abs(S)), when the maximum is nonzero.

Logistic:
    alpha <= X/S             for entries with S>0,
    alpha <= (1-X)/(-S)      for entries with S<0.

Poisson:
    alpha <= X/S             for entries with S>0.
```

The implementation takes the minimum of one and all applicable bounds.
When scaling is needed it rounds toward zero. It then recomputes divergence
from the actual scaled array and verifies the domain directly, including
`X-1 <= S <= X` for logistic loss. Checking the divergence directly avoids
hiding tiny domain violations when a subtraction rounds `X-S` to zero or one.
If graph-reduction roundoff defeats a candidate, a few further inward
scalings are attempted before falling back to the feasible zero multiplier.

`dual_scale` reports the resulting scale. `result.dual` is the certificate
multiplier, and its KKT residual is evaluated with that same returned array.
The internal PDHG update continues using its own unscaled state.

Boundary observations can make this repair weak. If a Bernoulli-zero
observation has positive divergence, every strictly positive scale remains
infeasible. The same problem arises for a Poisson zero or a Bernoulli one
with the opposite sign. Here `alpha=0` is valid but may give a very loose
bound. A nearly stationary raw iterate is not reported as converged unless
the actual returned certificate also meets the requested gap and KKT tests.
This implementation does not replace global scaling with a more expensive
projection onto the full intersection of domains.

## Stable gap evaluation

The exact gap decomposes into nonnegative Fenchel terms:

```text
P(U)-D(Z) = [f(U)+f*(-S)+<U,S>]
           + sum_e [r_e*||BU_e|| - <Z_e,BU_e>].
```

For Huber, put `r=U-X` and `c=clip(r,-delta,delta)`. The data contribution
is evaluated as

```text
0.5*(c+S)^2 + max(abs(r)-delta,0)*(delta+sign(r)*S).
```

For logistic loss it is Bernoulli relative entropy between `q=X-S` and
`sigmoid(U)`. For Poisson it is the generalized scalar KL divergence
`q*log(q/exp(U))-q+exp(U)`. All scalar contributions are summed.

Direct KL evaluation can lose a tiny positive gap when large quantities
nearly cancel. For `t=(q-mean)/mean` close to zero, the implementation uses

```text
KL(q,mean) = mean * [t²/2 - t³/6 + t⁴/12 - t⁵/20 + ...].
```

When `exp(U)` underflows but `U` remains finite, a log-space evaluation
retains a finite KL contribution instead of treating the mathematical mean
as exactly zero. Logistic success and failure probabilities are evaluated
separately to retain the small tail probability.

Only local cancellation-size negative edge slacks are rounded to zero;
larger violations raise an error. The nonnegative decomposition avoids
subtracting the full primal and dual objective values. Therefore
`objective-dual_objective` may differ slightly from the more accurately
evaluated `gap` at finite precision.

The normalized gap is
`gap/(1+abs(objective)+abs(dual_objective))`. The original requested KKT
tolerance is not relaxed for large counts. A machine-accurate scalar
Poisson proximal root can therefore produce a finite but unconverged solve
when floating-point natural parameters cannot meet the global tolerance.

## Independent checks

The generalized test suite includes separate CVXPY maximizations of all
three constrained duals with all three edge norms, independently assembling
the incidence matrix and fidelity-domain constraints. Returned dual bounds
are also checked against independent primal oracle solutions.

Additional regressions verify global domain repair, boundary wrong-sign
fallback, positive KL gaps against 80-digit Decimal arithmetic, exponent
underflow, and large-count precision limits. Tests also construct logistic
and Poisson points with gradients near `1e-8` but a relative gap above one
half: a small gradient alone cannot certify those points. A large-data
Huber example similarly prevents KKT-only acceptance at iteration zero.

These certificates bound objective error, not distance to a unique Huber
solution; Huber solutions can be nonunique. No global strong-convexity
parameter is assumed for the likelihood models. The formulas are numerical
certificates rather than interval-arithmetic bounds.
