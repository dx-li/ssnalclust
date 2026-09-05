# Audit of the original SSNAL implementation

This audit refers to `ssnalclust.py` at commit
`567f7c697fa303472bdf5b695e0ea3f2737e34a1`. Line numbers below refer to that
historical version, not the replacement library.

Reference: Defeng Sun, Kim-Chuan Toh, and Yancheng Yuan (2021),
[Convex Clustering: Model, Theoretical Guarantee and Efficient Algorithm](https://www.jmlr.org/papers/volume22/18-694/18-694.pdf),
JMLR 22, Sections 5–6. The paper is published under CC BY 4.0. Equations below
use the paper's column-oriented notation; a samples-by-features Python API
transposes these arrays without changing the mathematics.

## Defects found by inspecting the original code

1. **The dual proximal operator changes the optimization problem when sigma
   changes** (lines 145–159, 232, 241). `proxdual_pU(U, tau)` projects onto balls
   of radius `tau * gamma * w`. The conjugate is an indicator function, so
   multiplying it by a positive `tau` does not change its proximal map. The
   correct radius is always `gamma * w`. Consequently the supplied `grad_phi`
   need not even be the gradient of the supplied `phi`.

   A direct scalar-feature example is `A=[0,0]`, one unit-weight edge,
   `gamma=1`, `X=[2,0]`, `Z=0`, `sigma=2`. The old `phi` is 4.5 and its
   derivative with respect to the first coordinate is 3, while the old
   `grad_phi` returns 4 for that coordinate. The correct reduced objective is
   3.75 with derivative 3.

2. **The generalized Hessian has two independent errors** (lines 469–486).
   Its active set tests `gamma*w / norm(D) < 1`, omitting `sigma` from the
   denominator. More seriously, the complement blocks are set to zero. For
   the derivative of a projection onto a ball, an interior block is the
   identity. This drops the entire graph-Laplacian contribution from fused
   edges. With one edge, `sigma=1`, `D=0`, positive radius, and direction
   `H=[1,-1]`, the correct Hessian product is `[3,-3]`; the old code produces
   `[1,-1]`. These defects exist independently of any line-search choice.

3. **The reported gap is not a valid certificate** (lines 161–172, 329–338).
   The reported primal objective uses the current auxiliary `U`, although
   `U` need not equal `B(X)`. It therefore need not upper-bound the true
   optimum. The dual objective also lacks a feasibility check. In exact
   arithmetic the ALM multiplier update with the correct primal prox yields
   a feasible multiplier, but the standalone objective and arbitrary warm
   starts do not guarantee this. Taking the absolute difference hides sign
   problems, and the gap-only branch can declare success with failed KKT
   conditions. For a literal example, `A=X=[2,0]`, `U=0`, `Z=0`, and one edge
   with unit penalty give old primal and dual values both zero, although
   `B(X)-U=2` and the true primal value is 2.

4. **Zero-penalty cases can produce NaN** (lines 140–143, 156–159, 472).
   `gamma=0` is explicitly accepted, but zero columns with zero radius cause
   `0/0`. The original active-set computation also divides by zero at fused
   edges. Handle zero radii and zero norms with masked division and explicit
   branches, including the edgeless graph and identical data.

5. **Diagonal weights create incorrect edges** (lines 95–102). The default
   upper triangle includes the diagonal. For a diagonal entry, assigning
   first +1 then -1 leaves a column with a single -1 instead of a zero
   self-difference. This adds a penalty on an individual centroid. An
   undirected graph should contain only `i < j`; diagonal weights must be
   ignored or rejected consistently. Negative, asymmetric, nonfinite, and
   incorrectly shaped weights and data also lack validation. Extracted
   sparse-matrix weights must be flattened to a one-dimensional edge array.

6. **The line search can run forever or accept an unverified step**
   (lines 530–573). `zoom` has an unbounded loop and no stagnation guard;
   the outer search returns an unevaluated doubled step after exhaustion.
   The implemented strong-Wolfe-style procedure also ignores the `delta`
   argument supplied to SSNCG. Finite Armijo backtracking on the correct
   objective is sufficient and follows Algorithm 2 directly.

7. **The outer/inner accuracy policy does not establish the paper's
   convergence assumptions** (lines 315, 445–465, 350–358). Inner accuracy
   has a fixed floor independent of the requested outer tolerance. Early
   exits can return without achieving that accuracy. Sigma can decrease,
   whereas Algorithm 1 specifies a nondecreasing sequence. Do not claim the
   paper's theorem for an unqualified heuristic variant. Report inner
   failures and the achieved certificate honestly.

8. **The implementation is a prototype rather than a reusable estimator.**
   Its operator metadata does not describe the matrix-oriented action used
   internally; private operator methods bypass the advertised abstraction.
   Fit-time numerical data live in the constructor, `fit` returns a tuple,
   no fitted labels or standard diagnostics exist, and debugging prints
   occur unconditionally. Input ownership and zero iteration handling are
   not consistently defined. The repository initially had no packaging,
   tests, examples, or CI.

## Equations to implement and test

Let each edge be an unordered pair represented once as `i < j`, with
`r_e = gamma * w_e >= 0`. Define `B(X)_e = X_i - X_j` and let `B*` scatter
edge vectors with + at `i` and - at `j`. Thus `B*B` is the graph Laplacian,
and the adjoint identity is `<B(X), Z> = <X, B*(Z)>`.

The objective and a feasible dual lower bound are

```text
p(U) = sum_e r_e ||U_e||_2
P(X) = 0.5 ||X-A||_F^2 + p(B(X))
C = {Z : ||Z_e||_2 <= r_e for every edge}
D(Z) = <A, B*(Z)> - 0.5 ||B*(Z)||_F^2,  Z in C.
```

These also follow directly by minimizing the Lagrangian over `X` and `U`.
Project a candidate multiplier onto `C` before reporting a finite dual lower
bound. Use `P(X) - D(project_C(Z))` for the gap; tolerate only roundoff-size
negative values. A finite gap and KKT residual together expose infeasibility
and stationarity failures.

For fixed positive `sigma` and multiplier `Z`, define

```text
T = B(X) + Z/sigma
U = prox_(p/sigma)(T)
Q = project_C(sigma * B(X) + Z) = sigma * (T-U)
phi(X) = 0.5 ||X-A||_F^2 + p(U)
         + ||Q||_F^2/(2 sigma) - ||Z||_F^2/(2 sigma)
grad_phi(X) = X-A + B*(Q).
```

The Euclidean proximal blocks are vector soft thresholding at radius
`r_e/sigma`; the dual projection radius remains `r_e`. The multiplier update
is `Z_new = Z + sigma*(B(X)-U) = Q`. These identities correspond to
Sections 5.2–5.3, including equations (20)–(21).

For a Hessian-vector product set `v_e = T_e/||T_e||`, and define the edge
operator `M` by

```text
if r_e == 0:
    M_e(h) = 0
elif ||T_e|| <= r_e/sigma:
    M_e(h) = h
else:
    a_e = r_e / (sigma * ||T_e||)
    M_e(h) = a_e * (h - v_e * <v_e,h>)

Hessian(H) = H + sigma * B*(M(B(H))).
```

The boundary branch selects a valid generalized derivative. This is
`I + sigma B*(I-P)B` from equation (22), expressed using the projection
derivative rather than the shrinkage derivative. Its quadratic form lies
between those of `I` and `I + sigma B*B`. Implement it without assembling a
dense `(n*d)` square matrix.

Solve `Hessian(direction) = -grad_phi` by CG with absolute residual at most
`min(eta_bar, ||grad_phi||^(1+tau))`. Start Armijo backtracking at step 1,
multiply by `delta`, and require

```text
phi(X + step*direction) <= phi(X) + mu*step*<grad_phi,direction>.
```

Guard nonfinite values, non-descent directions, CG failures, and exhaustion.
Equation (24)'s sufficient ALM subproblem criterion is
`||grad_phi_k|| <= epsilon_k/max(1,sqrt(sigma_k))` for a summable sequence
`epsilon_k`; for example `epsilon_0/(k+1)^2`. Sigma is positive and
nondecreasing in the stated algorithm. Finite computational budgets and
floating-point floors require explicit nonconvergence diagnostics when the
requested accuracy cannot be reached.

Section 6 uses the following relative KKT checks:

```text
eta_P = ||B(X)-U|| / (1+||U||)
eta_D = sum_e max(0, ||Z_e||_2-r_e) / (1+||A||)
eta   = (||B*(Z)+X-A|| + ||U-prox_p(U+Z)||) / (1+||A||+||U||)
stop when max(eta_P, eta_D, eta) <= tolerance.
```

Other explicitly documented residual normalizations can be valid, but must
not be represented as identical to the paper's experimental criterion.

## Independent correctness checks

- Compare adjoints and Laplacian products against explicit incidence
  matrices, including disconnected graphs.
- Check Moreau decomposition for several `sigma` values, zero radii, zero
  vectors, and points inside/outside/on projection boundaries.
- Check reduced-objective gradients and Hessian products by finite
  differences away from nonsmooth boundaries. The examples above distinguish
  each historical bug.
- For two unit-mass observations with one edge, their fitted difference is
  `prox_(2*r*||.||)(a_1-a_2)` and their midpoint is preserved. This gives an
  exact solution for l1, l2, and linfinity penalties using their respective
  proximal maps, including both unfused and fused regimes.
- Compare small random weighted graphs against an independent convex
  optimization formulation, rather than only comparing two implementations
  sharing the same proximal helpers. Check objective, centroids, feasible
  dual bounds, and KKT residuals.
- Verify zero gamma returns the data, edgeless graphs return the data,
  component means are preserved, translations commute with fitting, and
  Euclidean solutions commute with orthogonal feature transformations.
- Verify max-iteration and inner-solver failures cannot be reported as
  convergence, and that tighter tolerances improve actual certificates.

The paper's detailed SSNAL derivation is for the Euclidean edge norm.
Supporting l1 and linfinity convex clustering is a natural library feature,
but each additional solver or generalized derivative needs its own tests
and an accurate description of its algorithmic scope.

## Cross-check of the later survey

The user also supplied Chi, Molstad, Gao, and Chi (2025),
[The Why and How of Convex Clustering, version 2](https://arxiv.org/html/2507.09077v2).
Its Section 8.2 is useful exposition, but direct algebra reveals apparent
typographical discrepancies. These are this audit's findings, not published
author errata:

- Equation (8.6) omits `gamma*w_ij` from its shrinkage threshold.
- Algorithm 1 updates the multiplier using `v^k`; it requires `v^(k+1)`.
- Algorithm 2's multiplier-update indices lag its newly computed variables.
- Equation (8.8)'s `vtilde = v + z/rho` has the wrong sign for (8.4).
  Differentiating (8.5) gives `U+rho*B*(B(U))=X+B*(rho*V-Z)`.
- The assertion that rho must diverge is stronger than JMLR Algorithm 1,
  which permits a finite limiting penalty.
- Superlinear local convergence in the JMLR theorem concerns semismooth
  Newton iterates; it should not be attributed unqualifiedly to CG.

Implementations should use the consistent Lagrangian derivation above and
the original convergence assumptions.
