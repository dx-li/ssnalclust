# External reference: cvxclustr AMA conventions

This audit inspected CRAN's **cvxclustr 1.1.1**, GitHub mirror commit
`d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf`, on 2026-09-05. Context7's
`library cvxclustr` lookup returned no indexed library, so the evidence below
comes directly from the package's R wrappers and C implementation. The
reference is an independent implementation, not an unquestioned certificate
oracle. Its [DESCRIPTION](https://github.com/cran/cvxclustr/blob/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf/DESCRIPTION)
identifies the version and CC BY-NC-SA 4.0 license; this audit describes behavior
without incorporating reference implementation code.

## Objective and array mapping

For samples in columns, the intended problem is

$$
 F_\gamma(U)=\tfrac12\sum_{i=1}^n\|U_{:,i}-X_{:,i}\|_2^2
       +\gamma\sum_{i<j}w_{ij}\|U_{:,i}-U_{:,j}\|_q.
$$

There is no normalization by sample count, feature count, or edge count, and
each undirected edge appears once. This is equation (1.1) of Chi and Lange,
[Splitting Methods for Convex Clustering, arXiv v2](https://arxiv.org/pdf/1304.0499v2),
and is implemented correctly for `type=2` by `loss_primal_L2` in the
[C source, lines 21–41](https://github.com/cran/cvxclustr/blob/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf/src/cvxclustr.c#L21-L41).
The package only exposes `type=1` (l1 fusion) and `type=2` (l2 fusion); the
routine name `update_Lambda_ama_Linf` refers to the *dual projection* for l1
fusion, not support for primal linf fusion.

| Quantity | cvxclustr | ssnalclust mapping |
| --- | --- | --- |
| Data | `X`, features × samples | `X_R = X_python.T` |
| Centers | `U`, features × samples | `centers = U_R.T` |
| Full weights | vector of length n(n−1)/2 | symmetric graph, each edge once |
| Active multipliers | `Lambda`, features × positive edges | `Z = -Lambda_R.T` |
| Difference auxiliary | `V`, features × positive edges | same edge orientation, transpose |
| Regularization | `gamma*w` | same gamma and weights, no rescaling |

Comparisons here use unit sample fidelity weights; the R interface has no
sample-weight argument. It does not automatically center or standardize X.

## Exact edge order and indexing

The full weight vector enumerates `(1,2), (1,3), …, (1,n), (2,3), …, (n−1,n)`.
The one-based position is

$$
 k=n(i-1)-i(i-1)/2+j-i,\qquad 1\le i<j\le n.
$$

For n=4 this is `(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)`. This follows from
`tri2vec`, `vec2tri`, and `compactify_edges` in
[R/util.r](https://github.com/cran/cvxclustr/blob/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf/R/util.r),
and independently from the nested loops of C `kernel_weights`.
For a symmetric R weight matrix, extracting its lower triangle in R's
column-major order produces this order. Extracting its upper triangle in
column-major order generally does not. Build the list of pairs explicitly
when transferring weights across languages.

`cvxclust_path_ama` takes the **full** vector, retains positions `w>0`, and
uses that retained order for `Lambda`, `V`, and the compact edge list. Zero
weights have no multiplier columns. Its internal call to `cvxclust_ama` passes
only positive weights, with `ix`, `M1`, and `M2` reduced to zero-based indices;
`s1` and `s2` remain counts. Direct calls must follow this convention. The
compactification helper is internal and may be accessed with R's `:::` when
building a direct-call audit. See the
[path wrapper](https://github.com/cran/cvxclustr/blob/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf/R/cvxclust_path_ama.r).

## Multiplier sign and independently recomputed certificate

Let B have row `e_i-e_j` for i<j, and write L = `Lambda_R.T` in sample-row
notation. The package uses the Lagrangian term `⟨L,V−BU⟩`. Consequently its
AMA reconstruction is `U=X+B.T L` and its update projects
`L−nu*B U` onto balls of radius `gamma*w`. Both signs are explicit in C
`update_U_ama` and `update_Lambda_ama_L2` / `update_Lambda_ama_Linf`.
The library's convention `⟨Z,BU−V⟩` therefore requires **Z=−L**.

For a dual-feasible L, C `loss_dual` computes

$$
 D(L)=-\tfrac12\|B^T L\|_F^2-\langle BX,L\rangle.
$$

Equivalently, for the mapped Z,

$$
 D(Z)=\langle BX,Z\rangle-\tfrac12\|B^T Z\|_F^2,
 \qquad \|Z_e\|_{q^*}\le\gamma w_e.
$$

Recompute the weighted primal at the **returned** centers and the dual at
the **returned** mapped multipliers; do not reconstruct centers from the
final multiplier and silently compare a different iterate. A stable gap is

$$
 G=\tfrac12\|U-X+B^TZ\|_F^2+
   \sum_e\bigl(\gamma w_e\|(BU)_e\|_q-\langle Z_e,(BU)_e\rangle\bigr).
$$

Check ball feasibility separately. C `loss_dual` omits the indicator of the
dual domain because its updates project onto that domain. Finite arithmetic
still requires numerical feasibility checks. These are floating-point
diagnostics, not interval-arithmetic bounds. The formulas agree with the
paper's dual equation (3.12) and absolute-gap criterion in section 3.4.1;
source implementation details below are established by the package itself.

## Reported gap and stopping: source-level discrepancies

1. **Weighted l1 primal is wrong in the reference.** C `loss_primal_L1`
   (lines 43–58) sums absolute edge differences without multiplying by w.
   Dual projection and final V proximal updates do use `gamma*w`. Its
   reported primal and gap therefore describe inconsistent objectives when
   positive weights differ from one. For example X=(0,2), one edge w=2,
   gamma=0.1 has weighted optimum U=(0.2,1.8), primal=dual=0.36. The faulty
   primal at this optimum is 0.20 and the reported gap is −0.16. Premature
   stopping is possible, including before the actual weighted optimum.
   Use independent weighted certificates for l1; unit positive weights avoid
   this particular bug. The l2 reporting routine includes w correctly.

2. **The actual stopping test is an absolute gap:** `fp-fd < tol`, with no
   KKT criterion, absolute value, or denominator. The accelerated routine's
   relative-gap test is commented out. Equal numeric tolerances between
   packages therefore do not imply equal convergence requirements.
   ssnalclust requires both KKT and a gap divided by
   `1+abs(primal)+abs(dual)`. Compare independently recomputed certificates
   against common targets instead of treating either package's iteration
   count as a matched-accuracy benchmark.

3. **Early-stop iteration and history indexing is off by one.** C assigns
   `iter=its`, where `its` is a zero-based index when it breaks. The wrapper
   slices `primal[1:iter]` and `dual[1:iter]`, dropping the final, terminating
   history entry. A first-update break gives `iter=0` and R's `1:0` index
   expression, not a conventional empty history. On budget exhaustion,
   `its=max_iter`, and history length matches updates. No separate success
   flag is returned. Do not infer the final gap from the last exposed entry.

4. **Returned U and Lambda come from different update stages.** Plain AMA
   forms U from the old Lambda, then projects to a new Lambda and evaluates
   primal(U), dual(new Lambda). Accelerated AMA forms U from an extrapolated
   multiplier and returns the subsequent projected multiplier. A gap between
   these feasible primal and dual points is valid for l2 (and unit-weight
   l1), even though stationarity need not hold exactly. The final V is
   `prox_{gamma*w/nu}(BU−Lambda/nu)` and is not necessarily exactly BU at a
   finite iterate. Recompute center differences when comparing objectives;
   V's exact zero pattern is a separate cluster-extraction convention.

These conclusions follow from
[C AMA routines, lines 230–327](https://github.com/cran/cvxclustr/blob/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf/src/cvxclustr.c#L230-L327)
and the
[single-solve R wrapper](https://github.com/cran/cvxclustr/blob/d74d9e2c4356ca05ad5ca6e1a9102144ed68b9bf/R/cvxclust_ama.r).

## Step size, acceleration, and paths

The implementation uses a fixed nu, with no backtracking despite generic
documentation describing an initial backtracking step. If `nu>=2/n`, the
wrapper warns and replaces it by `max(AMA_step_size(w,n),1/n)`. The helper
uses graph degrees from positive-weight support, not weighted degrees;
weights belong in the dual ball radii. For sparse direct calls its helper
receives the already compressed w, whereas its indexing expects full-vector
positions. Avoid that replacement path in comparison runs: choose an
explicit positive `nu<2/n` or compute the helper from the full weight vector
before making a direct call.

Both entry points default to `accelerate=TRUE`. The accelerated C routine
performs two initial plain updates without testing the gap, then uses
momentum `(its−1)/(its+2)` and tests the absolute gap. It unconditionally
writes two initial history entries: **do not call it with max_iter<2**.
Matching the label FAMA does not imply identical momentum initialization,
restart behavior, or reported iteration counts across implementations.

The path initializes Lambda to zero once, visits gamma in supplied order,
and warm-starts each solve with the previous returned Lambda. It does not
sort gamma or stop after complete fusion (that early-exit block is commented
out). It carries the actual returned nu forward. Its output contains lists
U/V/Lambda, `nGamma`, `iters`, and `call`; unlike the single-solve output, it
does **not** expose primal/dual histories. For descending gamma, the carried
initial multiplier need not satisfy the new ball constraints, although each
subsequent projection enforces them. Record the gamma sequence, acceleration,
actual step size, budget, and independently measured final certificates with
any external comparison.
