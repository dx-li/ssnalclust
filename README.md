This implements the Semismooth Newton Augmented Lagrangian (SSNAL) Algorithm for convex clustering as described in the paper: https://proceedings.mlr.press/v80/yuan18a.html

## Status

The implementation now correctly follows the mathematical formulas from the paper. Recent fixes addressed critical bugs in the SSNCG subproblem solver that were preventing proper convergence.

## Recent Fixes

The following mathematical bugs have been fixed:

1. **Dual proximal operator scaling**: The `proxdual_pU` function now correctly scales projection ball radii by the `tau` parameter (σ·w_ij).

2. **Active-set handling in Hessian**: The semismooth Newton Hessian approximation now correctly zeros out active-set columns, ensuring only inactive constraints contribute to the Newton direction.

## Dependencies

* numpy
* scipy
* pylops

## To Do

* Tests
* Examples
* Performance benchmarking
