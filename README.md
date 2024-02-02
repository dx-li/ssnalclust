This implements the Semismooth Newton Augmented Lagrangian (SSNAL) Algorithm for convex clustering as described in the paper: https://proceedings.mlr.press/v80/yuan18a.html

As of the current commit, this seems to solve most problems relatively fast and correctly. There is a bug in the SSNCG part, however, which causes the subproblem to not converge, leading to the overall convergence being slower than it could be.

Dependencies: numpy, scipy, pylops

To do:
* Fix subproblem
* Tests
* Examples