This implements the Semismooth Newton Augmented Lagrangian (SSNAL) Algorithm for convex clustering as described in the paper: https://proceedings.mlr.press/v80/yuan18a.html

As of the current commit, this seems to solve small problems relatively fast. There is a bug in the SSNCG part, however, which causes the algorithm to converge slower than it should.

Dependencies: numpy, scipy, pylops

To do:
* Fix subproblem
* Tests
* Examples