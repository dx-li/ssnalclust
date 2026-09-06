"""Convex clustering with sparse solvers and scikit-learn estimators."""

from .estimator import ConvexClustering, convex_clustering_path, iter_convex_clustering_path
from .generalized import GeneralizedResult, solve_generalized
from .graph import graph_from_weights, k_neighbors_graph
from .graph_extra import connected_k_neighbors_graph, minimum_spanning_tree_graph, self_tuning_graph
from .legacy import SSNAL
from .missing import MissingResult, solve_missing
from .model_estimators import (
    ConvexBiclustering,
    GeneralizedConvexClustering,
    MissingConvexClustering,
    SparseConvexClustering,
)
from .path import summarize_path
from .problem import ConvexClusteringProblem
from .selection import SelectionResult, select_gamma
from .solvers import SolverResult, solve
from .structured import StructuredResult, solve_biclustering, solve_sparse

__version__ = "0.1.0"
__all__ = [
    "ConvexClusteringProblem",
    "iter_convex_clustering_path",
    "summarize_path",
    "minimum_spanning_tree_graph",
    "connected_k_neighbors_graph",
    "self_tuning_graph",
    "SSNAL",
    "ConvexClustering",
    "SolverResult",
    "solve",
    "convex_clustering_path",
    "graph_from_weights",
    "k_neighbors_graph",
    "MissingResult",
    "solve_missing",
    "GeneralizedResult",
    "solve_generalized",
    "StructuredResult",
    "solve_biclustering",
    "solve_sparse",
    "SelectionResult",
    "select_gamma",
    "ConvexBiclustering",
    "GeneralizedConvexClustering",
    "MissingConvexClustering",
    "SparseConvexClustering",
]
