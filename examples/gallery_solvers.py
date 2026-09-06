"""Real Wine chemistry: four solvers, fusion norms, graphs and streamed paths.

Fixed illustrative settings, not a parameter-selection or timing benchmark.
No cultivar labels are loaded. All plots fit in 13 features and project only
for display. Run with --output-dir NEW_DIRECTORY; --smoke uses first24rows.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from heldout_preprocessing import prepare_training
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ssnalclust import (
    ConvexClusteringProblem,
    connected_k_neighbors_graph,
    iter_path_summaries,
    k_neighbors_graph,
    minimum_spanning_tree_graph,
    self_tuning_graph,
    summarize_path,
)

METHODS = ("ssnal", "admm", "ama", "fama")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_problem(smoke=False):
    path = Path(__file__).parent / "data/wine.data"
    raw = np.loadtxt(path, delimiter=",", usecols=range(1, 14))
    if smoke:
        raw = raw[:24]
    neighbors = 5 if smoke else 10
    prepared = prepare_training(raw, np.ones(raw.shape, bool), neighbors)
    return raw, prepared, path, neighbors


def projection(X):
    mean = X.mean(axis=0)
    _, _, vectors = np.linalg.svd(X - mean, full_matrices=False)
    basis = vectors[:2].T.copy()
    for column in range(2):
        if basis[np.argmax(np.abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    return mean, basis


def record(result):
    summary = summarize_path([result])[0]
    return {
        key: summary[key]
        for key in [
            "n_clusters",
            "objective",
            "dual_objective",
            "gap",
            "relative_gap",
            "center_error_bound",
            "kkt_residual",
            "converged",
            "n_iter",
        ]
    }


def run(output_dir, smoke=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    raw, prepared, data_path, neighbors = load_problem(smoke)
    X, W = prepared.training_values, prepared.graph
    problem = ConvexClusteringProblem(X, W)
    gamma, tol = 2.5, 1e-6
    mean, basis = projection(X)
    points = (X - mean) @ basis
    arrays = dict(
        raw_features=raw,
        X=X,
        means=prepared.means,
        scales=prepared.scales,
        graph_indptr=W.indptr,
        graph_indices=W.indices,
        graph_data=W.data,
        pca_mean=mean,
        pca_basis=basis,
    )
    report = dict(
        status="running",
        smoke=smoke,
        samples=len(X),
        features=X.shape[1],
        gamma=gamma,
        tol=tol,
        max_iter=50000,
        cluster_tol=1e-4,
        neighbors=neighbors,
        bandwidth=prepared.bandwidth,
        edges=W.nnz // 2,
        components=prepared.components,
        data_sha256=digest(data_path),
        source_sha256=digest(__file__),
        helper_sha256=digest(Path(__file__).parent / "heldout_preprocessing.py"),
        versions={
            name: importlib.metadata.version(name)
            for name in ["ssnalclust", "numpy", "scipy", "scikit-learn", "matplotlib"]
        },
        fits=[],
        paths=[],
        graph_variants=[],
    )
    fits = {}

    def save():
        np.savez_compressed(output_dir / "solvers.npz", **arrays)
        (output_dir / "solvers.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )

    combinations = [(method, "l2") for method in METHODS] + [("admm", "l1"), ("admm", "linf")]
    for method, penalty in combinations:
        result = problem.solve(
            gamma=gamma,
            solver=method,
            penalty=penalty,
            tol=tol,
            max_iter=50000,
            check_every=1,
            store_history=True,
        )
        name = method + "_" + penalty
        fits[name] = result
        arrays[name + "_centers"] = result.centers
        arrays[name + "_dual"] = result.dual
        arrays[name + "_labels"] = summarize_path([result])[0]["labels"]
        arrays[name + "_iteration"] = np.array([item["iteration"] for item in result.history])
        arrays[name + "_residual"] = np.array(
            [max(item["relative_gap"], item["kkt_residual"]) for item in result.history]
        )
        report["fits"].append(dict(name=name, solver=method, penalty=penalty, **record(result)))
        save()
        print(name, report["fits"][-1], flush=True)
        if not result.converged:
            raise RuntimeError(f"{name} did not converge; inspect saved diagnostics")
    reference = fits["ssnal_l2"]
    report["l2_agreement"] = []
    for method in METHODS[1:]:
        other = fits[method + "_l2"]
        distance = float(np.linalg.norm(reference.centers - other.centers))
        bound = reference.center_error_bound + other.center_error_bound
        report["l2_agreement"].append(
            dict(
                method=method,
                center_distance=distance,
                summed_error_bounds=bound,
                within_bounds=distance <= bound + 1e-10,
            )
        )
        if distance > bound + 1e-10:
            raise RuntimeError("L2 solutions disagree beyond their numerical bounds")
    strengths = [0.0, 0.25, 1.0, 2.5, 5.0]
    source = problem.iter_path(
        strengths, solver="ssnal", tol=tol, max_iter=300, store_history=False
    )
    summaries = iter_path_summaries(source)
    gamma_iterator = iter(strengths)
    try:
        for point in summaries:
            strength = next(gamma_iterator)
            report["paths"].append(
                dict(
                    gamma=strength,
                    **{key: value for key, value in point.items() if key != "labels"},
                )
            )
            if not point["converged"]:
                save()
                raise RuntimeError("Path point did not converge")
            del point
    finally:
        summaries.close()
        source.close()
    graphs = {
        "Gaussian kNN": k_neighbors_graph(X, n_neighbors=neighbors, bandwidth=prepared.bandwidth),
        "Minimum spanning tree": minimum_spanning_tree_graph(X, bandwidth=prepared.bandwidth),
        "Connected kNN": connected_k_neighbors_graph(
            X, n_neighbors=neighbors, bandwidth=prepared.bandwidth
        ),
        "Local scales": self_tuning_graph(X, n_neighbors=neighbors, scale_neighbors=neighbors),
    }
    for index, (name, graph) in enumerate(graphs.items()):
        report["graph_variants"].append(
            dict(
                name=name,
                edges=graph.nnz // 2,
                components=int(connected_components(graph, directed=False)[0]),
            )
        )
        for field in ["indptr", "indices", "data"]:
            arrays[f"variant_{index}_{field}"] = getattr(graph, field)
    report["status"] = "completed"
    save()
    plot_solvers(points, mean, basis, fits, output_dir)
    plot_graphs(points, graphs, output_dir)
    plot_paths(report["paths"], output_dir)
    return report


def plot_solvers(points, mean, basis, fits, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.4))
    for ax, (name, result) in zip(axes.flat, fits.items()):
        fitted = (result.centers - mean) @ basis
        labels = summarize_path([result])[0]["labels"]
        for left, right in zip(points, fitted):
            ax.plot([left[0], right[0]], [left[1], right[1]], color=".8", lw=0.5, zorder=0)
        ax.scatter(points[:, 0], points[:, 1], c=".7", s=9, label="Measured")
        ax.scatter(fitted[:, 0], fitted[:, 1], c=labels, cmap="tab20", s=17, label="Fitted")
        ax.set_title(f"{name.upper().replace('_', ' / ')}: {labels.max() + 1} groups", fontsize=10)
        ax.set_xlabel("Display PC1")
        ax.set_ylabel("Display PC2")
        ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Wine chemistry: same data and strength, different solver/norm choices")
    fig.tight_layout()
    fig.savefig(output_dir / "gallery_solvers.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method in METHODS:
        r = fits[method + "_l2"]
        ax.semilogy(
            [h["iteration"] for h in r.history],
            [max(h["relative_gap"], h["kkt_residual"], 1e-16) for h in r.history],
            label=method.upper(),
        )
    ax.axhline(1e-6, color=".4", ls="--", label="Requested tolerance")
    ax.set(
        xlabel="Outer iterations (different work per method)",
        ylabel="Maximum of relative gap and KKT",
        title="Wine L2 fits: convergence, not a timing comparison",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "gallery_convergence.png", dpi=160)
    plt.close(fig)


def plot_graphs(points, graphs, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (name, graph) in zip(axes.flat, graphs.items()):
        upper = sparse.triu(graph, 1, format="coo")
        for i, j in zip(upper.row, upper.col):
            ax.plot(points[[i, j], 0], points[[i, j], 1], color="#427ca8", alpha=0.18, lw=0.5)
        ax.scatter(points[:, 0], points[:, 1], s=8, c="#182f46")
        count = connected_components(graph, directed=False)[0]
        ax.set_title(f"{name}: {upper.nnz} edges, {count} component(s)", fontsize=10)
        ax.set(xlabel="Display PC1", ylabel="Display PC2")
    fig.suptitle("Wine graph choices: distances use all 13 standardized features")
    fig.tight_layout()
    fig.savefig(output_dir / "gallery_graphs.png", dpi=160)
    plt.close(fig)


def plot_paths(path, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    gammas = [p["gamma"] for p in path]
    axes[0].plot(gammas, [p["n_clusters"] for p in path], "o-", color="#2266aa")
    axes[0].set(ylabel="Thresholded cluster count", title="Streamed SSNAL path")
    axes[1].semilogy(
        gammas, [max(p["relative_gap"], 1e-16) for p in path], "o-", label="Relative gap"
    )
    axes[1].semilogy(gammas, [max(p["kkt_residual"], 1e-16) for p in path], "s-", label="KKT")
    axes[1].axhline(1e-6, ls="--", c=".5")
    axes[1].set(
        title="Every point has its own certificate", ylabel="Diagnostic (display floor 1e-16)"
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Gamma")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "gallery_paths.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.output_dir, args.smoke)
