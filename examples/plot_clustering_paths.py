"""Plot a deterministic 2D convex clustering path and inspect recovery.

Install plotting support with ``python -m pip install -e '.[examples]'``.
Run ``python examples/plot_clustering_paths.py --output artifacts/clustering_paths.png``.
Omit --output to open the figure interactively. This synthetic illustration
uses known labels to choose a readable snapshot; it is not model selection.
"""

import argparse
from pathlib import Path

import numpy as np

from ssnalclust import connected_k_neighbors_graph, convex_clustering_path, summarize_path


def compute_gallery():
    """Return observations, known labels, certified path, and recovery snapshot."""
    rng = np.random.default_rng(27)
    truth = np.repeat(np.arange(3), 8)
    means = np.array([[-2.0, -0.4], [1.1, 1.8], [1.2, -1.7]])
    X = means[truth] + rng.normal(scale=0.18, size=(len(truth), 2))
    weights = connected_k_neighbors_graph(X, n_neighbors=5, bandwidth=1.5)
    gammas = np.r_[0.0, np.geomspace(0.01, 1000, 51)]
    results = convex_clustering_path(
        X, gammas, weights=weights, solver="ssnal", tol=1e-7, max_iter=300
    )
    for gamma, result in zip(gammas, results):
        if not result.converged:
            raise RuntimeError(f"Path point gamma={gamma:g} did not converge: {result.message}")
    summaries = summarize_path(results, cluster_tol=1e-4)
    truth_pairs = truth[:, None] == truth[None, :]
    recovering = [
        i
        for i, point in enumerate(summaries)
        if np.array_equal(point["labels"][:, None] == point["labels"][None, :], truth_pairs)
    ]
    if not recovering:
        raise RuntimeError("This numerical path did not recover the synthetic partition")
    snapshot = recovering[len(recovering) // 2]
    return X, truth, gammas, results, summaries, snapshot


def make_figure(data):
    """Build the Matplotlib figure; numerical work is separate from rendering."""
    import matplotlib.pyplot as plt

    X, truth, gammas, results, summaries, snapshot = data
    colors = np.array(["#2369a1", "#d66d32", "#428562"])
    centroids = np.stack([result.centers for result in results])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), layout="constrained")
    fig.suptitle("Convex clustering: observations become shared centroids", fontsize=17)
    original, trajectories, recovered, counts = axes.ravel()

    for label, color in enumerate(colors):
        group = truth == label
        original.scatter(X[group, 0], X[group, 1], color=color, s=36, label=f"Group {label + 1}")
        trajectories.scatter(X[group, 0], X[group, 1], color=color, s=20, alpha=0.4, zorder=3)
        for index in np.flatnonzero(group):
            trajectories.plot(
                centroids[:, index, 0], centroids[:, index, 1], color=color, alpha=0.58, lw=1
            )
        fitted = results[snapshot].centers[group]
        recovered.scatter(X[group, 0], X[group, 1], color=color, s=24, alpha=0.3)
        for point, center in zip(X[group], fitted):
            recovered.plot([point[0], center[0]], [point[1], center[1]], color=color, lw=0.8)
        recovered.scatter(
            fitted[:, 0].mean(),
            fitted[:, 1].mean(),
            color=color,
            s=160,
            marker="X",
            edgecolor="white",
            linewidth=1,
            zorder=5,
        )
    original.legend(loc="upper left", frameon=False, fontsize=9)
    trajectories.scatter(
        centroids[-1, 0, 0],
        centroids[-1, 0, 1],
        color="#202936",
        marker="*",
        s=160,
        zorder=5,
        label="Final common centroid",
    )
    trajectories.legend(loc="upper left", frameon=False, fontsize=9)
    original.set_title("A  ·  Observed data (known synthetic groups)", loc="left", fontsize=11)
    trajectories.set_title("B  ·  Every fitted centroid across the path", loc="left", fontsize=11)
    recovered.set_title(
        f"C  ·  Recovered partition at gamma = {gammas[snapshot]:.3g}", loc="left", fontsize=11
    )
    for axis in (original, trajectories, recovered):
        axis.set(xlabel="Feature 1", ylabel="Feature 2", xlim=(-2.7, 2.5), ylim=(-2.6, 2.8))
        axis.set_aspect("equal", adjustable="box")

    cluster_counts = [point["n_clusters"] for point in summaries]
    counts.step(gammas[1:], cluster_counts[1:], where="post", color="#283d58", lw=2)
    counts.scatter(gammas[snapshot], 3, color="#d66d32", s=65, zorder=4)
    counts.axvline(gammas[snapshot], color="#d66d32", ls="--", lw=1, alpha=0.7)
    counts.axhline(3, color="#8d969f", ls=":", lw=1)
    counts.set_xscale("log")
    counts.set(
        xlabel="Fusion strength gamma (log scale)",
        ylabel="Numerical cluster count",
        ylim=(0, len(X) + 1),
        yticks=[1, 3, 8, 16, 24],
    )
    counts.set_title("D  ·  Partition size along the evaluated path", loc="left", fontsize=11)
    counts.text(
        0.97,
        0.96,
        "gamma = 0: 24 observations\nThreshold: 1e-4 in feature units",
        transform=counts.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#4c5764",
    )
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.12)
        axis.set_axisbelow(True)
    max_gap = max(result.relative_gap for result in results)
    max_kkt = max(result.kkt_residual for result in results)
    fig.supxlabel(
        f"Fixed connected kNN + MST graph  ·  SSNAL  ·  {len(results)} converged path points\n"
        f"Maximum relative gap {max_gap:.1e}; maximum normalized KKT {max_kkt:.1e}. "
        "Snapshot chosen using known labels for illustration only.",
        fontsize=9,
        color="#4c5764",
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save a figure instead of displaying it")
    args = parser.parse_args(argv)
    if args.output is not None:
        import matplotlib

        matplotlib.use("Agg")
    data = compute_gallery()
    figure = make_figure(data)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.output, dpi=180)
        print(f"Saved {args.output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
    print(f"All {len(data[3])} path points converged; illustrative gamma={data[2][data[5]]:.6g}")
    return figure


if __name__ == "__main__":
    main()
