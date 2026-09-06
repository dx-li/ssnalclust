"""Render saved digits-study evidence without fitting or selecting models.

Usage: python examples/plot_digits_study.py report.json --arrays report.npz
Outputs digits_curves.png and digits_centroids.png in --output-dir.
Only previously computed posthoc ARI scores are read; true-label arrays are
never accessed. Incomplete selections remain unavailable in both figures.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _selected_position(graph):
    selection = graph.get("selection", {})
    position = selection.get("selected_position")
    if selection.get("status") != "valid" or position is None:
        return None
    if not 0 <= position < len(graph["ssnal"]):
        raise ValueError("Saved selected_position is outside the completed path")
    return position


def make_curves(report):
    """Plot saved metrics, marking the frozen silhouette choice for each graph."""
    import matplotlib.pyplot as plt

    factors = np.asarray(report["gamma_factors"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, layout="constrained")
    notes = []
    for graph in report["graphs"]:
        points = graph["ssnal"]
        x = factors[[point["grid_index"] for point in points]]
        selection = graph.get("selection", {})
        values = [
            selection.get("silhouette_scores", [None] * len(points)),
            graph.get("posthoc", {}).get("ari_by_position", [None] * len(points)),
            [point.get("n_clusters") for point in points],
        ]
        position = _selected_position(graph)
        label = f"k={graph['neighbors']}"
        for axis, scores in zip(axes, values):
            if len(scores) != len(points):
                raise ValueError("Saved metrics and path positions have different lengths")
            y = np.asarray([np.nan if value is None else value for value in scores])
            (line,) = axis.plot(x, y, marker="o", ms=4, label=label)
            if position is not None:
                axis.axvline(x[position], color=line.get_color(), ls=":", alpha=0.7)
                if np.isfinite(y[position]):
                    axis.scatter(
                        x[position],
                        y[position],
                        s=115,
                        marker="*",
                        color=line.get_color(),
                        zorder=5,
                    )
        if position is None:
            notes.append(f"{label}: selection {selection.get('status', 'pending')}")
    for axis, title in zip(
        axes,
        [
            "Silhouette (selection criterion)",
            "Posthoc ARI (evaluation only)",
            "Numerical cluster count",
        ],
    ):
        axis.set(title=title, xlabel="Fixed gamma factor (gamma = factor × bandwidth / k)")
        axis.set_xscale("log")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    for axis in axes:
        if not any(np.isfinite(line.get_ydata()).any() for line in axis.lines):
            axis.text(
                0.5,
                0.5,
                "No available metric values",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=10,
            )
    footer = "Stars mark the saved silhouette selection separately for each graph."
    if notes:
        footer += "\n" + "; ".join(notes)
    fig.suptitle(f"Digits study: n={report['samples']} | report status: {report['status']}")
    fig.supxlabel(footer, fontsize=9)
    return fig


def make_centroids(report, arrays):
    """Average optimized centers within predicted groups; order by group size."""
    import matplotlib.pyplot as plt

    graphs = report["graphs"]
    shape = tuple(report["image_shape"])
    fig, axes = plt.subplots(
        max(1, len(graphs)) * 2,
        6,
        figsize=(12, max(1, len(graphs)) * 4),
        squeeze=False,
        layout="constrained",
    )
    for axis in axes.flat:
        axis.set_axis_off()
    if not graphs:
        axes[0, 0].text(0, 0.5, "No completed graph selections", transform=axes[0, 0].transAxes)
    for row, graph in enumerate(graphs):
        group_axes = axes[2 * row : 2 * row + 2].ravel()
        position = _selected_position(graph)
        label = f"k={graph['neighbors']}"
        if position is None:
            status = graph.get("selection", {}).get("status", "pending")
            group_axes[0].text(
                0, 0.5, f"{label}\nSelection {status}", transform=group_axes[0].transAxes
            )
            continue
        point = graph["ssnal"][position]
        key = f"k{graph['neighbors']}_ssnal_{point['grid_index']}"
        if key + "_centers" not in arrays or key + "_labels" not in arrays:
            group_axes[0].text(
                0, 0.5, f"{label}\nSelected arrays unavailable", transform=group_axes[0].transAxes
            )
            continue
        centers = np.asarray(arrays[key + "_centers"])
        labels = np.asarray(arrays[key + "_labels"])
        if centers.shape != (report["samples"], int(np.prod(shape))) or labels.shape != (
            len(centers),
        ):
            raise ValueError("Selected arrays do not match report sample/image dimensions")
        unique, first, counts = np.unique(labels, return_index=True, return_counts=True)
        order = np.lexsort((first, -counts))[:12]
        for rank, (axis, index) in enumerate(zip(group_axes, order)):
            image = centers[labels == unique[index]].mean(axis=0).reshape(shape)
            axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            heading = f"{label}, gamma={point['gamma']:.3g}\n" if rank == 0 else ""
            axis.set_title(f"{heading}Size rank {rank + 1} · n={counts[index]}", fontsize=9)
    fig.suptitle("Saved silhouette selections: largest 12 predicted groups per graph")
    fig.supxlabel(
        "Mean optimized centers within each group; grayscale fixed to [0, 1]. "
        "Ties use first observation order; no truth-based ordering.",
        fontsize=9,
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--arrays", type=Path, help="Defaults to report filename with .npz suffix")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    curves = make_curves(report)
    curves.savefig(args.output_dir / "digits_curves.png", dpi=170)
    plt.close(curves)
    with np.load(args.arrays or args.report.with_suffix(".npz"), allow_pickle=False) as arrays:
        centroids = make_centroids(report, arrays)
        centroids.savefig(args.output_dir / "digits_centroids.png", dpi=170)
        plt.close(centroids)


if __name__ == "__main__":
    main()
