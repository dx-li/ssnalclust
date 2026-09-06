"""Render saved Wine entry-holdout tuning and posthoc clustering evidence.

Usage: python examples/plot_wine_holdout.py docs/wine_holdout/study.json
No fitting, factor selection or true-label array access takes place here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_study(path):
    """Load local scalar checkpoints, verifying their recorded hashes if present."""
    study = json.loads(path.read_text(encoding="utf-8"))
    checkpoints = []
    for process in study["processes"]:
        artifact = path.parent / process["checkpoint"]
        if not artifact.is_file():
            checkpoints.append((process, {}))
            continue
        raw = artifact.read_bytes()
        expected = process.get("json_sha256")
        if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Checkpoint hash mismatch: {artifact}")
        checkpoints.append((process, json.loads(raw)))
    return study, checkpoints


def make_figure(study, checkpoints):
    """Plot saved choices only; incomplete candidates remain explicitly labeled."""
    import matplotlib.pyplot as plt

    selection = study.get("selection", {})
    factors = np.asarray(selection["factors"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.3), sharex=True, layout="constrained")
    notices = []
    full = {}
    for process, report in checkpoints:
        name = process["name"]
        if process.get("process_status") != "completed":
            notices.append(f"{name}: {process.get('process_status', 'pending')}")
        if name == "full":
            full = report
            continue
        points = report.get("points", [])
        scores = {p["index"]: p.get("score_mse") for p in points}
        y = [np.nan if scores.get(i) is None else scores[i] for i in range(len(factors))]
        axes[0].plot(
            factors, y, "o-", alpha=0.55, lw=1, ms=4, label=f"Draw seed {report.get('seed', name)}"
        )
        failures = [p["index"] for p in points if not p.get("converged", False)]
        if failures:
            notices.append(f"{name}: unconverged candidate indices {failures}")
        if len(points) != len(factors):
            notices.append(f"{name}: {len(points)}/{len(factors)} candidate records")
    pooled = selection.get("aggregate_mse")
    if pooled is not None:
        axes[0].plot(factors, pooled, "o-", color="#252f40", lw=3, ms=5, label="Pooled tuning MSE")
    baseline = selection.get("baseline_mse")
    if baseline is not None:
        axes[0].axhline(
            baseline, color="black", ls="--", lw=1.3, label="Pooled training-mean baseline"
        )
    full_points = {point["index"]: point for point in full.get("points", [])}
    if len(full_points) != len(factors):
        notices.append(f"Full path: {len(full_points)}/{len(factors)} candidate records")
    for axis, field in zip(axes[1:], ("n_clusters", "posthoc_ari")):
        values = [full_points.get(i, {}).get(field) for i in range(len(factors))]
        y = [np.nan if value is None else value for value in values]
        axis.plot(factors, y, "o-", color="#2369a1", lw=2)
        failed = [i for i, point in full_points.items() if not point.get("converged", False)]
        for i in failed:
            if values[i] is not None:
                axis.scatter(factors[i], values[i], marker="x", s=80, color="#b22222", zorder=5)
        if not np.isfinite(y).any():
            axis.text(
                0.5,
                0.5,
                "No available full-data metrics",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    selected = selection.get("selected_index") if selection.get("status") == "valid" else None
    if selected is not None:
        axes[0].axvline(factors[selected], color="#bd6b20", ls=":")
        if pooled is not None:
            axes[0].scatter(
                factors[selected], pooled[selected], marker="*", s=150, color="#bd6b20", zorder=6
            )
        if full.get("selected_status") == "valid" and full_points.get(selected, {}).get(
            "converged", False
        ):
            for axis, field in zip(axes[1:], ("n_clusters", "posthoc_ari")):
                axis.axvline(factors[selected], color="#bd6b20", ls=":")
                value = full_points[selected].get(field)
                if value is not None:
                    axis.scatter(
                        factors[selected], value, marker="*", s=150, color="#bd6b20", zorder=6
                    )
        else:
            notices.append("Selected full-data refit unavailable or unconverged")
    else:
        notices.append(f"Factor selection: {selection.get('status', 'pending')}")
    failed_full = [i for i, point in full_points.items() if not point.get("converged", False)]
    if failed_full:
        notices.append(f"Full path: unconverged indices {failed_full} (red crosses)")
    for axis, title, ylabel in zip(
        axes,
        (
            "Held-out-entry tuning scores",
            "Full-data numerical cluster count",
            "Full-data posthoc ARI",
        ),
        ("MSE in training-standardized units", "Clusters", "Adjusted Rand index (evaluation only)"),
    ):
        axis.set(title=title, ylabel=ylabel, xlabel="Prespecified gamma factor", xscale="log")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Wine entry-holdout study" + (" · SMOKE ONLY" if study.get("smoke") else ""))
    footer = "Stars mark the saved tuning choice, never an ARI-based choice. "
    footer += "Repeated tuning draws are not independent test estimates."
    if notices:
        footer += "\n" + "; ".join(notices)
    fig.supxlabel(footer, fontsize=8)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/_static/wine_holdout.png"))
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = make_figure(*load_study(args.report))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
