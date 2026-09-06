"""Plot recorded tuning, audit and post-selection results without refitting."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def plot(directory, output):
    directory = Path(directory)
    manifest = json.loads((directory / "study.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Plot requires a completed study; inspect partial checkpoints separately")
    reports = []
    for process in manifest["processes"]:
        path = directory / process["checkpoint"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != process["checkpoint_sha256"]:
            raise ValueError("Worker checkpoint hash mismatch")
        reports.append(json.loads(path.read_text()))
    scenarios = list(dict.fromkeys(r["scenario"] for r in reports))
    seeds = list(dict.fromkeys(r["seed"] for r in reports))
    colors = {seed: color for seed, color in zip(seeds, ["#2266aa", "#dd7722", "#118866"])}
    fig, axes = plt.subplots(3, len(scenarios), figsize=(4 * len(scenarios), 8.8), squeeze=False)
    for col, scenario in enumerate(scenarios):
        for report in [r for r in reports if r["scenario"] == scenario]:
            factors = np.array(report["factors"])
            decision = report["selection"]
            color = colors[report["seed"]]
            scores = np.array([np.nan if x is None else x for x in decision["mean_scores"]])
            axes[0, col].plot(factors, scores, "o-", color=color, ms=4)
            invalid = ~np.isfinite(scores)
            axes[0, col].plot(
                factors[invalid], np.full(invalid.sum(), 1.06), "x", color=color, ms=5
            )
            full = next(s for s in report["samples_fitted"] if s["name"] == "full")
            counts = [p["n_clusters"] for p in full["points"]]
            ari = [p["posthoc_ari"] for p in full["points"]]
            axes[1, col].plot(factors, counts, "o-", color=color, ms=4)
            axes[2, col].plot(factors, ari, "o-", color=color, ms=4)
            index = decision["selected_index"]
            if index is not None:
                for row, values in enumerate([scores, counts, ari]):
                    axes[row, col].plot(
                        factors[index],
                        values[index],
                        "*",
                        color=color,
                        ms=13,
                        markeredgecolor="white",
                        markeredgewidth=0.6,
                    )
                if report["audit_summary"]["eligible"]:
                    axes[0, col].plot(
                        factors[index],
                        report["audit_summary"]["mean_score"],
                        "D",
                        color=color,
                        fillstyle="none",
                        ms=7,
                    )
            else:
                axes[0, col].text(
                    0.03,
                    0.08 + 0.07 * seeds.index(report["seed"]),
                    f"{report['seed']}: no selection",
                    color=color,
                    transform=axes[0, col].transAxes,
                    fontsize=8,
                )
        axes[0, col].set_title(scenario.replace("_", " ").title())
        axes[0, col].set_ylim(-1.1, 1.15)
        axes[1, col].set_ylim(0, reports[0]["samples"] * 1.06)
        axes[2, col].set_ylim(-0.1, 1.07)
        for row in range(3):
            axes[row, col].set_xscale("symlog", linthresh=0.1)
            axes[row, col].set_xticks([0, 0.1, 1, 10, 30], labels=["0", "0.1", "1", "10", "30"])
            axes[row, col].grid(alpha=0.2)
            axes[row, col].set_xlabel("Strength factor")
    axes[0, 0].set_ylabel("Corrected disagreement (lower is better)")
    axes[1, 0].set_ylabel("Full-data cluster count")
    axes[2, 0].set_ylabel("Post-selection ARI")
    handles = [Line2D([], [], color=colors[seed], label=f"seed {seed}") for seed in seeds]
    handles += [
        Line2D([], [], color="black", marker="*", ls="", ms=10, label="Selected"),
        Line2D([], [], color="black", marker="D", fillstyle="none", ls="", label="Audit mean"),
        Line2D([], [], color="black", marker="x", ls="", label="Ineligible tuning factor"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    title = "Corrected partition stability: frozen simulation study"
    if manifest["smoke"]:
        title += " (smoke only)"
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.directory, args.output)
