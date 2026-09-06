"""Wine magnesium entry holdout using disjoint, fully observed graph covariates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from heldout_preprocessing import prepare_training

from ssnalclust import select_gamma

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/data/wine.data"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_features(smoke=False):
    # Read feature columns only: the cultivar-label column is never loaded.
    values = np.loadtxt(DATA, delimiter=",", usecols=range(1, 14))
    return values[:24] if smoke else values


def graph_from_covariates(features, neighbors=10):
    covariates = np.asarray(features)[:, [0, 12]]
    return prepare_training(covariates, np.ones(covariates.shape, dtype=bool), neighbors)


def diagnostics(result):
    return {
        key: value.item() if isinstance(value := getattr(result, key), np.generic) else value
        for key in (
            "objective",
            "dual_objective",
            "gap",
            "relative_gap",
            "kkt_residual",
            "converged",
            "n_iter",
            "certificate_model",
        )
    }


def run(output_dir, smoke=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = load_features(smoke)
    target = features[:, [4]]
    prepared = graph_from_covariates(features, 5 if smoke else 10)
    gammas = [1.0, 10.0] if smoke else [0.1, 1.0, 10.0, 100.0]
    report = dict(
        status="running",
        smoke=smoke,
        samples=len(features),
        target="magnesium",
        target_scale="original Wine measurement units; no target standardization",
        graph_covariates=["alcohol", "proline"],
        neighbors=5 if smoke else 10,
        bandwidth=prepared.bandwidth,
        graph_components=prepared.components,
        graph_edges=prepared.graph.nnz // 2,
        gammas=gammas,
        random_state=41,
        validation_fraction=0.1,
        tol=1e-6,
        max_iter=20000,
        data_sha256=digest(DATA),
        source_sha256=digest(__file__),
        preprocessing_sha256=digest(ROOT / "examples/heldout_preprocessing.py"),
        solver_source_sha256={
            p.name: digest(p) for p in sorted((ROOT / "src/ssnalclust").glob("*.py"))
        },
        missingness="simulated entry holdout on complete real measurements",
    )
    arrays = dict(
        raw_target=target,
        covariates=features[:, [0, 12]],
        graph_inputs=prepared.graph_inputs,
        covariate_means=prepared.means,
        covariate_scales=prepared.scales,
        graph_indptr=prepared.graph.indptr,
        graph_indices=prepared.graph.indices,
        graph_data=prepared.graph.data,
    )
    try:
        selected = select_gamma(
            target,
            gammas,
            prepared.graph,
            validation_fraction=0.1,
            random_state=41,
            tol=1e-6,
            max_iter=20000,
        )
        best = int(np.argmin(selected.validation_errors))
        mean = float(target[selected.training_mask].mean())
        baseline = float(np.mean((target[selected.validation_mask] - mean) ** 2))
        report.update(
            status="completed",
            selected_index=best,
            selected_gamma=selected.best_gamma,
            validation_count=int(selected.validation_mask.sum()),
            training_mean=mean,
            baseline_mse=baseline,
            selected_mse=float(selected.validation_errors[best]),
            candidates=[
                dict(gamma=g, mse=float(score), **diagnostics(result))
                for g, score, result in zip(gammas, selected.validation_errors, selected.results)
            ],
            full_refit=diagnostics(selected.best_result),
        )
        arrays.update(
            training_mask=selected.training_mask,
            validation_mask=selected.validation_mask,
            full_centers=selected.best_result.centers,
            full_dual=selected.best_result.dual,
        )
        for index, result in enumerate(selected.results):
            arrays[f"candidate_{index}_centers"] = result.centers
            arrays[f"candidate_{index}_dual"] = result.dual
        plot(
            output_dir / "gallery_missing.png",
            prepared.graph_inputs,
            target,
            selected,
            best,
            mean,
            baseline,
        )
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        np.savez_compressed(output_dir / "gallery_missing.npz", **arrays)
        report["arrays_sha256"] = digest(output_dir / "gallery_missing.npz")
        (output_dir / "gallery_missing.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def plot(path, covariates, target, selected, best, mean, baseline):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hidden = selected.validation_mask[:, 0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    cloud = axes[0].scatter(*covariates[~hidden].T, c=target[~hidden, 0], s=24, cmap="viridis")
    axes[0].scatter(*covariates[hidden].T, marker="x", c="black", label="Held-out magnesium")
    axes[0].set(
        xlabel="Alcohol (standardized)",
        ylabel="Proline (standardized)",
        title="Graph covariates; observed target color",
    )
    axes[0].legend(fontsize=8)
    fig.colorbar(cloud, ax=axes[0], label="Magnesium")
    truth = target[hidden, 0]
    predictions = selected.results[best].centers[hidden, 0]
    axes[1].scatter(truth, predictions)
    low, high = min(truth.min(), predictions.min(), mean), max(truth.max(), predictions.max(), mean)
    axes[1].plot([low, high], [low, high], color="gray", linestyle=":", label="Equality")
    axes[1].axhline(mean, color="darkorange", linestyle="--", label="Training-mean baseline")
    axes[1].set(
        xlabel="Withheld magnesium",
        ylabel="Fitted magnesium",
        title=f"Selected gamma = {selected.best_gamma:g}",
    )
    axes[1].legend(fontsize=8)
    axes[2].semilogx(selected.gammas, selected.validation_errors, "o-")
    axes[2].scatter(
        [selected.best_gamma],
        [selected.validation_errors[best]],
        marker="*",
        s=180,
        color="red",
        zorder=3,
    )
    axes[2].axhline(baseline, color="darkorange", linestyle="--", label="Training-mean baseline")
    axes[2].set(
        xlabel="Gamma",
        ylabel="Tuning MSE (magnesium units squared)",
        title="One shared entry holdout",
    )
    axes[2].legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build/gallery-missing")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    report = run(args.output_dir, args.smoke)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["selected_gamma", "selected_mse", "baseline_mse", "validation_count"]
            }
        )
    )


if __name__ == "__main__":
    main()
