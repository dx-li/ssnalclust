"""Real-data illustrations of Huber, logistic and Poisson fusion losses.

Run --output-dir NEW_DIRECTORY after installing the examples extra. --smoke
uses fixed smaller row prefixes. Parameters are illustrative, not selected
using truth labels or a predictive validation score. No download is performed.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse

DATA = Path(__file__).resolve().parent / "data"


def load_votes(path, limit=None):
    """Keep complete vote rows in file order; never retain party labels."""
    ids, values = [], []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            votes = line.strip().split(",")[1:]
            if len(votes) != 16:
                raise ValueError("Expected 16 Congressional vote fields")
            if any(value not in {"y", "n", "?"} for value in votes):
                raise ValueError("Unknown vote symbol")
            if "?" not in votes:
                ids.append(index)
                values.append([int(value == "y") for value in votes])
    X = np.asarray(values, dtype=float)
    ids = np.asarray(ids)
    if limit is not None:
        X, ids = X[:limit], ids[:limit]
    if len(X) < 2:
        raise ValueError("Need at least two complete voting rows")
    # An all-zero/one issue has no finite logistic optimum in a connected graph.
    retained_columns = np.flatnonzero((X.min(axis=0) == 0) & (X.max(axis=0) == 1))
    if not len(retained_columns):
        raise ValueError("No issues with both outcomes in the selected rows")
    return X[:, retained_columns], ids, retained_columns


def load_bikes(path, limit=90):
    """Return original daily counts and dates in strictly chronological order."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))[:limit]
    dates = np.asarray([row["dteday"] for row in rows])
    counts = np.asarray([int(row["cnt"]) for row in rows], dtype=float)[:, None]
    if len(rows) < 2 or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("Daily bike rows must be in strictly increasing date order")
    if np.any(counts < 0):
        raise ValueError("Counts must be nonnegative")
    return counts, dates, np.arange(len(rows))


def chain_graph(samples):
    """Unit adjacency of consecutive calendar observations; no count-based edges."""
    if samples < 2:
        raise ValueError("Need at least two temporal observations")
    return sparse.diags(
        [np.ones(samples - 1), np.ones(samples - 1)],
        [-1, 1],
        shape=(samples, samples),
        format="csr",
    )


def certificate(result, elapsed):
    return dict(
        converged=bool(result.converged),
        objective=result.objective,
        dual_objective=result.dual_objective,
        gap=result.gap,
        relative_gap=result.relative_gap,
        kkt_residual=result.kkt_residual,
        n_iter=result.n_iter,
        seconds=elapsed,
    )


def fit_generalized(X, graph, loss, gamma, max_iter=20000):
    from ssnalclust import solve_generalized

    started = time.perf_counter()
    result = solve_generalized(
        X, weights=graph, loss=loss, gamma=gamma, huber_delta=1.0, tol=1e-6, max_iter=max_iter
    )
    diagnostics = certificate(result, time.perf_counter() - started)
    diagnostics["max_iter"] = max_iter
    return result, diagnostics


def fit_huber(smoke=False):
    from heldout_preprocessing import prepare_training

    from ssnalclust import solve

    path = DATA / "wine.data"
    raw = np.loadtxt(path, delimiter=",", usecols=range(1, 14))
    raw = raw[:24] if smoke else raw
    prepared = prepare_training(raw, np.ones(raw.shape, dtype=bool), neighbors=10)
    X, graph = prepared.training_values, prepared.graph
    gamma = 3.0
    robust, metrics = fit_generalized(X, graph, "huber", gamma)
    started = time.perf_counter()
    squared = solve(X, weights=graph, gamma=gamma, tol=1e-6, max_iter=300)
    comparison = certificate(squared, time.perf_counter() - started)
    arrays = dict(
        raw_features=raw,
        selected_ids=np.arange(len(raw)),
        optimization_data=X,
        feature_means=prepared.means,
        feature_scales=prepared.scales,
        centers=robust.centers,
        fitted_means=robust.fitted_means,
        dual=robust.dual,
        squared_centers=squared.centers,
        squared_dual=squared.dual,
    )
    record = dict(
        loss="huber",
        gamma=gamma,
        huber_delta=1.0,
        samples=len(X),
        features=X.shape[1],
        graph="union 10-neighbor, median positive distance Gaussian",
        bandwidth=prepared.bandwidth,
        preprocessing="all selected feature rows standardized with population mean/std",
        diagnostics=metrics,
        squared_comparison=comparison,
    )
    return path, record, arrays, graph


def fit_logistic(smoke=False):
    from ssnalclust import connected_k_neighbors_graph, summarize_path

    path = DATA / "house-votes-84.data"
    X, ids, columns = load_votes(path, limit=40 if smoke else None)
    graph = connected_k_neighbors_graph(X, n_neighbors=10, bandwidth=2.0)
    gamma = 1.0
    result, metrics = fit_generalized(X, graph, "logistic", gamma)
    summary = summarize_path([result])[0]
    arrays = dict(
        optimization_data=X,
        selected_ids=ids,
        retained_issue_columns=columns,
        centers=result.centers,
        fitted_means=result.fitted_means,
        dual=result.dual,
        labels=summary["labels"],
    )
    record = dict(
        loss="logistic",
        gamma=gamma,
        samples=len(X),
        features=X.shape[1],
        graph="connected union 10-neighbor plus MST edges; Gaussian bandwidth 2",
        bandwidth=2.0,
        excluded_constant_columns=sorted(set(range(16)) - set(columns.tolist())),
        n_clusters=summary["n_clusters"],
        cluster_tol=1e-4,
        row_selection="complete rows in original file order" + (", first 40" if smoke else ""),
        diagnostics=metrics,
    )
    return path, record, arrays, graph


def fit_poisson(smoke=False):
    path = DATA / "bike_day.csv"
    X, dates, ids = load_bikes(path, limit=14 if smoke else 90)
    graph = chain_graph(len(X))
    gamma = 100.0
    result, metrics = fit_generalized(X, graph, "poisson", gamma, max_iter=100000)
    arrays = dict(
        optimization_data=X,
        dates=dates,
        selected_ids=ids,
        centers=result.centers,
        fitted_means=result.fitted_means,
        dual=result.dual,
    )
    record = dict(
        loss="poisson",
        gamma=gamma,
        samples=len(X),
        features=1,
        graph="unit chain in original chronological order",
        count_units="raw daily cnt rentals",
        date_start=str(dates[0]),
        date_end=str(dates[-1]),
        diagnostics=metrics,
    )
    return path, record, arrays, graph


def plot_huber(arrays, record):
    import matplotlib.pyplot as plt

    X, U, Q = arrays["optimization_data"], arrays["centers"], arrays["squared_centers"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    order = np.argsort(X[:, 0], kind="stable")
    axes[0].plot(X[order, 0], ".", color="0.6", label="Measured alcohol")
    axes[0].plot(U[order, 0], lw=1.4, label="Huber fit")
    axes[0].plot(Q[order, 0], lw=1.2, label="Squared-loss fit")
    axes[0].set(
        xlabel="Rows sorted by measured alcohol (display only)", ylabel="Standardized alcohol"
    )
    residual = (U - X).ravel()
    order = np.argsort(residual)
    axes[1].plot(residual[order], residual[order], label="Squared derivative r")
    axes[1].plot(
        residual[order],
        np.clip(residual[order], -1, 1),
        lw=2,
        label="Huber derivative clip(r, −1, 1)",
    )
    axes[1].set(
        xlabel="Actual Huber-fit residuals across all 13 features",
        ylabel="Fidelity derivative per entry",
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    fig.suptitle(f"Wine: bounded Huber fidelity influence | gamma={record['gamma']}, delta=1")
    return fig


def plot_logistic(arrays, record):
    import matplotlib.pyplot as plt

    order = np.argsort(arrays["labels"], kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), layout="constrained")
    for axis, field, title in zip(
        axes,
        ("optimization_data", "fitted_means"),
        ("Observed votes: no=0, yes=1", "Fitted yes probability: sigmoid(centers)"),
    ):
        image = axis.imshow(
            arrays[field][order],
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.set(
            title=title,
            xlabel="Retained vote issue",
            ylabel="Complete rows grouped by fitted cluster",
        )
    fig.colorbar(image, ax=axes, shrink=0.75, label="Vote / probability")
    fig.suptitle(
        f"Congressional votes | gamma={record['gamma']}, {record['n_clusters']} numerical clusters\n"
        "Same row order in both panels; party labels unused"
    )
    return fig


def plot_poisson(arrays, record):
    import matplotlib.pyplot as plt

    dates = arrays["dates"].astype("datetime64[D]")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained")
    axes[0].plot(
        dates,
        arrays["optimization_data"][:, 0],
        ".-",
        color="0.65",
        lw=0.8,
        label="Raw daily rentals",
    )
    axes[0].plot(dates, arrays["fitted_means"][:, 0], lw=2, label="Fitted mean exp(centers)")
    axes[0].set_ylabel("Rentals per day")
    axes[0].legend(frameon=False)
    axes[1].plot(dates, arrays["centers"][:, 0], color="#c26b2c", lw=2)
    axes[1].set(
        xlabel="Calendar date (original chronological chain)", ylabel="Fitted log intensity"
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"Bike rentals: Poisson log-intensity fusion | gamma={record['gamma']:g}\n"
        "Descriptive smoothing; no weather adjustment or forecast evaluation"
    )
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only", choices=("huber", "logistic", "poisson"), help="Render one illustration"
    )
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import ssnalclust

    args.output_dir.mkdir(parents=True, exist_ok=False)
    source = Path(ssnalclust.__file__).parent

    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    report = dict(
        status="running",
        smoke=args.smoke,
        versions={
            name: importlib.metadata.version(name)
            for name in ("ssnalclust", "numpy", "scipy", "scikit-learn", "matplotlib")
        },
        source_sha256={p.name: digest(p) for p in sorted(source.glob("*.py"))},
        harness_sha256=digest(__file__),
        preprocessing_sha256=digest(Path(__file__).with_name("heldout_preprocessing.py")),
        tol=1e-6,
        generalized_max_iter=20000,
        cases={},
    )
    arrays = {}
    for name, fit, plot in (
        ("huber", fit_huber, plot_huber),
        ("logistic", fit_logistic, plot_logistic),
        ("poisson", fit_poisson, plot_poisson),
    ):
        if args.only and args.only != name:
            continue
        path, record, saved, graph = fit(args.smoke)
        record["data_sha256"] = digest(path)
        record["data_file"] = path.name
        saved.update(graph_indptr=graph.indptr, graph_indices=graph.indices, graph_data=graph.data)
        record["arrays"] = {key: name + "_" + key for key in saved}
        arrays.update({record["arrays"][key]: value for key, value in saved.items()})
        report["cases"][name] = record
        np.savez_compressed(args.output_dir / "fits.npz", **arrays)
        (args.output_dir / "diagnostics.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        if (
            not record["diagnostics"]["converged"]
            or not record.get("squared_comparison", {"converged": True})["converged"]
        ):
            report["status"] = "failed"
            (args.output_dir / "diagnostics.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            raise RuntimeError(
                f"{name} illustration did not meet its certificate tolerance; diagnostics preserved"
            )
        figure = plot(saved, record)
        figure.savefig(args.output_dir / f"gallery_{name}.png", dpi=160)
        plt.close(figure)
    report["status"] = "completed"
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
