"""Illustrate sparse convex clustering and biclustering on UCI Wine features.

Cultivar labels are never loaded. Run --output-dir DIR; --smoke uses first24
rows and a shorter alpha path. This gallery does not select parameters or
claim validated cluster recovery. All fits must satisfy their stopping tests.
"""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

from ssnalclust.estimator import _centroid_labels
from ssnalclust.structured import solve_biclustering, solve_sparse

FEATURE_NAMES = (
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315",
    "Proline",
)
ALPHAS = (0.0, 2.0, 5.0, 6.0, 6.1, 7.0)
GAMMA_ROW = 2.5
CLUSTER_TOL = 1e-4
FEATURE_ZERO_TOL = 1e-4


def load_features(path, smoke=False):
    """Load only the13 chemistry columns; the cultivar column is not read."""
    X = np.loadtxt(path, delimiter=",", usecols=range(1, 14), ndmin=2)
    if X.shape[1] != 13 or not np.isfinite(X).all():
        raise ValueError("Wine data must contain13 finite chemical measurements per row")
    rows = np.arange(min(24, len(X)) if smoke else len(X))
    return X[rows].copy(), rows


def column_graph(standardized, neighbors=4):
    """Exact union kNN on feature profiles, without restandardizing X.T."""
    distances = cdist(standardized.T, standardized.T)
    k = min(neighbors, standardized.shape[1] - 1)
    if k < 1:
        raise ValueError("Biclustering gallery requires at least two features")
    ranking = distances.copy()
    np.fill_diagonal(ranking, np.inf)
    columns = np.argsort(ranking, axis=1, kind="stable")[:, :k].ravel()
    rows = np.repeat(np.arange(len(distances)), k)
    directed = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=distances.shape)
    edges = sparse.triu(directed.maximum(directed.T), 1, format="coo")
    lengths = distances[edges.row, edges.col]
    positive = lengths[lengths > 0]
    if not len(positive):
        raise ValueError("Column graph needs a positive retained-edge distance")
    bandwidth = float(np.median(positive))
    upper = sparse.csr_matrix(
        (np.exp(-0.5 * (lengths / bandwidth) ** 2), (edges.row, edges.col)), shape=distances.shape
    )
    return (upper + upper.T).tocsr(), bandwidth, k


def _diagnostics(result, seconds):
    report = dict(
        converged=bool(result.converged),
        n_iter=result.n_iter,
        objective=result.objective,
        dual_objective=result.dual_objective,
        gap=result.gap,
        relative_gap=result.relative_gap,
        kkt_residual=result.kkt_residual,
        solve_seconds=seconds,
    )
    if not result.converged or not np.isfinite(list(report.values())).all():
        raise RuntimeError(f"Structured gallery fit failed convergence: {report}")
    return report


def _graph_arrays(arrays, name, graph):
    arrays[name + "_indptr"] = graph.indptr
    arrays[name + "_indices"] = graph.indices
    arrays[name + "_data"] = graph.data
    return hashlib.sha256(
        graph.indptr.tobytes() + graph.indices.tobytes() + graph.data.tobytes()
    ).hexdigest()


def fit_models(original, smoke=False):
    """Fit fixed illustrative models; no labels or parameter selection inputs."""
    from heldout_preprocessing import prepare_training

    original = np.asarray(original, dtype=float)
    prepared = prepare_training(
        original, np.ones_like(original, dtype=bool), neighbors=min(10, len(original) - 1)
    )
    X = prepared.training_values
    columns, column_bandwidth, column_neighbors = column_graph(X)
    alphas = (0.0, 2.0, 5.0) if smoke else ALPHAS
    arrays = dict(
        original=original,
        standardized=X,
        feature_means=prepared.means,
        feature_scales=prepared.scales,
        original_column_indices=np.arange(X.shape[1]),
    )
    report = dict(
        samples=len(X),
        features=X.shape[1],
        smoke=smoke,
        tol=1e-6,
        max_iter=20000,
        feature_zero_tol=FEATURE_ZERO_TOL,
        cluster_tol=CLUSTER_TOL,
        feature_names=list(FEATURE_NAMES),
        gamma_row=GAMMA_ROW,
        row_neighbors=min(10, len(X) - 1),
        row_bandwidth=prepared.bandwidth,
        row_edges=prepared.graph.nnz // 2,
        row_components=prepared.components,
        row_graph_sha256=_graph_arrays(arrays, "row_graph", prepared.graph),
        column_neighbors=column_neighbors,
        column_bandwidth=column_bandwidth,
        column_edges=columns.nnz // 2,
        column_graph_sha256=_graph_arrays(arrays, "column_graph", columns),
        sparse_path=[],
        parameter_policy="fixed illustrative choices; no statistical model selection or cultivar evaluation",
    )
    norms = []
    for index, alpha in enumerate(alphas):
        started = time.perf_counter()
        result = solve_sparse(
            X, weights=prepared.graph, gamma=GAMMA_ROW, alpha=alpha, tol=1e-6, max_iter=20000
        )
        diagnostics = _diagnostics(result, time.perf_counter() - started)
        restored = result.centers + result.offset
        labels = _centroid_labels(restored, CLUSTER_TOL)
        diagnostics.update(
            alpha=alpha,
            index=index,
            active_features=int(np.count_nonzero(result.feature_norms > FEATURE_ZERO_TOL)),
            n_clusters=int(np.unique(labels).size),
        )
        report["sparse_path"].append(diagnostics)
        norms.append(result.feature_norms)
        arrays[f"sparse_{index}_centered"] = result.centers
        arrays[f"sparse_{index}_offset"] = result.offset
        arrays[f"sparse_{index}_standardized_fitted"] = restored
        arrays[f"sparse_{index}_original_units_fitted"] = (
            restored * prepared.scales + prepared.means
        )
        arrays[f"sparse_{index}_labels"] = labels
        for name, dual in result.duals.items():
            arrays[f"sparse_{index}_dual_{name}"] = dual
    arrays["sparse_feature_norms"] = np.asarray(norms)
    arrays["sparse_alphas"] = np.asarray(alphas)
    report["sparse_display_index"] = 1  # alpha=2, fixed independently of observed outcomes.
    report["sparse_display_alpha"] = alphas[1]
    arrays["sparse_display_row_order"] = np.argsort(arrays["sparse_1_labels"], kind="stable")
    gamma_col = column_bandwidth / 4
    started = time.perf_counter()
    result = solve_biclustering(
        X,
        row_weights=prepared.graph,
        column_weights=columns,
        gamma_row=GAMMA_ROW,
        gamma_col=gamma_col,
        tol=1e-6,
        max_iter=20000,
    )
    diagnostics = _diagnostics(result, time.perf_counter() - started)
    row_labels = _centroid_labels(result.centers, CLUSTER_TOL)
    col_labels = _centroid_labels(result.centers.T, CLUSTER_TOL)
    row_order = np.argsort(row_labels, kind="stable")
    column_order = np.argsort(col_labels, kind="stable")
    diagnostics.update(
        gamma_row=GAMMA_ROW,
        gamma_col=gamma_col,
        row_clusters=int(np.unique(row_labels).size),
        column_clusters=int(np.unique(col_labels).size),
    )
    report["biclustering"] = diagnostics
    arrays.update(
        biclustering_standardized_fitted=result.centers,
        biclustering_original_units_fitted=result.centers * prepared.scales + prepared.means,
        biclustering_row_labels=row_labels,
        biclustering_column_labels=col_labels,
        row_order=row_order,
        column_order=column_order,
    )
    for name, dual in result.duals.items():
        arrays["biclustering_dual_" + name] = dual
    return report, arrays


def _boundaries(labels, order):
    return np.flatnonzero(np.diff(labels[order]) != 0) + 0.5


def plot_results(report, arrays, output_dir):
    """Plot saved model coordinates using identical input/fitted permutations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    index = report["sparse_display_index"]
    order = arrays["sparse_display_row_order"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for feature, name in enumerate(report["feature_names"]):
        axes[0].plot(
            arrays["sparse_alphas"],
            arrays["sparse_feature_norms"][:, feature],
            marker="o",
            markersize=3,
            label=name,
        )
    axes[0].axvline(report["sparse_display_alpha"], color="black", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="Feature penalty alpha",
        ylabel="Centered fitted feature norm",
        title="Feature shrinkage at gamma=2.5",
    )
    axes[0].legend(fontsize=6, ncol=2)
    limit = float(np.max(np.abs(arrays["standardized"])))
    for axis, matrix, title in zip(
        axes[1:],
        [arrays["standardized"], arrays[f"sparse_{index}_standardized_fitted"]],
        ["Standardized input", "Sparse fitted matrix (offset restored)"],
    ):
        im = axis.imshow(
            matrix[order],
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        axis.set(title=title, xlabel="Chemical feature", ylabel="Rows grouped by fitted partition")
        axis.set_xticks(
            np.arange(report["features"]), report["feature_names"], rotation=90, fontsize=7
        )
    fig.colorbar(im, ax=axes[1:], label="Standardized units", shrink=0.75)
    chosen = report["sparse_path"][index]
    fig.suptitle(
        f"Wine sparse convex clustering: alpha={chosen['alpha']:g}, {chosen['active_features']}/13 active features; zero norm <=1e-4"
    )
    fig.savefig(output_dir / "gallery_sparse.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    rows, cols = arrays["row_order"], arrays["column_order"]
    for axis, matrix, title in zip(
        axes,
        [arrays["standardized"], arrays["biclustering_standardized_fitted"]],
        ["Standardized input", "Convex biclustering fit"],
    ):
        im = axis.imshow(
            matrix[np.ix_(rows, cols)],
            aspect="auto",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        for boundary in _boundaries(arrays["biclustering_column_labels"], cols):
            axis.axvline(boundary, color="black", linewidth=0.7)
        for boundary in _boundaries(arrays["biclustering_row_labels"], rows):
            axis.axhline(boundary, color="black", linewidth=0.5, alpha=0.4)
        axis.set(title=title, ylabel="Rows grouped by fitted row partition")
        axis.set_xticks(
            np.arange(len(cols)),
            [report["feature_names"][j] for j in cols],
            rotation=90,
            fontsize=8,
        )
    fig.colorbar(im, ax=axes, label="Standardized units", shrink=0.75)
    fitted = report["biclustering"]
    fig.suptitle(
        f"Wine convex biclustering: {fitted['row_clusters']} row groups, {fitted['column_clusters']} feature groups\nSame fitted row/column order in both panels; original indices saved"
    )
    fig.savefig(output_dir / "gallery_biclustering.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/wine.data")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original, source_rows = load_features(args.data, args.smoke)
    report, arrays = fit_models(original, args.smoke)
    arrays["source_row_indices"] = source_rows
    arrays["ordered_source_row_indices"] = source_rows[arrays["row_order"]]
    arrays["ordered_source_column_indices"] = arrays["original_column_indices"][
        arrays["column_order"]
    ]
    arrays["sparse_ordered_source_row_indices"] = source_rows[arrays["sparse_display_row_order"]]
    import ssnalclust

    root = Path(ssnalclust.__file__).parent
    report.update(
        data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preprocessing_sha256=hashlib.sha256(
            Path(__file__).with_name("heldout_preprocessing.py").read_bytes()
        ).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ["ssnalclust", "numpy", "scipy", "matplotlib"]
        },
    )
    np.savez_compressed(args.output_dir / "gallery_structured.npz", **arrays)
    (args.output_dir / "gallery_structured.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot_results(report, arrays, args.output_dir)
    print(
        json.dumps(
            dict(sparse_path=report["sparse_path"], biclustering=report["biclustering"]), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
