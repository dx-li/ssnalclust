"""Prespecified UCI optical digits study with label-blind silhouette selection.

No download occurs here. Supply the original optdigits.tes CSV. Full paths
retain failures; --smoke and --skip-admm are exploratory/CI modes, not the full
study protocol. JSON and compressed arrays are checkpointed after every point.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_score

from ssnalclust import ConvexClusteringProblem
from ssnalclust.estimator import _centroid_labels

FACTORS = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])


def load_digits_csv(path, smoke=False):
    """Load original 64 integer pixel columns and a separate digit label."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 65 or not np.isfinite(data).all() or np.any(data != np.floor(data)):
        raise ValueError("Expected 64 integer pixels and one integer label per row")
    if np.any((data[:, :64] < 0) | (data[:, :64] > 16)):
        raise ValueError("Pixels must be integers in [0,16]")
    if np.any((data[:, 64] < 0) | (data[:, 64] > 9)):
        raise ValueError("Labels must be integers in [0,9]")
    if smoke:
        data = data[:80]
    return data[:, :64].copy() / 16.0, data[:, 64].astype(int)


def make_graph(X, neighbors, distances=None):
    """Union kNN with stable row-order tie breaks and median positive edge scale."""
    if not 1 <= neighbors < len(X):
        raise ValueError("neighbors must be in [1,n_samples-1]")
    distances = cdist(X, X) if distances is None else distances
    ranking = distances.copy()
    np.fill_diagonal(ranking, np.inf)
    columns = np.argsort(ranking, axis=1, kind="stable")[:, :neighbors].ravel()
    rows = np.repeat(np.arange(len(X)), neighbors)
    adjacency = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(X), len(X)))
    adjacency = adjacency.maximum(adjacency.T)
    edges = sparse.triu(adjacency, 1, format="coo")
    lengths = distances[edges.row, edges.col]
    positive = lengths[lengths > 0]
    if not len(positive):
        raise ValueError("Median positive retained-edge distance is undefined")
    bandwidth = float(np.median(positive))
    values = np.exp(-0.5 * (lengths / bandwidth) ** 2)
    upper = sparse.csr_matrix((values, (edges.row, edges.col)), shape=adjacency.shape)
    graph = (upper + upper.T).tocsr()
    graph.eliminate_zeros()
    return graph, bandwidth


def _peak_rss():
    """Native process high-water RSS on Linux/macOS, explicitly unavailable elsewhere."""
    if sys.platform not in {"linux", "darwin"}:
        return None
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def canonical_partition(labels):
    """Consecutive labels ordered by first occurrence, invariant to label names."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=int)
    rank[np.argsort(first)] = np.arange(len(first))
    return rank[inverse]


def _point_record(X, graph, gamma, index, result, seconds, cluster_tol):
    started = time.perf_counter()
    labels = canonical_partition(_centroid_labels(result.centers, cluster_tol))
    edges = sparse.triu(graph, 1, format="coo")
    fused = 0
    for begin in range(0, edges.nnz, 4096):
        differences = (
            result.centers[edges.row[begin : begin + 4096]]
            - result.centers[edges.col[begin : begin + 4096]]
        )
        fused += int(np.count_nonzero(np.linalg.norm(differences, axis=1) <= cluster_tol))
    movement = float(np.linalg.norm(result.centers - X))
    baseline = float(np.linalg.norm(X - X.mean(axis=0)))
    record = dict(
        grid_index=int(index),
        gamma=float(gamma),
        status="completed",
        converged=bool(result.converged),
        iterations=result.n_iter,
        objective=result.objective,
        dual_objective=result.dual_objective,
        gap=result.gap,
        relative_gap=result.relative_gap,
        kkt_residual=result.kkt_residual,
        center_error_bound=result.center_error_bound,
        n_clusters=int(labels.max()) + 1,
        displacement_frobenius=movement,
        relative_displacement=movement / baseline if baseline else 0.0 if movement == 0 else None,
        direct_fused_edge_fraction=float(fused / edges.nnz) if edges.nnz else None,
        solve_seconds=seconds,
        diagnostics_seconds=time.perf_counter() - started,
        message=result.message,
        peak_rss_bytes=_peak_rss(),
    )
    return record, labels


def fit_path(
    X,
    graph,
    gammas,
    indices=None,
    *,
    tol=1e-6,
    max_iter=300,
    cluster_tol=1e-4,
    problem_factory=ConvexClusteringProblem,
):
    """Yield SSNAL fits using features only; no truth or desired cluster count."""
    preparation_started = time.perf_counter()
    problem = problem_factory(X, weights=graph)
    preparation_seconds = time.perf_counter() - preparation_started
    x0 = dual0 = None
    indices = range(len(gammas)) if indices is None else indices
    for index, gamma in zip(indices, gammas):
        started = time.perf_counter()
        try:
            result = problem.solve(
                gamma=float(gamma),
                solver="ssnal",
                tol=tol,
                max_iter=max_iter,
                x0=x0,
                dual0=dual0,
                store_history=False,
            )
            seconds = time.perf_counter() - started
            record, labels = _point_record(X, graph, gamma, index, result, seconds, cluster_tol)
            record["prepared_problem_seconds"] = preparation_seconds
            x0, dual0 = result.centers, result.dual
            yield record, result.centers, labels
        except Exception as error:
            yield (
                dict(
                    grid_index=int(index),
                    gamma=float(gamma),
                    status="error",
                    converged=False,
                    error_type=type(error).__name__,
                    error=str(error),
                    solve_seconds=time.perf_counter() - started,
                ),
                None,
                None,
            )


def select_by_silhouette(X, points, partitions, *, expected_points, distances=None):
    """Select from features and fitted partitions only, with no label access.

    Scores are exact Euclidean silhouettes on original scaled features,
    including sklearn's zero contribution for singleton clusters. First input
    point wins ties. Any missing or unconverged path point invalidates selection.
    """
    distances = cdist(X, X) if distances is None else distances
    started = time.perf_counter()
    scores = []
    for labels in partitions:
        count = len(np.unique(labels)) if labels is not None else 0
        score = (
            float(silhouette_score(distances, labels, metric="precomputed"))
            if 2 <= count < len(X)
            else None
        )
        scores.append(score if score is not None and np.isfinite(score) else None)
    complete = len(points) == expected_points and all(
        point.get("converged", False) for point in points
    )
    eligible = [index for index, score in enumerate(scores) if score is not None]
    selected = max(eligible, key=lambda index: scores[index]) if complete and eligible else None
    return dict(
        status="valid" if selected is not None else "incomplete" if not complete else "unavailable",
        selected_position=selected,
        selected_grid_index=points[selected]["grid_index"] if selected is not None else None,
        selected_gamma=points[selected]["gamma"] if selected is not None else None,
        silhouette_scores=scores,
        selection_seconds=time.perf_counter() - started,
        criterion="exact_euclidean_silhouette_on_original_scaled_features",
        all_path_points_required=True,
    )


def evaluate_truth(truth, partitions, selection):
    """Evaluate digit truth only after selection; never choose or adjust a fit."""
    scores = [
        float(adjusted_rand_score(truth, labels)) if labels is not None else None
        for labels in partitions
    ]
    selected = selection["selected_position"]
    return dict(
        purpose="posthoc_evaluation_only",
        ari_by_position=scores,
        selected_ari=scores[selected] if selected is not None else None,
    )


def crosscheck_admm(X, graph, gammas, indices, *, tol=1e-6, max_iter=10000, cluster_tol=1e-4):
    """Cold ADMM calls at prespecified grid indices, independent of silhouette."""
    for index, gamma in zip(indices, gammas):
        if index not in (0, 4, 8):
            continue
        started = time.perf_counter()
        try:
            # A fresh prepared instance also makes factorization timing cold.
            result = ConvexClusteringProblem(X, weights=graph).solve(
                gamma=float(gamma), solver="admm", tol=tol, max_iter=max_iter, store_history=False
            )
            seconds = time.perf_counter() - started
            record, labels = _point_record(X, graph, gamma, index, result, seconds, cluster_tol)
            yield record, result.centers, labels
        except Exception as error:
            yield (
                dict(
                    grid_index=int(index),
                    gamma=float(gamma),
                    status="error",
                    converged=False,
                    error_type=type(error).__name__,
                    error=str(error),
                    solve_seconds=time.perf_counter() - started,
                ),
                None,
                None,
            )


def _checkpoint(report, arrays, output, array_path):
    """Replace arrays before JSON so an interruption preserves a usable prefix.

    Each file replacement is atomic. Arrays only accumulate during a run:
    interruption between replacements leaves the older JSON with a superset of
    its required arrays. The pair is not a transactional snapshot. Use unique
    output paths per run; concurrent writers or reuse across different studies
    are not supported.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    array_temporary = array_path.with_name(array_path.name + ".tmp")
    with array_temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(array_temporary, array_path)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def run_study(data_path, output, array_path, *, neighbors=(10, 20), smoke=False, skip_admm=False):
    output, array_path = Path(output), Path(array_path)
    if output.resolve() == array_path.resolve():
        raise ValueError("JSON and NPZ output paths must differ")
    if output.exists() or array_path.exists():
        raise FileExistsError("Use fresh output paths to preserve interrupted-run checkpoints")
    worker_started = time.perf_counter()
    X, truth = load_digits_csv(data_path, smoke=smoke)
    import ssnalclust

    root = Path(ssnalclust.__file__).parent
    report = dict(
        status="running",
        protocol="optdigits_test_features_divided_by_16",
        smoke=smoke,
        admm_skipped=skip_admm,
        data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        scaled_features_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))
        },
        versions={
            name: importlib.metadata.version(name)
            for name in ["ssnalclust", "numpy", "scipy", "scikit-learn"]
        },
        python=platform.python_version(),
        platform=platform.platform(),
        samples=len(X),
        features=X.shape[1],
        image_shape=[8, 8],
        pixel_divisor=16,
        gamma_factors=FACTORS.tolist(),
        solve_timing="SSNAL excludes shared preparation; ADMM includes cold preparation",
        preparation_timing="prepared_problem_seconds is repeated per SSNAL point; count once per graph",
        tol=1e-6,
        ssnal_max_iter=300,
        ssnal_store_history=False,
        admm_initialization="fresh prepared problem and cold iterates at each index; no factor reuse",
        checkpoint_consistency="arrays replaced before JSON; older JSON may have extra unreferenced arrays; unique paths per run",
        admm_max_iter=10000,
        cluster_tol=1e-4,
        arrays=str(array_path),
        graphs=[],
    )
    arrays = {"scaled_features": X, "truth_posthoc_only": truth}
    report["checkpoint_seconds"] = 0.0
    report["checkpoint_count"] = 0
    report["checkpoint_timing"] = "completed_JSON_and_NPZ_writes_excluding_final_summary_refresh"
    report["peak_rss_supported"] = sys.platform in {"linux", "darwin"}

    def save():
        checkpoint_started = time.perf_counter()
        report["worker_seconds"] = time.perf_counter() - worker_started
        report["peak_rss_bytes"] = _peak_rss()
        _checkpoint(report, arrays, output, array_path)
        report["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
        report["checkpoint_count"] += 1

    started = time.perf_counter()
    distances = cdist(X, X)
    report["distance_matrix_seconds"] = time.perf_counter() - started
    report["distance_matrix_bytes"] = distances.nbytes
    indices = np.array([0, 4, 8]) if smoke else np.arange(len(FACTORS))
    for k in neighbors:
        started = time.perf_counter()
        graph, bandwidth = make_graph(X, k, distances)
        gammas = bandwidth / k * FACTORS[indices]
        graph_record = dict(
            neighbors=int(k),
            bandwidth=bandwidth,
            gamma_indices=indices.tolist(),
            gammas=gammas.tolist(),
            graph_seconds=time.perf_counter() - started,
            edges=graph.nnz // 2,
            components=int(connected_components(graph)[0]),
            graph_sha256=hashlib.sha256(
                graph.indptr.tobytes() + graph.indices.tobytes() + graph.data.tobytes()
            ).hexdigest(),
            ssnal=[],
            admm=[],
            selection={"status": "pending"},
        )
        report["graphs"].append(graph_record)
        arrays[f"k{k}_graph_indptr"], arrays[f"k{k}_graph_indices"], arrays[f"k{k}_graph_data"] = (
            graph.indptr,
            graph.indices,
            graph.data,
        )
        save()
        partitions = []
        for point, centers, labels in fit_path(X, graph, gammas, indices):
            graph_record["ssnal"].append(point)
            partitions.append(labels)
            if centers is not None:
                arrays[f"k{k}_ssnal_{point['grid_index']}_centers"] = centers
                arrays[f"k{k}_ssnal_{point['grid_index']}_labels"] = labels
            save()
            print(
                f"k={k} SSNAL index={point['grid_index']} gamma={point['gamma']:.6g} "
                f"status={point['status']} converged={point['converged']}",
                flush=True,
            )
        selection = select_by_silhouette(
            X, graph_record["ssnal"], partitions, expected_points=len(gammas), distances=distances
        )
        graph_record["selection"] = selection
        graph_record["posthoc"] = evaluate_truth(truth, partitions, selection)
        save()
        if not skip_admm:
            for point, centers, labels in crosscheck_admm(X, graph, gammas, indices):
                graph_record["admm"].append(point)
                if centers is not None:
                    arrays[f"k{k}_admm_{point['grid_index']}_centers"] = centers
                    arrays[f"k{k}_admm_{point['grid_index']}_labels"] = labels
                    ssnal_centers = arrays.get(f"k{k}_ssnal_{point['grid_index']}_centers")
                    if ssnal_centers is not None:
                        original = next(
                            p
                            for p in graph_record["ssnal"]
                            if p["grid_index"] == point["grid_index"]
                        )
                        point["ssnal_center_distance"] = float(
                            np.linalg.norm(centers - ssnal_centers)
                        )
                        point["combined_center_bound"] = (
                            point["center_error_bound"] + original["center_error_bound"]
                        )
                        point["within_combined_center_bound"] = bool(
                            point["ssnal_center_distance"]
                            <= point["combined_center_bound"]
                            + 128 * np.finfo(float).eps * (1 + np.linalg.norm(X))
                        )
                        point["same_partition_as_ssnal"] = bool(
                            np.array_equal(
                                canonical_partition(labels),
                                canonical_partition(
                                    arrays[f"k{k}_ssnal_{point['grid_index']}_labels"]
                                ),
                            )
                        )
                save()
                print(
                    f"k={k} ADMM index={point['grid_index']} status={point['status']} "
                    f"converged={point['converged']}",
                    flush=True,
                )
        graph_record["admm_crosscheck_status"] = (
            "skipped_exploratory"
            if skip_admm
            else "complete"
            if len(graph_record["admm"]) == 3
            and all(p.get("converged", False) for p in graph_record["admm"])
            else "incomplete"
        )
    report["status"] = "completed"
    report["selection_valid_for_all_graphs"] = all(
        g["selection"]["status"] == "valid" for g in report["graphs"]
    )
    save()
    # Final lightweight summary reflects all measured full checkpoints.
    report["worker_seconds"] = time.perf_counter() - worker_started
    report["peak_rss_bytes"] = _peak_rss()
    report["peak_rss_scope"] = "observed_through_final_full_checkpoint"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).parent / "data" / "optdigits.tes"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[10, 20])
    parser.add_argument(
        "--skip-admm", action="store_true", help="Exploratory only: omit prescribed crosschecks"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="First80rows and gridindices0,4,8 only"
    )
    args = parser.parse_args()
    run_study(
        args.data,
        args.output,
        args.arrays,
        neighbors=args.neighbors,
        smoke=args.smoke,
        skip_admm=args.skip_admm,
    )


if __name__ == "__main__":
    main()
