"""Audit saved Digits study artifacts without importing or running any solver.

Run: python examples/audit_digits_study.py --output docs/digits_audit.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, silhouette_score

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_digest(graph):
    return hashlib.sha256(
        graph.indptr.tobytes() + graph.indices.tobytes() + graph.data.tobytes()
    ).hexdigest()


def canonical(labels):
    mapping = {}
    return np.array([mapping.setdefault(int(label), len(mapping)) for label in labels])


def summary(values):
    return dict(zip(["minimum", "median", "maximum"], map(float, np.quantile(values, [0, 0.5, 1]))))


def audit(report_path):
    report = json.loads(report_path.read_text())
    arrays_path = report_path.with_suffix(".npz")
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    with np.load(arrays_path, allow_pickle=False) as arrays:
        x = arrays["scaled_features"]
        truth = arrays["truth_posthoc_only"]
        original = np.loadtxt(ROOT / "examples/data/optdigits.tes", delimiter=",")
        check(
            "original_data_hash",
            digest(ROOT / "examples/data/optdigits.tes") == report["data_sha256"],
        )
        check("original_features", np.array_equal(x, original[:, :64] / 16))
        check("original_truth", np.array_equal(truth, original[:, 64]))
        check(
            "feature_hash",
            hashlib.sha256(x.tobytes()).hexdigest() == report["scaled_features_sha256"],
        )
        check(
            "full_completed_study",
            report["status"] == "completed" and not report["smoke"] and not report["admm_skipped"],
        )
        distances = cdist(x, x)
        graph_reports = []
        for recorded in report["graphs"]:
            k = recorded["neighbors"]
            prefix = f"k{k}"
            stored = sparse.csr_matrix(
                (
                    arrays[prefix + "_graph_data"],
                    arrays[prefix + "_graph_indices"],
                    arrays[prefix + "_graph_indptr"],
                ),
                shape=(len(x), len(x)),
            )
            check(prefix + "_stored_hash", graph_digest(stored) == recorded["graph_sha256"])
            # Independent explicit lexicographic ranking; self is excluded before sorting.
            edges = set()
            for i in range(len(x)):
                candidates = np.arange(len(x))
                candidates = candidates[candidates != i]
                nearest = candidates[np.lexsort((candidates, distances[i, candidates]))[:k]]
                edges.update((min(i, int(j)), max(i, int(j))) for j in nearest)
            pairs = np.array(sorted(edges))
            lengths = distances[pairs[:, 0], pairs[:, 1]]
            bandwidth = float(np.median(lengths[lengths > 0]))
            weights = np.exp(-0.5 * (lengths / bandwidth) ** 2)
            upper = sparse.csr_matrix((weights, (pairs[:, 0], pairs[:, 1])), shape=stored.shape)
            rebuilt = (upper + upper.T).tocsr()
            rebuilt.eliminate_zeros()
            check(prefix + "_rebuilt_hash", graph_digest(rebuilt) == recorded["graph_sha256"])
            check(prefix + "_bandwidth", bandwidth == recorded["bandwidth"])
            check(prefix + "_edge_count", len(pairs) == recorded["edges"])
            components = int(connected_components(stored)[0])
            check(prefix + "_components", components == recorded["components"])
            degree = np.diff(stored.indptr)
            weighted_degree = np.asarray(stored.sum(axis=1)).ravel()
            point_reports = []
            partitions = []
            for method in ("ssnal", "admm"):
                records = recorded[method]
                check(
                    prefix + "_" + method + "_grid",
                    [p["grid_index"] for p in records]
                    == (list(range(9)) if method == "ssnal" else [0, 4, 8]),
                )
                for point in records:
                    index = point["grid_index"]
                    name = f"{prefix}_{method}_{index}"
                    expected_gamma = bandwidth / k * [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10][index]
                    check(name + "_frozen_gamma", point["gamma"] == expected_gamma)
                    centers, labels = arrays[name + "_centers"], arrays[name + "_labels"]
                    check(
                        name + "_convergence",
                        point["converged"]
                        and point["relative_gap"] <= report["tol"]
                        and point["kkt_residual"] <= report["tol"],
                    )
                    delta = centers[pairs[:, 0]] - centers[pairs[:, 1]]
                    edge_lengths = np.linalg.norm(delta, axis=1)
                    primal = float(
                        0.5 * np.sum((centers - x) ** 2)
                        + point["gamma"] * np.dot(weights, edge_lengths)
                    )
                    allowance = 128 * np.finfo(float).eps * (1 + abs(primal))
                    check(name + "_primal", abs(primal - point["objective"]) <= allowance)
                    center_distances = cdist(centers, centers)
                    membership = sparse.csr_matrix(center_distances <= report["cluster_tol"])
                    count, derived_labels = connected_components(membership)
                    check(
                        name + "_memberships",
                        np.array_equal(canonical(labels), canonical(derived_labels)),
                    )
                    check(name + "_cluster_count", count == point["n_clusters"])
                    fused = float(np.mean(edge_lengths <= report["cluster_tol"]))
                    check(name + "_fused_edges", fused == point["direct_fused_edge_fraction"])
                    check(
                        name + "_error_bound_formula",
                        np.isclose(
                            point["center_error_bound"],
                            np.sqrt(2 * point["gap"]),
                            rtol=1e-14,
                            atol=0,
                        ),
                    )
                    check(
                        name + "_gap_vs_objectives",
                        abs(point["gap"] - (point["objective"] - point["dual_objective"]))
                        <= allowance,
                    )
                    item = dict(
                        method=method,
                        grid_index=index,
                        recomputed_primal=primal,
                        primal_difference=primal - point["objective"],
                        n_clusters=int(count),
                        direct_fused_edge_fraction=fused,
                    )
                    if method == "ssnal":
                        partitions.append(labels.copy())
                    else:
                        other = recorded["ssnal"][index]
                        other_centers = arrays[f"{prefix}_ssnal_{index}_centers"]
                        center_distance = float(np.linalg.norm(centers - other_centers))
                        combined = point["center_error_bound"] + other["center_error_bound"]
                        difference = abs(point["objective"] - other["objective"])
                        gap_sum = point["gap"] + other["gap"]
                        interval_left = max(point["dual_objective"], other["dual_objective"])
                        interval_right = min(point["objective"], other["objective"])
                        check(name + "_objective_bound", difference <= gap_sum + allowance)
                        check(
                            name + "_interval_overlap", interval_left <= interval_right + allowance
                        )
                        check(
                            name + "_center_bound",
                            center_distance
                            <= combined + 128 * np.finfo(float).eps * (1 + np.linalg.norm(x)),
                        )
                        same = bool(
                            np.array_equal(
                                canonical(labels),
                                canonical(arrays[f"{prefix}_ssnal_{index}_labels"]),
                            )
                        )
                        check(
                            name + "_recorded_partition_crosscheck",
                            same == point["same_partition_as_ssnal"],
                        )
                        check(
                            name + "_recorded_center_distance",
                            np.isclose(
                                center_distance,
                                point["ssnal_center_distance"],
                                rtol=1e-14,
                                atol=1e-14,
                            ),
                        )
                        item.update(
                            objective_difference=difference,
                            summed_gaps=gap_sum,
                            certificate_interval_left=interval_left,
                            certificate_interval_right=interval_right,
                            exact_recorded_interval_overlap=interval_left <= interval_right,
                            floating_point_allowance=allowance,
                            centroid_distance=center_distance,
                            summed_center_bounds=combined,
                            same_partition=same,
                        )
                    point_reports.append(item)
            # Recompute selection from input distances and partitions, before consulting truth.
            scores = [
                float(silhouette_score(distances, labels, metric="precomputed"))
                if 2 <= len(np.unique(labels)) < len(x)
                else None
                for labels in partitions
            ]
            eligible = [
                i for i, value in enumerate(scores) if value is not None and np.isfinite(value)
            ]
            selected = max(eligible, key=lambda i: scores[i]) if eligible else None
            check(
                prefix + "_selected_index", selected == recorded["selection"]["selected_grid_index"]
            )
            check(
                prefix + "_silhouette_scores",
                all(
                    a == b
                    or (
                        a is not None and b is not None and np.isclose(a, b, rtol=1e-12, atol=1e-14)
                    )
                    for a, b in zip(scores, recorded["selection"]["silhouette_scores"])
                ),
            )
            ari = [float(adjusted_rand_score(truth, labels)) for labels in partitions]
            check(
                prefix + "_posthoc_ari",
                np.allclose(ari, recorded["posthoc"]["ari_by_position"], rtol=1e-14, atol=1e-14),
            )
            check(
                prefix + "_selected_ari",
                recorded["posthoc"]["selected_ari"]
                == (ari[selected] if selected is not None else None),
            )
            graph_reports.append(
                dict(
                    neighbors=k,
                    exact_graph_hash=graph_digest(rebuilt),
                    edges=len(pairs),
                    components=components,
                    isolates=int(np.count_nonzero(degree == 0)),
                    weight_summary=summary(weights),
                    degree_summary=summary(degree),
                    weighted_degree_summary=summary(weighted_degree),
                    bandwidth=bandwidth,
                    selected_index=selected,
                    selected_silhouette=scores[selected] if selected is not None else None,
                    selected_ari=ari[selected] if selected is not None else None,
                    points=point_reports,
                )
            )
    return dict(
        report_file=report_path.name,
        report_sha256=digest(report_path),
        arrays_file=arrays_path.name,
        arrays_sha256=digest(arrays_path),
        status="passed" if all(checks.values()) else "failed",
        checks=checks,
        graphs=graph_reports,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports",
        type=Path,
        nargs="+",
        default=[ROOT / "docs/digits_k10.json", ROOT / "docs/digits_k20.json"],
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/digits_audit.json")
    args = parser.parse_args()
    reports = [audit(path) for path in args.reports]
    output = dict(
        status="passed" if all(r["status"] == "passed" for r in reports) else "failed",
        audit_source_sha256=digest(__file__),
        scope="Artifact reconstruction and recorded-certificate consistency; no optimization run",
        limitations=[
            "Edge dual arrays were not saved, so dual feasibility and dual objectives cannot be independently recomputed.",
            "Certificate intervals and center bounds use recorded floating-point certificates, not interval arithmetic.",
            "Equal partitions are observed outcomes, not guaranteed by objective accuracy.",
        ],
        reports=reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(output["status"], args.output)
    if output["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
