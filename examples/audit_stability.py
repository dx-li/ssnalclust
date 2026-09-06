"""Audit saved stability studies with explicit pairs and independent certificates.

Uses neither the study's comparator/selector nor production solver diagnostics.
An interrupted or failed study is not eligible for a passed audit.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from audit_wine_holdout import canonical, certificate, reconstruct_graph
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_oracle(a, b, minimum):
    rows, cols = np.triu_indices(len(a), 1)
    left, right = a[rows] == a[cols], b[rows] == b[cols]
    margins = [int(left.sum()), int((~left).sum()), int(right.sum()), int((~right).sum())]
    defined = min(margins) > 0
    eligible = min(margins) >= minimum
    return dict(
        n_shared=len(a),
        total_pairs=len(rows),
        same_a=margins[0],
        same_b=margins[2],
        both_same=int((left & right).sum()),
        raw_disagreement=float(np.mean(left != right)),
        corrected_disagreement=-float(np.corrcoef(left.astype(float), right.astype(float))[0, 1])
        if defined
        else None,
        eligible=eligible,
        reason=None
        if eligible
        else "insufficient_pair_support"
        if defined
        else "degenerate_partition",
    )


def audit_study(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "study.json").read_text())
    failures, workers = [], []
    checks = 0

    def check(name, condition):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(name)

    def close(name, actual, expected, atol=1e-10):
        check(name, np.allclose(actual, expected, atol=atol, rtol=2e-11))

    def comparison(name, actual, expected):
        check(name + "/fields", set(actual) == set(expected))
        for key, value in expected.items():
            if isinstance(value, float):
                close(name + "/" + key, actual[key], value, atol=2e-14)
            else:
                check(name + "/" + key, actual[key] == value)

    smoke = manifest["smoke"]
    scenarios = ["separated"] if smoke else ["separated", "overlap", "single_gaussian"]
    seeds = [8101] if smoke else [8101, 8102, 8103]
    n, count, neighbors, minimum = (24, 1, 5, 1) if smoke else (120, 5, 10, 10)
    factors = [0.0, 1.0, 30.0] if smoke else [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    check("parent_completed", manifest["status"] == "completed")
    expected_names = [f"{scenario}_{seed}" for scenario in scenarios for seed in seeds]
    check("prescribed_workers", [p["name"] for p in manifest["processes"]] == expected_names)
    source_hashes = None
    for process in manifest["processes"]:
        name = process["name"]
        check(
            name + "/process",
            process["process_status"] == "completed" and process["returncode"] == 0,
        )
        for field in ["checkpoint", "arrays", "selection", "log"]:
            check(
                name + "/hash/" + field,
                digest(directory / process[field]) == process[field + "_sha256"],
            )
        report = json.loads((directory / process["checkpoint"]).read_text())
        selection = json.loads((directory / process["selection"]).read_text())
        check(
            name + "/decision_hash",
            digest(directory / process["selection"]) == report["selection_sha256"],
        )
        check(name + "/same_decision", selection == report["selection"])
        check(name + "/completed", report["status"] == "completed")
        check(
            name + "/identity",
            report["scenario"] == process["scenario"] and report["seed"] == process["seed"],
        )
        for key, value in dict(
            samples=n,
            features=2,
            factors=factors,
            tuning_pairs=count,
            audit_pairs=count,
            neighbors=neighbors,
            min_pairs=minimum,
            cluster_tol=1e-4,
            tol=1e-6,
            max_iter=300,
            check_every=1,
            store_history=False,
            solver="ssnal",
            smoke=smoke,
        ).items():
            check(name + "/config/" + key, report[key] == value)
        hashes = (report["source_sha256"], report["helper_sha256"])
        if source_hashes is None:
            source_hashes = hashes
        check(name + "/shared_sources", source_hashes == hashes)
        check(
            name + "/harness_hash",
            report["helper_sha256"]["stability_study.py"] == manifest["harness_sha256"],
        )
        check(name + "/threads", set(report["requested_thread_environment"].values()) == {"1"})
        with np.load(directory / process["arrays"], allow_pickle=False) as arrays:
            X, truth = arrays["source_X"], arrays["truth_posthoc_only"]
            rng = np.random.default_rng(process["seed"])
            if process["scenario"] == "single_gaussian":
                expected_X, expected_truth = rng.normal(size=(n, 2)), np.zeros(n, int)
            else:
                expected_truth = np.repeat(np.arange(3), n // 3)
                means = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
                expected_X = means[expected_truth] + rng.normal(
                    scale=0.25 if process["scenario"] == "separated" else 0.9, size=(n, 2)
                )
            order = rng.permutation(n)
            check(name + "/generated_features", np.array_equal(X, expected_X[order]))
            check(name + "/generated_truth", np.array_equal(truth, expected_truth[order]))
            sample_lookup = {s["name"]: s for s in report["samples_fitted"]}
            check(name + "/unique_samples", len(sample_lookup) == len(report["samples_fitted"]))
            expected_samples = []
            oracle_pairs = {}
            for phase, offset in [("tuning", 100000), ("audit", 200000)]:
                rng = np.random.default_rng(offset + process["seed"])
                expected_count = (
                    count if phase == "tuning" or selection["selected_index"] is not None else 0
                )
                check(name + "/" + phase + "/count", len(report[phase]) == expected_count)
                oracle_pairs[phase] = []
                for index, pair in enumerate(report[phase]):
                    expected_ids = [
                        np.sort(rng.choice(n, int(0.8 * n), replace=False)) for _ in range(2)
                    ]
                    actual_names = [f"{phase}_{index}_{member}" for member in range(2)]
                    expected_samples.extend(actual_names)
                    check(
                        name + "/pair_names",
                        pair["samples"] == actual_names and pair["index"] == index,
                    )
                    sample = [sample_lookup[s] for s in actual_names]
                    ids = [arrays[s["arrays"]["ids"]] for s in sample]
                    for member in range(2):
                        check(
                            name + "/sample_ids", np.array_equal(ids[member], expected_ids[member])
                        )
                    shared, a, b = np.intersect1d(*ids, return_indices=True)
                    check(name + "/shared_ids", pair["shared_ids"] == shared.tolist())
                    expected_comparisons = []
                    for left, right in zip(sample[0]["points"], sample[1]["points"]):
                        expected_comparisons.append(
                            pair_oracle(
                                arrays[left["arrays"]["labels"]][a],
                                arrays[right["arrays"]["labels"]][b],
                                minimum,
                            )
                        )
                    check(
                        name + "/comparison_count",
                        len(pair["comparisons"]) == len(expected_comparisons),
                    )
                    for actual, expected in zip(pair["comparisons"], expected_comparisons):
                        comparison(name + "/comparison", actual, expected)
                    check(name + "/pair_converged", pair["all_converged"] is True)
                    oracle_pairs[phase].append(expected_comparisons)
            expected_samples.append("full")
            check(name + "/scheduled_samples", list(sample_lookup) == expected_samples)
            means = []
            for index in range(len(factors)):
                column = [pair[index] for pair in oracle_pairs["tuning"]]
                means.append(
                    math.fsum(p["corrected_disagreement"] for p in column) / count
                    if all(p["eligible"] for p in column)
                    else None
                )
            valid_indices = [i for i, value in enumerate(means) if value is not None]
            # Oracle arithmetic can differ at roundoff; use recorded means only
            # after comparing them, retaining the protocol's exact tie behavior.
            for actual, expected in zip(selection["mean_scores"], means):
                if expected is None:
                    check(name + "/ineligible_mean", actual is None)
                else:
                    close(name + "/mean", actual, expected, atol=2e-14)
            selected = (
                min(valid_indices, key=lambda i: selection["mean_scores"][i])
                if valid_indices
                else None
            )
            check(name + "/selected_index", selected == selection["selected_index"])
            check(
                name + "/selected_factor",
                selection["selected_factor"] == (None if selected is None else factors[selected]),
            )
            check(name + "/eligibility", selection["eligible"] == [v is not None for v in means])
            check(
                name + "/selection_status",
                selection["status"] == ("no_selection" if selected is None else "valid"),
            )
            check(
                name + "/selection_reason",
                selection["no_selection_reason"]
                == ("no_factor_eligible_for_every_pair" if selected is None else None),
            )
            audit_valid = selected is not None and all(
                p[0]["eligible"] for p in oracle_pairs["audit"]
            )
            check(name + "/audit_eligible", report["audit_summary"]["eligible"] == audit_valid)
            if audit_valid:
                close(
                    name + "/audit_mean",
                    report["audit_summary"]["mean_score"],
                    np.mean([p[0]["corrected_disagreement"] for p in oracle_pairs["audit"]]),
                )
            else:
                check(name + "/audit_undefined", report["audit_summary"]["mean_score"] is None)
            check(name + "/completion_flag", report["selection_audit_complete"] == audit_valid)
            maxima = dict(kkt=0.0, relative_gap=0.0, dual_ball_excess=0.0)
            points = 0
            for sample in report["samples_fitted"]:
                check(name + "/sample_completed", sample["status"] == "completed")
                refs = sample["arrays"]
                ids = arrays[refs["ids"]]
                if sample["name"] == "full":
                    check(name + "/full_ids", np.array_equal(ids, np.arange(n)))
                expected_factors = (
                    [selection["selected_factor"]]
                    if sample["name"].startswith("audit")
                    else factors
                )
                check(name + "/sample_factors", sample["factors"] == expected_factors)
                check(name + "/point_count", len(sample["points"]) == len(expected_factors))
                data = X[ids]
                mean, scale = data.mean(axis=0), data.std(axis=0)
                scale[scale == 0] = 1
                standardized = (data - mean) / scale
                close(name + "/means", arrays[refs["means"]], mean)
                close(name + "/scales", arrays[refs["scales"]], scale)
                close(name + "/standardized", arrays[refs["training_values"]], standardized)
                graph = sparse.csr_matrix(
                    (
                        arrays[refs["graph_data"]],
                        arrays[refs["graph_indices"]],
                        arrays[refs["graph_indptr"]],
                    ),
                    shape=(len(ids), len(ids)),
                )
                expected_graph, bandwidth = reconstruct_graph(standardized, neighbors)
                close(name + "/graph", graph.toarray(), expected_graph.toarray())
                close(name + "/bandwidth", sample["bandwidth"], bandwidth)
                check(
                    name + "/graph_details",
                    sample["edges"] == graph.nnz // 2
                    and sample["components"] == connected_components(graph, directed=False)[0],
                )
                for index, point in enumerate(sample["points"]):
                    points += 1
                    centers, dual, labels = [
                        arrays[point["arrays"][key]] for key in ["centers", "dual", "labels"]
                    ]
                    expected_labels = connected_components(
                        sparse.csr_matrix(cdist(centers, centers) <= 1e-4), directed=False
                    )[1]
                    check(
                        name + "/partition",
                        np.array_equal(canonical(labels), canonical(expected_labels)),
                    )
                    clusters, counts = np.unique(labels, return_counts=True)
                    check(name + "/clusters", len(clusters) == point["n_clusters"])
                    close(
                        name + "/singletons",
                        point["singleton_fraction"],
                        np.sum(counts == 1) / len(ids),
                    )
                    gamma = expected_factors[index] * bandwidth / neighbors
                    check(
                        name + "/point_order",
                        point["index"] == index and point["factor"] == expected_factors[index],
                    )
                    close(name + "/gamma", point["gamma"], gamma)
                    numerical = certificate(
                        standardized, np.ones_like(standardized, bool), graph, gamma, centers, dual
                    )
                    for key in [
                        "objective",
                        "dual_objective",
                        "gap",
                        "relative_gap",
                        "kkt_residual",
                    ]:
                        close(name + "/certificate/" + key, point[key], numerical[key])
                    roundoff = 5e-11 * (1 + abs(numerical["objective"]))
                    check(
                        name + "/certified",
                        point["converged"]
                        and numerical["kkt_residual"] <= 1e-6 + 1e-12
                        and -roundoff <= numerical["gap"]
                        and numerical["relative_gap"] <= 1e-6 + 1e-12
                        and numerical["maximum_dual_ball_excess"] <= roundoff,
                    )
                    close(
                        name + "/center_bound_squared",
                        point["center_error_bound"] ** 2,
                        2 * max(numerical["gap"], 0),
                    )
                    if sample["name"] == "full":
                        close(
                            name + "/ari", point["posthoc_ari"], adjusted_rand_score(truth, labels)
                        )
                    maxima["kkt"] = max(maxima["kkt"], numerical["kkt_residual"])
                    maxima["relative_gap"] = max(maxima["relative_gap"], numerical["relative_gap"])
                    maxima["dual_ball_excess"] = max(
                        maxima["dual_ball_excess"], numerical["maximum_dual_ball_excess"]
                    )
            workers.append(
                dict(
                    name=name,
                    points=points,
                    maxima=maxima,
                    selected_factor=selection["selected_factor"],
                    audit_eligible=audit_valid,
                )
            )
    return dict(
        status="passed" if not failures else "failed",
        checks=checks,
        failures=failures,
        workers=workers,
        source_manifest_sha256=digest(directory / "study.json"),
        scope="Floating-point reconstruction, not interval certification or population validation",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_study(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
