"""Reconstruct Wine holdout artifacts and certificates without fitting models."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
FACTORS = np.array([0.1, 0.3, 1.0, 3.0, 10.0])


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_hash(graph):
    return hashlib.sha256(
        graph.indptr.tobytes() + graph.indices.tobytes() + graph.data.tobytes()
    ).hexdigest()


def reconstruct_mask(shape, seed):
    observed = np.ones(shape, dtype=bool)
    rng = np.random.default_rng(seed)
    for feature in range(shape[1]):
        observed[rng.permutation(shape[0])[: math.floor(0.1 * shape[0])], feature] = False
    return observed


def reconstruct_graph(inputs, neighbors):
    """Explicit lexicographic ranking, independent of the preprocessing helper."""
    distances = cdist(inputs, inputs)
    pairs = set()
    for i in range(len(inputs)):
        candidates = np.arange(len(inputs))
        candidates = candidates[candidates != i]
        order = np.lexsort((candidates, distances[i, candidates]))
        pairs.update((min(i, int(j)), max(i, int(j))) for j in candidates[order[:neighbors]])
    pairs = np.array(sorted(pairs))
    lengths = distances[pairs[:, 0], pairs[:, 1]]
    bandwidth = float(np.median(lengths[lengths > 0]))
    weights = np.exp(-0.5 * (lengths / bandwidth) ** 2)
    upper = sparse.csr_matrix(
        (weights, (pairs[:, 0], pairs[:, 1])), shape=(len(inputs), len(inputs))
    )
    graph = (upper + upper.T).tocsr()
    graph.eliminate_zeros()
    return graph, bandwidth


def certificate(training, observed, graph, gamma, centers, dual):
    """Independent l2 primal, original KKT, and fidelity-domain dual values.

    The box dual is evaluated as separate scalar constrained infima, rather
    than calling the production solver's centered/slack diagnostics.
    """
    edges = sparse.triu(graph, 1, format="coo")
    order = np.lexsort((edges.col, edges.row))
    first, second, weights = edges.row[order], edges.col[order], edges.data[order]
    radii = gamma * weights
    if dual.shape != (len(radii), centers.shape[1]):
        raise ValueError("Saved edge dual has the wrong shape")
    differences = centers[first] - centers[second]
    divergence = np.zeros_like(centers)
    np.add.at(divergence, first, dual)
    np.add.at(divergence, second, -dual)
    gradient = np.zeros_like(centers)
    gradient[observed] = centers[observed] - training[observed]
    fusion = float(np.dot(radii, np.linalg.norm(differences, axis=1)))
    primal = float(0.5 * np.sum(gradient**2) + fusion)
    dual_norms = np.linalg.norm(dual, axis=1)
    ball_excess = float(np.max(np.maximum(dual_norms - radii, 0), initial=0))
    q = differences + dual
    qnorm = np.linalg.norm(q, axis=1)
    shrink = np.zeros_like(qnorm)
    np.divide(radii, qnorm, out=shrink, where=qnorm > 0)
    proximal = q * np.maximum(1 - shrink, 0)[:, None]
    stationarity = float(
        np.linalg.norm(gradient + divergence)
        / (1 + np.linalg.norm(gradient) + np.linalg.norm(divergence))
    )
    subgradient = float(
        np.linalg.norm(differences - proximal)
        / (1 + np.linalg.norm(differences) + np.linalg.norm(dual))
    )
    inside_box = True
    if observed.all():
        lower_bound = float(
            np.sum(dual * (training[first] - training[second])) - 0.5 * np.sum(divergence**2)
        )
        model = "unconstrained_squared_fidelity"
    else:
        _, components = connected_components(graph, directed=False)
        terms = []
        for component in np.unique(components):
            rows = np.flatnonzero(components == component)
            for feature in range(centers.shape[1]):
                values = training[rows[observed[rows, feature]], feature]
                if not len(values):
                    raise ValueError("Unanchored graph-component feature")
                lower, upper = float(values.min()), float(values.max())
                for row in rows:
                    a = float(divergence[row, feature])
                    u = float(centers[row, feature])
                    inside_box = inside_box and lower <= u <= upper
                    if observed[row, feature]:
                        x = float(training[row, feature])
                        t = min(max(x - a, lower), upper)
                        terms.append(0.5 * (t - x) ** 2 + a * t)
                    else:
                        terms.append(min(a * lower, a * upper))
        lower_bound = math.fsum(terms)
        model = "observed_range_box"
    gap = primal - lower_bound
    return dict(
        objective=primal,
        dual_objective=lower_bound,
        gap=gap,
        relative_gap=gap / (1 + abs(primal) + abs(lower_bound)),
        kkt_residual=max(stationarity, subgradient),
        stationarity=stationarity,
        edge_proximal_residual=subgradient,
        maximum_dual_ball_excess=ball_excess,
        centers_inside_observed_box=inside_box,
        certificate_model=model,
    )


def canonical(labels):
    mapping = {}
    return np.array([mapping.setdefault(int(value), len(mapping)) for value in labels])


def audit_worker(path, original, smoke=False):
    report = json.loads(path.read_text())
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)

    def close(name, actual, expected, *, atol=1e-11):
        check(name, np.allclose(actual, expected, rtol=2e-12, atol=atol, equal_nan=True))

    x = original[:24, 1:] if smoke else original[:, 1:]
    heldout = report["kind"] == "holdout"
    factors = np.array([0.1, 10.0]) if smoke else FACTORS
    expected_mask = (
        reconstruct_mask(x.shape, report["seed"]) if heldout else np.ones(x.shape, dtype=bool)
    )
    result = dict(
        file=path.name,
        sha256=file_hash(path),
        arrays_sha256=file_hash(path.with_suffix(".npz")),
        worker_status=report["status"],
        kind=report["kind"],
        seed=report["seed"],
        checks=checks,
        points=[],
    )
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        if "training_values" not in arrays:
            check("missing_preprocessing_reported_incomplete", report["status"] != "completed")
            result["status"] = "incomplete"
            return result
        observed, values = arrays["training_mask"], arrays["training_values"]
        means, scales = arrays["feature_means"], arrays["feature_scales"]
        check("frozen_mask", np.array_equal(observed, expected_mask))
        check("hidden_values_erased", np.isnan(values[~observed]).all())
        independent_means, independent_scales = [], []
        for feature in range(x.shape[1]):
            training = x[expected_mask[:, feature], feature]
            mean = math.fsum(map(float, training)) / len(training)
            deviation = math.sqrt(
                math.fsum((float(v) - mean) ** 2 for v in training) / len(training)
            )
            independent_means.append(mean)
            independent_scales.append(deviation if deviation else 1.0)
        close("training_means", means, independent_means)
        close("population_scales", scales, independent_scales)
        expected_values = np.where(expected_mask, (x - means) / scales, np.nan)
        close("standardized_training_values", values, expected_values)
        expected_inputs = np.where(expected_mask, expected_values, 0)
        check("graph_inputs_training_only", np.array_equal(arrays["graph_inputs"], expected_inputs))
        graph = sparse.csr_matrix(
            (arrays["graph_data"], arrays["graph_indices"], arrays["graph_indptr"]),
            shape=(len(x), len(x)),
        )
        rebuilt, bandwidth = reconstruct_graph(expected_inputs, 10)
        check("stored_graph_hash", graph_hash(graph) == report["graph_sha256"])
        check("reconstructed_graph_hash", graph_hash(rebuilt) == report["graph_sha256"])
        check("bandwidth", bandwidth == report["bandwidth"])
        components, component_labels = connected_components(graph)
        check("component_count", components == report["graph_components"])
        check("edge_count", graph.nnz // 2 == report["graph_edges"])
        check(
            "component_feature_coverage",
            all(observed[component_labels == c].any(axis=0).all() for c in range(components)),
        )
        close("frozen_factors", report["factors"], factors)
        targets = ((x - means) / scales)[~expected_mask]
        if heldout:
            check("no_truth_in_tuning_archive", "truth_posthoc_only" not in arrays)
            check("validation_mask", np.array_equal(arrays["validation_mask"], ~expected_mask))
            check("validation_count", len(targets) == report["validation_count"])
            check(
                "raw_targets",
                np.array_equal(arrays["validation_targets_original"], x[~expected_mask]),
            )
            close("standardized_targets", arrays["validation_targets_standardized"], targets)
            baseline = math.fsum(float(t) ** 2 for t in targets) / len(targets)
            close("baseline_mse", report["baseline_mse"], baseline)
            result.update(baseline_mse=baseline, validation_count=len(targets))
        expected_indices = list(range(len(factors)))
        if report["status"] == "completed":
            check("complete_grid", [p["index"] for p in report["points"]] == expected_indices)
        for point in report["points"]:
            index = point["index"]
            name = f"point_{index}"
            close(name + "_factor", point["factor"], factors[index])
            close(name + "_gamma", point["gamma"], factors[index] * bandwidth / 10)
            entry = dict(
                index=index,
                factor=point["factor"],
                fit_status=point["fit_status"],
                converged=point["converged"],
            )
            result["points"].append(entry)
            if point["fit_status"] != "completed":
                check(name + "_failure_not_converged", not point["converged"])
                continue
            centers, dual = arrays[name + "_centers"], arrays[name + "_dual"]
            check(name + "_finite_arrays", np.isfinite(centers).all() and np.isfinite(dual).all())
            independent = certificate(values, observed, graph, point["gamma"], centers, dual)
            entry["independent_certificate"] = independent
            allowance = 256 * np.finfo(float).eps * (1 + abs(independent["objective"]))
            for key in ("objective", "dual_objective", "gap", "relative_gap", "kkt_residual"):
                close(name + "_" + key, point[key], independent[key], atol=allowance)
            radii_max = point["gamma"] * graph.data.max()
            check(
                name + "_dual_feasible",
                independent["maximum_dual_ball_excess"]
                <= 64 * np.finfo(float).eps * (1 + radii_max),
            )
            check(name + "_box_membership", independent["centers_inside_observed_box"])
            check(name + "_nonnegative_gap", independent["gap"] >= -allowance)
            recorded_converged = max(point["relative_gap"], point["kkt_residual"]) <= report["tol"]
            check(name + "_convergence_flag", recorded_converged == point["converged"])
            if point["converged"]:
                check(
                    name + "_independent_tolerance",
                    max(independent["relative_gap"], independent["kkt_residual"])
                    <= report["tol"] + allowance,
                )
            if heldout:
                residual = centers[~observed] - targets
                mse = math.fsum(float(r) ** 2 for r in residual) / len(targets)
                entry["recomputed_score_mse"] = mse
                close(name + "_score_mse", point["score_mse"], mse)
            else:
                labels = arrays[name + "_labels"]
                adjacency = sparse.csr_matrix(cdist(centers, centers) <= report["cluster_tol"])
                count, derived = connected_components(adjacency)
                check(name + "_partition", np.array_equal(canonical(labels), canonical(derived)))
                check(name + "_cluster_count", count == point["n_clusters"])
                truth = original[: len(x), 0].astype(int)
                check(name + "_original_truth", np.array_equal(arrays["truth_posthoc_only"], truth))
                ari = float(adjusted_rand_score(truth, derived))
                close(name + "_ari", ari, point["posthoc_ari"])
                close(
                    name + "_center_bound", point["center_error_bound"], math.sqrt(2 * point["gap"])
                )
                entry.update(n_clusters=int(count), posthoc_ari=ari)
        result["status"] = "passed" if all(checks.values()) else "failed"
    return result


def audit_study(directory):
    manifest = json.loads((directory / "study.json").read_text())
    selection = json.loads((directory / "selection.json").read_text())
    original_path = ROOT / "examples/data/wine.data"
    original = np.loadtxt(original_path, delimiter=",")
    smoke = bool(manifest["smoke"])
    checks = dict(
        data_hash=file_hash(original_path) == manifest["data_sha256"],
        selection_hash=file_hash(directory / "selection.json") == manifest["selection_sha256"],
        selection_matches_manifest=selection == manifest["selection"],
    )
    workers, reports = [], []
    for process in manifest["processes"]:
        path = directory / process["checkpoint"]
        checks[process["name"] + "_checkpoint_hash"] = file_hash(path) == process["json_sha256"]
        checks[process["name"] + "_arrays_hash"] = (
            file_hash(path.with_suffix(".npz")) == process["npz_sha256"]
        )
        report = json.loads(path.read_text())
        checks[process["name"] + "_data_hash"] = report["data_sha256"] == manifest["data_sha256"]
        result = audit_worker(path, original, smoke)
        result["process_status"] = process["process_status"]
        workers.append(result)
        reports.append(report)
    expected_seeds = [1729] if smoke else [1729, 1730, 1731]
    draws = [w for w in workers if w["kind"] == "holdout"]
    checks["frozen_seeds"] = [w["seed"] for w in draws] == expected_seeds
    factors = [0.1, 10.0] if smoke else FACTORS.tolist()
    complete = len(draws) == len(expected_seeds) and all(
        w["worker_status"] == "completed"
        and w["process_status"] == "completed"
        and len(w["points"]) == len(factors)
        and all(
            p["converged"] and math.isfinite(p.get("recomputed_score_mse", float("nan")))
            for p in w["points"]
        )
        for w in draws
    )
    aggregate = baseline = chosen = None
    if complete:
        total = sum(w["validation_count"] for w in draws)
        aggregate = [
            math.fsum(w["points"][i]["recomputed_score_mse"] * w["validation_count"] for w in draws)
            / total
            for i in range(len(factors))
        ]
        baseline = math.fsum(w["baseline_mse"] * w["validation_count"] for w in draws) / total
        chosen = min(range(len(factors)), key=lambda i: aggregate[i])
        checks["pooled_scores"] = bool(
            np.allclose(selection["aggregate_mse"], aggregate, rtol=1e-12, atol=1e-12)
        )
        checks["pooled_baseline"] = bool(
            np.isclose(selection["baseline_mse"], baseline, rtol=1e-12, atol=1e-12)
        )
        checks["scored_entries"] = selection["scored_entries"] == total
        checks["selected_index"] = selection["selected_index"] == chosen
    checks["selection_gate"] = selection["status"] == ("valid" if complete else "incomplete")
    checks["selected_factor"] = selection["selected_factor"] == (
        factors[chosen] if chosen is not None else None
    )
    full = next(r for r in reports if r["kind"] == "full")
    checks["full_refit_factor"] = full.get("selected_factor") == selection["selected_factor"]
    if chosen is not None:
        selected_point = next((p for p in full["points"] if p["index"] == chosen), None)
        if selected_point is None:
            checks["selected_refit_status"] = full.get("selected_status") != "valid"
        else:
            expected_refit = "valid" if selected_point["converged"] else "refit_incomplete"
            checks["selected_refit_status"] = full.get("selected_status") == expected_refit
    checks["manifest_refit_status"] = manifest["selected_refit_status"] == full.get(
        "selected_status", "incomplete"
    )
    return dict(
        status="passed"
        if all(checks.values()) and all(w["status"] == "passed" for w in workers)
        else "failed",
        smoke=smoke,
        audit_source_sha256=file_hash(__file__),
        manifest_sha256=file_hash(directory / "study.json"),
        scope="Independent artifact reconstruction, masked/full primal-dual certificates, original KKT, and selection; no fitting",
        numeric_policy={
            "reconstruction_rtol": 2e-12,
            "reconstruction_default_atol": 1e-11,
            "certificate_atol": "256 * float64 epsilon * (1 + abs(primal objective))",
            "dual_feasibility_allowance": "64 * float64 epsilon * (1 + maximum edge radius)",
            "selection_rtol_and_atol": 1e-12,
        },
        limitations=[
            "All comparisons are floating-point checks with stated scale-aware tolerances, not interval-arithmetic proofs.",
            "The masked box dual certifies the original optimal objective value, not a unique held-out prediction or prediction-error bound.",
            "Artifact consistency cannot by itself prove the historical execution order or exclude all possible information leakage; that additionally requires source/protocol review.",
        ],
        checks=checks,
        recomputed_aggregate_mse=aggregate,
        recomputed_baseline_mse=baseline,
        recomputed_selected_index=chosen,
        workers=workers,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=ROOT / "docs/wine_holdout")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_study(args.directory)
    output = args.output or args.directory / "audit.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["status"], output)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
