"""Prespecified, process-bounded Wine held-out-entry tuning study.

No data download occurs. --smoke uses first24rows, seed1729, factors0.1/10.
Every worker has a900second deadline. Use a fresh --output-dir: JSON/NPZ pairs
replace arrays first, preserving an older JSON prefix after interruption.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)
SEEDS = (1729, 1730, 1731)
THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _peak_rss():
    if sys.platform not in {"linux", "darwin"}:
        return None
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _write_json(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint(path, report, arrays):
    import numpy as np

    array_path = path.with_suffix(".npz")
    temporary = array_path.with_name(array_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, array_path)
    _write_json(path, report)


def load_features(path, smoke=False):
    """Read feature columns only; cultivar labels are not loaded by tuning."""
    import numpy as np

    X = np.loadtxt(path, delimiter=",", usecols=range(1, 14), ndmin=2)
    if X.shape[1] != 13 or not np.isfinite(X).all():
        raise ValueError("Wine features must have13 finite numeric columns")
    return X[:24].copy() if smoke else X


def make_training_mask(shape, seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    observed = np.ones(shape, dtype=bool)
    count = int(0.1 * shape[0])
    if count < 1 or count >= shape[0]:
        raise ValueError("Need at least one held-out and one training entry per feature")
    for feature in range(shape[1]):
        observed[rng.permutation(shape[0])[:count], feature] = False
    return observed


def _save_training(prepared, arrays, report):
    import numpy as np

    graph = prepared.graph
    arrays.update(
        training_values=prepared.training_values,
        training_mask=prepared.observed,
        feature_means=prepared.means,
        feature_scales=prepared.scales,
        graph_inputs=prepared.graph_inputs,
        graph_indptr=graph.indptr,
        graph_indices=graph.indices,
        graph_data=graph.data,
    )
    degree = np.asarray(graph.sum(axis=1)).ravel()
    report.update(
        samples=len(prepared.training_values),
        features=prepared.training_values.shape[1],
        neighbors=10,
        bandwidth=prepared.bandwidth,
        graph_components=prepared.components,
        graph_edges=graph.nnz // 2,
        graph_sha256=hashlib.sha256(
            graph.indptr.tobytes() + graph.indices.tobytes() + graph.data.tobytes()
        ).hexdigest(),
        weight_min=float(graph.data.min()),
        weight_median=float(np.median(graph.data)),
        weight_max=float(graph.data.max()),
        weighted_degree_min=float(degree.min()),
        weighted_degree_median=float(np.median(degree)),
        weighted_degree_max=float(degree.max()),
    )


def _certificate(result, seconds):
    return dict(
        fit_status="completed",
        converged=bool(result.converged),
        n_iter=result.n_iter,
        objective=result.objective,
        dual_objective=result.dual_objective,
        gap=result.gap,
        relative_gap=result.relative_gap,
        kkt_residual=result.kkt_residual,
        solve_seconds=seconds,
        peak_rss_bytes=_peak_rss(),
    )


def _holdout(X, args, report, arrays, save):
    import numpy as np
    from heldout_preprocessing import prepare_training

    from ssnalclust import solve_missing
    from ssnalclust.selection import _finite_heldout_mse

    training = make_training_mask(X.shape, args.seed)
    # Physical erasure precedes every statistic and graph calculation.
    erased = np.where(training, X, np.nan)
    started = time.perf_counter()
    prepared = prepare_training(erased, training, neighbors=10)
    report["preprocessing_seconds"] = time.perf_counter() - started
    _save_training(prepared, arrays, report)
    validation = ~training
    arrays["validation_mask"] = validation
    arrays["validation_targets_original"] = X[validation].copy()
    rows, columns = np.nonzero(validation)
    targets = (X[rows, columns] - prepared.means[columns]) / prepared.scales[columns]
    arrays["validation_targets_standardized"] = targets
    report["validation_count"] = len(targets)
    report["baseline_mse"] = _finite_heldout_mse(np.zeros_like(targets), targets)
    report["baseline"] = "training feature mean; standardized prediction zero"
    report["score_units"] = "training-standardized feature units"
    report["certificate_model"] = "observed_range_box; no imputation error bound"
    save()
    for index, factor in enumerate(report["factors"]):
        gamma = factor * prepared.bandwidth / 10
        point = dict(index=index, factor=factor, gamma=gamma, score_mse=None)
        started = time.perf_counter()
        try:
            result = solve_missing(
                prepared.training_values,
                prepared.graph,
                gamma=gamma,
                observed=prepared.observed,
                penalty="l2",
                tol=1e-6,
                max_iter=30000,
            )
            point.update(_certificate(result, time.perf_counter() - started))
            arrays[f"point_{index}_centers"] = result.centers
            arrays[f"point_{index}_dual"] = result.dual
            diagnostic_started = time.perf_counter()
            try:
                point["score_mse"] = _finite_heldout_mse(result.centers[validation], targets)
                point["score_status"] = "finite"
            except ValueError as error:
                point["score_status"] = "error"
                point["score_error"] = str(error)
            point["diagnostics_seconds"] = time.perf_counter() - diagnostic_started
        except Exception as error:
            point.update(
                fit_status="error",
                converged=False,
                error_type=type(error).__name__,
                error=str(error),
                solve_seconds=time.perf_counter() - started,
            )
        report["points"].append(point)
        save()
        print(
            f"seed={args.seed} factor={factor} fit={point['fit_status']} converged={point['converged']}",
            flush=True,
        )


def _full(X, args, report, arrays, save):
    import numpy as np
    from heldout_preprocessing import prepare_training

    from ssnalclust import ConvexClusteringProblem
    from ssnalclust.estimator import _centroid_labels

    started = time.perf_counter()
    prepared = prepare_training(X, np.ones_like(X, dtype=bool), neighbors=10)
    report["preprocessing_seconds"] = time.perf_counter() - started
    _save_training(prepared, arrays, report)
    report["selected_factor"] = args.selected_factor
    report["selected_status"] = (
        "pending" if args.selected_factor is not None else "selection_incomplete"
    )
    report["cluster_tol"] = 1e-4
    started = time.perf_counter()
    problem = ConvexClusteringProblem(prepared.training_values, weights=prepared.graph)
    report["preparation_seconds"] = time.perf_counter() - started
    save()
    x0 = dual0 = None
    for index, factor in enumerate(report["factors"]):
        gamma = factor * prepared.bandwidth / 10
        point = dict(index=index, factor=factor, gamma=gamma)
        started = time.perf_counter()
        try:
            result = problem.solve(
                gamma=gamma,
                solver="ssnal",
                penalty="l2",
                tol=1e-6,
                max_iter=300,
                check_every=1,
                store_history=False,
                x0=x0,
                dual0=dual0,
            )
            point.update(_certificate(result, time.perf_counter() - started))
            diagnostic_started = time.perf_counter()
            point["center_error_bound"] = result.center_error_bound
            labels = _centroid_labels(result.centers, 1e-4)
            arrays[f"point_{index}_centers"] = result.centers
            arrays[f"point_{index}_dual"] = result.dual
            arrays[f"point_{index}_labels"] = labels
            point["n_clusters"] = int(np.unique(labels).size)
            point["diagnostics_seconds"] = time.perf_counter() - diagnostic_started
            x0, dual0 = result.centers, result.dual
        except Exception as error:
            point.update(
                fit_status="error",
                converged=False,
                error_type=type(error).__name__,
                error=str(error),
                solve_seconds=time.perf_counter() - started,
            )
        report["points"].append(point)
        if factor == args.selected_factor:
            report["selected_status"] = "valid" if point["converged"] else "refit_incomplete"
            report["selected_index"] = index
            report["selected_gamma"] = gamma
        save()
        print(
            f"full factor={factor} fit={point['fit_status']} converged={point['converged']}",
            flush=True,
        )
    # Cultivar labels are first loaded only after selection and all full fits.
    from sklearn.metrics import adjusted_rand_score

    truth = np.loadtxt(args.data, delimiter=",", usecols=0, ndmin=1)
    truth = truth[:24] if args.smoke else truth
    if not np.isfinite(truth).all() or not np.isin(truth, [1, 2, 3]).all():
        raise ValueError("Wine cultivar labels must be1,2,3")
    arrays["truth_posthoc_only"] = truth.astype(int)
    report["evaluation_purpose"] = "posthoc cultivar ARI; never selection"
    for point in report["points"]:
        labels = arrays.get(f"point_{point['index']}_labels")
        point["posthoc_ari"] = (
            float(adjusted_rand_score(truth, labels)) if labels is not None else None
        )
    save()


def _worker(args):
    import ssnalclust

    path = (
        args.output_dir
        / f"{args.kind}{'_' + str(args.seed) if args.kind == 'holdout' else ''}.json"
    )
    if path.exists() or path.with_suffix(".npz").exists():
        raise FileExistsError("Worker requires fresh checkpoint paths")
    root = Path(ssnalclust.__file__).parent
    report = dict(
        status="running",
        kind=args.kind,
        seed=args.seed if args.kind == "holdout" else None,
        smoke=args.smoke,
        factors=[0.1, 10.0] if args.smoke else list(FACTORS),
        points=[],
        data_sha256=_hash(args.data),
        source_sha256={p.name: _hash(p) for p in sorted(root.glob("*.py"))},
        harness_sha256=_hash(__file__),
        preprocessing_sha256=_hash(Path(__file__).with_name("heldout_preprocessing.py")),
        versions={
            name: importlib.metadata.version(name)
            for name in ("ssnalclust", "numpy", "scipy", "scikit-learn")
        },
        python=platform.python_version(),
        platform=platform.platform(),
        requested_thread_environment={name: os.environ.get(name) for name in THREAD_NAMES},
        checkpoint_consistency="arrays accumulate and replace before JSON; older JSON can have extra unreferenced arrays",
        checkpoint_seconds=0.0,
        checkpoint_count=0,
        checkpoint_timing="completed JSON+NPZ writes excluding final scalar refresh",
        solve_timing="solver call only; preprocessing, diagnostics and checkpoints separate",
        tol=1e-6,
        max_iter=30000 if args.kind == "holdout" else 300,
        penalty="l2",
        initialization="cold" if args.kind == "holdout" else "warm from previous returned point",
        solver="pdhg_missing" if args.kind == "holdout" else "ssnal",
        store_history=True if args.kind == "holdout" else False,
        peak_rss_scope="native process high-water, not current RSS; null when unsupported",
    )
    arrays = {}
    started = time.perf_counter()

    def save():
        report["worker_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = _peak_rss()
        checkpoint_started = time.perf_counter()
        _checkpoint(path, report, arrays)
        report["checkpoint_seconds"] += time.perf_counter() - checkpoint_started
        report["checkpoint_count"] += 1

    save()
    try:
        X = load_features(args.data, smoke=args.smoke)
        (_holdout if args.kind == "holdout" else _full)(X, args, report, arrays, save)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="error", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    save()
    report["worker_seconds"] = time.perf_counter() - started
    report["peak_rss_bytes"] = _peak_rss()
    _write_json(path, report)
    return 0 if report["status"] == "completed" else 1


def select_factor(reports, factors, expected_seeds):
    """Aggregate finite tuning scores only; never receives full-data fits or truth."""
    complete = (
        len(reports) == len(expected_seeds)
        and {r.get("seed") for r in reports} == set(expected_seeds)
        and all(
            r.get("status") == "completed"
            and len(r.get("points", [])) == len(factors)
            and r.get("validation_count", 0) > 0
            and r.get("baseline_mse") is not None
            and math.isfinite(r.get("baseline_mse", float("nan")))
            and r["baseline_mse"] >= 0
            and all(
                p.get("factor") == factor
                and p.get("index") == index
                and p.get("converged", False)
                and p.get("score_mse") is not None
                and math.isfinite(p["score_mse"])
                and p["score_mse"] >= 0
                for index, (p, factor) in enumerate(zip(r["points"], factors))
            )
            for r in reports
        )
    )
    result = dict(
        status="incomplete",
        selected_factor=None,
        factors=list(factors),
        aggregate_mse=None,
        baseline_mse=None,
        aggregation="sum squared errors / total held-out entries across draws; repeated entries count repeatedly",
    )
    if not complete:
        return result
    total = sum(r["validation_count"] for r in reports)
    try:
        scores = [
            math.fsum(
                r["points"][i]["score_mse"] * (r["validation_count"] / total) for r in reports
            )
            for i in range(len(factors))
        ]
        baseline = math.fsum(r["baseline_mse"] * (r["validation_count"] / total) for r in reports)
    except OverflowError:
        return result
    if not all(math.isfinite(value) for value in [*scores, baseline]):
        return result
    best = min(range(len(scores)), key=lambda i: scores[i])
    result.update(
        status="valid",
        selected_factor=factors[best],
        selected_index=best,
        aggregate_mse=scores,
        baseline_mse=baseline,
        scored_entries=total,
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data/wine.data")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--kind", choices=("holdout", "full"), default="holdout", help=argparse.SUPPRESS
    )
    parser.add_argument("--seed", type=int, default=1729, help=argparse.SUPPRESS)
    parser.add_argument("--selected-factor", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.output_dir, args.data = args.output_dir.resolve(), args.data.resolve()
    if args.worker:
        return _worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_NAMES})
    manifest = dict(
        status="running",
        smoke=args.smoke,
        data_sha256=_hash(args.data),
        harness_sha256=_hash(__file__),
        processes=[],
        selection={"status": "pending"},
    )
    destination = args.output_dir / "study.json"
    _write_json(destination, manifest)

    def run(kind, seed=None, selected_factor=None):
        name = f"holdout_{seed}" if kind == "holdout" else "full"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--kind",
            kind,
            "--output-dir",
            str(args.output_dir),
            "--data",
            str(args.data),
        ]
        if seed is not None:
            command += ["--seed", str(seed)]
        if selected_factor is not None:
            command += ["--selected-factor", str(selected_factor)]
        if args.smoke:
            command += ["--smoke"]
        process = dict(
            name=name,
            process_status="running",
            command=command,
            timeout_seconds=900,
            checkpoint=name + ".json",
            arrays=name + ".npz",
            log=name + ".txt",
            requested_thread_environment={key: env[key] for key in THREAD_NAMES},
        )
        manifest["processes"].append(process)
        _write_json(destination, manifest)
        started = time.perf_counter()
        print(f"Starting {name}", flush=True)
        with (args.output_dir / (name + ".txt")).open("x", encoding="utf-8") as log:
            try:
                child = subprocess.run(
                    command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900
                )
                process.update(
                    process_status="completed" if child.returncode == 0 else "error",
                    returncode=child.returncode,
                )
            except subprocess.TimeoutExpired:
                process.update(process_status="timeout", returncode=None)
            except OSError as error:
                process.update(process_status="error", returncode=None, error=str(error))
        process["process_wall_seconds"] = time.perf_counter() - started
        for suffix in ("json", "npz", "txt"):
            artifact = args.output_dir / (name + "." + suffix)
            process[suffix + "_exists"] = artifact.exists()
            if artifact.exists():
                process[suffix + "_sha256"] = _hash(artifact)
        report_path = args.output_dir / (name + ".json")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        process["returned_points"] = len(report.get("points", []))
        _write_json(destination, manifest)
        print(f"Finished {name}: {process['process_status']}", flush=True)
        return (
            report
            if process["process_status"] == "completed"
            else {**report, "status": process["process_status"]}
        )

    seeds = (1729,) if args.smoke else SEEDS
    reports = [run("holdout", seed) for seed in seeds]
    factors = (0.1, 10.0) if args.smoke else FACTORS
    selection = select_factor(reports, factors, seeds)
    manifest["selection"] = selection
    _write_json(args.output_dir / "selection.json", selection)
    manifest["selection_sha256"] = _hash(args.output_dir / "selection.json")
    _write_json(destination, manifest)
    full = run("full", selected_factor=selection["selected_factor"])
    manifest.update(
        status="completed",
        selected_refit_status=full.get("selected_status", "incomplete")
        if full.get("status") == "completed"
        else "incomplete",
    )
    _write_json(destination, manifest)
    return 0 if all(p["process_status"] == "completed" for p in manifest["processes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
