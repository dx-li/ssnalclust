"""Frozen corrected partition-stability experiment in sequential fresh workers.

Run --output-dir NEW_DIRECTORY; --smoke checks plumbing, not scientific evidence.
Each worker checkpoints scalar JSON and referenced arrays, then freezes a separate
selection.json before audit fits. Truth is used only for final posthoc evaluation.
"""

import argparse
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from wine_holdout_study import THREAD_NAMES, _checkpoint, _hash, _peak_rss, _write_json

FACTORS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
SCENARIOS = ("separated", "overlap", "single_gaussian")
SEEDS = (8101, 8102, 8103)


def generate_data(scenario, seed, samples=120):
    """Generate fixed features and separate posthoc truth; preserve shuffled IDs."""
    import numpy as np

    rng = np.random.default_rng(seed)
    if scenario == "single_gaussian":
        X = rng.normal(size=(samples, 2))
        truth = np.zeros(samples, dtype=int)
    elif scenario in {"separated", "overlap"}:
        truth = np.repeat(np.arange(3), samples // 3)
        means = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        X = means[truth] + rng.normal(
            scale=0.25 if scenario == "separated" else 0.9, size=(samples, 2)
        )
    else:
        raise ValueError("Unknown scenario")
    order = rng.permutation(samples)
    return X[order], truth[order]


def sample_pairs(samples, seed, count):
    """Independent members; sorted indices align fits to original observations."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [
        (
            np.sort(rng.choice(samples, int(0.8 * samples), replace=False)),
            np.sort(rng.choice(samples, int(0.8 * samples), replace=False)),
        )
        for _ in range(count)
    ]


def _fit_sample(X, ids, factors, name, report, arrays, save, neighbors):
    import numpy as np
    from heldout_preprocessing import prepare_training

    from ssnalclust import ConvexClusteringProblem, summarize_path

    record = dict(name=name, status="running", factors=list(factors), points=[])
    report["samples_fitted"].append(record)
    started = time.perf_counter()
    subset = X[ids]
    prepared = prepare_training(subset, np.ones(subset.shape, dtype=bool), neighbors=neighbors)
    graph = prepared.graph
    members = dict(
        ids=ids,
        training_values=prepared.training_values,
        means=prepared.means,
        scales=prepared.scales,
        graph_indptr=graph.indptr,
        graph_indices=graph.indices,
        graph_data=graph.data,
    )
    record["arrays"] = {key: name + "_" + key for key in members}
    arrays.update({record["arrays"][key]: value for key, value in members.items()})
    record.update(
        bandwidth=prepared.bandwidth,
        components=prepared.components,
        edges=graph.nnz // 2,
        preparation_seconds=time.perf_counter() - started,
    )
    gammas = np.asarray(factors) * prepared.bandwidth / neighbors
    problem = ConvexClusteringProblem(prepared.training_values, weights=graph)
    iterator = problem.iter_path(
        gammas, tol=1e-6, solver="ssnal", max_iter=300, check_every=1, store_history=False
    )
    partitions = []
    save()
    try:
        for index, factor in enumerate(factors):
            started = time.perf_counter()
            result = next(iterator)
            seconds = time.perf_counter() - started
            summary = summarize_path([result], cluster_tol=1e-4)[0]
            prefix = f"{name}_point_{index}"
            refs = {key: prefix + "_" + key for key in ("centers", "dual", "labels")}
            arrays[refs["centers"]] = result.centers
            arrays[refs["dual"]] = result.dual
            arrays[refs["labels"]] = summary["labels"]
            partitions.append(summary["labels"])
            counts = np.bincount(summary["labels"])
            record["points"].append(
                dict(
                    index=index,
                    factor=float(factor),
                    gamma=float(gammas[index]),
                    converged=bool(result.converged),
                    n_iter=result.n_iter,
                    objective=result.objective,
                    dual_objective=result.dual_objective,
                    gap=result.gap,
                    relative_gap=result.relative_gap,
                    kkt_residual=result.kkt_residual,
                    center_error_bound=result.center_error_bound,
                    n_clusters=summary["n_clusters"],
                    singleton_fraction=float(np.count_nonzero(counts == 1) / len(ids)),
                    arrays=refs,
                    solve_seconds=seconds,
                    peak_rss_bytes=_peak_rss(),
                )
            )
            save()
    finally:
        iterator.close()
    record["status"] = "completed" if all(p["converged"] for p in record["points"]) else "failed"
    save()
    return record, partitions


def run_worker(args):
    import numpy as np
    from partition_stability import compare_partitions, select_stable_factor

    import ssnalclust

    n = 24 if args.smoke else 120
    factors = [0.0, 1.0, 30.0] if args.smoke else list(FACTORS)
    count, neighbors, min_pairs = (1, 5, 1) if args.smoke else (5, 10, 10)
    X, truth = generate_data(args.scenario, args.seed, n)
    arrays = dict(source_X=X, truth_posthoc_only=truth)
    root = Path(ssnalclust.__file__).parent
    helper = Path(__file__).parent
    report = dict(
        status="running",
        scenario=args.scenario,
        seed=args.seed,
        smoke=args.smoke,
        samples=n,
        features=2,
        factors=factors,
        tuning_pairs=count,
        audit_pairs=count,
        neighbors=neighbors,
        min_pairs=min_pairs,
        cluster_tol=1e-4,
        tol=1e-6,
        solver="ssnal",
        max_iter=300,
        check_every=1,
        store_history=False,
        versions={
            name: importlib.metadata.version(name)
            for name in ("ssnalclust", "numpy", "scipy", "scikit-learn")
        },
        python=platform.python_version(),
        platform=platform.platform(),
        source_sha256={p.name: _hash(p) for p in sorted(root.glob("*.py"))},
        helper_sha256={
            name: _hash(helper / name)
            for name in (
                "stability_study.py",
                "heldout_preprocessing.py",
                "partition_stability.py",
                "wine_holdout_study.py",
            )
        },
        requested_thread_environment={key: os.environ.get(key) for key in THREAD_NAMES},
        source_arrays={"features": "source_X", "truth": "truth_posthoc_only"},
        samples_fitted=[],
        tuning=[],
        audit=[],
        selection={"status": "pending"},
    )
    destination = args.output_dir / "report.json"
    started = time.perf_counter()

    def save():
        report["worker_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = _peak_rss()
        _checkpoint(destination, report, arrays)
        # Include array compression/checkpoint allocation in the final observed
        # high-water mark and elapsed time, not just the preceding solver call.
        report["worker_seconds"] = time.perf_counter() - started
        report["peak_rss_bytes"] = _peak_rss()
        _write_json(destination, report)

    def fit_pair(ids_pair, phase, pair_index, selected_factors):
        fitted = []
        for member, ids in enumerate(ids_pair):
            fitted.append(
                _fit_sample(
                    X,
                    ids,
                    selected_factors,
                    f"{phase}_{pair_index}_{member}",
                    report,
                    arrays,
                    save,
                    neighbors,
                )
            )
        shared, a, b = np.intersect1d(*ids_pair, return_indices=True)
        comparisons = [
            compare_partitions(left[a], right[b], min_pairs=min_pairs)
            for left, right in zip(fitted[0][1], fitted[1][1])
        ]
        pair_report = dict(
            index=pair_index,
            samples=[item[0]["name"] for item in fitted],
            shared_ids=shared.tolist(),
            comparisons=comparisons,
            all_converged=all(item[0]["status"] == "completed" for item in fitted),
        )
        report[phase].append(pair_report)
        save()
        return pair_report

    save()
    try:
        for index, pair in enumerate(sample_pairs(n, 100000 + args.seed, count)):
            fit_pair(pair, "tuning", index, factors)
        if all(pair["all_converged"] for pair in report["tuning"]):
            selection = select_stable_factor([p["comparisons"] for p in report["tuning"]], factors)
            selection["status"] = (
                "valid" if selection["selected_index"] is not None else "no_selection"
            )
        else:
            selection = dict(
                status="failed",
                selected_index=None,
                selected_factor=None,
                no_selection_reason="unconverged_tuning_fit",
            )
        # This separate immutable-by-convention decision artifact precedes every audit fit.
        _write_json(args.output_dir / "selection.json", selection)
        report["selection"] = selection
        report["selection_sha256"] = _hash(args.output_dir / "selection.json")
        save()
        selected = selection["selected_factor"]
        if selected is not None:
            for index, pair in enumerate(sample_pairs(n, 200000 + args.seed, count)):
                fit_pair(pair, "audit", index, [selected])
            comparisons = [pair["comparisons"][0] for pair in report["audit"]]
            valid = all(pair["all_converged"] for pair in report["audit"]) and all(
                comparison["eligible"] for comparison in comparisons
            )
            report["audit_summary"] = dict(
                status="valid" if valid else "ineligible_or_failed",
                eligible=valid,
                mean_score=float(np.mean([p["corrected_disagreement"] for p in comparisons]))
                if valid
                else None,
            )
        else:
            report["audit_summary"] = dict(status="no_selection", eligible=False, mean_score=None)
        full, partitions = _fit_sample(
            X, np.arange(n), factors, "full", report, arrays, save, neighbors
        )
        # All fits and the selection decision exist before truth-based evaluation.
        from sklearn.metrics import adjusted_rand_score

        for point, labels in zip(full["points"], partitions):
            point["posthoc_ari"] = float(adjusted_rand_score(truth, labels))
        report["status"] = (
            "completed"
            if all(s["status"] == "completed" for s in report["samples_fitted"])
            else "failed"
        )
        report["selection_audit_complete"] = (
            selection["status"] == "valid"
            and report["audit_summary"]["eligible"]
            and report["status"] == "completed"
        )
        save()
        return 0 if report["status"] == "completed" else 1
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
            selection_audit_complete=False,
        )
        save()
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scenario", choices=SCENARIOS, default="separated", help=argparse.SUPPRESS
    )
    parser.add_argument("--seed", type=int, default=8101, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return run_worker(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        status="running",
        smoke=args.smoke,
        processes=[],
        harness_sha256=_hash(__file__),
        protocol_sha256=_hash(Path(__file__).resolve().parents[1] / "docs/stability_protocol.md"),
    )
    destination = args.output_dir / "study.json"
    _write_json(destination, manifest)
    environment = os.environ.copy()
    environment.update({key: "1" for key in THREAD_NAMES})
    for scenario in SCENARIOS[:1] if args.smoke else SCENARIOS:
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            name = f"{scenario}_{seed}"
            folder = args.output_dir / name
            folder.mkdir()
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--output-dir",
                str(folder.resolve()),
            ]
            if args.smoke:
                command.append("--smoke")
            process = dict(
                name=name,
                scenario=scenario,
                seed=seed,
                process_status="running",
                timeout_seconds=600,
                checkpoint=f"{name}/report.json",
                arrays=f"{name}/report.npz",
                selection=f"{name}/selection.json",
                log=f"{name}/worker.txt",
            )
            manifest["processes"].append(process)
            _write_json(destination, manifest)
            started = time.perf_counter()
            print(f"Starting {name}", flush=True)
            with (folder / "worker.txt").open("w", encoding="utf-8") as log:
                try:
                    child = subprocess.run(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600
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
            for field in ("checkpoint", "arrays", "selection", "log"):
                path = args.output_dir / process[field]
                if path.is_file():
                    process[field + "_sha256"] = _hash(path)
            _write_json(destination, manifest)
            print(f"Finished {name}: {process['process_status']}", flush=True)
    success = all(p["process_status"] == "completed" for p in manifest["processes"])
    manifest["status"] = "completed" if success else "failed"
    _write_json(destination, manifest)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
