"""Fixed-grid recovery experiments; truth is used only after fitting.

Quick check: python examples/recovery_study.py --smoke --self-test
Full study: python examples/recovery_study.py --output docs/recovery_study_results.jsonl
Each dataset/graph combination runs in a fresh worker with its own deadline.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path


def fit_fixed_grid(
    X, *, neighbors, bandwidth, gammas, solver, tol, max_iter, cluster_tol, problem_factory=None
):
    """Fit a prespecified path using X alone; no truth or desired k is accepted."""
    from scipy.sparse.csgraph import connected_components

    from ssnalclust import ConvexClusteringProblem, k_neighbors_graph
    from ssnalclust.estimator import _centroid_labels

    start = time.perf_counter()
    graph = k_neighbors_graph(X, n_neighbors=min(neighbors, len(X) - 1), bandwidth=bandwidth)
    graph_seconds = time.perf_counter() - start
    factory = ConvexClusteringProblem if problem_factory is None else problem_factory
    problem = factory(X, weights=graph)
    graph_details = dict(
        edges=graph.nnz // 2,
        components=int(connected_components(graph)[0]),
        graph_seconds=graph_seconds,
    )
    x0 = dual0 = None
    for gamma in gammas:
        start = time.perf_counter()
        result = problem.solve(
            gamma=gamma,
            solver=solver,
            tol=tol,
            max_iter=max_iter,
            x0=x0,
            dual0=dual0,
            store_history=False,
        )
        labels = _centroid_labels(result.centers, cluster_tol)
        elapsed = time.perf_counter() - start
        x0, dual0 = result.centers, result.dual
        yield dict(
            gamma=gamma,
            labels=labels,
            centers=result.centers,
            seconds=elapsed,
            converged=result.converged,
            iterations=result.n_iter,
            relative_gap=result.relative_gap,
            kkt_residual=result.kkt_residual,
            center_error_bound=result.center_error_bound,
            objective=result.objective,
            **graph_details,
        )


def recovery_metrics(truth, labels):
    """ARI and one-to-one matching accuracy for the two-class study design."""
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    classes, y = np.unique(truth, return_inverse=True)
    clusters, z = np.unique(labels, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("This study's matching accuracy is defined for two true classes")
    counts = np.zeros((2, len(clusters)), dtype=int)
    np.add.at(counts, (y, z), 1)
    if len(clusters) == 1:
        matched = counts.max()
    else:
        assignments = counts[0, :, None] + counts[1, None, :]
        np.fill_diagonal(assignments, -1)
        matched = assignments.max()
    return dict(
        ari=float(adjusted_rand_score(truth, labels)),
        matched_accuracy=float(matched / len(truth)),
        n_clusters=len(clusters),
    )


def _dataset(config):
    from sklearn.datasets import make_circles, make_moons

    kwargs = dict(
        n_samples=config["samples"],
        noise=config["noise"],
        random_state=config["seed"],
        shuffle=True,
    )
    return (
        make_moons(**kwargs)
        if config["dataset"] == "moons"
        else make_circles(**kwargs, factor=config["circle_factor"])
    )


def _options(config):
    return {
        key: config[key]
        for key in ["neighbors", "bandwidth", "gammas", "solver", "tol", "max_iter", "cluster_tol"]
    }


def _worker(config):
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering, KMeans

    import ssnalclust

    def emit(kind, **fields):
        print(json.dumps(dict(kind=kind, **fields), allow_nan=False), flush=True)

    source = Path(ssnalclust.__file__).parent
    emit(
        "metadata",
        python=platform.python_version(),
        platform=platform.platform(),
        versions={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn", "ssnalclust"]
        },
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        },
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        requested_threads=1,
    )
    try:
        X, truth = _dataset(config)
        # The fit routine cannot access truth: it receives X and a fixed grid.
        scored = []
        for fitted in fit_fixed_grid(X, **_options(config)):
            metrics = recovery_metrics(truth, fitted["labels"])
            report = {
                key: value for key, value in fitted.items() if key not in ("labels", "centers")
            }
            report.update(metrics)
            scored.append(report)
            emit("convex_grid_point", **report)

        # These comparators receive the true k and are explicitly oracle-k.
        oracle_k = len(np.unique(truth))
        for name, estimator in [
            (
                "kmeans_oracle_k",
                KMeans(n_clusters=oracle_k, n_init=20, random_state=config["seed"]),
            ),
            (
                "single_linkage_oracle_k",
                AgglomerativeClustering(n_clusters=oracle_k, linkage="single"),
            ),
        ]:
            start = time.perf_counter()
            labels = estimator.fit_predict(X)
            emit(
                "comparator",
                method=name,
                oracle_k=oracle_k,
                seconds=time.perf_counter() - start,
                **recovery_metrics(truth, labels),
            )
        eligible = [point for point in scored if point["converged"]]
        best = max(eligible, key=lambda point: point["ari"]) if eligible else None
        emit(
            "summary",
            status="completed",
            all_grid_points_converged=len(eligible) == len(scored),
            oracle_best_converged_gamma=None if best is None else best["gamma"],
            oracle_best_converged_ari=None if best is None else best["ari"],
            oracle_best_converged_n_clusters=None if best is None else best["n_clusters"],
            oracle_warning="Best-over-grid ARI uses truth: an upper-envelope diagnostic, not unsupervised selection performance.",
        )
    except Exception as exc:
        emit(
            "summary",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )


def test_truth_is_not_supplied_to_fitting():
    """Runtime audit of fitter inputs, plus invariance to evaluation labels."""
    import numpy as np

    from ssnalclust import ConvexClusteringProblem

    config = dict(
        dataset="moons",
        samples=20,
        noise=0.02,
        seed=19,
        circle_factor=0.5,
        neighbors=3,
        bandwidth=0.3,
        gammas=[0.0, 0.2],
        solver="ssnal",
        tol=1e-6,
        max_iter=100,
        cluster_tol=1e-3,
    )
    X, truth = _dataset(config)
    calls = []

    class AuditedProblem:
        def __init__(self, observations, *, weights):
            np.testing.assert_array_equal(observations, X)
            self.actual = ConvexClusteringProblem(observations, weights=weights)

        def solve(self, **kwargs):
            assert set(kwargs) == {
                "gamma",
                "solver",
                "tol",
                "max_iter",
                "x0",
                "dual0",
                "store_history",
            }
            calls.append(kwargs["gamma"])
            return self.actual.solve(**kwargs)

    fits = list(fit_fixed_grid(X, problem_factory=AuditedProblem, **_options(config)))
    assert calls == config["gammas"]
    snapshots = [(f["labels"].copy(), f["centers"].copy()) for f in fits]
    for fitted, (labels, centers) in zip(fits, snapshots):
        recovery_metrics(truth, fitted["labels"])
        recovery_metrics(np.random.default_rng(18).permutation(truth), fitted["labels"])
        np.testing.assert_array_equal(fitted["labels"], labels)
        np.testing.assert_array_equal(fitted["centers"], centers)
    assert recovery_metrics([0, 0, 1, 1], [5, 5, 2, 2])["matched_accuracy"] == 1
    assert recovery_metrics([0, 0, 1, 1], [0, 1, 2, 3])["matched_accuracy"] == 0.5


def _run(config, timeout):
    environment = os.environ.copy()
    environment.update(
        {
            key: "1"
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        }
    )
    started = time.perf_counter()
    status = None
    try:
        worker = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        stdout, stderr = worker.stdout, worker.stderr
        if worker.returncode:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        # run kills and reaps this worker. Completed grid points are retained.
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
        status = "timeout"
    records = []
    for line in stdout.splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    summary = next((record for record in records if record["kind"] == "summary"), {})
    return dict(
        config=config,
        status=status or summary.get("status", "error"),
        process_seconds=time.perf_counter() - started,
        timeout_seconds=timeout,
        records=records,
        stderr=stderr,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[100, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--datasets", choices=["moons", "circles"], nargs="+", default=["moons", "circles"]
    )
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.05])
    parser.add_argument(
        "--graphs", nargs="+", default=["5:0.2", "10:0.5"], help="neighbor:bandwidth pairs"
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    )
    parser.add_argument("--circle-factor", type=float, default=0.5)
    parser.add_argument("--solver", choices=["ssnal", "admm", "ama", "fama"], default="ssnal")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--cluster-tol", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(args.worker))
        return
    if args.self_test:
        test_truth_is_not_supplied_to_fitting()
        print("Truth-separation runtime audit passed", file=sys.stderr, flush=True)
    if args.smoke:
        args.samples, args.seeds, args.datasets = [40], [0], ["moons"]
        args.noise, args.graphs, args.gammas = [0.05], ["5:0.2"], [0.0, 0.3, 3.0]
    try:
        graphs = []
        for graph in args.graphs:
            neighbor, bandwidth = graph.split(":")
            graphs.append((int(neighbor), float(bandwidth)))
    except ValueError:
        parser.error("graphs must be neighbor:bandwidth pairs")
    if (
        min(args.samples) < 2
        or args.max_iter < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.tol)
        or args.tol <= 0
        or not math.isfinite(args.cluster_tol)
        or args.cluster_tol < 0
        or not math.isfinite(args.circle_factor)
        or not 0 <= args.circle_factor < 1
        or any(not math.isfinite(v) or v < 0 for v in args.noise + args.gammas)
        or any(k < 1 or not math.isfinite(b) or b <= 0 for k, b in graphs)
    ):
        parser.error(
            "require samples>=2, positive iteration/time/tolerance/graph settings, "
            "nonnegative finite noise/gamma/cluster_tol, and circle_factor in [0,1)"
        )
    for dataset, n, noise, seed, (neighbors, bandwidth) in itertools.product(
        args.datasets, args.samples, args.noise, args.seeds, graphs
    ):
        config = dict(
            dataset=dataset,
            samples=n,
            noise=noise,
            seed=seed,
            neighbors=neighbors,
            bandwidth=bandwidth,
            gammas=args.gammas,
            circle_factor=args.circle_factor,
            solver=args.solver,
            tol=args.tol,
            cluster_tol=args.cluster_tol,
            max_iter=args.max_iter,
        )
        print(
            f"Starting {dataset}, n={n}, noise={noise}, seed={seed}, graph={neighbors}:{bandwidth}",
            file=sys.stderr,
            flush=True,
        )
        result = _run(config, args.timeout)
        encoded = json.dumps(result, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as output:
                output.write(encoded + "\n")
        else:
            print(encoded, flush=True)
        points = [r for r in result["records"] if r["kind"] == "convex_grid_point"]
        print(
            f"Finished {result['status']}, {len(points)} grid points, {result['process_seconds']:.2f}s",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
