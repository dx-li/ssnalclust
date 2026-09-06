"""Process-isolated pipeline benchmarks with timeouts and native peak RSS.

Example: python examples/scalability.py --samples 1000 5000 10000 --solver ssnal
Each JSONL record describes one fresh worker. Timing results are measurements,
not pass/fail tests. A timeout never counts as an unconverged completed solve.
"""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _rss():
    # macOS reports bytes; Linux reports KiB. This is the process high-water
    # mark, including native numerical allocations and imported libraries.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _graph_evidence(graph):
    """Summarize a validated symmetric positive-edge CSR graph in O(n+m) storage."""
    import numpy as np
    from scipy.sparse.csgraph import connected_components

    degree = np.asarray(graph.sum(axis=1)).ravel()
    positive = graph.data[graph.data > 0]
    statistics = {}
    for name, values in [("positive_edge_weight", positive), ("weighted_degree", degree)]:
        statistics.update(
            {
                name + "_min": float(values.min()) if len(values) else None,
                name + "_median": float(np.median(values)) if len(values) else None,
                name + "_max": float(values.max()) if len(values) else None,
            }
        )
    return dict(
        connected_components=int(connected_components(graph, directed=False, return_labels=False)),
        isolated_vertices=int(np.count_nonzero(degree == 0)),
        **statistics,
    )


def _weighted_input_edge_penalty(X, graph):
    """Unscaled l2 fusion penalty at the input, using bounded edge batches."""
    import numpy as np
    from scipy import sparse

    edges = sparse.triu(graph, k=1, format="coo")
    batch = max(1, min(4096, 1_000_000 // X.shape[1]))
    total = 0.0
    for begin in range(0, edges.nnz, batch):
        end = begin + batch
        difference = X[edges.row[begin:end]] - X[edges.col[begin:end]]
        total += float(edges.data[begin:end] @ np.linalg.norm(difference, axis=1))
    return total


def _solution_evidence(
    X, graph, gamma, result, cluster_tol, centered_input_norm=None, initial_edge_penalty=None
):
    """Measure displacement and direct edge fusion, separate from label connectivity.

    With zero centered input norm, relative displacement is zero for an
    unchanged solution and null otherwise. No-edge fusion fractions are null.
    Edge differences are processed in bounded batches, never an m-by-p array.
    """
    import numpy as np
    from scipy import sparse

    displacement = float(np.linalg.norm(result.centers - X))
    if centered_input_norm is None:
        centered_input_norm = float(np.linalg.norm(X - X.mean(axis=0)))
    if centered_input_norm:
        relative = displacement / centered_input_norm
        normalization = "centered_input_norm"
    elif displacement == 0:
        relative = 0.0
        normalization = "zero_baseline_no_displacement"
    else:
        relative = None
        normalization = "undefined_zero_baseline"
    edges = sparse.triu(graph, k=1, format="coo")
    edges.eliminate_zeros()
    fused = 0
    # Bound temporary edge-by-feature differences to roughly one million
    # entries even when benchmarking high-dimensional observations.
    batch = max(1, min(4096, 1_000_000 // X.shape[1]))
    for begin in range(0, edges.nnz, batch):
        end = begin + batch
        difference = result.centers[edges.row[begin:end]] - result.centers[edges.col[begin:end]]
        fused += int(np.count_nonzero(np.linalg.norm(difference, axis=1) <= cluster_tol))
    if initial_edge_penalty is None:
        initial_edge_penalty = _weighted_input_edge_penalty(X, graph)
    return dict(
        gamma=float(gamma),
        gap=float(result.gap),
        zero_dual_initial_gap=float(gamma * initial_edge_penalty),
        centroid_displacement_frobenius=displacement,
        centered_input_norm=centered_input_norm,
        relative_centroid_displacement=relative,
        relative_displacement_normalization=normalization,
        direct_fused_edge_fraction=float(fused / edges.nnz) if edges.nnz else None,
    )


def _worker(config):
    emit_lock = threading.Lock()
    stage_timings = {}
    stage_calls = {}
    active_calls = {}

    def emit(event, **payload):
        with emit_lock:
            print(json.dumps(dict(event=event, **payload), allow_nan=False), flush=True)

    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(1):
            emit(
                "memory_sample",
                peak_rss_observed_bytes=_rss(),
                partial_stage_seconds=stage_timings.copy(),
                partial_stage_calls=stage_calls.copy(),
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    worker_start = time.perf_counter()
    try:
        import numpy as np

        import ssnalclust.estimator as estimator
        import ssnalclust.problem as problem
        import ssnalclust.solvers as solvers
        from ssnalclust import ConvexClustering, convex_clustering_path, k_neighbors_graph

        source_root = Path(estimator.__file__).parent
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source_root.glob("*.py"))
        }
        metadata = dict(
            python=platform.python_version(),
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            logical_cpu_count=os.cpu_count(),
            versions={
                name: importlib.metadata.version(name)
                for name in ["numpy", "scipy", "scikit-learn", "ssnalclust"]
            },
            requested_thread_environment={key: os.environ.get(key) for key in THREAD_VARIABLES},
            source_sha256=hashes,
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        emit("metadata", **metadata)

        def timed(name, function, events=False):
            def call(*args, **kwargs):
                outermost = active_calls.get(name, 0) == 0
                active_calls[name] = active_calls.get(name, 0) + 1
                if events and outermost:
                    emit("stage_start", stage=name, peak_rss_observed_bytes=_rss())
                start = time.perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    elapsed = time.perf_counter() - start
                    active_calls[name] -= 1
                    if outermost:
                        stage_timings[name] = stage_timings.get(name, 0.0) + elapsed
                    stage_calls[name] = stage_calls.get(name, 0) + 1
                    if events and outermost:
                        emit(
                            "stage_complete",
                            stage=name,
                            seconds=elapsed,
                            peak_rss_observed_bytes=_rss(),
                        )

            return call

        # Instrument actual pipeline calls, without replacing their algorithms.
        estimator._centroid_labels = timed("labels", estimator._centroid_labels, events=True)
        original_solve = estimator.solve

        def configured_solve(*args, **kwargs):
            kwargs.setdefault("check_every", config["check_every"])
            return original_solve(*args, **kwargs)

        estimator.solve = timed("solve", configured_solve, events=True)
        for attribute, name in [
            ("graph_from_weights", "incidence"),
            ("_diagnostics", "diagnostics"),
            ("_ssn_step", "newton_inner"),
            ("_newton_operator", "newton_operator"),
            ("_reduced", "reduced_objective_gradient"),
            ("_solve_prepared", "prepared_solve"),
        ]:
            if hasattr(solvers, attribute):
                setattr(solvers, attribute, timed(name, getattr(solvers, attribute)))
        original_prepared_solve = solvers._solve_prepared

        def reported_prepared_solve(prepared, *args, **kwargs):
            result = original_prepared_solve(prepared, *args, **kwargs)
            # A materialized path may later time out. Preserve completed point
            # certificates immediately without retaining more centroid arrays.
            gamma = kwargs.get("gamma", args[0] if args else 1.0)
            emit(
                "solution_complete",
                gamma=float(gamma),
                objective=result.objective,
                gap=result.gap,
                relative_gap=result.relative_gap,
                kkt_residual=result.kkt_residual,
                converged=result.converged,
                iterations=result.n_iter,
            )
            return result

        solvers._solve_prepared = reported_prepared_solve
        problem.graph_from_weights = timed("incidence", problem.graph_from_weights)
        problem.factorized = timed("factorization", problem.factorized)
        original_cg = solvers.cg

        def counted_cg(*args, **kwargs):
            original_callback = kwargs.pop("callback", None)

            def callback(iterate):
                stage_calls["cg_iterations"] = stage_calls.get("cg_iterations", 0) + 1
                if original_callback is not None:
                    original_callback(iterate)

            return original_cg(*args, callback=callback, **kwargs)

        solvers.cg = timed("conjugate_gradient", counted_cg)

        rng = np.random.default_rng(config["seed"])
        n, p = config["samples"], config["features"]
        if config["data"] == "gaussian":
            X = rng.normal(size=(n, p))
        else:
            # Fixed, separated groups; duplicate mode isolates label cost.
            groups = np.arange(n) % 5
            locations = np.arange(5)[:, None] * np.ones((5, p)) * 4
            X = locations[groups].copy()
            if config["data"] == "blobs":
                X += rng.normal(scale=0.25, size=X.shape)
        graph = timed("graph", k_neighbors_graph, events=True)(
            X, n_neighbors=min(config["neighbors"], n - 1), bandwidth=config["bandwidth"]
        )
        graph_evidence = timed("graph_evidence", _graph_evidence, events=True)(graph)
        emit(
            "graph_summary",
            edges=graph.nnz // 2,
            graph_nnz=graph.nnz,
            graph_storage_bytes=graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes,
            **graph_evidence,
        )
        pipeline_start = time.perf_counter()
        if config["scenario"] == "fit":
            model = ConvexClustering(
                weights=graph,
                gamma=config["gamma"],
                solver=config["solver"],
                tol=config["tol"],
                max_iter=config["max_iter"],
                cluster_tol=config["cluster_tol"],
            )
            timed("fit", model.fit, events=True)(X)
            results = [model.result_]
            clusters = [model.n_clusters_]
        elif config["scenario"] == "path":
            results = timed("path", convex_clustering_path, events=True)(
                X,
                config["path_gammas"],
                weights=graph,
                solver=config["solver"],
                tol=config["tol"],
                max_iter=config["max_iter"],
                check_every=config["check_every"],
            )
            clusters = [
                int(estimator._centroid_labels(result.centers, config["cluster_tol"]).max()) + 1
                for result in results
            ]
        else:
            labels = estimator._centroid_labels(X, config["cluster_tol"])
            results, clusters = [], [int(labels.max()) + 1]
        pipeline_seconds = time.perf_counter() - pipeline_start

        def summarize_solutions():
            if not results:
                return []
            centered_norm = float(np.linalg.norm(X - X.mean(axis=0)))
            initial_edge_penalty = _weighted_input_edge_penalty(X, graph)
            gammas = config["path_gammas"] if config["scenario"] == "path" else [config["gamma"]]
            return [
                dict(
                    converged=result.converged,
                    iterations=result.n_iter,
                    objective=result.objective,
                    relative_gap=result.relative_gap,
                    kkt_residual=result.kkt_residual,
                    center_error_bound=getattr(result, "center_error_bound", None),
                    **_solution_evidence(
                        X,
                        graph,
                        gamma,
                        result,
                        config["cluster_tol"],
                        centered_norm,
                        initial_edge_penalty,
                    ),
                )
                for gamma, result in zip(gammas, results)
            ]

        # Evidence collection is excluded from pipeline/solver timings, but
        # included in worker wall time and measured process peak memory.
        solutions = timed("solution_evidence", summarize_solutions, events=True)()
        emit(
            "result",
            status="completed",
            pipeline_seconds=pipeline_seconds,
            graph_plus_pipeline_seconds=stage_timings["graph"] + pipeline_seconds,
            evidence_seconds=stage_timings["graph_evidence"] + stage_timings["solution_evidence"],
            evidence_timing="excluded_from_pipeline_included_in_worker",
            worker_seconds=time.perf_counter() - worker_start,
            stage_seconds=stage_timings,
            stage_calls=stage_calls,
            solutions=solutions,
            clusters=clusters,
            all_converged=all(r.converged for r in results) if results else None,
            peak_rss_bytes=_rss(),
            peak_rss_complete=True,
        )
    except Exception as exc:
        emit(
            "result",
            status="error",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            worker_seconds=time.perf_counter() - worker_start,
            peak_rss_bytes=_rss(),
            peak_rss_complete=True,
        )
    finally:
        stopped.set()


def _run(config, timeout, threads):
    environment = os.environ.copy()
    environment.update({key: str(threads) for key in THREAD_VARIABLES})
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
            check=False,
        )
        stdout, stderr = completed.stdout, completed.stderr
        status = "error" if completed.returncode else None
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and reaps this specific worker before raising.
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
        status, returncode = "timeout", None
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # A worker killed while printing can leave one partial line.
    record = dict(
        config=config,
        timeout_seconds=timeout,
        process_wall_seconds=time.perf_counter() - start,
        returncode=returncode,
    )
    for event in events:
        if event["event"] in ("metadata", "graph_summary", "result"):
            record.update({key: value for key, value in event.items() if key != "event"})
    record["events"] = [event for event in events if event["event"] != "memory_sample"]
    record["peak_rss_observed_bytes"] = max(
        [event.get("peak_rss_observed_bytes", 0) for event in events], default=0
    )
    samples = [event for event in events if event["event"] == "memory_sample"]
    if samples:
        record["partial_stage_seconds"] = samples[-1].get("partial_stage_seconds", {})
        record["partial_stage_calls"] = samples[-1].get("partial_stage_calls", {})
    if status == "timeout":
        record.update(status="timeout", peak_rss_complete=False)
    elif "status" not in record:
        record["status"] = status or "error"
    if stderr:
        record["stderr"] = stderr
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=[1000, 5000, 10000])
    parser.add_argument("--features", type=int, nargs="+", default=[3])
    parser.add_argument("--solver", choices=["ssnal", "admm", "ama", "fama"], default="ssnal")
    parser.add_argument("--scenario", choices=["fit", "path", "labels"], default="fit")
    parser.add_argument("--data", choices=["gaussian", "blobs", "duplicates"], default="gaussian")
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--path-gammas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--cluster-tol", type=float, default=1e-4)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--check-every", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(args.worker))
        return
    if (
        min(args.samples) < 2
        or min(args.features) < 1
        or args.threads < 1
        or args.timeout <= 0
        or args.repeat < 1
        or args.neighbors < 1
        or args.max_iter < 1
        or args.check_every < 1
    ):
        parser.error(
            "samples >= 2 and positive features/threads/timeout/repeat/neighbors/"
            "max-iter/check-every required"
        )
    for n, p, repetition in itertools.product(args.samples, args.features, range(args.repeat)):
        config = {
            key: getattr(args, key)
            for key in [
                "solver",
                "scenario",
                "data",
                "gamma",
                "path_gammas",
                "neighbors",
                "bandwidth",
                "cluster_tol",
                "tol",
                "max_iter",
                "check_every",
                "seed",
            ]
        }
        config.update(samples=n, features=p, repetition=repetition)
        print(
            f"Starting {args.scenario}: n={n}, p={p}, {args.solver}, timeout={args.timeout:g}s",
            file=sys.stderr,
            flush=True,
        )
        record = _run(config, args.timeout, args.threads)
        encoded = json.dumps(record, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("a") as output:
                output.write(encoded + "\n")
        else:
            print(encoded, flush=True)
        print(
            f"Finished: {record['status']}, wall={record['process_wall_seconds']:.2f}s, "
            f"all_converged={record.get('all_converged')}",
            file=sys.stderr,
            flush=True,
        )


if __name__ == "__main__":
    main()
