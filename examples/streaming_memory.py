"""Fresh-process memory evidence for public materialized and streaming paths.

Run with --output NEW.jsonl; --smoke uses 40 rows, 3 features, lengths 3 and 5.
RSS is native process peak RSS, never current RSS. Array-byte counts cover only
consumer-owned result centers and dual arrays, not solver workspaces or the
iterator's warm-start snapshots. Each materialized point is recorded only after
public path() returns; a timeout therefore may have no returned-point evidence.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _peak_rss():
    if sys.platform not in {"linux", "darwin"}:
        return None
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _emit(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def _graph(X, k):
    import numpy as np
    from scipy import sparse
    from scipy.spatial.distance import cdist

    distances = cdist(X, X)
    ranking = distances.copy()
    np.fill_diagonal(ranking, np.inf)
    columns = np.argsort(ranking, axis=1, kind="stable")[:, :k].ravel()
    rows = np.repeat(np.arange(len(X)), k)
    directed = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(X), len(X)))
    edges = sparse.triu(directed.maximum(directed.T), 1, format="coo")
    lengths = distances[edges.row, edges.col]
    bandwidth = float(np.median(lengths[lengths > 0]))
    values = np.exp(-0.5 * (lengths / bandwidth) ** 2)
    upper = sparse.csr_matrix((values, (edges.row, edges.col)), shape=directed.shape)
    return (upper + upper.T).tocsr(), bandwidth


def _worker(args):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    import ssnalclust
    from ssnalclust import ConvexClusteringProblem

    identity = dict(
        mode=args.mode, length=args.length, samples=args.samples, features=args.features
    )
    rng = np.random.default_rng(7301)
    means = rng.normal(size=(4, args.features)) * 2
    X = means[np.arange(args.samples) % 4] + rng.normal(size=(args.samples, args.features)) * 0.35
    started = time.perf_counter()
    graph, bandwidth = _graph(X, args.neighbors)
    graph_seconds = time.perf_counter() - started
    started = time.perf_counter()
    problem = ConvexClusteringProblem(X, weights=graph)
    preparation_seconds = time.perf_counter() - started
    gammas = np.geomspace(0.005, 0.5, args.length)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    edges = sparse.triu(graph, 1, format="coo")
    centered_input_norm = float(np.linalg.norm(X - X.mean(axis=0)))
    baseline = _peak_rss()
    source = Path(ssnalclust.__file__).parent
    _emit(
        dict(
            event="prepared",
            **identity,
            seed=7301,
            neighbors=args.neighbors,
            bandwidth=bandwidth,
            edges=graph.nnz // 2,
            components=int(connected_components(graph, directed=False, return_labels=False)),
            isolated_vertices=int(np.count_nonzero(degree == 0)),
            weighted_degree_min=float(degree.min()),
            weighted_degree_median=float(np.median(degree)),
            weighted_degree_max=float(degree.max()),
            edge_weight_min=float(graph.data.min()),
            edge_weight_median=float(np.median(graph.data)),
            edge_weight_max=float(graph.data.max()),
            graph_csr_bytes=graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes,
            graph_sha256=hashlib.sha256(
                graph.indptr.tobytes() + graph.indices.tobytes() + graph.data.tobytes()
            ).hexdigest(),
            data_sha256=hashlib.sha256(X.tobytes()).hexdigest(),
            gammas=gammas.tolist(),
            graph_seconds=graph_seconds,
            preparation_seconds=preparation_seconds,
            baseline_peak_rss_bytes=baseline,
            peak_rss_bytes=baseline,
            rss_scope="process high-water including imports and graph construction; not current RSS",
            retained_bytes_scope="consumer centers and dual only; excludes iterator snapshots and solver workspaces",
            versions={
                name: importlib.metadata.version(name)
                for name in ("ssnalclust", "numpy", "scipy", "scikit-learn")
            },
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.glob("*.py"))
            },
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            python=platform.python_version(),
            platform=platform.platform(),
            requested_thread_environment={key: os.environ.get(key) for key in THREAD_NAMES},
            solver="ssnal",
            tol=1e-6,
            max_iter=300,
            store_history=False,
            check_every=1,
            cluster_tol=1e-4,
        )
    )
    options = dict(solver="ssnal", tol=1e-6, max_iter=300, check_every=1, store_history=False)
    started = time.perf_counter()
    if args.mode == "materialized":
        results = problem.path(gammas, **options)
        path_seconds = time.perf_counter() - started
        retained = sum(r.centers.nbytes + r.dual.nbytes for r in results)
        iterator = iter(results)
        _emit(
            dict(
                event="path_returned",
                **identity,
                path_seconds=path_seconds,
                consumer_retained_center_dual_bytes=retained,
                peak_rss_bytes=_peak_rss(),
            )
        )
    else:
        iterator = problem.iter_path(gammas, **options)
        retained = 0
        path_seconds = None
    all_converged = True
    count = 0
    diagnostics_seconds = 0.0
    generated_bytes = 0
    while True:
        try:
            result = next(iterator)
        except StopIteration:
            break
        diagnostic_started = time.perf_counter()
        count += 1
        center_bytes, dual_bytes = result.centers.nbytes, result.dual.nbytes
        generated_bytes += center_bytes + dual_bytes
        displacement = float(np.linalg.norm(result.centers - X))
        fused = 0
        for begin in range(0, edges.nnz, 4096):
            differences = (
                result.centers[edges.row[begin : begin + 4096]]
                - result.centers[edges.col[begin : begin + 4096]]
            )
            fused += int(np.count_nonzero(np.linalg.norm(differences, axis=1) <= 1e-4))
        del differences
        all_converged = all_converged and result.converged
        record = dict(
            event="point",
            **identity,
            index=count - 1,
            gamma=float(gammas[count - 1]),
            converged=bool(result.converged),
            n_iter=result.n_iter,
            objective=result.objective,
            dual_objective=result.dual_objective,
            gap=result.gap,
            relative_gap=result.relative_gap,
            kkt_residual=result.kkt_residual,
            center_error_bound=result.center_error_bound,
            centers_sha256=hashlib.sha256(result.centers.tobytes()).hexdigest(),
            dual_sha256=hashlib.sha256(result.dual.tobytes()).hexdigest(),
            displacement_frobenius=displacement,
            relative_displacement=displacement / centered_input_norm
            if centered_input_norm
            else None,
            direct_fused_edge_fraction=fused / edges.nnz if edges.nnz else None,
            cumulative_generated_center_dual_bytes=generated_bytes,
            center_array_bytes=center_bytes,
            dual_array_bytes=dual_bytes,
            consumer_retained_center_dual_bytes=retained
            if args.mode == "materialized"
            else center_bytes + dual_bytes,
            peak_rss_bytes=_peak_rss(),
            elapsed_since_path_start_seconds=time.perf_counter() - started,
            history_entries=len(result.history),
        )
        del result
        record["consumer_retained_bytes_after_discard"] = (
            retained if args.mode == "materialized" else 0
        )
        record["diagnostics_seconds"] = time.perf_counter() - diagnostic_started
        diagnostics_seconds += record["diagnostics_seconds"]
        _emit(record)
    _emit(
        dict(
            event="finished",
            **identity,
            returned_points=count,
            cumulative_generated_center_dual_bytes=generated_bytes,
            all_converged=bool(all_converged),
            peak_rss_bytes=_peak_rss(),
            baseline_peak_rss_bytes=baseline,
            consumer_retained_center_dual_bytes=retained,
            elapsed_since_path_start_seconds=time.perf_counter() - started,
            materialized_path_seconds=path_seconds,
            diagnostics_seconds=diagnostics_seconds,
            timing_scope="elapsed includes diagnostics and JSON output; materialized_path_seconds excludes both",
        )
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("stream", "materialized"), default="stream", help=argparse.SUPPRESS
    )
    parser.add_argument("--length", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--neighbors", type=int, default=10)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            _worker(args)
        except Exception as error:
            _emit(
                dict(
                    event="worker_error",
                    mode=args.mode,
                    length=args.length,
                    error_type=type(error).__name__,
                    error=str(error),
                )
            )
            traceback.print_exc(file=sys.stderr)
            return 1
        return 0
    if args.output is None:
        parser.error("--output is required")
    if args.smoke:
        args.samples, args.features = 40, 3
    if args.samples < 4 or args.features < 1 or not 1 <= args.neighbors < args.samples:
        parser.error("Require samples>=4, features>=1, and 1<=neighbors<samples")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    logs = output.with_name(output.name + ".workers")
    if output.exists() or logs.exists():
        raise FileExistsError("Use a fresh output path and worker directory")
    logs.mkdir()
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_NAMES})
    lengths = (3, 5) if args.smoke else (8, 64, 256)
    success = True
    harness_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parents[1] / "src" / "ssnalclust"
    source_sha256 = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.glob("*.py"))
    }
    with output.open("x", encoding="utf-8") as summary:
        for length in lengths:
            for mode in ("stream", "materialized"):
                identity = dict(
                    mode=mode, length=length, samples=args.samples, features=args.features
                )
                checkpoint = logs / f"{mode}-{length}.jsonl"
                error_log = logs / f"{mode}-{length}.stderr.txt"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--mode",
                    mode,
                    "--length",
                    str(length),
                    "--samples",
                    str(args.samples),
                    "--features",
                    str(args.features),
                    "--neighbors",
                    str(args.neighbors),
                ]
                summary.write(
                    json.dumps(
                        dict(
                            event="process_started",
                            harness_sha256=harness_sha256,
                            checkout_source_sha256=source_sha256,
                            **identity,
                            command=command,
                            checkpoint=str(checkpoint),
                            stderr=str(error_log),
                            timeout_seconds=120,
                            returned_point_visibility="after path returns only"
                            if mode == "materialized"
                            else "after each yielded point",
                        )
                    )
                    + "\n"
                )
                summary.flush()
                started = time.perf_counter()
                returncode = None
                print(f"Starting {mode} length={length}", flush=True)
                with (
                    checkpoint.open("x", encoding="utf-8") as out,
                    error_log.open("x", encoding="utf-8") as err,
                ):
                    try:
                        child = subprocess.run(
                            command, env=env, stdout=out, stderr=err, timeout=120
                        )
                        returncode = child.returncode
                        status = "completed" if returncode == 0 else "error"
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                    except OSError as error:
                        status = "error"
                        err.write(str(error))
                success = success and status == "completed"
                record = dict(
                    event="process_finished",
                    harness_sha256=harness_sha256,
                    checkout_source_sha256=source_sha256,
                    **identity,
                    process_status=status,
                    returncode=returncode,
                    process_wall_seconds=time.perf_counter() - started,
                    checkpoint=str(checkpoint),
                    stderr=str(error_log),
                    timeout_seconds=120,
                    returned_point_visibility="after path returns only"
                    if mode == "materialized"
                    else "after each yielded point",
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    requested_thread_environment={key: env[key] for key in THREAD_NAMES},
                )
                summary.write(json.dumps(record, allow_nan=False) + "\n")
                summary.flush()
                print(f"Finished {mode} length={length}: {status}", flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
