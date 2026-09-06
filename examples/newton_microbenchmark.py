"""Isolated Newton matvec timings with an independent small dense oracle.

Example: python examples/newton_microbenchmark.py --source /checkout/src
Pass multiple --source paths together to compare fixed checkouts sequentially.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path


def _worker(config):
    import numpy as np
    from scipy import sparse

    import ssnalclust.solvers as solvers
    from ssnalclust.graph import graph_from_weights

    source = Path(solvers.__file__).resolve().parent
    if source != Path(config["source_dir"]).resolve() / "ssnalclust":
        raise RuntimeError(f"Wrong source imported: {source}")
    rng = np.random.default_rng(config["seed"])
    p, sigma = config["features"], config["sigma"]

    # Independently assemble all dense generalized-Hessian blocks on a small
    # complete graph. This checks action, not just an output checksum.
    small_b, _ = graph_from_weights(None, 6)
    u = rng.normal(size=(6, p))
    z = rng.normal(size=(small_b.shape[0], p))
    mass = rng.uniform(0.5, 2.0, size=(6, 1))
    q = sigma * (small_b @ u) + z
    norms = np.linalg.norm(q, axis=1)
    radii = norms * np.where(np.arange(len(norms)) % 2, 0.5, 2.0)
    jacobian = np.zeros((len(radii) * p, len(radii) * p))
    for edge in range(len(radii)):
        block = np.eye(p)
        if norms[edge] > radii[edge]:
            unit = q[edge] / norms[edge]
            block = radii[edge] / norms[edge] * (np.eye(p) - np.outer(unit, unit))
        jacobian[edge * p : (edge + 1) * p, edge * p : (edge + 1) * p] = block
    incidence = np.kron(small_b.toarray(), np.eye(p))
    expected = np.diag(np.repeat(mass[:, 0], p)) + sigma * incidence.T @ jacobian @ incidence
    operator, preconditioner = solvers._newton_operator(u, z, small_b, radii, sigma, mass)
    direction = rng.normal(size=u.size)
    actual = operator @ direction
    np.testing.assert_allclose(actual, expected @ direction, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(
        preconditioner @ direction, direction / np.diag(expected), rtol=2e-13, atol=2e-13
    )
    oracle_error = float(np.max(np.abs(actual - expected @ direction)))

    # Fixed circulant graph: each node connects to offsets 1,...,10 in both
    # directions, giving exactly 10*n undirected edges when n>20.
    n = config["samples"]
    graph_start = time.perf_counter()
    i = np.repeat(np.arange(n), 10)
    j = (i + np.tile(np.arange(1, 11), n)) % n
    rows, cols = np.concatenate((i, j)), np.concatenate((j, i))
    weights = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    b, _ = graph_from_weights(weights, n)
    graph_seconds = time.perf_counter() - graph_start
    u = rng.normal(size=(n, p))
    z = rng.normal(size=(b.shape[0], p))
    mass = rng.uniform(0.5, 2.0, size=(n, 1))
    q = sigma * (b @ u) + z
    radii = np.linalg.norm(q, axis=1) * np.where(np.arange(len(z)) % 2, 0.5, 2.0)
    direction = rng.normal(size=u.size)
    start = time.perf_counter()
    operator, _ = solvers._newton_operator(u, z, b, radii, sigma, mass)
    construction_seconds = time.perf_counter() - start
    for _ in range(10):
        output = operator @ direction
    batches = []
    for _ in range(config["rounds"]):
        start = time.perf_counter()
        for _ in range(config["calls_per_round"]):
            output = operator @ direction
        batches.append(time.perf_counter() - start)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        kind="matvec_microbenchmark",
        status="completed",
        config=config,
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        },
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(),
        platform=platform.platform(),
        versions={name: importlib.metadata.version(name) for name in ["numpy", "scipy"]},
        edges=b.shape[0],
        graph="circulant_offsets_1_to_10_union",
        mass="uniform[.5,2]",
        radii="alternating 2*||q|| and .5*||q||",
        outside_fraction=0.5,
        graph_seconds=graph_seconds,
        operator_construction_seconds=construction_seconds,
        batch_seconds=batches,
        median_seconds_per_matvec=statistics.median(batches) / config["calls_per_round"],
        timed_matvec_calls=config["rounds"] * config["calls_per_round"],
        warmup_calls=10,
        output_norm=float(np.linalg.norm(output)),
        output_sum=float(np.sum(output)),
        dense_oracle_max_absolute_error=oracle_error,
        peak_rss_bytes=int(peak if sys.platform == "darwin" else peak * 1024),
        requested_threads=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--features", type=int, nargs="+", default=[3, 20])
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=317)
    parser.add_argument("--calls-per-round", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            result = _worker(json.loads(args.worker))
        except Exception as exc:
            result = dict(
                kind="matvec_microbenchmark",
                status="error",
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        print(json.dumps(result, allow_nan=False), flush=True)
        return
    if (
        not args.source
        or args.samples <= 20
        or min(args.features) < 1
        or args.rounds < 1
        or args.calls_per_round < 1
        or args.timeout <= 0
        or args.sigma <= 0
    ):
        parser.error(
            "source required; samples>20 and positive features/rounds/calls/timeout/sigma required"
        )
    for source in args.source:
        for features in args.features:
            config = dict(
                source_dir=str(source.resolve()),
                samples=args.samples,
                features=features,
                sigma=args.sigma,
                seed=args.seed,
                calls_per_round=args.calls_per_round,
                rounds=args.rounds,
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(source.resolve())
            for name in [
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            ]:
                environment[name] = "1"
            print(
                f"Starting operator n={args.samples}, p={features}, source={source}",
                file=sys.stderr,
                flush=True,
            )
            start = time.perf_counter()
            try:
                worker = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(config)],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    env=environment,
                    check=False,
                )
                result = (
                    json.loads(worker.stdout)
                    if worker.stdout
                    else dict(kind="matvec_microbenchmark", status="error", stderr=worker.stderr)
                )
            except subprocess.TimeoutExpired:
                result = dict(kind="matvec_microbenchmark", status="timeout")
            result.update(
                config=config,
                timeout_seconds=args.timeout,
                process_seconds=time.perf_counter() - start,
            )
            encoded = json.dumps(result, allow_nan=False)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a") as output:
                    output.write(encoded + "\n")
            else:
                print(encoded, flush=True)
            print(
                f"Finished {result['status']}: {result.get('median_seconds_per_matvec')} s/matvec",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
