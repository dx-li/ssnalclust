"""Compare unmodified R cvxclustr fits using independently recomputed certificates.

R and cvxclustr are optional external reference dependencies, never package
runtime dependencies. See docs/external_reference.md for installation/protocol.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np


def certify_candidate(X, weights, gamma, penalty, centers, dual):
    """Evaluate unit-mass quadratic certificates independently of solver helpers.

    Dual rows follow lexicographic positive-weight (i,j), i<j, with incidence
    +1 at i and -1 at j. Only the reference package's l1/l2 norms are supported.
    Materially infeasible multipliers cannot furnish a finite dual certificate.
    """
    x, w, u, z = [np.asarray(a, dtype=float) for a in (X, weights, centers, dual)]
    if penalty not in ("l1", "l2"):
        raise ValueError("reference certificate penalty must be l1 or l2")
    if x.ndim != 2 or not all(x.shape) or u.shape != x.shape:
        raise ValueError("centers and X must have the same nonempty two-dimensional shape")
    n = len(x)
    if w.shape != (n, n) or not np.array_equal(w, w.T) or np.any(w < 0):
        raise ValueError("weights must be a nonnegative symmetric adjacency")
    if np.any(np.diag(w) != 0) or not np.isfinite(gamma) or gamma < 0:
        raise ValueError("invalid adjacency diagonal or gamma")
    i, j = np.where(np.triu(w, 1) > 0)
    if z.shape != (len(i), x.shape[1]):
        raise ValueError("dual shape does not match positive graph edges")
    if not all(np.isfinite(a).all() for a in (x, w, u, z)):
        raise ValueError("certificate inputs must be finite")
    radii = gamma * w[i, j]
    if not np.isfinite(radii).all():
        raise ValueError("certificate radii overflow")
    norms = np.max(np.abs(z), axis=1) if penalty == "l1" else np.hypot.reduce(z, axis=1)
    violation = float(np.maximum(norms - radii, 0).max(initial=0))
    eps = 64 * np.finfo(float).eps
    if np.any(norms > radii + eps * radii):
        raise ValueError("dual is outside its feasible norm balls")
    adjoint = np.zeros_like(x)
    np.add.at(adjoint, i, z)
    np.add.at(adjoint, j, -z)
    differences = u[i] - u[j]
    order = 1 if penalty == "l1" else 2
    penalties = radii * np.linalg.norm(differences, ord=order, axis=1)
    pairings = np.einsum("ij,ij->i", z, differences)
    slacks = penalties - pairings
    if np.any(slacks < -eps * (penalties + np.abs(pairings))):
        raise ValueError("negative dual Fenchel slack exceeds roundoff")
    residual = u - x
    objective = float(0.5 * np.sum(residual**2) + penalties.sum())
    dual_objective = float(np.sum(z * (x[i] - x[j])) - 0.5 * np.sum(adjoint**2))
    gap = float(0.5 * np.sum((residual + adjoint) ** 2) + np.maximum(slacks, 0).sum())
    trial = differences + z
    if penalty == "l1":
        projected = np.clip(trial, -radii[:, None], radii[:, None])
    else:
        lengths = np.linalg.norm(trial, axis=1)
        scale = np.ones_like(lengths)
        np.divide(radii, lengths, out=scale, where=lengths > radii)
        projected = trial * scale[:, None]
    inclusion = projected - z
    stationarity = np.linalg.norm(residual + adjoint) / (
        1 + np.linalg.norm(residual) + np.linalg.norm(adjoint)
    )
    subgradient = np.linalg.norm(inclusion) / (1 + np.linalg.norm(differences) + np.linalg.norm(z))
    if not np.isfinite([objective, dual_objective, gap, stationarity, subgradient]).all():
        raise ValueError("certificate diagnostics overflow; rescale inputs")
    return dict(
        objective=objective,
        dual_objective=dual_objective,
        gap=gap,
        relative_gap=gap / (1 + abs(objective) + abs(dual_objective)),
        kkt_residual=float(max(stationarity, subgradient)),
        dual_violation=violation,
        center_error_bound=float(np.sqrt(2 * gap)),
    )


def reference_fit(x, w, gamma, penalty, *, accelerate, rscript, r_library, tol, max_iter, timeout):
    """Fresh R process; timeout and native stopping data remain explicit."""
    with tempfile.TemporaryDirectory(prefix="ssnalclust-reference-") as folder:
        root = Path(folder)
        np.savetxt(root / "X.csv", x, delimiter=",", fmt="%.17g")
        np.savetxt(root / "W.csv", w, delimiter=",", fmt="%.17g")
        env = os.environ.copy()
        if r_library:
            env["R_LIBS"] = r_library
        for key in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[key] = "1"
        command = [
            rscript,
            str(Path(__file__).with_suffix(".R")),
            folder,
            folder,
            str(gamma),
            "1" if penalty == "l1" else "2",
            str(accelerate).upper(),
            str(tol),
            str(max_iter),
        ]
        started = time.perf_counter()
        run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
        if run.returncode:
            raise RuntimeError(run.stderr[-4000:])
        with (root / "metadata.csv").open() as stream:
            metadata = {row["name"]: row["value"] for row in csv.DictReader(stream)}
        edges = np.loadtxt(root / "edges.csv", delimiter=",", ndmin=2).astype(int) - 1
        expected = np.column_stack(np.where(np.triu(w, 1) > 0))
        if not np.array_equal(edges, expected):
            raise ValueError("reference edge ordering differs from common certificate ordering")
        u = np.loadtxt(root / "centers.csv", delimiter=",", ndmin=2)
        # cvxclustr uses the opposite Lagrange sign to +B.T@Z stationarity.
        z = -np.loadtxt(root / "lambda.csv", delimiter=",", ndmin=2)
        metadata.update(
            process_seconds=time.perf_counter() - started,
            stderr=run.stderr,
            seconds=float(metadata["seconds"]),
            iterations=int(metadata["iterations"]),
            native_primal=float(metadata["native_primal"]),
            native_dual=float(metadata["native_dual"]),
        )
        return u, z, metadata


def cases(smoke, stress=False):
    """Fixed datasets/graphs; truth labels never enter fitting or parameter choice."""
    if stress:
        x = np.array([[0.0, 0.1], [0.2, -0.3], [0.4, 0.8], [1.2, -0.7], [2.0, 0.4], [3.0, 1.5]])
        w = np.zeros((6, 6))
        for i, weight in enumerate([2.0, 4.0, 3.0, 5.0, 2.0]):
            w[i, i + 1] = w[i + 1, i] = weight
        yield "weighted_stop_stress", x, w
        return
    yield "two_point", np.array([[0.0, 1.0], [3.0, -1.0]]), np.array([[0.0, 0.7], [0.7, 0.0]])
    if smoke:
        return
    rng = np.random.default_rng(20260905)
    x = rng.normal(size=(12, 4))
    w = rng.uniform(0.2, 1.5, size=(12, 12))
    w = np.triu(w, 1)
    w += w.T
    yield "random_complete", x, w
    sparse = np.zeros_like(w)
    for a, b in [(k, k + 1) for k in range(5)] + [(k, k + 1) for k in range(6, 11)]:
        sparse[a, b] = sparse[b, a] = w[a, b]
    yield "random_disconnected", x, sparse
    from sklearn.datasets import load_iris

    from ssnalclust import k_neighbors_graph

    iris = load_iris().data
    iris = (iris - iris.mean(axis=0)) / iris.std(axis=0, ddof=0)
    graph = k_neighbors_graph(iris, n_neighbors=10, bandwidth=1.0).toarray()
    yield "iris_standardized_knn10", iris, graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--r-library", default="")
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--stress", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.reference_source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(args.reference_source), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        parser.error("reference source checkout must be clean")
    reference_hashes = {
        str(p.relative_to(args.reference_source)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("R", "src")
        for p in (args.reference_source / folder).rglob("*")
        if p.is_file() and p.suffix.lower() in ("r", ".r", ".c", ".h")
    }
    from threadpoolctl import threadpool_limits

    import ssnalclust
    from ssnalclust import solve

    with threadpool_limits(limits=1):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as stream:

            def emit(record):
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()

            emit(
                dict(
                    kind="metadata",
                    reference_revision=revision,
                    reference_source_sha256=reference_hashes,
                    harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    bridge_sha256=hashlib.sha256(
                        Path(__file__).with_suffix(".R").read_bytes()
                    ).hexdigest(),
                    mode="stress" if args.stress else "smoke" if args.smoke else "full",
                    python=platform.python_version(),
                    platform=platform.platform(),
                    versions={
                        name: importlib.metadata.version(name)
                        for name in ["numpy", "scipy", "scikit-learn", "ssnalclust"]
                    },
                    source_sha256={
                        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in Path(ssnalclust.__file__).parent.glob("*.py")
                    },
                    reference_tol=1e-10,
                    reference_max_iter=100000,
                    local_tol=1e-8,
                    gamma_grid=[0.1] if args.smoke else [0.01, 0.1, 1.0, 10.0],
                    thread_limit=1,
                )
            )
            for name, x, w in cases(args.smoke, args.stress):
                emit(dict(kind="problem", case=name, X=x.tolist(), weights=w.tolist()))
                for penalty in ("l1", "l2"):
                    for gamma in [0.1] if args.smoke else [0.01, 0.1, 1.0, 10.0]:
                        common = dict(
                            case=name,
                            n=len(x),
                            p=x.shape[1],
                            penalty=penalty,
                            gamma=gamma,
                            data_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                            graph_sha256=hashlib.sha256(w.tobytes()).hexdigest(),
                        )
                        local_solver = "ssnal" if penalty == "l2" else "fama"
                        start = time.perf_counter()
                        fit = solve(
                            x,
                            w,
                            gamma=gamma,
                            penalty=penalty,
                            solver=local_solver,
                            tol=1e-8,
                            max_iter=100000,
                            store_history=False,
                        )
                        seconds = time.perf_counter() - start
                        local = certify_candidate(x, w, gamma, penalty, fit.centers, fit.dual)
                        emit(
                            dict(
                                kind="local",
                                **common,
                                solver=local_solver,
                                seconds=seconds,
                                n_iter=fit.n_iter,
                                converged=fit.converged,
                                certificate=local,
                                centers=fit.centers.tolist(),
                                dual=fit.dual.tolist(),
                            )
                        )
                        for accelerate in (False, True):
                            engine = "cvxclustr_fama" if accelerate else "cvxclustr_ama"
                            try:
                                u, z, meta = reference_fit(
                                    x,
                                    w,
                                    gamma,
                                    penalty,
                                    accelerate=accelerate,
                                    rscript=args.rscript,
                                    r_library=args.r_library,
                                    tol=1e-10,
                                    max_iter=100000,
                                    timeout=args.timeout,
                                )
                                cert = certify_candidate(x, w, gamma, penalty, u, z)
                                distance = float(np.linalg.norm(u - fit.centers))
                                emit(
                                    dict(
                                        kind="reference",
                                        status="completed",
                                        **common,
                                        solver=engine,
                                        metadata=meta,
                                        certificate=cert,
                                        centers=u.tolist(),
                                        dual=z.tolist(),
                                        common_accuracy_pass=max(
                                            cert["relative_gap"], cert["kkt_residual"]
                                        )
                                        <= 1e-8,
                                        center_distance=distance,
                                        combined_center_bound=local["center_error_bound"]
                                        + cert["center_error_bound"],
                                    )
                                )
                            except subprocess.TimeoutExpired:
                                emit(
                                    dict(
                                        kind="reference",
                                        status="timeout",
                                        **common,
                                        solver=engine,
                                        timeout=args.timeout,
                                    )
                                )
                            except (RuntimeError, ValueError) as exc:
                                emit(
                                    dict(
                                        kind="reference",
                                        status="error",
                                        **common,
                                        solver=engine,
                                        error=str(exc),
                                    )
                                )
                        print(f"Finished {name} {penalty} gamma={gamma}", flush=True)


if __name__ == "__main__":
    main()
