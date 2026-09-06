"""Validate installed wheel and optional sdist artifacts in fresh environments.

Usage: python scripts/check_distribution.py dist/package.whl \
           --sdist dist/package.tar.gz --report artifacts/distribution.json

Requires only the standard library in the invoking interpreter. Runtime and
isolated build dependencies are downloaded by pip; no development extras are
installed in either smoke-test environment. No package is published.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import venv
import zipfile
from datetime import datetime, timezone
from email.parser import BytesParser
from pathlib import Path

SMOKE = r"""
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import site
import sys

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import ssnalclust as sc

expected_version, expected_prefix, checkout = sys.argv[1:]
module = Path(sc.__file__).resolve()
prefix = Path(expected_prefix).resolve()
assert Path(sys.prefix).resolve() == prefix
assert any(module.is_relative_to(Path(folder).resolve()) for folder in site.getsitepackages())
assert module.is_relative_to(prefix), (module, prefix)
assert not module.is_relative_to(Path(checkout).resolve()), module
assert sc.__version__ == metadata.version("ssnalclust") == expected_version
assert all(hasattr(sc, name) for name in sc.__all__)
for optional in ["cvxpy", "pytest", "pandas", "matplotlib", "build"]:
    assert importlib.util.find_spec(optional) is None, f"Unexpected optional dependency: {optional}"

checks = []
def certified(name, result, shape, tol=1e-7):
    assert result.converged, (name, result)
    assert result.centers.shape == shape, name
    assert np.isfinite(result.centers).all(), name
    assert np.isfinite(result.objective), name
    assert result.kkt_residual <= tol, (name, result.kkt_residual)
    assert result.relative_gap <= tol, (name, result.relative_gap)
    checks.append(dict(name=name, objective=float(result.objective),
                       relative_gap=float(result.relative_gap),
                       kkt_residual=float(result.kkt_residual)))

X = np.array([[0., 0.], [3., 4.]])
W = np.array([[0., 1.], [1., 0.]])
expected_l2 = np.array([[.6, .8], [2.4, 3.2]])
for method in ["ssnal", "admm", "ama", "fama"]:
    result = sc.solve(X, W, gamma=1., solver=method, tol=1e-7, max_iter=10000)
    certified("core_" + method, result, X.shape)
    assert_allclose(result.centers, expected_l2, atol=2e-6)
    assert_allclose(result.objective, 4., atol=2e-7)
l1 = sc.solve(X, W, gamma=1., solver="admm", penalty="l1", tol=1e-7, max_iter=10000)
certified("core_l1", l1, X.shape)
assert_allclose(l1.centers, [[1., 1.], [2., 3.]], atol=2e-6)
assert_allclose(l1.objective, 5., atol=2e-7)

problem = sc.ConvexClusteringProblem(X, weights=W)
assert (problem.n_samples, problem.n_features, problem.n_edges) == (2, 2, 1)
prepared = problem.solve(gamma=1., tol=1e-7)
certified("prepared", prepared, X.shape)
for label, path in [
    ("prepared_stream", list(problem.iter_path([0., 1., 10.], solver="admm", tol=1e-7,
                                               max_iter=10000, store_history=False))),
    ("public_stream", list(sc.iter_convex_clustering_path(X, [0., 1., 10.], weights=W,
                                                         tol=1e-7, store_history=False))),
    ("public_path", sc.convex_clustering_path(X, [0., 1., 10.], weights=W, tol=1e-7,
                                             store_history=False)),
]:
    assert len(path) == 3
    assert_array_equal(path[0].centers, X)
    assert_allclose(path[-1].centers, np.tile(X.mean(axis=0), (2, 1)), atol=2e-5)
    assert [point["n_clusters"] for point in sc.summarize_path(path)] == [2, 2, 1]
    for index, point in enumerate(path):
        certified(f"{label}_{index}", point, X.shape)
        assert point.history == []
summary_clusters = []
upstream = problem.iter_path([0., 1., 10.], tol=1e-7, store_history=False)
summaries = sc.iter_path_summaries(upstream)
try:
    for summary in summaries:
        assert summary["converged"]
        assert summary["relative_gap"] <= 1e-7
        assert summary["kkt_residual"] <= 1e-7
        assert np.isfinite(summary["dual_objective"])
        assert summary["gap"] >= 0.
        assert summary["center_error_bound"] >= 0.
        summary_clusters.append(summary["n_clusters"])
        checks.append(dict(name=f"streamed_summary_{len(summary_clusters) - 1}",
                           objective=float(summary["objective"]),
                           relative_gap=float(summary["relative_gap"]),
                           kkt_residual=float(summary["kkt_residual"])))
        del summary
finally:
    summaries.close()
    upstream.close()
assert summary_clusters == [2, 2, 1]

model = sc.ConvexClustering(weights=W, gamma=10., tol=1e-7).fit(X)
assert_array_equal(model.labels_, [0, 0])
assert model.n_clusters_ == 1
assert_allclose(model.cluster_centers_, [X.mean(axis=0)], atol=2e-5)

missing = sc.solve_missing([[2., np.nan], [np.nan, 4.]], W, tol=1e-7)
certified("missing", missing, (2, 2))
assert_allclose(missing.centers, [[2., 4.], [2., 4.]])
assert_allclose(missing.objective, 0.)
assert missing.certificate_model == "observed_range_box"

huber = sc.solve_generalized(X, W, gamma=1., loss="huber", huber_delta=10., tol=1e-7)
certified("huber", huber, X.shape)
assert_allclose(huber.centers, expected_l2, atol=2e-5)
assert_allclose(huber.objective, 4., atol=2e-6)
for loss, value, mean, objective in [
    ("logistic", .5, .5, 4 * np.log(2)),
    ("poisson", 2., 2., 4 * (2 - 2 * np.log(2))),
]:
    result = sc.solve_generalized(np.full((2, 2), value), W, loss=loss, tol=1e-7)
    certified(loss, result, (2, 2))
    assert_allclose(result.fitted_means, mean, atol=2e-6)
    assert_allclose(result.objective, objective, atol=2e-6)

sparse_result = sc.solve_sparse(X, W, gamma=0., alpha=1., tol=1e-7)
certified("feature_sparse", sparse_result, X.shape)
centered = X - X.mean(axis=0)
expected_sparse = centered * np.maximum(0., 1 - 1 / np.linalg.norm(centered, axis=0))
assert_allclose(sparse_result.centers, expected_sparse, atol=2e-5)
assert_allclose(sparse_result.offset, X.mean(axis=0))
expected_objective = .5 * np.sum((expected_sparse - centered)**2) + np.linalg.norm(expected_sparse, axis=0).sum()
assert_allclose(sparse_result.objective, expected_objective, atol=2e-6)
bicluster = sc.solve_biclustering(X, row_weights=W, gamma_row=1., gamma_col=0., tol=1e-7)
certified("biclustering", bicluster, X.shape)
assert_allclose(bicluster.centers, expected_l2, atol=2e-5)
assert_allclose(bicluster.objective, 4., atol=2e-6)

print(json.dumps(dict(status="passed", module=str(module), prefix=str(prefix),
                     python=sys.version.split()[0], version=metadata.version("ssnalclust"),
                     runtime_versions={name: metadata.version(name)
                                       for name in ["numpy", "scipy", "scikit-learn"]},
                     optional_dependencies_absent=True, checks=checks), allow_nan=False))
"""


def _run(command, cwd, timeout=600):
    environment = os.environ.copy()
    for name in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"]:
        environment.pop(name, None)
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    environment["PIP_NO_INPUT"] = "1"
    process = subprocess.run(
        command, cwd=cwd, env=environment, text=True, capture_output=True, timeout=timeout
    )
    if process.returncode:
        raise RuntimeError(
            f"Command failed ({process.returncode}): {command}\n"
            f"{process.stdout[-10000:]}\n{process.stderr[-10000:]}"
        )
    return process.stdout


def _python(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _wheel_metadata(wheel):
    with zipfile.ZipFile(wheel) as archive:
        candidates = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(candidates) != 1:
            raise ValueError("Wheel must contain exactly one METADATA file")
        metadata = BytesParser().parsebytes(archive.read(candidates[0]))
    if metadata["Name"] != "ssnalclust":
        raise ValueError("Expected an ssnalclust wheel")
    if not metadata["Version"]:
        raise ValueError("Wheel metadata has no version")
    return {
        "name": metadata["Name"],
        "version": metadata["Version"],
        "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist", []),
    }


def _inspect_sdist(sdist):
    with tarfile.open(sdist, "r:gz") as archive:
        members = {member.name for member in archive.getmembers() if member.isfile()}
    for required in ["pyproject.toml", "LICENSE", "src/ssnalclust/__init__.py"]:
        if not any(name.endswith("/" + required) for name in members):
            raise ValueError(f"Source distribution is missing {required}")
    return {"file_count": len(members), "sha256": hashlib.sha256(sdist.read_bytes()).hexdigest()}


def _check_wheel(wheel, folder, checkout):
    metadata = _wheel_metadata(wheel)
    environment = folder / "runtime-env"
    unrelated = folder / "unrelated-working-directory"
    unrelated.mkdir(parents=True)
    print(f"Installing {wheel.name} in a fresh runtime environment", flush=True)
    started = time.perf_counter()
    venv.EnvBuilder(with_pip=True, system_site_packages=False, symlinks=os.name != "nt").create(
        environment
    )
    python = _python(environment)
    _run([str(python), "-I", "-m", "pip", "install", str(wheel)], unrelated)
    _run([str(python), "-I", "-m", "pip", "check"], unrelated)
    output = _run(
        [str(python), "-I", "-c", SMOKE, metadata["version"], str(environment), str(checkout)],
        unrelated,
    )
    smoke = json.loads(output.strip().splitlines()[-1])
    print(f"Passed {len(smoke['checks'])} numerical checks for {wheel.name}", flush=True)
    return {
        "wheel": str(wheel),
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "metadata": metadata,
        "seconds": time.perf_counter() - started,
        "smoke": smoke,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Wheel artifact to install and validate")
    parser.add_argument(
        "--sdist", type=Path, help="Also build and validate this .tar.gz source artifact"
    )
    parser.add_argument("--report", type=Path, help="Write structured results here")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve(strict=True)
    sdist = args.sdist.resolve(strict=True) if args.sdist else None
    checkout = Path(__file__).resolve().parents[1]
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "invoking_python": sys.version.split()[0],
        "artifacts": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="ssnalclust-distribution-") as temporary:
            workspace = Path(temporary)
            report["artifacts"].append(_check_wheel(wheel, workspace / "wheel", checkout))
            if sdist is not None:
                report["sdist"] = {"path": str(sdist), **_inspect_sdist(sdist)}
                print(
                    f"Building a wheel from {sdist.name} with isolated build dependencies",
                    flush=True,
                )
                builder = workspace / "build-env"
                venv.EnvBuilder(
                    with_pip=True, system_site_packages=False, symlinks=os.name != "nt"
                ).create(builder)
                destination = workspace / "built-wheels"
                destination.mkdir()
                _run(
                    [
                        str(_python(builder)),
                        "-I",
                        "-m",
                        "pip",
                        "wheel",
                        "--no-deps",
                        "--wheel-dir",
                        str(destination),
                        str(sdist),
                    ],
                    workspace,
                )
                rebuilt = list(destination.glob("*.whl"))
                if len(rebuilt) != 1:
                    raise ValueError("Source build did not produce exactly one wheel")
                if _wheel_metadata(rebuilt[0])["version"] != _wheel_metadata(wheel)["version"]:
                    raise ValueError("Wheel and source-distribution versions disagree")
                report["artifacts"].append(_check_wheel(rebuilt[0], workspace / "sdist", checkout))
                report["sdist"]["isolated_build"] = "passed"
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Distribution validation passed; temporary environments removed", flush=True)
    return report


if __name__ == "__main__":
    main()
