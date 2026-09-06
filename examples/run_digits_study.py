"""Run separate, bounded digits-study processes and retain their evidence.

Each graph receives a fresh process with a 900-second deadline and one requested
native thread. Use --smoke for the offline first-80-row CI/exploratory protocol.
The output directory must not already exist, preserving previous checkpoints.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _revision(root):
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _save_records(path, records):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--data", type=Path, help="Local original UCI optdigits.tes; never downloaded"
    )
    parser.add_argument("--smoke", action="store_true", help="First 80 rows and grid indices 0,4,8")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    data = args.data.resolve() if args.data is not None else root / "examples/data/optdigits.tes"
    env = os.environ.copy()
    env.update({key: "1" for key in THREAD_NAMES})
    revision = _revision(root)
    records = []
    for k in (10, 20):
        command = [
            sys.executable,
            str(root / "examples/digits_study.py"),
            "--data",
            str(data),
            "--neighbors",
            str(k),
            "--output",
            str(output / f"digits_k{k}.json"),
            "--arrays",
            str(output / f"digits_k{k}.npz"),
        ]
        if args.smoke:
            command.append("--smoke")
        log_path = output / f"digits_k{k}.log"
        record = dict(
            neighbors=k,
            process_status="running",
            returncode=None,
            timeout_seconds=900,
            requested_thread_environment={key: env[key] for key in THREAD_NAMES},
            command=command,
            git_revision=revision,
            smoke=args.smoke,
            log=str(log_path),
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            harness_sha256=hashlib.sha256(
                (root / "examples/digits_study.py").read_bytes()
            ).hexdigest(),
        )
        records.append(record)
        _save_records(output / "digits_processes.json", records)
        started = time.perf_counter()
        print(f"Starting graph k={k}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            try:
                child = subprocess.run(
                    command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=900
                )
                record["process_status"] = "completed" if child.returncode == 0 else "error"
                record["returncode"] = child.returncode
            except subprocess.TimeoutExpired:
                record["process_status"] = "timeout"
            except OSError as error:
                record["process_status"] = "error"
                record["error"] = str(error)
        record["process_wall_seconds"] = time.perf_counter() - started
        for suffix in ("json", "npz"):
            path = output / f"digits_k{k}.{suffix}"
            record[f"{suffix}_exists"] = path.exists()
            if path.exists():
                record[f"{suffix}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        _save_records(output / "digits_processes.json", records)
        print(f"Finished graph k={k}: {record['process_status']}", flush=True)
    return 0 if all(record["process_status"] == "completed" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
