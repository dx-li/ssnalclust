"""Plot measured native peak RSS and retained arrays from streaming checkpoints.

Usage: python examples/plot_streaming_memory.py docs/streaming_results.jsonl
No solver is imported or run. A failed/incomplete process is annotated rather
than treated as a zero-memory result. Checkpoint hashes are verified.
"""

import argparse
import hashlib
import json
from pathlib import Path


def read_results(path):
    """Read process summaries and their hash-verified scalar checkpoint events."""
    rows = []
    pending = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        process = json.loads(line)
        if process.get("event") == "process_started":
            identity = (process["mode"], process["length"])
            if identity in pending:
                raise ValueError(f"Repeated process start without completion: {identity}")
            pending.add(identity)
            continue
        if process.get("event") != "process_finished":
            continue
        pending.discard((process["mode"], process["length"]))
        recorded = Path(process["checkpoint"])
        candidates = [
            path.parent / (path.name + ".workers") / recorded.name,
            path.parent / recorded.name,
            path.parent / recorded,
            recorded,
        ]
        checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError(f"Checkpoint unavailable: {recorded}")
        raw = checkpoint.read_bytes()
        if hashlib.sha256(raw).hexdigest() != process["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
        events = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        process = dict(process, checkpoint_events=events)
        prepared = [event for event in events if event.get("event") == "prepared"]
        finished = [event for event in events if event.get("event") == "finished"]
        if len(prepared) > 1 or len(finished) > 1:
            raise ValueError(f"Ambiguous repeated worker events: {checkpoint}")
        rows.append((process, prepared[0] if prepared else {}, finished[0] if finished else {}))
    if pending:
        raise ValueError(f"Started processes have no finished record: {sorted(pending)}")
    if not rows:
        raise ValueError("No process_finished records to plot")
    identities = [(row[0]["mode"], row[0]["length"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Multiple runs per mode/length require explicit replicate handling")
    return rows


def validate_pairs(rows):
    """Independently audit complete numerical parity and retained-array accounting.

    Exact scalar equality is intentional: these are the same installed solver
    on identical inputs, in fresh single-thread workers on the same host.
    A timeout cannot establish full-path parity, even with a matching prefix.
    """
    errors = []
    pairs = []
    by_length = {}
    for process, prepared, finished in rows:
        length, mode = process["length"], process["mode"]
        bucket = by_length.setdefault(length, {})
        if mode in bucket:
            errors.append(f"L={length}: duplicate {mode}")
        points = [e for e in process["checkpoint_events"] if e.get("event") == "point"]
        bucket[mode] = (prepared, finished, points)
        prefix = f"{mode} L={length}"
        if process["process_status"] != "completed" or finished.get("returned_points") != length:
            errors.append(f"{prefix}: incomplete process cannot establish full-path parity")
        if [p["index"] for p in points] != list(range(length)):
            errors.append(f"{prefix}: missing, duplicate or out-of-order points")
        if finished.get("all_converged") != all(p["converged"] for p in points):
            errors.append(f"{prefix}: all_converged disagrees with point records")
        generated = sum(p["center_array_bytes"] + p["dual_array_bytes"] for p in points)
        expected_retained = generated if mode == "materialized" else 0
        if finished.get("consumer_retained_center_dual_bytes") != expected_retained:
            errors.append(f"{prefix}: retained-array accounting mismatch")
        if finished.get("cumulative_generated_center_dual_bytes") != generated:
            errors.append(f"{prefix}: generated-array accounting mismatch")
    metadata = (
        "data_sha256",
        "graph_sha256",
        "gammas",
        "source_sha256",
        "harness_sha256",
        "samples",
        "features",
        "seed",
        "neighbors",
        "solver",
        "tol",
        "max_iter",
        "store_history",
        "requested_thread_environment",
        "versions",
        "platform",
    )
    numerical = (
        "index",
        "gamma",
        "converged",
        "n_iter",
        "objective",
        "dual_objective",
        "gap",
        "relative_gap",
        "kkt_residual",
        "center_error_bound",
        "centers_sha256",
        "dual_sha256",
        "displacement_frobenius",
        "relative_displacement",
        "direct_fused_edge_fraction",
        "center_array_bytes",
        "dual_array_bytes",
    )
    if rows:
        baseline = rows[0][1]
        for process, prepared, _ in rows[1:]:
            changed = [
                field
                for field in metadata
                if field != "gammas"
                and (
                    field not in baseline
                    or field not in prepared
                    or baseline[field] != prepared[field]
                )
            ]
            if changed:
                errors.append(
                    f"{process['mode']} L={process['length']}: scaling configuration differs: "
                    + ", ".join(changed)
                )
    for length, modes in sorted(by_length.items()):
        if set(modes) != {"stream", "materialized"}:
            errors.append(f"L={length}: missing mode pair")
            continue
        left, right = modes["stream"], modes["materialized"]
        mismatches = [
            field
            for field in metadata
            if field not in left[0] or field not in right[0] or left[0][field] != right[0][field]
        ]
        for a, b in zip(left[2], right[2]):
            mismatches.extend(
                f"point {a['index']} {field}"
                for field in numerical
                if field not in a or field not in b or a[field] != b[field]
            )
        if mismatches:
            errors.append(f"L={length}: mismatched " + ", ".join(mismatches))
        pairs.append(
            dict(
                length=length,
                compared_points=min(len(left[2]), len(right[2])),
                mismatches=mismatches,
            )
        )
    if not rows:
        errors.append("No runs supplied")
    return dict(
        passed=not errors,
        pairs=pairs,
        errors=errors,
        scope="Exact full-path numerical parity and consumer returned-array accounting",
    )


def make_figure(rows):
    """Return two panels, with native high-water and consumer-array scopes separate."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), layout="constrained", sharex=True)
    notices = []
    sources = set()
    for mode, color in (("stream", "#2369a1"), ("materialized", "#c66b25")):
        selected = sorted(
            (row for row in rows if row[0]["mode"] == mode), key=lambda row: row[0]["length"]
        )
        x, rss, baseline, retained = [], [], [], []
        for process, prepared, finished in selected:
            length = process["length"]
            x.append(length)
            if prepared:
                sources.add(
                    hashlib.sha256(
                        json.dumps(prepared["source_sha256"], sort_keys=True).encode()
                    ).hexdigest()[:12]
                )
            complete = (
                process["process_status"] == "completed"
                and finished.get("returned_points") == length
            )
            if not complete:
                notices.append(
                    f"{mode} L={length}: {process['process_status']}; final measurement unavailable"
                )
            elif not finished.get("all_converged", False):
                notices.append(f"{mode} L={length}: completed, some points unconverged")
            if complete and finished.get("peak_rss_bytes") is None:
                notices.append(f"{mode} L={length}: native peak RSS unavailable")
            rss.append(finished.get("peak_rss_bytes") if complete else None)
            baseline.append(prepared.get("baseline_peak_rss_bytes"))
            retained.append(
                finished.get("consumer_retained_center_dual_bytes") if complete else None
            )

        def convert(values):
            return np.asarray([np.nan if value is None else value / 2**20 for value in values])

        axes[0].plot(x, convert(rss), "o-", color=color, label=f"{mode}: final peak")
        axes[0].plot(
            x, convert(baseline), "x:", color=color, alpha=0.65, label=f"{mode}: preparation peak"
        )
        axes[1].plot(x, convert(retained), "o-", color=color, label=mode)
    axes[0].set(title="Native process peak RSS (not current memory)", ylabel="MiB")
    axes[1].set(title="Consumer-retained centers and duals at completion", ylabel="MiB")
    lengths = sorted({row[0]["length"] for row in rows})
    for axis in axes:
        axis.set(xlabel="Path length", xticks=lengths)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    configurations = sorted({(row[0]["samples"], row[0]["features"]) for row in rows})
    fig.suptitle(f"Streaming versus retained paths | (samples, features): {configurations}")
    footer = "RSS includes imports, temporary arrays and allocator retention. Right panel excludes solver/iterator storage."
    footer += "\nSource fingerprints: " + ", ".join(sorted(sources))
    if len(sources) > 1:
        footer += " (different sources: not a controlled comparison)"
    if notices:
        footer += "\n" + "\n".join(notices)
    fig.supxlabel(footer, fontsize=8)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, default=Path("docs/_static/streaming_memory.png"))
    parser.add_argument("--audit", type=Path, help="Write independent parity/accounting audit JSON")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = read_results(args.report)
    audit = validate_pairs(rows)
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    figure = make_figure(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)
    if not audit["passed"]:
        raise SystemExit("Independent parity audit failed: " + "; ".join(audit["errors"]))


if __name__ == "__main__":
    main()
