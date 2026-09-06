"""Source data and recorded evidence must survive cross-platform checkout intact."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "name,expected",
    [
        ("wine.data", "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659"),
        ("wine.names", "f1b84f2ef845e0bdebf13e14fa7a213e56de4f1baa40c5974dbd1ee51c5ae710"),
        ("optdigits.tes", "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8"),
        ("optdigits.names", "3e82f7202d72a2b7dbdbc324c8c90fe8853164f5d6ab978a071357ad3de89f02"),
    ],
)
def test_original_uci_bytes(name, expected):
    assert digest(ROOT / "examples/data" / name) == expected


def test_recorded_digits_artifact_hashes():
    docs = ROOT / "docs"
    for record in json.loads((docs / "digits_processes.json").read_text()):
        for suffix in ("json", "npz"):
            path = docs / f"digits_k{record['neighbors']}.{suffix}"
            assert digest(path) == record[suffix + "_sha256"], path.name


def test_recorded_streaming_checkpoint_hashes():
    docs = ROOT / "docs"
    for line in (docs / "streaming_results.jsonl").read_text().splitlines():
        record = json.loads(line)
        if record["event"] == "process_finished":
            path = docs / "streaming_results.jsonl.workers" / Path(record["checkpoint"]).name
            assert digest(path) == record["checkpoint_sha256"], path.name


def test_recorded_wine_artifact_hashes():
    directory = ROOT / "docs/wine_holdout"
    manifest = json.loads((directory / "study.json").read_text())
    assert digest(directory / "selection.json") == manifest["selection_sha256"]
    for record in manifest["processes"]:
        for field, suffix in (("checkpoint", "json"), ("arrays", "npz"), ("log", "txt")):
            path = directory / record[field]
            assert digest(path) == record[suffix + "_sha256"], path.name


def test_recorded_stability_artifact_hashes():
    directory = ROOT / "docs/stability_results"
    manifest = json.loads((directory / "study.json").read_text())
    assert digest(ROOT / "docs/stability_protocol.md") == manifest["protocol_sha256"]
    for record in manifest["processes"]:
        for field in ("checkpoint", "arrays", "selection", "log"):
            path = directory / record[field]
            assert digest(path) == record[field + "_sha256"], path.name
