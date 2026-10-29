"""Secret-free provenance and cost reports for offline deliveries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from contracts.index_format import sha256_of
from contracts.retrieval import corpus_fingerprint

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_FILE = "build_provenance.json"
REPORT_FILE = "build_report.json"


def _atomic_json(path: Path, value: Mapping) -> None:
    path = Path(path)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    pending.replace(path)


def make_provenance(corpus_path: Path, dataset: str, reader, *, vector_manifest=None) -> dict:
    """Build a portable identity record without machine paths or credentials."""

    source = Path(corpus_path)
    value = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "dataset": dataset,
        "source": {
            "filename": source.name,
            "bytes": source.stat().st_size,
            "sha256": sha256_of(source),
        },
        "corpus": {"n_docs": reader.n_docs,
                   "ordered_fingerprint": corpus_fingerprint(reader)},
        "bm25": reader.manifest().get("bm25", {}),
    }
    if vector_manifest is not None:
        value["embedding"] = dict(vector_manifest.embedding.__dict__)
        value["vector"] = {
            "collection": vector_manifest.collection,
            "backend": vector_manifest.backend,
            "index_params": dict(vector_manifest.index_params),
            "snapshot_files": dict(vector_manifest.snapshot_files),
        }
    return value


def write_provenance(path: Path, value: Mapping) -> None:
    _atomic_json(Path(path), value)


def make_report(build_dir: Path, *, dataset: str, n_docs: int,
                phase_seconds: Mapping[str, float]) -> dict:
    """Record phase timing and artifact bytes/hashes without inventing peaks."""

    root = Path(build_dir)
    artifacts = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()
                       and p.name not in (REPORT_FILE, REPORT_FILE + ".tmp")):
        relative = path.relative_to(root).as_posix()
        artifacts[relative] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "dataset": dataset,
        "n_docs": n_docs,
        "phase_seconds": {key: round(float(value), 6)
                          for key, value in phase_seconds.items()},
        "artifacts": artifacts,
        "resources": {
            "peak_host_ram_bytes": None,
            "peak_gpu_memory_bytes": None,
            "note": "标准库未可靠测量峰值；正式作业须由作业监控补录。",
        },
    }


def write_report(path: Path, value: Mapping) -> None:
    _atomic_json(Path(path), value)
