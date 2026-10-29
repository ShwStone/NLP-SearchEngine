"""Validate a released offline dual-index directory without online imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from contracts.index_format import MANIFEST, REQUIRED_FILES, load_manifest, sha256_of, verify_files
from contracts.retrieval import VECTOR_MANIFEST, VectorManifest, corpus_fingerprint
from .disk_index import DiskIndex
from .provenance import PROVENANCE_FILE, PROVENANCE_SCHEMA_VERSION, REPORT_FILE


def _safe_artifact_path(root: Path, relative: str) -> Path:
    """Resolve a report/manifest path while rejecting traversal and absolutes."""

    if not isinstance(relative, str) or not relative:
        raise ValueError("交付产物路径必须是非空字符串")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"交付产物路径非法: {relative}")
    resolved_root = root.resolve()
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"交付产物路径越界: {relative}") from error
    return resolved


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def validate_build(build_dir: str | Path) -> dict:
    """Fail closed unless A1, vector identity, snapshot hashes and reports align."""

    root = Path(build_dir)
    lexical_dir = root / "bm25"
    lexical_manifest = load_manifest(lexical_dir)
    verify_files(lexical_dir, lexical_manifest, require_hashes=True)
    reader = DiskIndex(lexical_dir)
    vector_path = root / VECTOR_MANIFEST
    if not vector_path.is_file():
        raise ValueError("双索引缺少 vector_manifest.json")
    vector_manifest = VectorManifest.from_dict(json.loads(vector_path.read_text(encoding="utf-8")))
    if vector_manifest.n_docs != reader.n_docs:
        raise ValueError("向量 n_docs 与 BM25 不一致")
    if vector_manifest.corpus_sha256 != corpus_fingerprint(reader):
        raise ValueError("向量 corpus_sha256 与 BM25 不一致")
    if vector_manifest.backend == "qdrant-hnsw" and not vector_manifest.snapshot_files:
        raise ValueError("生产 Qdrant 产物必须声明快照 SHA256")
    snapshot_hashes = {}
    for name, digest in vector_manifest.snapshot_files.items():
        if Path(name).name != name:
            raise ValueError(f"快照路径非法: {name}")
        path = _safe_artifact_path(root, name)
        if not path.is_file():
            raise ValueError(f"快照路径非法或不存在: {name}")
        actual = sha256_of(path)
        if actual != digest:
            raise ValueError(f"快照 SHA256 不匹配: {name}")
        snapshot_hashes[name] = actual
    for required in (PROVENANCE_FILE, REPORT_FILE):
        if not (root / required).is_file():
            raise ValueError(f"交付缺少 {required}")
    provenance = json.loads((root / PROVENANCE_FILE).read_text(encoding="utf-8"))
    corpus = provenance.get("corpus") if isinstance(provenance, dict) else None
    source = provenance.get("source") if isinstance(provenance, dict) else None
    embedding = provenance.get("embedding") if isinstance(provenance, dict) else None
    vector = provenance.get("vector") if isinstance(provenance, dict) else None
    expected_embedding = dict(vector_manifest.embedding.__dict__)
    expected_vector = {
        "collection": vector_manifest.collection,
        "backend": vector_manifest.backend,
        "index_params": dict(vector_manifest.index_params),
        "snapshot_files": dict(vector_manifest.snapshot_files),
    }
    source_filename = source.get("filename") if isinstance(source, dict) else None
    if (not isinstance(corpus, dict)
            or provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION
            or provenance.get("dataset") != lexical_manifest.dataset
            or corpus.get("ordered_fingerprint") != vector_manifest.corpus_sha256
            or corpus.get("n_docs") != reader.n_docs
            or not isinstance(source, dict)
            or not isinstance(source_filename, str)
            or Path(source_filename).name != source_filename
            or not isinstance(source.get("bytes"), int) or source.get("bytes") < 0
            or not _is_sha256(source.get("sha256"))
            or embedding != expected_embedding
            or vector != expected_vector
            or provenance.get("bm25") != lexical_manifest.bm25.to_dict()):
        raise ValueError("provenance 与向量 corpus 指纹不一致")
    report = json.loads((root / REPORT_FILE).read_text(encoding="utf-8"))
    artifacts = report.get("artifacts") if isinstance(report, dict) else None
    if (not isinstance(report, dict)
            or report.get("schema_version") != PROVENANCE_SCHEMA_VERSION
            or report.get("dataset") != lexical_manifest.dataset
            or report.get("n_docs") != reader.n_docs
            or not isinstance(artifacts, dict)):
        raise ValueError("成本报告缺少 artifacts 对象")
    required_artifacts = {
        *(f"bm25/{name}" for name in [MANIFEST, *REQUIRED_FILES]),
        VECTOR_MANIFEST,
        PROVENANCE_FILE,
        *vector_manifest.snapshot_files,
    }
    missing = sorted(required_artifacts - set(artifacts))
    if missing:
        raise ValueError(f"成本报告缺少必需产物: {', '.join(missing)}")
    for relative, entry in artifacts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"成本报告缺少产物摘要: {relative}")
        path = _safe_artifact_path(root, relative)
        if not path.is_file():
            raise ValueError(f"成本报告引用的产物不存在: {relative}")
        if entry.get("sha256") != sha256_of(path):
            raise ValueError(f"成本报告哈希不匹配: {relative}")
    return {
        "dataset": lexical_manifest.dataset,
        "n_docs": reader.n_docs,
        "corpus_sha256": vector_manifest.corpus_sha256,
        "backend": vector_manifest.backend,
        "snapshot_sha256": snapshot_hashes,
        "provenance": str(root / PROVENANCE_FILE),
        "report": str(root / REPORT_FILE),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验已发布的离线双索引产物")
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate_build(args.build_dir), ensure_ascii=False, indent=2))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"validation error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
