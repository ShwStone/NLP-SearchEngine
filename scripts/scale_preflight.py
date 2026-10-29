"""Estimate storage and vector capacity before a large offline build.

The report is deliberately an estimate: it measures a bounded corpus sample,
records the exact source hash, and never turns an extrapolation into a claim
that an 8M build fits.  Run it before allocating a cluster job and keep its
JSON output beside the private run receipt, outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts import analyzer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return int(pages * page_size) if pages > 0 and page_size > 0 else None


def sample_corpus(path: Path, sample_docs: int) -> dict:
    if sample_docs < 1:
        raise ValueError("sample_docs must be positive")
    total_bytes = 0
    rows = 0
    text_bytes = 0
    token_count = 0
    posting_rows = 0
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("doc_id"), str):
                raise ValueError(f"invalid corpus row at line {line_no}")
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"empty text at line {line_no}")
            rows += 1
            total_bytes += len(line.encode("utf-8"))
            text_bytes += len(text.encode("utf-8"))
            tokens = analyzer.analyze(text)
            token_count += len(tokens)
            posting_rows += len({token.term for token in tokens})
            if rows >= sample_docs:
                break
    if rows == 0:
        raise ValueError("corpus sample is empty")
    return {
        "sample_documents": rows,
        "sample_jsonl_bytes": total_bytes,
        "sample_text_bytes": text_bytes,
        "sample_tokens": token_count,
        "sample_unique_terms_per_doc": posting_rows,
        "avg_jsonl_bytes": total_bytes / rows,
        "avg_text_bytes": text_bytes / rows,
        "avg_tokens": token_count / rows,
        "avg_unique_terms_per_doc": posting_rows / rows,
    }


def preflight(path: str | Path, *, target_docs: int = 8_096_668,
              sample_docs: int = 10_000, dimension: int = 512) -> dict:
    corpus = Path(path)
    if target_docs < 1 or dimension < 1:
        raise ValueError("target_docs and dimension must be positive")
    sample = sample_corpus(corpus, sample_docs)
    scale = target_docs / sample["sample_documents"]
    vector_raw = target_docs * dimension * 4
    estimated_corpus = sample["sample_jsonl_bytes"] * scale
    # The factor is a planning margin for build staging, WAL/segments, and a
    # snapshot. It is not a measured Qdrant overhead and must be replaced by
    # the cluster receipt after the real run.
    recommended_disk = (estimated_corpus + vector_raw) * 3.0
    usage = shutil.disk_usage(corpus.parent)
    return {
        "source": {"filename": corpus.name, "bytes": corpus.stat().st_size,
                   "sha256": _sha256(corpus)},
        "analyzer": analyzer.config(),
        "target_documents": target_docs,
        "embedding_dimension": dimension,
        "sample": sample,
        "estimates": {
            "corpus_jsonl_bytes": int(estimated_corpus),
            "raw_float32_vectors_bytes": vector_raw,
            "planning_disk_bytes_with_3x_margin": int(recommended_disk),
            "avg_tokens_per_document": sample["avg_tokens"],
        },
        "host": {
            "available_disk_bytes_at_check": usage.free,
            "physical_memory_bytes": _host_memory_bytes(),
        },
        "interpretation": {
            "status": "estimate_only",
            "memory_bound": "not_proven; measure peak RSS for the selected builder",
            "qdrant_overhead": "not measured; include WAL, segments, payload, HNSW and snapshot",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight large offline build capacity")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--target-docs", type=int, default=8_096_668)
    parser.add_argument("--sample-docs", type=int, default=10_000)
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight(args.corpus, target_docs=args.target_docs,
                       sample_docs=args.sample_docs, dimension=args.dimension)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
