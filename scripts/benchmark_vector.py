"""Isolate Qdrant HNSW ef/candidate tradeoffs on frozen query embeddings.

This is a backend latency and top-10 overlap experiment, not an HTTP or qrels
relevance evaluation. The final selected configuration still needs HTTP validation.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time

from adapters.embedding import BGEEncoder
from contracts.retrieval import VectorManifest
from online.vector_index import QdrantVectorIndex
from scripts.benchmark_http import read_queries, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--ef", type=int, nargs="+", default=[64, 128, 256])
    parser.add_argument("--candidates", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--reference-ef", type=int, default=256)
    parser.add_argument("--reference-candidates", type=int, default=200)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    variants = [(ef, candidates) for candidates in args.candidates for ef in args.ef]
    reference = (args.reference_ef, args.reference_candidates)
    if reference not in variants or any(ef < 1 or k < 10 for ef, k in variants):
        parser.error("ef must be positive, candidates >=10, and the reference variant must be included")
    rows = read_queries(args.queries, args.limit)
    manifest = VectorManifest.from_dict(json.loads(args.manifest.read_text(encoding="utf-8")))
    encoder = BGEEncoder(spec=manifest.embedding, model_path=args.model_path, device="cpu")
    vectors = {}
    embeddings = []
    for row in rows:
        started = time.perf_counter()
        vector = encoder.encode_query(row["query"])
        embeddings.append(vector)
        row["encode_ms"] = (time.perf_counter() - started) * 1000
    for ef, candidates in variants:
        vectors[(ef, candidates)] = QdrantVectorIndex(manifest, args.url, hnsw_ef=ef)
    measurements = []
    # Warm each variant with the first query, then preserve the same query order.
    for variant in variants:
        ef, candidates = variant
        vectors[variant].search(embeddings[0], candidates)
        for row, embedding in zip(rows, embeddings):
            started = time.perf_counter()
            hits = vectors[variant].search(embedding, candidates)
            elapsed = (time.perf_counter() - started) * 1000
            measurements.append({"qid": row["qid"], "ef": ef, "candidates": candidates,
                                 "vector_ms": elapsed,
                                 "doc_ids": [hit.doc_id for hit in hits[:10]]})
    reference_rows = {(row["qid"]): row for row in measurements
                      if (row["ef"], row["candidates"]) == reference}
    summaries = {}
    for variant in variants:
        ef, candidates = variant
        selected = [row for row in measurements if (row["ef"], row["candidates"]) == variant]
        overlaps = []
        for row in selected:
            expected = reference_rows[row["qid"]]["doc_ids"]
            actual = row["doc_ids"]
            overlaps.append(len(set(expected) & set(actual)) / 10)
            row["reference_overlap_at_10"] = overlaps[-1]
            row["exact_ordered_match_at_10"] = actual == expected
        summaries[f"ef{ef}-k{candidates}"] = {
            "vector_ms": summarize([row["vector_ms"] for row in selected]),
            "reference_overlap_at_10_mean": statistics.fmean(overlaps),
            "reference_overlap_at_10_min": min(overlaps),
            "exact_ordered_matches": sum(row["exact_ordered_match_at_10"] for row in selected),
            "queries": len(selected),
        }
    receipt = {
        "schema_version": 1,
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Qdrant vector-only latency and overlap with reference ANN; no qrels relevance claim",
        "collection": manifest.collection,
        "n_docs": manifest.n_docs,
        "reference": {"ef": args.reference_ef, "candidates": args.reference_candidates},
        "config": {"ef": args.ef, "candidates": args.candidates, "queries": len(rows)},
        "encode_ms": summarize([row["encode_ms"] for row in rows]),
        "summary": summaries,
        "queries": rows,
        "measurements": measurements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"label": args.label, "out": str(args.out), "summary": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
