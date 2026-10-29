# RFC-0004 scale path

This document describes the optional scale support currently present in the
repository. It is an engineering handoff and does not claim that the course
corpus or the 8M-document build has completed.

## Validated pieces

- `offline.prepare` validates a JSONL corpus, rejects duplicate external IDs,
  writes a normalized corpus and an ID map, and records source/analyzer hashes.
- `offline.stream_build` uses SQLite-backed uniqueness, length, and posting
  storage and streams the unchanged v1 JSONL output. It removes the legacy
  builder's full Python conformance oracle from the lexical build path.
- `offline.compact.compile_index` converts a checksum-verified v1 lexical
  index into a read-only compact artifact. `offline.packed_index.PackedIndex`
  provides bounded decoded-postings caching and a portable positional-read
  fallback on Windows.
- `offline.disk_documents.DiskDocuments` reads one document at a time for
  vector encoding. `adapters.embedding.BGEEncoder` pins the BGE-small-zh
  model revision, tokenizer behavior, dimension, prefix, truncation and
  normalization; `adapters.model_files` verifies every downloaded file.
- `scripts/scale_preflight.py` records source bytes, a bounded token sample,
  raw vector bytes and a planning disk estimate. Its output is explicitly
  `estimate_only`; it does not prove peak RAM, Qdrant overhead or throughput.

The compact reader and the preparation/vector helper pass the tiny-corpus
conformance regression in `tests/test_scale_support.py` on Windows.

As an external, non-versioned smoke receipt, the same path was run on the
1,029-document depth-4 government-web crawl: `stream_build` completed in
66.69 seconds with average document length 2,945.7, and `compact` completed in
15.22 seconds with 178,504 terms. The resulting manifests and their hashes are
recorded in `evaluation/RESOURCES.md`. This is evidence that the staged path is
usable on a small external corpus; it is not an 8M peak-RSS, disk, throughput,
or Qdrant HNSW receipt.

The depth-4 corpus was then merged by URL with 108 pages fetched from the
site's public sitemap batch, producing 1,137 documents. The merged stream build
completed in 74.19 seconds with average document length 2,797.2; compact
conversion completed in 13.97 seconds with 186,263 terms. The merged manifest
and compact hash are recorded in `evaluation/RESOURCES.md`. These measurements
remain bounded engineering smoke evidence, not an 8M resource receipt.

A separate `contracts.conformance.check_reader` attempt against the 1,029-document
packed artifact exceeded 20 minutes of local CPU time without producing a
receipt and was stopped. The tiny packed-reader conformance remains green; the
large conformance path is therefore unvalidated and must be bounded before it
is used as a production gate.

## Remaining production gate

The legacy `offline.build` path still materializes the conformance oracle in
memory; use `offline.stream_build` when measuring a large lexical build. The
optional compact conversion happens after a source build and therefore does
not remove that source peak. `offline.vector_build.QdrantSnapshotWriter` now
uses a temporary SQLite identity table; the 8M vector build still needs a
complete resource receipt before it is called bounded. A Qdrant HNSW result
must additionally include server version,
collection metadata, exact point count, snapshot hash, clean-instance restore,
peak RSS/disk, throughput and four-mode evaluator output.

The tiny `scripts/smoke_offline.py` command only imports a snapshot into the
same isolated Qdrant process after deleting its collection. It is a dependency
and wiring smoke test, not clean-instance recovery or HNSW performance
evidence.

Large corpora, model files, Qdrant collections/snapshots, generated indexes,
runs and resource receipts belong outside Git.
