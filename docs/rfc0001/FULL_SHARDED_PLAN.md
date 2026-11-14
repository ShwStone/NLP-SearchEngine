# Full corpus sharded BM25 plan

The in-memory builder cannot stay below the container memory limit on the full corpus. The normalized corpus is split into 81 deterministic shards of at most 100,000 documents. Each shard retains the existing index contract (docs, postings, offsets, stats, manifest). The full build remains incomplete until all shards and their input identities are audited.

Run the resumable launcher from the repository root:

```bash
python -u -m evaluation.build_full_sharded_bm25 \
  --corpus-dir /path/to/corpus100k --out /path/to/index100k --workers 3
```

The launcher uses the current interpreter, a process lock, and at most three child builders. Before reusing a manifest it verifies every required file hash, analyzer identity, dataset name, and BM25 parameters. A corrupt completed shard fails closed. File hashes alone do not establish correspondence with source corpus records; that needs the final corpus/index identity audit.

## Global scoring invariant

Shard-local BM25 scores must **not** be merged directly. Every shard must score with the same full-corpus document count N, summed document frequency df(term), and document-weighted global average length. Only then are each shard's Top-k results sufficient to obtain exact global Top-k. Global document IDs use cumulative shard sizes and ties use ascending global document ID.

`evaluation/sharded_bm25.py` implements this scoring view and deterministic merge without materializing all shard readers simultaneously. The parity regression compares unequal shard sizes, unequal term distributions, repeated query terms, absent terms, and several cutoffs against a monolithic index. This proves the scoring primitive on the test corpus; it is not full-corpus retrieval or latency evidence. The remaining work includes building global statistics, binding full-corpus inputs, querying all shards, and independent end-to-end audit.

## Query-scoped global statistics

After every shard is built, prepare statistics with:

```bash
python -m evaluation.prepare_sharded_bm25_statistics \
  --corpus-dir /path/to/corpus100k --index-dir /path/to/index100k \
  --queries /path/to/queries.jsonl --output /path/to/global-statistics.json
```

The generator verifies every shard's required file hashes, computes average length from the integer document lengths, and sums query-term df across all shards. Its output binds query bytes, source shard bytes, manifest bytes and cumulative document offsets. It fails before publishing output if any declared shard is missing. It deliberately records `source_record_parity_verified=false`: the final audit must still compare actual source records to indexed documents. This output covers only the declared query terms, so serving arbitrary new query terms requires a full vocabulary statistics artifact or an equivalent exact lookup.

## Source identity audit

After build completion, run:

```bash
python -m evaluation.audit_sharded_bm25_sources \
  --corpus-dir /path/to/corpus100k --index-dir /path/to/index100k \
  --full-corpus /path/to/full-corpus.jsonl --expected-documents 8096668 \
  --output /path/to/source-parity-audit.json
```

This streams source and indexed documents in lockstep, checking all document IDs, titles, texts and contiguous internal IDs. It also hashes the raw concatenation of ordered source shards and compares it with the full normalized corpus hash. The report is published only when every shard is complete, required index hashes verify, the document counts agree, and source parity passes. It does not claim semantic conformance of every posting or correctness of the upstream normalization; those remain separate audit requirements.

## Full BM25 candidate capture

After global statistics and source identity are verified:

```bash
python -m evaluation.capture_sharded_bm25 \
  --index-dir /path/to/index100k --queries /path/to/queries.jsonl \
  --statistics /path/to/global-statistics.json --output /path/to/full-bm25-capture --k 200
```

One shard reader is resident at a time. All queries are scored before moving to the next shard, and only the merged Top-k candidates are retained. Explicit boolean/phrase filters are evaluated within each shard; NOT-only zero-score candidates retain global document-order ties. Statistics, queries and per-shard manifests are bound by SHA256, and index file checksums are verified when readers load. The resulting candidate capture is local-only and must not be committed to the PR. Batch duration is recorded as candidate generation work, never presented as online latency. A monolithic-engine comparison covers ordinary queries, AND, NOT and phrase queries on an unequal-shard test corpus.

## Full/subset normalization parity

`evaluation.audit_full_subset_parity` scanned all 8,096,668 normalized records and verified every one of the frozen 100,000 subset records by ID, title, text, and relative order. The full normalized corpus SHA256 is `bc67f3ccfe7423c039f04c7a5e7f6eae07a7f722087edf0caa7cfc694a8c45e8`. The frozen subset corpus SHA256 remains `f24b50b6012b47f6fb5f72272666dbf187124a869f19d1c8d83f7e30d6cb62b4`. The aggregate audit is in `full-subset-parity-audit.json`. This establishes exact subset coverage in the full normalized file, not a record-by-record identity audit of all full-corpus documents against the official archive.

## Bounded-memory dense encoding

The existing Qdrant writer materializes `records = list(records)`, so it must not be used unchanged for the full corpus. `evaluation.encode_full_dense_chunks` encodes documents directly from corpus JSONL in chunks, preserving `title + '\n' + text` and the frozen vector manifest's embedding specification. The shared encoder verifies the local pinned model inventory. Each completed chunk binds the source SHA256, spec, IDs, offset, dimensions, dtype and vector file hash. A process lock prevents concurrent encoders in the same output directory. Resume verifies existing chunks and re-encodes only unpublished chunks. Numerical shape, finiteness and unit-norm checks run before a chunk is accepted.

```bash
python -u -m evaluation.encode_full_dense_chunks \
  --corpus-dir /path/to/corpus100k --out /path/to/dense-chunks \
  --model-path /path/to/bge-small-zh-v1.5 \
  --vector-manifest /path/to/frozen-100k/vector_manifest.json \
  --adapter-root /path/to/shared-adapters --limit-shards 1
```

The first run is limited to one shard to measure actual throughput and resource use. Encoding summaries explicitly state `qdrant_built=false`. A streaming Qdrant ingestion path, corpus-wide identity alignment, snapshots and ANN checks remain required before the dense backend can be evaluated as a full-corpus index.

`evaluation.ingest_dense_shard` now provides a bounded-batch ingestion primitive.
It binds the collection to the exact build metadata and checks dimensions and
distance before writing. Retrying a partially uploaded shard replays validated
records with stable global point IDs, so upserts do not duplicate documents.
After ingestion it checks the exact count for the declared shard IDs. A failure
can leave partial points in the unpublished build collection; successful ingestion
also explicitly reports `collection_published=false`. Final snapshots, complete
corpus coverage, HNSW configuration/readiness and server-side vector readback
remain separate gates. The fixture integration test uses Qdrant's local engine;
it is not evidence of full-corpus ingestion or HTTP server parity.

## Full BM25 shard completion

All 81 BM25 shards are now built and independently verified. The source parity audit reports 8,096,668 documents, exact source/index document fields and local IDs, contiguous offsets, required file SHA256 checks, and matching concatenation with the normalized full corpus. The full corpus SHA256 is `bc67f3ccfe7423c039f04c7a5e7f6eae07a7f722087edf0caa7cfc694a8c45e8`; the last shard contains 96,668 documents. The machine-readable audit is `full-sharded-bm25-source-parity.json`. This verifies the sharded lexical artifacts and source identity; it does not yet publish global statistics, dense/Qdrant, or retrieval-quality results.
