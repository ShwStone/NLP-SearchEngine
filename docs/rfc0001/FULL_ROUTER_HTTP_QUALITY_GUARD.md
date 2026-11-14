# Full-corpus quality guard HTTP audit

The loopback HTTP benchmark evaluates a separate CE-33/CE-40 policy. It selects reranking depth using BM25 statistics before CE scoring. Queries above the configured threshold `0.15146` use CE-40 and the rest use CE-33. Explicit lexical constraints take the fixed path. This benchmark uses CUDA BGE, Qdrant `hnsw_ef=128`, and a dedicated evaluation handler. Production serving configuration and the four-arm CE holdout with a matched random control are tracked separately.

## Frozen inputs

- Corpus: official 8,096,668-document DuReader corpus.
- Expected holdout: 808 queries from the frozen local split of official-dev. The verifier binds the exact query and qrels hashes in the configuration.
- Candidate retrieval: frozen BM25 + BGE-small dense + equal-RRF.
- Gate features: `n_terms`, `vocab_coverage`, `relative_gap`, `candidate_hits`.
- Threshold: `0.15146`. The author reports that it was frozen before reading holdout labels. The threshold selection receipt has yet to be verified.
- Actions: heavy `CE-40`; cheap `fixed CE-33`.

## Reported HTTP result, pending request-log readback

| quantity | quality-guard-40 | fixed CE-33 |
| --- | ---: | ---: |
| queries | 808 | 808 |
| CE-40 queries | 322 of 808 (39.85%) | 0 (declared control) |
| mean CE pairs | 35.790 | 33.000 |
| p50 wall latency (ms) | 316.10 | 259.29 |
| p95 wall latency (ms) | 2149.09 | 2109.67 |

The reported differences relative to CE-33 are `+0.00498` nDCG@10 (paired-query bootstrap 95% CI `[+0.00149, +0.00897]`) and `+0.00622` Recall@10 (95% CI `[+0.00271, +0.01022]`). The reported mean is below the absolute cap of `36.49` CE pairs and above the 33-pair comparison cost. The author reports 91 newly promoted requests and 717 reused decisions from the earlier threshold. The table's wall latencies combine those sources. The confidence intervals use query-level resampling.

Re-verify using the original request log, frozen model bundle, query file, and qrels. Use the original log digest in the configuration, or record a fresh capture's digest before scoring. The historical log predates the explicit fallback field required for fresh captures.

The original digest identifies the historical capture. Its metrics are written to `audit-verified.json`. Its CLI exit status is 0 when the historical quality gate passes and 2 when it fails. To recapture, run `evaluation.benchmark_dureader_policies_http` with `--policies quality-guard-40,fixed-pair-budget` and `--reader-source RFC4_CHECKOUT` on the complete frozen split. The benchmark records the ordered candidate IDs before reranking. The audit requires both requests for each query to start with identical candidates and BM25 gate features, and requires each arm to reorder only its declared CE prefix. After these checks pass, the benchmark saves the request SHA-256 in `capture-before-scoring.json` before opening the qrels.

A fresh capture audited using only its request digest receives the source label `unbound_recapture`, writes its metrics to `audit-unbound.json`, and exits with status 3 even when its quality gate passes. For a runtime-attested recapture, pass `--capture-only` to end the benchmark without opening qrels. A separate verifier must witness an isolated Qdrant instance restored from the frozen snapshot, verify that the live benchmark connects to that same instance, and record the capture receipt digest before scoring. The benchmark itself validates the loaded compact files, corpus alignment, collection compatibility, and actual model inventory and runtime. Collection metadata and point count alone cannot certify a native restore.

```bash
python -m evaluation.audit_quality_guard_http --folder RUN_DIR --qrels QRELS --models MODELS_JSON \
  --queries QUERIES --requests-sha256 REQUEST_LOG_SHA256
```

For the runtime-attested path, append `--capture-sha256 CAPTURE_SHA256 --runtime-attestation ATTESTATION_JSON TRUSTED_ATTESTATION_SHA256 --restore-receipt RESTORE_ACCEPTED_JSON --live-restore-witness WITNESS_JSON`. The verifier's attestation is a JSON object with `schema_version: 1`, `status: "VERIFIED_ISOLATED_RESTORE_AND_CAPTURE"`, `capture_sha256`, `restore_receipt_sha256`, `live_restore_witness_sha256`, and the independently checked `runtime` and `runtime_source_sha256` objects. Record the attestation digest separately from the run directory.

The independent witness records `schema_version: 1`, `capture_sha256`, `native_restore_sha256`, `snapshot_sha256`, `collection`, `exact_points`, `restored_instance`, and `benchmark_instance`. Use a non-secret instance identifier that the verifier ties to the actual process and endpoint in its retained launcher evidence. The two instance identifiers must match. The auditor verifies the witness digest and checks the native restore receipt's pinned delivery, compact index, vector manifest and snapshot before opening qrels. This path writes `audit-runtime-attested.json` and uses exit 0 or 2 for the quality gate. A recapture can verify the current runtime and outcomes. The historical threshold-selection chronology still requires its original evidence.
