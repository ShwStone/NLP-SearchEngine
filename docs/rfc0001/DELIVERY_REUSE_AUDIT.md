# RFC-0004 delivery reuse: current evidence and remaining gates

The local `NLP-RFC4-Delivery-20261107` delivery is useful, but its entire
directory has **not** passed the delivery verifier: vector-cache files are
missing or have unexpected sizes, including zero-byte placeholders. Do not
delete or overwrite the delivered originals. Reuse individually verified
artifacts in independent directories.

## Verified identities

The official baseline archive has additionally passed an independent direct
source audit: pinned archive SHA256, hashes of the extracted four passage
files/ID map/dev files, and all 8,096,668 normalized document IDs, texts and
positions agree. See `official-full-source-audit.json`. This closes the prior
gap between subset-coverage checks and all-document official-source identity.

- The extracted native Qdrant snapshot is 18,477,460,992 bytes with SHA256
  `2df5c889386db365e4d72d089f90bfb736c8a6a24d5bede4a10f727d9f0976e0`.
- The existing Qdrant 1.16.0 musl release archive matches the delivered
  inventory SHA256 `d4752e9bf673c7f98153264cb8f416398928c7787e1269350c179fb856de3ef7`.
  The running binary reports version 1.16.0, commit
  `0d79528705f04def1b52842f747e5a00ab1e338a`.
- Snapshot restore uses a new storage directory and loopback port 6377.
  Server configuration and exact count verify against the vector manifest;
  the server reports green status and 8,096,668 indexed vectors.
- An independent streaming scan of all 8,096,668 normalized documents confirms
  file SHA256 `bc67f3ccfe7423c039f04c7a5e7f6eae07a7f722087edf0caa7cfc694a8c45e8`
  and vector-contract fingerprint
  `0ae8b1174d0ee8e55f269c3c6d8e5f12d7e9c79fcb72c3966e99de18c9061019`.
  These hashes use different serialization and must not be compared directly.
- Exhaustive server scroll now verifies all 8,096,668 point IDs and external
  document IDs against the ordered corpus. Every point is covered, not only
  the first/last samples. A separate numeric audit reencodes one fixed document
  from each of 81 shards using the pinned model: maximum absolute vector error
  `2.1606683731079102e-07`, minimum cosine `0.9999998807907104`. This is a
  sampled numerical check and does not claim all vectors were reencoded.
- The delivered train and dev query bytes match the frozen 100k recipe:
  172 train queries (`0fc31d70fb1c4548483b3c3d115d0e89fe957c813312d6823ff155b4ef488290`)
  and 1,020 dev queries (`ce010e6ea3a498a696d653b58bcf7976b5101d4034341983429df0a44837cfd9`).

## Historical results are a separate evaluation

The delivered `root-verified-results.json` evaluates 808 test queries. It is
not this goal's 1,020-query internal-dev comparison. Its own
`latency_comparability` is `per_arm_only_mixed_epochs`; cross-arm speedup claims
must not be inferred from those timings. The test set must not become a tuning
set. Existing aggregate historical results are external reference evidence,
not substitutes for learned Router, CE-0/28/40, or equal-budget controls.

## Search parameter defect and rerun

The router worktree's original `QdrantVectorIndex` ignored attributes assigned
as `vec.search_params`, although multiple 100k harnesses assigned ef=128 that
way. Its `query_points` calls did not forward the parameters. Earlier reported
100k scores must be treated as results under the backend's effective default
configuration, not proof that ef=128 was applied.

The adapter now validates and forwards explicit ANN/exact parameters for both
supported SDK query paths. Regression tests cover forwarding, default behavior,
invalid values, and legacy search. `evaluation.capture_full_dense` additionally
passes ef=128/exact=false directly and records the actual request parameters.
Full-corpus train/dev captures are complete (172/1,020 rows, Top-200; one dev
query with explicit lexical constraints skips dense). Corrected 100k captures
are rerun separately before updating any controlled quality comparison.
Component timings recorded during concurrent audit jobs are not HTTP latency
evidence and must not be added to offline BM25 batch time.

The corrected, explicit-ef=128 dense captures are complete for both corpus
sizes. On the same 1,019 supported dev queries, 100k nDCG@10 is
`0.2647649751`, and full-corpus nDCG@10 is `0.2600519719`; Recall@10 is
`0.2908343473` and `0.2837592503`, respectively. The paired full-minus-100k
nDCG difference is `-0.0047130031` (95% query bootstrap CI
`[-0.0080967504, -0.0019563530]`, 2,000 samples, seed 42). This is a
corpus-size comparison using separately frozen HNSW builds, not a Router
improvement or an end-to-end performance result. One explicit-constraint
query is excluded from dense quality for both sizes. MAP@100 differs as
well, so the 100k subset must not stand in for full-corpus evaluation.

Compact identity/readback reports and the controlled dense aggregates are
stored alongside this document as `full-delivery-*.json`. Large captures
remain outside the repository. The complete worktree regression suite passed
259 tests after the initial audit/merge changes; subsequent parameter and
Unicode regressions must be included in the final rerun before delivery.

## Outstanding acceptance work

The compact full lexical archive is now independently extracted and audited:
all seven compact file hashes match, and every one of 8,096,668 document IDs,
titles, texts and relative positions matches the normalized full corpus and
vector fingerprint. See `full-delivery-compact-audit.json`. This is not a claim
that every posting was rederived from source text.

The frozen delivery source at commit `dfa697687ddf1a0204d42a4c52992a0ac159212d`
provides the compact reader. `evaluation.delivery_backend` checks both reader
source hashes before loading them; it never silently imports an unrelated
checkout. The float64 accelerator retains the reference BM25 term order and
uses the reader's bounded postings cache. The compact and independent
81-shard BM25 captures agree on the ordered candidate IDs and scores for all
172 training queries and 1,020 development queries. The two receipts cover
34,400 and 204,000 candidate hits, respectively, with maximum score
difference 0. [Train receipt](full-train-sharded-compact-parity.json) · [Development receipt](full-dev-sharded-compact-parity.json)

`evaluation.continue_full_dureader` serializes fusion, CE scoring and nested
training using only completed, hashed outputs. It checks upstream process
identity while waiting, fails if a producer exits without publishing, and never
restarts an unfinished producer or triggers shutdown. The automatic pipeline
does not replace final audit, all-query serving validation or remote delivery.

The full-corpus development candidate and quality comparisons are recorded
with the grouped confidence intervals and fixed-depth controls. The loopback
HTTP run took place alongside other batch work. The remaining isolated timing
check needs a quiescent worker using the same frozen retrieval and CE policies,
warmup, request order, and coverage checks. The independent 808-query CE
holdout also remains pending. [Quality guard](DUREADER_QUALITY_GUARD.md) · [Native HTTP evidence](DUREADER_FULL_FINAL_REPORT.json) · [Course report](../../REPORT.md#10-ce-重排与逐查询成本分配)

Reproduce the corpus identity check with:

```bash
python -m evaluation.audit_delivery_corpus \
  --corpus /path/to/full-corpus.jsonl \
  --manifest-path /path/to/vector_manifest.json \
  --output /path/to/new-corpus-audit.json
```

Read back every server ID and external document ID with:

```bash
python -m evaluation.audit_delivery_qdrant \
  --corpus /path/to/full-corpus.jsonl \
  --manifest-path /path/to/vector_manifest.json \
  --url http://127.0.0.1:6377 \
  --output /path/to/new-server-audit.json
```

The second audit verifies all ordered IDs/payloads and the corpus fingerprint;
it explicitly does not claim to validate every vector's numeric value.
