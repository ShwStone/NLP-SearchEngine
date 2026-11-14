# HTTP evaluator

`evaluation/evaluate.py` evaluates the public `POST /search` API without importing
the search engine implementation. It reads TREC qrels, sends each query over
HTTP, optionally writes a TREC run, and reports nDCG@10, Recall@10, MRR@10,
MAP@100, and latency.

Query files accept `qid<TAB>query` lines or JSONL records such as
`{"qid":"q1","query":"heart attack"}`. Qrels use the usual TREC columns:
`qid iteration docid relevance`.

```bash
python3 -m evaluation.evaluate \
  --endpoint http://127.0.0.1:8000/search \
  --queries queries.tsv --qrels qrels.tsv \
  --output-run runs/bm25.trec \
  --include-per-query
```

### Retrieval depth and routing telemetry

HTTP requests fetch `max(--k, --map-k)` results (100 by default), while
nDCG/Recall/MRR and MAP apply their own cutoffs (10 and 100 by default).
The TREC run retains the full retrieved list, and per-query `retrieved` counts
that list rather than only the top-10 metric window. The public API permits
at most 200 results per request; non-lexical modes also require the server's
`candidate_k` to cover the requested depth. Latency measures this full-depth
HTTP request, including serialization and transfer, not a separate top-10 call.

API v2 routing fields are validated and retained with `--include-per-query`.
The aggregate `routing` object always reports query count, `vector_attempts`,
`vector_successes`, `fallbacks`, and their rates. Attempt and fallback rates use
all observed queries as the denominator; vector success rate uses attempts and
is `null` if no vector call was attempted. An adaptive run with missing vector
dependencies is therefore visibly degraded rather than silently labeled as a
successful hybrid experiment. Add `--fail-on-fallback` to reject any such query
with a nonzero exit status.

For server-side timing analysis, pass a distinct `--request-tag` per HTTP
attempt. The evaluator sends it as the public `run_tag` field and records it
in the request journal and snapshot. The tag starts with an ASCII letter or
digit, can also contain `.`, `_` and `-`, and is at most 128 characters.
`--tag` continues to name the TREC run.

Keep the full server log, including its startup record, and pass it together
with the finalized request sidecar:

```bash
python3 -m evaluation.serving_cost --run runs/bm25.trec.requests.json service.log
```

Use a distinct tag per serial attempt. The log intentionally contains no query
IDs. The checker verifies ordered HTTP status and routing agreement within one
startup epoch. It reports server stage timings and client HTTP latency
separately. Repeated `--run` arguments preserve the per-attempt timing scopes.

Standalone TREC runs contain no routing or latency evidence: those fields are
`null`, not zero failures or zero latency.

The ordinary HTTP evaluator rejects hybrid responses that carry CE reranking
metadata or timings. Use the dedicated CE evaluator below to retain arm identity,
the served ranking, actual pair counts, and paired measurements.

### Train and evaluate a CE40 gate

Freeze train, development and independent holdout queries/qrels before capture.
Keep their query IDs and normalized texts disjoint, and use a warmup set
disjoint from all three. Use the same lexical/vector artifacts, retrieval
parameters and fixed CE model throughout; see the [serving profiles](../docs/online-serving.md#学习门控-ce40).
The evaluator requests 100 results in fixed hybrid mode with BM25 and no
expansion, so set `candidate_k` to at least 100.

Capture **only RRF and fixed CE40** for training, with `capture_features: true`
on CE40 and no gate. Then fit from the completed SQLite capture and the exact
train query/qrels files. The trainer requires scikit-learn; the online gate
loads plain JSON. Use two new output paths in an existing directory.

```bash
python -m evaluation.ce_http \
  --baseline http://127.0.0.1:8000 --rerank 40 http://127.0.0.1:8040 \
  --require-encoder-runtime \
  --queries data/queries.train.jsonl --qrels data/qrels.train.tsv \
  --warmup-queries data/queries.warmup.jsonl --out runs/ce-train
python -m evaluation.train_ce_gate \
  --capture runs/ce-train/capture.sqlite3 \
  --train-queries data/queries.train.jsonl --train-qrels data/qrels.train.tsv \
  --model-out runs/ce-train/ce40-gate.json --receipt-out runs/ce-train/fit-receipt.json
```

The trainer checks the input digests and the exact set of paired queries.
It skips explicit constraints and empty candidate pools. It standardizes the
four BM25 features (`n_terms`, `vocab_coverage`, `relative_gap`, `candidate_hits`)
and fits balanced logistic regression. A query is positive only if fixed CE40
strictly improves nDCG@10 over RRF; ties and losses are nonpositive. Retain the
receipt with the capture and model: it records input and model digests, service
identities, fit settings, runtime and label counts.

Start the learned profile on a separate port with the frozen JSON and receipt
digest. Keep fixed CE40 available and add fixed CE28 for the four-arm comparison:

```bash
python -m evaluation.ce_http \
  --baseline http://127.0.0.1:8000 --rerank 28 http://127.0.0.1:8028 \
  --rerank 40 http://127.0.0.1:8040 --learned http://127.0.0.1:8041 --measure-random \
  --require-instance-id --require-encoder-runtime \
  --fit-receipt runs/ce-train/fit-receipt.json "$FIT_RECEIPT_SHA256" \
  --queries data/queries.dev.jsonl --qrels data/qrels.dev.tsv \
  --warmup-queries data/queries.warmup.jsonl --out runs/ce-dev
```

Repeat with independent holdout files and a new output directory, leaving the
gate and comparison settings fixed. The learned arm records real HTTP latency
and CE pair counts; report latency by capture epoch after a resume.
`--require-instance-id` checks the server-generated `X-Search-Instance` header
on every identity and search response. A restart with identical configuration
stops the current invocation before committing a mixed query. A new invocation
can resume the journal in a separate latency epoch. Both the fixed-arm summary
and live random summary record the observed service instances by epoch. Legacy
journals can be replayed with their original options.
`--require-encoder-runtime` binds the loaded query encoder identity in all arms.
Use the same option and services when resuming a capture. An existing journal
from a service that did not expose the runtime cannot establish this provenance.
Record the training receipt's SHA256 when freezing the gate. `--fit-receipt`
checks that digest and compares the learned arm with the recorded gate,
retrieval binding and actual training query encoder before admitting the capture.
`matched-random-plan.json` freezes a seeded random assignment at exactly the
observed learned arm's total pair count, using each query's actual fixed CE40
capacity. Its quality results in `summary.json` and its TREC file replay
observed fixed RRF/CE40 rankings. `learned-diagnostics.json` reports paired
quality differences from each fixed CE arm, missed positive gains, applied
losses and the two-action oracle. The oracle uses qrels after capture and
is only an upper reference for these two observed actions.

`--measure-random` separately sends each query to the learned service and
the fixed endpoint selected by the frozen random plan. It counterbalances
request order, checks every observed ranking and CE count against the original
capture, and writes measured wall times to `matched-random-live.json`.
The comparison times the HTTP requests through their body reads; client-side
endpoint selection is outside this timing. The separate SQLite journal can
resume after a restart on new ports with matching service identities. With
this flag omitted, random replay quality is still available, but no random
latency is measured.

### Offline verification of completed CE captures

Use the capture's source revision and Python runtime. Keep its journal and
the original query, warmup and qrels inputs. With the input files in place,
recompute the outputs into a new or empty directory:

```bash
python -m evaluation.ce_http \
  --replay-from runs/ce-holdout --out runs/ce-holdout-replay \
  --queries data/queries.holdout.jsonl --qrels data/qrels.holdout.tsv \
  --warmup-queries data/queries.warmup.jsonl \
  --fit-receipt runs/ce-train/fit-receipt.json "$FIT_RECEIPT_SHA256" \
  --measure-random
```

The command reads the arms and identity requirements from the main journal.
Supply the fit receipt and its recorded SHA256 when the capture used one.
`--measure-random` requires the completed live random journal. A journal
present in the source directory is replayed automatically. Use sealed
journals after the capture process has exited and any SQLite WAL has been
checkpointed. Replay opens those journals as immutable, makes no HTTP
requests and rejects incomplete captures, changed inputs and invalid epochs.

Compare the SHA256 of each regenerated TREC and JSON file against its sealed
digest. This checks the metrics, paired intervals, matched random plan and
latency summaries against the recorded rows. The original service instance
IDs and request timings are retained. Replay does not measure new latency.

### Frozen unsupported-request policy and coverage

The default remains strict: any HTTP error aborts evaluation. For a separately
predeclared capability evaluation, `--unsupported-policy policy.json` allows
only the specific dense-mode rejection of explicit lexical constraints. It
still sends every query to the public `POST /search` endpoint, without changing
query text, parsing, routing, or API semantics. No other HTTP error is ignored.

The policy JSON has exactly these fields (digest placeholders below are a
schema example, not a runnable policy):

```json
{
  "schema_version": 1,
  "queries_sha256": "<SHA256 of original query file bytes>",
  "preflight_parser_sha256": "<parser SHA256 recorded before evaluation>",
  "request": {"mode": "dense", "scorer": "bm25", "expander": "none"},
  "expected_unsupported": {
    "query_ids": ["predeclared-query-id"],
    "status": 400,
    "body": {"error": "dense 模式无法保证显式词法约束，请使用 lexical、hybrid 或 adaptive"}
  }
}
```

Digests must be lowercase 64-character hex strings. Unknown or duplicate JSON
fields, duplicate/absent qids, changed query bytes or records, and mismatched
request parameters are rejected before creating outputs or sending requests.
The parser digest is **preflight provenance only**; this independent evaluator
does not import the parser or claim to verify the server's runtime parser.
The exact observed HTTP status and JSON error body form the behavioral check.
A predeclared qid returning HTTP200, another error, or malformed JSON aborts.

```bash
python3 -m evaluation.evaluate \
  --endpoint http://127.0.0.1:8000/search \
  --queries queries.jsonl --qrels qrels.tsv --mode dense \
  --unsupported-policy policy.json --output-run runs/dense.trec \
  --output-json runs/dense.json --include-per-query
```

Existing output paths are rejected, including sidecars and pending `.tmp`
files. Use fresh paths for each attempt. With `--output-run`, request evidence
defaults to `OUTPUT_RUN.requests.json`; `--output-requests` can choose another
path. A policy requires one of these durable request outputs. Alongside the
snapshot, `OUTPUT_REQUESTS.events.jsonl` is exclusively created as an append-only
journal. Its `START` line stores the ordered query universe and full policy once;
each request appends a sequenced `BEGIN` followed by its terminal `OUTCOME`,
including unexpected errors. Every event is written as one complete JSON line
and flushed. Per-step atomic `INCOMPLETE` snapshots contain only counters,
query-universe SHA256, compact policy identity, the last/inflight outcome, and
journal basename; they never rewrite the growing full history. Serialization
and logical write volume grow linearly with request count.

Only after all requests and metrics finish does the snapshot become `COMPLETE`
and include full query IDs/outcomes and the final journal SHA256. Unexpected
failures leave a partial run and an `INCOMPLETE` snapshot, and CLI exit status
is nonzero; there is no successful summary JSON. Recover history from the
complete journal lines: validate `START`'s universe digest, monotonically
increasing sequence numbers, attempt/qid matching, and each BEGIN/OUTCOME pair.
An unpaired BEGIN means an attempted request with unknown terminal outcome,
never a successful response or an empty ranking. A crash can leave a partial
final journal line; disregard only that trailing incomplete line and preserve
it as crash evidence. A malformed complete line, missing START, or inconsistent
sequence is an invalid receipt. Snapshot replacement can lag a flushed event,
so the journal is authoritative for history; it does not authorize automatic
resumption or treating an interrupted run as complete. Preserve pending `.tmp`
files and start a new attempt with fresh paths. Do not score partial TREC
artifacts as completed experiments. Existing journals are included in stale
output and path-collision checks.

An exactly matching HTTP400 is recorded as `EXPECTED_UNSUPPORTED`, not HTTP
success or fallback. Its ranking is empty, all four ranking metrics are zero,
and routing and success latency are null. The original query remains in the
full evaluation denominator. The `requests` summary and sidecar retain the
policy bytes SHA256, provenance, ordered qids/outcomes, attempted requests,
validated `http_200` responses, expected rejections, unexpected errors, and
coverage. `http_success_coverage` is validated HTTP200 / all input queries;
`protocol_accounted_coverage` is (validated HTTP200 + expected rejection) / all
input queries. The latter is not an HTTP success rate. Existing latency
aggregates use validated successful HTTP200 requests only; rejection latency
is retained separately in request outcomes. Routing rates use only responses
with validated routing, with their query count explicitly reported.

Empty rankings create no TREC rows. To reproduce full-query metrics from a
TREC file, preserve the sidecar's query universe and build selected qrels with
an entry (possibly empty) for every input qid. Standalone scoring otherwise
uses the union of qrels/run IDs and cannot distinguish missing requests from
empty rankings. A secondary common-support comparison must freeze its qid
subset before looking at relevance or outcomes, and filter **both runs and
qrels** to that same subset before using `evaluation.compare`. Report it
alongside full-query metrics and request coverage, not as a replacement for
the full-query result. Never select the subset from whichever requests happened
to succeed or from favorable scores.

An existing run can be scored without a server:

```bash
python3 -m evaluation.evaluate --qrels qrels.tsv --run runs/bm25.trec
```

For a bounded JSONL corpus that fits in memory, normalize and freeze an
external release without committing the data. Supply its actual expected counts:

```bash
python3 -m evaluation.prepare_dataset \
  --corpus data/raw/corpus.jsonl --queries data/raw/queries.jsonl \
  --qrels data/raw/qrels.tsv --out data/duretrieval
```

Ordinary queries retain TSV output. Queries containing tabs, newlines or Unicode
line separators automatically select escaped JSONL; explicit `--query-format tsv`
rejects these inputs. All output files use LF, including on Windows.

For the public RocketQA DuReader-Retrieval baseline archive, use the dedicated
streaming adapter instead of treating the archive as generic JSONL. It writes
`corpus.jsonl`, `queries.dev.jsonl`, `qrels.dev.tsv`, and a SHA256/count manifest;
the output remains an official `dev` split because the archive has no official
test labels. Queries use escaped JSONL rather than TSV so embedded newlines,
tabs, and Unicode line separators round-trip without creating extra query IDs.
Pass `queries.dev.jsonl` directly to the evaluator's `--queries` option.
`--archive-sha256` requires `--archive`; missing archives or mismatched hashes
are rejected before any output files or directory are created:

```bash
python3 -m evaluation.prepare_dureader \
  --root /data/dureader-retrieval-baseline-dataset \
  --out /data/prepared/duretrieval-official-dev \
  --archive /data/dureader-retrieval-baseline-dataset.tar.gz \
  --archive-sha256 <archive-sha256> \
  --expected-queries 2000 --expected-qrels 9839
```

After corpus validation, split just the prepared queries and qrels without
loading or copying the 8M-document corpus. This uses the same SHA256 seed/query-ID
assignment as the normalizer and reproduces the six frozen internal-split files:

```bash
python3 -m evaluation.prepare_dataset --queries-only \
  --queries /data/prepared/duretrieval-official-dev/queries.dev.jsonl \
  --qrels /data/prepared/duretrieval-official-dev/qrels.dev.tsv \
  --out /data/prepared/official-internal-splits \
  --seed 20261028 --dev-fraction 0.5 --test-fraction 0.4 \
  --expected-queries 2000 --expected-qrels 9839
```

This entry validates query IDs, qrel query references, input counts and source
hashes, then publishes a fresh directory. It does not validate corpus membership;
its manifest explicitly records that boundary. Run the corpus adapter first.

Compare two runs with a deterministic query-level paired bootstrap. The
reported difference is `run-b - run-a`:

```bash
python3 -m evaluation.compare --qrels data/duretrieval/qrels.test.tsv \
  --run-a runs/hybrid.trec --run-b runs/adaptive.trec \
  --metric ndcg --samples 10000 --seed 2026
```

Install the test dependencies (a CPU PyTorch build is sufficient), then run
pytest to collect both unittest classes and function-based evaluation tests:

```bash
python3 -m pip install -r requirements-router-research.txt pytest torch ijson==3.4.0.post0
python3 -m pytest evaluation
python3 -m unittest discover -s evaluation -p 'test_*.py'
```

Install `bm25s==0.2.14` as well to exercise the optional external reference.

## External TREC runs and the DuReader 100k development subset

The shared [DuReader 100k rebuild recipe](https://huggingface.co/datasets/ShwStone/dureader-mini-100k/tree/2db5eadeb4adabec463324b4ca3562b7b79fda94)
pins 100,000 documents and 172 train / 1,020 development queries from the
public official-dev release. Use its published hashes to verify the rebuilt
files. These development results use a different candidate corpus from the
full-corpus 808-query holdout.

External baselines can generate a six-column TREC file in a separate
environment. The validator checks one run tag, unique documents and
contiguous ranks per query, corpus membership and minimum depth. It scores
every query in the supplied order, with zero metrics for queries without qrels.
Latency and routing are unavailable from a TREC run alone. Use the hashes in the
receipt to check that each compared run used the same frozen input bytes.

```bash
python -m evaluation.external_run \
  --corpus data/mini100k/corpus.jsonl \
  --queries data/mini100k/queries.dev.jsonl --qrels data/mini100k/qrels.dev.tsv \
  --run runs/reference.trec --depth 100 --output-json runs/reference.validation.json
```

A method that legitimately yields fewer than 100 documents needs a reviewed
`--exceptions` JSON file. Freeze the eligibility rule before inspecting
quality metrics and verify the exact counts independently of the submitted run.
An explicit-constraint rejection can be accounted for with an expected count of
zero. The evaluator keeps the query in the denominator. An unexplained missing
or truncated query fails validation.

```json
{
  "schema_version": 1,
  "corpus_sha256": "<frozen corpus SHA256>",
  "queries_sha256": "<frozen queries SHA256>",
  "qrels_sha256": "<frozen qrels SHA256>",
  "depth": 100,
  "short_queries": {
    "<frozen query ID>": {"expected_count": 0, "reason": "<reviewed reason>"}
  }
}
```

The optional formula-aligned BM25S reference scores document body tokens using
the shared analyzer and `bm25s==0.2.14` with ATIRE term-frequency saturation,
Lucene IDF, `k1=1.2`, `b=0.75` and float64. Corpus order breaks score ties.
Queries with Boolean or phrase constraints use the project's positional filter
on a separately built lexical index whose ordered corpus fingerprint must match.
BM25S supplies all scores. The reference does not use qrels while retrieving.

```bash
python -m pip install bm25s==0.2.14
python -m evaluation.bm25s_reference \
  --corpus data/mini100k/corpus.jsonl --n-docs 100000 \
  --corpus-sha256 f24b50b6012b47f6fb5f72272666dbf187124a869f19d1c8d83f7e30d6cb62b4 \
  --queries data/mini100k/queries.dev.jsonl --depth 100 \
  --constraint-index artifacts/mini100k/compact --out runs/bm25s-mini100k
```

The output contains `run.trec` and a manifest with hashes, versions and
observed short rankings. Use `evaluation.external_run` to validate and score
the TREC file. Observed short counts are diagnostic until independently checked.

### Exact dense reference

The exact reference uses the same frozen BGE model and verified FP32 document
cache as the Qdrant arm. It evaluates every document with float64 cosine,
breaking score ties by corpus order. The paired comparison measures agreement
between exact search and ANN on the shared embeddings. Record the exact and ANN
backends separately when reporting results.

```bash
python -m evaluation.dense_reference \
  --corpus data/mini100k/corpus.jsonl --n-docs 100000 \
  --corpus-sha256 f24b50b6012b47f6fb5f72272666dbf187124a869f19d1c8d83f7e30d6cb62b4 \
  --queries data/mini100k/queries.dev.jsonl --qrels data/mini100k/qrels.dev.tsv \
  --exceptions data/mini100k/dense-depth-exceptions.json \
  --cache-dir artifacts/mini100k/bge-cache --model-path models/bge-small-zh-v1.5 \
  --chunk-size 1024 --batch-size 32 --shards 1 --depth 100 --out runs/bge-exact-mini100k
```

The default backend is NumPy. Install `faiss-cpu==1.10.0` and add
`--backend faiss` to run the independent FAISS CPU reference on the same
embeddings. FAISS uses a flat inner-product index with L2-normalized float32
inputs and resolves exact score ties in corpus order, including ties across
the requested cutoff. Query encoding runs in a separate process so Windows can load each backend's
OpenMP library independently. The manifest identifies the backend, arithmetic, and
package version. The [FAISS metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
describes the equivalence of normalized inner product and cosine similarity.

To compare against a sealed historical NumPy run, also pass
`--reference-manifest /path/to/manifest.json <verified-SHA256>`. This admits
the historical cache contract only if the other cache fields and the frozen
query inputs match. The document and query vector SHA256 values must match
before the FAISS ranking is published.

Pass the same cache partition parameters used by the producer. When the frozen
query set includes explicit lexical constraints, the exceptions file must list
exactly those queries with `expected_count: 0`. Keep their query IDs in the
evaluation denominator. Run `evaluation.external_run` on the emitted TREC
file with the same exceptions. The qrels bind the policy by SHA256 and do not
participate in ranking.

### Matched HTTP runs on the shared development corpus

Point the serving profile at the 100k compact lexical index and the matching
Qdrant snapshot. Use the same verified BGE cache for the snapshot and the exact
reference above. The development protocol uses BGE
`BAAI/bge-small-zh-v1.5@7999e1d3359715c523056ef9478215996d62a620`,
512-dimensional FP32 vectors, `candidate_k=200`, `rrf_c=60`,
`hnsw_ef=256`, and the built-in `rule` policy. Verify the corpus fingerprint,
embedding, search parameters, collection count and HNSW readiness before and
after each arm. Record an immutable copy of the service log for each arm.
The [serving guide](../docs/online-serving.md) describes the profile and log
formats.

For each of `lexical`, `dense`, `hybrid` and `adaptive`, use a fresh output
path and unique request tag. This example shows the dense arm, for which the
frozen HTTP policy records one unsupported explicit-constraint query:

```bash
python -m evaluation.evaluate --endpoint http://127.0.0.1:8000/search \
  --queries data/mini100k/queries.dev.jsonl --qrels data/mini100k/qrels.dev.tsv \
  --mode dense --scorer bm25 --expander none --k 10 --map-k 100 \
  --unsupported-policy data/mini100k/dense-http-unsupported.json \
  --include-per-query --fail-on-fallback --timeout 30 \
  --request-tag mini100k-dense-attempt-001 --tag mini100k-dense \
  --output-run runs/mini100k/dense.trec --output-json runs/mini100k/dense.json

python -m evaluation.serving_cost \
  --run runs/mini100k/dense.trec.requests.json runs/mini100k/dense.server.log
```

Omit `--unsupported-policy` in the other three arms. Keep the constraint query
in all quality denominators; it contributes an empty ranking to dense and
exact dense. Confirm that hybrid and adaptive preserve its lexical filter.
For ANN agreement, compare the exact and ANN document sets at depths 10 and
100 over the remaining 1,019 queries. Run paired bootstrap with all 1,020
queries for quality and report the direction of each subtraction.

Measure HTTP latency in a separate warm run with one client request at a time,
the same frozen query sample for all modes, and an interleaved mode order.
Keep the per-request wall-clock observations and their matching server log
rows. Time measured inside the server and wall time measured by the client
have different boundaries. The exact reference is an offline run, so it has
no comparable HTTP latency.

To reconstruct the 100k cost schedule, exclude the declared constraint ID,
sort the remaining IDs by the bytes of SHA-256 of UTF-8 `2026:<qid>`, and
take the first 96. For each zero-based repeat `r = 0, 1, 2`, copy that list
and shuffle it with Python `random.Random(2026 + r)`. At zero-based position
`i` in the shuffled list, request the mode indices
`[(j + i + r) % 4 for j in (0, 1, 3, 2)]` from
`[lexical, dense, hybrid, adaptive]`, at depth 100. Before measuring, warm up
each mode with the first ten eligible training queries. Average each
query's three repeats before resampling paired latency differences.

### Paired live HTTP measurements

For a new paired measurement, freeze a JSON plan with these fields:

| Field | Value |
| --- | --- |
| `schema_version` | `2` for new runtime-bound measurements. Use `1` to verify historical journals under their original identity rules. |
| `queries_sha256`, `corpus_sha256` | SHA-256 of the exact query file and the expected corpus identity from `/index/info` |
| `query_ids`, `warmup_query_ids` | Ordered, disjoint ID arrays from that file. Preselect the measured queries. Include warmup query records in the same file. |
| `repeats`, `depth`, `timeout_s` | Positive repeat count, retrieval depth and request timeout in seconds. Use depth 100 for MAP@100. |
| `arms` | Ordered objects with `label`, HTTP(S) origin `endpoint`, `mode`, and expected `policy_version`. Set optional `identity_sha256` to pin the full serving identity for that arm. |

Use one vector-ready endpoint with four modes for a mode comparison, or two
endpoints with `adaptive` and different policy versions for a Router comparison.
Both endpoints must expose the same index, vector and effective search configuration.
Version 2 also binds the query encoder implementation, loaded device, weight dtype
and available verified model inventory to both the HTTP identity and the server
startup event. CPU and CUDA arms have different runtime identities.
Record hardware models and load separately when interpreting latency.
For a strict rerun, copy each arm's `identity_sha256` from the prior verified
report from the same schema version. Version 1 digests cover the index, vector,
search configuration and policy objects. Version 2 adds the encoder runtime.
For a policy that can route adaptive requests to dense retrieval, set
`allow_adaptive_dense: true` on the adaptive arm and pin its `identity_sha256`.
The paired verifier checks the selected route in both the response and the
matching server log. For a standalone HTTP evaluation, use
`--mode adaptive --allow-adaptive-dense`.
Requests use BM25 and expansion `none`. When a dense arm is present, choose a
panel it can execute successfully and record any exclusions when freezing the plan.
Within each repeat the runner walks the listed query IDs and rotates which arm goes
first after each query. Freeze the query order before inspecting results.

```bash
mkdir -p runs
python -m evaluation.paired_http collect --plan paired.json \
  --queries panel.jsonl --out runs/router-paired
python -m evaluation.paired_http verify --plan paired.json \
  --queries panel.jsonl --journal runs/router-paired/requests.jsonl \
  --server-log rules=rules.server.log --server-log gain=gain.server.log \
  --run-dir runs/router-paired-trec > runs/router-paired-verified.json
python -m evaluation.compare --qrels panel.qrels.tsv \
  --run-a runs/router-paired-trec/rules.trec \
  --run-b runs/router-paired-trec/gain.trec --metric ndcg --k 10 --map-k 100
```

Supply the same server log path for labels served by the same endpoint. Keep each
server's startup event and request logs. The verifier checks the entire schedule,
including warmup, against unique request tags, the startup identity and the
client's routing record. A failed request, fallback, changed identity or incomplete
journal prevents acceptance. Preserve an interrupted journal and collect again into
a fresh directory so the latency sample covers one measurement session.

The report includes client and server timings, vector attempts and successes,
and 95% paired intervals for right arm minus left arm. Repeats are averaged
per query before bootstrap sampling. TREC output uses the first repeat and records
rank-derived scores. Use qrels restricted to the plan's measured IDs, including
queries with empty results. The historical 100k schedule above remains the
reference for the published measurement.

## Human review of a frozen candidate pool

Use the frozen corpus, query split and candidate pool already admitted for the
experiment. Export all pairs, including any source-anchor pairs that were
graded separately in a previous run. Each input JSONL row requires
`query_id`, `query`, `doc_id`, `title`, `text`, `document_sha256` and
`query_sha256`. Hash the query as UTF-8 and the document as the UTF-8 bytes of
the compact JSON array `[doc_id,title,text]` with unescaped Unicode.

```bash
python -m evaluation.annotation_packet \
  --candidates data/frozen/blinded-pairs.jsonl \
  --candidates data/frozen/seed-candidates.jsonl \
  --packet data/review/reviewer.jsonl --private-map data/review/private/mapping.json
```

Keep both outputs outside Git. Share only `reviewer.jsonl` with reviewers.
Each row contains `pair_id`, `query`, `title` and `text`; keep pool-system
membership, original IDs, seed roles and old judgments with the coordinator.
Use relevance grades 0 (irrelevant), 1 (partly relevant) and 2 (directly
relevant), and record each grade with its opaque `pair_id`. Assign a subset to
independent reviewers, resolve disagreements and preserve the adjudication
before deriving qrels. Exporting a packet records candidate coverage and does
not assign labels.

Freeze the reviewer assignments and record the plan SHA256 before collecting
grades. A version 1 plan contains `schema_version`, `packet_sha256`,
`private_map_sha256` and `reviewers`. The last field maps pseudonymous
reviewer IDs to their exact assigned `pair_id` lists. Assign every pair to at
least one reviewer and include an independently reviewed overlap.

Once every reviewer has submitted their complete assignment, prepare the
adjudication materials against the same frozen inputs:

```bash
python -m evaluation.annotation_intake \
  --packet data/review/reviewer.jsonl --private-map data/review/private/mapping.json \
  --plan data/review/private/plan.json --plan-sha256 "$FROZEN_PLAN_SHA256" \
  --reviewer reviewer-a data/review/private/reviewer-a.jsonl \
  --reviewer reviewer-b data/review/private/reviewer-b.jsonl \
  --prepare-adjudication data/review/disputes.jsonl data/review/adjudication-template.jsonl \
    data/review/private/pre-adjudication.json
```

Use fresh paths for all three outputs. The private report binds the input and
output hashes and records the number of assigned pairs, independently reviewed
pairs, unanimous overlaps and disputes. The agreement rate divides unanimous
overlaps by all overlaps. When disputes exist, give the adjudicator only the
dispute packet and matching template. The packet retains the original blind
order and contains `pair_id`, `query`, `title` and `text` for disputed pairs.
Copy the template to a new file, replace every `null` grade with an integer
`0/1/2`, and keep the completed file private. When every overlap is unanimous,
the command writes the private report alone. Record the private report's SHA256
before sending the adjudication materials. Use it at final import to verify
the exact reviewer submissions and the reconstructed dispute material.

Import the completed judgments to create qrels:

```bash
python -m evaluation.annotation_intake \
  --packet data/review/reviewer.jsonl --private-map data/review/private/mapping.json \
  --plan data/review/private/plan.json --plan-sha256 "$FROZEN_PLAN_SHA256" \
  --reviewer reviewer-a data/review/private/reviewer-a.jsonl \
  --reviewer reviewer-b data/review/private/reviewer-b.jsonl \
  --pre-adjudication data/review/private/pre-adjudication.json "$TRUSTED_PRE_ADJUDICATION_SHA256" \
  --adjudication data/review/private/adjudication.jsonl \
  --qrels data/review/news-human.qrels.tsv --receipt data/review/private/intake.json
```

Each submission row contains only `pair_id` and an integer `relevance` in
`0/1/2`. Provide adjudication for exactly the pairs whose grades disagree.
Omit `--adjudication` when the independent submissions agree on every
overlapping pair. Keep the reviewer inputs, private map and receipt outside Git.

## Simple bounded crawler

The legacy `evaluation.crawl_web` entry rejects all HTTP redirects before any
redirected request, including same-origin and robots redirects. Robots and page
fetches share the configured socket timeout, byte limit, User-Agent and minimum
request-start interval; errors, non-HTML responses and leaves do not bypass pacing.
Robots failures are denied unless explicitly opted into `allow_robots_failure`;
HTTP 401/403 still deny and the existing 404 policy is retained. This is not a
complete SSRF sandbox. Use restricted egress for untrusted targets and prefer the
persistent workflows below for formal collection.

The merge hardening changes `crawl_web.py`'s source hash, which participates in
persistent crawl bindings even though its HTML parser is unchanged. Resume an
existing state only with its originally recorded commit/source archive, not the
new code. Do not relax source-identity checks or alter frozen corpus/index IDs.
See [receiver acceptance and fixes](../docs/contributions/rfc0004-receiver-acceptance.md).

## Durable collection of unlabelled web candidates

For a long crawl, use the persistent FIFO workflow rather than keeping the
frontier in process memory. It enforces same-host HTTP(S), fail-closed robots,
redirect validation before the redirected request, and pacing for every request
including robots, failed fetches and leaves. Site crawl-delay/request-rate may
increase the configured interval.

```bash
python -m evaluation.crawl_resume --state-dir /data/news-crawl \
  --seed https://www.news.cn/ --target-documents 20000 --max-depth 6 \
  --max-attempts 100000 --delay-seconds 0.5 --timeout 8 --max-bytes 1000000
```

Repeat exactly the same command with the same frozen code after interruption.
SQLite is authoritative; completed pages retain their IDs, interrupted attempts
remain charged, and an OS file lock rejects concurrent writers. The binding
includes seeds, settings, both crawler and cleaner source hashes, and robots
policy. A temporary robots fetch failure produces `INTERRUPTED` so the page can
be retried after connectivity returns; it is not treated as a permanent denial.
`corpus.jsonl` and `receipt.json` are atomic exported snapshots. If a hard stop
occurs between their replacements, re-export from SQLite on recovery rather
than treating the two files as a joint transaction.

The 20k target counts raw, URL-distinct candidate pages. It does not establish
20k cleaned, deduplicated, searchable documents. These candidates still require
the RFC-0002 cleaning/source review, frozen IDs, query design and pooled
judgments. This command creates no qrels or retrieval-quality claims.

## Cleaned article collection

For the new Xinhua article version, count accepted unique articles directly:

```bash
python -m evaluation.crawl_articles --state-dir /data/news-clean \
  --seed https://www.news.cn/ --target-documents 20000 --max-depth 6 \
  --max-attempts 100000 --delay-seconds 1 --timeout 8 --max-bytes 1000000
python -m unittest evaluation.test_clean_news evaluation.test_crawl_resume \
  evaluation.test_crawl_articles -v
```

The frozen `news-cn-body-v1` rule accepts article `/c.html` URLs on `www.news.cn`,
extracts `#detailContent` (fallback `#detail`), removes specified template/editor
nodes and requires at least 80 nonspace body characters, 40 CJK characters and
no more than 1% replacement characters. Navigation, filtered and duplicate pages
still discover links; they do not consume accepted document IDs. Exact dedup uses
NFKC/whitespace-normalized body SHA256, and near dedup requires SimHash64 Hamming
distance <=3 plus length ratio >=0.9. These rules never consult qrels.

Raw response bytes are stored as SHA-addressed deterministic gzip before decoding;
sidecars distinguish that SHA from decoded UTF-8 HTML and cleaned-body identities.
The corpus keeps only `doc_id/title/text`; `article_sidecar.jsonl` records source,
time, selector, filtering/duplicate reasons and capture identity. Reuse exactly
the same command and frozen code to resume. Optional `--seed-file` takes JSONL
URLs, binds its bytes SHA and treats imported seeds as depth 0 in a new version.
A raw crawl state cannot be silently upgraded to this policy.

SQLite remains authoritative. Corpus, sidecar and receipt are separate atomic
exports; readers must check both hashes/counts from a matching extended receipt
and refuse an incomplete/intermediate combination. Freeze only a completed,
reviewed export before indexing and qrel pooling. Public availability and robots
permission do not establish redistribution rights; full text/caches stay outside
Git. The collector creates no queries, labels or quality scores.
