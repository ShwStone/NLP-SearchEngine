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

Standalone TREC runs contain no routing or latency evidence: those fields are
`null`, not zero failures or zero latency.

An existing run can be scored without a server:

```bash
python3 -m evaluation.evaluate --qrels qrels.tsv --run runs/bm25.trec
```

Before a full experiment, normalize and freeze an externally downloaded
DuRetrieval release without committing the data:

```bash
python3 -m evaluation.prepare_dataset \
  --corpus data/raw/corpus.jsonl --queries data/raw/queries.jsonl \
  --qrels data/raw/qrels.tsv --out data/duretrieval \
  --expected-docs 100001 --expected-queries 2000 --expected-qrels 9839
```

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

Compare two runs with a deterministic query-level paired bootstrap. The
reported difference is `run-b - run-a`:

```bash
python3 -m evaluation.compare --qrels data/duretrieval/qrels.test.tsv \
  --run-a runs/hybrid.trec --run-b runs/adaptive.trec \
  --metric ndcg --samples 10000 --seed 2026
```

Run the evaluator tests with:

```bash
python3 -m unittest discover -s evaluation -p 'test_*.py'
```

## Bounded web test-set workflow

`evaluation.crawl_web` is a bounded corpus constructor for the teacher-requested
network test set. It only accepts HTTP(S), normalizes URLs, rejects credentials,
uses same-domain breadth-first traversal by default, checks `robots.txt`, and
limits depth, document count, response bytes, timeout, and request spacing.
It writes only page IDs, titles, text, and URLs; it does not infer relevance.

```bash
python3 -m evaluation.crawl_web crawl \
  --seed https://example.org/ \
  --out-corpus data/crawl/corpus.jsonl \
  --annotation-template data/crawl/annotations.template.jsonl
```

After a human annotator fills `query` and `relevance` (0/1/2) in a copy of the
template, finalize it with:

```bash
python3 -m evaluation.crawl_web finalize \
  --corpus data/crawl/corpus.jsonl \
  --annotations data/crawl/annotations.jsonl \
  --queries-out data/crawl/queries.jsonl \
  --qrels-out data/crawl/qrels.tsv
```

Finalization rejects unknown document IDs, duplicate judgments, empty queries,
and queries without a positive judgment. The generated files are evaluation
inputs and remain outside version control. This workflow is separate from the
official DuReader dev resources and must not be mixed with their IDs or qrels.
