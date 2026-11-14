# DuReader 100k v1.0 baseline

This report uses the frozen `ShwStone/dureader-mini-100k` v1.0 recipe and the official DuReader archive. The corpus contains 100,000 documents, 1,020 internal-dev queries, and 4,960 qrels. The index and vector manifest are generated from the frozen corpus; qrels are read only after the indexes are frozen.

Run the reproducible evaluator with the repository virtual environment:

```bash
.venv/bin/python -u evaluation/evaluate_dureader100k_baselines.py \
  --index-root /path/to/dureader-100k-index-v1 \
  --data-root /path/to/dureader-mini-100k-v1.0 \
  --model-path .runtime/models/bge-small-zh-v1.5 \
  --qdrant-binary .runtime/bin/qdrant
```

Dense is skipped for the one query containing an explicit boolean/phrase constraint, as required by the parser contract. All other variants run on all 1,020 queries. No per-query result file is written by the evaluator.

| variant | nDCG@10 | Recall@10 | MRR | MAP@100 | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| lexical/BM25 | 0.13396 | 0.15712 | 0.21732 | 0.13183 | 51.35 | 102.31 |
| dense/BGE-small | 0.26476 | 0.29083 | 0.41025 | 0.24778 | 42.94 | 57.43 |
| equal RRF hybrid | 0.22787 | 0.26479 | 0.34746 | 0.20741 | 67.26 | 117.27 |
| existing adaptive Router (rules-v1) | 0.20464 | 0.24055 | 0.31686 | 0.18630 | 60.02 | 114.77 |
| repaired adaptive Router (hybrid-v1) | 0.22768 | 0.26430 | 0.34734 | 0.20737 | 70.77 | 126.19 |

The existing `rules-v1` Router is a reproducible quality regression against equal RRF on this frozen domain (`-10.2%` relative nDCG). We repaired the frozen policy selection by evaluating the existing `hybrid-v1` action: it raises nDCG by 11.3% relative to `rules-v1` and is within 0.08% of equal-RRF quality, while preserving lexical handling for explicit constraints. It costs more than rules-v1, so this is a quality-first Router configuration rather than a latency win. The dense result is a baseline, not a claim that dense is safe for explicit constraints.

Input hashes and the full machine-readable result are in `dureader-100k-baseline-summary.json`.
