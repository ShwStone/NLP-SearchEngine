# DuReader 100k CE and learned Router experiment

This stage uses only the frozen 100k train split for fitting and the frozen internal-dev split for the final report. Candidate capture is Top-200 lexical/dense/hybrid with the same BGE-small/Qdrant manifest as the baseline. The BGE reranker is the pinned local `bge-reranker-base` inventory; no dev labels are used to fit the ranker or Router.

The raw CE action reranks the first 28 or 40 hybrid candidates and leaves the tail in first-stage order. `CE-0` is equal-RRF with no reranking. The learned ranker is a train-only ridge model over CE score and candidate position. The learned Router is a train-only logistic model over query-side features (term count, lexical candidate count, dense overlap, and lexical score gap); it decides whether to spend 40 CE pairs.

| action | nDCG@10 | Recall@10 | MRR | MAP@100 | CE p50/p95 (ms) |
|---|---:|---:|---:|---:|---:|
| CE-0 | 0.22787 | 0.26479 | 0.34746 | 0.20741 | 0 / 0 |
| CE-28 raw | 0.40900 | 0.42592 | 0.58177 | 0.34723 | 38.01 / 48.36 |
| CE-40 raw | 0.48548 | 0.52831 | 0.63583 | 0.41379 | 49.00 / 60.90 |
| CE-28 + learned ranker | 0.40331 | 0.42303 | 0.57604 | 0.34125 | 49.04 / 60.22 |
| CE-40 + learned ranker | 0.51276 | 0.56261 | 0.65878 | 0.42935 | 49.04 / 60.22 |
| learned Router (CE-40 action) | 0.39438 | 0.43529 | 0.53826 | 0.33819 | 60.7% heavy |

The CE-40 raw action improves nDCG by 0.25761 over equal RRF (paired 95% CI [0.24238, 0.27312]). The train-only learned ranker adds another 0.02728 nDCG (paired 95% CI [0.01780, 0.03676]). The learned Router improves nDCG by 0.16650 over equal RRF (paired 95% CI [0.15174, 0.18110]) at a 60.7% heavy rate in the final no-position-feature rerun. Its cost and quality must still be qualified through the production HTTP harness before being treated as an online acceptance result.

Run candidate capture with `evaluation/capture_dureader100k.py`, then train/evaluate with `evaluation/train_eval_ce_router_dureader100k.py`. The machine-readable result, input hashes and split protocol are in `dureader-100k-ce-router-summary.json`. Large per-query captures remain local and are intentionally excluded from the PR.

## Loopback HTTP confirmation

The same frozen BM25/Qdrant backend and reranker were exercised through a real loopback HTTP server. Each request performs hybrid retrieval, CE scoring, JSON serialization and response parsing. Across all 1,020 dev queries:

| action | nDCG@10 | CE p50/p95 ms | end-to-end p50/p95 ms |
|---|---:|---:|---:|
| CE-0 | 0.22787 | 0 / 0 | 77.45 / 128.93 |
| CE-28 | 0.40900 | 43.31 / 53.29 | 115.64 / 170.93 |
| CE-40 | 0.48548 | 55.41 / 67.68 | 130.12 / 185.54 |

CE-40’s paired nDCG gain over CE-0 is 0.25761 (95% CI [0.24238, 0.27312]). The HTTP harness is reproducible with `evaluation/benchmark_dureader100k_ce_http.py`; its aggregate output is included in `dureader-100k-ce-router-summary.json`. The learned Router still requires an equivalent HTTP run with its train-only model bound into the serving path before it can be called an online acceptance result.

## Same-budget controls

At the learned Router's target heavy rate (60.7%), fixed and random controls were evaluated with the same CE-40 action. Fixed CE-28 and fixed CE-40 are useful cost-quality endpoints; two deterministic random policies match the budget within sampling error.

| policy | nDCG@10 | Recall@10 | heavy rate | paired nDCG delta vs hybrid (95% CI) |
|---|---:|---:|---:|---:|
| fixed CE-28 | 0.40899 | 0.42592 | 100% | +0.18111 [0.16717, 0.19525] |
| fixed CE-40 | 0.48548 | 0.52831 | 100% | +0.25761 [0.24238, 0.27312] |
| random budget v1 | 0.38563 | 0.43060 | 59.5% | +0.15776 [0.14249, 0.17225] |
| random budget v2 | 0.38549 | 0.42256 | 59.8% | +0.15761 [0.14339, 0.17161] |
| learned Router | 0.39438 | 0.43529 | 60.7% | +0.16650 [0.15174, 0.18110] |

The learned Router exceeds both random same-budget controls on nDCG, but its margin is an offline result; the serving binding still needs a separate HTTP run. Use `evaluation/evaluate_dureader100k_budget_controls.py` to reproduce the controls.

## Router HTTP confirmation

The repaired quality-first policy was also exercised through the real HTTP serving path. On the same 1,020 dev queries, `hybrid-v1` improves nDCG over the old `rules-v1` by 0.02305 (paired 95% CI [0.01776, 0.02895]).

| policy | nDCG@10 | Recall@10 | vector attempt rate | e2e p50/p95 ms |
|---|---:|---:|---:|---:|
| rules-v1 | 0.20464 | 0.24055 | 73.3% | 65.51 / 119.27 |
| repaired hybrid-v1 | 0.22768 | 0.26430 | 99.9% | 68.85 / 120.95 |

The serving harness is `evaluation/benchmark_dureader100k_router_http.py`; aggregate output is included in `dureader-100k-ce-router-summary.json`. This confirms the policy repair over HTTP, while the separate train-only learned Router remains an offline experimental model.

## Learned Router HTTP confirmation

The train-only learned Router was bound to a loopback HTTP handler and evaluated on all 1,020 dev queries. The handler computes runtime lexical features, performs hybrid retrieval, and conditionally runs CE-40. Training used only the 172 train queries.

- nDCG@10: `0.36810`
- Recall@10: `0.41090`
- heavy rate: `53.0%`
- e2e p50/p95: `122.35/217.05 ms`
- CE p50/p95 on heavy requests: `54.35/64.65 ms`
- paired nDCG delta vs equal-RRF: `+0.14023`, 95% CI `[0.12557, 0.15345]`

This is the production-path confirmation for the train-only model. It is intentionally reported separately from the earlier capture-based learned Router because runtime feature extraction changes the decision distribution. Reproduce with `evaluation/benchmark_dureader100k_learned_router_http.py`.
