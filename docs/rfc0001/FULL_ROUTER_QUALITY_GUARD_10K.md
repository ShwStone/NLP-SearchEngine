# Full-corpus quality-first Router after 10k-query training expansion

This is the first full-corpus Router variant with a statistically positive holdout result. It is a quality-first policy and has an explicit cost increase.

## Frozen protocol

- Corpus: official DuReader full corpus, 8,096,668 documents.
- Training: 10,000 legal official DuReader training queries with exact corpus-positive passage matches.
- Evaluation: 808 official-dev queries held out from all training and model selection.
- Candidate retrieval: BM25 compact reader, BGE-small dense HNSW `ef=128`, equal-RRF.
- CE actions: independently scored CE-28, CE-40; fixed CE-33 is the reference.
- Features: full CE-28 score profile, adjacent score gaps, lexical/dense rank statistics, candidate overlap, and query statistics.
- Gate: ExtraTrees classifier trained to predict whether CE-40 improves nDCG over CE-28.
- Budget policy: route the top 50% eligible queries by classifier score to CE-40; route the rest through fixed CE-33.
- Model selection: five-fold grouped OOF on the 10k training queries. The 50% heavy rate was selected by the predeclared quality-first OOF policy grid; no holdout labels were used.

## Independent holdout result

Compared with fixed CE-33 on all 808 holdout queries:

| Metric | Delta | Grouped bootstrap 95% CI |
| --- | ---: | ---: |
| nDCG@10 | `+0.00686` | `[+0.00222, +0.01210]` |
| Recall@10 | `+0.01040` | `[+0.00562, +0.01571]` |

The Router uses approximately `36.49` mean CE pairs versus `33` for fixed CE-33, an estimated `+10.6%` CE-pair budget. The result is therefore a quality improvement with an explicit cost increase, not a same-cost Pareto claim.

## Decision

This establishes that Router decisions can have a meaningful positive effect on the full corpus when the objective is quality-first budget allocation and the training set is expanded. It should replace the full-corpus all-fixed fallback only after a real HTTP benchmark confirms the same action decisions and measures the latency increase. The earlier same-cost Router remains unpromoted.
