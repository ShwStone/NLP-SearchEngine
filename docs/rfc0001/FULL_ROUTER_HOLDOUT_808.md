# Full-corpus Router: 808-query holdout audit

This audit extends the training universe from the original 172-query train split to the frozen 1,020-query internal-dev split, then evaluates only on the 808 remaining official-dev queries. The 808 queries were never used to fit or select a model. All candidates were regenerated against the full 8,096,668-document corpus with the frozen BM25 compact reader, BGE-small dense index (`ef=128`), equal-RRF fusion, and independently scored CE budgets 28/40/60/80/100.

## Protocol

- Training: 1,020 internal-dev queries and their official answer paragraphs.
- Holdout: 808 remaining official-dev queries and answer paragraphs.
- Router features: cheap CE-28 score confidence, lexical/dense score statistics, candidate overlap, and query statistics.
- Actions: CE-28/40, CE-28/60, CE-28/80, CE-28/100, and CE-0/60/80/100 probes.
- Target budget: 33.2 mean CE pairs, matching fixed CE-33.
- Model selection: grouped five-fold training-only selection among ridge, ExtraTrees, RandomForest, and HistGradientBoosting.
- Holdout was read only after model selection.

## Results

| Action pair | Selected model | Holdout nDCG delta vs fixed CE-33 | Holdout Recall delta |
| --- | --- | ---: | ---: |
| CE-28/40 | ExtraTrees | -0.00045 | -0.00171 |
| CE-28/60 | Ridge | -0.00107 | -0.00132 |
| CE-28/80 | HistGradientBoosting | -0.00152 | -0.00427 |
| CE-28/100 | RandomForest | +0.00148 | -0.00061 |
| CE-0/60 | Ridge | -0.03102 | -0.01886 |

The CE-28/100 positive nDCG result is not statistically reliable: grouped bootstrap on the 808 holdout gives nDCG delta `+0.00148`, 95% CI `[-0.00437, +0.00711]`; Recall delta `-0.00061`, 95% CI `[-0.00783, +0.00667]`. Its average budget is `33.17` pairs, so the cost matching is correct, but the quality evidence does not pass the promotion gate.

## Decision

The larger training set and deeper CE actions did not produce a statistically credible Router improvement on the independent full-corpus holdout. The current quality guard and fixed CE-33 remain the only defensible full-corpus default. A future promotion requires either more legal training queries or stronger decision-time signals that improve both nDCG and Recall under the same budget.

A BGE-small query embedding gate and small MLP/Ridge variants were also trained on the 1,020-query training universe and evaluated on the same untouched 808-query holdout. They all regressed against fixed CE-33 (best Ridge nDCG delta `-0.00234`, Recall delta `-0.00514`), so semantic query embeddings alone do not repair the budget decision.

## 5,000-query training expansion

A further legal training expansion used 5,000 positive cross-encoder training queries from the official DuReader training file; 4,960 had exact passage identity matches in the full corpus and were scored with full BM25/dense/CE-28/40/60/80/100 captures. Models were fit only on these 4,960 rows and evaluated on the same untouched 808-query holdout.

The best exploratory result was a KNN CE-28/40 selector at the target 33.2-pair budget: nDCG delta `+0.00190`, Recall delta `+0.00142`, with grouped bootstrap CIs `[-0.00302,+0.00700]` and `[-0.00465,+0.00775]`. Training-only nested model selection selected a conservative histogram model whose holdout nDCG delta was only `+0.00089` and whose CI was not positive. CE-28/60 and CE-28/100 selectors likewise lacked a reliable joint gain. Adding the original 172 queries to the 4,960-query set did not change this conclusion.

The larger training set improves point estimates but does not yet produce a statistically credible full-corpus Router advantage. The result remains research evidence and is not promoted to the PR.

A classification gate trained on the 4,960-query expansion (label: whether CE-40 improves nDCG over CE-28) gives the strongest point result so far: at 33.2 mean pairs, holdout nDCG delta `+0.00473` and Recall delta `+0.00451`. However grouped bootstrap CIs remain `[-0.00073,+0.01050]` and `[-0.00190,+0.01141]`; the training-only OOF policy delta is only `+0.00127`. Combining the original 172 queries with the expansion reduces the holdout delta to `+0.00136` nDCG and `-0.00038` Recall. The 4,960-query classifier is therefore promising but not yet statistically sufficient for promotion.

## 10,000-query expansion

A second disjoint 5,000-query official training batch was captured and scored, giving 10,000 training queries total. The ExtraTrees positivity classifier for CE-40 over CE-28 retained a positive holdout point estimate: nDCG delta `+0.00382`, Recall delta `+0.00451`. The 808-query grouped CIs remained `[-0.00166,+0.00966]` and `[-0.00177,+0.01130]`. A joint-positive label (both nDCG and Recall gain above zero) produced nDCG `+0.00398` and Recall `+0.00404`, with CIs still crossing zero. The two halves of the 808 holdout were both positive (`+0.00467/+0.00310` nDCG), but neither half was individually conclusive.

The larger data expansion stabilizes the direction of the point estimate, yet does not meet the promotion requirement of a statistically credible same-budget improvement. Fixed CE-33 and the quality guard remain the production choice.

A conservative quality-guard variant was also probed: send only the highest-confidence classifier queries to CE-40 and route all others through fixed CE-33. At 40% CE-40, mean cost is 35.79 pairs (about 8% above CE-33) and the holdout point gains are nDCG `+0.00596`, Recall `+0.00938`. This is a quality/cost tradeoff, not a same-cost result; confidence intervals and HTTP latency still need confirmation before it could be considered for deployment.

The quality-first 50% CE-40 / 50% fixed-CE-33 guard is documented separately in `FULL_ROUTER_QUALITY_GUARD_10K.md`. It is the first variant whose independent 808-query nDCG and Recall CIs are both above zero, at an explicit average cost of 36.49 CE pairs.
