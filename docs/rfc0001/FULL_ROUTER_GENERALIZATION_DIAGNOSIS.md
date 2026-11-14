# Full-corpus Router generalization diagnosis

This is an exploratory diagnosis based on the frozen full-corpus score captures. It is not yet a PR-ready method result.

## Findings

The deployed full-corpus gate uses four features: `n_terms`, `vocab_coverage`, `relative_gap`, and `candidate_hits`.

- `candidate_hits` is exactly 200 for all 172 training queries and all 1,020 full-dev queries, so its coefficient is zero and it cannot discriminate queries.
- `vocab_coverage` is almost constant: training median and full-dev median are both 1.0. It has only two distinct training values.
- The effective gate is therefore almost entirely a two-feature model over `n_terms` and `relative_gap`.
- The frozen threshold is 0.14446. It produces 77.3% heavy on training predictions but 83.0% heavy on full-dev predictions.
- The full-dev gate prediction has almost no relationship to the observed CE-40 minus CE-0 gain: Pearson correlation 0.022 and rank correlation 0.028. The train OOF correlation is also negative (-0.216), so this is not only a corpus-size shift; the gain target itself is poorly predicted by this gate.
- A fixed-rate quota over the existing predictions reproduces the same regression: at 83% heavy, nDCG is 0.33638 versus fixed CE-33 at 0.35346.

## Action-pair diagnosis

The original learned Router chooses CE-40 or CE-0. This is not the same action pair as the full-corpus fixed CE-33 control. A prototype that chooses CE-40 or CE-28 was tested using the same frozen captures and training-only labels derived from the official dev judgments.

- At approximately 43.3% CE-40, the mean budget is approximately 33.2 pairs, matching the deployed Router and fixed CE-33 cost. The best tested model was still below fixed CE-33 (ExtraTrees: about -0.0025 nDCG).
- At 83% CE-40, the CE-28/CE-40 prototype reaches about 0.3577 nDCG, above fixed CE-33, but its mean budget is about 38 pairs, so this is not a same-cost improvement.

## Current conclusion

Two separate problems are present:

1. The exported gate is not a reliable predictor of per-query CE gain, even on train OOF.
2. The original cheap/heavy action pair is mismatched with the fixed CE-33 baseline; replacing CE-0 with CE-28 helps only when allowing a larger average CE budget.

The next repair should replace the regression gate with an explicitly budget-constrained CE-28/CE-40 selector trained on the CE-40 minus CE-28 objective, add non-degenerate BM25 confidence features, and evaluate it with grouped nested selection at a predeclared mean budget of 33.2 pairs. It must beat fixed CE-33 on an independent holdout before being promoted.

## Additional selector probes

The following training-only selectors were also probed against the frozen full-dev judgments:

- Rich BM25/dense confidence features, ExtraTrees/RF/HistGradientBoosting and SVR models: no same-budget CE-28/CE-40 selector exceeded fixed CE-33.
- Cheap CE-28 score confidence features: same-budget nDCG remained below fixed CE-33; the best positive deltas appeared only after raising the heavy rate and therefore the average budget.
- BGE-small query embeddings, query text TF-IDF, quality prediction, and simple domain rules: none produced a same-budget improvement.
- An oracle selector using the true held-out CE-40 minus CE-28 gains would reach about `0.37350` nDCG at the 33.2-pair budget, about `+0.0201` over fixed CE-33. This confirms that headroom exists, but the currently available online features do not identify the profitable queries.

This separates an action-quality ceiling from a prediction failure: the Router action space has potential, while the current train-only supervision and observable features do not recover it.

The reproducible probe is `evaluation/diagnose_full_router_generalization.py`; its output is stored locally as `router-generalization-diagnosis.json`. The best tested ExtraTrees selector reaches `0.35668` nDCG at 35.19 mean CE pairs, but remains `-0.00191` below fixed CE-33 at the matched 33.19-pair budget. This is a measurable quality/cost tradeoff, not a valid same-budget promotion.
