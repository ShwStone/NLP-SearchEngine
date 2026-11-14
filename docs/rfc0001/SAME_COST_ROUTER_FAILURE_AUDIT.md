# Same-cost full-corpus Router failure audit

This local audit records the final bounded exploration for the strict same-cost goal. No result below is promoted as the final Router or pushed as a success claim.

## Acceptance gate

- Independent holdout: 808 full-corpus queries.
- Reference: fixed CE-33.
- Mean CE pairs: at most 33.1.
- nDCG@10 and Recall@10: both at least 5% relative improvement, with grouped bootstrap 95% CI lower bounds above zero.
- Decision features must be available before CE; threshold/model selection must use train or internal calibration data only.

## Final HTTP results

| policy | mean CE pairs | heavy rate | nDCG@10 | Recall@10 | comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed CE-33 | 33.000 | 100% | 0.34943 | 0.35417 | reference |
| CE-20/40 pre-CE Router | 32.104 | 60.52% | 0.34369 | 0.34668 | quality regression |
| CE-20/40 HTTP-trained scalar calibrator | 31.856 | 59.28% | 0.34194 | 0.34624 | quality regression |
| CE-20/40 HTTP-feature Router | 32.426 | 62.13% | 0.34526 | 0.34954 | quality regression |
| CE-28/40 HTTP-feature Router | 32.886 | 40.72% | 0.35098 | 0.35339 | nDCG +0.44%, Recall -0.22% |

All final HTTP runs used the same full-corpus loopback harness, qrels read after request capture, and fixed CE-33 controls. The CE-28/40 action space was the last bounded candidate because it preserves the 33-pair budget with a 5/12 heavy fraction.

## Offline versus online finding

Offline full-corpus captures contain an oracle action ceiling and several positive pre-CE models. Replaying those models through the real HTTP path removes the apparent gain. The mismatch persists after fixing a feature-schema offset and after training on 100k internal-dev HTTP CE-20/40 labels. This indicates a corpus/runtime distribution shift in per-query action gains, rather than a missing threshold adjustment.

The quality-first 40% heavy policy remains a valid separate result with higher cost, but it is outside this same-cost gate. No same-cost Router is promoted, and PR #9 is not updated with a success claim.

## Resumed blocked-audit CE-24/40 experiment

A final high-ceiling action space was trained from real 100k HTTP CE-24/CE-40 labels with an OOF-selected RandomForest and evaluated over real HTTP on the independent 808 holdout. It produced:

- mean CE pairs `34.020` (fails the `33.1` cost gate);
- nDCG `0.35041` versus fixed CE-33 `0.34943` (+0.28%);
- Recall `0.35507` versus `0.35417` (+0.25%).

Thus even the higher-ceiling CE-24/40 action space does not approach the 5% quality target and slightly exceeds the cost gate. This reinforces the full-corpus generalization blocker.
