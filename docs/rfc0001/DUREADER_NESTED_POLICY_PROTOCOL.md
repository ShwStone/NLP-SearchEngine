# Corrected DuReader CE/Router experiment

This follow-up repairs reproducibility gaps in the earlier 100k report.
Dense retrieval explicitly uses HNSW ef=128/exact=false. Historical lexical
captures are reused with external/internal IDs bound to the verified index;
this reuse is recorded rather than presented as a new lexical run. Equal RRF
uses c=60 and global internal docid ties. Explicit constraints remain lexical.

## Label and split discipline

`capture_ce_actions` scans the corpus with bounded memory, keeps only selected
candidate texts, verifies the source bytes/count, and scores CE-28 and CE-40
separately without accessing qrels. Large scores/captures stay local.

`train_dureader_policies` trains only on 172 internal-train queries. Normalized
duplicate query texts and shared positive-document IDs define connected
training groups (171 groups). Four outer GroupKFold splits and three inner
splits preserve group separation. Inner folds select ridge regularization
from 0.1/1/10/100; all OOF assignments and selections are recorded locally.
Dev predictions and model exports are saved before opening dev qrels. The
1,020 internal-dev queries remain historically observed development evidence,
not an untouched final test. Grouped paired bootstrap uses 1,014 dev groups.
The frozen train/dev splits have no normalized duplicate queries, but share
one positive document. The maintainer's split is preserved. Training folds
group shared positives internally; this does not remove that cross-split
document overlap or turn the development results into independent test claims.

The gate estimates CE-40's nDCG gain using only term count, corpus vocabulary
coverage, BM25 relative gap and lexical candidate count. Its threshold comes
from training OOF predictions with a predeclared 60% train heavy-budget ceiling.
The exact same functions and exported scaler/linear coefficients are used in
serving. Learned candidate ranking uses CE score and hybrid/lexical/dense rank
features; no relevance grades are features.

Random controls use label-independent seeded query hashes and exactly match
the Router's realized heavy count. A fixed CE candidate count matches average
pair budget after rounding down; its quality can reuse cached pair scores, but
its latency must be measured by separately executing that budget over HTTP.

## Observed v3 diagnostic results

| Policy | nDCG@10 | Recall@10 | Mean CE pairs |
|---|---:|---:|---:|
| equal RRF / CE-0 | 0.22767 | 0.26430 | 0 |
| old rules Router | 0.20464 | 0.24055 | 0 |
| raw CE-28 | 0.40895 | 0.42601 | 28 |
| raw CE-40 | 0.48586 | 0.52905 | 40 |
| learned Router | 0.38770 | 0.42404 | 24 |
| random seed 20261113 | 0.38441 | 0.42496 | 24 |
| random seed 20261114 | 0.38400 | 0.42346 | 24 |
| fixed CE-24 | 0.38116 | 0.38593 | 24 |
| learned CE-28 ranker | 0.38454 | 0.42047 | 28 |
| learned CE-40 ranker | 0.47839 | 0.52118 | 40 |

Router minus same-budget random nDCG differences are +0.00329 and +0.00370,
with grouped 95% CIs [-0.01232, 0.01848] and [-0.01210, 0.01975]. There is no
reliable evidence of superiority over random gating. Improvement over CE-0
alone is insufficient to establish useful routing.

The added rank features regress against raw CE. The v4 repair therefore adds
raw CE as an explicit candidate in training-only inner selection, with raw
preferred on ties. Both full-train selections reject learned reranking and
choose raw CE. This protects quality using training evidence; it is not a
claim that a learned ranker improved accuracy. The v3 negative results remain
preserved and must not be silently replaced. The gate remains experimental.

Independent audits reconstruct all exported gate/ranker predictions and
aggregate quality metrics, outer group separation and OOF membership, plus
70 recorded inner selection trials across four outer folds and final training.
Every inner ridge fit, validation score and gate threshold matches the recorded
result without opening dev labels. See `dureader-100k-nested-v4-audit.json` and
`dureader-100k-nested-v4-inner-audit.json`.

## Remaining gates

All-query HTTP verification exposed two adjacent-order differences for the
fixed CE-24 action when compared with a prefix of cached CE-40 scores.
Different last-batch padding/shape in float16 can perturb near-tied logits.
The complete HTTP run is retained as measured evidence, but the reused-score
fixed-depth result is not considered exact serving parity. A separate CE-24
capture is required for the corrected fixed-budget comparison. This affects
neither the frozen gate parameters nor the need for a final isolated run.

The separate CE-24 capture is now complete for 1,020 queries. Ten complete
rankings differ from the reused CE-40 prefix; aggregate nDCG/Recall are
unchanged. Router minus independently executed CE-24 is +0.00653 nDCG
(grouped 95% CI [-0.00849, 0.02180]) and +0.03812 Recall@10
([0.02075, 0.05505]). This does not establish nDCG superiority. See
`dureader-100k-fixed24-corrected-summary.json`. CE component timings were
recorded under concurrent load and are not isolated latency acceptance.

The 8-query HTTP pilot matched all ten policies' predicted document lists and
the learned gate decisions. The later all-query reports retain the observed
ranking comparisons and independently scored CE-24/CE-33 controls. Benchmark
latency was sampled under concurrent batch load. Use future quiescent
measurements for the isolated timing gate, and the frozen internal holdout
for evaluation outside the CE gate's training and development splits.
[Quality guard](DUREADER_QUALITY_GUARD.md) · [Full development evidence](DUREADER_FULL_FINAL_REPORT.json) · [Course report](../../REPORT.md#10-ce-重排与逐查询成本分配)
