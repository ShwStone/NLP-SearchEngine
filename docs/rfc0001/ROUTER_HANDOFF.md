# RFC-0001 Router handoff

This note is the handoff point for the next contributor. The branch contains the
reproducible implementation and aggregate reports for the current full-corpus
Router exploration. Large corpora, indexes, model files, request logs, and
per-query captures remain local and are intentionally excluded from Git.

## What is established

- The full-corpus quality-first HTTP policy is reproducible and improves both
  nDCG@10 and Recall@10 on the independent 808-query holdout, at a mean cost of
  35.790 CE pairs versus 33 for fixed CE-33. See
  `FULL_ROUTER_HTTP_QUALITY_GUARD.md`.
- The original learned Router and the new same-cost CE-20/40 and CE-28/40
  selectors do not establish a same-cost improvement over fixed CE-33. The
  latest CE-28/40 HTTP result is nDCG +0.44% and Recall -0.22% at 32.886 mean
  CE pairs; CE-24/40 is +0.28%/+0.25% at 34.020 pairs and exceeds the cost
  gate. See `SAME_COST_ROUTER_FAILURE_AUDIT.md`.
- The negative result is a valid research outcome: the online features and
  train-only gain labels do not reliably identify profitable CE-40 decisions on
  the full corpus. Do not describe the same-cost Router as a completed success.

## Reproducible code added in this handoff

- `evaluation/prece_router.py`: frozen pre-CE feature schema and portable tree
  inference for exported models.
- `evaluation/train_prece_router.py`: grouped five-fold OOF training/export
  entry point for CE-20/40, CE-24/40, or CE-28/40 action pairs.
- `evaluation/benchmark_dureader_policies_http.py`: real loopback HTTP policies,
  pre-CE model loading, scalar calibration, feature logging, and HTTP error
  reporting.

The training script expects frozen score captures and qrels supplied by the
local data recipe. Exported model JSON files are local artifacts and are not
required to review the implementation.

## Suggested continuation

1. Keep fixed CE-33 as the full-corpus same-cost reference.
2. If pursuing Router quality, obtain or construct additional legal training
   queries and freeze a new decision-time feature set before touching the 808
   holdout.
3. Use the HTTP benchmark for every promoted policy and report mean CE pairs,
   heavy rate, CE/e2e p50 and p95, nDCG, Recall, and grouped bootstrap intervals.
4. Promote a same-cost policy only after it beats fixed CE-33 on both quality
   metrics with a predeclared independent evaluation protocol.

## Validation for this handoff

The handoff change is syntax-checked and covered by the targeted Router tests.
The existing branch-level aggregate and HTTP audits remain the evidence for the
earlier results; no local artifact is silently treated as committed evidence.
