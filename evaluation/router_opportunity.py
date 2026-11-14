"""Measure retrospective two-action routing headroom on judged queries."""

import hashlib
import json
import math

import numpy as np

from .dureader_policy_features import query_groups
from .evaluate import _metric_for_query
from .router_metrics import GROUP_BOOTSTRAP_SAMPLES, GROUP_BOOTSTRAP_SEED, group_bootstrap


METRICS = ("ndcg_at_10", "recall_at_10", "mrr_at_10", "map_at_100")


def analyze_opportunity(records, qrels, runs, rule_calls, learned_calls, constrained):
    """Compare observed policies with a constrained, nDCG-maximizing oracle."""
    qids = [record.qid for record in records]
    eligible = set(qids) - constrained
    budget = len(rule_calls)
    lexical = {qid: _metric_for_query(qrels[qid], runs["lexical"].get(qid, ()), 10)[:4]
               for qid in qids}
    deltas = {}
    for qid in eligible:
        hybrid = _metric_for_query(qrels[qid], runs["hybrid"].get(qid, ()), 10)[:4]
        deltas[qid] = tuple(right - left for left, right in zip(lexical[qid], hybrid, strict=True))
    ordered = sorted(eligible, key=lambda qid: (-deltas[qid][0], qid))
    exact = set(ordered[:budget])
    at_most = {qid for qid in exact if deltas[qid][0] > 0}

    def summary(calls):
        means = {metric: (math.fsum(lexical[qid][i] for qid in qids)
                          + math.fsum(deltas[qid][i] for qid in sorted(calls))) / len(qids)
                 for i, metric in enumerate(METRICS)}
        return {"vector_calls": len(calls), "metrics": means,
                "calls_with_positive_ndcg_gain": sum(deltas[qid][0] > 0 for qid in calls),
                "calls_with_zero_ndcg_gain": sum(deltas[qid][0] == 0 for qid in calls),
                "calls_with_negative_ndcg_gain": sum(deltas[qid][0] < 0 for qid in calls)}

    groups = query_groups([{"query_id": record.qid, "text": record.query} for record in records], qrels)
    groups_sha = hashlib.sha256(json.dumps(sorted(zip(qids, groups.tolist(), strict=True)),
                                            ensure_ascii=False).encode("utf-8")).hexdigest()

    def paired_groups(left, right):
        return {metric: group_bootstrap(np.array([
            ((qid in right) - (qid in left)) * deltas[qid][i] if qid in eligible else 0.0
            for qid in qids]), groups) for i, metric in enumerate(METRICS)}

    return {"scope": "retrospective_judged_two_action_replay", "query_count": len(qids),
            "eligible_queries": len(eligible), "vector_call_budget": budget,
            "positive_gain_queries": sum(gain[0] > 0 for gain in deltas.values()),
            "oracle_objective": "ndcg_at_10", "oracle_tie_break": "qid_asc",
            "policies": {"rules": summary(rule_calls), "learned_same_budget": summary(learned_calls),
                         "oracle_exact_budget": summary(exact), "oracle_at_most_budget": summary(at_most)},
            "observed_policy_grouped_sensitivity": {
                "group_count": len(set(groups)), "groups_sha256": groups_sha, "samples": GROUP_BOOTSTRAP_SAMPLES,
                "seed": GROUP_BOOTSTRAP_SEED, "grouping": "normalized_query_or_shared_positive_connected_components",
                "learned_minus_rules": paired_groups(rule_calls, learned_calls),
                "rules_minus_constraint_aware_hybrid": paired_groups(eligible, rule_calls)}}
