"""Error and cost analysis for observed fixed and learned CE controls."""

from collections import Counter
from statistics import fmean

from .compare import paired_bootstrap
from .evaluate import _metric_for_query


def learned_diagnostics(qrels, runs, plan, *, samples, seed):
    """Explain learned actions without using qrels to change any action."""
    outcomes = ("win", "loss", "tie")
    eligible = Counter(dict.fromkeys(outcomes, 0))
    action_counts = Counter(dict.fromkeys(("ineligible", *(
        f"{action}_{outcome}" for action in ("applied", "skipped") for outcome in outcomes
    )), 0))
    oracle_sum = missed_gain = unnecessary_loss = 0.0
    ndcg_delta = []
    for qid, labels in qrels.items():
        base, fixed, learned = (
            _metric_for_query(labels, [doc_id for doc_id, _ in runs[arm][qid]], 10)[0]
            for arm in ("rrf", "ce40", "learned")
        )
        delta = fixed - base
        ndcg_delta.append(learned - base)
        oracle_sum += max(base, fixed)
        if plan["capacity"][qid] == 0:
            action_counts["ineligible"] += 1
            continue
        active = plan["learned_actions"][qid]
        outcome = "win" if delta > 0 else "loss" if delta < 0 else "tie"
        eligible[outcome] += 1
        action_counts[f"{'applied' if active else 'skipped'}_{outcome}"] += 1
        if not active:
            missed_gain += max(0.0, delta)
        else:
            unnecessary_loss += max(0.0, -delta)
    n = len(qrels)
    contrasts = {
        arm: {
            metric: paired_bootstrap(qrels, runs[arm], runs["learned"], metric=metric,
                                     samples=samples, seed=seed).to_dict()
            for metric in ("ndcg", "recall", "mrr", "map")
        }
        for arm in ("ce28", "ce40") if arm in runs
    }
    fixed_pairs = sum(plan["capacity"].values())
    return {
        "cutoff": 10, "eligible_fixed_ce40_outcomes": dict(eligible),
        "learned_actions_by_ce40_outcome": dict(action_counts),
        "fixed_ce40_pairs": fixed_pairs,
        "learned_ce_pairs": plan["ce_pairs"],
        "ce40_pair_fraction": plan["ce_pairs"] / fixed_pairs if fixed_pairs else None,
        "mean_learned_delta_vs_rrf": fmean(ndcg_delta),
        "mean_oracle_ndcg": oracle_sum / n,
        "mean_missed_positive_ndcg": missed_gain / n,
        "mean_applied_negative_ndcg": unnecessary_loss / n,
        "learned_minus_fixed": contrasts,
    }
