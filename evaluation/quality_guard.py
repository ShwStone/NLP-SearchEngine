"""Record a development-only decision for a learned CE budget Router."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile


def _metric(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid {name}")
    return value


def guard(summary, fixed_control, output, margin=0.0):
    summary, fixed_control, output = map(Path, (summary, fixed_control, output))
    if os.path.lexists(output):
        raise FileExistsError(output)
    margin = _metric(margin, "acceptance margin")
    summary_bytes, fixed_bytes = summary.read_bytes(), fixed_control.read_bytes()
    learned, fixed = json.loads(summary_bytes), json.loads(fixed_bytes)
    source, control = learned["protocol"], fixed["protocol"]
    summary_sha = sha256(summary_bytes).hexdigest()
    qrels_sha = source["dev_qrels_sha256"]
    if (source["all_dev_metrics_exploratory"] is not True
            or control["separately_scored"] is not True
            or control["model_or_threshold_changed"] is not False
            or control["policy_summary_sha256"] != summary_sha
            or not isinstance(qrels_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", qrels_sha)
            or control["qrels_sha256"] != qrels_sha):
        raise ValueError("fixed control is not bound to the same exploratory development run")
    queries, budget = source["dev_queries"], source["fixed_pair_budget"]
    if (type(queries) is not int or queries <= 0 or type(budget) is not int or budget <= 0
            or type(fixed["queries"]) is not int or fixed["queries"] != queries
            or type(fixed["fixed_ce_pairs"]) is not int or fixed["fixed_ce_pairs"] != budget):
        raise ValueError("fixed control query count or CE budget differs from the Router run")
    policy = learned["learned-router"]
    learned_ndcg = _metric(policy["metrics"]["ndcg"], "learned nDCG")
    fixed_ndcg = _metric(fixed["metrics"]["ndcg"], "fixed nDCG")
    fixed_recall = _metric(fixed["metrics"]["recall"], "fixed recall")
    heavy_rate = _metric(policy["heavy_rate"], "learned trigger rate")
    mean_pairs = policy["ce_pairs_mean"]
    if type(mean_pairs) not in (int, float) or not math.isfinite(mean_pairs) or math.floor(mean_pairs) != budget:
        raise ValueError("learned mean CE pairs do not agree with the fixed budget")
    delta = fixed["router_minus_fixed"]["ndcg"]["mean"]
    if (type(delta) not in (int, float)
            or not math.isclose(delta, learned_ndcg - fixed_ndcg, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("reported nDCG delta disagrees with the bound inputs")
    fallback = delta < -margin
    result = {
        "schema_version": 2,
        "policy": "quality-guarded-fixed-heavy",
        "scope": "internal_development_only",
        "fallback_activated": fallback,
        "acceptance_margin": margin,
        "learned_router_ndcg": learned_ndcg,
        "fixed_action_ndcg": fixed_ndcg,
        "learned_minus_fixed_ndcg": delta,
        "fixed_action_recall": fixed_recall,
        "heavy_rate": 1.0 if fallback else heavy_rate,
        "ce_pairs_mean": budget if fallback else mean_pairs,
        "quality_guard_reason": "fixed depth preferred on dev" if fallback else "point estimate within dev margin",
        "production_default_safe": False,
        "inputs": {"policy_summary_sha256": summary_sha,
                   "fixed_control_sha256": sha256(fixed_bytes).hexdigest(),
                   "qrels_sha256": qrels_sha, "queries": queries},
    }
    pending = None
    try:
        with NamedTemporaryFile(dir=output.parent, delete=False) as file:
            pending = Path(file.name)
            file.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8"))
            file.flush()
            os.fsync(file.fileno())
        os.link(pending, output)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "fixed-control", "output"):
        cli.add_argument("--" + name, required=True)
    cli.add_argument("--margin", type=float, default=0.0)
    print(json.dumps(guard(**vars(cli.parse_args())), indent=2))
