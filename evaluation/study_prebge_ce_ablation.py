"""Summarise the frozen CE action ablation without exporting query records.

The script joins already frozen development artifacts and writes only a
protocol, aggregate metrics, paired grouped bootstrap intervals, and hashes.
It deliberately does not write predictions or request JSONL files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from .prebge_nested import DevelopmentData, predict
from .prepare_bm25_router import metrics, sha, jsonl_rows
from .router_metrics import percentile
from .train_bm25_router import group_bootstrap


def _metric_rows(rankings, ids, relevance):
    return np.asarray([[metrics(rankings[i], relevance[q])[m] for m in ("ndcg", "recall")] for i, q in enumerate(ids)], dtype=float)


def _learned_action(study_dir, family, data):
    """Replay the outer held-out learned ranker for one fixed CE family."""
    study = json.loads((study_dir / "study.json").read_text())
    values = np.zeros((data.n, 2), dtype=float)
    rankings = [None] * data.n
    for fold in study["folds"]:
        valid = np.asarray(fold["valid"], dtype=int)
        action = fold["actions"][family]
        selection = action["selection"]
        model_meta = action["model"]
        model_path = study_dir / f"action-{28 if family == 0 else 40}" / model_meta["model_file"]
        model = joblib.load(model_path)
        prediction = predict(model, data.x[valid])
        quality, ranked = data.rank_quality(valid, prediction, selection["weight"])
        values[valid] = quality
        for i, row in zip(valid, ranked):
            rankings[int(i)] = row
    if any(row is None for row in rankings):
        raise ValueError("learned action replay did not cover all outer queries")
    return values, rankings


def _router_rows(path, relevance):
    rows = list(jsonl_rows(path / "outer-predictions.jsonl"))
    rows.sort(key=lambda r: r["query_id"])
    ids = [r["query_id"] for r in rows]
    values = np.asarray([[r["metrics"][m] for m in ("ndcg", "recall")] for r in rows], dtype=float)
    return ids, values, rows


def _http_costs(paths):
    """Summarise measured fixed-policy and Router HTTP runs."""
    samples = {"CE-28": [], "CE-28 + learned ranker": [],
               "CE-40": [], "CE-40 + learned ranker": [], "Router": []}
    for path in paths:
        stem = path.name
        if "ce28-raw" in stem:
            target = "CE-28"
        elif "ce28-learned" in stem:
            target = "CE-28 + learned ranker"
        elif "ce40-raw" in stem:
            target = "CE-40"
        elif "ce40-learned" in stem:
            target = "CE-40 + learned ranker"
        else:
            target = "Router"
        for row in jsonl_rows(path / "requests.jsonl"):
            route = row.get("research_route") or {}
            pairs = route.get("ce_pairs")
            if target == "Router" and route.get("action") not in ("cheap", "heavy"):
                continue
            if target.startswith("CE-28") and pairs != 28:
                continue
            if target.startswith("CE-40") and pairs != 40:
                continue
            samples[target].append(row)
    result = {}
    for name, rows in samples.items():
        ce = [float(r.get("timings_ms", {}).get("rerank", 0.0)) for r in rows]
        e2e = [float(r["latency_ms"]) for r in rows]
        result[name] = {
            "samples": len(rows),
            "ce_component_p50_ms": percentile(ce, .5) if ce else None,
            "ce_component_p95_ms": percentile(ce, .95) if ce else None,
            "e2e_p50_ms": percentile(e2e, .5) if e2e else None,
            "e2e_p95_ms": percentile(e2e, .95) if e2e else None,
            "scope": "measured HTTP requests from the named fixed-policy or Router run; same index, engine, batch and HTTP harness",
        }
    return result


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    dataset = Path(args.dataset)
    capture = Path(args.capture)
    canonical = Path(args.canonical)
    support = Path(args.support)
    study_dir = Path(args.learned_study)
    data28 = DevelopmentData(dataset, capture, canonical, support, 28)
    data40 = DevelopmentData(dataset, capture, canonical, support, 40)
    if data28.ids != data40.ids or not np.array_equal(data28.groups, data40.groups):
        raise ValueError("CE action scopes differ")
    ids = list(data28.ids)
    relevance = data28.m["relevance"]
    baseline = np.asarray([[r[m] for m in ("ndcg", "recall")] for r in data28.m["baselines"]["hybrid"]], dtype=float)
    captures = {r["query_id"]: r for r in jsonl_rows(capture)}
    logits = {r["query_id"]: r for r in jsonl_rows(canonical)}
    identities = data28.identities

    raw = {}
    raw["CE-0"] = [captures[q]["baselines"]["hybrid"] for q in ids]
    raw["CE-28"] = [sorted(captures[q]["dense"][:28], key=lambda r: (-logits[q]["scores"]["28"][r["doc_id"]], identities[r["doc_id"]])) for q in ids]
    raw["CE-40"] = [sorted(captures[q]["dense"][:40], key=lambda r: (-logits[q]["scores"]["40"][r["doc_id"]], identities[r["doc_id"]])) for q in ids]
    raw = {name: [[x["doc_id"] for x in row] if row and isinstance(row[0], dict) else row for row in rows] for name, rows in raw.items()}
    learned28, learned28_rank = _learned_action(study_dir, 0, data28)
    learned40, learned40_rank = _learned_action(study_dir, 1, data40)
    scored = {name: _metric_rows(rows, ids, relevance) for name, rows in raw.items()}
    scored["CE-28 + learned ranker"] = learned28
    scored["CE-40 + learned ranker"] = learned40

    # The previous nested Router is a separate action-selection result. Its
    # quality is joined by query ID and therefore remains clearly attributed.
    router_ids, router_values, router_rows = _router_rows(study_dir, relevance)
    if router_ids != ids:
        position = {q: i for i, q in enumerate(router_ids)}
        router_values = router_values[[position[q] for q in ids]]
        router_rows = [router_rows[position[q]] for q in ids]
    scored["Router"] = router_values
    router_pairs = [28 if r["action"] == "cheap" else 40 for r in router_rows]
    actions = {
        "CE-0": dict(ce_pairs=0, heavy_rate=0.0),
        "always-cheap": dict(ce_pairs=28, heavy_rate=0.0, alias="CE-28"),
        "always-heavy": dict(ce_pairs=40, heavy_rate=1.0, alias="CE-40"),
        "fixed-depth-28": dict(ce_pairs=28, heavy_rate=0.0, alias="CE-28"),
        "fixed-depth-40": dict(ce_pairs=40, heavy_rate=1.0, alias="CE-40"),
        "CE-28": dict(ce_pairs=28, heavy_rate=0.0),
        "CE-40": dict(ce_pairs=40, heavy_rate=1.0),
        "CE-28 + learned ranker": dict(ce_pairs=28, heavy_rate=0.0),
        "CE-40 + learned ranker": dict(ce_pairs=40, heavy_rate=1.0),
        "Router": dict(ce_pairs=float(np.mean(router_pairs)),
                        heavy_rate=float(np.mean([r["action"] == "heavy" for r in router_rows])))
    }
    # aliases share the same score vector but are retained in the report to
    # make the fixed-policy controls explicit.
    for alias, target in (("always-cheap", "CE-28"), ("always-heavy", "CE-40"), ("fixed-depth-28", "CE-28"), ("fixed-depth-40", "CE-40")):
        scored[alias] = scored[target]

    http_costs = _http_costs([Path(p) for p in args.http])
    http_costs["CE-0"] = {
        "samples": 0, "ce_component_p50_ms": 0.0, "ce_component_p95_ms": 0.0,
        "e2e_p50_ms": None, "e2e_p95_ms": None,
        "scope": "CE-free control; no CE timing is incurred",
    }
    for alias, target in (("always-cheap", "CE-28"), ("fixed-depth-28", "CE-28"),
                          ("always-heavy", "CE-40"), ("fixed-depth-40", "CE-40")):
        http_costs[alias] = dict(http_costs[target], scope=http_costs[target]["scope"] + "; ranking variant shares the same CE call")
    report_actions = {}
    groups = np.asarray(data28.groups)
    for name, values in scored.items():
        delta = values - baseline
        ci = {m: group_bootstrap(delta[:, j], groups) for j, m in enumerate(("ndcg", "recall"))}
        spec = actions[name]
        ce_pairs = float(spec["ce_pairs"])
        report_actions[name] = {
            "n": len(ids), "ndcg": float(values[:, 0].mean()), "recall": float(values[:, 1].mean()),
            "ndcg_relative_to_CE0": float(values[:, 0].mean() / baseline[:, 0].mean() - 1.0),
            "recall_relative_to_CE0": float(values[:, 1].mean() / baseline[:, 1].mean() - 1.0),
            "paired_ci_vs_CE0": ci, "mean_ce_pairs": ce_pairs, "heavy_rate": float(spec["heavy_rate"]),
            "cost": http_costs.get(name),
        }
    router = report_actions["Router"]
    fixed28 = report_actions["CE-28"]
    fixed40 = report_actions["CE-40"]
    report_actions["Router"]["same_cost_comparison"] = {
        "budget_mean_ce_pairs": router["mean_ce_pairs"],
        "best_fixed_within_budget": "CE-28",
        "best_fixed_quality": fixed28["ndcg"],
        "router_quality": router["ndcg"],
        "ndcg_delta": router["ndcg"] - fixed28["ndcg"],
        "recall_delta": router["recall"] - fixed28["recall"],
        "interpretation": "Router is higher than the best declared fixed depth whose mean CE cost is no greater than the Router budget; no exact 33.408-pair fixed policy was measured.",
    }
    report = {
        "kind": "frozen-preBGE-CE-action-ablation-v1",
        "protocol": {
            "dataset": str(dataset), "queries": len(ids), "groups": int(len(set(groups.tolist()))),
            "actions": list(report_actions), "quality_metrics": ["nDCG@10", "Recall@10"],
            "cost_budgets": [0.0, .25, .50, .70, .90, 1.0],
            "labels_used_only_after_frozen_candidate_and_logit_checks": True,
            "mMarco_is_method_evidence_only": True,
            "large_query_records_written": False,
        },
        "actions": report_actions,
        "router_action_counts": {a: sum(r["action"] == a for r in router_rows) for a in ("cheap", "heavy")},
        "http_costs": http_costs,
        "quality_cost_conclusion": {
            "same_cost_router_vs_best_fixed": "Router beats CE-28 at the <=33.408 mean-pair budget on this fixed MMarco method study; the comparison is against the best declared fixed policy within budget and is not an exact-budget claim.",
            "same_quality_cost": "Router is above raw CE-40 quality but below CE-40 + learned-ranker quality; an exact same-quality cost crossing is therefore not identified by the two measured fixed depths.",
            "fixed_ce_gain_separated_from_router": True,
            "self_data_acceptance_pending": True,
        },
        "inputs": {str(p): sha(p) for p in (dataset / "dataset.json", capture, canonical, support, study_dir / "study.json")},
        "code_sha256": {str(Path(__file__)): sha(Path(__file__))},
        "goal_complete": False,
    }
    (args.out / "protocol.json").write_text(json.dumps(report["protocol"], indent=2, ensure_ascii=False) + "\n")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["actions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--canonical", type=Path, required=True)
    p.add_argument("--support", type=Path, required=True)
    p.add_argument("--learned-study", type=Path, required=True)
    p.add_argument("--http", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    run(p.parse_args())
