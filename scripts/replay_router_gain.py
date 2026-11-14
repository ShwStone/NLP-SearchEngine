"""Freeze learned Router actions, then score their recorded retrieval branches."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import hashlib
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.index_format import sha256_of
from evaluation.compare import paired_bootstrap
from evaluation.evaluate import QueryRouting, _metric_for_query, evaluate_run, load_qrels, load_queries
from evaluation.router_budget import check_journal, parse_json, read_rankings, require, VECTOR_REASONS, LEXICAL_REASONS
from evaluation.train_router_gain import FEATURE_GROUPS, load_train_features, model_json, train_router_gain
from online.linear_gain_router import FEATURE_VERSION, LinearGainRouter


MODES = ("lexical", "hybrid", "adaptive")
METRICS = ("ndcg_at_10", "recall_at_10", "mrr_at_10", "map_at_100")
BUDGET_PERCENTAGES = (0, 10, 25, 50, 75, 90, 100)


def load_binding(path):
    """Pin the externally audited experiment and every archived arm source."""
    value = read(path)
    fields = {"schema_version", "corpus_sha256", "candidate_k", "rrf_c", "hnsw_ef",
              "feature_version", "model_sha256", "features_sha256", "queries_sha256",
              "training_queries_sha256", "arm_sources"}
    require(isinstance(value, dict) and set(value) == fields
            and type(value["schema_version"]) is int and value["schema_version"] == 1,
            "Experiment binding schema disagrees")
    hashes = ("corpus_sha256", "model_sha256", "features_sha256", "queries_sha256",
              "training_queries_sha256")
    valid_digest = lambda item: isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item)
    require(all(valid_digest(value[key]) for key in hashes)
            and value["feature_version"] == FEATURE_VERSION,
            "Experiment binding identity disagrees")
    require(type(value["candidate_k"]) is int and value["candidate_k"] >= 100
            and type(value["rrf_c"]) in (int, float) and math.isfinite(value["rrf_c"])
            and value["rrf_c"] > 0 and type(value["hnsw_ef"]) is int and value["hnsw_ef"] > 0,
            "Experiment binding retrieval configuration disagrees")
    arms = value["arm_sources"]
    require(isinstance(arms, dict) and set(arms) == set(MODES), "Missing bound retrieval arms")
    for mode in MODES:
        source = arms[mode]
        require(isinstance(source, dict) and set(source) == {
                    "run_sha256", "metrics_sha256", "request_log_sha256"}
                and all(valid_digest(entry) for entry in source.values()),
                f"Invalid {mode} source binding")
    return value


def read_requests(path, mode, qids, rankings, summaries, candidate_k):
    """Validate current journal receipts or explicitly recognized archived JSONL."""
    raw = path.read_bytes()
    try:
        payload = parse_json(raw)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict) and "schema_version" in payload:
        receipt = payload
        n = len(qids)
        request = receipt["request"]
        require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
                and receipt["status"] == "COMPLETE" and receipt["query_ids"] == qids
                and receipt["policy"] is None
                and request["mode"] == mode and request["scorer"] == "bm25"
                and request["expander"] == "none" and type(request["k"]) is int
                and 100 <= request["k"] <= candidate_k,
                f"{mode} request receipt identity or depth disagrees")
        universe_sha = hashlib.sha256(json.dumps(qids, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
        expected = {"requested_queries": n, "attempted_requests": n, "http_200": n,
                    "routing_observed_queries": n, "expected_unsupported": 0,
                    "unexpected_errors": 0, "last_event_sequence": 2 * n}
        require(receipt["query_universe_sha256"] == universe_sha
                and all(type(receipt[key]) is int and receipt[key] == number
                        for key, number in expected.items())
                and receipt["http_success_coverage"] == 1.0
                and receipt["protocol_accounted_coverage"] == 1.0
                and receipt["inflight_outcome"] is None,
                f"{mode} request receipt coverage disagrees")
        outcomes = receipt["outcomes"]
        require(isinstance(outcomes, list) and len(outcomes) == n
                and receipt["last_outcome"] == outcomes[-1], f"Incomplete {mode} request outcomes")
        require(check_journal(path, receipt, qids) is not None,
                f"{mode} request receipt lacks a finalized journal")
        for qid, outcome in zip(qids, outcomes):
            require(outcome["qid"] == qid and outcome["status"] == "SUCCESS"
                    and type(outcome["http_status"]) is int and outcome["http_status"] == 200,
                    f"{mode} request failed at {qid}")
        kind = "complete_receipt_with_journal"
    else:
        require(raw.endswith(b"\n"), f"{mode} archived request log lacks final newline")
        outcomes = [parse_json(line) for line in raw.splitlines()]
        require(len(outcomes) == len(qids), f"Incomplete {mode} archived request log")
        for qid, outcome in zip(qids, outcomes):
            require(isinstance(outcome, dict) and set(outcome) == {
                        "query_id", "status", "returned", "latency_ms", "routing"}
                    and outcome["query_id"] == qid and type(outcome["status"]) is int
                    and outcome["status"] == 200 and type(outcome["returned"]) is int
                    and outcome["returned"] == len(rankings.get(qid, [])),
                    f"{mode} archived request coverage or returned depth disagrees at {qid}")
        kind = "legacy_ordered_http200_jsonl_no_recorded_request_configuration"
    for qid, outcome in zip(qids, outcomes):
        latency = outcome["latency_ms"]
        require(type(latency) in (int, float) and math.isfinite(latency) and latency >= 0
                and len(rankings.get(qid, [])) <= candidate_k,
                f"{mode} invalid latency or ranking depth at {qid}")
        route = QueryRouting.from_response(outcome["routing"], mode)
        require(not route.fallback
                and route == QueryRouting.from_response(summaries[qid]["routing"], mode),
                f"{mode} request routing differs from metrics at {qid}")
        if mode == "adaptive":
            require(route.reason in (VECTOR_REASONS if route.vector_used else LEXICAL_REASONS),
                    f"Unknown adaptive rule reason at {qid}")
    return kind


def read(path):
    return parse_json(path.read_bytes())


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def freeze(args):
    binding = load_binding(args.experiment_binding)
    for key, path in (("model_sha256", args.model), ("features_sha256", args.features),
                      ("queries_sha256", args.queries),
                      ("training_queries_sha256", args.training_queries)):
        require(binding[key] == sha256_of(path), f"Experiment binding {key} disagrees")
    require(sha256_of(args.model) == args.model_sha256, "Model SHA256 disagrees")
    raw = read(args.model)
    expected_model_binding = {key: binding[key] for key in
                              ("corpus_sha256", "candidate_k", "rrf_c", "feature_version")}
    require(raw["provenance"].get("binding") == expected_model_binding,
            "Model corpus or retrieval binding disagrees with experiment")
    require(raw["provenance"]["inputs_sha256"]["train_queries"] == sha256_of(args.training_queries),
            "Training query identity does not match model provenance")
    policy = LinearGainRouter.from_dict(raw, args.model_sha256)
    train, records = load_queries(args.training_queries), load_queries(args.queries)
    qids = [row.qid for row in records]
    receipt = read(args.feature_receipt)
    expected_receipt = {**expected_model_binding, "features_sha256": binding["features_sha256"],
                        "queries_sha256": binding["queries_sha256"],
                        "lexical_run_sha256": binding["arm_sources"]["lexical"]["run_sha256"]}
    require(isinstance(receipt, dict)
            and all(receipt.get(key) == value for key, value in expected_receipt.items())
            and type(receipt.get("features")) is int and receipt["features"] == len(qids)
            and type(receipt.get("rank_matches")) is int and receipt["rank_matches"] == len(qids)
            and type(receipt.get("depth")) is int and 100 <= receipt["depth"] <= binding["candidate_k"],
            "Feature receipt corpus, query, ranking or feature binding disagrees")
    normalize = lambda text: "".join(text.casefold().split())
    require(bool(qids) and len(qids) == len(set(qids))
            and not set(qids).intersection(row.qid for row in train)
            and not {normalize(row.query) for row in records}.intersection(normalize(row.query) for row in train),
            "Evaluation cohort overlaps training or contains duplicate query IDs")
    features = load_train_features(args.features, set(qids))
    require(type(receipt.get("constrained")) is int
            and receipt["constrained"] == sum(feature.explicit_constraints for feature in features.values()),
            "Feature receipt constraint count disagrees")
    scores = {qid: policy.predict(features[qid]) for qid in qids if not features[qid].explicit_constraints}
    require(0 <= args.fixed_budget <= len(scores), "Fixed budget exceeds eligible query count")
    allocated = set(sorted(scores, key=lambda qid: (-scores[qid], qid))[:args.fixed_budget])
    rows = [{"qid": qid, "predicted_gain": scores.get(qid),
             "explicit_constraints": features[qid].explicit_constraints,
             "threshold_vector": policy.decide(features[qid]).use_vector,
             "fixed_budget_vector": qid in allocated} for qid in qids]
    value = {"schema_version": 1, "model_sha256": args.model_sha256,
             "policy_version": policy.policy_version, "queries_sha256": sha256_of(args.queries),
             "features_sha256": sha256_of(args.features), "fixed_budget": args.fixed_budget,
             "tie_break": "predicted_gain_desc_then_qid_asc", "rows": rows}
    value["experiment_binding_sha256"] = sha256_of(args.experiment_binding)
    value["feature_receipt_sha256"] = sha256_of(args.feature_receipt)
    write(args.output, value)
    return {"queries": len(rows), "fixed_budget_calls": len(allocated),
            "threshold_calls": sum(row["threshold_vector"] for row in rows),
            "predictions_sha256": sha256_of(args.output)}


def development_budget_curve(rows, qrels, runs, rule_calls):
    """Compare score-ordered replay and uniform allocation at fixed call counts."""
    eligible = sorted((row for row in rows if not row["explicit_constraints"]),
                      key=lambda row: (-row["predicted_gain"], row["qid"]))
    qids = [row["qid"] for row in rows]
    lexical = {
        qid: _metric_for_query(qrels[qid], runs["lexical"].get(qid, ()), 10)[:4]
        for qid in qids
    }
    deltas = []
    for row in eligible:
        qid = row["qid"]
        hybrid = _metric_for_query(qrels[qid], runs["hybrid"].get(qid, ()), 10)[:4]
        deltas.append(tuple(right - left for left, right in zip(lexical[qid], hybrid)))
    base = tuple(math.fsum(lexical[qid][i] for qid in qids) for i in range(len(METRICS)))
    full_gain = tuple(math.fsum(delta[i] for delta in deltas) for i in range(len(METRICS)))
    budgets = sorted({rule_calls, *(len(eligible) * percentage // 100 for percentage in BUDGET_PERCENTAGES)})
    points = []
    for budget in budgets:
        probability = budget / len(eligible) if eligible else 0.0
        replay_gain = tuple(math.fsum(delta[i] for delta in deltas[:budget]) for i in range(len(METRICS)))
        points.append({
            "vector_calls": budget,
            "query_trigger_rate": budget / len(qids),
            "gain_replay": {metric: (base[i] + replay_gain[i]) / len(qids)
                            for i, metric in enumerate(METRICS)},
            "uniform_fixed_budget_expectation": {metric: (base[i] + probability * full_gain[i]) / len(qids)
                                                 for i, metric in enumerate(METRICS)},
        })
    return {"scope": "offline_development_replay", "cost_unit": "vector_calls",
            "query_count": len(qids), "eligible_queries": len(eligible),
            "eligible_budget_percentages": list(BUDGET_PERCENTAGES),
            "observed_rule_calls": rule_calls, "points": points}


def development_feature_ablation(inputs_path, binding, rows, records, qrels, runs, learned, budget):
    """Refit each intervention on the original training files and replay at equal cost."""
    paths = read(inputs_path)
    train_names = {"train_features", "train_queries", "train_qrels", "lexical_run",
                   "hybrid_run", "adaptive_receipt", "feature_receipt"}
    required = train_names | {"model", "development_features"}
    require(isinstance(paths, dict) and set(paths) == required
            and all(isinstance(value, str) and value for value in paths.values()),
            "Invalid ablation input file mapping")
    paths = {name: inputs_path.parent / value for name, value in paths.items()}
    hashes = {name: sha256_of(path) for name, path in paths.items()}
    require(hashes["model"] == binding["model_sha256"]
            and hashes["development_features"] == binding["features_sha256"]
            and hashes["train_queries"] == binding["training_queries_sha256"],
            "Ablation inputs disagree with frozen model, features or training cohort")
    model = read(paths["model"])
    expected = {key: binding[key] for key in ("corpus_sha256", "candidate_k", "rrf_c", "feature_version")}
    provenance = model["provenance"]
    require(provenance["binding"] == expected and provenance["split_role"] == "train"
            and all(provenance["inputs_sha256"].get(name) == hashes[name] for name in train_names),
            "Ablation training files disagree with original model provenance")
    train = load_queries(paths["train_queries"])
    normalize = lambda text: "".join(text.casefold().split())
    qids = {row.qid for row in records}
    require(not qids.intersection(row.qid for row in train)
            and not {normalize(row.query) for row in records}.intersection(normalize(row.query) for row in train),
            "Ablation development cohort overlaps training")
    features = load_train_features(paths["development_features"], qids)
    training = {name: paths[name] for name in train_names}
    alpha = provenance["fit"]["alpha"]
    refitted = train_router_gain(**training, alpha=alpha)
    require(model_json(refitted).encode("utf-8") == paths["model"].read_bytes(),
            "Full model refit disagrees with archive")
    full = LinearGainRouter.from_dict(model, hashes["model"])
    eligible = [row["qid"] for row in rows if not row["explicit_constraints"]]
    require(all(row["explicit_constraints"] == features[row["qid"]].explicit_constraints
                and (row["explicit_constraints"] or row["predicted_gain"] == full.predict(features[row["qid"]]))
                and row["threshold_vector"] == full.decide(features[row["qid"]]).use_vector for row in rows),
            "Full model predictions disagree with frozen replay")

    def allocate(policy):
        return set(sorted(eligible, key=lambda qid: (-policy.predict(features[qid]), qid))[:budget])

    full_calls = allocate(full)
    require(full_calls == {row["qid"] for row in rows if row["fixed_budget_vector"]},
            "Full model allocation disagrees with frozen replay")
    ndcg_gains = {
        qid: _metric_for_query(qrels[qid], runs["hybrid"].get(qid, ()), 10)[0]
        - _metric_for_query(qrels[qid], runs["lexical"].get(qid, ()), 10)[0]
        for qid in eligible
    }

    def diagnostics(calls):
        return {"missed_positive_gain": sum(qid not in calls and gain > 0 for qid, gain in ndcg_gains.items()),
                "called_negative_gain": sum(qid in calls and gain < 0 for qid, gain in ndcg_gains.items()),
                "called_zero_gain": sum(qid in calls and gain == 0 for qid, gain in ndcg_gains.items())}

    variants = {}
    for group in FEATURE_GROUPS:
        ablated = train_router_gain(**training, alpha=alpha, omit_feature_group=group)
        ablated_digest = hashlib.sha256(model_json(ablated).encode("utf-8")).hexdigest()
        policy = LinearGainRouter.from_dict(ablated, ablated_digest)
        calls = allocate(policy)
        rankings = {qid: runs["hybrid" if qid in calls else "lexical"].get(qid, []) for qid in qids}
        deltas = {metric: asdict(paired_bootstrap(qrels, learned, rankings, metric=metric))
                  for metric in ("ndcg", "recall", "mrr", "map")}
        variants[group] = {"model_sha256": ablated_digest, "changed_actions": len(calls ^ full_calls),
                           "vector_calls": len(calls), "minus_full": deltas, **diagnostics(calls)}
    require(all(sha256_of(path) == hashes[name] for name, path in paths.items()),
            "An ablation input changed during the analysis")
    return {"scope": "exploratory_offline_development_ablation", "alpha": alpha,
            "input_sha256": hashes, "vector_calls": budget, "eligible_queries": len(eligible),
            "training_queries": len(train), "development_queries": len(records),
            "full": diagnostics(full_calls), "variants": variants}


def score(args):
    binding = load_binding(args.experiment_binding)
    require(sha256_of(args.predictions) == args.predictions_sha256, "Prediction SHA256 disagrees")
    predictions = read(args.predictions)
    require(predictions.get("experiment_binding_sha256") == sha256_of(args.experiment_binding)
            and predictions["model_sha256"] == binding["model_sha256"]
            and predictions["features_sha256"] == binding["features_sha256"]
            and predictions["queries_sha256"] == binding["queries_sha256"],
            "Predictions disagree with the experiment binding")
    require(predictions["schema_version"] == 1 and predictions["queries_sha256"] == sha256_of(args.queries),
            "Predictions are for a different query universe")
    records = load_queries(args.queries)
    qids = [row.qid for row in records]
    universe = set(qids)
    rows = predictions["rows"]
    require([row["qid"] for row in rows] == qids and len(qids) == len(universe),
            "Prediction rows do not cover the ordered universe")
    for row in rows:
        explicit, gain = row["explicit_constraints"], row["predicted_gain"]
        require(all(type(row[key]) is bool for key in
                    ("explicit_constraints", "threshold_vector", "fixed_budget_vector"))
                and ((explicit and gain is None and not row["threshold_vector"] and not row["fixed_budget_vector"])
                     or (not explicit and type(gain) in (float, int) and math.isfinite(gain))),
                "Invalid or constraint-violating predictions")
    qrels_sha256 = sha256_of(args.qrels)
    ablation_inputs = getattr(args, "development_ablation_inputs", None)
    opportunity = getattr(args, "routing_opportunity", False)
    if ablation_inputs is not None or opportunity:
        require(qrels_sha256 == getattr(args, "expected_development_qrels_sha256", None),
                "Offline analysis requires the exact frozen qrels SHA256")
    qrels = load_qrels(args.qrels)
    require(set(qrels) == universe, "Qrels do not cover the exact scoring universe")
    runs, summaries, evidence_formats = {}, {}, {}
    for mode in MODES:
        run = getattr(args, f"{mode}_run")
        metrics = getattr(args, f"{mode}_metrics")
        requests = getattr(args, f"{mode}_requests")
        source = binding["arm_sources"][mode]
        require(sha256_of(run) == source["run_sha256"]
                and sha256_of(metrics) == source["metrics_sha256"]
                and sha256_of(requests) == source["request_log_sha256"],
                f"{mode} archived source hashes disagree with experiment binding")
        ranked = read_rankings(run, universe)
        reported = read(metrics)["per_query"]
        summary = {row["qid"]: row for row in reported}
        require(len(reported) == len(summary) and summary.keys() == universe,
                f"{mode} metrics do not cover the scoring universe")
        require(all(type(summary[qid]["retrieved"]) is int
                    and summary[qid]["retrieved"] == len(ranked.get(qid, [])) for qid in qids),
                f"{mode} ranking depth disagrees with request evidence")
        evidence_formats[mode] = read_requests(requests, mode, qids, ranked, summary,
                                               binding["candidate_k"])
        runs[mode] = ranked
        summaries[mode] = summary
    learned = {"gain_fixed_budget": {}, "gain_train_threshold": {}}
    rule_call_ids = set()
    for row in rows:
        qid = row["qid"]
        route = QueryRouting.from_response(summaries["adaptive"][qid]["routing"], "adaptive")
        require(not route.fallback and (route.reason == "explicit_constraints") == row["explicit_constraints"],
                "Recorded rule route falls back or disagrees with constraint evidence")
        if route.vector_attempted:
            rule_call_ids.add(qid)
        forced = {mode: runs[mode].get(qid, []) for mode in MODES}
        require(forced["adaptive"] == forced[route.executed_mode],
                "Adaptive ranking differs from its recorded forced route")
        for arm, flag in (("gain_fixed_budget", "fixed_budget_vector"),
                          ("gain_train_threshold", "threshold_vector")):
            learned[arm][qid] = forced["hybrid" if row[flag] else "lexical"]
    rule_calls = len(rule_call_ids)
    require(rule_calls == predictions["fixed_budget"] == sum(row["fixed_budget_vector"] for row in rows),
            "Learned and observed rule budgets disagree")
    all_arms = {**runs, **learned}
    # Evaluation accepts document IDs; scores are immaterial after rank order
    # has been validated against the TREC and request evidence.
    summaries = {}
    for arm, rankings in all_arms.items():
        observed = {qid: [(doc_id, 0.0) for doc_id in ranking] for qid, ranking in rankings.items()}
        value = evaluate_run(qrels, observed).to_dict()
        summaries[arm] = {metric: value[metric] for metric in METRICS}
    comparisons = {metric: asdict(paired_bootstrap(
        qrels, runs["adaptive"], learned["gain_fixed_budget"], metric=metric,
    )) for metric in ("ndcg", "recall", "mrr", "map")}
    result = {"predictions_sha256": args.predictions_sha256, "queries": len(qids),
              "experiment_binding_sha256": sha256_of(args.experiment_binding),
              "experiment_binding": binding, "request_evidence_formats": evidence_formats,
              "qrels_sha256": qrels_sha256,
              "vector_calls": {"rules": rule_calls, "gain_fixed_budget": rule_calls,
                               "gain_train_threshold": sum(row["threshold_vector"] for row in rows)},
              "metrics": summaries, "gain_fixed_budget_minus_rules": comparisons}
    if getattr(args, "development_budget_curve", False):
        result["development_budget_curve"] = development_budget_curve(rows, qrels, runs, rule_calls)
    if ablation_inputs is not None:
        result["development_feature_ablation"] = development_feature_ablation(
            ablation_inputs, binding, rows, records, qrels, runs, learned["gain_fixed_budget"], rule_calls)
    if opportunity:
        from evaluation.router_opportunity import analyze_opportunity

        constrained = {row["qid"] for row in rows if row["explicit_constraints"]}
        # The observed adaptive ranking was already checked against lexical
        # for these queries. The forced hybrid arm is outside their action set.
        require(not rule_call_ids & constrained, "Rules called vectors for a protected query")
        result["routing_opportunity"] = analyze_opportunity(
            records, qrels, runs, rule_call_ids, {row["qid"] for row in rows if row["fixed_budget_vector"]},
            constrained)
    if ablation_inputs is not None or opportunity:
        require(sha256_of(args.qrels) == qrels_sha256, "Qrels changed during offline analysis")
    write(args.output, result)
    return result


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    phases = cli.add_subparsers(dest="phase", required=True)
    make = phases.add_parser("freeze")
    verify = phases.add_parser("score")
    for phase in (make, verify):
        phase.add_argument("--experiment-binding", required=True, type=Path)
        phase.add_argument("--queries", required=True, type=Path)
        phase.add_argument("--output", required=True, type=Path)
    for flag in ("training-queries", "features", "feature-receipt", "model"):
        make.add_argument(f"--{flag}", required=True, type=Path)
    make.add_argument("--model-sha256", required=True)
    make.add_argument("--fixed-budget", required=True, type=int)
    for flag in ("predictions", "qrels", *(f"{mode}-{kind}" for mode in MODES for kind in ("run", "metrics", "requests"))):
        verify.add_argument(f"--{flag}", required=True, type=Path)
    verify.add_argument("--predictions-sha256", required=True)
    verify.add_argument("--development-budget-curve", action="store_true",
                        help="Add offline call-budget replay on the bound development split")
    verify.add_argument("--development-ablation-inputs", type=Path,
                        help="JSON mapping for the archived training files, model and dev features")
    verify.add_argument("--routing-opportunity", action="store_true",
                        help="Add a judged oracle and grouped sensitivity for frozen ranks (NumPy required)")
    verify.add_argument("--expected-qrels-sha256", "--expected-development-qrels-sha256",
                        dest="expected_development_qrels_sha256",
                        help="Frozen qrels digest required for ablation or opportunity analysis")
    args = cli.parse_args(argv)
    if args.phase == "score":
        needs_digest = bool(args.development_ablation_inputs or args.routing_opportunity)
        if needs_digest != bool(args.expected_development_qrels_sha256):
            cli.error("provide the frozen qrels SHA256 with ablation or opportunity analysis")
    result = freeze(args) if args.phase == "freeze" else score(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
