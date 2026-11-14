"""Fit the adaptive linear gain policy using only the supplied training split."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import fmean
import sys

from online.linear_gain_router import FEATURE_VERSION, feature_vector, predict_gain
from online.router import QueryFeatures

from .evaluate import _metric_for_query, load_qrels, load_queries
from .router_budget import digest, parse_json, read_adaptive_receipt, read_rankings, require


FEATURE_FIELDS = {"qid", "explicit_constraints", "n_terms", "vocab_coverage", "lexical_hits", "relative_gap"}
FEATURE_GROUPS = {"query_length": (0,), "vocabulary_coverage": (1,),
                  "lexical_hits": (2,), "relative_gap": (3, 4)}
TARGET_DEPTH = 10


def in_unit_interval(value):
    return type(value) in (int, float) and 0 <= value <= 1


def load_train_features(path, qids):
    features = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        row = parse_json(line)
        require(isinstance(row, dict) and set(row) == FEATURE_FIELDS,
                f"Unexpected training feature fields on line {line_number}")
        qid = row["qid"]
        require(isinstance(qid, str) and qid in qids and qid not in features,
                f"Foreign or duplicate training feature qid on line {line_number}")
        explicit, terms, coverage, hits, gap = (
            row[field] for field in ("explicit_constraints", "n_terms", "vocab_coverage",
                                     "lexical_hits", "relative_gap")
        )
        require(type(explicit) is bool and type(terms) is int and terms >= 0
                and type(hits) is int and hits >= 0 and in_unit_interval(coverage)
                and (terms > 0 or coverage == 0)
                and ((hits < 2 and gap is None) or (hits >= 2 and in_unit_interval(gap))),
                f"Invalid BM25 prefix feature values for {qid!r}")
        features[qid] = QueryFeatures(explicit, terms, coverage, hits, gap)
    require(set(features) == qids, "Training features do not cover the exact query universe")
    return features


def solve_positive_definite(matrix, rhs):
    """Solve a small symmetric ridge system by Cholesky factorization."""
    size = len(rhs)
    lower = [[0.0] * size for _ in range(size)]
    for row in range(size):
        for col in range(row + 1):
            residual = matrix[row][col] - math.fsum(
                lower[row][i] * lower[col][i] for i in range(col)
            )
            if row == col:
                require(residual > 0 and math.isfinite(residual),
                        "Ridge system is not numerically positive definite")
                lower[row][col] = math.sqrt(residual)
            else:
                lower[row][col] = residual / lower[col][col]
    forward = [0.0] * size
    for row in range(size):
        forward[row] = (rhs[row] - math.fsum(
            lower[row][col] * forward[col] for col in range(row)
        )) / lower[row][row]
    weights = [0.0] * size
    for row in reversed(range(size)):
        weights[row] = (forward[row] - math.fsum(
            lower[col][row] * weights[col] for col in range(row + 1, size)
        )) / lower[row][row]
    return weights


def fit_ridge(vectors, targets, alpha):
    columns = list(zip(*vectors))
    mean = [fmean(values) for values in columns]
    scale = [math.sqrt(fmean((value - center) ** 2 for value in values)) or 1.0
             for values, center in zip(columns, mean)]
    standardized = [[(value - center) / deviation for value in values]
                    for values, center, deviation in zip(columns, mean, scale)]
    column_means = [fmean(values) for values in standardized]
    centered = [[value - center for value in values]
                for values, center in zip(standardized, column_means)]
    target_mean = fmean(targets)
    centered_target = [value - target_mean for value in targets]
    gram = [[math.fsum(x * y for x, y in zip(left, right)) + (alpha if i == j else 0.0)
             for j, right in enumerate(centered)]
            for i, left in enumerate(centered)]
    rhs = [math.fsum(x * y for x, y in zip(values, centered_target)) for values in centered]
    weights = solve_positive_definite(gram, rhs)
    bias = target_mean - math.fsum(weight * center for weight, center in zip(weights, column_means))
    require(all(math.isfinite(value) for value in (*mean, *scale, *weights, bias)),
            "Ridge fit produced non-finite parameters")
    return mean, scale, weights, bias


def calibrate_threshold(scores, target_calls):
    require(scores and all(math.isfinite(score) for score in scores)
            and 0 <= target_calls <= len(scores), "Invalid training calibration scores or call target")
    counts = Counter(scores)
    zero_calls = math.nextafter(max(counts), math.inf)
    require(math.isfinite(zero_calls), "Could not represent a finite no-call threshold")
    best = (target_calls, 0, zero_calls)
    calls = 0
    for score in sorted(counts, reverse=True):
        calls += counts[score]
        candidate = (abs(calls - target_calls), calls, score)
        best = min(best, candidate)
    _, selected_calls, threshold = best
    return threshold, selected_calls


def model_json(model):
    return json.dumps(model, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def train_router_gain(train_features, train_queries, train_qrels, lexical_run, hybrid_run,
                      adaptive_receipt, *, feature_receipt, alpha=1.0, omit_feature_group=None):
    require(type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0,
            "Ridge alpha must be a positive finite number")
    require(omit_feature_group is None or omit_feature_group in FEATURE_GROUPS,
            "Unknown omitted feature group")
    omitted_columns = FEATURE_GROUPS.get(omit_feature_group, ())
    sources = {name: Path(path) for name, path in {
        "train_features": train_features, "train_queries": train_queries, "train_qrels": train_qrels,
        "lexical_run": lexical_run, "hybrid_run": hybrid_run, "adaptive_receipt": adaptive_receipt,
        "feature_receipt": feature_receipt,
    }.items()}
    source_hashes = {name: digest(path.read_bytes()) for name, path in sources.items()}
    qids = [record.qid for record in load_queries(sources["train_queries"])]
    universe = set(qids)
    require(qids and len(universe) == len(qids), "Training queries are empty or have duplicate qids")
    qrels = load_qrels(sources["train_qrels"])
    require(set(qrels) == universe, "Training qrels do not cover the exact query universe")
    features = load_train_features(sources["train_features"], universe)
    receipt = parse_json(sources["feature_receipt"].read_bytes())
    require(isinstance(receipt, dict) and receipt.get("feature_version") == FEATURE_VERSION
            and receipt.get("features") == receipt.get("rank_matches") == len(qids)
            and type(receipt.get("depth")) is int and receipt["depth"] >= TARGET_DEPTH
            and type(receipt.get("candidate_k")) is int
            and receipt["depth"] <= receipt["candidate_k"] <= 200
            and type(receipt.get("rrf_c")) in (int, float)
            and math.isfinite(receipt["rrf_c"]) and receipt["rrf_c"] > 0,
            "Feature receipt has invalid feature or retrieval identity")
    for field, source in (("features_sha256", "train_features"),
                          ("queries_sha256", "train_queries"),
                          ("lexical_run_sha256", "lexical_run")):
        require(receipt.get(field) == source_hashes[source],
                f"Feature receipt {field} disagrees with its training input")
    corpus_sha = receipt.get("corpus_sha256")
    require(isinstance(corpus_sha, str) and len(corpus_sha) == 64
            and all(char in "0123456789abcdef" for char in corpus_sha),
            "Feature receipt corpus identity is invalid")
    constrained = {qid for qid, values in features.items() if values.explicit_constraints}
    require(receipt.get("constrained") == len(constrained),
            "Feature receipt constrained query count disagrees")
    eligible = [qid for qid in qids if qid not in constrained]
    require(eligible, "Training has no vector-eligible queries")
    lexical = read_rankings(sources["lexical_run"], universe)
    hybrid = read_rankings(sources["hybrid_run"], universe)
    rules, universe_sha, journal_sha = read_adaptive_receipt(
        sources["adaptive_receipt"], qids, constrained, {}, TARGET_DEPTH)
    require(journal_sha is not None, "Training adaptive receipt must bind a finalized request journal")
    targets = []
    vectors = []
    for qid in eligible:
        first = lexical.get(qid, [])
        fused = hybrid.get(qid, [])
        hits = features[qid].lexical_hits
        require(min(hits, TARGET_DEPTH) <= len(first) <= hits,
                f"Training lexical run and BM25 hit count disagree for {qid!r}")
        require(fused and len(fused) >= min(hits, TARGET_DEPTH),
                f"Training hybrid run lacks a corroborated ranking for {qid!r}")
        lexical_ndcg = _metric_for_query(qrels[qid], first, TARGET_DEPTH, TARGET_DEPTH)[0]
        hybrid_ndcg = _metric_for_query(qrels[qid], fused, TARGET_DEPTH, TARGET_DEPTH)[0]
        targets.append(hybrid_ndcg - lexical_ndcg)
        vectors.append(tuple(0.0 if i in omitted_columns else value
                             for i, value in enumerate(feature_vector(features[qid]))))
    mean, scale, weights, bias = fit_ridge(vectors, targets, alpha)
    scores = [predict_gain(features[qid], mean, scale, weights, bias) for qid in eligible]
    rule_calls = sum(rules[qid].vector_attempted for qid in eligible)
    threshold, selected_calls = calibrate_threshold(scores, rule_calls)
    require(sum(score >= threshold for score in scores) == selected_calls,
            "Calibrated threshold disagrees with the runtime predicate")
    require(all(digest(path.read_bytes()) == source_hashes[name] for name, path in sources.items()),
            "A training input changed during fitting")
    model = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "bias": bias,
        "threshold": threshold,
        "provenance": {
            "binding": {"corpus_sha256": corpus_sha,
                        "candidate_k": receipt["candidate_k"],
                        "rrf_c": receipt["rrf_c"], "feature_version": FEATURE_VERSION},
            "inputs_sha256": {**source_hashes, "adaptive_journal": journal_sha,
                              "ordered_query_universe": universe_sha},
            "split_role": "train",
            "training_queries": len(qids),
            "eligible_queries": len(eligible),
            "excluded_explicit_constraints": len(constrained),
            "feature_order": ["log1p_n_terms", "vocab_coverage", "log1p_lexical_hits",
                              "relative_gap_or_zero", "relative_gap_missing"],
            "target": "hybrid_ndcg_at_10_minus_lexical_ndcg_at_10",
            "fit": {"method": "ridge_with_unpenalized_intercept", "alpha": alpha,
                    "standardization": "eligible_training_population_std_zero_scale_to_one"},
            "calibration": {
                "policy_reference": "rules-v1-compatible adaptive receipt",
                "predicate": "predicted_gain >= threshold",
                "selection": "minimize_absolute_call_error_then_fewer_calls",
                "rule_vector_calls": rule_calls,
                "selected_vector_calls": selected_calls,
                "absolute_call_error": abs(selected_calls - rule_calls),
            },
        },
    }
    if omit_feature_group is not None:
        model["provenance"]["omitted_feature_group"] = omit_feature_group
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("train-features", "train-queries", "train-qrels", "lexical-run", "hybrid-run",
                 "adaptive-receipt", "feature-receipt", "output-model"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    options = vars(parser.parse_args(argv))
    target = options.pop("output_model")
    try:
        require(not target.exists(), "Output model already exists")
        model = train_router_gain(**options)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(model_json(model))
    except (ValueError, TypeError, KeyError, OSError, OverflowError) as error:
        print(f"train router gain error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
