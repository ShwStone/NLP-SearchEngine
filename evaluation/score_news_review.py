"""Rescore the frozen RFC-0002 rankings using verified relevance labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .audit_crawled_dataset import (
    NEWS_ARMS, QRELS_MANIFEST_SHA256, RUN_STATE_SHA256, SELECTED_MANIFEST_SHA256,
    coverage_for_rankings, load_frozen_rankings, read_json, require,
)
from .compare import paired_bootstrap
from .evaluate import evaluate_run


REVIEW_PLAN_SHA256 = "c4f137c3c7c77a150f4bcd6cc60aaf8a6be17eb4db9bb2d35e2b16916d0cccee"
REVIEW_PACKET_SHA256 = "235ec7e372bb817bc65855f6ed8c110a193e11cf0bf7187e6e86146d0485d466"
REVIEW_MAP_SHA256 = "fa4404df73dc204d66bfdcc2f9b741dd8c3cd3dbd303a2c23ad5d7defb323e5a"
REVIEW_ASSIGNMENTS = {"reviewer-a": 2485, "reviewer-b": 2484, "reviewer-c": 2484, "reviewer-d": 2484}
REVIEW_PAIRS = 8281
REVIEW_OVERLAP = 1656
TEST_PAIRS = 6214
TEST_QUERIES = 150
ASSISTANT_SOURCE = "assistant_semantic_review_not_human_gold"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20261028
METRIC_FIELDS = {"ndcg_at_10": "ndcg", "recall_at_10": "recall", "mrr_at_10": "mrr", "map_at_100": "map"}


def load_review_inputs(evidence_root):
    selection = read_json(evidence_root / "news-query-design-20261102/selected-200.manifest.json",
                          SELECTED_MANIFEST_SHA256)
    test_ids = set(selection["splits"]["test"])
    dev_ids = set(selection["splits"]["dev"])
    require(len(test_ids) == TEST_QUERIES and len(dev_ids) == 50 and not test_ids & dev_ids,
            "Frozen news split does not match the review")
    historical = read_json(evidence_root / "news-formal-pooled-qrels-20261104/manifest.json",
                           QRELS_MANIFEST_SHA256)
    pool_dir = evidence_root / "news-formal-http-pool-rm3-resume-20261103/pool"
    pool_digest = historical["pool_manifest_sha256"]
    pool = read_json(pool_dir / "manifest.json", pool_digest)
    provenance = pool_dir / "pool-provenance.jsonl"
    pool_bytes = provenance.read_bytes()
    require(hashlib.sha256(pool_bytes).hexdigest() == pool["files"][provenance.name],
            "SHA-256 mismatch for the pool provenance")
    rows = (json.loads(line) for line in pool_bytes.splitlines())
    pool_rows = [(row["query_id"], row["doc_id"]) for row in rows]
    pairs = set(pool_rows)
    require(len(pool_rows) == len(pairs) == pool["pool_pairs"] == REVIEW_PAIRS
            and {qid for qid, _ in pairs} == dev_ids | test_ids
            and sum(qid in test_ids for qid, _ in pairs) == TEST_PAIRS,
            "Frozen news pool coverage does not match the review")
    state, runs = load_frozen_rankings(evidence_root / "news-formal-quality-test-20261104/formal-test",
                                      QRELS_MANIFEST_SHA256, test_ids)
    require(state["status"] == "COMPLETE" and state["split"] == "test"
            and state["n_queries"] == TEST_QUERIES and state["metrics_cutoff"] == 10
            and state["retrieval_and_map_cutoff"] == 100
            and state["pool_manifest_sha256"] == pool_digest
            and state["selected_queries_sha256"] == selection["selected_queries_sha256"]
            == historical["selected_queries_sha256"],
            "Frozen news rankings are bound to another experiment")
    return test_ids, pairs, state, runs


def verify_intake(receipt):
    def judgment_file(row, count):
        return (isinstance(row, dict) and type(row.get("pair_count")) is int
                and row["pair_count"] == count and isinstance(row.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None)

    require(receipt.get("schema_version") == 1 and receipt.get("label_source") == "reviewer_submissions"
            and receipt.get("pair_count") == REVIEW_PAIRS and receipt.get("overlap_pair_count") == REVIEW_OVERLAP,
            "Intake does not describe the frozen four-reviewer plan")
    for field, digest in (("plan", REVIEW_PLAN_SHA256), ("packet", REVIEW_PACKET_SHA256),
                          ("private_map", REVIEW_MAP_SHA256)):
        binding = receipt.get(field)
        require(isinstance(binding, dict) and binding.get("sha256") == digest,
                f"Intake {field} differs from the frozen review")
    reviewers = receipt.get("reviewers")
    require(isinstance(reviewers, dict) and reviewers.keys() == REVIEW_ASSIGNMENTS.keys()
            and all(judgment_file(reviewers[name], count) for name, count in REVIEW_ASSIGNMENTS.items()),
            "Intake reviewer coverage differs from the frozen assignments")
    disputes = receipt.get("disputed_pair_count")
    require(type(disputes) is int and 0 <= disputes <= REVIEW_OVERLAP
            and "adjudication" in receipt, "Intake lacks the dispute count or adjudication record")
    adjudication = receipt["adjudication"]
    require((not disputes and adjudication is None) or judgment_file(adjudication, disputes),
            "Intake adjudication does not cover the recorded disputes")
    preparation = receipt.get("pre_adjudication")
    require(isinstance(preparation, dict) and isinstance(preparation.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", preparation["sha256"]) is not None,
            "Intake lacks the frozen pre-adjudication receipt")
    return preparation["sha256"]


def load_review_grades(qrels, intake, expected_intake_sha256, pairs):
    receipt = read_json(intake, expected_intake_sha256)
    preparation_sha256 = verify_intake(receipt)
    grades, digest = read_grades(qrels, receipt["qrels"]["sha256"], pairs)
    return grades, digest, preparation_sha256


def read_grades(qrels, expected_sha256, pairs):
    payload = qrels.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_sha256, "Qrels SHA-256 differs from the verified label record")
    grades = {}
    for row in payload.decode("utf-8").splitlines():
        qid, iteration, doc_id, grade = row.split()
        pair = (qid, doc_id)
        require(iteration == "0" and grade in ("0", "1", "2") and pair not in grades,
                "Invalid or repeated qrels row")
        grades[pair] = int(grade)
    require(set(grades) == pairs, "Qrels do not cover the exact frozen news pool")
    return grades, digest


def score_news_review(evidence_root, qrels, intake, expected_intake_sha256):
    test_ids, pairs, state, runs = load_review_inputs(evidence_root)
    grades, qrels_digest, preparation_sha256 = load_review_grades(qrels, intake, expected_intake_sha256, pairs)
    return score_rankings(test_ids, state, runs, grades, "reviewer_submissions", {
        "qrels_sha256": qrels_digest, "intake_sha256": expected_intake_sha256,
        "pre_adjudication_sha256": preparation_sha256, "review_plan_sha256": REVIEW_PLAN_SHA256,
    })


def score_news_assistant(evidence_root, *, exclude_source_seeds=False):
    test_ids, pairs, state, runs = load_review_inputs(evidence_root)
    directory = evidence_root / "news-formal-pooled-qrels-20261104"
    manifest = read_json(directory / "manifest.json", QRELS_MANIFEST_SHA256)
    require(manifest["status"] == "POOLED_QRELS_COMPLETE"
            and manifest["reviewer_provenance"] == ASSISTANT_SOURCE,
            "Expected the frozen assistant label provenance")
    grades, digest = read_grades(directory / "all.qrels.tsv", manifest["files"]["all.qrels.tsv"], pairs)
    identity = {
        "qrels_sha256": digest, "qrels_manifest_sha256": QRELS_MANIFEST_SHA256,
    }
    if not exclude_source_seeds:
        return score_rankings(test_ids, state, runs, grades, ASSISTANT_SOURCE, identity)

    seed_path = evidence_root / "news-judgments-20261102/seed-judgments-200.validated.jsonl"
    seed_bytes = seed_path.read_bytes()
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    require(seed_sha256 == manifest["seed_judgments_sha256"], "Frozen source-seed judgments differ")
    seed_rows = [json.loads(line) for line in seed_bytes.splitlines()]
    seed_pairs = {(row["query_id"], row["doc_id"]): row["relevance"] for row in seed_rows}
    require(len(seed_pairs) == len(seed_rows) == manifest["seed_reused"]
            and all(grades.get(pair) == grade for pair, grade in seed_pairs.items()),
            "Source seeds do not bind unique frozen judgments")
    retained = {pair: grade for pair, grade in grades.items() if pair not in seed_pairs}
    result = score_rankings(test_ids, state, runs, retained, ASSISTANT_SOURCE, {
        **identity, "excluded_source_seed_sha256": seed_sha256,
    })
    result["scope"] = "frozen_news20k_source_seeds_treated_as_unjudged"
    result["excluded_test_judgments"] = sum(qid in test_ids for qid, _ in seed_pairs)
    return result


def score_rankings(test_ids, state, runs, grades, label_source, label_identity):
    test_qrels = {qid: {} for qid in sorted(test_ids)}
    for (qid, doc_id), grade in grades.items():
        if qid in test_qrels:
            test_qrels[qid][doc_id] = grade
    scores = {}
    for arm, run in runs.items():
        summary = evaluate_run(test_qrels, run, k=10, map_k=100)
        scores[arm] = {field: getattr(summary, field) for field in METRIC_FIELDS}
    comparison = {
        field: paired_bootstrap(test_qrels, runs["hybrid"], runs["adaptive"], metric=metric, k=10, map_k=100,
                                samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED).to_dict()
        for field, metric in METRIC_FIELDS.items()
    }
    return {
        "schema_version": 1, "scope": "frozen_news20k_test_rankings_rescored",
        "label_source": label_source, "test_queries": len(test_qrels),
        "test_queries_without_positive": sum(not any(rows.values()) for rows in test_qrels.values()),
        "test_grade_counts": dict(sorted(Counter(grade for rows in test_qrels.values()
                                                for grade in rows.values()).items())),
        "identity": {
            **label_identity, "run_state_sha256": RUN_STATE_SHA256,
            "pool_manifest_sha256": state["pool_manifest_sha256"],
            "run_sha256": {arm: state["arms"][arm]["sha256"] for arm in NEWS_ARMS},
        },
        "metrics": scores, "judged_coverage": coverage_for_rankings(runs, grades),
        "adaptive_minus_hybrid": comparison,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assistant-labels", action="store_true", help="Use the frozen assistant qrels and manifest")
    source.add_argument("--qrels", type=Path, help="Qrels from optional independent review")
    parser.add_argument("--exclude-source-seeds", action="store_true",
                        help="Sensitivity analysis: treat the frozen source seeds as unjudged")
    parser.add_argument("--intake", type=Path)
    parser.add_argument("--intake-sha256", help="Trusted digest recorded after successful intake")
    parser.add_argument("--out", required=True, type=Path, help="Fresh aggregate JSON path outside Git")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        parser.error("output already exists")
    if args.assistant_labels:
        if args.intake is not None or args.intake_sha256 is not None:
            parser.error("--assistant-labels does not use reviewer intake")
        result = score_news_assistant(args.evidence_root, exclude_source_seeds=args.exclude_source_seeds)
    else:
        if args.exclude_source_seeds:
            parser.error("--exclude-source-seeds requires --assistant-labels")
        if args.intake is None or args.intake_sha256 is None:
            parser.error("--qrels requires --intake and --intake-sha256")
        result = score_news_review(args.evidence_root, args.qrels, args.intake, args.intake_sha256)
    encoded = json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
    with args.out.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")


if __name__ == "__main__":
    main()
