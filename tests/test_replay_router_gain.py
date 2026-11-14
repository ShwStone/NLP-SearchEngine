"""Replay rejects missing request evidence and altered experiment inputs."""
import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.evaluate import EvaluationError
from online.linear_gain_router import FEATURE_VERSION
from scripts.replay_router_gain import MODES, development_budget_curve, freeze, main, score


class ReplayGainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = argparse.Namespace()
        self.args.queries = self.write("queries.tsv", "q\tquery\n")
        self.args.qrels = self.write("qrels.tsv", "q 0 d 1\n")
        self.args.output = self.root / "result.json"
        self.binding = {"schema_version": 1, "corpus_sha256": "a" * 64,
                        "candidate_k": 200, "rrf_c": 60, "hnsw_ef": 256,
                        "feature_version": FEATURE_VERSION, "model_sha256": "b" * 64,
                        "features_sha256": "c" * 64,
                        "queries_sha256": self.sha(self.args.queries),
                        "training_queries_sha256": "d" * 64, "arm_sources": {}}
        for mode in MODES:
            route = {"requested_mode": mode, "executed_mode": "hybrid" if mode == "hybrid" else "lexical",
                     "reason": "lexical_sufficient" if mode == "adaptive" else f"forced_{mode}",
                     "vector_attempted": mode == "hybrid", "vector_used": mode == "hybrid", "fallback": False}
            run = self.write(f"{mode}.trec", "q Q0 d 1 1 fixture\n")
            metrics = self.write(f"{mode}.json", {"per_query": [{"qid": "q", "retrieved": 1,
                                                               "routing": route}]})
            requests = self.write(f"{mode}.jsonl", json.dumps({"query_id": "q", "status": 200,
                    "returned": 1, "latency_ms": 1.0, "routing": route}) + "\n")
            for name, path in (("run", run), ("metrics", metrics), ("requests", requests)):
                setattr(self.args, f"{mode}_{name}", path)
            self.binding["arm_sources"][mode] = {"run_sha256": self.sha(run),
                    "metrics_sha256": self.sha(metrics), "request_log_sha256": self.sha(requests)}
        self.args.experiment_binding = self.write("binding.json", self.binding)
        self.predictions = {"schema_version": 1, "model_sha256": "b" * 64,
                "features_sha256": "c" * 64, "queries_sha256": self.sha(self.args.queries),
                "experiment_binding_sha256": self.sha(self.args.experiment_binding), "fixed_budget": 0,
                "rows": [{"qid": "q", "explicit_constraints": False, "predicted_gain": -.1,
                          "threshold_vector": False, "fixed_budget_vector": False}]}
        self.args.predictions = self.write("predictions.json", self.predictions)
        self.args.predictions_sha256 = self.sha(self.args.predictions)

    def write(self, name, value):
        path = self.root / name
        path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")
        return path

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def refresh_binding(self):
        self.write("binding.json", self.binding)
        self.predictions["experiment_binding_sha256"] = self.sha(self.args.experiment_binding)
        self.write("predictions.json", self.predictions)
        self.args.predictions_sha256 = self.sha(self.args.predictions)

    def alter_request(self, **changes):
        row = json.loads(self.args.hybrid_requests.read_text())
        row.update(changes)
        self.write("hybrid.jsonl", json.dumps(row) + "\n")
        self.binding["arm_sources"]["hybrid"]["request_log_sha256"] = self.sha(self.args.hybrid_requests)
        self.refresh_binding()

    def test_valid_archived_requests_are_classified_and_bound(self):
        result = score(self.args)
        self.assertEqual(result["metrics"]["gain_fixed_budget"]["ndcg_at_10"], 1.0)
        self.assertEqual(result["experiment_binding"]["corpus_sha256"], "a" * 64)
        self.assertIn("legacy_ordered_http200", result["request_evidence_formats"]["hybrid"])

    def test_curve_is_optional_and_uses_validated_score_inputs(self):
        self.assertNotIn("development_budget_curve", score(self.args))
        self.args.output = self.root / "curve.json"
        self.args.development_budget_curve = True
        curve = score(self.args)["development_budget_curve"]
        self.assertEqual([point["vector_calls"] for point in curve["points"]], [0, 1])
        self.write("hybrid.trec", "q Q0 foreign 1 1 fixture\n")
        self.args.output = self.root / "invalid-curve.json"
        with self.assertRaisesRegex(EvaluationError, "source hashes"):
            score(self.args)

    def test_curve_preserves_lexical_constraints_and_uses_deterministic_ties(self):
        rows = [
            {"qid": "z", "explicit_constraints": False, "predicted_gain": 0.5},
            {"qid": "a", "explicit_constraints": False, "predicted_gain": 0.5},
            {"qid": "c", "explicit_constraints": True, "predicted_gain": None},
        ]
        qrels = {row["qid"]: {"right": 1.0} for row in rows}
        runs = {"lexical": {"z": ["right"], "c": ["right"]},
                "hybrid": {"a": ["right"]}}
        curve = development_budget_curve(rows, qrels, runs, 1)
        self.assertEqual([point["vector_calls"] for point in curve["points"]], [0, 1, 2])
        zero, one, full = curve["points"]
        self.assertAlmostEqual(zero["gain_replay"]["ndcg_at_10"], 2 / 3)
        self.assertAlmostEqual(one["query_trigger_rate"], 1 / 3)
        self.assertEqual(one["gain_replay"]["ndcg_at_10"], 1.0)
        self.assertAlmostEqual(one["uniform_fixed_budget_expectation"]["ndcg_at_10"], 2 / 3)
        self.assertAlmostEqual(full["gain_replay"]["ndcg_at_10"], 2 / 3)

    def test_ablation_rejects_unpinned_or_changed_development_qrels(self):
        self.args.development_ablation_inputs = self.root / "unused-map.json"
        with self.assertRaisesRegex(EvaluationError, "frozen qrels SHA256"):
            score(self.args)
        self.args.expected_development_qrels_sha256 = self.sha(self.args.qrels)
        self.args.qrels.write_text("q 0 d 2\n", encoding="utf-8")
        with self.assertRaisesRegex(EvaluationError, "frozen qrels SHA256"):
            score(self.args)
        self.assertFalse(self.args.output.exists())

    def test_opportunity_requires_frozen_qrels_before_scoring(self):
        self.args.routing_opportunity = True
        with self.assertRaisesRegex(EvaluationError, "frozen qrels SHA256"):
            score(self.args)
        self.assertFalse(self.args.output.exists())

    @unittest.skipUnless(find_spec("numpy"), "NumPy required for grouped sensitivity")
    def test_opportunity_cli_scores_and_keeps_original_result_fields(self):
        previous = score(self.args)
        self.args.output = self.root / "opportunity.json"
        arguments = ["score", "--routing-opportunity", "--expected-qrels-sha256", self.sha(self.args.qrels)]
        for name, value in vars(self.args).items():
            arguments.extend([f"--{name.replace('_', '-')}", str(value)])
        with redirect_stdout(StringIO()):
            self.assertEqual(main(arguments), 0)
        current = json.loads(self.args.output.read_bytes())
        opportunity = current.pop("routing_opportunity")
        self.assertEqual(current, previous)
        self.assertEqual(opportunity["vector_call_budget"], 0)
        self.assertEqual(opportunity["observed_policy_grouped_sensitivity"]["group_count"], 1)

    def test_foreign_corpus_binding_rejected_by_frozen_identity(self):
        self.binding["corpus_sha256"] = "e" * 64
        self.write("binding.json", self.binding)
        with self.assertRaisesRegex(EvaluationError, "experiment binding"):
            score(self.args)

    def test_changed_run_rejected_even_when_depth_unchanged(self):
        self.write("hybrid.trec", "q Q0 foreign 1 1 fixture\n")
        with self.assertRaisesRegex(EvaluationError, "source hashes"):
            score(self.args)

    def test_failed_missing_foreign_and_wrong_depth_requests_rejected(self):
        for changes in ({"status": 500}, {"query_id": "foreign"}, {"returned": 0}):
            with self.subTest(changes=changes):
                self.alter_request(**changes)
                with self.assertRaises(EvaluationError):
                    score(self.args)
        self.write("hybrid.jsonl", "")
        self.binding["arm_sources"]["hybrid"]["request_log_sha256"] = self.sha(self.args.hybrid_requests)
        self.refresh_binding()
        with self.assertRaises(EvaluationError):
            score(self.args)

    def test_request_route_must_match_forced_mode(self):
        row = json.loads(self.args.hybrid_requests.read_text())
        row["routing"]["requested_mode"] = "lexical"
        self.alter_request(routing=row["routing"])
        with self.assertRaisesRegex(EvaluationError, "requested_mode"):
            score(self.args)

    def current_receipt(self):
        route = json.loads(self.args.adaptive_requests.read_text())["routing"]
        request = {"mode": "adaptive", "scorer": "bm25", "expander": "none", "k": 100}
        outcome = {"qid": "q", "status": "SUCCESS", "http_status": 200,
                   "latency_ms": 1.0, "routing": route}
        universe = hashlib.sha256(b'["q"]').hexdigest()
        events = [
            {"event": "START", "sequence": 0, "request": request, "query_ids": ["q"],
             "query_universe_sha256": universe, "policy": None},
            {"event": "BEGIN", "sequence": 1, "attempt": 1, "outcome": {
                "qid": "q", "status": "INCOMPLETE", "http_status": None,
                "latency_ms": None, "routing": None}},
            {"event": "OUTCOME", "sequence": 2, "attempt": 1, "outcome": outcome},
        ]
        journal = self.write("adaptive.events.jsonl", "".join(json.dumps(row) + "\n" for row in events))
        receipt = {"schema_version": 1, "status": "COMPLETE", "request": request,
                   "query_ids": ["q"], "query_universe_sha256": universe, "policy": None,
                   "requested_queries": 1, "attempted_requests": 1, "http_200": 1,
                   "routing_observed_queries": 1, "expected_unsupported": 0, "unexpected_errors": 0,
                   "last_event_sequence": 2, "http_success_coverage": 1.0,
                   "protocol_accounted_coverage": 1.0, "inflight_outcome": None,
                   "outcomes": [outcome], "last_outcome": outcome,
                   "journal": {"file": journal.name, "sha256": self.sha(journal)}}
        self.args.adaptive_requests = self.write("adaptive.requests.json", receipt)
        self.binding["arm_sources"]["adaptive"]["request_log_sha256"] = self.sha(self.args.adaptive_requests)
        self.refresh_binding()
        return journal

    def test_current_receipt_requires_matching_complete_journal(self):
        self.current_receipt()
        result = score(self.args)
        self.assertEqual(result["request_evidence_formats"]["adaptive"], "complete_receipt_with_journal")

    def test_changed_current_journal_rejected(self):
        journal = self.current_receipt()
        journal.write_bytes(journal.read_bytes() + b'{}\n')
        with self.assertRaisesRegex(EvaluationError, "journal SHA256"):
            score(self.args)

    def test_freeze_requires_exact_feature_receipt_corpus_binding(self):
        self.args.training_queries = self.write("train.tsv", "train\ttraining\n")
        self.args.features = self.write("features.jsonl", json.dumps({"qid": "q",
            "explicit_constraints": False, "n_terms": 1, "vocab_coverage": 1,
            "lexical_hits": 1, "relative_gap": None}) + "\n")
        model_binding = {key: self.binding[key] for key in
                         ("corpus_sha256", "candidate_k", "rrf_c", "feature_version")}
        model = {"schema_version": 1, "feature_version": FEATURE_VERSION, "mean": [0] * 5,
                 "scale": [1] * 5, "weights": [0] * 5, "bias": 0, "threshold": 1,
                 "provenance": {"binding": model_binding,
                     "inputs_sha256": {"train_queries": self.sha(self.args.training_queries)}}}
        self.args.model = self.write("model.json", model)
        self.args.model_sha256 = self.sha(self.args.model)
        self.args.fixed_budget = 0
        self.binding.update(model_sha256=self.args.model_sha256,
                            features_sha256=self.sha(self.args.features),
                            training_queries_sha256=self.sha(self.args.training_queries))
        self.write("binding.json", self.binding)
        receipt = {**model_binding, "features_sha256": self.sha(self.args.features),
                   "queries_sha256": self.sha(self.args.queries),
                   "lexical_run_sha256": self.sha(self.args.lexical_run), "features": 1,
                   "rank_matches": 1, "constrained": 0, "depth": 100}
        self.args.feature_receipt = self.write("features.receipt.json", receipt)
        self.assertEqual(freeze(self.args)["queries"], 1)
        receipt["corpus_sha256"] = "e" * 64
        self.write("features.receipt.json", receipt)
        self.args.output = self.root / "must-not-exist.json"
        with self.assertRaisesRegex(EvaluationError, "Feature receipt"):
            freeze(self.args)
        self.assertFalse(self.args.output.exists())


if __name__ == "__main__":
    unittest.main()
