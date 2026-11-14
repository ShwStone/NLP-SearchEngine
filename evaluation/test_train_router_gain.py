"""Deterministic train-only ridge and runtime threshold calibration checks."""

from contextlib import redirect_stderr
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.evaluate import EvaluationError, QueryRouting, evaluate_http, load_qrels, load_queries
from evaluation.train_router_gain import calibrate_threshold, main, train_router_gain
from online.linear_gain_router import FEATURE_VERSION, LinearGainRouter, predict_gain
from online.router import QueryFeatures


class TrainRouterGainTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.files = {name: root / filename for name, filename in {
            "train_features": "train.features.jsonl", "train_queries": "train.queries.tsv",
            "train_qrels": "train.qrels.tsv", "lexical_run": "lexical.trec", "hybrid_run": "hybrid.trec",
            "adaptive_receipt": "adaptive.requests.json",
            "feature_receipt": "feature-receipt.json",
        }.items()}
        self.output = root / "model.json"
        self.files["train_queries"].write_text("a\tfirst\nb\tsecond\nc\texplicit\n", encoding="utf-8")
        self.files["train_qrels"].write_text("a 0 hit 1\nb 0 hit 1\nc 0 hit 1\n", encoding="utf-8")
        self.files["lexical_run"].write_text("a Q0 miss 1 1 fixture\nb Q0 hit 1 1 fixture\n", encoding="utf-8")
        self.files["hybrid_run"].write_text("a Q0 hit 1 1 fixture\nb Q0 miss 1 1 fixture\n", encoding="utf-8")
        self.feature_rows = [
            {"qid": "a", "explicit_constraints": False, "n_terms": 4, "vocab_coverage": .25,
             "lexical_hits": 1, "relative_gap": None},
            {"qid": "b", "explicit_constraints": False, "n_terms": 4, "vocab_coverage": .75,
             "lexical_hits": 1, "relative_gap": None},
            {"qid": "c", "explicit_constraints": True, "n_terms": 900, "vocab_coverage": .1,
             "lexical_hits": 900, "relative_gap": .123},
        ]
        self.write_features()
        routes = [
            QueryRouting("adaptive", "hybrid", "low_vocab_coverage", True, True, False),
            QueryRouting("adaptive", "lexical", "lexical_sufficient", False, False, False),
            QueryRouting("adaptive", "lexical", "explicit_constraints", False, False, False),
        ]
        with patch("evaluation.evaluate._request_search", side_effect=[([], 2.0, route) for route in routes]):
            evaluate_http("http://unused", load_queries(self.files["train_queries"]),
                          load_qrels(self.files["train_qrels"]), mode="adaptive",
                          request_output=self.files["adaptive_receipt"])

    def write_features(self):
        self.files["train_features"].write_text("".join(
            json.dumps(row) + "\n" for row in self.feature_rows
        ), encoding="utf-8")
        receipt = {
            "features": len(self.feature_rows), "rank_matches": len(self.feature_rows),
            "constrained": sum(row["explicit_constraints"] for row in self.feature_rows),
            "feature_version": FEATURE_VERSION, "candidate_k": 200, "rrf_c": 60.0, "depth": 100,
            "corpus_sha256": "a" * 64,
            "features_sha256": hashlib.sha256(self.files["train_features"].read_bytes()).hexdigest(),
            "queries_sha256": hashlib.sha256(self.files["train_queries"].read_bytes()).hexdigest(),
            "lexical_run_sha256": hashlib.sha256(self.files["lexical_run"].read_bytes()).hexdigest(),
        }
        self.files["feature_receipt"].write_text(json.dumps(receipt), encoding="utf-8")

    def train(self, **kwargs):
        return train_router_gain(**self.files, **kwargs)

    def test_train_only_ridge_calibrates_runtime_boundary(self):
        model = self.train()
        self.assertEqual(model, self.train())
        self.assertEqual(model["feature_version"], FEATURE_VERSION)
        self.assertEqual(model["mean"], [math.log1p(4), .5, math.log1p(1), 0.0, 1.0])
        self.assertEqual(model["scale"], [1.0, .25, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(model["weights"][1], -2 / 3)
        self.assertEqual(model["bias"], 0.0)
        scores = [
            predict_gain(QueryFeatures(False, 4, coverage, 1, None),
                         model["mean"], model["scale"], model["weights"], model["bias"])
            for coverage in (.25, .75)
        ]
        self.assertEqual(model["threshold"], scores[0])
        self.assertEqual(sum(score >= model["threshold"] for score in scores), 1)
        router = LinearGainRouter.from_dict(model, "a" * 64)
        self.assertTrue(router.decide(QueryFeatures(False, 4, .25, 1, None)).use_vector)
        self.assertFalse(router.decide(QueryFeatures(False, 4, .75, 1, None)).use_vector)
        self.assertFalse(router.decide(QueryFeatures(True, 4, .25, 1, None)).use_vector)
        self.assertEqual(model["provenance"]["excluded_explicit_constraints"], 1)
        calibration = model["provenance"]["calibration"]
        self.assertEqual(calibration["rule_vector_calls"], 1)
        self.assertEqual(calibration["selected_vector_calls"], 1)
        self.assertEqual(model["provenance"]["inputs_sha256"]["train_features"],
                         hashlib.sha256(self.files["train_features"].read_bytes()).hexdigest())
        self.assertAlmostEqual(self.train(alpha=2.0)["weights"][1], -.5)

    def test_calibration_ties_choose_lower_calls_and_endpoints(self):
        scores = [.5, .5, -.2]
        threshold, calls = calibrate_threshold(scores, 1)
        self.assertTrue(math.isfinite(threshold))
        self.assertGreater(threshold, .5)
        self.assertEqual(calls, 0)
        self.assertEqual(calibrate_threshold(scores, 2), (.5, 2))
        self.assertEqual(calibrate_threshold(scores, 3), (-.2, 3))
        self.assertEqual(calibrate_threshold(scores, 0)[1], 0)

    def test_ablation_refits_and_removes_both_relative_gap_columns(self):
        self.feature_rows[1].update(lexical_hits=2, relative_gap=0.7)
        for name in ("lexical_run", "hybrid_run"):
            with self.files[name].open("a", encoding="utf-8") as run:
                run.write("b Q0 extra 2 0 fixture\n")
        self.write_features()
        original = self.train()
        self.assertNotEqual(original["weights"][3:], [0.0, 0.0])
        without_gap = self.train(omit_feature_group="relative_gap")
        self.assertEqual(without_gap["weights"][3:], [0.0, 0.0])
        self.assertEqual(without_gap["provenance"]["omitted_feature_group"], "relative_gap")
        self.assertEqual(without_gap["provenance"]["inputs_sha256"], original["provenance"]["inputs_sha256"])
        policy = LinearGainRouter.from_dict(without_gap, "a" * 64)
        self.assertEqual(policy.predict(QueryFeatures(False, 4, .25, 2, .2)),
                         policy.predict(QueryFeatures(False, 4, .25, 2, .8)))
        self.assertFalse(policy.decide(QueryFeatures(True, 4, .25, 2, .2)).use_vector)
        with self.assertRaisesRegex(EvaluationError, "feature group"):
            self.train(omit_feature_group="unknown")

    def test_malformed_features_and_incomplete_rankings_fail(self):
        self.feature_rows[0]["target"] = 1.0
        self.write_features()
        with self.assertRaisesRegex(EvaluationError, "feature fields"):
            self.train()
        self.feature_rows[0].pop("target")
        self.feature_rows[0]["relative_gap"] = float("nan")
        self.write_features()
        with self.assertRaisesRegex(EvaluationError, "feature values"):
            self.train()
        self.feature_rows[0]["relative_gap"] = None
        self.write_features()
        self.files["hybrid_run"].write_text("", encoding="utf-8")
        with self.assertRaisesRegex(EvaluationError, "hybrid run lacks"):
            self.train()

    def test_feature_constraints_and_finalized_receipt_must_agree(self):
        self.feature_rows[0]["explicit_constraints"] = True
        self.write_features()
        with self.assertRaisesRegex(EvaluationError, "explicit_constraints"):
            self.train()
        self.feature_rows[0]["explicit_constraints"] = False
        self.write_features()
        receipt_path = self.files["adaptive_receipt"]
        receipt = json.loads(receipt_path.read_bytes())
        journal_path = receipt_path.with_name(receipt["journal"]["file"])
        journal_path.write_bytes(journal_path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(EvaluationError, "journal SHA256"):
            self.train()

    def test_cli_writes_fresh_model_and_protects_existing_file(self):
        args = []
        for key, path in self.files.items():
            args.extend([f"--{key.replace('_', '-')}", str(path)])
        args.extend(["--output-model", str(self.output)])
        self.assertEqual(main(args), 0)
        raw = self.output.read_bytes()
        self.assertEqual(json.loads(raw), self.train())
        with redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(args), 2)
        self.assertIn("already exists", error.getvalue())
        self.assertEqual(self.output.read_bytes(), raw)

    def test_feature_receipt_rejects_input_and_identity_drift(self):
        path = self.files["feature_receipt"]
        original = json.loads(path.read_bytes())
        for key, value in (("features_sha256", "b" * 64), ("queries_sha256", "b" * 64),
                           ("corpus_sha256", "invalid"), ("candidate_k", 99),
                           ("feature_version", "other")):
            path.write_text(json.dumps({**original, key: value}), encoding="utf-8")
            with self.subTest(field=key), self.assertRaises(EvaluationError):
                self.train()
        path.write_text(json.dumps(original), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
