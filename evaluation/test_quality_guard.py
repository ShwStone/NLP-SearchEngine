import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from evaluation.quality_guard import guard


class QualityGuardTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = temporary.name
        self.summary, self.fixed, self.out = (Path(folder) / name for name in ("summary", "fixed", "out"))
        self.learned = {
            "learned-router": {"metrics": {"ndcg": 0.30}, "heavy_rate": 0.6, "ce_pairs_mean": 24},
            "protocol": {"dev_qrels_sha256": "a" * 64, "all_dev_metrics_exploratory": True,
                         "fixed_pair_budget": 24, "dev_queries": 10},
        }
        self.control = {
            "queries": 10, "fixed_ce_pairs": 24, "metrics": {"ndcg": 0.32, "recall": 0.4},
            "router_minus_fixed": {"ndcg": {"mean": -0.02}},
            "protocol": {"qrels_sha256": "a" * 64, "separately_scored": True,
                         "model_or_threshold_changed": False},
        }

    def inputs(self):
        raw = json.dumps(self.learned).encode("utf-8")
        self.summary.write_bytes(raw)
        self.control["protocol"]["policy_summary_sha256"] = hashlib.sha256(raw).hexdigest()
        self.fixed.write_text(json.dumps(self.control), encoding="utf-8")

    def test_bound_development_fallback_never_approves_production(self):
        self.inputs()
        result = guard(self.summary, self.fixed, self.out)
        self.assertTrue(result["fallback_activated"])
        self.assertEqual(result["heavy_rate"], 1.0)
        self.assertEqual(result["ce_pairs_mean"], 24)
        self.assertFalse(result["production_default_safe"])
        self.assertEqual(result["inputs"]["fixed_control_sha256"],
                         hashlib.sha256(self.fixed.read_bytes()).hexdigest())
        with self.assertRaises(FileExistsError):
            guard(self.summary, self.fixed, self.out)

    def test_margin_does_not_turn_a_dev_result_into_production_approval(self):
        self.inputs()
        result = guard(self.summary, self.fixed, self.out, margin=0.03)
        self.assertFalse(result["fallback_activated"])
        self.assertFalse(result["production_default_safe"])

    def test_rejects_mismatched_inputs_without_publishing(self):
        self.inputs()
        for key, value in (("policy_summary_sha256", "c" * 64), ("qrels_sha256", "b" * 64),
                           ("separately_scored", False)):
            with self.subTest(key=key):
                altered = {**self.control, "protocol": {**self.control["protocol"], key: value}}
                self.fixed.write_text(json.dumps(altered), encoding="utf-8")
                with self.assertRaises(ValueError):
                    guard(self.summary, self.fixed, self.out)
                self.assertFalse(self.out.exists())

    def test_rejects_invalid_margin_and_disagreed_delta(self):
        self.inputs()
        with self.assertRaises(ValueError):
            guard(self.summary, self.fixed, self.out, margin=float("nan"))
        self.control["router_minus_fixed"]["ndcg"]["mean"] = 0.02
        self.inputs()
        with self.assertRaises(ValueError):
            guard(self.summary, self.fixed, self.out)
        self.assertFalse(self.out.exists())
