"""Verify that gate error analysis preserves its fixed and skipped outcomes."""

import unittest

from evaluation.ce_analysis import learned_diagnostics


class CeAnalysisTests(unittest.TestCase):
    def test_missed_gain_and_applied_loss_remain_visible(self):
        a, b = ("a", .2), ("b", .1)
        qrels = {qid: {label: 1} for qid, label in
                 (("miss", "b"), ("harm", "a"), ("tie", "a"), ("bypass", "b"))}
        rrf = dict.fromkeys(qrels, [a, b])
        fixed = {**rrf, "miss": [b, a], "harm": [b, a]}
        learned = {**rrf, "harm": [b, a]}
        plan = {"capacity": {"miss": 2, "harm": 2, "tie": 2, "bypass": 0},
                "learned_actions": {"miss": False, "harm": True, "tie": True, "bypass": False},
                "ce_pairs": 4}
        result = learned_diagnostics(qrels, {"rrf": rrf, "ce40": fixed, "learned": learned},
                                     plan, samples=10, seed=2026)
        self.assertEqual(result["eligible_fixed_ce40_outcomes"], {"win": 1, "loss": 1, "tie": 1})
        counts = result["learned_actions_by_ce40_outcome"]
        self.assertEqual((counts["skipped_win"], counts["applied_loss"], counts["applied_tie"],
                          counts["ineligible"]), (1, 1, 1, 1))
        self.assertAlmostEqual(result["mean_missed_positive_ndcg"], result["mean_applied_negative_ndcg"])
        self.assertLess(result["mean_learned_delta_vs_rrf"], 0)
        self.assertAlmostEqual(result["ce40_pair_fraction"], 2/3)
