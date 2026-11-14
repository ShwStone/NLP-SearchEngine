"""Oracle selection and grouped resampling on a fixed judged cohort."""

from importlib.util import find_spec
import unittest

from evaluation.evaluate import QueryRecord


@unittest.skipUnless(find_spec("numpy"), "NumPy required for grouped sensitivity")
class OpportunityTests(unittest.TestCase):
    def test_exact_and_upper_bound_budgets_preserve_constraint(self):
        from evaluation.router_opportunity import analyze_opportunity

        records = [QueryRecord(qid, text) for qid, text in (
            ("c", "Ｃ"), ("b", "negative"), ("a", "positive"), ("d", "c"))]
        qrels = {record.qid: {"hit": 1.0} for record in records}
        lexical = {"a": [], "b": ["hit"], "c": [], "d": []}
        hybrid = {"a": ["hit"], "b": [], "c": ["hit"], "d": ["hit"]}
        result = analyze_opportunity(records, qrels, {"lexical": lexical, "hybrid": hybrid},
                                     {"a", "b", "c"}, {"a", "b", "c"}, {"d"})
        exact = result["policies"]["oracle_exact_budget"]
        capped = result["policies"]["oracle_at_most_budget"]
        self.assertEqual((exact["vector_calls"], exact["calls_with_negative_ndcg_gain"]), (3, 1))
        self.assertEqual((capped["vector_calls"], capped["calls_with_negative_ndcg_gain"]), (2, 0))
        self.assertEqual(exact["metrics"]["ndcg_at_10"], .5)
        self.assertEqual(capped["metrics"]["ndcg_at_10"], .75)
        grouped = result["observed_policy_grouped_sensitivity"]
        self.assertEqual(grouped["group_count"], 1)
        self.assertEqual(grouped["learned_minus_rules"]["ndcg_at_10"]["ci95"], [0.0, 0.0])

    def test_tied_oracle_uses_query_id_without_optimizing_secondary_metrics(self):
        from evaluation.router_opportunity import analyze_opportunity

        records = [QueryRecord("z", "query z"), QueryRecord("a", "query a")]
        qrels = {"z": {"z1": 1.0}, "a": {"a1": 1.0, "a2": 1.0}}
        lexical = {"z": [], "a": ["a1", "a2"]}
        # Both gains become exactly zero at cutoff 10, but only z changes MAP@100.
        hybrid = {"z": [f"miss{i}" for i in range(10)] + ["z1"], "a": ["a1", "a2"]}
        result = analyze_opportunity(records, qrels, {"lexical": lexical, "hybrid": hybrid},
                                     {"z"}, {"z"}, set())
        exact = result["policies"]["oracle_exact_budget"]
        self.assertEqual(exact["calls_with_zero_ndcg_gain"], 1)
        self.assertEqual(exact["metrics"]["map_at_100"], .5)
        self.assertGreater(result["policies"]["rules"]["metrics"]["map_at_100"], .5)
