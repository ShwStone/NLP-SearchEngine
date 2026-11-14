"""Regression tests for the label-free CE40 budget control."""

import unittest

from evaluation.ce_budget import matched_random


class MatchedRandomTests(unittest.TestCase):
    def test_matches_each_capacity_and_actual_scored_pairs(self):
        capacity = {
            "explicit": 0, "empty": 0, "a": 1, "b": 1, "c": 7,
            "d": 7, "e": 7, "f": 40, "g": 40, "h": 4,
        }
        selected = {query_id: query_id in {"a", "c", "d", "f"} for query_id in capacity}

        matched = matched_random(selected, capacity)

        self.assertEqual(matched.keys(), selected.keys())
        self.assertFalse(matched["explicit"])
        self.assertFalse(matched["empty"])
        for pairs in set(capacity.values()):
            query_ids = [query_id for query_id in capacity if capacity[query_id] == pairs]
            self.assertEqual(
                sum(matched[query_id] for query_id in query_ids),
                sum(selected[query_id] for query_id in query_ids),
            )
        self.assertEqual(sum(capacity[qid] for qid in capacity if matched[qid]), 55)
        self.assertEqual(matched, matched_random(selected, capacity, seed=2026))
        self.assertEqual(matched, matched_random(
            dict(reversed(list(selected.items()))), dict(reversed(list(capacity.items())))
        ))

    def test_rejects_invalid_budget_inputs(self):
        cases = (
            ({"a": True}, {"a": 0}, 2026),
            ({"a": True}, {"b": 1}, 2026),
            ({"": False}, {"": 0}, 2026),
            ({"a": 1}, {"a": 1}, 2026),
            ({"a": False}, {"a": -1}, 2026),
            ({"a": False}, {"a": 41}, 2026),
            ({"a": False}, {"a": True}, 2026),
            ({"a": False}, {"a": 1}, True),
        )
        for selected, capacity, seed in cases:
            with self.subTest(selected=selected, capacity=capacity, seed=seed):
                with self.assertRaises(ValueError):
                    matched_random(selected, capacity, seed)
