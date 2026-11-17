"""Exact score/candidate regression checks for optional position-free readers."""

import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline.build import build
from offline.compact import CompactIndex, compile_index
from offline.packed_index import PackedIndex
from online.memory_index import MemoryIndex
import online.scorers as scorer_module
from online.scorers import _candidate_docs, bm25, tfidf, topk


CORPUS = [
    {"doc_id": str(i), "title": "", "text": text}
    for i, text in enumerate(("alpha alpha common beta", "beta common gamma",
                              "common gamma gamma", "common alpha"))
]
TERMS = ["alpha", "common", "beta", "alpha", "gamma", "absent"]
WEIGHTS = {"alpha": 0.125, "common": 0.0, "beta": 1.75, "gamma": -0.25}


class FrequenciesOnly:
    """A single-pass frequency reader; positions must never be requested."""

    def __init__(self, legacy):
        self.legacy = legacy
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def term_frequencies(self, term):
        self.calls.append(term)
        return ((docid, tf) for docid, tf, _ in self.legacy.postings(term) or ())

    def postings(self, term):
        raise AssertionError("frequency-only operation materialized positions")


class ScorerFastpathTests(unittest.TestCase):
    def assert_exact(self, left, right):
        # Compare insertion order as well as the exact floating-point values.
        self.assertEqual(list(left.items()), list(right.items()))
        self.assertEqual(sorted(left, key=lambda d: (-left[d], d)),
                         sorted(right, key=lambda d: (-right[d], d)))

    def test_fast_and_legacy_readers_preserve_scores_and_filters(self):
        legacy = MemoryIndex.from_corpus(CORPUS)
        fast = FrequenciesOnly(legacy)
        for scorer in (bm25, tfidf):
            for terms in (TERMS, ["absent"], [], ["common"], ["alpha", "absent"]):
                for options in ({}, {"match_all": True},
                                {"candidate_docs": {0, 2, 99}},
                                {"candidate_docs": set()},
                                {"match_all": True, "candidate_docs": {1, 3}}):
                    for weights in (None, WEIGHTS):
                        with self.subTest(scorer=scorer.__name__, terms=terms,
                                          options=options, weights=weights):
                            self.assert_exact(
                                scorer(legacy, terms, term_weights=weights, **options),
                                scorer(fast, terms, term_weights=weights, **options))
        self.assertTrue(fast.calls)
        self.assertEqual(_candidate_docs(fast, ["alpha", "absent"], True), set())
        self.assertEqual(_candidate_docs(fast, ["alpha", "beta"], True), {0})

    def test_frozen_legacy_scores_preserve_duplicate_accumulation(self):
        # Captured from b57955e's original scorers, not recomputed by a second
        # implementation of their formulas. Hex preserves all float bits.
        expected = {
            bm25: [(0, '0x1.4908f4dae3672p+0'), (3, '0x1.9aed44fa878fap-3'),
                   (1, '0x1.0a2b23f3bab73p+0'), (2, '-0x1.e7f9c1e980fa8p-3')],
            tfidf: [(0, '0x1.81a3edef26647p+0'), (3, '0x1.62e42fefa39efp-3'),
                    (1, '0x1.0a2b23f3bab73p+0'), (2, '-0x1.2c710ff5dce5bp-2')],
        }
        reader = FrequenciesOnly(MemoryIndex.from_corpus(CORPUS))
        for scorer, values in expected.items():
            actual = scorer(reader, TERMS, term_weights=WEIGHTS)
            self.assertEqual([(d, v.hex()) for d, v in actual.items()], values)

    def test_high_df_ties_and_zero_scores_retain_all_documents(self):
        corpus = [{"doc_id": str(i), "title": "", "text": "common common"}
                  for i in range(257)]
        reader = FrequenciesOnly(MemoryIndex.from_corpus(corpus))
        for scorer in (bm25, tfidf):
            scores = scorer(reader, ["common", "common"])
            self.assertEqual(list(scores), list(range(257)))
            self.assertEqual(len(set(scores.values())), 1)
            self.assertEqual(sorted(scores, key=lambda d: (-scores[d], d)), list(range(257)))
            if scorer is tfidf:
                self.assertEqual(set(scores.values()), {0.0})

    def test_default_union_reads_each_scored_term_occurrence_once(self):
        legacy = MemoryIndex.from_corpus(CORPUS)
        expected_calls = Counter(term for term in TERMS if legacy.df(term))
        for scorer in (bm25, tfidf):
            reader = FrequenciesOnly(legacy)
            scores = scorer(reader, TERMS, term_weights=WEIGHTS)
            self.assertEqual(Counter(reader.calls), expected_calls)
            # An explicit union is a separate reference path that still filters.
            union = {0, 1, 2, 3}
            self.assert_exact(scores, scorer(legacy, TERMS, candidate_docs=union,
                                            term_weights=WEIGHTS))
            self.assertEqual(scorer(reader, TERMS, candidate_docs=set()), {})
            self.assertEqual(scorer(reader, TERMS, match_all=True), {})
            # Explicit candidates continue to override match_all, including when
            # an unknown query term makes the implicit intersection empty.
            self.assert_exact(scorer(reader, TERMS, match_all=True, candidate_docs={0}),
                              scorer(legacy, TERMS, candidate_docs={0}))

    def test_default_union_legacy_fallback_also_avoids_candidate_pass(self):
        reader = MemoryIndex.from_corpus(CORPUS)
        expected_calls = Counter(term for term in TERMS if reader.df(term))
        for scorer in (bm25, tfidf):
            with patch.object(reader, "postings", wraps=reader.postings) as postings:
                actual = scorer(reader, TERMS, term_weights=WEIGHTS)
                self.assertEqual(Counter(call.args[0] for call in postings.call_args_list),
                                 expected_calls)
            self.assert_exact(actual, scorer(reader, TERMS, candidate_docs={0, 1, 2, 3},
                                             term_weights=WEIGHTS))

    def test_noncallable_optional_interface_uses_postings_fallback(self):
        reader = MemoryIndex.from_corpus(CORPUS)
        for scorer in (bm25, tfidf):
            expected = scorer(reader, TERMS)
            reader.term_frequencies = None
            with patch.object(reader, "postings", wraps=reader.postings) as postings:
                self.assert_exact(expected, scorer(reader, TERMS))
                self.assertGreater(postings.call_count, 0)

    def test_real_packed_frequency_path_matches_compact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(row) + "\n" for row in CORPUS), encoding="utf-8")
            build(corpus, root / "lexical", "scorer-fastpaths")
            compile_index(root / "lexical", root / "compact")
            # Zero cache also exercises fresh single-pass iterators on each use.
            with CompactIndex(root / "compact") as legacy, PackedIndex(
                    root / "compact", cache_bytes=0) as fast:
                for term in TERMS:
                    self.assertEqual(list(fast.term_frequencies(term)),
                                     [(d, tf) for d, tf, _ in legacy.postings(term) or ()])
                with patch.object(fast, "postings", side_effect=AssertionError("positions requested")):
                    for scorer in (bm25, tfidf):
                        for options in ({}, {"match_all": True}, {"candidate_docs": {0, 3}}):
                            self.assert_exact(scorer(legacy, TERMS, term_weights=WEIGHTS, **options),
                                              scorer(fast, TERMS, term_weights=WEIGHTS, **options))

    def test_numpy_topk_preserves_exact_scores_ties_and_duplicate_terms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(row) + "\n" for row in CORPUS), encoding="utf-8")
            build(corpus, root / "lexical", "numpy-topk")
            compile_index(root / "lexical", root / "compact")
            with PackedIndex(root / "compact", cache_bytes=0) as reader:
                for terms in (TERMS, ["common"], ["common", "common"], ["absent"], []):
                    for weights in (None, WEIGHTS, {"common": 0.0}):
                        expected_scores = bm25(reader, terms, term_weights=weights)
                        expected_order = sorted(expected_scores, key=lambda d: (-expected_scores[d], d))
                        for limit in (1, 2, 3, 4, 20):
                            with self.subTest(terms=terms, weights=weights, limit=limit):
                                actual = topk("bm25", reader, terms, limit, term_weights=weights)
                                expected = [(docid, expected_scores[docid])
                                            for docid in expected_order[:limit]]
                                self.assertEqual([docid for docid, _ in actual],
                                                 [docid for docid, _ in expected])
                                for (_, left), (_, right) in zip(actual, expected):
                                    self.assertAlmostEqual(left, right, delta=1e-12)

    def test_sparse_numpy_topk_is_bit_exact_above_the_postings_gate(self):
        rows = [{"doc_id": str(i), "title": "", "text": "common common shared"}
                for i in range(300)]
        rows[42] = {"doc_id": "42", "title": "", "text": "common"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            build(corpus, root / "lexical", "numpy-sparse-topk")
            compile_index(root / "lexical", root / "compact")
            with PackedIndex(root / "compact") as reader:
                for terms in (["common"], ["common", "common"], ["common", "shared"]):
                    for weights in (None, {"common": 0.0}, {"common": -0.5}):
                        expected_scores = bm25(reader, terms, term_weights=weights)
                        expected = sorted(expected_scores.items(),
                                          key=lambda row: (-row[1], row[0]))[:10]
                        with patch("online.scorers._rank_docids",
                                   wraps=scorer_module._rank_docids) as rank:
                            actual = topk("bm25", reader, terms, 10, term_weights=weights)
                        self.assertTrue(rank.called, "postings gate did not enter numpy ranking")
                        self.assertEqual([(d, float(v).hex()) for d, v in actual],
                                         [(d, float(v).hex()) for d, v in expected])
                expected_scores = bm25(reader, ["common", "shared"])
                expected = sorted(expected_scores.items(), key=lambda row: (-row[1], row[0]))[:10]
                with patch.object(scorer_module, "DENSE_MIN_POSTINGS", 0), \
                        patch.object(scorer_module, "DENSE_POSTINGS_PER_DOC", 1):
                    actual = topk("bm25", reader, ["common", "shared"], 10)
                self.assertEqual([(d, float(v).hex()) for d, v in actual],
                                 [(d, float(v).hex()) for d, v in expected])


if __name__ == "__main__":
    unittest.main()
