"""Regressions for PackedIndex cache accounting and in-flight reads."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

from offline.build import build
from offline.compact import compile_index
from offline.packed_index import PackedIndex


@contextmanager
def _compact_fixture():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        row = {"doc_id": "a", "title": "", "text": "alpha beta gamma delta"}
        corpus.write_text(json.dumps(row) + "\n", encoding="utf-8")
        build(corpus, root / "lexical", "packed-cache-test")
        compile_index(root / "lexical", root / "compact")
        yield root / "compact"


def test_packed_cache_evicts_to_stay_within_byte_budget():
    terms = ("alpha", "beta", "gamma", "delta")
    with _compact_fixture() as compact:
        with PackedIndex(compact) as sizing_reader:
            budget = max(sizing_reader._load_block(term).charge(term) for term in terms)
        with PackedIndex(compact, cache_bytes=budget) as reader:
            for position, term in enumerate(terms):
                assert reader.postings(term) == [(0, 1, [position])]
                info = reader.cache_info()
                assert info["terms"] == 1
                assert 0 < info["cached_bytes"] <= budget
                assert info["evictions"] == position
            assert reader.postings("delta") == [(0, 1, [3])]
            assert reader.cache_info()["hits"] == 1
            reader.clear_cache()
            assert reader.cache_info()["cached_bytes"] == 0
            assert reader.cache_info()["terms"] == 0


def test_packed_zero_cache_coalesces_same_term_concurrent_reads():
    with _compact_fixture() as compact, PackedIndex(compact, cache_bytes=0) as reader:
        entered = threading.Event()
        release = threading.Event()
        joined = threading.Event()
        original = reader._load_block

        def controlled_load(term):
            entered.set()
            assert release.wait(5), "test did not release the owner's load"
            return original(term)

        with patch.object(reader, "_load_block", side_effect=controlled_load) as load:
            with ThreadPoolExecutor(max_workers=2) as pool:
                owner = pool.submit(reader.postings, "alpha")
                try:
                    assert entered.wait(5), "owner did not start loading"
                    pending = reader._pending["alpha"]
                    original_result = pending.result

                    def observe_follower():
                        joined.set()
                        return original_result(timeout=5)

                    with patch.object(pending, "result", side_effect=observe_follower):
                        follower = pool.submit(reader.postings, "alpha")
                        assert joined.wait(5), "follower did not join the pending load"
                        release.set()
                        assert owner.result(timeout=5) == follower.result(timeout=5) == [(0, 1, [0])]
                finally:
                    release.set()
            load.assert_called_once_with("alpha")
        info = reader.cache_info()
        assert info["coalesced"] == info["loads"] == 1
        assert info["cached_bytes"] == info["terms"] == info["pending"] == 0


def test_packed_failed_read_clears_pending_and_can_retry():
    with _compact_fixture() as compact, PackedIndex(compact, cache_bytes=0) as reader:
        with patch.object(reader, "_load_block", side_effect=OSError("injected read failure")) as load:
            try:
                reader.postings("beta")
            except OSError as error:
                assert str(error) == "injected read failure"
            else:
                raise AssertionError("PackedIndex did not propagate the read failure")
            load.assert_called_once_with("beta")
        assert reader.cache_info()["pending"] == 0
        assert reader.postings("beta") == [(0, 1, [1])]
        assert reader.cache_info()["loads"] == 1


ALL_TESTS = [test_packed_cache_evicts_to_stay_within_byte_budget,
             test_packed_zero_cache_coalesces_same_term_concurrent_reads,
             test_packed_failed_read_clears_pending_and_can_retry]
