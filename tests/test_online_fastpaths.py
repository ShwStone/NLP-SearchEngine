"""Fast paths must preserve complete responses and retain strict reader bounds."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from contracts.api import SearchRequest
from contracts.reader import Document
from contracts.retrieval import RetrievalConfig, corpus_fingerprint
from offline.build import build
from offline.compact import CompactIndex, compile_index
from offline.packed_index import PackedIndex
from online.engine import SearchEngine
from tests.test_hybrid import _setup


def semantic(response):
    value = response.to_dict()
    value.pop("took_ms")
    value.pop("timings_ms")
    return value


def test_light_document_and_identity_paths_preserve_four_modes():
    engine, encoder, vectors = _setup(RetrievalConfig(candidate_k=2))
    original = engine.reader

    class LightReader:
        def __init__(self):
            self.identities = []
            self.rendered = []

        def __getattr__(self, name):
            return getattr(original, name)

        def external_doc_id(self, docid):
            self.identities.append(docid)
            return original.doc(docid).doc_id

        def document_text(self, docid):
            self.rendered.append(docid)
            doc = original.doc(docid)
            return replace(doc, offsets=[])

        def doc(self, docid):
            raise AssertionError("query path must not materialize stored offsets")

    def full_sort(limit, rows, key):
        return sorted(rows, key=key)[:limit]

    for mode in ("lexical", "dense", "hybrid", "adaptive"):
        for query in ("phone", "camera", "phone OR NOT camera"):
            if mode == "dense" and " OR " in query:
                continue
            request = SearchRequest(query=query, k=2, mode=mode)
            engine.reader = original
            with patch("online.engine.nsmallest", side_effect=full_sort):
                expected = semantic(engine.search(request))
            fast = LightReader()
            engine.reader = fast
            assert semantic(engine.search(request)) == expected
            assert len(fast.rendered) == len(expected["results"])
            if mode in ("dense", "hybrid"):
                # External IDs are checked through the already-rendered final
                # documents; no separate read occurs for every ANN candidate.
                assert fast.identities == []


def test_bounded_topk_matches_full_sort_with_ties_and_pagination():
    class Reader:
        n_docs = 257

        def doc(self, docid):
            return Document(docid, str(docid), "", "alpha")

    scores = {docid: float((docid * 17) % 9 - 4) for docid in reversed(range(257))}
    engine = SearchEngine(Reader())
    expected = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
    with patch("online.engine.scorers.SCORERS", {"bm25": lambda *args, **kwargs: scores}):
        for offset, count in ((0, 1), (7, 11), (250, 7), (300, 2)):
            result = engine.search(SearchRequest(query="alpha", k=count, offset=offset))
            assert [(int(row.doc_id), row.score) for row in result.results] == expected[offset:offset + count]
            assert [row.rank for row in result.results] == list(range(offset + 1, offset + 1 + len(result.results)))


def compact_fixture(root):
    corpus = root / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in (
        {"doc_id": "small", "title": "", "text": "alpha beta"},
        {"doc_id": "large", "title": "", "text": "x" * (70 * 1024)},
    )), encoding="utf-8")
    build(corpus, root / "lexical", "fast-doc-test")
    compile_index(root / "lexical", root / "compact")


def test_compact_rendering_avoids_offsets_and_reuses_bounded_small_rows():
    import offline.compact as compact
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        compact_fixture(root)
        for reader_class in (CompactIndex, PackedIndex):
            with reader_class(root / "compact") as reader:
                calls = []
                original = compact._pread

                def record(fd, size, offset):
                    calls.append(fd)
                    return original(fd, size, offset)

                with patch.object(compact, "_pread", side_effect=record):
                    assert reader.external_doc_id(0) == "small"
                    light = reader.document_text(0)
                    assert (light.doc_id, light.text, light.offsets) == ("small", "alpha beta", [])
                    assert calls == [reader._files["docs.jsonl"]]
                    complete = reader.doc(0)
                    assert complete.offsets
                    assert complete.text == light.text
                    assert calls[-1] == reader._files["offsets.bin"]
                    before = reader._document_bytes.cache_info()
                    assert reader.external_doc_id(1) == reader.external_doc_id(1) == "large"
                    after = reader._document_bytes.cache_info()
                    assert before == after  # Oversized rows never enter the bounded cache.
                    assert calls[-2:] == [reader._files["docs.jsonl"]] * 2


def test_cached_documents_still_reject_invalid_ids_and_closed_readers():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        compact_fixture(root)
        for reader_class in (CompactIndex, PackedIndex):
            reader = reader_class(root / "compact")
            reader.external_doc_id(0)
            for bad in (True, False, -1, 2, 0.0):
                try:
                    reader.external_doc_id(bad)
                except IndexError:
                    pass
                else:
                    raise AssertionError("cache bypassed strict internal ID validation")
            reader.close()
            assert reader._document_bytes.cache_info().currsize == 0
            try:
                reader.document_text(0)
            except ValueError:
                pass
            else:
                raise AssertionError("closed document cache remained queryable")


def test_streamed_fingerprint_matches_full_documents_and_rejects_partial_order():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        compact_fixture(root)
        with PackedIndex(root / "compact") as reader:
            class Legacy:
                n_docs = reader.n_docs

                def doc(self, docid):
                    return reader.doc(docid)

            expected = corpus_fingerprint(Legacy())
            with patch.object(reader, "doc", side_effect=AssertionError("offset reads forbidden")), \
                 patch.object(reader, "_document_bytes", side_effect=AssertionError("random document reads forbidden")):
                assert corpus_fingerprint(reader) == expected
            reader._doc_index = reader._doc_index[16:32] + reader._doc_index[:16]
            assert corpus_fingerprint(Legacy()) != expected
            try:
                corpus_fingerprint(reader)
            except ValueError as error:
                assert "document byte index mismatch" in str(error)
            else:
                raise AssertionError("stream ignored a swapped logical document index")

    class Broken:
        n_docs = 2

        def __init__(self, rows):
            self.rows = rows

        def iter_document_text(self):
            return iter(self.rows)

    for rows in ([Document(0, "a", "", "")],
                 [Document(1, "a", "", ""), Document(0, "b", "", "")],
                 [Document(0, "a", "", ""), Document(1, "b", "", ""), Document(2, "c", "", "")]):
        try:
            corpus_fingerprint(Broken(rows))
        except ValueError:
            pass
        else:
            raise AssertionError("fingerprint accepted truncated or reordered documents")


ALL_TESTS = [test_light_document_and_identity_paths_preserve_four_modes,
             test_bounded_topk_matches_full_sort_with_ties_and_pagination,
             test_compact_rendering_avoids_offsets_and_reuses_bounded_small_rows,
             test_cached_documents_still_reject_invalid_ids_and_closed_readers,
             test_streamed_fingerprint_matches_full_documents_and_rejects_partial_order]
