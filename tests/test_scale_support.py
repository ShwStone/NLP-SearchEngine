"""Portable regression checks for the optional compact/offline scale path."""

from contextlib import closing
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

from contracts.api import SearchRequest
from contracts.conformance import check_reader
from contracts.index_format import sha256_of
from offline.build import build
from offline.compact import CompactIndex, compile_index
from offline.disk_documents import DiskDocuments
from offline.disk_index import DiskIndex
from offline.packed_index import PackedIndex
from offline.prepare import prepare
from offline.stream_build import build as stream_build
from online.engine import SearchEngine


CORPUS = [
    {"doc_id": "d0", "title": "红苹果", "text": "红苹果 手机 进水"},
    {"doc_id": "d1", "title": "蓝苹果", "text": "蓝苹果 手机 充电"},
]


def _write_corpus(path: Path) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in CORPUS),
                    encoding="utf-8")


def test_prepare_compact_and_packed_readers_preserve_contract():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        prepared = root / "prepared"
        report = prepare(corpus, prepared)
        assert report["accepted"] == len(CORPUS)
        assert json.loads((prepared / "source_manifest.json").read_text(encoding="utf-8"))["accepted"] == 2

        lexical = root / "lexical"
        build(prepared / "corpus.jsonl", lexical, "compact-test")
        compact = root / "compact"
        compile_index(lexical, compact)
        with PackedIndex(compact, cache_bytes=1024) as reader:
            check_reader(reader, CORPUS)
            assert reader.external_doc_id(0) == "d0"
            assert reader.positions_for("手机", 0)
            assert reader.cache_info()["budget_bytes"] == 1024
        with CompactIndex(compact) as reader:
            assert reader.doc(1).doc_id == "d1"
        with DiskDocuments(lexical) as documents:
            assert documents.doc(0).doc_id == "d0"


def test_compact_readers_preserve_full_lexical_results():
    # Preserve the cross-reader regression introduced in PR #7.
    rows = [
        {"doc_id": "d0", "title": "", "text": "alpha beta beta gamma"},
        {"doc_id": "d1", "title": "", "text": "alpha gamma"},
        {"doc_id": "d2", "title": "", "text": "😀心肌梗死急救。"},
        {"doc_id": "d3", "title": "", "text": "delta beta"},
        {"doc_id": "d4", "title": "", "text": "alpha alpha beta"},
    ]
    requests = [
        SearchRequest(query=query, k=3)
        for query in ("alpha beta", "心肌梗死", "NOT alpha",
                      "alpha AND NOT gamma", '"alpha beta"', "absent")
    ] + [
        SearchRequest(query="alpha", expander="rm3", k=3),
        SearchRequest(query="alpha beta", scorer="tfidf", k=2, offset=1),
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        corpus.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                          encoding="utf-8")
        lexical, compact = root / "lexical", root / "compact"
        build(corpus, lexical, "reader-parity")
        compile_index(lexical, compact)
        reference = DiskIndex(lexical)
        baseline = SearchEngine(reference)
        expected = [baseline.search(request).to_dict() for request in requests]
        assert expected[1]["results"] and expected[4]["results"]
        assert expected[-1]["results"] and expected[-2]["expanded"]

        for reader_type in (CompactIndex, PackedIndex):
            with reader_type(compact) as reader:
                check_reader(reader, rows)
                assert reader.manifest() == reference.manifest()
                engine = SearchEngine(reader)
                for request, want in zip(requests, expected):
                    actual = engine.search(request).to_dict()
                    assert actual["results"] == want["results"], (reader_type.__name__, request)
                    assert actual["expanded"] == want["expanded"], (reader_type.__name__, request)


def test_compact_startup_rejects_bad_terms_and_releases_partial_resources():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        lexical, compact = root / "lexical", root / "compact"
        build(corpus, lexical, "compact-startup")
        compile_index(lexical, compact)

        open_file, connect = os.open, sqlite3.connect
        descriptors, connections = [], []

        def failing_open(path, *args, **kwargs):
            if Path(path).name == "offsets.bin":
                raise OSError("injected second file open failure")
            fd = open_file(path, *args, **kwargs)
            descriptors.append(fd)
            return fd

        def tracked_connect(*args, **kwargs):
            db = connect(*args, **kwargs)
            connections.append(db)
            return db

        with patch("offline.compact.os.open", side_effect=failing_open), \
                patch("offline.compact.sqlite3.connect", side_effect=tracked_connect):
            try:
                PackedIndex(compact)
            except OSError as error:
                assert str(error) == "injected second file open failure"
            else:
                raise AssertionError("compact reader ignored a required file open failure")
        assert len(descriptors) == len(connections) == 1
        try:
            os.fstat(descriptors[0])
        except OSError:
            pass
        else:
            os.close(descriptors[0])
            raise AssertionError("startup left an open artifact descriptor")
        try:
            connections[0].execute("SELECT 1")
        except sqlite3.ProgrammingError:
            pass
        else:
            connections[0].close()
            raise AssertionError("startup left an open term database")

        database = compact / "terms.sqlite"
        with closing(connect(database)) as db, db:
            db.execute("DROP TABLE terms")
        metadata = compact / "compact.json"
        manifest = json.loads(metadata.read_text(encoding="utf-8"))
        manifest["files"]["terms.sqlite"] = sha256_of(database)
        metadata.write_text(json.dumps(manifest), encoding="utf-8")
        for reader_type in (CompactIndex, PackedIndex):
            try:
                with reader_type(compact):
                    raise AssertionError("reader admitted a term database with no terms table")
            except ValueError as error:
                assert str(error) == "invalid compact term index"


def test_prepare_preserves_tokenless_documents_in_both_policies():
    punct = {"doc_id": "punct", "title": "", "text": " () \n"}
    for rows in ([punct, *CORPUS], [punct]):
        for policy in ("error", "skip"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                corpus = root / "corpus.jsonl"
                original = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
                corpus.write_text(original, encoding="utf-8")
                target = root / "prepared"
                report = prepare(corpus, target, invalid=policy)
                assert report["accepted"] == report["source_records"] == len(rows)
                assert report["excluded"] == 0
                assert [json.loads(line) for line in (target / "corpus.jsonl").read_text(encoding="utf-8").splitlines()] == rows
                assert [json.loads(line) for line in (target / "id_map.jsonl").read_text(encoding="utf-8").splitlines()] == [
                    {"docid": i, "doc_id": row["doc_id"], "source_line": i + 1}
                    for i, row in enumerate(rows)]
                assert (target / "excluded.jsonl").read_bytes() == b""
                assert corpus.read_text(encoding="utf-8") == original


def test_prepare_still_rejects_empty_bodies_and_duplicate_ids():
    empty = {"doc_id": "empty", "title": "", "text": " "}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        corpus.write_text(json.dumps(empty) + "\n", encoding="utf-8")
        try:
            prepare(corpus, root / "empty")
        except ValueError as error:
            assert "empty_body" in str(error)
        else:
            raise AssertionError("empty body accepted")
        assert not (root / "empty").exists()
        corpus.write_text("".join(json.dumps(row) + "\n" for row in [*CORPUS, CORPUS[0]]), encoding="utf-8")
        for policy in ("error", "skip"):
            target = root / policy
            try:
                prepare(corpus, target, invalid=policy)
            except ValueError as error:
                assert "duplicate doc_id" in str(error)
            else:
                raise AssertionError("duplicate external ID accepted")
            assert not target.exists()


def test_stream_builder_is_deterministic_and_conformant():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        first, second = root / "first", root / "second"
        stream_build(corpus, first, "stream-test")
        stream_build(corpus, second, "stream-test")
        compact_first = root / "compact-first"
        compile_index(first, compact_first)
        with PackedIndex(compact_first, cache_bytes=1024) as reader:
            check_reader(reader, CORPUS)
        first_manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        second_manifest = json.loads((second / "manifest.json").read_text(encoding="utf-8"))
        assert first_manifest["files"] == second_manifest["files"]


def test_tokenless_documents_preserve_ids_and_exact_oracle_lengths():
    from offline.disk_index import DiskIndex
    from online.memory_index import MemoryIndex
    from online.scorers import bm25, tfidf

    for rows in ([{"doc_id": "punct", "title": "", "text": "()"}, *CORPUS],
                 [{"doc_id": "punct", "title": "", "text": "()"}]):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "corpus.jsonl"
            corpus.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            expected = MemoryIndex.from_corpus(rows)
            check_reader(expected, rows)
            for builder in (build, stream_build):
                target = root / builder.__module__.replace(".", "-")
                builder(corpus, target, "tokenless-test")
                disk = DiskIndex(target)
                check_reader(disk, rows)
                assert disk.n_docs == len(rows)
                assert disk.doc(0).doc_id == "punct" and disk.doc(0).offsets == []
                for score in (bm25, tfidf):
                    assert score(disk, ["手机"]) == score(expected, ["手机"])

    # A tokenful document falsified as zero still fails the exact oracle gate.
    false_reader = MemoryIndex.from_corpus(CORPUS)
    false_reader._exp.doc_lengths[0] = 0
    try:
        check_reader(false_reader, CORPUS)
    except AssertionError:
        pass
    else:
        raise AssertionError("conformance accepted a false zero token length")


def test_separate_sqlite_work_dir_matches_default_and_cleans_failures():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        work = root / "work"
        work.mkdir()
        sentinel = work / "protected.txt"
        sentinel.write_bytes(b"protected")
        default = stream_build(corpus, root / "default", "work-dir-test")
        separate = stream_build(corpus, root / "separate", "work-dir-test", work_dir=work)
        assert default.to_dict() == separate.to_dict()
        for name in default.files:
            assert (root / "default" / name).read_bytes() == (root / "separate" / name).read_bytes()
        corpus.write_text(corpus.read_text(encoding="utf-8") + "invalid JSON\n", encoding="utf-8")
        try:
            stream_build(corpus, root / "failed", "failed", work_dir=work)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed input was accepted")
        assert not (root / "failed").exists()
        assert [p.name for p in work.iterdir()] == [sentinel.name]
        assert sentinel.read_bytes() == b"protected"
        assert not list(root.glob(".*.stream-*"))


def test_disk_index_compact_dispatch_serves_public_http():
    import threading
    from http.server import ThreadingHTTPServer
    from unittest.mock import patch
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from offline.disk_index import DiskIndex
    from online.engine import SearchEngine
    from online.server import make_handler

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        stream_build(corpus, root / "v1", "http-compact")
        compile_index(root / "v1", root / "compact")
        with patch("offline.disk_index.load_manifest", side_effect=AssertionError("legacy path entered")):
            reader = DiskIndex(root / "compact")
        try:
            assert reader.storage_format == "compact-packed-v1"
            check_reader(reader, CORPUS)
            server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(SearchEngine(reader)))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                endpoint = f"http://127.0.0.1:{server.server_port}"
                for path in ("/health", "/index/info"):
                    with urlopen(endpoint + path, timeout=5) as response:
                        assert response.status == 200
                request = Request(endpoint + "/search", data=json.dumps({
                    "query": "手机 AND NOT 充电", "mode": "lexical", "k": 5,
                }).encode(), headers={"Content-Type": "application/json"})
                with urlopen(request, timeout=5) as response:
                    payload = json.load(response)
                assert [r["doc_id"] for r in payload["results"]] == ["d0"]
                request = Request(endpoint + "/search", data=json.dumps({
                    "query": "手机", "mode": "dense", "k": 5,
                }).encode(), headers={"Content-Type": "application/json"})
                try:
                    urlopen(request, timeout=5)
                except HTTPError as error:
                    assert error.code == 503
                    error.close()
                else:
                    raise AssertionError("compact server hid missing vectors")
            finally:
                server.shutdown()
                worker.join(timeout=5)
                server.server_close()
        finally:
            reader.close()


def test_compact_dispatch_rejects_ambiguity_and_disabled_checksum_bypass():
    from offline.disk_index import DiskIndex
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        stream_build(corpus, root / "v1", "compact-dispatch")
        compile_index(root / "v1", root / "compact")
        (root / "compact" / "manifest.json").write_bytes((root / "v1" / "manifest.json").read_bytes())
        try:
            DiskIndex(root / "compact")
        except ValueError as error:
            assert "ambiguous" in str(error)
        else:
            raise AssertionError("mixed index formats were silently selected")
        (root / "compact" / "manifest.json").unlink()
        with (root / "compact" / "docs.idx").open("ab") as output:
            output.write(b"\0")
        try:
            DiskIndex(root / "compact", verify_checksums=False)
        except ValueError as error:
            assert "checksum" in str(error)
        else:
            raise AssertionError("legacy flag bypassed compact integrity validation")


def test_compact_reader_validates_source_version_statistics_and_identity():
    from offline.disk_index import DiskIndex
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        _write_corpus(corpus)
        stream_build(corpus, root / "v1", "metadata-validation")
        target = root / "compact #目录"
        compile_index(root / "v1", target)
        with DiskIndex(target) as reader:
            assert reader.doc(0).doc_id == "d0"
            assert list(reader.vocab())
        path = target / "compact.json"
        original = path.read_text(encoding="utf-8")
        for key, value in (("schema_version", 999), ("avgdl", 999),
                           ("avgdl", float("nan")), ("n_docs", True),
                           ("files", {"docs.jsonl": None})):
            metadata = json.loads(original)
            metadata["source_manifest"][key] = value
            path.write_text(json.dumps(metadata), encoding="utf-8")
            try:
                DiskIndex(target)
            except ValueError:
                pass
            else:
                raise AssertionError(f"compact accepted malformed source {key}")


def test_compact_compilers_reject_noncanonical_statistics_before_publication():
    import importlib.util
    from offline.checkpoint_compact import compile_checkpoint

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        corpus = root / "corpus.jsonl"
        corpus.write_text(json.dumps({"doc_id": "one", "title": "", "text": "()"}) + "\n",
                          encoding="utf-8")
        source = root / "source"
        stream_build(corpus, source, "noncanonical-stats")
        path = source / "manifest.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        variants = ["default"]
        if importlib.util.find_spec("ijson"):
            variants += ["streaming", "checkpoint"]
        for key, value in (("n_docs", True), ("avgdl", False),
                           ("avgdl", float("nan")), ("schema_version", True)):
            for variant in variants:
                manifest = dict(original, **{key: value})
                path.write_text(json.dumps(manifest), encoding="utf-8")
                target = root / (variant + "-target")
                try:
                    if variant == "checkpoint":
                        compile_checkpoint(source, target, state_dir=root / "state")
                    else:
                        compile_index(source, target, streaming=variant == "streaming")
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"{variant} published invalid {key}={value!r}")
                assert not target.exists()


ALL_TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
