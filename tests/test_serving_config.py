"""ServingConfig v1 schema, path and CLI exclusivity tests (no real services)."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from contracts.retrieval import EmbeddingSpec, RetrievalConfig, VectorManifest
from online import server
from online.serving_config import ServingConfigError, load_serving_config


NEWS_PROFILE = {
    "schema_version": 1,
    "dataset": "rfc4-clean-news-20k",
    "index": "../artifacts/news20k/compact",
    "vector": {
        "url": "http://127.0.0.1:6333",
        "manifest": "../artifacts/news20k/vector/vector_manifest.json",
        "hnsw_ef": 256,
        "timeout_s": 30.0,
    },
    "embedding": {"model_path": "../models/bge-small-zh-v1.5-7999e1d3",
                  "device": "cpu"},
    "retrieval": {"candidate_k": 200, "rrf_c": 60.0},
    "serve": {"host": "127.0.0.1", "port": 8000, "max_body_bytes": 65536,
              "request_timeout_s": 10.0},
    "policy": {"factory": "rule"},
}
OFFICIAL_PROFILE = {
    "schema_version": 1,
    "dataset": "dureader-official-dev-full-8096668",
    "index": "../artifacts/official/compact",
    "vector": {
        "url": "http://127.0.0.1:6333",
        "manifest": "../artifacts/official/vector/vector_manifest.json",
        "hnsw_ef": 256,
        "timeout_s": 30.0,
    },
    "embedding": {"model_path": "../models/bge-small-zh-v1.5-7999e1d3",
                  "device": "cpu"},
    "retrieval": {"candidate_k": 200, "rrf_c": 60.0},
    "serve": {"host": "127.0.0.1", "port": 8000, "max_body_bytes": 65536,
              "request_timeout_s": 10.0},
    "policy": {"factory": "rule"},
}


def _manifest(dataset="fixture"):
    return VectorManifest(
        "a" * 64, 2, EmbeddingSpec("fixture", "v1", dimension=2), "fixture-v1",
        snapshot_files={"fixture.snapshot": "b" * 64},
    )


def _engine_info(dataset="fixture"):
    """CLI 测试把 SearchEngine 换成 mock，这里提供真实 engine 会回显的身份块。"""
    return {
        "index_identity": {"dataset": dataset, "n_docs": 2, "schema_version": 1,
                           "analyzer_hash": "-", "corpus_sha256": "a" * 64,
                           "bm25": {"k1": 1.2, "b": 0.75, "idf": "log1p-lucene"}},
        "vector_identity": {"corpus_sha256": "a" * 64, "n_docs": 2,
                            "collection": "fixture-v1", "backend": "qdrant-hnsw",
                            "search": {"hnsw_ef": 256, "exact": False,
                                       "timeout_s": 12.5}},
        "search_config": {"candidate_k": 120, "rrf_c": 25.0},
        "policy": {"name": "rule", "policy_version": "rules-v1",
                   "requires_statistics": False},
    }


class ServingConfigLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, value, name="serving.json", directory="config"):
        path = self.root / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="utf-8")
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def load(self, value):
        return load_serving_config(self.write(value))

    def assert_invalid(self, value, *fragments):
        with self.assertRaises(ServingConfigError) as raised:
            self.load(value)
        for fragment in fragments:
            self.assertIn(fragment, str(raised.exception))

    def test_full_profile_resolves_paths_and_serving_info(self):
        path = self.write(NEWS_PROFILE)
        config = load_serving_config(path)
        base = path.parent
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.dataset, "rfc4-clean-news-20k")
        self.assertEqual(config.index, (base / "../artifacts/news20k/compact").resolve())
        self.assertEqual(config.vector.url, "http://127.0.0.1:6333")
        self.assertEqual(config.vector.manifest,
                         (base / "../artifacts/news20k/vector/vector_manifest.json").resolve())
        self.assertEqual(config.vector.hnsw_ef, 256)
        self.assertEqual(config.vector.timeout_s, 30.0)
        self.assertEqual(config.embedding.model_path,
                         (base / "../models/bge-small-zh-v1.5-7999e1d3").resolve())
        self.assertEqual(config.embedding.device, "cpu")
        self.assertEqual(config.retrieval_config(), RetrievalConfig(candidate_k=200, rrf_c=60.0))
        self.assertIsNone(config.retrieval.lexical_cache_bytes)
        self.assertEqual(config.serving_info(), {
            "schema_version": 1,
            "profile": "rfc4-clean-news-20k",
            "host": "127.0.0.1",
            "port": 8000,
            "max_body_bytes": 65536,
            "request_timeout_s": 10.0,
            "router_factory": "rule",
        })

    def test_minimal_profile_uses_frozen_defaults(self):
        config = self.load({"schema_version": 1, "dataset": "memory", "index": "idx"})
        self.assertIsNone(config.vector)
        self.assertIsNone(config.embedding)
        self.assertEqual(config.retrieval.candidate_k, 200)
        self.assertEqual(config.retrieval.rrf_c, 60.0)
        self.assertEqual(config.serve.host, "127.0.0.1")
        self.assertEqual(config.serve.port, 8000)
        self.assertEqual(config.serve.max_body_bytes, 65536)
        self.assertEqual(config.serve.request_timeout_s, 10.0)
        self.assertEqual(config.policy_factory, "rule")
        self.assertEqual(config.index, (self.root / "config" / "idx").resolve())

    def test_partial_vector_and_embedding_defaults(self):
        value = {"schema_version": 1, "dataset": "memory", "index": "idx",
                 "vector": {"url": "http://127.0.0.1:6333", "manifest": "v.json"},
                 "embedding": {}}
        config = self.load(value)
        self.assertEqual(config.vector.hnsw_ef, None)
        self.assertEqual(config.vector.timeout_s, 30.0)
        self.assertIsNone(config.embedding.model_path)
        self.assertIsNone(config.embedding.device)

    def test_example_profiles_load_with_distinct_exact_datasets(self):
        news = load_serving_config(self.write(NEWS_PROFILE, directory="news"))
        official = load_serving_config(self.write(OFFICIAL_PROFILE, directory="official"))
        self.assertEqual(news.dataset, "rfc4-clean-news-20k")
        self.assertEqual(official.dataset, "dureader-official-dev-full-8096668")
        self.assertNotEqual(news.dataset, official.dataset)
        # The loader never normalizes aliases; exact match is enforced at startup.
        alias = load_serving_config(self.write(dict(NEWS_PROFILE, dataset="news20k"),
                                               directory="alias"))
        self.assertEqual(alias.dataset, "news20k")

    def test_paths_are_config_relative_and_not_interpolated(self):
        path = self.write({"schema_version": 1, "dataset": "memory",
                           "index": "../idx/$HOME", "vector": {
                               "url": "http://127.0.0.1:6333",
                               "manifest": "~/vector/manifest.json"}})
        config = load_serving_config(path)
        base = path.parent
        self.assertEqual(config.index, (base / "../idx/$HOME").resolve())
        self.assertEqual(config.index.parts[-2:], ("idx", "$HOME"))
        self.assertEqual(config.vector.manifest, (base / "~/vector/manifest.json").resolve())
        self.assertNotEqual(config.vector.manifest, Path.home() / "vector/manifest.json")

    def test_rejects_missing_required_fields_and_unknown_keys(self):
        for missing in ("schema_version", "dataset", "index"):
            value = dict(NEWS_PROFILE)
            value.pop(missing)
            self.assert_invalid(value, "缺少必填字段")
        self.assert_invalid(dict(NEWS_PROFILE, extra=1), "未知字段")
        self.assert_invalid(dict(NEWS_PROFILE, vector=dict(NEWS_PROFILE["vector"], extra=1)),
                            "未知字段")
        self.assert_invalid(dict(NEWS_PROFILE, embedding={"model_path": "m", "extra": 1}),
                            "未知字段")
        self.assert_invalid(dict(NEWS_PROFILE, retrieval={"candidate_k": 10, "extra": 1}),
                            "未知字段")
        self.assert_invalid(dict(NEWS_PROFILE, serve={"host": "127.0.0.1", "extra": 1}),
                            "未知字段")
        self.assert_invalid(dict(NEWS_PROFILE, policy={"factory": "rule", "extra": 1}),
                            "未知字段")

    def test_rejects_wrong_but_coercible_types(self):
        cases = [
            {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": "1"},
            {"dataset": 1}, {"dataset": ""}, {"index": 1}, {"index": ""},
            {"vector": {"url": "http://h", "manifest": "v", "hnsw_ef": True}},
            {"vector": {"url": "http://h", "manifest": "v", "hnsw_ef": 1.0}},
            {"vector": {"url": "http://h", "manifest": "v", "timeout_s": "30"}},
            {"retrieval": {"candidate_k": True}},
            {"retrieval": {"candidate_k": 2.0}},
            {"serve": {"port": True}}, {"serve": {"port": 8000.0}},
            {"serve": {"max_body_bytes": True}},
            {"serve": {"request_timeout_s": "10"}},
        ]
        for override in cases:
            with self.subTest(override=override):
                value = dict(NEWS_PROFILE)
                value.update(override)
                self.assert_invalid(value)

    def test_rejects_nonfinite_numbers(self):
        raw = json.dumps(NEWS_PROFILE)
        for needle in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(needle=needle):
                for key in ("rrf_c", "timeout_s", "request_timeout_s"):
                    text = raw.replace('"rrf_c": 60.0', f'"rrf_c": {needle}') \
                        .replace('"timeout_s": 30.0', f'"timeout_s": {needle}') \
                        .replace('"request_timeout_s": 10.0',
                                 f'"request_timeout_s": {needle}')
                    with self.assertRaises(ServingConfigError):
                        load_serving_config(self.write(text, name=f"{needle}-{key}.json"))

    def test_rejects_out_of_range_values(self):
        cases = [
            ({"serve": {"port": -1}}, "port"),
            ({"serve": {"port": 65536}}, "port"),
            ({"serve": {"max_body_bytes": 0}}, "max_body_bytes"),
            ({"retrieval": {"candidate_k": 0}}, "candidate_k"),
            ({"retrieval": {"rrf_c": 0}}, "rrf_c"),
            ({"retrieval": {"rrf_c": -1.0}}, "rrf_c"),
            ({"vector": {"url": "http://h", "manifest": "v", "hnsw_ef": 0}}, "hnsw_ef"),
            ({"vector": {"url": "http://h", "manifest": "v", "timeout_s": 0}},
             "timeout_s"),
        ]
        for override, fragment in cases:
            with self.subTest(override=override):
                value = dict(NEWS_PROFILE)
                value.update(override)
                self.assert_invalid(value, fragment)

    def test_rejects_blank_paths_and_paired_embedding_rules(self):
        self.assert_invalid(dict(NEWS_PROFILE, index="   "), "index")
        self.assert_invalid(dict(NEWS_PROFILE, vector={
            "url": "http://h", "manifest": " "}), "vector.manifest")
        self.assert_invalid(dict(NEWS_PROFILE, embedding={
            "model_path": " "}), "embedding.model_path")
        value = dict(NEWS_PROFILE)
        value.pop("vector")
        self.assert_invalid(value, "embedding")
        self.assert_invalid(dict(NEWS_PROFILE, embedding={"device": "cpu"}), "device")
        self.assert_invalid(dict(NEWS_PROFILE, vector={"url": "http://h"}), "manifest")
        self.assert_invalid(dict(NEWS_PROFILE, vector={"manifest": "v"}), "url")

    def test_rejects_credential_or_ambiguous_vector_urls_without_echo(self):
        for url in ("http://alice:s3cr3t@127.0.0.1:6333",
                    "http://alice@127.0.0.1:6333",
                    "http://127.0.0.1:6333?token=s3cr3t",
                    "http://127.0.0.1:6333#s3cr3t",
                    "ftp://127.0.0.1:6333",
                    "127.0.0.1:6333"):
            with self.subTest(url=url):
                value = dict(NEWS_PROFILE, vector=dict(NEWS_PROFILE["vector"], url=url))
                with self.assertRaises(ServingConfigError) as raised:
                    self.load(value)
                message = str(raised.exception)
                for secret in ("alice", "s3cr3t", "token", "ftp://"):
                    self.assertNotIn(secret, message)

    def test_rejects_invalid_json_and_nonobject_root(self):
        self.assert_invalid("{", "JSON")
        self.assert_invalid("[]", "JSON 对象")
        self.assert_invalid("null", "JSON 对象")


class ServingCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config_path = self.root / "config" / "serving.json"
        self.config_path.parent.mkdir(parents=True)
        self.manifest = _manifest()
        self.manifest_path = self.root / "config" / "vector_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest.to_dict()), encoding="utf-8")
        self.profile = {
            "schema_version": 1,
            "dataset": "fixture",
            "index": "../index",
            "vector": {
                "url": "http://127.0.0.1:6333",
                "manifest": "vector_manifest.json",
                "hnsw_ef": 256,
                "timeout_s": 12.5,
            },
            "embedding": {"model_path": "../models/bge", "device": "cpu"},
            "retrieval": {"candidate_k": 120, "rrf_c": 25.0},
            "serve": {"host": "127.0.0.1", "port": 8123,
                      "max_body_bytes": 2048, "request_timeout_s": 3.0},
            "policy": {"factory": "rule"},
        }
        self.write_profile(self.profile)

    def write_profile(self, value):
        self.config_path.write_text(json.dumps(value), encoding="utf-8")

    @contextlib.contextmanager
    def cli(self, dataset="fixture", disk_manifest=None):
        with contextlib.ExitStack() as stack:
            mocks = {name: stack.enter_context(patch.object(server, name)) for name in
                     ("DiskIndex", "SearchEngine", "QdrantVectorIndex",
                      "ThreadingHTTPServer", "SentenceTransformerEncoder")}
            mocks["DiskIndex"].return_value.manifest.return_value = (
                {"dataset": dataset, "n_docs": 2, "schema_version": 1,
                 "analyzer": {"config_hash": "-"}}
                if disk_manifest is None else disk_manifest)
            mocks["ThreadingHTTPServer"].return_value.server_address = ("127.0.0.1", 8123)
            mocks["SearchEngine"].return_value.info.return_value = _engine_info(dataset)
            mocks["_emit_startup"] = stack.enter_context(
                patch.object(server, "_emit_startup"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield mocks

    def test_config_is_exclusive_with_every_explicit_legacy_flag(self):
        flags = [
            ("--index", "unused"), ("--host", "127.0.0.1"), ("--port", "8000"),
            ("--candidate-k", "200"), ("--rrf-c", "60.0"),
            ("--vector-url", "http://127.0.0.1:6333"),
            ("--vector-manifest", str(self.manifest_path)),
            ("--hnsw-ef", "256"), ("--bge-model-path", "model"),
            ("--encoder-device", "cpu"), ("--router-factory", "rule"),
            ("--lexical-cache-bytes", "536870912"),
            ("--vector-timeout-s", "30.0"), ("--max-body-bytes", "65536"),
            ("--request-timeout-s", "10.0"),
        ]
        for flag, value in flags:
            with self.subTest(flag=flag), patch.object(server, "DiskIndex") as reader, \
                    patch.object(server, "ThreadingHTTPServer") as http, \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    server.main(["--config", str(self.config_path), flag, value])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("--config", stderr.getvalue())
                reader.assert_not_called()
                http.assert_not_called()

    def test_dataset_mismatch_fails_before_model_backend_or_bind(self):
        with self.cli(dataset="a-different-dataset") as mocks, \
                patch("adapters.embedding.BGEEncoder") as bge, \
                patch.object(server, "load_router") as load_router, \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                server.main(["--config", str(self.config_path)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("dataset", stderr.getvalue())
            self.assertIn("a-different-dataset", stderr.getvalue())
            bge.assert_not_called()
            mocks["QdrantVectorIndex"].assert_not_called()
            load_router.assert_not_called()
            mocks["SearchEngine"].assert_not_called()
            mocks["ThreadingHTTPServer"].assert_not_called()

    def test_config_forwards_relative_paths_hnsw_and_timeout(self):
        with self.cli() as mocks, patch("adapters.embedding.BGEEncoder") as bge:
            server.main(["--config", str(self.config_path)])
            bge.assert_called_once_with(
                spec=self.manifest.embedding,
                model_path=str((self.config_path.parent / "../models/bge").resolve()),
                device="cpu")
            mocks["QdrantVectorIndex"].assert_called_once_with(
                self.manifest, "http://127.0.0.1:6333",
                hnsw_ef=256, timeout_s=12.5)
            retrieval = mocks["SearchEngine"].call_args.args[3]
            self.assertEqual((retrieval.candidate_k, retrieval.rrf_c), (120, 25.0))
            mocks["ThreadingHTTPServer"].assert_called_once()
            self.assertEqual(mocks["ThreadingHTTPServer"].call_args.args[0],
                             ("127.0.0.1", 8123))
            handler = mocks["ThreadingHTTPServer"].call_args.args[1]
            self.assertEqual(handler.timeout, 3.0)

    def test_startup_log_echoes_engine_identity_policy_and_search_config(self):
        with self.cli() as mocks, patch("adapters.embedding.BGEEncoder"):
            server.main(["--config", str(self.config_path)])
            record = mocks["_emit_startup"].call_args.args[0]
        expected = _engine_info()
        self.assertEqual(record["event"], "startup")
        self.assertEqual(record["dataset"], "fixture")
        for name in ("index_identity", "vector_identity", "search_config", "policy"):
            self.assertEqual(record[name], expected[name])
        self.assertEqual(record["policy_version"], "rules-v1")
        # 启动日志只带身份/配置，不带 query/请求体/凭据。
        text = json.dumps(record, ensure_ascii=False)
        for forbidden in ("query", "Authorization", "password"):
            self.assertNotIn(forbidden, text)

    def test_config_uses_router_factory_selector_and_profile_defaults(self):
        value = {"schema_version": 1, "dataset": "fixture", "index": "../index",
                 "policy": {"factory": "custom:factory"}}
        self.write_profile(value)
        fake_router = type("FakeRouter", (), {"policy_version": "custom-v1"})()
        with self.cli() as mocks, patch.object(
                server, "load_router", return_value=fake_router) as load_router:
            server.main(["--config", str(self.config_path)])
            selector, retrieval = load_router.call_args.args
            self.assertEqual(selector, "custom:factory")
            self.assertEqual((retrieval.candidate_k, retrieval.rrf_c), (200, 60.0))
            # No vector block: no encoder/backend is constructed.
            self.assertIsNone(mocks["QdrantVectorIndex"].call_args)
            self.assertIsNone(mocks["SentenceTransformerEncoder"].call_args)
            handler = mocks["ThreadingHTTPServer"].call_args.args[1]
            self.assertEqual(handler.timeout, 10.0)

    def test_legacy_flags_forward_new_bounds_and_vector_timeout(self):
        with self.cli() as mocks:
            server.main([
                "--index", "unused", "--vector-url", "http://127.0.0.1:6333",
                "--vector-manifest", str(self.manifest_path), "--hnsw-ef", "128",
                "--vector-timeout-s", "5.5", "--router-factory", "rule",
                "--max-body-bytes", "1024", "--request-timeout-s", "2.0",
            ])
            mocks["QdrantVectorIndex"].assert_called_once_with(
                self.manifest, "http://127.0.0.1:6333",
                hnsw_ef=128, timeout_s=5.5)
            retrieval = mocks["SearchEngine"].call_args.args[3]
            self.assertEqual((retrieval.candidate_k, retrieval.rrf_c), (200, 60.0))
            handler = mocks["ThreadingHTTPServer"].call_args.args[1]
            self.assertEqual(handler.timeout, 2.0)

    def test_profile_forwards_explicit_packed_cache_budget(self):
        self.profile["retrieval"]["lexical_cache_bytes"] = 536870912
        self.write_profile(self.profile)
        with self.cli() as mocks, patch("adapters.embedding.BGEEncoder"):
            server.main(["--config", str(self.config_path)])
            mocks["DiskIndex"].assert_called_once_with(
                str((self.config_path.parent / "../index").resolve()),
                cache_bytes=536870912)

    def test_legacy_defaults_keep_legacy_index_path(self):
        with self.cli() as mocks:
            server.main(["--index", "fixtures/tiny_index"])
            mocks["DiskIndex"].assert_called_once_with(str(Path("fixtures/tiny_index")))
            mocks["QdrantVectorIndex"].assert_not_called()
            mocks["SentenceTransformerEncoder"].assert_not_called()
            retrieval = mocks["SearchEngine"].call_args.args[3]
            self.assertEqual((retrieval.candidate_k, retrieval.rrf_c), (200, 60.0))
            self.assertEqual(mocks["ThreadingHTTPServer"].call_args.args[0],
                             ("127.0.0.1", 8000))
            self.assertEqual(mocks["ThreadingHTTPServer"].call_args.args[1].timeout, 10.0)

    def test_nonloopback_host_prints_explicit_warning(self):
        with self.cli(), contextlib.redirect_stderr(io.StringIO()) as stderr:
            server.main(["--index", "unused", "--host", "0.0.0.0"])
            self.assertIn("警告", stderr.getvalue())
            self.assertIn("0.0.0.0", stderr.getvalue())

    def test_invalid_legacy_values_fail_before_loading(self):
        cases = [
            ["--port", "70000"], ["--max-body-bytes", "0"],
            ["--lexical-cache-bytes", "-1"], ["--request-timeout-s", "nan"],
            ["--vector-url", "http://127.0.0.1:6333",
             "--vector-manifest", str(self.manifest_path), "--vector-timeout-s", "0"],
            ["--router-factory", ""],
        ]
        for options in cases:
            with self.subTest(options=options), patch.object(server, "DiskIndex") as reader, \
                    patch.object(server, "QdrantVectorIndex") as vectors, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    server.main(["--index", "unused", *options])
                self.assertEqual(error.exception.code, 2)
                reader.assert_not_called()
                vectors.assert_not_called()


if __name__ == "__main__":
    unittest.main()
