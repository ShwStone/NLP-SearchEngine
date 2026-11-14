"""Exercise the paired evaluator through the public search handler."""

from collections import Counter
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from contracts.retrieval import EmbeddingSpec, VectorManifest, VectorRecord, corpus_fingerprint
from online.engine import SearchEngine
from online.memory_index import MemoryIndex
from online.observability import SearchLogger
from online.router import RouteDecision
from online.server import make_handler
from online.vector_index import MemoryVectorIndex

from evaluation.evaluate import load_run
from evaluation.paired_http import collect, get_identities, identity_digest, load_protocol, service_identity, verify
from evaluation.serving_cost import sha256


class FixtureEncoder:
    spec = EmbeddingSpec("test-only", "fixture-v1", dimension=2)
    query_encoder_runtime = {"implementation": "fixture", "device": "cpu", "dtype": "float32",
                             "inventory_sha256": None}

    def encode_query(self, query):
        return [0.0, 1.0] if query == "camera" else [1.0, 0.0]


class DensePolicy:
    policy_version = "paired-dense-fixture-v1"

    def decide(self, features):
        return RouteDecision(True, "fixture_dense", semantic_mode="dense")


def fixture_service(router=None):
    docs = [
        {"doc_id": "phone", "title": "", "text": "phone water"},
        {"doc_id": "case", "title": "", "text": "phone case"},
        {"doc_id": "camera", "title": "", "text": "camera lens"},
    ]
    reader = MemoryIndex.from_corpus(docs)
    encoder = FixtureEncoder()
    corpus = corpus_fingerprint(reader)
    manifest = VectorManifest(corpus, reader.n_docs, encoder.spec,
                              "paired-fixture", backend="memory-exact")
    vectors = MemoryVectorIndex(manifest, [
        VectorRecord(i, doc["doc_id"], encoder.encode_query(doc["doc_id"]))
        for i, doc in enumerate(docs)
    ])
    engine = SearchEngine(reader, encoder, vectors, router=router)
    identity = engine.info()
    bound_identity = {key: identity[key] for key in ("index_identity", "vector_identity",
                                                     "search_config", "policy")}
    startup = {"event": "startup", "dataset": identity["index_identity"]["dataset"],
               "policy_version": identity["policy"]["policy_version"], **bound_identity,
               "query_encoder_runtime": identity["query_encoder_runtime"]}
    return engine, corpus, startup, bound_identity


class PairedHTTPTest(unittest.TestCase):
    def test_version_two_checks_runtime_across_http_profiles_and_startup(self):
        engine, corpus, startup, bound_identity = fixture_service()
        runtime = startup["query_encoder_runtime"]
        other_encoder = FixtureEncoder()
        other_encoder.query_encoder_runtime = {**runtime, "device": "cuda:0"}
        other = SearchEngine(engine.reader, other_encoder, engine.vectors)
        sink = io.StringIO()
        logger = SearchLogger(sink)
        logger.emit(startup)
        with (tempfile.TemporaryDirectory() as temp,
              ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, search_log=logger)) as first,
              ThreadingHTTPServer(("127.0.0.1", 0), make_handler(other)) as second):
            threads = [Thread(target=server.serve_forever, daemon=True) for server in (first, second)]
            for thread in threads:
                thread.start()
            try:
                root = Path(temp)
                query_path = root / "queries.tsv"
                query_path.write_text("q1\tphone\n", encoding="utf-8")
                plan_path = root / "plan.json"
                arms = [{"label": mode, "mode": mode, "policy_version": startup["policy_version"],
                         "endpoint": f"http://127.0.0.1:{profile.server_port}"}
                        for mode, profile in (("lexical", first), ("hybrid", second))]
                plan_path.write_text(json.dumps({
                    "schema_version": 2, "queries_sha256": sha256(query_path.read_bytes()),
                    "corpus_sha256": corpus, "query_ids": ["q1"], "warmup_query_ids": [],
                    "repeats": 1, "depth": 3, "timeout_s": 5, "arms": arms,
                }), encoding="utf-8")
                plan, _, _ = load_protocol(plan_path, query_path)
                with self.assertRaisesRegex(ValueError, "different.*query encoder runtime"):
                    get_identities(plan)
                self.assertEqual(get_identities({**plan, "schema_version": 1}),
                                 dict.fromkeys(("lexical", "hybrid"), bound_identity))
                arms[1]["endpoint"] = arms[0]["endpoint"]
                plan_path.write_text(json.dumps({**plan, "arms": arms}), encoding="utf-8")
                plan, digest, queries = load_protocol(plan_path, query_path)
                collected = collect(plan, digest, queries, root / "capture")
            finally:
                for server, thread in zip((first, second), threads):
                    server.shutdown()
                    thread.join(timeout=5)
            journal = Path(collected["journal"])
            log = root / "server.log"
            log.write_text(sink.getvalue(), encoding="utf-8")
            logs = dict.fromkeys(("lexical", "hybrid"), log)
            report = verify(plan, digest, journal, logs)
            self.assertEqual(report["query_encoder_runtime"], runtime)
            self.assertEqual(report["arms"]["hybrid"]["identity_sha256"],
                             identity_digest({**bound_identity, "query_encoder_runtime": runtime}))
            rows = sink.getvalue().splitlines()
            log.write_text("\n".join([json.dumps({**startup, "query_encoder_runtime":
                           other_encoder.query_encoder_runtime}), *rows[1:]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Client and server startup identity differ"):
                verify(plan, digest, journal, logs)
            with self.assertRaisesRegex(ValueError, "Missing or invalid query encoder runtime"):
                service_identity(bound_identity, corpus, startup["policy_version"], schema_version=2)

    def test_public_handler_roundtrip_and_unverifiable_runs(self):
        engine, corpus, startup, bound_identity = fixture_service()
        sink = io.StringIO()
        logger = SearchLogger(sink)
        logger.emit(startup)
        with (tempfile.TemporaryDirectory() as temp,
              ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, search_log=logger)) as server):
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                root = Path(temp)
                queries_file = root / "queries.jsonl"
                queries_file.write_text("".join(
                    json.dumps({"qid": qid, "query": text}) + "\n"
                    for qid, text in [("q1", "phone"), ("q2", "camera"), ("warm", "case")]
                ), encoding="utf-8")
                modes = ["lexical", "dense", "hybrid", "adaptive"]
                plan_file = root / "plan.json"
                plan_file.write_text(json.dumps({
                    "schema_version": 1, "queries_sha256": sha256(queries_file.read_bytes()),
                    "corpus_sha256": corpus, "query_ids": ["q1", "q2"], "warmup_query_ids": ["warm"],
                    "repeats": 2, "depth": 3, "timeout_s": 5,
                    "arms": [{"label": mode, "mode": mode, "policy_version": startup["policy_version"],
                              "identity_sha256": identity_digest(bound_identity),
                              "endpoint": f"http://127.0.0.1:{server.server_port}"} for mode in modes],
                }), encoding="utf-8")
                plan, digest, queries = load_protocol(plan_file, queries_file)
                collected = collect(plan, digest, queries, root / "collect")
            finally:
                server.shutdown()
                thread.join(timeout=5)
            journal = Path(collected["journal"])
            first_arms = Counter(
                event["request"]["label"]
                for event in map(json.loads, journal.read_bytes().splitlines())
                if event["event"] == "BEGIN" and event["request"]["phase"] == "measure"
                and event["request"]["position"] == 0
            )
            self.assertEqual(first_arms, dict.fromkeys(modes, 1))
            log = root / "server.log"
            lines = sink.getvalue().splitlines()
            log.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logs = dict.fromkeys(modes, log)
            output = root / "trec"
            result = verify(plan, digest, journal, logs, output)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(result["trec_runs"]["repeat"], 0)
            self.assertEqual(len(result["paired_client_latency"]), 6)
            self.assertEqual({r["queries"] for r in result["paired_client_latency"]}, {2})
            for mode in modes:
                self.assertNotIn("allow_adaptive_dense", result["arms"][mode])
                self.assertEqual(result["arms"][mode]["client_ms"]["n"], 4)
                self.assertEqual(result["arms"][mode]["warmup_requests"], 1)
                path = output / f"{mode}.trec"
                self.assertEqual(sha256(path.read_bytes()), result["trec_runs"]["arms"][mode]["sha256"])
                self.assertEqual(set(load_run(path)), {"q1", "q2"})
            self.assertEqual(result["arms"]["dense"]["vector_attempts"], 4)
            self.assertEqual(result["arms"]["dense"]["route_counts"], {"dense": 4})
            self.assertEqual(result["arms"]["dense"]["identity_sha256"], identity_digest(bound_identity))
            self.assertEqual(result["arms"]["lexical"]["vector_attempts"], 0)

            startup_with_ce = {**json.loads(lines[0]), "reranker": {"depth": 40}}
            log.write_text("\n".join([json.dumps(startup_with_ce), *lines[1:]]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CE-enabled hybrid arm"):
                verify(plan, digest, journal, logs)

            # A duplicate server event and a second startup both invalidate the
            # time sample, even if every individual response still looks valid.
            bad = root / "rejected"
            for mutated in [lines + lines[-1:], lines[:-1] + [lines[0], lines[-1]]]:
                log.write_text("\n".join(mutated) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify(plan, digest, journal, logs, bad)
                self.assertFalse(bad.exists())
            journal.write_bytes(journal.read_bytes().rsplit(b"\n", 2)[0] + b"\n")
            with self.assertRaises(ValueError):
                verify(plan, digest, journal, logs, bad)
            self.assertFalse(bad.exists())

    def test_hybrid_baseline_rejects_ce_at_collection_even_with_matching_digest(self):
        from online.reranking import HybridReranker
        from tests.test_peer_routing_integration import RecordingScorer

        engine, corpus, _, identity = fixture_service()
        engine = SearchEngine(engine.reader, engine.encoder, engine.vectors,
                              reranker=HybridReranker(RecordingScorer(), depth=3))
        with ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine)) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                endpoint = f"http://127.0.0.1:{server.server_port}"
                arm = {"label": "baseline", "mode": "hybrid", "endpoint": endpoint,
                       "policy_version": identity["policy"]["policy_version"],
                       "identity_sha256": identity_digest(identity)}
                plan = {"arms": [arm], "corpus_sha256": corpus, "depth": 3, "timeout_s": 5}
                with self.assertRaisesRegex(ValueError, "CE-enabled hybrid arm"):
                    get_identities(plan)
                arm["mode"] = "lexical"
                self.assertEqual(get_identities(plan)["baseline"], identity)
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_adaptive_dense_requires_an_opt_in_and_frozen_service_identity(self):
        engine, corpus, startup, bound_identity = fixture_service(DensePolicy())
        sink = io.StringIO()
        logger = SearchLogger(sink)
        logger.emit(startup)
        with (tempfile.TemporaryDirectory() as temp,
              ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine, search_log=logger)) as server):
            root = Path(temp)
            query_file = root / "queries.jsonl"
            query_file.write_text('{"qid":"q","query":"phone"}\n', encoding="utf-8")
            plan_file = root / "plan.json"
            identity = identity_digest(bound_identity)
            service = {"endpoint": f"http://127.0.0.1:{server.server_port}",
                       "policy_version": startup["policy_version"]}
            lexical = {**service, "label": "base", "mode": "lexical"}
            adaptive = {**service, "label": "route", "mode": "adaptive", "identity_sha256": identity}

            def frozen(arm):
                plan_file.write_text(json.dumps({
                    "schema_version": 1, "queries_sha256": sha256(query_file.read_bytes()),
                    "corpus_sha256": corpus, "query_ids": ["q"], "warmup_query_ids": [],
                    "repeats": 1, "depth": 3, "timeout_s": 5, "arms": [lexical, arm],
                }), encoding="utf-8")
                return load_protocol(plan_file, query_file)

            enabled = {**adaptive, "allow_adaptive_dense": True}
            for arm, error in (
                ({**adaptive, "allow_adaptive_dense": "true"}, "must be a boolean"),
                ({**enabled, "mode": "dense"}, "requires an adaptive arm"),
                ({**service, "label": "route", "mode": "adaptive", "allow_adaptive_dense": True},
                 "requires an adaptive arm"),
                ({**enabled, "identity_sha256": False}, "Invalid arm identity SHA256"),
            ):
                with self.subTest(arm=arm), self.assertRaisesRegex(ValueError, error):
                    frozen(arm)

            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                omitted, omitted_digest, queries = frozen(adaptive)
                with self.assertRaisesRegex(ValueError, "执行模式或向量标记不一致"):
                    collect(omitted, omitted_digest, queries, root / "omitted")
                wrong, wrong_digest, queries = frozen({**enabled, "identity_sha256": "0" * 64})
                with self.assertRaisesRegex(ValueError, "Effective service identity differs"):
                    collect(wrong, wrong_digest, queries, root / "wrong")
                plan, digest, queries = frozen(enabled)
                journal = Path(collect(plan, digest, queries, root / "allowed")["journal"])
            finally:
                server.shutdown()
                thread.join(timeout=5)

            log = root / "server.log"
            log.write_text(sink.getvalue(), encoding="utf-8")
            logs = {"base": log, "route": log}
            result = verify(plan, digest, journal, logs)
            self.assertIs(result["arms"]["route"]["allow_adaptive_dense"], True)
            self.assertEqual(result["arms"]["route"]["route_counts"], {"dense": 1})

            # A matching START digest is insufficient when the frozen plan forbids the route.
            events = [json.loads(line) for line in journal.read_bytes().splitlines()]
            events[0]["plan_sha256"] = omitted_digest
            rebound = root / "rebound.jsonl"
            rebound.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "执行模式或向量标记不一致"):
                verify(omitted, omitted_digest, rebound, logs)

            logged = [json.loads(line) for line in log.read_bytes().splitlines()]
            tag = next(event["tag"] for event in events if event["event"] == "OUTCOME"
                       and event["routing"]["executed_mode"] == "dense")
            next(event for event in logged if event.get("run_tag") == tag)["executed_mode"] = "hybrid"
            log.write_text("".join(json.dumps(event) + "\n" for event in logged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "routing disagreement"):
                verify(plan, digest, journal, logs)


if __name__ == "__main__":
    unittest.main()
