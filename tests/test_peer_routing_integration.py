"""Exercise the peer routing and CE seams through the public search handler."""

from contextlib import closing, redirect_stderr
from dataclasses import asdict
from http.server import ThreadingHTTPServer
import hashlib
import io
import json
import math
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from contracts.ce import GATE_DEPTH, GATE_FEATURE_NAMES, gate_binding
from contracts.index_format import BM25Spec
from contracts.retrieval import EmbeddingSpec, VectorManifest, VectorRecord, corpus_fingerprint
from evaluation import ce_http
from evaluation.evaluate import EvaluationError, QueryRecord, evaluate_http
from online import server
from online.ce_gate import load_gate
from online.engine import SearchEngine
from online.gain_router import load_gain_router
from online.memory_index import MemoryIndex
from online.observability import SearchLogger
from online.reranking import HybridReranker
from online.router import RouteDecision
from online.router_features import FEATURE_NAMES, FEATURE_VERSION
from online.server import make_handler
from online.vector_index import MemoryVectorIndex


CORPUS = [
    {"doc_id": "z", "title": "", "text": "phone alpha"},
    {"doc_id": "a", "title": "", "text": "phone bravo"},
    {"doc_id": "m", "title": "", "text": "gadget charlie"},
    {"doc_id": "b", "title": "", "text": "camera delta"},
]


class FixtureEncoder:
    spec = EmbeddingSpec("test-only", "fixture-v1", dimension=2)

    def __init__(self):
        self.calls = []

    def encode_query(self, query):
        self.calls.append(query)
        return [1.0, 0.0]


class RecordingScorer:
    identity = {"model_id": "fixture-ce", "revision": "test-v1"}

    def __init__(self):
        self.calls = []
        self.scores = [2.0, 2.0, 9.0]

    def score(self, query, texts):
        self.calls.append((query, tuple(texts)))
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores


class QueryAwareDensePolicy:
    policy_version = "test-query-aware-dense-v1"
    requires_statistics = True

    def __init__(self):
        self.statistics_calls = []
        self.decisions = []

    def statistics_for_query(self, query, reader, terms, ranked):
        self.statistics_calls.append((query, reader, terms, ranked))
        return {"query_chars": len(query)}

    def decide(self, features):
        self.decisions.append(features)
        return RouteDecision(True, "fixture_dense", semantic_mode="dense")


class PeerRoutingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.reader = MemoryIndex.from_corpus(CORPUS)
        self.encoder = FixtureEncoder()
        manifest = VectorManifest(corpus_fingerprint(self.reader), self.reader.n_docs,
                                  self.encoder.spec, "peer-integration", backend="memory-exact")
        self.vectors = MemoryVectorIndex(manifest, [
            VectorRecord(i, row["doc_id"], [1.0, 0.0]) for i, row in enumerate(CORPUS)
        ])
        self.opener = build_opener(ProxyHandler({}))

    def serve(self, *, router=None, reranker=None):
        engine = SearchEngine(self.reader, self.encoder, self.vectors,
                              router=router, reranker=reranker)
        sink = io.StringIO()
        service = self.start_service(engine, sink)
        return f"http://127.0.0.1:{service.server_port}/search", sink

    def start_service(self, engine, sink):
        handler = make_handler(engine, search_log=SearchLogger(sink))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01},
                                  daemon=True)
        thread.start()

        def stop():
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.addCleanup(stop)
        return server

    def post(self, endpoint, **payload):
        request = Request(endpoint, json.dumps(payload).encode("utf-8"),
                          {"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(request, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            with error:
                return error.code, json.load(error)

    def search(self, endpoint, **payload):
        status, body = self.post(endpoint, **payload)
        self.assertEqual(status, 200, body)
        return body

    def test_index_info_only_exposes_supported_vector_search_parameters(self):
        endpoint, _ = self.serve()
        info = endpoint.removesuffix("/search") + "/index/info"
        self.vectors.search_params = SimpleNamespace(hnsw_ef=128, exact=False)
        with self.opener.open(info, timeout=5) as response:
            self.assertIsNone(json.load(response)["retrieval"]["vector_search_params"])
        self.vectors.search_params = {"hnsw_ef": 128, "exact": False, "internal": object()}
        with self.opener.open(info, timeout=5) as response:
            self.assertEqual(json.load(response)["retrieval"]["vector_search_params"],
                             {"hnsw_ef": 128, "exact": False})

    def write_gate(self, path, scorer):
        engine = SearchEngine(self.reader, self.encoder, self.vectors)
        artifact = {
            "schema_version": 1, "feature_names": list(GATE_FEATURE_NAMES),
            "mean": [0.0] * 4, "scale": [1.0] * 4,
            "coef": [0.0, -3.0, 0.0, 0.0], "intercept": 2.5,
            "binding": gate_binding(engine.info(), scorer.identity),
        }
        raw = json.dumps(artifact).encode("utf-8")
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def test_ce_capture_resumes_in_a_new_epoch_after_same_config_restart(self):
        self.encoder.query_encoder_runtime = {
            "implementation": "fixture", "device": "cpu", "dtype": "float32", "inventory_sha256": None,
        }
        def ce_scorer():
            scorer = RecordingScorer()
            scorer.identity = {**scorer.identity, "weights_sha256": "a" * 64,
                               "inventory_sha256": "b" * 64}
            scorer.scores = [1.0, 2.0, 3.0, 4.0]
            return scorer

        def engine(reranker=None):
            return SearchEngine(self.reader, self.encoder, self.vectors, reranker=reranker)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            gate_path = root / "gate.json"
            gate = load_gate(gate_path, self.write_gate(gate_path, ce_scorer()))
            fixed = engine(HybridReranker(ce_scorer(), depth=40, capture_features=True))
            sink = io.StringIO()
            ce40 = self.start_service(fixed, sink)
            profiles = {
                "rrf": self.start_service(engine(), io.StringIO()),
                "ce28": self.start_service(engine(HybridReranker(ce_scorer(), depth=28)), io.StringIO()),
                "ce40": ce40,
                "learned": self.start_service(
                    engine(HybridReranker(ce_scorer(), depth=40, gate=gate)), io.StringIO()),
            }
            urls = {name: f"http://127.0.0.1:{service.server_port}" for name, service in profiles.items()}
            gpu_encoder = FixtureEncoder()
            gpu_encoder.query_encoder_runtime = {**self.encoder.query_encoder_runtime, "device": "cuda:0"}
            mismatched = self.start_service(SearchEngine(
                self.reader, gpu_encoder, self.vectors,
                reranker=HybridReranker(ce_scorer(), depth=40, capture_features=True),
            ), io.StringIO())
            with self.assertRaisesRegex(EvaluationError, "service identity differs from baseline"):
                ce_http.check_services(ce_http.http_client(True), {
                    "rrf": (0, urls["rrf"]),
                    "ce40": (40, f"http://127.0.0.1:{mismatched.server_port}"),
                })
            (root / "queries.tsv").write_text("q1\tphone\nq2\tphone alpha missing\n", encoding="utf-8")
            (root / "warmup.tsv").write_text("w1\tcamera\n", encoding="utf-8")
            (root / "qrels.tsv").write_text("q1 0 z 1\nq2 0 z 1\n", encoding="utf-8")
            out = root / "capture"
            args = [
                "--baseline", urls["rrf"], "--rerank", "28", urls["ce28"],
                "--rerank", "40", urls["ce40"], "--learned", urls["learned"],
                "--queries", str(root / "queries.tsv"), "--qrels", str(root / "qrels.tsv"),
                "--warmup-queries", str(root / "warmup.tsv"), "--out", str(out),
                "--require-instance-id", "--require-encoder-runtime", "--measure-random",
            ]
            fit_receipt = root / "fit-receipt.json"
            fit_identity = engine().info()
            fit_identity["query_encoder_runtime"]["device"] = "cuda:0"
            fit = {"schema_version": 1, "model_sha256": gate.sha256,
                   "capture_identities": {"common": {"retrieval": fit_identity},
                                          "rerankers": {"ce40": {"model": ce_scorer().identity}}}}
            fit_receipt.write_text(json.dumps(fit), encoding="utf-8")
            with patch.object(ce_http, "capture_query", side_effect=AssertionError("capture attempted")):
                with self.assertRaisesRegex(EvaluationError, "fit receipt SHA256 differs"):
                    ce_http.main([*args, "--fit-receipt", str(fit_receipt), "0" * 64])
                with self.assertRaisesRegex(EvaluationError, "differs from the frozen training capture"):
                    ce_http.main([*args, "--fit-receipt", str(fit_receipt), ce_http.sha256(fit_receipt)])
            fit_identity["query_encoder_runtime"] = self.encoder.query_encoder_runtime
            fit_identity["bm25_accelerator"] = "compiled-float64-topk"
            fit_receipt.write_text(json.dumps(fit), encoding="utf-8")
            with patch.object(ce_http, "capture_query", side_effect=AssertionError("capture attempted")):
                with self.assertRaisesRegex(EvaluationError, "BM25 accelerator differs"):
                    ce_http.main([*args, "--fit-receipt", str(fit_receipt), ce_http.sha256(fit_receipt)])
            fit_identity["bm25_accelerator"] = None
            fit_receipt.write_text(json.dumps(fit), encoding="utf-8")
            fit_digest = ce_http.sha256(fit_receipt)
            args.extend(["--fit-receipt", str(fit_receipt), fit_digest])
            capture = ce_http.capture_query

            def restart_before_second_query(client, arms, record, *rest):
                if record.qid == "q2":
                    # Keep the endpoint and effective profile identical to a process restart.
                    ce40.RequestHandlerClass = make_handler(fixed, search_log=SearchLogger(sink))
                return capture(client, arms, record, *rest)

            with patch.object(ce_http, "capture_query", side_effect=restart_before_second_query):
                with self.assertRaisesRegex(EvaluationError, "q2.*ce40.*restarted"):
                    ce_http.main(args)
            self.assertFalse((out / "summary.json").exists())
            with closing(sqlite3.connect(out / "capture.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT qid FROM queries").fetchall(), [("q1",)])

            ce_http.main(args)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual(summary["protocol"]["identities"]["common"]["retrieval"]["query_encoder_runtime"],
                             self.encoder.query_encoder_runtime)
            self.assertEqual(summary["protocol"]["fit_receipt_sha256"], fit_digest)
            instances = list(summary["service_instances_by_epoch"].values())
            self.assertEqual(len(instances), 2)
            self.assertNotEqual(instances[0]["ce40"], instances[1]["ce40"])
            self.assertEqual(instances[0]["rrf"], instances[1]["rrf"])
            live = json.loads((out / "matched-random-live.json").read_bytes())
            self.assertEqual(live["ce_pairs"], {"learned": 4, "random": 4})
            self.assertEqual(len(live["service_instances_by_epoch"]), 1)
            live_instances = next(iter(live["service_instances_by_epoch"].values()))
            self.assertTrue(any(live_instances.items() <= row.items() for row in instances))
            replay_dir = root / "replay"
            replay = [
                "--replay-from", str(out), "--out", str(replay_dir), "--measure-random",
                *args[args.index("--queries"):args.index("--out")],
            ]
            with patch.object(ce_http, "http_client", side_effect=AssertionError("network used")):
                with redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit):
                    ce_http.main(replay)
                self.assertIn("requires its original --fit-receipt", error.getvalue())
                ce_http.main([*replay, "--fit-receipt", str(fit_receipt), fit_digest])
            self.assertEqual(
                {path.name: path.read_bytes() for path in replay_dir.iterdir()},
                {path.name: path.read_bytes() for path in out.iterdir() if path.suffix != ".sqlite3"},
            )

    def test_learned_gate_skips_or_applies_the_same_fixed_ce40(self):
        scorer, fixed_scorer = RecordingScorer(), RecordingScorer()
        scorer.scores = fixed_scorer.scores = [1.0, 2.0, 3.0, 4.0]
        baseline_endpoint, _ = self.serve()
        fixed_endpoint, _ = self.serve(reranker=HybridReranker(
            fixed_scorer, depth=GATE_DEPTH, capture_features=True))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            digest = self.write_gate(path, scorer)
            gate = load_gate(path, digest)
        learned_endpoint, _ = self.serve(reranker=HybridReranker(
            scorer, depth=GATE_DEPTH, gate=gate))
        info_url = learned_endpoint.rsplit("/", 1)[0] + "/index/info"
        with self.opener.open(info_url, timeout=5) as response:
            reranker_info = json.load(response)["retrieval"]["reranker"]
        self.assertEqual(reranker_info["gate_sha256"], digest)
        self.assertIs(reranker_info["capture_features"], True)

        request = {"query": "phone", "mode": "hybrid", "k": 4}
        baseline = self.search(baseline_endpoint, **request)
        fixed = self.search(fixed_endpoint, **request)
        skipped = self.search(learned_endpoint, **request)
        self.assertNotIn("rerank", baseline)
        self.assertEqual(fixed["rerank"], {"depth": GATE_DEPTH, "scored": 4,
                                          "features": [1.0, 1.0, 0.0, 2.0]})
        self.assertEqual(skipped["rerank"], {**fixed["rerank"], "scored": 0,
                                            "gate_sha256": digest})
        self.assertEqual(skipped["results"], baseline["results"])
        self.assertNotEqual(fixed["results"], baseline["results"])
        self.assertEqual(scorer.calls, [])

        # Two of three distinct terms are in the vocabulary. BM25 hits only
        # two documents; the fused list also contains two vector-only hits.
        request["query"] = "phone alpha missing"
        lexical = self.search(baseline_endpoint, **{**request, "mode": "lexical"})
        first, second = [row["score"] for row in lexical["results"]]
        self.assertGreater(first, second)
        fixed = self.search(fixed_endpoint, **request)
        applied = self.search(learned_endpoint, **request)
        features = fixed["rerank"]["features"]
        self.assertEqual((features[0], features[1], features[3]), (3.0, 2 / 3, 2.0))
        # Lexical scores on the wire are rounded, unlike the gate inputs.
        self.assertAlmostEqual(features[2], (first - second) / first, delta=1e-6)
        self.assertEqual(fixed["rerank"], {"depth": GATE_DEPTH, "scored": 4,
                                          "features": features})
        self.assertEqual(applied["rerank"], {**fixed["rerank"], "gate_sha256": digest})
        self.assertEqual(applied["results"], fixed["results"])
        self.assertEqual(scorer.calls, fixed_scorer.calls[1:])
        self.assertEqual(len(scorer.calls), 1)

        constrained = self.search(learned_endpoint, query="phone AND alpha", mode="hybrid", k=4)
        self.assertNotIn("rerank", constrained)
        self.assertEqual(len(scorer.calls), 1)

    def test_cli_rejects_wrong_gate_bindings_before_listening(self):
        scorer = RecordingScorer()
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "serving.json"
            digest = self.write_gate(config_path.with_name("gate.json"), scorer)
            config = {
                "schema_version": 1, "dataset": "memory", "index": "unused-index",
                "vector": {"url": "http://127.0.0.1:6333", "manifest": "unused-manifest"},
                "rerank": {"model_path": "unused-model", "depth": GATE_DEPTH,
                           "gate": {"model_path": "gate.json", "sha256": digest}},
            }
            for mismatch in ("retrieval", "model"):
                with self.subTest(mismatch=mismatch):
                    active = dict(config)
                    if mismatch == "retrieval":
                        active["retrieval"] = {"rrf_c": 25.0}
                    changed_scorer = RecordingScorer()
                    if mismatch == "model":
                        changed_scorer.identity = {**scorer.identity, "revision": "wrong"}
                    config_path.write_text(json.dumps(active), encoding="utf-8")
                    with (patch.object(server, "_load_reader", return_value=self.reader),
                          patch.object(server, "_load_vector_backend",
                                       return_value=(self.encoder, self.vectors)),
                          patch("adapters.reranker.FrozenCrossEncoder", return_value=changed_scorer),
                          patch.object(server, "ThreadingHTTPServer") as listen,
                          redirect_stderr(io.StringIO()) as stderr):
                        with self.assertRaises(SystemExit) as raised:
                            server.main(["--config", str(config_path)])
                        self.assertEqual(raised.exception.code, 2)
                        self.assertIn(f"CE gate {mismatch} binding", stderr.getvalue())
                        listen.assert_not_called()

    def test_hybrid_reranks_fixed_prefix_before_pagination(self):
        baseline_endpoint, _ = self.serve()
        scorer = RecordingScorer()
        rerank_endpoint, _ = self.serve(reranker=HybridReranker(scorer, depth=3))
        request = {"query": "phone", "mode": "hybrid"}

        baseline = self.search(baseline_endpoint, **request, k=4)
        self.assertEqual([row["doc_id"] for row in baseline["results"]], ["z", "a", "m", "b"])
        reranked = self.search(rerank_endpoint, **request, k=4)
        self.assertEqual([row["doc_id"] for row in reranked["results"]], ["m", "a", "z", "b"])
        self.assertEqual([row["rank"] for row in reranked["results"]], [1, 2, 3, 4])
        self.assertEqual({row["doc_id"]: row["score"] for row in reranked["results"]},
                         {row["doc_id"]: row["score"] for row in baseline["results"]})
        self.assertEqual(reranked["results"][-1], baseline["results"][-1])
        self.assertEqual(reranked["rerank"]["depth"], 3)
        self.assertEqual(reranked["rerank"]["scored"], 3)
        self.assertTrue(math.isfinite(reranked["timings_ms"]["rerank"]))
        self.assertGreaterEqual(reranked["timings_ms"]["rerank"], 0)
        self.assertNotIn("rerank", baseline)

        page = self.search(rerank_endpoint, **request, offset=1, k=1)
        self.assertEqual(page["results"], reranked["results"][1:2])
        self.assertEqual(page["rerank"]["scored"], 3)
        self.assertEqual(len(scorer.calls), 2)
        self.assertEqual(scorer.calls[0], scorer.calls[1])
        query, texts = scorer.calls[0]
        self.assertEqual(query, "phone")
        self.assertEqual([text.split()[-1] for text in texts], ["alpha", "bravo", "charlie"])

    def test_compiled_bm25_and_cross_encoder_share_the_same_hybrid_prefix(self):
        try:
            from online.compiled_bm25 import CompiledBM25
        except ModuleNotFoundError as error:
            if error.name != "numpy":
                raise
            self.skipTest("optional NumPy accelerator is not installed")
        reference_scorer, compiled_scorer = RecordingScorer(), RecordingScorer()
        reference, _ = self.serve(reranker=HybridReranker(reference_scorer, depth=3))
        engine = SearchEngine(
            self.reader, self.encoder, self.vectors,
            bm25_accelerator=CompiledBM25(self.reader),
            reranker=HybridReranker(compiled_scorer, depth=3),
        )
        service = self.start_service(engine, io.StringIO())
        compiled = f"http://127.0.0.1:{service.server_port}/search"
        request = {"query": "phone", "mode": "hybrid", "offset": 1, "k": 2}
        expected = self.search(reference, **request)
        actual = self.search(compiled, **request)
        self.assertEqual(actual["results"], expected["results"])
        self.assertEqual(actual["rerank"], expected["rerank"])
        self.assertEqual(compiled_scorer.calls, reference_scorer.calls)
        self.assertEqual(len(compiled_scorer.calls), 1)

        constrained = self.search(compiled, query="phone AND NOT camera", mode="hybrid", k=3)
        self.assertNotIn("rerank", constrained)
        self.assertEqual(len(compiled_scorer.calls), 1)

    def test_rerank_metadata_counts_actual_pairs_for_a_short_pool(self):
        scorer = RecordingScorer()
        scorer.scores = [1.0, 2.0, 3.0, 4.0]
        endpoint, _ = self.serve(reranker=HybridReranker(scorer, depth=6))
        body = self.search(endpoint, query="phone", mode="hybrid", k=1)
        self.assertEqual(body["rerank"]["depth"], 6)
        self.assertEqual(body["rerank"]["scored"], 4)
        self.assertEqual(body["results"][0]["doc_id"], "b")
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(len(scorer.calls[0][1]), 4)

    def test_reranker_bypassed_for_explicit_constraints_and_other_modes(self):
        baseline_endpoint, _ = self.serve()
        scorer = RecordingScorer()
        rerank_endpoint, _ = self.serve(reranker=HybridReranker(scorer, depth=3))
        for query, mode in [("phone", "lexical"), ("phone", "dense"),
                            ("phone", "adaptive"), ("phone AND alpha", "hybrid")]:
            with self.subTest(query=query, mode=mode):
                baseline = self.search(baseline_endpoint, query=query, mode=mode, k=4)
                bypassed = self.search(rerank_endpoint, query=query, mode=mode, k=4)
                self.assertEqual(bypassed["results"], baseline["results"])
                self.assertEqual(bypassed["routing"], baseline["routing"])
                self.assertNotIn("rerank", bypassed)
                self.assertNotIn("rerank", bypassed["timings_ms"])
        self.assertEqual(scorer.calls, [])

    def test_scorer_failures_are_sanitized_on_the_public_endpoint(self):
        scorer = RecordingScorer()
        endpoint, sink = self.serve(reranker=HybridReranker(scorer, depth=3))
        secret = "private fixture scorer exception"
        for scores in (RuntimeError(secret), [1.0], [1.0, math.nan, 2.0]):
            with self.subTest(scores=scores):
                scorer.scores = scores
                status, body = self.post(endpoint, query="phone", mode="hybrid", k=4)
                self.assertEqual(status, 503, body)
                self.assertEqual(body["code"], "reranker_unavailable")
                self.assertNotIn(secret, json.dumps(body))
        self.assertNotIn(secret, sink.getvalue())

    def test_query_statistics_can_select_dense_without_bypassing_constraints(self):
        router = QueryAwareDensePolicy()
        endpoint, _ = self.serve(router=router)
        adaptive = self.search(endpoint, query="camera", mode="adaptive", k=4)
        dense = self.search(endpoint, query="camera", mode="dense", k=4)
        hybrid = self.search(endpoint, query="camera", mode="hybrid", k=4)
        self.assertEqual(adaptive["results"], dense["results"])
        self.assertNotEqual(adaptive["results"][0]["doc_id"], hybrid["results"][0]["doc_id"])
        route = adaptive["routing"]
        self.assertEqual(route["requested_mode"], "adaptive")
        self.assertEqual(route["executed_mode"], "dense")
        self.assertEqual(route["score_kind"], "cosine")
        self.assertEqual(route["reason"], "fixture_dense")
        self.assertEqual(route["policy_version"], router.policy_version)
        self.assertTrue(route["vector_attempted"])
        self.assertTrue(route["vector_used"])

        vector_calls = len(self.encoder.calls)
        constrained = self.search(endpoint, query="phone AND alpha", mode="adaptive", k=4)
        self.assertEqual([row["doc_id"] for row in constrained["results"]], ["z"])
        self.assertEqual(constrained["routing"]["reason"], "explicit_constraints")
        self.assertFalse(constrained["routing"]["vector_attempted"])
        self.assertEqual(len(self.encoder.calls), vector_calls)

        self.assertEqual(len(router.statistics_calls), 1)
        query, reader, terms, ranked = router.statistics_calls[0]
        self.assertEqual(query, "camera")
        self.assertIs(reader, self.reader)
        self.assertEqual(terms, ("camera",))
        self.assertIsInstance(ranked, tuple)
        self.assertEqual([docid for docid, _ in ranked], [3])
        self.assertEqual(len(router.decisions), 1)
        self.assertEqual(router.decisions[0].statistics, {"query_chars": 6})

    def test_http_evaluator_requires_opt_in_for_the_dense_adaptive_action(self):
        endpoint, _ = self.serve(router=QueryAwareDensePolicy())
        records = [QueryRecord("camera-qid", "camera")]
        qrels = {"camera-qid": {"b": 1.0}}
        with self.assertRaises(EvaluationError):
            evaluate_http(endpoint, records, qrels, mode="adaptive")
        summary = evaluate_http(endpoint, records, qrels, mode="adaptive",
                                allow_adaptive_dense=True, fail_on_fallback=True)
        self.assertEqual(summary.n_queries, 1)
        self.assertEqual(summary.routing.vector_successes, 1)
        self.assertEqual(summary.per_query[0].routing.executed_mode, "dense")

    def test_peer_factory_binds_model_digest_and_effective_vector_search(self):
        # Peer artifacts bind the published BM25 manifest. The generic
        # MemoryIndex omits that block and relies on the scorer's defaults.
        manifest = {**self.reader.manifest(), "bm25": asdict(BM25Spec())}
        manifest_override = patch.object(self.reader, "manifest", return_value=manifest)
        manifest_override.start()
        self.addCleanup(manifest_override.stop)
        model = {
            "schema_version": 1, "kind": "action-linear",
            "feature_version": FEATURE_VERSION, "feature_names": list(FEATURE_NAMES),
            "selected_features": ["query_chars"], "mean": [0.0], "scale": [1.0],
            "heads": {"dense": {"coef": [0.0], "intercept": 1.0},
                      "hybrid": {"coef": [0.0], "intercept": 0.0}},
            "threshold": 0.0, "margin": 0.0,
            "provenance": {"source": {
                "candidate_k": 200, "rrf_c": 60.0, "source_manifest": self.reader.manifest(),
                "vector_manifest": self.vectors.manifest().to_dict(),
                "vector_search_params": {"hnsw_ef": None, "exact": True},
            }},
        }
        raw = json.dumps(model).encode("utf-8")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "test-model.json"
            path.write_bytes(raw)
            with patch.dict("os.environ", {"NLP_PEER_GAIN_MODEL": str(path),
                                            "NLP_PEER_GAIN_SHA256": hashlib.sha256(raw).hexdigest()}):
                router = load_gain_router(None)
                endpoint, _ = self.serve(router=router)
                actual = self.search(endpoint, query="camera", mode="adaptive", k=4)
                self.assertEqual(actual["routing"]["reason"], "predicted_dense_gain")
                router.model["provenance"]["source"]["vector_search_params"] = {
                    "hnsw_ef": 128, "exact": False,
                }
                with self.assertRaisesRegex(ValueError, "ANN configuration mismatch"):
                    SearchEngine(self.reader, self.encoder, self.vectors, router=router)
                path.write_bytes(raw + b"\n")
                with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                    load_gain_router(None)

if __name__ == "__main__":
    unittest.main()
