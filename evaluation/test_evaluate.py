"""Small stdlib-only tests for the HTTP-only evaluator."""

import io
import hashlib
import json
import tempfile
import threading
import unittest
from contextlib import contextmanager, redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluate import (
    DENSE_CONSTRAINT_ERROR,
    EvaluationError,
    QueryRecord,
    QueryRouting,
    _atomic_json,
    evaluate_http,
    evaluate_run,
    load_qrels,
    load_queries,
    load_run,
    main,
)
from evaluation.compare import paired_bootstrap


class RequestSnapshotTests(unittest.TestCase):
    def test_transient_windows_reader_lock_keeps_old_snapshot_until_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "requests.json"
            snapshot.write_text('{"count":1}', encoding="utf-8")
            pending = snapshot.with_suffix(".json.tmp")
            replace = Path.replace
            attempts = []

            def replace_after_reader_closes(source, target):
                attempts.append(source)
                if len(attempts) == 1:
                    self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), {"count": 1})
                    self.assertEqual(json.loads(pending.read_text(encoding="utf-8")), {"count": 2})
                    error = PermissionError("reader is open without delete sharing")
                    error.winerror = 5
                    raise error
                return replace(source, target)

            with patch.object(Path, "replace", new=replace_after_reader_closes), \
                    patch("evaluation.evaluate.time.sleep") as pause:
                _atomic_json(snapshot, {"count": 2})
            pause.assert_called_once_with(0.01)
            self.assertEqual(attempts, [pending, pending])
            self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), {"count": 2})
            self.assertFalse(pending.exists())

    def test_persistent_windows_lock_fails_with_both_snapshots_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "requests.json"
            snapshot.write_text('{"count":1}', encoding="utf-8")
            error = PermissionError("persistent reader lock")
            error.winerror = 32
            with patch.object(Path, "replace", side_effect=error) as rename, \
                    patch("evaluation.evaluate.time.sleep") as pause, self.assertRaises(PermissionError):
                _atomic_json(snapshot, {"count": 2})
            self.assertEqual(rename.call_count, 7)
            self.assertEqual(pause.call_count, 6)
            self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), {"count": 1})
            pending = snapshot.with_suffix(".json.tmp")
            self.assertEqual(json.loads(pending.read_text(encoding="utf-8")), {"count": 2})


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        self.server.requests.append(request)
        query, mode = request["query"], request["mode"]
        if query in ("unsupported", "wrong-error", "unavailable", "non-json"):
            status = 503 if query == "unavailable" else 400
            body = (b"not json" if query == "non-json" else json.dumps({"error":
                    DENSE_CONSTRAINT_ERROR if query == "unsupported" else "different error"}).encode())
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        rows = [{"doc_id": "d1", "score": 2.0}, {"doc_id": "d2", "score": 1.0}]
        if query == "deep":
            rows = [{"doc_id": f"d{i}", "score": float(121 - i)} for i in range(1, 121)]
        elif query == "q2":
            rows = [{"doc_id": "d3", "score": 1.0}]
        fallback = mode == "adaptive" and query == "fallback"
        used = mode in ("dense", "hybrid") or mode == "adaptive" and query not in ("skip", "fallback")
        routing = {
            "requested_mode": mode,
            "executed_mode": ("dense" if mode == "dense" else "hybrid") if used else "lexical",
            "reason": "vector_unavailable" if fallback else "test",
            "vector_attempted": used or fallback,
            "vector_used": used,
            "fallback": fallback,
        }
        payload = {"results": rows[:request["k"]], "routing": routing}
        if query == "ce-result":
            payload["rerank"] = {"depth": 2, "scored": 2}
            payload["results"].reverse()
        elif query == "ce-timing":
            payload["timings_ms"] = {"rerank": 0.0}
        elif query == "missing-routing":
            payload.pop("routing")
        elif query == "bad-routing":
            routing["fallback"] = "false"
        elif query == "wrong-mode":
            routing["requested_mode"] = "unexpected"
        elif query == "inconsistent-routing":
            routing["vector_used"] = True
            routing["vector_attempted"] = False
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@contextmanager
def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/search", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class EvaluatorTests(unittest.TestCase):
    def _policy_inputs(self, root, middle="unsupported"):
        queries = root / "queries.tsv"
        queries.write_text(f"q1\tq1\nq2\t{middle}\nq3\tq1\n", encoding="utf-8")
        policy = {"schema_version": 1,
                  "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
                  "preflight_parser_sha256": "a" * 64,
                  "request": {"mode": "dense", "scorer": "bm25", "expander": "none"},
                  "expected_unsupported": {"query_ids": ["q2"], "status": 400,
                                           "body": {"error": DENSE_CONSTRAINT_ERROR}}}
        path = root / "policy.json"
        path.write_text(json.dumps(policy, ensure_ascii=False), encoding="utf-8")
        return queries, path, policy

    def test_expected_unsupported_keeps_all_queries_and_durable_outcomes(self):
        with _serve() as (endpoint, server), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, policy, _ = self._policy_inputs(root)
            run = root / "dense.trec"
            qrels = {qid: {"d1": 1.0} for qid in ("q1", "q2", "q3")}
            summary = evaluate_http(endpoint, load_queries(queries), qrels, mode="dense",
                                    unsupported_policy=policy, queries_source=queries, run_output=run)
            self.assertEqual([x["query"] for x in server.requests], ["q1", "unsupported", "q1"])
            self.assertTrue(all(x["mode"] == "dense" and x["k"] == 100 for x in server.requests))
            self.assertEqual(summary.n_queries, 3)
            rejected = summary.per_query[1]
            self.assertEqual((rejected.ndcg, rejected.recall, rejected.mrr, rejected.map_at_100), (0, 0, 0, 0))
            self.assertEqual(rejected.retrieved, 0)
            self.assertIsNone(rejected.routing)
            self.assertIsNone(rejected.latency_ms)
            self.assertEqual(summary.routing.n_queries, 2)
            self.assertEqual(summary.routing.fallbacks, 0)
            self.assertAlmostEqual(summary.ndcg_at_10, 2 / 3)
            self.assertEqual(evaluate_run(qrels, load_run(run)).ndcg_at_10, summary.ndcg_at_10)
            self.assertNotIn("q2", load_run(run))
            state = json.loads(Path(str(run) + ".requests.json").read_text(encoding="utf-8"))
            self.assertEqual(state, summary.requests)
            self.assertEqual(state["status"], "COMPLETE")
            self.assertEqual((state["attempted_requests"], state["http_200"], state["expected_unsupported"]), (3, 2, 1))
            self.assertEqual(state["unexpected_errors"], 0)
            self.assertEqual(state["http_success_coverage"], 2 / 3)
            self.assertEqual(state["protocol_accounted_coverage"], 1)
            self.assertEqual(state["policy"]["sha256"], hashlib.sha256(policy.read_bytes()).hexdigest())
            self.assertIn("runtime_parser_not_verified", state["policy"]["parser_identity_scope"])
            self.assertEqual(state["outcomes"][1]["status"], "EXPECTED_UNSUPPORTED")
            self.assertIsNone(state["outcomes"][1]["routing"])
            self.assertGreaterEqual(state["outcomes"][1]["latency_ms"], 0)
            journal = run.parent / state["journal"]["file"]
            self.assertEqual(hashlib.sha256(journal.read_bytes()).hexdigest(), state["journal"]["sha256"])
            events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["START", "BEGIN", "OUTCOME", "BEGIN", "OUTCOME", "BEGIN", "OUTCOME"])
            self.assertEqual(events[0]["query_ids"], state["query_ids"])
            self.assertEqual([event["sequence"] for event in events], list(range(7)))
            self.assertEqual([event["outcome"] for event in events if event["event"] == "OUTCOME"], state["outcomes"])

    def test_default_strict_and_unexpected_errors_leave_incomplete_evidence(self):
        for middle, use_policy in (("unsupported", False), ("wrong-error", True),
                                   ("unavailable", True), ("non-json", True),
                                   ("q1", True), ("missing-routing", True)):
            with self.subTest(middle=middle, policy=use_policy), _serve() as (endpoint, server), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                queries, policy, _ = self._policy_inputs(root, middle)
                run = root / "partial.trec"
                with self.assertRaises(EvaluationError):
                    evaluate_http(endpoint, load_queries(queries), {}, mode="dense", run_output=run,
                                  unsupported_policy=policy if use_policy else None, queries_source=queries)
                state = json.loads(Path(str(run) + ".requests.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "INCOMPLETE")
                self.assertEqual(state["unexpected_errors"], 1)
                self.assertEqual(state["expected_unsupported"], 0)
                self.assertEqual(state["attempted_requests"], 2)
                self.assertNotIn("outcomes", state)
                self.assertNotIn("query_ids", state)
                self.assertEqual(state["last_outcome"]["status"], "UNEXPECTED_ERROR")
                journal = run.parent / state["journal"]["file"]
                events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(events[-1]["outcome"], state["last_outcome"])
                self.assertEqual(events[0]["query_ids"], ["q1", "q2", "q3"])
                self.assertEqual(len(server.requests), 2)
                self.assertEqual(set(load_run(run)), {"q1"})

    def test_policy_rejects_drift_before_outputs_or_http(self):
        mutations = [lambda p: p.update(extra=True),
                     lambda p: p.update(schema_version=True),
                     lambda p: p.update(queries_sha256="b" * 64),
                     lambda p: p.update(preflight_parser_sha256="bad"),
                     lambda p: p["request"].update(extra=True),
                     lambda p: p["expected_unsupported"].update(extra=True),
                     lambda p: p["expected_unsupported"].update(query_ids=[]),
                     lambda p: p["expected_unsupported"].update(query_ids=["q2", "q2"]),
                     lambda p: p["expected_unsupported"].update(query_ids=["missing"]),
                     lambda p: p["expected_unsupported"].update(status=503),
                     lambda p: p["expected_unsupported"].update(body={"error": "different"})]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutate in mutations:
                queries, policy_path, policy = self._policy_inputs(root)
                mutate(policy)
                policy_path.write_text(json.dumps(policy), encoding="utf-8")
                with patch("evaluation.evaluate._request_search") as request, self.assertRaises(EvaluationError):
                    evaluate_http("http://unused", load_queries(queries), {}, mode="dense",
                                  unsupported_policy=policy_path, queries_source=queries,
                                  run_output=root / "should-not-exist.trec")
                request.assert_not_called()
                self.assertFalse((root / "should-not-exist.trec").exists())
                self.assertFalse((root / "should-not-exist.trec.requests.json").exists())
                self.assertFalse((root / "should-not-exist.trec.requests.json.events.jsonl").exists())
            queries, policy_path, _ = self._policy_inputs(root)
            for kwargs in ({"mode": "lexical", "queries_source": queries},
                           {"mode": "dense", "scorer": "tfidf", "queries_source": queries},
                           {"mode": "dense", "expander": "rm3", "queries_source": queries},
                           {"mode": "dense"},
                           {"mode": "dense", "queries_source": queries}):
                with patch("evaluation.evaluate._request_search") as request, self.assertRaises(EvaluationError):
                    evaluate_http("http://unused", load_queries(queries), {}, unsupported_policy=policy_path, **kwargs)
                request.assert_not_called()

    def test_stale_outputs_are_rejected_and_cli_cannot_leave_stale_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, policy, _ = self._policy_inputs(root)
            run = root / "existing.trec"
            run.write_text("protected", encoding="utf-8")
            with patch("evaluation.evaluate._request_search") as request, self.assertRaisesRegex(EvaluationError, "已存在"):
                evaluate_http("http://unused", load_queries(queries), {}, run_output=run)
            request.assert_not_called()
            self.assertEqual(run.read_text(), "protected")
            qrels = root / "qrels.tsv"
            qrels.write_text("q1 0 d1 1\n", encoding="utf-8")
            output = root / "summary.json"
            output.write_text("protected summary", encoding="utf-8")
            with redirect_stderr(io.StringIO()), patch("evaluation.evaluate._request_search") as request:
                code = main(["--endpoint", "http://unused", "--queries", str(queries), "--qrels", str(qrels),
                             "--output-json", str(output), "--output-run", str(root / "new.trec")])
            self.assertEqual(code, 2)
            request.assert_not_called()
            self.assertFalse((root / "new.trec").exists())
            self.assertEqual(output.read_text(), "protected summary")

    def test_policy_rejects_duplicate_json_and_changed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, policy, _ = self._policy_inputs(root)
            policy.write_text(policy.read_text(encoding="utf-8").replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'), encoding="utf-8")
            with patch("evaluation.evaluate._request_search") as request, self.assertRaisesRegex(EvaluationError, "重复"):
                evaluate_http("http://unused", load_queries(queries), {}, mode="dense", unsupported_policy=policy,
                              queries_source=queries, request_output=root / "out.json")
            request.assert_not_called()
            queries, policy, _ = self._policy_inputs(root)
            changed = load_queries(queries)
            changed[0] = QueryRecord("q1", "changed")
            with patch("evaluation.evaluate._request_search") as request, self.assertRaisesRegex(EvaluationError, "记录不匹配"):
                evaluate_http("http://unused", changed, {}, mode="dense", unsupported_policy=policy,
                              queries_source=queries, request_output=root / "out.json")
            request.assert_not_called()
            self.assertFalse((root / "out.json").exists())

    def test_request_persistence_serialization_is_linear_for_2000_queries(self):
        totals = []
        for count in (1000, 2000):
            stats = {"bytes": 0, "snapshots": 0, "events": 0, "complete": 0}

            def snapshot_writer(path, value):
                stats["snapshots"] += 1
                stats["bytes"] += len(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
                if value["status"] == "INCOMPLETE":
                    self.assertNotIn("query_ids", value)
                    self.assertNotIn("outcomes", value)
                    self.assertLess(len(json.dumps(value)), 2500)
                else:
                    stats["complete"] += 1
                    self.assertEqual(len(value["query_ids"]), count)
                    self.assertEqual(len(value["outcomes"]), count)

            def event_writer(path, value):
                stats["events"] += 1
                stats["bytes"] += len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) + 1

            with tempfile.TemporaryDirectory() as directory, \
                    patch("evaluation.evaluate._atomic_json", side_effect=snapshot_writer), \
                    patch("evaluation.evaluate._append_request_event", side_effect=event_writer), \
                    patch("evaluation.evaluate._request_search", return_value=([], 1.0, QueryRouting("lexical", "lexical", "test", False, False, False))) as request:
                summary = evaluate_http("http://unused", [QueryRecord(f"q{i:04d}", "query") for i in range(count)], {},
                                        request_output=Path(directory) / "requests.json")
                self.assertEqual(request.call_count, count)
                self.assertEqual(summary.requests["attempted_requests"], count)
            self.assertEqual(stats["complete"], 1)
            self.assertEqual(stats["events"], 2 * count + 1)
            self.assertEqual(stats["snapshots"], 2 * count + 2)
            totals.append(stats["bytes"])
        self.assertLess(totals[1], 2.1 * totals[0])
        self.assertLess(totals[1], 12_000_000)

    def test_stale_journal_and_output_collision_are_rejected_before_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome = root / "requests.json"
            journal = Path(str(outcome) + ".events.jsonl")
            journal.write_text("protected journal\n", encoding="utf-8")
            with patch("evaluation.evaluate._request_search") as request, self.assertRaisesRegex(EvaluationError, "已存在"):
                evaluate_http("http://unused", [QueryRecord("q1", "q1")], {}, request_output=outcome)
            request.assert_not_called()
            self.assertFalse(outcome.exists())
            self.assertEqual(journal.read_text(), "protected journal\n")
            queries, policy, _ = self._policy_inputs(root)
            qrels = root / "qrels.tsv"
            qrels.write_text("q1 0 d1 1\n", encoding="utf-8")
            with patch("evaluation.evaluate._request_search") as request, redirect_stderr(io.StringIO()):
                code = main(["--endpoint", "http://unused", "--queries", str(queries), "--qrels", str(qrels),
                             "--output-requests", str(root / "new.json"),
                             "--output-json", str(root / "new.json.events.jsonl")])
            self.assertEqual(code, 2)
            request.assert_not_called()
            self.assertFalse((root / "new.json").exists())
            for collision in (root / "fresh.json.events.jsonl", root / "fresh.json.tmp"):
                with patch("evaluation.evaluate._request_search") as request, self.assertRaisesRegex(EvaluationError, "相同"):
                    evaluate_http("http://unused", [QueryRecord("q1", "query")], {},
                                  run_output=collision, request_output=root / "fresh.json")
                request.assert_not_called()
                self.assertFalse((root / "fresh.json").exists())

    def test_request_exception_is_journaled_and_interrupt_keeps_inflight(self):
        for failure in (RuntimeError("synthetic error"), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "requests.json"
                with patch("evaluation.evaluate._request_search", side_effect=failure), self.assertRaises(type(failure)):
                    evaluate_http("http://unused", [QueryRecord("q1", "query")], {}, request_output=output)
                state = json.loads(output.read_text(encoding="utf-8"))
                events = [json.loads(line) for line in Path(str(output) + ".events.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(state["status"], "INCOMPLETE")
                self.assertEqual(state["http_200"], 0)
                if isinstance(failure, KeyboardInterrupt):
                    self.assertEqual(events[-1]["event"], "BEGIN")
                    self.assertEqual(state["inflight_outcome"]["qid"], "q1")
                    self.assertIsNone(state["inflight_outcome"]["http_status"])
                else:
                    self.assertEqual(events[-1]["event"], "OUTCOME")
                    self.assertEqual(events[-1]["outcome"]["error_type"], "RuntimeError")
                    self.assertEqual(state["unexpected_errors"], 1)

    def test_cli_policy_and_unknown_qid_rejection_use_real_http(self):
        with _serve() as (endpoint, _), tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queries, policy_path, policy = self._policy_inputs(root)
            qrels = root / "qrels.tsv"
            qrels.write_text("q1 0 d1 1\nq2 0 d1 1\nq3 0 d1 1\n", encoding="utf-8")
            output = root / "summary.json"
            with patch("sys.stdout", new=io.StringIO()):
                code = main(["--endpoint", endpoint, "--queries", str(queries), "--qrels", str(qrels),
                             "--mode", "dense", "--unsupported-policy", str(policy_path),
                             "--output-requests", str(root / "requests.json"), "--output-json", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["requests"]["status"], "COMPLETE")
            policy["expected_unsupported"]["query_ids"] = ["q3"]
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                evaluate_http(endpoint, load_queries(queries), {}, mode="dense", unsupported_policy=policy_path,
                              queries_source=queries, request_output=root / "unknown-qid.json")
            state = json.loads((root / "unknown-qid.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "INCOMPLETE")
            self.assertEqual(state["expected_unsupported"], 0)

    def test_metrics_and_trec_loaders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qrels_path = root / "qrels.tsv"
            qrels_path.write_text("q1 0 d1 2\nq1 0 d9 1\nq2 0 d3 1\n")
            run_path = root / "run.trec"
            run_path.write_text("q1 Q0 d2 1 3.0 tag\nq1 Q0 d1 2 2.0 tag\nq2 Q0 d3 1 1.0 tag\n")
            query_path = root / "queries.tsv"
            query_path.write_text("q1\tquery one\nq2\tquery two\n")
            qrels = load_qrels(qrels_path)
            summary = evaluate_run(qrels, load_run(run_path), k=2)
            self.assertEqual(summary.n_queries, 2)
            self.assertGreater(summary.ndcg_at_10, 0.0)
            self.assertGreater(summary.map_at_100, 0.0)
            self.assertEqual([q.qid for q in load_queries(query_path)], ["q1", "q2"])
            self.assertIsNone(summary.routing)
            self.assertTrue(all(item.routing is None for item in summary.per_query))

    def test_http_mode_and_run_output(self):
        with _serve() as (endpoint, _), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            summary = evaluate_http(
                endpoint, [QueryRecord("q1", "q1"), QueryRecord("q2", "q2")],
                {"q1": {"d1": 1.0}, "q2": {"d3": 1.0}}, run_output=path,
            )
            self.assertEqual(summary.n_queries, 2)
            self.assertIsNotNone(summary.p95_latency_ms)
            self.assertEqual(len(path.read_text().splitlines()), 3)
            self.assertEqual(summary.routing.vector_attempts, 0)
            self.assertIsNone(summary.routing.vector_success_rate)

    def test_hybrid_baseline_rejects_ce_response_and_preserves_incomplete_run(self):
        for marker in ("ce-result", "ce-timing"):
            with self.subTest(marker=marker), _serve() as (endpoint, server), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "baseline.trec"
                queries = [QueryRecord("plain", "q1"), QueryRecord("ce", marker)]
                with self.assertRaisesRegex(EvaluationError, r"evaluation\.ce_http"):
                    evaluate_http(endpoint, queries, {}, mode="hybrid", run_output=path)
                self.assertEqual(len(server.requests), 2)
                self.assertEqual(set(load_run(path)), {"plain"})
                state = json.loads(Path(str(path) + ".requests.json").read_text(encoding="utf-8"))
                self.assertEqual(state["status"], "INCOMPLETE")
                self.assertEqual(state["unexpected_errors"], 1)

    def test_http_fetches_map_depth_without_changing_top_ten_metrics(self):
        with _serve() as (endpoint, server), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            summary = evaluate_http(
                endpoint, [QueryRecord("q1", "deep")], {"q1": {"d11": 1.0}},
                run_output=path,
            )
            self.assertEqual(server.requests[0]["k"], 100)
            self.assertEqual(len(load_run(path)["q1"]), 100)
            self.assertEqual(summary.per_query[0].retrieved, 100)
            self.assertEqual(summary.ndcg_at_10, 0.0)
            self.assertEqual(summary.recall_at_10, 0.0)
            self.assertEqual(summary.mrr_at_10, 0.0)
            self.assertAlmostEqual(summary.map_at_100, 1.0 / 11)
            reloaded = evaluate_run({"q1": {"d11": 1.0}}, load_run(path))
            self.assertEqual(reloaded.map_at_100, summary.map_at_100)

    def test_http_request_tag_binds_payload_and_journal_independently_of_trec_tag(self):
        with _serve() as (endpoint, server), tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run.trec"
            summary = evaluate_http(
                endpoint, [QueryRecord("q1", "q1")], {"q1": {"d1": 1.0}},
                run_output=run, tag="ranking-arm", request_tag="experiment-attempt.1",
            )
            self.assertEqual(server.requests[0]["run_tag"], "experiment-attempt.1")
            self.assertEqual(summary.requests["request"]["run_tag"], "experiment-attempt.1")
            sidecar = Path(str(run) + ".requests.json")
            journal = Path(str(sidecar) + ".events.jsonl")
            complete = json.loads(sidecar.read_text(encoding="utf-8"))
            start = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(start["request"], complete["request"])
            self.assertEqual(complete["request"]["run_tag"], "experiment-attempt.1")
            self.assertTrue(all(line.split()[-1] == "ranking-arm"
                                for line in run.read_text(encoding="utf-8").splitlines()))

    def test_invalid_http_request_tags_leave_no_requests_or_outputs(self):
        with _serve() as (endpoint, server), tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run.trec"
            for tag in ("", "with space", "x" * 129, "x\nnext", "中文"):
                with self.subTest(tag=tag), self.assertRaisesRegex(EvaluationError, "request tag"):
                    evaluate_http(endpoint, [QueryRecord("q1", "q1")], {},
                                  run_output=run, request_tag=tag)
            self.assertEqual(server.requests, [])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_http_fetches_larger_ranking_cutoff_independently_of_map_cutoff(self):
        with _serve() as (endpoint, server):
            summary = evaluate_http(
                endpoint, [QueryRecord("q1", "deep")], {"q1": {"d11": 1.0}},
                k=15, map_k=10,
            )
            self.assertEqual(server.requests[0]["k"], 15)
            self.assertEqual(summary.recall_at_10, 1.0)
            self.assertEqual(summary.map_at_100, 0.0)

    def test_http_rejects_unsupported_depth_before_touching_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            path.write_text("previous run", encoding="utf-8")
            with patch("evaluation.evaluate._request_search") as request:
                for k, map_k in ((201, 100), (10, 201)):
                    with self.assertRaisesRegex(EvaluationError, "200"):
                        evaluate_http("http://unused", [QueryRecord("q1", "q1")], {},
                                      k=k, map_k=map_k, run_output=path)
                request.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "previous run")

    def test_http_reports_success_skip_and_fallback_per_query(self):
        with _serve() as (endpoint, _):
            summary = evaluate_http(
                endpoint, [QueryRecord("success", "q1"), QueryRecord("skip", "skip"),
                           QueryRecord("failed", "fallback")],
                {}, mode="adaptive",
            )
            self.assertEqual(summary.routing.n_queries, 3)
            self.assertEqual(summary.routing.vector_attempts, 2)
            self.assertEqual(summary.routing.vector_successes, 1)
            self.assertEqual(summary.routing.fallbacks, 1)
            self.assertAlmostEqual(summary.routing.vector_attempt_rate, 2.0 / 3)
            self.assertEqual(summary.routing.vector_success_rate, 0.5)
            self.assertAlmostEqual(summary.routing.fallback_rate, 1.0 / 3)
            value = summary.to_dict(include_per_query=True)
            self.assertTrue(value["per_query"][2]["routing"]["fallback"])
            self.assertEqual(value["routing"]["fallbacks"], 1)

    def test_http_validates_routing_instead_of_silently_losing_telemetry(self):
        with _serve() as (endpoint, _):
            for query in ("missing-routing", "bad-routing", "wrong-mode", "inconsistent-routing"):
                with self.subTest(query=query), self.assertRaisesRegex(EvaluationError, "routing"):
                    evaluate_http(endpoint, [QueryRecord("q1", query)], {}, mode="adaptive")
            for mode in ("lexical", "dense", "hybrid", "adaptive"):
                with self.subTest(mode=mode):
                    summary = evaluate_http(endpoint, [QueryRecord("q1", "q1")], {}, mode=mode)
                    self.assertEqual(summary.per_query[0].routing.requested_mode, mode)

    def test_strict_fallback_rejection_is_exposed_by_cli(self):
        with _serve() as (endpoint, _), tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(EvaluationError, "q1.*降级"):
                evaluate_http(endpoint, [QueryRecord("q1", "fallback")], {},
                              mode="adaptive", fail_on_fallback=True)
            root = Path(directory)
            (root / "queries.tsv").write_text("q1\tfallback\n", encoding="utf-8")
            (root / "qrels.tsv").write_text("q1 0 d1 1\n", encoding="utf-8")
            errors = io.StringIO()
            with redirect_stderr(errors):
                code = main(["--endpoint", endpoint, "--queries", str(root / "queries.tsv"),
                             "--qrels", str(root / "qrels.tsv"), "--mode", "adaptive",
                             "--fail-on-fallback"])
            self.assertEqual(code, 2)
            self.assertIn("降级", errors.getvalue())

    def test_map_and_paired_bootstrap_are_reproducible(self):
        qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        baseline = {"q1": [("d9", 2.0), ("d1", 1.0)], "q2": [("d2", 1.0)]}
        improved = {"q1": [("d1", 2.0)], "q2": [("d2", 1.0)]}
        first = paired_bootstrap(qrels, baseline, improved, samples=200, seed=7)
        second = paired_bootstrap(qrels, baseline, improved, samples=200, seed=7)
        self.assertEqual(first, second)
        self.assertGreater(first.observed_difference, 0.0)
        self.assertLessEqual(first.ci_lower, first.observed_difference)
        self.assertGreaterEqual(first.ci_upper, first.observed_difference)


if __name__ == "__main__":
    unittest.main()
