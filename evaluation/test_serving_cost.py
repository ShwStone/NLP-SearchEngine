"""Regression checks for binding evaluator receipts to server observations."""

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.evaluate import (
    DENSE_CONSTRAINT_ERROR,
    HTTPResponseError,
    QueryRouting,
    evaluate_http,
    load_queries,
)
from evaluation.serving_cost import main, summarize_run
from online.observability import SearchLogger


class ServingCostTests(unittest.TestCase):
    TAG = "dense-fixture-01"
    CORPUS = "a" * 64

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sidecar = self.root / "run.requests.json"
        self.journal = self.root / "run.requests.json.events.jsonl"
        self.log = self.root / "server.log"
        self.startup = {
            "event": "startup", "dataset": "fixture", "policy_version": "rules-v1",
            "index_identity": {"corpus_sha256": self.CORPUS},
        }
        queries = self.root / "queries.tsv"
        queries.write_text("q1\tfirst\nq2\tunsupported\nq3\tthird\n", encoding="utf-8")
        policy = self.root / "policy.json"
        rejection = {"error": DENSE_CONSTRAINT_ERROR}
        policy.write_text(json.dumps({
            "schema_version": 1,
            "queries_sha256": hashlib.sha256(queries.read_bytes()).hexdigest(),
            "preflight_parser_sha256": "b" * 64,
            "request": {"mode": "dense", "scorer": "bm25", "expander": "none"},
            "expected_unsupported": {"query_ids": ["q2"], "status": 400, "body": rejection},
        }), encoding="utf-8")
        routing = QueryRouting("dense", "dense", "forced_dense", True, True, False)
        transport = [
            ([("d", 1.0)], 20.0, routing),
            HTTPResponseError(400, rejection, 2.0),
            ([("d", 1.0)], 40.0, routing),
        ]
        with patch("evaluation.evaluate._request_search", side_effect=transport):
            evaluate_http("http://unused", load_queries(queries), {"q1": {"d": 1}},
                          mode="dense", request_tag=self.TAG, request_output=self.sidecar,
                          unsupported_policy=policy, queries_source=queries)
        with self.log.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.startup) + "\nServer startup notice\n")
            logger = SearchLogger(handle)
            identity = {"dataset": "fixture", "policy_version": "rules-v1",
                        "corpus_sha256": self.CORPUS}
            for tag, encode in (("", 99.0), (self.TAG + "-other", 99.0), (self.TAG, 4.0),
                                (self.TAG, None), (self.TAG, 12.0)):
                if encode is None:
                    logger.search(status=400, run_tag=tag, requested_mode="dense",
                                  reason="invalid_request", http_elapsed_ms=1.0,
                                  serialization_ms=0.1, **identity)
                    continue
                timings = {"encode": encode, "vector": 2.0, "lexical": 0.0,
                           "router": 0.0, "fusion": 0.0}
                logger.search(status=200, run_tag=tag, took_ms=15, timings_ms=timings,
                              http_elapsed_ms=17.0, serialization_ms=0.1,
                              **identity, **asdict(routing))

    def write_log(self, events):
        self.log.write_text("".join(json.dumps(event) + "\n" for event in events),
                            encoding="utf-8")

    def logged_events(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()
                if line.startswith("{")]

    def test_cost_coverage_and_cli(self):
        with redirect_stdout(io.StringIO()) as output:
            result_code = main(["--run", str(self.sidecar), str(self.log)])
        report = json.loads(output.getvalue())
        self.assertEqual(result_code, 0)
        self.assertEqual(report["observed_log_startup_epochs"], 1)
        run, = report["runs"]
        self.assertEqual(run["excluded_server_searches"], {"untagged": 1, "other_tags": 1})
        coverage = run["coverage"]
        self.assertEqual(coverage["requests"], 3)
        self.assertEqual(coverage["matched_server_events"], 3)
        self.assertEqual(coverage["http_success_coverage"], 2 / 3)
        self.assertEqual(coverage["protocol_accounted_coverage"], 1.0)
        self.assertEqual(coverage["dense_explicit_constraint_http_400"], 1)
        self.assertEqual(coverage["fallback_rate_of_http_200"], 0.0)
        success = run["validated_http_200_cost_ms"]
        self.assertEqual(success["server"]["encode"]["p50"], 8.0)
        self.assertEqual(success["client_latency_ms"]["p50"], 30.0)
        rejected = run["expected_dense_http_400_cost_ms"]
        self.assertEqual(rejected["server"]["http_elapsed_ms"]["p95"], 1.0)
        self.assertEqual(rejected["client_latency_ms"]["p95"], 2.0)
        self.assertNotIn("encode", rejected["server"])

    def test_hybrid_cost_rejects_a_ce_epoch(self):
        queries = self.root / "hybrid.tsv"
        queries.write_text("q1\tfirst\n", encoding="utf-8")
        sidecar = self.root / "hybrid.requests.json"
        routing = QueryRouting("hybrid", "hybrid", "forced_hybrid", True, True, False)
        with patch("evaluation.evaluate._request_search", return_value=([("d", 1.0)], 20.0, routing)):
            evaluate_http("http://unused", load_queries(queries), {"q1": {"d": 1}},
                          mode="hybrid", request_tag=self.TAG, request_output=sidecar)
        row = self.logged_events()[3]
        row.update(requested_mode="hybrid", executed_mode="hybrid", reason="forced_hybrid")
        self.write_log([self.startup, row])
        self.assertEqual(summarize_run(sidecar, self.log)["requested_mode"], "hybrid")
        for startup, search in (
            ({**self.startup, "reranker": {"depth": 40}}, row),
            (self.startup, {**row, "timings_ms": {**row["timings_ms"], "rerank": 0.8}}),
        ):
            self.write_log([startup, search])
            with self.assertRaisesRegex(ValueError, "CE-enabled hybrid run"):
                summarize_run(sidecar, self.log)

    def test_exact_tag_and_epoch_are_required(self):
        events = self.logged_events()
        for event in events[3:]:
            event["run_tag"] = self.TAG + "-suffix"
        self.write_log(events)
        with self.assertRaisesRegex(ValueError, "exact HTTP request tag"):
            summarize_run(self.sidecar, self.log)
        for event in events[3:]:
            event["run_tag"] = self.TAG
        self.write_log([*events[:-1], self.startup, events[-1]])
        with self.assertRaisesRegex(ValueError, "multiple server startups"):
            summarize_run(self.sidecar, self.log)

    def test_disagreement_and_invalid_measurement_are_rejected(self):
        events = self.logged_events()
        for patch_fields, error in (
            ({"reason": "different_route"}, "routing disagreement"),
            ({"http_elapsed_ms": float("inf")}, "measured http_elapsed_ms"),
            ({"status": 503}, "HTTP status disagreement"),
        ):
            with self.subTest(patch_fields=patch_fields):
                changed = {**events[-1], **patch_fields}
                self.write_log([*events[:-1], changed])
                with self.assertRaisesRegex(ValueError, error):
                    summarize_run(self.sidecar, self.log)

    def test_journal_digest_and_start_binding_are_required(self):
        raw = self.journal.read_bytes()
        self.journal.write_bytes(raw + b"{}\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            summarize_run(self.sidecar, self.log)
        first, remaining = raw.split(b"\n", 1)
        start = json.loads(first)
        start["request"]["run_tag"] += "-changed"
        tampered = json.dumps(start).encode("utf-8") + b"\n" + remaining
        self.journal.write_bytes(tampered)
        sidecar = json.loads(self.sidecar.read_bytes())
        sidecar["journal"]["sha256"] = hashlib.sha256(tampered).hexdigest()
        self.sidecar.write_text(json.dumps(sidecar), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Journal START differs"):
            summarize_run(self.sidecar, self.log)


if __name__ == "__main__":
    unittest.main()
