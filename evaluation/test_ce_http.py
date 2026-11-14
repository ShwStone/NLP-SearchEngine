"""Paired CE HTTP capture and live random controls support offline replay."""

from io import StringIO
import json
from pathlib import Path
import sqlite3
from contextlib import closing, redirect_stderr
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation import ce_http


class FixedCEResumeTests(unittest.TestCase):
    def test_final_identity_check_precedes_last_commit_and_offline_replay(self):
        identity = {"common": {"retrieval": {"fixture": "frozen"}}}

        def capture(client, arms, record, index, order, retrieval):
            return {
                name: {"ranking": [("doc", 0.5)], "ce_pairs": int(depth > 0),
                       "timings_ms": {}, "http_ms": 1.0, "explicit_constraints": False}
                for name, (depth, _) in arms.items()
            }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.tsv").write_text("q1\tone\nq2\ttwo\n", encoding="utf-8")
            (root / "warmup.tsv").write_text("w1\twarmup\n", encoding="utf-8")
            (root / "qrels.tsv").write_text("q1 0 doc 1\nq2 0 doc 1\n", encoding="utf-8")
            out = root / "output"
            args = [
                "--baseline", "http://127.0.0.1:18000",
                "--rerank", "28", "http://127.0.0.1:18028",
                "--queries", str(root / "queries.tsv"),
                "--warmup-queries", str(root / "warmup.tsv"),
                "--qrels", str(root / "qrels.tsv"), "--out", str(out),
            ]
            with (patch.object(ce_http, "check_services", side_effect=[identity, OSError("stopped")]),
                  patch.object(ce_http, "capture_query", side_effect=capture)):
                with self.assertRaisesRegex(OSError, "stopped"):
                    ce_http.main(args)
            self.assertFalse((out / "summary.json").exists())
            with closing(sqlite3.connect(out / "capture.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT qid FROM queries").fetchall(), [("q1",)])
            with (patch.object(ce_http, "check_services", return_value=identity),
                  patch.object(ce_http, "capture_query", side_effect=capture) as requests):
                ce_http.main(args)
            self.assertEqual([call.args[2].qid for call in requests.call_args_list], ["w1", "q2"])
            expected = (out / "summary.json").read_bytes()
            with (patch.object(ce_http, "check_services", side_effect=AssertionError("network used")),
                  patch.object(ce_http, "capture_query", side_effect=AssertionError("recaptured"))):
                ce_http.main(args)
            self.assertEqual((out / "summary.json").read_bytes(), expected)
            replay = [
                "--replay-from", str(out), "--out", str(root / "replay"),
                *args[args.index("--queries"):args.index("--out")],
            ]
            capture_path = out / "capture.sqlite3"
            capture_bytes = capture_path.read_bytes()
            with patch.object(ce_http, "http_client", side_effect=AssertionError("network used")):
                ce_http.main(replay)
                self.assertEqual(capture_path.read_bytes(), capture_bytes)
                with closing(sqlite3.connect(capture_path)) as db, db:
                    db.execute("DELETE FROM queries WHERE qid='q2'")
                with self.assertRaisesRegex(ce_http.EvaluationError, "journal is incomplete"):
                    ce_http.main([*replay, "--out", str(root / "incomplete")])
            self.assertEqual((root / "replay" / "summary.json").read_bytes(), expected)
            self.assertFalse((root / "replay" / "capture.sqlite3").exists())

    def test_live_random_checks_responses_resumes_pairs_and_republishes_offline(self):
        identity = {"common": {"retrieval": {"fixture": "frozen"}}, "rerankers": {
            "rrf": None, "ce28": {"depth": 28}, "ce40": {"depth": 40},
            "learned": {"depth": 40, "gate_sha256": "test-gate"},
        }}
        live_identity = {**identity, "rerankers": {
            name: identity["rerankers"][name] for name in ("rrf", "ce40", "learned")
        }}
        endpoints = {name: f"http://127.0.0.1:{port}" for name, port in
                     (("rrf", 18000), ("ce28", 18028), ("ce40", 18040), ("learned", 18041))}

        def row(qid, name):
            active = name == "ce40" or (name == "learned" and qid in ("q1", "q3"))
            return {"ranking": [("a", 0.5), ("b", 0.25)] if not active else
                    [("b", 0.25), ("a", 0.5)], "ce_pairs": 2 if active else 0,
                    "timings_ms": {}, "http_ms": {"rrf": 1.0, "ce28": 2.0, "ce40": 4.0,
                                                  "learned": 3.0}[name],
                    "explicit_constraints": False, "features": None}

        def capture(client, arms, record, index, order, identities):
            return {name: row(record.qid, name) for name in arms}

        def live_search(client, base, depth, record, retrieval, reranker):
            name = next(name for name, endpoint in endpoints.items() if endpoint == base)
            return row(record.qid, name)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queries.tsv").write_text("q1\tone\nq2\ttwo\nq3\tthree\nq4\tfour\n", encoding="utf-8")
            (root / "warmup.tsv").write_text("w1\twarmup\n", encoding="utf-8")
            (root / "qrels.tsv").write_text("".join(f"q{i} 0 b 1\n" for i in range(1, 5)),
                                           encoding="utf-8")
            out = root / "output"
            args = [
                "--baseline", endpoints["rrf"], "--rerank", "28", endpoints["ce28"],
                "--rerank", "40", endpoints["ce40"], "--learned", endpoints["learned"],
                "--queries", str(root / "queries.tsv"),
                "--warmup-queries", str(root / "warmup.tsv"),
                "--qrels", str(root / "qrels.tsv"), "--out", str(out),
            ]
            with (patch.object(ce_http, "check_services", return_value=identity),
                  patch.object(ce_http, "capture_query", side_effect=capture)):
                ce_http.main(args)
            original = (out / "summary.json").read_bytes()
            diagnostic_bytes = (out / "learned-diagnostics.json").read_bytes()
            diagnostics = json.loads(diagnostic_bytes)
            self.assertEqual(diagnostics["learned_actions_by_ce40_outcome"]["skipped_win"], 2)
            self.assertEqual(set(diagnostics["learned_minus_fixed"]), {"ce28", "ce40"})
            random_run = (out / "matched-random-replay.trec").read_text(encoding="utf-8")
            first_docs = {qid: doc for qid, _, doc, rank, _, _ in map(str.split, random_run.splitlines())
                          if rank == "1"}
            self.assertEqual(first_docs, {"q1": "b", "q2": "b", "q3": "a", "q4": "a"})
            live_path = out / "matched-random-live.json"
            self.assertFalse(live_path.exists())
            args.append("--measure-random")

            for field, wrong_value in (("ce_pairs", 0), ("ranking", row("q2", "rrf")["ranking"])):
                def drift(*call):
                    observed = live_search(*call)
                    if call[3].qid == "q2" and call[1] == endpoints["ce40"]:
                        observed[field] = wrong_value
                    return observed

                with (self.subTest(drift=field),
                      patch.object(ce_http, "check_services", return_value=live_identity),
                      patch.object(ce_http, "capture_query", side_effect=capture),
                      patch.object(ce_http, "search", side_effect=drift)):
                    with self.assertRaisesRegex(ce_http.EvaluationError, "q2.*ce40.*frozen capture"):
                        ce_http.main(args)
                self.assertFalse(live_path.exists())
                with closing(sqlite3.connect(out / "matched-random-live.sqlite3")) as db:
                    self.assertEqual(db.execute("SELECT qid FROM queries").fetchall(), [("q1",)])

            # The final identity probe must succeed before committing the final pair.
            with (patch.object(ce_http, "check_services",
                               side_effect=[live_identity, OSError("stopped")]) as check,
                  patch.object(ce_http, "capture_query", side_effect=capture) as warmups,
                  patch.object(ce_http, "search", side_effect=live_search) as requests):
                with self.assertRaisesRegex(OSError, "stopped"):
                    ce_http.main(args)
            self.assertFalse(live_path.exists())
            self.assertEqual([call.args[2].qid for call in warmups.call_args_list], ["w1"])
            self.assertEqual(set(check.call_args.args[1]), {"rrf", "ce40", "learned"})
            for i, (qid, control) in enumerate((("q2", "ce40"), ("q3", "rrf"), ("q4", "rrf"))):
                pair = requests.call_args_list[2*i:2*i + 2]
                self.assertEqual([call.args[3].qid for call in pair], [qid, qid])
                self.assertEqual({call.args[1] for call in pair},
                                 {endpoints["learned"], endpoints[control]})
            with closing(sqlite3.connect(out / "matched-random-live.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT qid FROM queries ORDER BY rowid").fetchall(),
                                 [("q1",), ("q2",), ("q3",)])

            # A launcher may pick new loopback ports after restarting the services.
            for i, (name, old_url) in enumerate(endpoints.items()):
                endpoints[name] = f"http://127.0.0.1:{19000 + i}"
                args[args.index(old_url)] = endpoints[name]
            with (patch.object(ce_http, "check_services", return_value=live_identity),
                  patch.object(ce_http, "capture_query", side_effect=capture),
                  patch.object(ce_http, "search", side_effect=live_search) as requests):
                ce_http.main(args)
            self.assertEqual([call.args[3].qid for call in requests.call_args_list], ["q4", "q4"])
            live_bytes = live_path.read_bytes()
            live = json.loads(live_bytes)
            self.assertEqual(live["ce_pairs"], {"learned": 4, "random": 4})
            self.assertEqual(live["queries_by_ce_pair_count"], {
                "learned": {"0": 2, "2": 2}, "random": {"0": 2, "2": 2},
            })
            self.assertEqual(sorted(row["n_queries"] for row in live["latency_by_epoch"].values()),
                             [1, 1, 2])
            self.assertTrue(all(not row["covers_all_queries"] for row in
                                live["latency_by_epoch"].values()))
            final_epoch = next(row for row in live["latency_by_epoch"].values()
                               if row["ce_pairs"] == {"learned": 0, "random": 0})
            self.assertEqual(final_epoch["paired_learned_minus_random_http_ms"],
                             {"mean": 2.0, "p50": 2.0, "p95": 2.0})

            live_path.unlink()
            with (patch.object(ce_http, "check_services", side_effect=AssertionError("network used")),
                  patch.object(ce_http, "capture_query", side_effect=AssertionError("recaptured")),
                  patch.object(ce_http, "search", side_effect=AssertionError("remeasured"))):
                ce_http.main(args)
            self.assertEqual(live_path.read_bytes(), live_bytes)
            self.assertEqual((out / "summary.json").read_bytes(), original)
            self.assertEqual((out / "learned-diagnostics.json").read_bytes(), diagnostic_bytes)
            replay_out = root / "replay"
            replay = [
                "--replay-from", str(out), "--out", str(replay_out),
                *args[args.index("--queries"):args.index("--out")],
            ]
            journals = {path.name: path.read_bytes() for path in out.glob("*.sqlite3")}
            with patch.object(ce_http, "http_client", side_effect=AssertionError("network used")):
                ce_http.main(replay)
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    ce_http.main([*replay, "--out", str(out / "nested")])
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    ce_http.main(replay)
            self.assertEqual({path.name: path.read_bytes() for path in out.glob("*.sqlite3")}, journals)
            self.assertEqual(
                {path.name: path.read_bytes() for path in replay_out.iterdir()},
                {path.name: path.read_bytes() for path in out.iterdir() if path.suffix != ".sqlite3"},
            )

            random_journal = out / "matched-random-live.sqlite3"
            with closing(sqlite3.connect(random_journal)) as db:
                db.execute("PRAGMA journal_mode=WAL")
                with db:
                    db.execute("DELETE FROM queries WHERE qid='q4'")
                with self.assertRaisesRegex(ce_http.EvaluationError, "uncheckpointed WAL"):
                    ce_http.main([*replay, "--out", str(root / "wal")])
            with (patch.object(ce_http, "http_client", side_effect=AssertionError("network used")),
                  self.assertRaisesRegex(ce_http.EvaluationError, "live random journal is incomplete")):
                ce_http.main([*replay, "--out", str(root / "incomplete")])
            self.assertFalse((root / "incomplete" / "summary.json").exists())
            random_journal.unlink()
            with (patch.object(ce_http, "http_client", side_effect=AssertionError("network used")),
                  self.assertRaisesRegex(ce_http.EvaluationError, "live random journal is missing")):
                ce_http.main([*replay, "--out", str(root / "missing"), "--measure-random"])
            self.assertFalse((root / "missing" / "summary.json").exists())
