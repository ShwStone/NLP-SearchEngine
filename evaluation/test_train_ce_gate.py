"""Train and load a small gate from the evaluator's durable capture schema."""

from contextlib import closing
import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from contracts.ce import gate_binding
from evaluation import ce_http, train_ce_gate
from evaluation.evaluate import EvaluationError
from online.ce_gate import load_gate


class GateTrainingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional")
    def test_complete_training_only_capture_exports_bound_gate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queries, qrels, capture = (root / name for name in ("train.tsv", "qrels.tsv", "capture.sqlite3"))
            queries.write_text("positive\tone\nnegative\ttwo\n", encoding="utf-8")
            qrels.write_text("positive 0 b 1\nnegative 0 a 1\n", encoding="utf-8")
            identity = {"index_identity": {"dataset": "test"}, "vector_identity": {"ef": 128},
                        "search_config": {"candidate_k": 200}}
            model = {key: "fixture" for key in ("model_id", "revision", "weights_sha256", "inventory_sha256")}
            model["device"] = "cpu"
            protocol = {"version": ce_http.CAPTURE_VERSION, "request": ce_http.REQUEST,
                        "query_count": 2, "depths": {"rrf": 0, "ce40": 40},
                        "inputs_sha256": {"queries": ce_http.sha256(queries), "qrels": ce_http.sha256(qrels)},
                        "identities": {"common": {"retrieval": identity},
                                       "rerankers": {"rrf": None, "ce40": {
                                           "depth": 40, "candidate_source": "hybrid_rrf",
                                           "capture_features": True, "model": model}}}}
            output, receipt = root / "gate.json", root / "receipt.json"
            with closing(sqlite3.connect(capture)) as db:
                ce_http.journal(db, protocol)
                for index, qid in enumerate(("positive", "negative")):
                    base = [["a", .2], ["b", .1]]
                    row = {"rrf": {"ranking": base, "explicit_constraints": False, "ce_pairs": 0},
                           "ce40": {"ranking": list(reversed(base)), "explicit_constraints": False,
                                    "ce_pairs": 2, "features": [index + 1., 1., .5, 2.]}}
                    with db:
                        db.execute("INSERT INTO queries VALUES (?, ?, ?)", (qid, "epoch", json.dumps(row)))
            result = train_ce_gate.train(capture, queries, qrels, output, receipt)
            self.assertEqual((result["counts"]["positive"], result["counts"]["losses"]), (1, 1))
            gate = load_gate(output, result["model_sha256"])
            gate.check_binding(**{
                "retrieval_identity": identity,
                "model_identity": gate_binding(identity, model)["model"],
            })
            self.assertTrue(gate.use_ce([1., 1., .5, 2.]))
            self.assertFalse(gate.use_ce([2., 1., .5, 2.]))
            with closing(sqlite3.connect(capture)) as db, db:
                protocol["version"] -= 1
                db.execute("UPDATE protocol SET body=?", (json.dumps(protocol),))
            with self.assertRaisesRegex(EvaluationError, "capture protocol"):
                train_ce_gate.train(capture, queries, qrels, root / "other.json", root / "other-receipt.json")
