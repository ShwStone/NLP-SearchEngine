"""Train a fixed CE40 gate from a complete DuReader train HTTP capture."""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
import warnings

from contracts.ce import GATE_DEPTH, GATE_FEATURE_NAMES, gate_binding
from .ce_http import CAPTURE_VERSION, REQUEST, sha256
from .evaluate import EvaluationError, _metric_for_query, load_qrels, load_queries


FIT = {"C": .5, "max_iter": 1000, "class_weight": "balanced", "random_state": 2026}
CUTOFF = 10


def _valid_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _ranking(arm, qid):
    if not isinstance(arm, dict) or not isinstance(arm.get("ranking"), list):
        raise EvaluationError(f"{qid}: missing ranking")
    ranking = arm["ranking"]
    if len(ranking) > REQUEST["k"] or any(
        not isinstance(item, list) or len(item) != 2
        or not isinstance(item[0], str) or not item[0]
        or any(char.isspace() for char in item[0]) or not _valid_number(item[1])
        for item in ranking
    ) or len({item[0] for item in ranking}) != len(ranking):
        raise EvaluationError(f"{qid}: invalid ranking")
    return ranking


def _protocol_identity(protocol, hashes, count):
    try:
        identities = protocol["identities"]
        retrieval = identities["common"]["retrieval"]
        rerankers = identities["rerankers"]
        reranker = rerankers["ce40"]
        model = reranker["model"]
        binding = gate_binding(retrieval, model)
        valid = (
            protocol["version"] == CAPTURE_VERSION and protocol["request"] == REQUEST
            and type(protocol["query_count"]) is int and protocol["query_count"] == count
            and all(protocol["inputs_sha256"][key] == value for key, value in hashes.items())
            and protocol["depths"].keys() == rerankers.keys() == {"rrf", "ce40"}
            and type(protocol["depths"]["rrf"]) is int and protocol["depths"]["rrf"] == 0
            and type(protocol["depths"]["ce40"]) is int and protocol["depths"]["ce40"] == GATE_DEPTH
            and rerankers["rrf"] is None
            and type(reranker["depth"]) is int and reranker["depth"] == GATE_DEPTH
            and reranker["candidate_source"] == "hybrid_rrf"
            and reranker["capture_features"] is True and reranker.get("gate_sha256") is None
            and all(isinstance(model.get(key), str) and model[key] for key in
                    ("model_id", "revision", "weights_sha256", "inventory_sha256"))
            and all(isinstance(value, dict) and value for value in binding["retrieval"].values())
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise EvaluationError("training capture has an invalid protocol") from error
    if not valid:
        raise EvaluationError("training capture protocol does not match fixed RRF/CE40 train inputs")
    return binding


def _read_capture(path, hashes, query_ids):
    wal = path.with_name(path.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise EvaluationError("training capture has an uncheckpointed SQLite WAL")
    digest = sha256(path)
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        if db.execute("SELECT 1 FROM invalid_epochs LIMIT 1").fetchone() is not None:
            raise EvaluationError("training capture contains an invalid service epoch")
        protocol_rows = db.execute("SELECT id, body FROM protocol").fetchall()
        if len(protocol_rows) != 1 or protocol_rows[0][0] != 1:
            raise EvaluationError("training capture must contain one protocol")
        protocol = json.loads(protocol_rows[0][1])
        binding = _protocol_identity(protocol, hashes, len(query_ids))
        rows = db.execute("SELECT qid, epoch, arms FROM queries").fetchall()
    if len(rows) != len(query_ids) or {qid for qid, _, _ in rows} != set(query_ids):
        raise EvaluationError("capture query IDs and count must match the train queries exactly")
    if (wal.exists() and wal.stat().st_size) or sha256(path) != digest:
        raise EvaluationError("training capture changed while reading")
    if any(not isinstance(epoch, str) or not epoch for _, epoch, _ in rows):
        raise EvaluationError("training capture has a missing service epoch")
    return protocol, binding, digest, {qid: json.loads(arms) for qid, _, arms in rows}


def _training_rows(query_ids, qrels, captured, arm_names):
    matrix, labels = [], []
    counts = Counter()
    for qid in query_ids:
        arms = captured[qid]
        if not isinstance(arms, dict) or arms.keys() != arm_names:
            raise EvaluationError(f"{qid}: training capture is missing paired arms")
        rrf, ce = arms["rrf"], arms["ce40"]
        original = _ranking(rrf, qid)
        reranked = _ranking(ce, qid)
        explicit = rrf.get("explicit_constraints")
        if (type(explicit) is not bool or ce.get("explicit_constraints") is not explicit
                or type(rrf.get("ce_pairs")) is not int or rrf["ce_pairs"] != 0):
            raise EvaluationError(f"{qid}: invalid RRF/CE40 eligibility evidence")
        expected = 0 if explicit else min(GATE_DEPTH, len(original))
        if (type(ce.get("ce_pairs")) is not int or ce["ce_pairs"] != expected
                or original[expected:] != reranked[expected:]
                or {tuple(row) for row in original[:expected]}
                != {tuple(row) for row in reranked[:expected]}):
            raise EvaluationError(f"{qid}: CE40 pair count or fixed RRF pool mismatch")
        vector = ce.get("features")
        if expected == 0:
            if vector is not None:
                raise EvaluationError(f"{qid}: bypassed CE40 row contains gate features")
            counts["explicit" if explicit else "empty_pool"] += 1
            continue
        if (not isinstance(vector, list) or len(vector) != len(GATE_FEATURE_NAMES)
                or not all(_valid_number(value) for value in vector)):
            raise EvaluationError(f"{qid}: invalid CE40 feature vector")
        try:
            base_ndcg = _metric_for_query(qrels[qid], [item[0] for item in original], CUTOFF)[0]
            ce_ndcg = _metric_for_query(qrels[qid], [item[0] for item in reranked], CUTOFF)[0]
        except OverflowError as error:
            raise EvaluationError(f"{qid}: qrels exceed the nDCG calculation range") from error
        if not all(math.isfinite(score) and 0.0 <= score <= 1.0 + 1e-12
                   for score in (base_ndcg, ce_ndcg)):
            raise EvaluationError(f"{qid}: invalid nDCG label")
        matrix.append(vector)
        labels.append(int(ce_ndcg > base_ndcg))
        counts["wins" if ce_ndcg > base_ndcg else "ties" if ce_ndcg == base_ndcg else "losses"] += 1
    if set(labels) != {0, 1}:
        raise EvaluationError("training requires both positive and nonpositive labels")
    return matrix, labels, counts


def _fit(matrix, labels, binding):
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    classifier = LogisticRegression(**FIT)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(scaler.fit_transform(matrix), labels)
    if classifier.classes_.tolist() != [0, 1]:
        raise EvaluationError("unexpected fitted CE gate classes")
    artifact = {
        "schema_version": 1,
        "feature_names": list(GATE_FEATURE_NAMES),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "binding": binding,
    }
    runtime = {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
               "iterations": classifier.n_iter_.tolist()}
    return artifact, runtime


def _encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _publish_pair(model_out, artifact, receipt_out, receipt):
    """Stage complete files and publish with create-only hard links."""
    staged, linked = [], []
    try:
        for target, data in ((model_out, artifact), (receipt_out, receipt)):
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                pending = Path(stream.name)
                staged.append((target, pending))
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        for target, pending in staged:
            os.link(pending, target)
            linked.append((target, pending))
    except BaseException:
        for target, pending in reversed(linked):
            if target.exists() and target.samefile(pending):
                target.unlink()
        raise
    finally:
        for _, pending in staged:
            pending.unlink()


def train(capture, train_queries, train_qrels, model_out, receipt_out):
    if (model_out.resolve() == receipt_out.resolve()
            or os.path.lexists(model_out) or os.path.lexists(receipt_out)):
        raise EvaluationError("model and receipt require two distinct new output paths")

    inputs = {"queries": train_queries, "qrels": train_qrels}
    hashes = {key: sha256(path) for key, path in inputs.items()}
    query_ids = [record.qid for record in load_queries(train_queries)]
    qrels = load_qrels(train_qrels)
    if any(sha256(path) != hashes[key] for key, path in inputs.items()):
        raise EvaluationError("train inputs changed while loading")
    if set(query_ids) != qrels.keys():
        raise EvaluationError("train query IDs and qrel IDs must match exactly")
    protocol, binding, capture_digest, captured = _read_capture(capture, hashes, query_ids)
    matrix, labels, counts = _training_rows(query_ids, qrels, captured, protocol["depths"].keys())
    artifact, runtime = _fit(matrix, labels, binding)
    model_bytes = _encode(artifact)
    receipt = {
        "schema_version": 1,
        "inputs_sha256": {**hashes, "capture": capture_digest},
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "capture_identities": protocol["identities"],
        "fit": {"scaler": "StandardScaler", "classifier": "LogisticRegression", **FIT},
        "runtime": runtime,
        "label_rule": "ndcg@10(ce40) > ndcg@10(rrf)",
        "counts": {
            "queries": len(query_ids), "eligible": len(labels),
            "positive": counts["wins"], "nonpositive": counts["ties"] + counts["losses"],
            "ties": counts["ties"], "losses": counts["losses"],
            "skipped_explicit": counts["explicit"], "skipped_empty_pool": counts["empty_pool"],
        },
    }
    _publish_pair(model_out, model_bytes, receipt_out, _encode(receipt))
    return receipt


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "train-queries", "train-qrels", "model-out", "receipt-out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args(argv)
    receipt = train(args.capture, args.train_queries, args.train_qrels, args.model_out, args.receipt_out)
    print(receipt["model_sha256"])


if __name__ == "__main__":
    main()
