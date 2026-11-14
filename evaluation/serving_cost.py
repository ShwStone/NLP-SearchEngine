"""Reconcile serial HTTP evaluation receipts with search timing JSONL.

Each measured attempt must use a unique HTTP request tag. Pass a complete,
immutable server log containing its startup event and the finalized sidecar.
The server exposes no query IDs. Matching relies on the exact tag, ordered
status and routing agreement, and a single startup epoch per measured arm.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
from string import hexdigits

from .evaluate import (
    DENSE_CONSTRAINT_ERROR,
    QueryRouting,
    _REQUEST_TAG,
    _percentile,
    _unique_json_object,
)


MODES = ("lexical", "dense", "hybrid", "adaptive")
STAGES = ("lexical", "router", "encode", "vector", "fusion")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def query_hash(ids):
    return sha256(json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def read_object(raw):
    value = json.loads(raw, object_pairs_hook=_unique_json_object)
    require(isinstance(value, dict), "Expected a JSON object")
    return value


def read_evaluation(sidecar):
    raw = sidecar.read_bytes()
    receipt = read_object(raw)
    require(type(receipt["schema_version"]) is int and receipt["schema_version"] == 1
            and receipt["status"] == "COMPLETE",
            "HTTP sidecar is not finalized")
    request = receipt["request"]
    require(isinstance(request, dict), "Sidecar request is invalid")
    tag = request.get("run_tag")
    require(isinstance(tag, str) and _REQUEST_TAG.fullmatch(tag),
            "Finalized sidecar has no valid HTTP request.run_tag binding")
    mode = request["mode"]
    require(mode in MODES, "Unsupported HTTP mode")
    journal = receipt["journal"]
    name = sidecar.name + ".events.jsonl"
    require(journal["file"] == name, "Journal filename differs from evaluator convention")
    journal_raw = sidecar.with_name(name).read_bytes()
    require(journal_raw.endswith(b"\n") and sha256(journal_raw) == journal["sha256"],
            "Journal is truncated or SHA256 disagrees with sidecar")
    events = [read_object(line) for line in journal_raw.splitlines()]
    qids = receipt["query_ids"]
    require(isinstance(qids, list) and qids
            and all(isinstance(qid, str) and qid.strip() == qid and qid for qid in qids)
            and len(qids) == len(set(qids)) and receipt["query_universe_sha256"] == query_hash(qids),
            "Sidecar query universe is invalid")
    count = len(qids)
    outcomes = receipt["outcomes"]
    require(isinstance(outcomes, list)
            and all(type(receipt[field]) is int for field in (
                "requested_queries", "attempted_requests", "last_event_sequence", "expected_unsupported",
                "unexpected_errors", "http_200", "routing_observed_queries"
            ))
            and receipt["requested_queries"] == receipt["attempted_requests"] == len(outcomes) == count
            and receipt["last_event_sequence"] == 2 * count
            and len(events) == 1 + 2 * count and receipt["inflight_outcome"] is None
            and receipt["last_outcome"] == outcomes[-1], "Evaluator event counts disagree")
    require(events[0] == {"event": "START", "sequence": 0, "query_ids": qids,
                          "query_universe_sha256": receipt["query_universe_sha256"],
                          "policy": receipt["policy"], "request": request},
            "Journal START differs from finalized sidecar")
    expected_rejections = set()
    if receipt["policy"] is not None:
        policy = receipt["policy"]
        expected = policy["expected_unsupported"]
        expected_rejections = set(expected["query_ids"])
        require(mode == "dense" and policy["request"] == {
            field: request[field] for field in ("mode", "scorer", "expander")
        } and expected["status"] == 400 and expected["body"] == {"error": DENSE_CONSTRAINT_ERROR}
            and len(expected["query_ids"]) == len(expected_rejections)
            and expected_rejections.issubset(qids), "Invalid dense constraint policy evidence")
    for attempt, (qid, outcome) in enumerate(zip(qids, outcomes), 1):
        begin_outcome = {"qid": qid, "status": "INCOMPLETE", "http_status": None,
                         "latency_ms": None, "routing": None}
        require(events[2 * attempt - 1] == {"event": "BEGIN", "sequence": 2 * attempt - 1,
                                             "attempt": attempt, "outcome": begin_outcome}
                and events[2 * attempt] == {"event": "OUTCOME", "sequence": 2 * attempt,
                                            "attempt": attempt, "outcome": outcome}
                and outcome["qid"] == qid, f"Journal outcome differs at request {attempt}")
        if qid in expected_rejections:
            require(outcome["status"] == "EXPECTED_UNSUPPORTED" and outcome["http_status"] == 400
                    and outcome["routing"] is None and outcome["body"] == expected["body"],
                    f"Dense explicit-constraint rejection differs at request {attempt}")
        else:
            require(outcome["status"] == "SUCCESS" and outcome["http_status"] == 200,
                    f"Unexpected evaluator outcome at request {attempt}")
    rejected = len(expected_rejections)
    require(receipt["expected_unsupported"] == rejected and receipt["unexpected_errors"] == 0
            and receipt["http_200"] == receipt["routing_observed_queries"] == count - rejected
            and receipt["http_success_coverage"] == (count - rejected) / count
            and receipt["protocol_accounted_coverage"] == 1.0,
            "Evaluator coverage differs from ordered journal outcomes")
    return receipt, sha256(raw)


def read_server_log(path, tag):
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "Server log snapshot ends in a partial line")
    epoch = None
    matched = []
    epochs = {}
    untagged_searches = 0
    other_tag_searches = 0
    for number, line in enumerate(raw.decode("utf-8-sig").split("\n"), 1):
        if not line.lstrip().startswith("{"):
            continue  # The serving CLI also emits human-readable messages.
        event = read_object(line)
        if event.get("event") == "startup":
            epoch = number
            epochs[epoch] = event
        if event.get("event") != "search":
            continue
        observed_tag = event.get("run_tag")
        if observed_tag == tag:
            require(epoch is not None, "Tagged search precedes the first startup event")
            matched.append((epoch, event))
        elif observed_tag == "":
            untagged_searches += 1
        else:
            other_tag_searches += 1
    require(matched, "No search log events match the exact HTTP request tag")
    require(len({line for line, _ in matched}) == 1, "Exact HTTP tag spans multiple server startups")
    epoch = matched[0][0]
    return (epochs[epoch], epoch, [row for _, row in matched], sha256(raw),
            untagged_searches, other_tag_searches)


def milliseconds(value, field):
    require(type(value) in (float, int) and 0 <= value and math.isfinite(value),
            f"Missing or invalid measured {field}")
    return value


def distribution(values):
    return {"n": len(values), "mean": statistics.fmean(values) if values else None,
            "p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95)}


def summarize_run(sidecar, server_log):
    """Return measured evidence for one bound HTTP evaluation attempt."""
    sidecar, server_log = Path(sidecar), Path(server_log)
    receipt, receipt_hash = read_evaluation(sidecar)
    mode = receipt["request"]["mode"]
    allow_adaptive_dense = receipt["request"].get("allow_adaptive_dense", False)
    require(type(allow_adaptive_dense) is bool and (not allow_adaptive_dense or mode == "adaptive"),
            "Invalid adaptive dense request policy")
    tag = receipt["request"]["run_tag"]
    startup, startup_line, logged, log_hash, untagged, other_tag = read_server_log(server_log, tag)
    require(mode != "hybrid" or ("reranker" not in startup and all(
        "rerank" not in row and "rerank" not in (row.get("timings_ms") or {}) for row in logged)),
        "Use evaluation.ce_http for a CE-enabled hybrid run")
    outcomes = receipt["outcomes"]
    require(len(logged) == len(outcomes),
            f"Exact HTTP tag count disagreement (server {len(logged)}, evaluator {len(outcomes)})")
    corpus = startup["index_identity"]["corpus_sha256"]
    dataset = startup["dataset"]
    policy_version = startup["policy_version"]
    require(isinstance(corpus, str) and len(corpus) == 64 and all(char in hexdigits for char in corpus)
            and isinstance(dataset, str) and dataset and isinstance(policy_version, str) and policy_version,
            "Incomplete startup corpus or policy identity")
    server_200 = {field: [] for field in (*STAGES, "serialization_ms", "http_elapsed_ms", "took_ms")}
    client_200 = []
    server_400 = {field: [] for field in ("serialization_ms", "http_elapsed_ms")}
    client_400 = []
    execution, reasons = Counter(), Counter()
    fallback, attempted, used = 0, 0, 0
    for attempt, (outcome, row) in enumerate(zip(outcomes, logged), 1):
        require(row["requested_mode"] == mode and type(row["status"]) is int
                and type(outcome["http_status"]) is int and row["status"] == outcome["http_status"],
                f"Ordered mode or HTTP status disagreement at request {attempt}")
        require((row["dataset"], row["corpus_identity"], row["policy_version"]) ==
                (dataset, corpus[:12], policy_version), f"Serving identity disagreement at request {attempt}")
        if row["status"] == 400:
            require(mode == "dense" and row["reason"] == "invalid_request"
                    and all(row[field] is None for field in (
                        "timings_ms", "took_ms", "executed_mode", "fallback", "vector_attempted", "vector_used"
                    )), f"Server dense HTTP 400 shape disagrees at request {attempt}")
            for field, values in server_400.items():
                values.append(milliseconds(row[field], field))
            client_400.append(milliseconds(outcome["latency_ms"], "client latency_ms"))
            continue
        routing = QueryRouting.from_response(row, mode, allow_adaptive_dense=allow_adaptive_dense)
        require(routing == QueryRouting.from_response(outcome["routing"], mode,
                                                      allow_adaptive_dense=allow_adaptive_dense),
                f"Ordered routing disagreement at request {attempt}")
        execution[routing.executed_mode] += 1
        reasons[routing.reason] += 1
        fallback += routing.fallback
        attempted += routing.vector_attempted
        used += routing.vector_used
        for field in STAGES:
            server_200[field].append(milliseconds(row["timings_ms"][field], field))
        for field in ("serialization_ms", "http_elapsed_ms", "took_ms"):
            server_200[field].append(milliseconds(row[field], field))
        client_200.append(milliseconds(outcome["latency_ms"], "client latency_ms"))
    return {
        "requested_mode": mode,
        "request_tag": tag,
        "inputs": {"sidecar_sha256": receipt_hash, "journal_sha256": receipt["journal"]["sha256"],
                   "server_log_sha256": log_hash, "query_universe_sha256": receipt["query_universe_sha256"]},
        "server_epoch": {"startup_line": startup_line, "dataset": dataset, "corpus_sha256": corpus,
                         "policy_version": policy_version, "first_search_time": logged[0]["time"],
                         "last_search_time": logged[-1]["time"]},
        "excluded_server_searches": {"untagged": untagged, "other_tags": other_tag},
        "coverage": {"requests": len(outcomes), "matched_server_events": len(logged),
                     "http_200": len(client_200), "dense_explicit_constraint_http_400": len(client_400),
                     "http_success_coverage": receipt["http_success_coverage"],
                     "protocol_accounted_coverage": receipt["protocol_accounted_coverage"],
                     "routing_observed_http_200": len(client_200), "fallbacks_http_200": fallback,
                     "fallback_rate_of_http_200": fallback / len(client_200) if client_200 else None,
                     "vector_attempts_http_200": attempted, "vector_successes_http_200": used,
                     "executed_modes_http_200": dict(execution), "routing_reasons_http_200": dict(reasons)},
        "validated_http_200_cost_ms": {
            "server": {field: distribution(values) for field, values in server_200.items()},
            "client_latency_ms": distribution(client_200),
        },
        "expected_dense_http_400_cost_ms": {
            "server": {field: distribution(values) for field, values in server_400.items()},
            "client_latency_ms": distribution(client_400),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("SIDECAR", "SERVER_LOG"),
                        required=True, help="one finalized evaluator sidecar and its full server log")
    args = parser.parse_args(argv)
    results, failures = [], []
    tags = set()
    for sidecar, server_log in args.run:
        try:
            result = summarize_run(sidecar, server_log)
            tag = result["request_tag"]
            require(tag not in tags, "Duplicate HTTP request tag supplied across evaluation attempts")
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            failures.append({"sidecar": sidecar, "server_log": server_log, "error": str(error)})
            continue
        tags.add(tag)
        results.append(result)
    epochs = {(run["inputs"]["server_log_sha256"], run["server_epoch"]["startup_line"]) for run in results}
    print(json.dumps({
        "schema_version": 1,
        "status": "PASS" if not failures else "UNVERIFIED",
        "latency_scope": "Within each run and observed server log startup epoch only",
        "percentile_method": "linear interpolation at (n - 1) * p",
        "observed_log_startup_epochs": len(epochs),
        "runs": results,
        "unverified": failures,
    }, ensure_ascii=True, indent=2, allow_nan=False))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
