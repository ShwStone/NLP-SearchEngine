"""Collect and verify interleaved HTTP latency on a frozen query panel.

Each collection uses a fresh output directory. Verification joins its journal
to a server startup and exactly one search log event per request, including
warmup. Only the measured requests contribute to the latency statistics.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import combinations
import json
import os
from pathlib import Path
import random
import re
from statistics import fmean
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

from contracts.retrieval import valid_query_encoder_runtime

from .compare import _percentile
from .evaluate import EvaluationError, QueryRouting, _request_search, load_queries
from .serving_cost import MODES, STAGES, distribution, milliseconds, read_object, require, sha256


DIGEST = re.compile(r"[0-9a-f]{64}")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}")
TREC_TOKEN = re.compile(r"[^\s\x00-\x1f\x7f]+")
RESERVED_FILENAME = re.compile(r"(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])")
IDENTITY_FIELDS = ("index_identity", "vector_identity", "search_config", "policy")
BOOTSTRAP_SEED = 2026
BOOTSTRAP_SAMPLES = 10_000


def load_protocol(plan_path, query_path):
    raw = plan_path.read_bytes()
    plan = read_object(raw)
    require(set(plan) == {
        "schema_version", "queries_sha256", "corpus_sha256", "query_ids",
        "warmup_query_ids", "repeats", "depth", "timeout_s", "arms",
    } and type(plan["schema_version"]) is int and plan["schema_version"] in (1, 2),
        "Invalid plan schema")
    query_raw = query_path.read_bytes()
    require(plan["queries_sha256"] == sha256(query_raw), "Query file SHA256 differs from plan")
    corpus = plan["corpus_sha256"]
    require(isinstance(corpus, str) and DIGEST.fullmatch(corpus), "Invalid corpus SHA256")
    for field in ("repeats", "depth"):
        require(type(plan[field]) is int and plan[field] > 0, f"{field} must be a positive integer")
    require(milliseconds(plan["timeout_s"], "timeout_s") > 0, "timeout_s must be positive")
    queries = {record.qid: record.query for record in load_queries(query_path)}
    require(query_path.read_bytes() == query_raw, "Query file changed during read")
    for field in ("query_ids", "warmup_query_ids"):
        ids = plan[field]
        require(isinstance(ids, list) and all(isinstance(qid, str) and qid in queries for qid in ids)
                and len(ids) == len(set(ids)), f"Invalid or duplicate IDs in {field}")
    require(plan["query_ids"] and not set(plan["query_ids"]).intersection(plan["warmup_query_ids"]),
            "Measured query IDs must be nonempty and disjoint from warmup query IDs")
    require(all(TREC_TOKEN.fullmatch(qid) for qid in plan["query_ids"]), "Invalid TREC query ID")
    arms = plan["arms"]
    require(isinstance(arms, list) and len(arms) >= 2, "At least two arms are required")
    profiles = {}
    labels = set()
    for arm in arms:
        required = {"label", "endpoint", "mode", "policy_version"}
        require(isinstance(arm, dict)
                and required <= set(arm) <= required | {"identity_sha256", "allow_adaptive_dense"}
                and all(isinstance(arm[field], str) and arm[field] for field in required),
                "Invalid arm schema")
        if "identity_sha256" in arm:
            identity = arm["identity_sha256"]
            require(isinstance(identity, str) and DIGEST.fullmatch(identity), "Invalid arm identity SHA256")
        label, mode, endpoint = arm["label"], arm["mode"], arm["endpoint"]
        allow_adaptive_dense = arm.get("allow_adaptive_dense", False)
        require(type(allow_adaptive_dense) is bool, "allow_adaptive_dense must be a boolean")
        require(not allow_adaptive_dense or (mode == "adaptive" and "identity_sha256" in arm),
                "allow_adaptive_dense requires an adaptive arm with a frozen identity_sha256")
        require(LABEL.fullmatch(label) and not RESERVED_FILENAME.fullmatch(label)
                and label.lower() not in labels and mode in MODES, "Invalid or duplicate arm label or mode")
        url = urlsplit(endpoint)
        require(url.scheme in ("http", "https") and url.hostname and url.username is None
                and url.password is None and url.path in ("", "/") and not url.query and not url.fragment
                and endpoint == endpoint.strip() and not any(char.isspace() for char in endpoint)
                and (url.port is None or url.port > 0), "Endpoint must be an HTTP(S) origin")
        endpoint = endpoint.rstrip("/")
        policy, modes = profiles.setdefault(endpoint, (arm["policy_version"], set()))
        require(policy == arm["policy_version"] and mode not in modes,
                "An endpoint must use one policy version and distinct modes")
        modes.add(mode)
        labels.add(label.lower())
        arm["endpoint"] = endpoint
    return plan, sha256(raw), queries


def schedule(plan, epoch):
    labels = [arm["label"] for arm in plan["arms"]]
    count = len(labels)
    for phase, ids, repeats in (
        ("warmup", plan["warmup_query_ids"], 1),
        ("measure", plan["query_ids"], plan["repeats"]),
    ):
        for repeat in range(repeats):
            for index, qid in enumerate(ids):
                slot = repeat * len(ids) + index
                offset = slot % count
                for position, label in enumerate(labels[offset:] + labels[:offset]):
                    yield {
                        "phase": phase, "repeat": repeat, "qid": qid, "label": label,
                        "position": position,
                        "tag": f"paired-{epoch}-{phase}-{slot}-{position}",
                    }


def service_identity(value, corpus, policy_version, *, schema_version=1):
    identity = {field: value[field] for field in IDENTITY_FIELDS}
    require(all(isinstance(item, dict) and item for item in identity.values()),
            "Missing index, vector, search or policy identity")
    index = identity["index_identity"]
    require(isinstance(index["dataset"], str) and index["dataset"] and index["corpus_sha256"] == corpus,
            "Index dataset or corpus SHA256 differs from plan")
    require(identity["vector_identity"]["corpus_sha256"] == corpus
            and identity["policy"]["policy_version"] == policy_version,
            "Vector corpus or effective policy version differs from plan")
    if schema_version == 2:
        runtime = value.get("query_encoder_runtime")
        require(valid_query_encoder_runtime(runtime), "Missing or invalid query encoder runtime identity")
        identity["query_encoder_runtime"] = runtime
    return identity


def identity_digest(identity):
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return sha256(raw)


def validate_identities(plan, identities):
    require(isinstance(identities, dict) and set(identities) == {arm["label"] for arm in plan["arms"]},
            "Client service identity coverage differs from plan")
    shared = None
    for arm in plan["arms"]:
        observed = identities[arm["label"]]
        require(arm["mode"] != "hybrid" or "reranker" not in observed,
                "Use evaluation.ce_http for a CE-enabled hybrid arm")
        identity = service_identity(observed, plan["corpus_sha256"], arm["policy_version"],
                                    schema_version=plan.get("schema_version", 1))
        if "identity_sha256" in arm:
            require(identity_digest(identity) == arm["identity_sha256"],
                    f"Effective service identity differs from the frozen digest for {arm['label']}")
        common = {field: value for field, value in identity.items() if field != "policy"}
        if shared is None:
            shared = common
        require(common == shared, "Arms use different index, vector, search or query encoder runtime identity")
        require(type(common["search_config"]["candidate_k"]) is int
                and plan["depth"] <= common["search_config"]["candidate_k"],
                "Requested depth exceeds the effective candidate pool")


def get_identities(plan):
    by_endpoint = {}
    identities = {}
    for arm in plan["arms"]:
        endpoint = arm["endpoint"]
        if endpoint not in by_endpoint:
            url = endpoint + "/index/info"
            with urlopen(url, timeout=plan["timeout_s"]) as response:
                require(response.geturl() == url and response.status == 200,
                        "Unexpected HTTP index identity response")
                value = read_object(response.read())
            retrieval = value["retrieval"]
            require(retrieval["vector_ready"] is True, "Expected a vector-ready HTTP endpoint")
            by_endpoint[endpoint] = retrieval
        retrieval = by_endpoint[endpoint]
        require(arm["mode"] != "hybrid" or "reranker" not in retrieval,
                "Use evaluation.ce_http for a CE-enabled hybrid arm")
        identities[arm["label"]] = service_identity(retrieval, plan["corpus_sha256"], arm["policy_version"],
                                                    schema_version=plan.get("schema_version", 1))
    validate_identities(plan, identities)
    return identities


def append(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def validate_doc_ids(doc_ids, depth):
    require(isinstance(doc_ids, list) and len(doc_ids) <= depth
            and all(isinstance(doc_id, str) and TREC_TOKEN.fullmatch(doc_id) for doc_id in doc_ids)
            and len(doc_ids) == len(set(doc_ids)), "Invalid ranked document IDs")


def collect(plan, digest, queries, out):
    out.mkdir()
    arms = {arm["label"]: arm for arm in plan["arms"]}
    epoch = uuid4().hex
    journal = out / "requests.jsonl"
    with journal.open("x", encoding="utf-8", newline="\n") as stream:
        append(stream, {"event": "START", "epoch": epoch, "plan_sha256": digest})
        before = get_identities(plan)
        append(stream, {"event": "IDENTITY", "services": before})
        for entry in schedule(plan, epoch):
            arm = arms[entry["label"]]
            append(stream, {"event": "BEGIN", "request": entry})
            results, latency, routing = _request_search(
                arm["endpoint"], queries[entry["qid"]], k=plan["depth"], mode=arm["mode"],
                scorer="bm25", expander="none", timeout=plan["timeout_s"], request_tag=entry["tag"],
                allow_adaptive_dense=arm.get("allow_adaptive_dense", False),
            )
            require(not routing.fallback, "Fallback observed in a no-fallback measurement")
            outcome = {"event": "OUTCOME", "tag": entry["tag"], "status": 200,
                       "latency_ms": milliseconds(latency, "client latency_ms"), "routing": asdict(routing)}
            if entry["phase"] == "measure":
                doc_ids = [doc_id for doc_id, _ in results]
                validate_doc_ids(doc_ids, plan["depth"])
                outcome["doc_ids"] = doc_ids
            append(stream, outcome)
        after = get_identities(plan)
        require(before == after, "Service identity changed during measurement")
        append(stream, {"event": "COMPLETE", "services": after})
    return {"status": "COLLECTED", "epoch": epoch, "journal": str(journal)}


def read_journal(path, plan, digest):
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "Client journal ends in a partial line")
    events = [read_object(line) for line in raw.splitlines()]
    require(len(events) >= 3 and events[-1]["event"] == "COMPLETE", "Client journal is incomplete")
    epoch = events[0]["epoch"]
    require(isinstance(epoch, str) and re.fullmatch(r"[0-9a-f]{32}", epoch)
            and events[0] == {"event": "START", "epoch": epoch, "plan_sha256": digest},
            "Client journal START differs from frozen plan")
    before = events[1]["services"]
    require(events[1] == {"event": "IDENTITY", "services": before}
            and events[-1] == {"event": "COMPLETE", "services": before},
            "Client service identity changed during measurement")
    validate_identities(plan, before)
    attempts = list(schedule(plan, epoch))
    require(len(events) == 3 + 2 * len(attempts), "Client journal request coverage differs from plan")
    arms = {arm["label"]: arm for arm in plan["arms"]}
    outcomes = {}
    for index, entry in enumerate(attempts):
        begin, outcome = events[2 + 2 * index:4 + 2 * index]
        tag = entry["tag"]
        fields = {"event", "tag", "status", "latency_ms", "routing"}
        if entry["phase"] == "measure":
            fields.add("doc_ids")
        require(begin == {"event": "BEGIN", "request": entry}
                and set(outcome) == fields
                and outcome["event"] == "OUTCOME" and outcome["tag"] == tag
                and type(outcome["status"]) is int and outcome["status"] == 200,
                f"Invalid journal outcome for {tag}")
        arm = arms[entry["label"]]
        routing = QueryRouting.from_response(outcome["routing"], arm["mode"],
                                             allow_adaptive_dense=arm.get("allow_adaptive_dense", False))
        require(not routing.fallback, f"Fallback observed for {tag}")
        doc_ids = outcome.get("doc_ids")
        if entry["phase"] == "measure":
            validate_doc_ids(doc_ids, plan["depth"])
        outcomes[tag] = (entry, routing, milliseconds(outcome["latency_ms"], "client latency_ms"), doc_ids)
    return epoch, before, outcomes, sha256(raw)


def read_logs(paths, arms, outcomes, epoch):
    require(set(paths) == set(arms), "Supply exactly one server log path per arm label")
    by_endpoint = {}
    for label, path in paths.items():
        endpoint = arms[label]["endpoint"]
        resolved = path.resolve()
        require(by_endpoint.setdefault(endpoint, resolved) == resolved,
                "Arms sharing an endpoint must share the same server log")
    logs = {}
    for path in set(by_endpoint.values()):
        raw = path.read_bytes()
        require(raw.endswith(b"\n"), "Server log snapshot ends in a partial line")
        digest = sha256(raw)
        events = {}
        startup = None
        for line_number, line in enumerate(raw.decode("utf-8-sig").split("\n"), 1):
            if not line.lstrip().startswith("{"):
                continue
            row = read_object(line)
            if row.get("event") == "startup":
                startup = (line_number, row)
            tag = row.get("run_tag")
            if row.get("event") != "search" or not isinstance(tag, str) or not tag.startswith(f"paired-{epoch}-"):
                continue
            require(tag in outcomes and tag not in events and startup is not None,
                    f"Unexpected, duplicate or startup-less server request {tag}")
            label = outcomes[tag][0]["label"]
            require(by_endpoint[arms[label]["endpoint"]] == path, f"Server log is bound to the wrong arm for {tag}")
            events[tag] = (startup, row)
        logs[path] = (digest, events)
    return {endpoint: logs[path] for endpoint, path in by_endpoint.items()}


def paired_interval(values):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    draws = [fmean(rng.choices(values, k=n)) for _ in range(BOOTSTRAP_SAMPLES)]
    return {
        "queries": n, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED,
        "mean_delta_ms": fmean(values),
        "ci95_lower_ms": _percentile(draws, 0.025), "ci95_upper_ms": _percentile(draws, 0.975),
    }


def publish_runs(run_dir, depth, rankings):
    runs = {}
    run_dir.mkdir()
    for label, by_query in rankings.items():
        path = run_dir / f"{label}.trec"
        raw = "".join(
            f"{qid} Q0 {doc_id} {rank} {depth - rank + 1} {label}\n"
            for qid, doc_ids in by_query.items()
            for rank, doc_id in enumerate(doc_ids, 1)
        ).encode("utf-8")
        with path.open("xb") as stream:
            stream.write(raw)
        runs[label] = {"file": path.name, "sha256": sha256(raw)}
    return {"repeat": 0, "arms": runs}


def verify(plan, digest, journal, paths, run_dir=None):
    epoch, identities, outcomes, journal_digest = read_journal(journal, plan, digest)
    arms = {arm["label"]: arm for arm in plan["arms"]}
    logs = read_logs(paths, arms, outcomes, epoch)
    startup_lines = {}
    log_epochs = {}
    latency = {label: defaultdict(list) for label in arms}
    rankings = {label: {} for label in arms}
    server_times = {label: defaultdict(list) for label in arms}
    vector_attempts = dict.fromkeys(arms, 0)
    vector_uses = dict.fromkeys(arms, 0)
    route_counts = {label: Counter() for label in arms}
    for tag, (entry, routing, elapsed, doc_ids) in outcomes.items():
        label = entry["label"]
        arm = arms[label]
        endpoint = arm["endpoint"]
        log_digest, logged = logs[endpoint]
        require(tag in logged, f"Missing server event for {tag}")
        (line, startup), row = logged[tag]
        require(arm["mode"] != "hybrid" or (
            "reranker" not in startup and "rerank" not in row
            and "rerank" not in (row.get("timings_ms") or {})),
            "Use evaluation.ce_http for a CE-enabled hybrid arm")
        if endpoint not in startup_lines:
            startup_lines[endpoint] = line
            require(service_identity(startup, plan["corpus_sha256"], arm["policy_version"],
                                     schema_version=plan.get("schema_version", 1)) == identities[label],
                    f"Client and server startup identity differ for {label}")
        require(line == startup_lines[endpoint], f"Endpoint {label} spans multiple server startups")
        require(log_epochs.setdefault(log_digest, line) == line, "A shared server log spans multiple startups")
        dataset = identities[label]["index_identity"]["dataset"]
        require(type(row["status"]) is int and row["status"] == 200
                and (startup["dataset"], startup["policy_version"]) == (dataset, arm["policy_version"])
                and (row["dataset"], row["corpus_identity"], row["policy_version"]) ==
                    (dataset, plan["corpus_sha256"][:12], arm["policy_version"])
                and QueryRouting.from_response(row, arm["mode"],
                                               allow_adaptive_dense=arm.get("allow_adaptive_dense", False)) == routing,
                f"HTTP status, identity or routing disagreement for {tag}")
        if entry["phase"] == "measure":
            if entry["repeat"] == 0:
                rankings[label][entry["qid"]] = doc_ids
            latency[label][entry["qid"]].append(elapsed)
            vector_attempts[label] += routing.vector_attempted
            vector_uses[label] += routing.vector_used
            route_counts[label][routing.executed_mode] += 1
            for field in STAGES:
                server_times[label][field].append(milliseconds(row["timings_ms"][field], field))
            for field in ("serialization_ms", "http_elapsed_ms", "took_ms"):
                server_times[label][field].append(milliseconds(row[field], field))

    comparisons = []
    for left, right in combinations(arms, 2):
        deltas = [fmean(b - a for a, b in zip(latency[left][qid], latency[right][qid]))
                  for qid in plan["query_ids"]]
        comparisons.append({"left": left, "right": right, **paired_interval(deltas)})
    report = {
        "status": "VERIFIED", "epoch": epoch, "plan_sha256": digest, "journal_sha256": journal_digest,
        "corpus_sha256": plan["corpus_sha256"], "paired_client_latency": comparisons,
        **({"query_encoder_runtime": next(iter(identities.values()))["query_encoder_runtime"]}
           if plan.get("schema_version", 1) == 2 else {}),
        "arms": {
            label: {
                "server_log_sha256": logs[arm["endpoint"]][0],
                "startup_line": startup_lines[arm["endpoint"]],
                "identity_sha256": identity_digest(identities[label]),
                **({"allow_adaptive_dense": True} if arm.get("allow_adaptive_dense") else {}),
                "warmup_requests": len(plan["warmup_query_ids"]),
                "vector_attempts": vector_attempts[label], "vector_uses": vector_uses[label],
                "route_counts": dict(route_counts[label]),
                "client_ms": distribution([value for values in latency[label].values() for value in values]),
                "server_ms": {field: distribution(values) for field, values in server_times[label].items()},
            }
            for label, arm in arms.items()
        },
    }
    if run_dir is not None:
        report["trec_runs"] = publish_runs(run_dir, plan["depth"], rankings)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--queries", required=True, type=Path)
    commands.choices["collect"].add_argument("--out", required=True, type=Path)
    verifier = commands.choices["verify"]
    verifier.add_argument("--journal", required=True, type=Path)
    verifier.add_argument("--server-log", required=True, action="append", metavar="LABEL=PATH")
    verifier.add_argument("--run-dir", type=Path, help="Export verified repeat-zero TREC runs to a new directory")
    args = parser.parse_args(argv)
    try:
        plan, digest, queries = load_protocol(args.plan, args.queries)
        if args.command == "collect":
            report = collect(plan, digest, queries, args.out)
        else:
            paths = {}
            for argument in args.server_log:
                label, separator, path = argument.partition("=")
                require(separator and label and path and label not in paths, "Use distinct LABEL=PATH arguments")
                paths[label] = Path(path)
            report = verify(plan, digest, args.journal, paths, args.run_dir)
    except (EvaluationError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"paired_http: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
