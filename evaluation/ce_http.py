"""Capture paired fixed RRF and cross-encoder controls through public HTTP."""

import argparse
from collections import Counter
from contextlib import closing, contextmanager
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import statistics
import tempfile
import time
import unicodedata
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

from contracts.api import SEARCH_INSTANCE_HEADER
from contracts.ce import GATE_DEPTH, GATE_FEATURE_NAMES, gate_binding
from contracts.retrieval import valid_query_encoder_runtime

from .ce_analysis import learned_diagnostics
from .ce_budget import matched_random
from .compare import paired_bootstrap
from .evaluate import (
    EvaluationError, _decode_search_response, _percentile, evaluate_run,
    _unique_json_object, load_qrels, load_queries,
)


REQUEST = {"k": 100, "mode": "hybrid", "scorer": "bm25", "expander": "none"}
CAPTURE_VERSION = 4
RANDOM_LIVE_VERSION = 1
SEED = 2026
BOOTSTRAP_SAMPLES = 10000
STAGES = ("lexical", "router", "encode", "vector", "fusion")
INSTANCE_ID = re.compile(r"[0-9a-f]{32}")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint(url):
    parts = urlsplit(url)
    host = parts.hostname
    try:
        local = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        local = False
    if (parts.scheme != "http" or not local or parts.port is None or not parts.port
            or parts.username is not None or parts.query or parts.fragment
            or parts.path.rstrip("/") not in ("", "/search")):
        raise EvaluationError("use an explicit loopback HTTP URL with a port")
    return f"http://{parts.netloc}"


def http_client(require_instance):
    client = build_opener(ProxyHandler({}), NoRedirect())
    client.service_instances = {} if require_instance else None
    return client


def http_json(client, base, path, body=None):
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(base + path, data=payload, headers={
        "Content-Type": "application/json", "Accept": "application/json",
    })
    started = time.perf_counter()
    with client.open(request, timeout=120) as response:
        if response.status != 200:
            raise EvaluationError(f"{path} returned HTTP {response.status}")
        raw = response.read()
    elapsed = (time.perf_counter() - started) * 1000
    instances = getattr(client, "service_instances", None)
    if instances is not None:
        tokens = response.headers.get_all(SEARCH_INSTANCE_HEADER) or []
        if len(tokens) != 1 or not INSTANCE_ID.fullmatch(tokens[0]):
            raise EvaluationError(f"{base}{path} did not return a valid search instance ID")
        if instances.setdefault(base, tokens[0]) != tokens[0]:
            raise EvaluationError(f"{base} restarted during the current capture epoch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise EvaluationError(f"{path} response must be a JSON object")
    return value, raw, elapsed


def check_services(client, arms):
    common = None
    model = None
    identities = {}
    for name, (depth, base) in arms.items():
        info, _, _ = http_json(client, base, "/index/info")
        retrieval = info.get("retrieval")
        if not isinstance(retrieval, dict) or retrieval.get("vector_ready") is not True:
            raise EvaluationError(f"{name} requires a ready vector index")
        for key in ("index_identity", "vector_identity", "search_config", "policy", "serving"):
            if not isinstance(retrieval.get(key), dict) or not retrieval[key]:
                raise EvaluationError(f"{name} missing retrieval.{key} identity")
        candidate_k = retrieval["search_config"].get("candidate_k")
        if type(candidate_k) is not int or candidate_k < REQUEST["k"]:
            raise EvaluationError(f"{name} candidate pool cannot support the fixed request")
        stable = {**info, "retrieval": {
            key: value for key, value in retrieval.items() if key != "reranker"
        }}
        stable["retrieval"]["serving"] = {
            key: value for key, value in retrieval["serving"].items() if key != "port"
        }
        if common is None:
            common = stable
        if ("query_encoder_runtime" in retrieval
                and not valid_query_encoder_runtime(retrieval["query_encoder_runtime"])):
            raise EvaluationError(f"{name} has an invalid query encoder runtime identity")
        if stable != common:
            raise EvaluationError(f"{name} service identity differs from baseline")
        reranker = retrieval.get("reranker")
        if depth:
            if (not isinstance(reranker, dict) or type(reranker.get("depth")) is not int
                    or reranker["depth"] != depth or reranker.get("candidate_source") != "hybrid_rrf"
                    or not isinstance(reranker.get("model"), dict)
                    or not all(reranker["model"].get(key) for key in
                               ("model_id", "revision", "weights_sha256", "inventory_sha256"))):
                raise EvaluationError(f"{name} has an invalid fixed CE identity")
            gate_sha = reranker.get("gate_sha256")
            if name == "learned":
                if (not isinstance(gate_sha, str) or len(gate_sha) != 64
                        or any(char not in "0123456789abcdef" for char in gate_sha)
                        or reranker.get("capture_features") is not True):
                    raise EvaluationError("learned profile requires a digest-bound gate")
            elif gate_sha is not None:
                raise EvaluationError(f"{name} must use a fixed-depth scorer")
            if model is None:
                model = reranker["model"]
            if reranker["model"] != model:
                raise EvaluationError(f"{name} CE model identity differs across arms")
        elif reranker is not None:
            raise EvaluationError("RRF baseline has an enabled reranker")
        identities[name] = reranker
    return {"common": common, "rerankers": identities}


def check_fit_receipt(path, expected_sha256, identity):
    """Bind a learned-arm capture to its frozen training runtime and gate."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise EvaluationError("fit receipt SHA256 differs from the frozen digest")
    try:
        receipt = json.loads(raw, object_pairs_hook=_unique_json_object)
        if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
            raise ValueError("unsupported schema version")
        training = receipt["capture_identities"]
        train_retrieval = training["common"]["retrieval"]
        train_model = training["rerankers"]["ce40"]["model"]
        retrieval = identity["common"]["retrieval"]
        learned = identity["rerankers"]["learned"]
        runtime = train_retrieval.get("query_encoder_runtime")
        matches_gate = receipt["model_sha256"] == learned["gate_sha256"]
        matches_retrieval = gate_binding(train_retrieval, train_model) == gate_binding(retrieval, learned["model"])
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise EvaluationError("fit receipt lacks the expected training and learned-arm identities") from error
    if not matches_gate or not matches_retrieval:
        raise EvaluationError("learned arm differs from the fitted gate or its retrieval binding")
    if not valid_query_encoder_runtime(runtime) or runtime != retrieval.get("query_encoder_runtime"):
        raise EvaluationError("query encoder runtime differs from the frozen training capture")
    if train_retrieval.get("bm25_accelerator") != retrieval.get("bm25_accelerator"):
        raise EvaluationError("BM25 accelerator differs from the frozen training capture")


def search(client, base, depth, record, identity, reranker_identity=None):
    response, raw, latency = http_json(client, base, "/search", {
        **REQUEST, "query": record.query,
    })
    ranking, routing = _decode_search_response(raw, "hybrid", allow_hybrid_rerank=depth > 0)
    items = response["results"]
    route_info = response["routing"]
    parsed = response.get("parsed")
    explicit = parsed.get("explicit_constraints") if isinstance(parsed, dict) else None
    if (routing.fallback or route_info.get("score_kind") != "rrf"
            or route_info.get("candidate_k") != identity["search_config"]["candidate_k"]
            or route_info.get("policy_version") != identity["policy"]["policy_version"]
            or response.get("expanded") != []
            or type(explicit) is not bool
            or not len(ranking) == len(items) <= REQUEST["k"]
            or any(type(item.get("rank")) is not int or item["rank"] != rank
                   or type(item.get("score")) not in (int, float)
                   or not item["doc_id"] or any(char.isspace() for char in item["doc_id"])
                   for rank, item in enumerate(items, 1))):
        raise EvaluationError("invalid fixed hybrid response")
    timings = response.get("timings_ms")
    expect_rerank = depth and not explicit and bool(ranking)
    expected_stages = {*STAGES, "rerank"} if expect_rerank else set(STAGES)
    if (not isinstance(timings, dict) or timings.keys() != expected_stages
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in timings.values())):
        raise EvaluationError("invalid fixed hybrid stage timings")
    rerank = response.get("rerank")
    scored = 0
    feature_row = None
    if expect_rerank:
        profile = reranker_identity or {}
        expected_gate = profile.get("gate_sha256")
        possible_pairs = {min(depth, len(ranking))}
        if expected_gate is not None:
            possible_pairs.add(0)
        if (not isinstance(rerank, dict) or type(rerank.get("depth")) is not int
                or rerank["depth"] != depth or type(rerank.get("scored")) is not int
                or rerank["scored"] not in possible_pairs or rerank.get("gate_sha256") != expected_gate):
            raise EvaluationError("CE response has missing or inconsistent pair counts")
        scored = rerank["scored"]
        feature_row = rerank.get("features")
        if profile.get("capture_features") is True:
            if (not isinstance(feature_row, list) or len(feature_row) != len(GATE_FEATURE_NAMES)
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for value in feature_row)):
                raise EvaluationError("CE response omitted the agreed pre-rerank features")
        elif feature_row is not None:
            raise EvaluationError("CE response returned unexpected gate features")
    elif "rerank" in response:
        raise EvaluationError("response reranked candidates where CE must be bypassed")
    result = {"ranking": ranking, "http_ms": latency, "timings_ms": timings, "ce_pairs": scored,
              "explicit_constraints": explicit, "features": feature_row}
    instances = getattr(client, "service_instances", None)
    if instances is not None:
        result["service_instance"] = instances[base]
    return result


def capture_query(client, arms, record, index, order, identity):
    shift = index % len(order)
    observations = {}
    for name in order[shift:] + order[:shift]:
        depth, base = arms[name]
        try:
            observations[name] = search(client, base, depth, record,
                                        identity["common"]["retrieval"], identity["rerankers"][name])
        except (OSError, ValueError, TypeError) as error:
            raise EvaluationError(f"qid={record.qid!r}, arm={name}: {error}") from error
    baseline = observations["rrf"]
    for name, result in observations.items():
        if result["explicit_constraints"] != baseline["explicit_constraints"]:
            raise EvaluationError(f"qid={record.qid!r}, arm={name}: query parsing differs from baseline")
        prefix = result["ce_pairs"]
        ranking = result["ranking"]
        original = baseline["ranking"]
        if ranking[prefix:] != original[prefix:] or set(ranking[:prefix]) != set(original[:prefix]):
            raise EvaluationError(f"qid={record.qid!r}, arm={name}: changed the fixed RRF candidate pool")
    if "learned" in observations:
        learned, fixed = observations["learned"], observations["ce40"]
        if learned["ce_pairs"] and (learned["ce_pairs"] != fixed["ce_pairs"]
                                   or learned["ranking"] != fixed["ranking"]):
            raise EvaluationError(f"qid={record.qid!r}: active gate differs from fixed CE40")
        if fixed["features"] is not None and learned["features"] != fixed["features"]:
            raise EvaluationError(f"qid={record.qid!r}: fixed and learned features disagree")
    return observations


@contextmanager
def open_journal(path, *, replay):
    if replay:
        wal = path.with_name(path.name + "-wal")
        if wal.is_file() and wal.stat().st_size:
            raise EvaluationError("replay requires a sealed journal with no uncheckpointed WAL")
        rollback = path.with_name(path.name + "-journal")
        if rollback.is_file() and rollback.stat().st_size:
            raise EvaluationError("replay requires a sealed journal with no pending rollback")
        if not path.is_file():
            raise EvaluationError(f"source journal is missing: {path.name}")
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    else:
        connection = sqlite3.connect(path)
    with closing(connection) as db:
        yield db


def journal(db, protocol, *, replay=False):
    body = json.dumps(protocol, sort_keys=True, ensure_ascii=False, allow_nan=False)
    if not replay:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS protocol (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS queries (
                qid TEXT PRIMARY KEY, epoch TEXT NOT NULL, arms TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS invalid_epochs (epoch TEXT PRIMARY KEY, reason TEXT NOT NULL);
        """)
        with db:
            db.execute("INSERT OR IGNORE INTO protocol VALUES (1, ?)", (body,))
    row = db.execute("SELECT body FROM protocol WHERE id=1").fetchone()
    if row is None or row[0] != body:
        raise EvaluationError("output journal belongs to a different capture protocol")
    if db.execute("SELECT 1 FROM invalid_epochs LIMIT 1").fetchone():
        raise EvaluationError("capture journal contains an invalid service epoch")
    return {qid: (epoch, json.loads(arms)) for qid, epoch, arms in
            db.execute("SELECT qid, epoch, arms FROM queries ORDER BY rowid")}


def completed_identity(db, expected_count):
    """Return the saved identity when the verified final query is committed."""
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"protocol", "queries", "invalid_epochs"} <= tables:
        return None
    if db.execute("SELECT COUNT(*) FROM queries").fetchone()[0] != expected_count:
        return None
    row = db.execute("SELECT body FROM protocol WHERE id=1").fetchone()
    return json.loads(row[0])["identities"] if row is not None else None


def distribution(values):
    return {"mean": statistics.fmean(values), "p50": _percentile(values, .50),
            "p95": _percentile(values, .95)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_query(query):
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def publish(path, contents):
    """Publish complete bytes once; on resume require identical output."""
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            pending.unlink()
            raise
    try:
        try:
            os.link(pending, path)
        except FileExistsError:
            if path.read_bytes() != contents:
                raise EvaluationError(f"existing output differs from completed capture: {path}")
    finally:
        pending.unlink()


def publish_trec(out, name, run):
    lines = (f"{qid} Q0 {doc_id} {rank} {score:.8f} {name}\n"
             for qid, ranking in run.items()
             for rank, (doc_id, score) in enumerate(ranking, 1))
    publish(out / f"{name}.trec", "".join(lines).encode("utf-8"))


def bind_epoch_instance(instances, epoch, arm, observation):
    token = observation.get("service_instance")
    if not isinstance(token, str) or not INSTANCE_ID.fullmatch(token):
        raise EvaluationError(f"missing or invalid service instance for {arm} in epoch {epoch}")
    if instances.setdefault(epoch, {}).setdefault(arm, token) != token:
        raise EvaluationError(f"mixed service instances for {arm} in epoch {epoch}")


def summarize(out, protocol, names, qrels, captured, random_plan=None, instances=None):
    runs = {}
    arms = {}
    for name in names:
        run = {qid: data[name]["ranking"] for qid, (_, data) in captured.items()}
        runs[name] = run
        publish_trec(out, name, run)
        epochs = {}
        ce_pairs = 0
        for epoch, data in captured.values():
            epochs.setdefault(epoch, []).append((data[name], data["rrf"]))
            ce_pairs += data[name]["ce_pairs"]
        latency = {}
        for epoch, pairs in epochs.items():
            rows = [row for row, _ in pairs]
            latency[epoch] = {
                "n_queries": len(rows),
                "covers_all_queries": len(rows) == len(captured),
                "http_ms": distribution([row["http_ms"] for row in rows]),
                "paired_mean_http_delta_vs_rrf_ms": statistics.fmean(
                    row["http_ms"] - baseline["http_ms"] for row, baseline in pairs),
                "timings_ms": {
                    stage: distribution([row["timings_ms"][stage] for row in rows
                                         if stage in row["timings_ms"]])
                    for stage in (*STAGES, "rerank")
                    if any(stage in row["timings_ms"] for row in rows)
                },
            }
        arms[name] = {
            "quality": evaluate_run(qrels, run).to_dict(),
            "ce_pairs": ce_pairs,
            "mean_ce_pairs": ce_pairs / len(captured),
            "latency_by_epoch": latency,
        }
    comparisons = {
        name: {
            metric: paired_bootstrap(qrels, runs["rrf"], runs[name], metric=metric,
                                     samples=BOOTSTRAP_SAMPLES, seed=SEED).to_dict()
            for metric in ("ndcg", "recall", "mrr", "map")
        }
        for name in names if name != "rrf"
    }
    summary = {
        "protocol": protocol, "arms": arms, "paired_vs_rrf": comparisons,
        "explicit_constraints_queries": sum(
            data["rrf"]["explicit_constraints"] for _, data in captured.values()),
    }
    if "learned" in names:
        random_run = {qid: runs["ce40" if active else "rrf"][qid]
                      for qid, active in random_plan["random_actions"].items()}
        publish_trec(out, "matched-random-replay", random_run)
        diagnostics = learned_diagnostics(qrels, runs, random_plan, samples=BOOTSTRAP_SAMPLES, seed=SEED)
        publish(out / "learned-diagnostics.json",
                (json.dumps(diagnostics, ensure_ascii=False, indent=2, sort_keys=True,
                            allow_nan=False) + "\n").encode("utf-8"))
        summary["matched_random_replay"] = random_replay(random_plan, qrels, runs["learned"], random_run)
    if instances is not None:
        summary["service_instances_by_epoch"] = instances
    publish(out / "summary.json", (json.dumps(summary, ensure_ascii=False, indent=2,
                                             sort_keys=True) + "\n").encode("utf-8"))


def freeze_random(captured):
    capacity = {qid: row["ce40"]["ce_pairs"] for qid, (_, row) in captured.items()}
    selected = {qid: row["learned"]["ce_pairs"] > 0 for qid, (_, row) in captured.items()}
    random_actions = matched_random(selected, capacity, SEED)
    count = sum(capacity[qid] for qid, active in random_actions.items() if active)
    actual = sum(row["learned"]["ce_pairs"] for _, row in captured.values())
    if count != actual:
        raise EvaluationError("exact random CE budget disagrees with actual learned work")
    return {"seed": SEED, "ce_pairs": count, "capacity": capacity,
            "learned_actions": selected, "random_actions": random_actions}


def random_replay(plan, qrels, learned, random_run):
    intervals = {
        metric: paired_bootstrap(qrels, random_run, learned, metric=metric,
                                 samples=BOOTSTRAP_SAMPLES, seed=SEED).to_dict()
        for metric in ("ndcg", "recall", "mrr", "map")
    }
    return {"source": "observed_fixed_http_rankings", "ce_pairs": plan["ce_pairs"],
            "seed": plan["seed"], "quality": evaluate_run(qrels, random_run).to_dict(),
            "learned_minus_random": intervals}


def json_digest(value):
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False, allow_nan=False)
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def random_live_schedule(records, plan):
    """Counterbalance first position within each pair of planned actions."""
    starts = {}
    rng = random.Random(plan["seed"])
    for record in records:
        learned = plan["learned_actions"][record.qid]
        random_active = plan["random_actions"][record.qid]
        stratum = (learned, random_active)
        if stratum not in starts:
            starts[stratum] = bool(rng.getrandbits(1))
        control = "ce40" if random_active else "rrf"
        order = ("learned", control) if starts[stratum] else (control, "learned")
        starts[stratum] = not starts[stratum]
        yield record, control, order


def random_live_pair(client, arms, identity, record, control, order, reference):
    rows = {}
    for name in order:
        depth, base = arms[name]
        try:
            observed = search(client, base, depth, record, identity["common"]["retrieval"],
                              identity["rerankers"][name])
        except (OSError, ValueError, TypeError) as error:
            raise EvaluationError(f"live qid={record.qid!r}, arm={name}: {error}") from error
        expected = reference[name]
        if ([tuple(item) for item in observed["ranking"]]
                != [tuple(item) for item in expected["ranking"]]
                or any(observed[key] != expected[key]
                       for key in ("ce_pairs", "explicit_constraints", "features"))):
            raise EvaluationError(f"live qid={record.qid!r}, arm={name}: differs from frozen capture")
        rows["learned" if name == "learned" else "random"] = {
            key: observed[key] for key in ("ce_pairs", "http_ms", "service_instance") if key in observed
        }
    return {"control": control, "order": list(order), **rows}


def measure_random(out, client, arms, records, warmup, capture_protocol, captured, plan, *, replay_from=None):
    arms = {name: arms[name] for name in ("rrf", "ce40", "learned")}
    frozen_identity = capture_protocol["identities"]
    identity = {"common": frozen_identity["common"], "rerankers": {
        name: frozen_identity["rerankers"][name] for name in arms
    }}
    live_protocol = {
        "version": RANDOM_LIVE_VERSION, "request": REQUEST, "identities": identity,
        "capture_protocol_sha256": json_digest(capture_protocol),
        "capture_rows_sha256": json_digest(captured),
        "random_plan_sha256": json_digest(plan),
        "order": "alternate_within_action_strata_v1",
        "control": "client_dispatch_to_fixed_rrf_or_ce40",
        "latency": "http_open_through_body_read_ms",
    }
    if "instance_header" in capture_protocol:
        live_protocol["instance_header"] = capture_protocol["instance_header"]
    instances = {}
    schedule = list(random_live_schedule(records, plan))
    live_path = (replay_from or out) / "matched-random-live.sqlite3"
    if replay_from is not None and not live_path.is_file():
        raise EvaluationError("live random journal is missing")
    with open_journal(live_path, replay=replay_from is not None) as db:
        saved = journal(db, live_protocol, replay=replay_from is not None)
        if list(saved) != [record.qid for record, _, _ in schedule[:len(saved)]]:
            raise EvaluationError("live random journal contains unexpected query IDs or order")
        for record, control, order in schedule[:len(saved)]:
            epoch, row = saved[record.qid]
            reference = captured[record.qid][1]
            if (row.get("control") != control or row.get("order") != list(order)
                    or any(row.get(name, {}).get("ce_pairs") != reference[arm]["ce_pairs"]
                           for name, arm in (("learned", "learned"), ("random", control)))
                    or any(type(row.get(name, {}).get("http_ms")) not in (float, int)
                           or not math.isfinite(row[name]["http_ms"]) or row[name]["http_ms"] < 0
                           for name in ("learned", "random"))):
                raise EvaluationError(f"live random journal has invalid pair for qid={record.qid!r}")
            if "instance_header" in live_protocol:
                bind_epoch_instance(instances, epoch, "learned", row["learned"])
                bind_epoch_instance(instances, epoch, control, row["random"])
        if len(saved) < len(schedule):
            if replay_from is not None:
                raise EvaluationError("live random journal is incomplete")
            if check_services(client, arms) != identity:
                raise EvaluationError("live random service identity differs from frozen capture")
            for i, record in enumerate(warmup):
                capture_query(client, arms, record, i, list(arms), identity)
            epoch = uuid4().hex
            for i, (record, control, order) in enumerate(schedule[len(saved):], len(saved) + 1):
                pair = random_live_pair(client, arms, identity, record, control, order,
                                        captured[record.qid][1])
                if "instance_header" in live_protocol:
                    bind_epoch_instance(instances, epoch, "learned", pair["learned"])
                    bind_epoch_instance(instances, epoch, control, pair["random"])
                if i == len(schedule) and check_services(client, arms) != identity:
                    with db:
                        db.execute("INSERT INTO invalid_epochs VALUES (?, ?)",
                                   (epoch, "service identity changed before final live pair commit"))
                    raise EvaluationError("service identity changed during live random measurement")
                with db:
                    db.execute("INSERT INTO queries VALUES (?, ?, ?)", (
                        record.qid, epoch, json.dumps(pair, allow_nan=False),
                    ))
                saved[record.qid] = (epoch, pair)
                if i % 100 == 0:
                    print(f"Measured {i}/{len(schedule)} live random pairs", flush=True)

    pairs = [row for _, row in saved.values()]
    work = {name: Counter(row[name]["ce_pairs"] for row in pairs)
            for name in ("learned", "random")}
    total = {name: sum(count * n_queries for count, n_queries in counts.items())
             for name, counts in work.items()}
    if work["learned"] != work["random"] or any(count != plan["ce_pairs"] for count in total.values()):
        raise EvaluationError("observed live random and learned CE work does not match the plan")
    by_epoch = {}
    for epoch, row in saved.values():
        by_epoch.setdefault(epoch, []).append(row)
    epochs = {}
    for epoch, rows in by_epoch.items():
        epochs[epoch] = {
            "n_queries": len(rows), "covers_all_queries": len(rows) == len(records),
            "learned_first": sum(row["order"][0] == "learned" for row in rows),
            "ce_pairs": {name: sum(row[name]["ce_pairs"] for row in rows)
                         for name in ("learned", "random")},
            "http_ms": {name: distribution([row[name]["http_ms"] for row in rows])
                        for name in ("learned", "random")},
            "paired_learned_minus_random_http_ms": distribution([
                row["learned"]["http_ms"] - row["random"]["http_ms"] for row in rows
            ]),
        }
    path = out / "matched-random-live.json"
    result = {"protocol": live_protocol, "ce_pairs": total, "queries_by_ce_pair_count": work,
              "latency_by_epoch": epochs}
    if "instance_header" in live_protocol:
        result["service_instances_by_epoch"] = instances
    publish(path, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2,
                              allow_nan=False) + "\n").encode("utf-8"))
    return path


def replay_configuration(directory):
    try:
        with open_journal(directory / "capture.sqlite3", replay=True) as db:
            row = db.execute("SELECT body FROM protocol WHERE id=1").fetchone()
        protocol = json.loads(row[0]) if row else None
        depths = protocol["depths"]
    except (sqlite3.Error, TypeError, ValueError, KeyError) as error:
        raise EvaluationError("cannot read the source capture protocol") from error
    expected = {"rrf": 0, "ce28": 28, "ce40": 40, "learned": GATE_DEPTH}
    if (not isinstance(depths, dict) or "rrf" not in depths
            or not ({"ce28", "ce40"} & depths.keys()) or ("learned" in depths and "ce40" not in depths)
            or any(type(depth) is not int or expected.get(name) != depth for name, depth in depths.items())):
        raise EvaluationError("capture journal has invalid arm depths")
    return protocol, {name: (depth, None) for name, depth in expected.items() if name in depths}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline")
    cli.add_argument("--rerank", action="append", nargs=2, metavar=("DEPTH", "URL"))
    cli.add_argument("--replay-from", type=Path,
                     help="Recompute reports from completed read-only journals without HTTP")
    cli.add_argument("--learned", help="Optional digest-bound CE40 gate endpoint")
    cli.add_argument("--measure-random", action="store_true",
                     help="Measure paired HTTP latency against the frozen matched random plan")
    cli.add_argument("--require-instance-id", action="store_true",
                     help="Verify that every HTTP response in an epoch comes from the same service instances")
    cli.add_argument("--require-encoder-runtime", action="store_true",
                     help="Require matching query encoder runtime identity in all profiles")
    cli.add_argument("--fit-receipt", nargs=2, metavar=("JSON", "SHA256"),
                     help="Bind a learned capture to a frozen train_ce_gate receipt and digest")
    for name in ("queries", "warmup-queries", "qrels", "out"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args(argv)
    if args.replay_from is not None:
        if args.baseline or args.rerank or args.learned:
            cli.error("--replay-from reads arm configuration from the source journal")
        if args.out.resolve().is_relative_to(args.replay_from.resolve()):
            cli.error("replay output must be outside the source capture directory")
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            cli.error("replay output must be a new or empty directory")
        saved_protocol, arms = replay_configuration(args.replay_from)
        args.learned = "learned" in arms
        args.require_instance_id |= "instance_header" in saved_protocol
        args.require_encoder_runtime |= saved_protocol.get("require_encoder_runtime") is True
        args.measure_random |= (args.replay_from / "matched-random-live.sqlite3").is_file()
        if "fit_receipt_sha256" in saved_protocol and not args.fit_receipt:
            cli.error("this capture requires its original --fit-receipt")
    else:
        if not args.baseline or not args.rerank:
            cli.error("capture requires --baseline and at least one --rerank")
        arms = {"rrf": (0, endpoint(args.baseline))}
        for depth, url in sorted(args.rerank):
            if depth not in ("28", "40") or f"ce{depth}" in arms:
                cli.error("--rerank requires each of depth 28 and 40 at most once")
            arms[f"ce{depth}"] = (int(depth), endpoint(url))
        if args.learned:
            if "ce40" not in arms:
                cli.error("--learned requires a fixed CE40 endpoint")
            arms["learned"] = (GATE_DEPTH, endpoint(args.learned))
        if len({base for _, base in arms.values()}) != len(arms):
            cli.error("each arm requires its own endpoint")
    if args.measure_random and not args.learned:
        cli.error("--measure-random requires --learned")
    if args.fit_receipt and (not args.learned or not re.fullmatch(r"[a-f0-9]{64}", args.fit_receipt[1])):
        cli.error("--fit-receipt requires --learned and a lowercase SHA256 digest")

    inputs = {"queries": args.queries, "warmup_queries": args.warmup_queries, "qrels": args.qrels}
    fingerprints = {name: sha256(path) for name, path in inputs.items()}
    records = load_queries(args.queries)
    warmup = load_queries(args.warmup_queries)
    if any(sha256(inputs[name]) != fingerprints[name] for name in ("queries", "warmup_queries")):
        raise EvaluationError("query inputs changed while loading")
    query_ids = [record.qid for record in records]
    if any(any(char.isspace() for char in qid) for qid in query_ids):
        raise EvaluationError("query IDs must be compatible with TREC runs")
    if set(query_ids).intersection(record.qid for record in warmup) or (
        {normalized_query(record.query) for record in records}
        & {normalized_query(record.query) for record in warmup}
    ):
        raise EvaluationError("warmup and evaluated queries must be disjoint by ID and normalized text")

    replay = args.replay_from is not None
    client = None if replay else http_client(args.require_instance_id)
    order = list(arms)
    random.Random(SEED).shuffle(order)
    protocol = {
        "version": CAPTURE_VERSION, "request": REQUEST, "seed": SEED, "base_arm_order": order,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "depths": {name: depth for name, (depth, _) in arms.items()},
        "inputs_sha256": fingerprints,
        "query_count": len(records), "warmup_query_count": len(warmup),
    }
    if args.require_instance_id:
        protocol["instance_header"] = SEARCH_INSTANCE_HEADER
    instances = {} if args.require_instance_id else None
    args.out.mkdir(parents=True, exist_ok=True)
    with open_journal((args.replay_from or args.out) / "capture.sqlite3", replay=replay) as db:
        identities = completed_identity(db, len(records))
        if identities is None:
            if replay:
                raise EvaluationError("capture journal is incomplete")
            identities = check_services(client, arms)
        if args.require_encoder_runtime or args.fit_receipt:
            if not valid_query_encoder_runtime(identities["common"]["retrieval"].get("query_encoder_runtime")):
                raise EvaluationError("capture is missing a verified query encoder runtime identity")
            protocol["require_encoder_runtime"] = True
        if args.fit_receipt:
            fit_path, fit_sha256 = args.fit_receipt
            check_fit_receipt(Path(fit_path), fit_sha256, identities)
            protocol["fit_receipt_sha256"] = fit_sha256
        protocol["identities"] = identities
        captured = journal(db, protocol, replay=replay)
        if replay and len(captured) != len(records):
            raise EvaluationError("capture journal is incomplete")
        if list(captured) != query_ids[:len(captured)] or any(
            data.keys() != arms.keys() for _, data in captured.values()
        ):
            raise EvaluationError("journal contains an unexpected query or incomplete paired arms")
        if instances is not None:
            for epoch, data in captured.values():
                for arm, observation in data.items():
                    bind_epoch_instance(instances, epoch, arm, observation)
        if len(captured) < len(records):
            for i, record in enumerate(warmup):
                capture_query(client, arms, record, i, order, identities)
            epoch = uuid4().hex
            for i in range(len(captured), len(records)):
                record = records[i]
                data = capture_query(client, arms, record, i, order, identities)
                if instances is not None:
                    for arm, observation in data.items():
                        bind_epoch_instance(instances, epoch, arm, observation)
                if i + 1 == len(records) and check_services(client, arms) != identities:
                    with db:
                        db.execute("INSERT INTO invalid_epochs VALUES (?, ?)",
                                   (epoch, "service identity changed before final query commit"))
                    raise EvaluationError("service identity changed during capture")
                with db:
                    db.execute("INSERT INTO queries VALUES (?, ?, ?)", (
                        record.qid, epoch, json.dumps(data, allow_nan=False, ensure_ascii=False),
                    ))
                captured[record.qid] = (epoch, data)
                if (i + 1) % 100 == 0:
                    print(f"Captured {i + 1}/{len(records)} paired queries", flush=True)
    random_plan = None
    if "learned" in arms:
        random_plan = freeze_random(captured)
        publish(args.out / "matched-random-plan.json",
                (json.dumps(random_plan, sort_keys=True, ensure_ascii=False, indent=2)
                 + "\n").encode("utf-8"))
    if args.measure_random:
        print(measure_random(args.out, None if replay else http_client(args.require_instance_id),
                             arms, records, warmup, protocol, captured, random_plan,
                             replay_from=args.replay_from))
    if sha256(args.qrels) != fingerprints["qrels"]:
        raise EvaluationError("qrels changed during capture")
    qrels = load_qrels(args.qrels)
    if sha256(args.qrels) != fingerprints["qrels"]:
        raise EvaluationError("qrels changed while loading")
    missing_qrels = set(query_ids) - qrels.keys()
    if missing_qrels:
        raise EvaluationError(f"{len(missing_qrels)} evaluated queries have no qrels")
    summarize(args.out, protocol, arms, {qid: qrels[qid] for qid in query_ids}, captured,
              random_plan, instances)
    print(args.out / "summary.json")


if __name__ == "__main__":
    main()
