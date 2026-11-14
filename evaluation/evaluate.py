"""Evaluate the public search HTTP API against TREC qrels.

The evaluator is deliberately independent from the retrieval implementation:
queries are sent to ``POST /search`` using :mod:`urllib`, and only the JSON
response schema is consumed.  It can also score an existing TREC run without
contacting a server.

Typical usage::

    python3 -m evaluation.evaluate \
        --endpoint http://127.0.0.1:8000/search \
        --queries queries.tsv --qrels qrels.tsv \
        --output-run runs/bm25.trec

``queries.tsv`` contains ``query_id<TAB>query text``.  TREC qrels use the
usual ``query_id iteration doc_id relevance`` columns.  JSONL query records
with ``qid``/``query`` (or ``query_id``/``text``) are accepted as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EvaluationError(ValueError):
    """Input, transport, or response errors raised by the evaluator."""

    def __init__(self, message: str, *, transport: dict | None = None):
        super().__init__(message)
        self.transport = transport


class HTTPResponseError(EvaluationError):
    """A non-200 or invalid HTTP200 response with observed transport metadata."""

    def __init__(self, status: int, body: object, latency_ms: float, reason: str | None = None):
        self.status, self.body, self.latency_ms = status, body, latency_ms
        super().__init__(reason or f"HTTP {status}: {str(body)[:500]}")


DENSE_CONSTRAINT_ERROR = "dense 模式无法保证显式词法约束，请使用 lexical、hybrid 或 adaptive"
_REQUEST_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationError(f"JSON 字段重复: {key}")
        result[key] = value
    return result


def _load_unsupported_policy(path, queries_source, records, mode, scorer, expander):
    """Validate a frozen policy without importing or claiming to check the parser."""
    if queries_source is None:
        raise EvaluationError("unsupported policy 需要原始 queries_source 文件")
    try:
        raw = Path(path).read_bytes()
        value = json.loads(raw, object_pairs_hook=_unique_json_object)
        query_digest = _sha256(queries_source)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationError("无法读取 unsupported policy 或 queries_source") from error
    fields = {"schema_version", "queries_sha256", "preflight_parser_sha256", "request", "expected_unsupported"}
    if not isinstance(value, dict) or set(value) != fields or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise EvaluationError("unsupported policy schema/字段无效")
    for key in ("queries_sha256", "preflight_parser_sha256"):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None:
            raise EvaluationError(f"unsupported policy {key} 无效")
    if value["queries_sha256"] != query_digest or list(records) != load_queries(queries_source):
        raise EvaluationError("unsupported policy query SHA256/记录不匹配")
    expected_request = {"mode": "dense", "scorer": "bm25", "expander": "none"}
    if value["request"] != expected_request or {"mode": mode, "scorer": scorer, "expander": expander} != expected_request:
        raise EvaluationError("unsupported policy 请求参数不匹配")
    expected = value["expected_unsupported"]
    if not isinstance(expected, dict) or set(expected) != {"query_ids", "status", "body"}:
        raise EvaluationError("unsupported policy expected_unsupported 字段无效")
    if type(expected["status"]) is not int or expected["status"] != 400 or expected["body"] != {"error": DENSE_CONSTRAINT_ERROR}:
        raise EvaluationError("unsupported policy 只允许特定 dense constraints HTTP400")
    ids = expected["query_ids"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(qid, str) or not qid.strip() or qid != qid.strip() for qid in ids)
            or len(set(ids)) != len(ids) or not set(ids).issubset({record.qid for record in records})):
        raise EvaluationError("unsupported policy qid 缺失、重复或无效")
    return set(ids), {"sha256": hashlib.sha256(raw).hexdigest(), **value,
                      "parser_identity_scope": "preflight_provenance_only_runtime_parser_not_verified"}


def _atomic_json(path: Path, value: dict) -> None:
    pending = path.with_name(path.name + ".tmp")
    with pending.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    # On Windows a reader may hold the destination without delete sharing.
    # Retry the same atomic rename; keep the previous snapshot and pending
    # bytes intact if the sharing violation does not clear.
    for delay in (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, None):
        try:
            pending.replace(path)
            return
        except PermissionError as error:
            if delay is None or getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(delay)


def _append_request_event(path: Path, value: dict) -> None:
    # The caller exclusively creates this owned journal before any appends.
    # A crash may leave an incomplete final line; only complete lines count.
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


Qrels = dict[str, dict[str, float]]
Run = dict[str, list[tuple[str, float]]]


@dataclass(frozen=True)
class QueryRecord:
    qid: str
    query: str


@dataclass(frozen=True)
class QueryRouting:
    requested_mode: str
    executed_mode: str
    reason: str
    vector_attempted: bool
    vector_used: bool
    fallback: bool

    @classmethod
    def from_response(cls, value: object, mode: str, *, allow_adaptive_dense: bool = False) -> "QueryRouting":
        if not isinstance(value, dict):
            raise EvaluationError("搜索服务响应缺少 routing 对象")
        for key in ("requested_mode", "executed_mode", "reason"):
            if not isinstance(value.get(key), str):
                raise EvaluationError(f"搜索服务 routing.{key} 必须是字符串")
        for key in ("vector_attempted", "vector_used", "fallback"):
            if type(value.get(key)) is not bool:
                raise EvaluationError(f"搜索服务 routing.{key} 必须是布尔值")
        routing = cls(**{key: value[key] for key in cls.__dataclass_fields__})
        if routing.requested_mode != mode:
            raise EvaluationError("搜索服务 routing.requested_mode 与请求模式不一致")
        if routing.vector_used and not routing.vector_attempted:
            raise EvaluationError("搜索服务 routing 向量成功但未尝试")
        if routing.fallback and (mode != "adaptive" or not routing.vector_attempted
                                 or routing.vector_used or routing.executed_mode != "lexical"):
            raise EvaluationError("搜索服务 routing 降级标记不一致")
        expected_mode = "lexical"
        if routing.vector_used:
            expected_mode = "dense" if mode == "dense" else "hybrid"
            if allow_adaptive_dense and mode == "adaptive" and routing.executed_mode == "dense":
                expected_mode = "dense"
        if (routing.executed_mode != expected_mode
                or (mode in ("dense", "hybrid") and not routing.vector_used)
                or (mode == "lexical" and routing.vector_attempted)
                or (routing.vector_attempted and not routing.vector_used and not routing.fallback)):
            raise EvaluationError("搜索服务 routing 执行模式或向量标记不一致")
        return routing


@dataclass(frozen=True)
class RoutingSummary:
    n_queries: int
    vector_attempts: int
    vector_successes: int
    fallbacks: int
    vector_attempt_rate: float
    vector_success_rate: float | None  # successes / attempts; null when never attempted
    fallback_rate: float             # fallbacks / all observed queries


@dataclass(frozen=True)
class QueryMetrics:
    qid: str
    ndcg: float
    recall: float
    mrr: float
    map_at_100: float
    latency_ms: float | None = None
    retrieved: int = 0
    relevant: int = 0
    routing: QueryRouting | None = None


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate metrics and per-query records returned by ``evaluate_run``."""

    n_queries: int
    queries_with_relevance: int
    ndcg_at_10: float
    recall_at_10: float
    mrr_at_10: float
    map_at_100: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    per_query: tuple[QueryMetrics, ...] = ()
    routing: RoutingSummary | None = None
    requests: dict | None = None

    def to_dict(self, include_per_query: bool = False) -> dict:
        value = asdict(self)
        if not include_per_query:
            value.pop("per_query", None)
        else:
            value["per_query"] = [asdict(item) for item in self.per_query]
        return value


def _ensure_nonempty(value: str, label: str) -> str:
    value = value.strip()
    if not value:
        raise EvaluationError(f"{label} 不能为空")
    return value


def _parse_rel(raw: str, line_no: int, path: Path) -> float:
    try:
        relevance = float(raw)
    except ValueError as error:
        raise EvaluationError(f"{path}:{line_no}: relevance 不是数字: {raw!r}") from error
    if not math.isfinite(relevance) or relevance < 0:
        raise EvaluationError(f"{path}:{line_no}: relevance 必须是非负有限数字")
    return relevance


def load_qrels(path: str | Path) -> Qrels:
    """Load TREC qrels (``qid iter docid rel``) into a mapping.

    A three-column ``qid docid rel`` shorthand is accepted for small fixtures.
    Duplicate qrel rows are rejected when they disagree and otherwise ignored.
    """

    path = Path(path)
    qrels: Qrels = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"无法读取 qrels: {path}: {error}") from error
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 4:
            qid, docid, relevance_raw = fields[0], fields[2], fields[3]
        elif len(fields) == 3:
            qid, docid, relevance_raw = fields
        else:
            raise EvaluationError(f"{path}:{line_no}: qrels 至少需要 3 列")
        qid = _ensure_nonempty(qid, "qid")
        docid = _ensure_nonempty(docid, "docid")
        relevance = _parse_rel(relevance_raw, line_no, path)
        docs = qrels.setdefault(qid, {})
        previous = docs.get(docid)
        if previous is not None and previous != relevance:
            raise EvaluationError(f"{path}:{line_no}: qrel 重复且 relevance 不一致")
        docs[docid] = relevance
    return qrels


def _query_from_json(value: object, line_no: int, path: Path) -> QueryRecord:
    if not isinstance(value, dict):
        raise EvaluationError(f"{path}:{line_no}: JSON 查询记录必须是对象")
    qid = value.get("qid", value.get("query_id", value.get("id")))
    query = value.get("query", value.get("text"))
    if not isinstance(qid, (str, int)) or not isinstance(query, str):
        raise EvaluationError(f"{path}:{line_no}: JSON 查询需要 qid/query 字段")
    return QueryRecord(_ensure_nonempty(str(qid), "qid"), _ensure_nonempty(query, "query"))


def load_queries(path: str | Path) -> list[QueryRecord]:
    """Load TSV/plain-text or JSONL query records.

    Text records use ``qid<TAB>query``; when no tab exists, the first
    whitespace-delimited token is treated as qid.  Query order is preserved
    and duplicate qids are rejected to avoid ambiguous run rows.
    """

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"无法读取 queries: {path}: {error}") from error
    records: list[QueryRecord] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvaluationError(f"{path}:{line_no}: JSON 查询记录格式错误") from error
            record = _query_from_json(value, line_no, path)
        elif "\t" in line:
            qid, query = line.split("\t", 1)
            record = QueryRecord(_ensure_nonempty(qid, "qid"), _ensure_nonempty(query, "query"))
        else:
            fields = line.split(maxsplit=1)
            if len(fields) != 2:
                raise EvaluationError(f"{path}:{line_no}: 查询需要 qid 和 query")
            record = QueryRecord(_ensure_nonempty(fields[0], "qid"), _ensure_nonempty(fields[1], "query"))
        if record.qid in seen:
            raise EvaluationError(f"{path}:{line_no}: qid 重复: {record.qid!r}")
        seen.add(record.qid)
        records.append(record)
    if not records:
        raise EvaluationError(f"queries 为空: {path}")
    return records


def _parse_run_row(fields: list[str], line_no: int, path: Path) -> tuple[str, str, int, float]:
    # TREC: qid Q0 docid rank score tag.  A compact qid docid rank score form
    # is also accepted for hand-written fixtures.
    if len(fields) >= 6:
        qid, docid, rank_raw, score_raw = fields[0], fields[2], fields[3], fields[4]
    elif len(fields) == 5:
        qid, docid, rank_raw, score_raw = fields[0], fields[1], fields[2], fields[3]
    elif len(fields) == 4:
        qid, docid, rank_raw, score_raw = fields
    else:
        raise EvaluationError(f"{path}:{line_no}: run 至少需要 4 列")
    qid = _ensure_nonempty(qid, "qid")
    docid = _ensure_nonempty(docid, "docid")
    try:
        rank = int(rank_raw)
        score = float(score_raw)
    except ValueError as error:
        raise EvaluationError(f"{path}:{line_no}: run rank/score 格式错误") from error
    if rank <= 0 or not math.isfinite(score):
        raise EvaluationError(f"{path}:{line_no}: run rank 必须为正且 score 有限")
    return qid, docid, rank, score


def load_run(path: str | Path) -> Run:
    """Load a TREC run, sorting each query by rank and removing duplicates."""

    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvaluationError(f"无法读取 run: {path}: {error}") from error
    rows: dict[str, list[tuple[int, str, float]]] = {}
    for line_no, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        qid, docid, rank, score = _parse_run_row(line.split(), line_no, path)
        rows.setdefault(qid, []).append((rank, docid, score))
    result: Run = {}
    for qid, values in rows.items():
        seen: set[str] = set()
        ordered: list[tuple[str, float]] = []
        for _, docid, score in sorted(values, key=lambda item: (item[0], item[1])):
            if docid in seen:
                continue
            seen.add(docid)
            ordered.append((docid, score))
        result[qid] = ordered
    return result


def _metric_for_query(qrels: Mapping[str, float], retrieved: Sequence[str], k: int,
                      map_k: int = 100) -> tuple[float, float, float, float, int]:
    if type(k) is not int or k <= 0 or type(map_k) is not int or map_k <= 0:
        raise EvaluationError("评测 cutoff 必须是正整数")
    top = list(retrieved[:k])
    relevant = {docid for docid, rel in qrels.items() if rel > 0}
    dcg = 0.0
    for rank, docid in enumerate(top, 1):
        rel = qrels.get(docid, 0.0)
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(rank + 1)
    ideal = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
    idcg = sum((2.0 ** rel - 1.0) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    ndcg = dcg / idcg if idcg else 0.0
    recall = len(relevant.intersection(top)) / len(relevant) if relevant else 0.0
    mrr = next((1.0 / rank for rank, docid in enumerate(top, 1) if docid in relevant), 0.0)
    ap_top = list(retrieved[:map_k])
    hits = 0
    precision_sum = 0.0
    for rank, docid in enumerate(ap_top, 1):
        if docid in relevant:
            hits += 1
            precision_sum += hits / rank
    average_precision = precision_sum / len(relevant) if relevant else 0.0
    return ndcg, recall, mrr, average_precision, len(relevant)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def evaluate_run(qrels: Mapping[str, Mapping[str, float]], run: Mapping[str, Sequence[tuple[str, float] | str]], k: int = 10, latencies: Mapping[str, float] | None = None, map_k: int = 100, routing: Mapping[str, QueryRouting] | None = None) -> EvaluationSummary:
    """Compute ranking metrics for a TREC run.

    ``run`` values may be ``(doc_id, score)`` pairs or plain doc IDs.  Queries
    are taken from the union of qrels and run IDs, so missing results receive
    zero scores. Latency values are milliseconds keyed by qid. Optional HTTP
    routing observations are retained per query and summarized separately;
    standalone TREC runs have no routing telemetry (null, not zero failures).
    """

    if type(k) is not int or k <= 0 or type(map_k) is not int or map_k <= 0:
        raise EvaluationError("评测 cutoff 必须是正整数")
    qids = list(dict.fromkeys([*qrels.keys(), *run.keys()]))
    if not qids:
        raise EvaluationError("qrels 和 run 均为空")
    metrics: list[QueryMetrics] = []
    for qid in qids:
        rows = run.get(qid, ())
        docids = [row[0] if isinstance(row, (tuple, list)) else row for row in rows]
        ndcg, recall, mrr, average_precision, relevant_count = _metric_for_query(
            qrels.get(qid, {}), docids, k, map_k)
        latency = None if latencies is None else latencies.get(qid)
        metrics.append(QueryMetrics(qid, ndcg, recall, mrr, average_precision,
                                    latency, len(docids), relevant_count,
                                    None if routing is None else routing.get(qid)))
    latency_values = [m.latency_ms for m in metrics if m.latency_ms is not None]
    relevant_queries = sum(m.relevant > 0 for m in metrics)
    observed_routing = [m.routing for m in metrics if m.routing is not None]
    routing_summary = None
    if observed_routing:
        attempts = sum(item.vector_attempted for item in observed_routing)
        successes = sum(item.vector_used for item in observed_routing)
        fallbacks = sum(item.fallback for item in observed_routing)
        count = len(observed_routing)
        routing_summary = RoutingSummary(
            count, attempts, successes, fallbacks, attempts / count,
            successes / attempts if attempts else None, fallbacks / count,
        )
    return EvaluationSummary(
        n_queries=len(metrics),
        queries_with_relevance=relevant_queries,
        ndcg_at_10=sum(m.ndcg for m in metrics) / len(metrics),
        recall_at_10=sum(m.recall for m in metrics) / len(metrics),
        mrr_at_10=sum(m.mrr for m in metrics) / len(metrics),
        map_at_100=sum(m.map_at_100 for m in metrics) / len(metrics),
        mean_latency_ms=statistics.fmean(latency_values) if latency_values else None,
        p50_latency_ms=_percentile(latency_values, 0.50),
        p95_latency_ms=_percentile(latency_values, 0.95),
        per_query=tuple(metrics),
        routing=routing_summary,
    )


def _endpoint_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if not endpoint:
        raise EvaluationError("endpoint 不能为空")
    return endpoint if endpoint.endswith("/search") else endpoint + "/search"


def _request_search(endpoint: str, query: str, *, k: int, mode: str, scorer: str,
                    expander: str, timeout: float, request_tag: str | None = None,
                    allow_adaptive_dense: bool = False,
                    ) -> tuple[list[tuple[str, float]], float, QueryRouting]:
    body = {
        "query": query,
        "k": k,
        "mode": mode,
        "scorer": scorer,
        "expander": expander,
    }
    if request_tag is not None:
        body["run_tag"] = request_tag
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = Request(_endpoint_url(endpoint), payload, {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }, method="POST")
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except HTTPError as error:
        try:
            raw = error.read()
        finally:
            error.close()
        try:
            error_body = json.loads(raw, object_pairs_hook=_unique_json_object)
        except (ValueError, UnicodeError):
            error_body = None
        raise HTTPResponseError(error.code, error_body,
                                (time.perf_counter() - started) * 1000.0) from error
    except (URLError, TimeoutError, OSError) as error:
        cause = error.reason if isinstance(error, URLError) else error
        detail = {"kind": "timeout" if isinstance(cause, TimeoutError) else "transport",
                  "cause_type": type(cause).__name__ if isinstance(cause, BaseException) else "URLError",
                  "errno": getattr(cause, "errno", None) if type(getattr(cause, "errno", None)) is int else None,
                  "elapsed_ms": (time.perf_counter() - started) * 1000.0}
        # Exception text may contain endpoint credentials. Keep structured cause
        # and elapsed time, without putting that text in logs or the CLI error.
        raise EvaluationError("请求搜索服务失败", transport=detail) from None
    latency_ms = (time.perf_counter() - started) * 1000.0
    if status != 200:
        try:
            error_body = json.loads(body, object_pairs_hook=_unique_json_object)
        except (ValueError, UnicodeError):
            error_body = None
        raise HTTPResponseError(status, error_body, latency_ms)
    try:
        results, routing = _decode_search_response(body, mode, allow_adaptive_dense=allow_adaptive_dense)
    except EvaluationError as error:
        raise HTTPResponseError(200, None, latency_ms, str(error)) from error
    return results, latency_ms, routing


def _decode_search_response(body: bytes, mode: str, *, allow_adaptive_dense: bool = False,
                            allow_hybrid_rerank: bool = False) -> tuple[list[tuple[str, float]], QueryRouting]:
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError) as error:
        raise EvaluationError("搜索服务返回的不是 JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise EvaluationError("搜索服务响应缺少 results 数组")
    timings = value.get("timings_ms")
    if (mode == "hybrid" and not allow_hybrid_rerank
            and ("rerank" in value or (isinstance(timings, dict) and "rerank" in timings))):
        raise EvaluationError("Use evaluation.ce_http for a CE-enabled hybrid response")
    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for position, item in enumerate(value["results"], 1):
        if not isinstance(item, dict) or not isinstance(item.get("doc_id"), str):
            raise EvaluationError("搜索服务 results 含无效 doc_id")
        docid = item["doc_id"]
        if docid in seen:
            continue
        seen.add(docid)
        raw_score = item.get("score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as error:
            raise EvaluationError("搜索服务 results 含无效 score") from error
        if not math.isfinite(score):
            raise EvaluationError("搜索服务 results 含非有限 score")
        results.append((docid, score))
    routing = QueryRouting.from_response(value.get("routing"), mode,
                                         allow_adaptive_dense=allow_adaptive_dense)
    return results, routing


def _write_trec_run(path: str | Path, qid: str, rows: Sequence[tuple[str, float]], tag: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This helper is used through the batch writer below; kept separate to
    # make its TREC formatting easy to test.
    with path.open("a", encoding="utf-8") as handle:
        for rank, (docid, score) in enumerate(rows, 1):
            handle.write(f"{qid} Q0 {docid} {rank} {score:.8f} {tag}\n")


def evaluate_http(endpoint: str, queries: Iterable[QueryRecord], qrels: Mapping[str, Mapping[str, float]], *, k: int = 10, mode: str = "lexical", scorer: str = "bm25", expander: str = "none", timeout: float = 10.0, run_output: str | Path | None = None, tag: str = "http-search", map_k: int = 100, fail_on_fallback: bool = False, unsupported_policy: str | Path | None = None, queries_source: str | Path | None = None, request_output: str | Path | None = None, request_tag: str | None = None, allow_adaptive_dense: bool = False) -> EvaluationSummary:
    """Query a public HTTP endpoint and evaluate its responses.

    Retrieve ``max(k, map_k)`` results, then apply each metric cutoff separately.
    When ``run_output`` is supplied, a TREC run is written while requests are
    made. Existing output paths are rejected. Routing
    outcomes are always reported; ``fail_on_fallback`` rejects degraded runs.
    An optional ``request_tag`` correlates the request journal with server logs.
    """

    if type(k) is not int or k <= 0 or type(map_k) is not int or map_k <= 0:
        raise EvaluationError("评测 cutoff 必须是正整数")
    retrieval_depth = max(k, map_k)
    if retrieval_depth > 200:
        raise EvaluationError("HTTP 召回深度 max(k, map_k) 不能超过 API 上限 200")
    if timeout <= 0 or not math.isfinite(timeout):
        raise EvaluationError("timeout 必须是正数")
    if (mode not in ("lexical", "dense", "hybrid", "adaptive")
            or scorer not in ("bm25", "tfidf") or expander not in ("none", "rm3", "slm")
            or mode != "lexical" and (scorer != "bm25" or expander != "none")):
        raise EvaluationError("HTTP 请求 mode/scorer/expander 参数无效")
    records = list(queries)
    if not records:
        raise EvaluationError("queries 为空")
    if len({record.qid for record in records}) != len(records):
        raise EvaluationError("queries qid 重复")
    if not tag or any(ch.isspace() for ch in tag):
        raise EvaluationError("run tag 必须是非空单词")
    if request_tag is not None and (
        not isinstance(request_tag, str) or not _REQUEST_TAG.fullmatch(request_tag)
    ):
        raise EvaluationError("HTTP request tag 格式无效")
    expected_ids, policy_metadata = set(), None
    if unsupported_policy is not None:
        expected_ids, policy_metadata = _load_unsupported_policy(
            unsupported_policy, queries_source, records, mode, scorer, expander)
    output_path = None if run_output is None else Path(run_output)
    outcome_path = (Path(request_output) if request_output is not None else
                    None if output_path is None else Path(str(output_path) + ".requests.json"))
    if unsupported_policy is not None and outcome_path is None:
        raise EvaluationError("unsupported policy 需要 run_output 或 request_output 持久记录")
    journal_path = None if outcome_path is None else Path(str(outcome_path) + ".events.jsonl")
    paths = [path for path in (output_path, outcome_path, journal_path) if path is not None]
    reserved_paths = paths + [path.with_name(path.name + ".tmp") for path in paths]
    if len({path.resolve() for path in reserved_paths}) != len(reserved_paths):
        raise EvaluationError("run/request output 路径不能相同")
    if any(path.exists() or path.with_name(path.name + ".tmp").exists() for path in paths):
        raise EvaluationError("output 已存在，拒绝覆盖旧 run/request 证据")
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema_version": 1, "status": "INCOMPLETE", "requested_queries": len(records),
             "attempted_requests": 0, "http_200": 0, "expected_unsupported": 0,
             "unexpected_errors": 0, "http_success_coverage": 0.0,
             "protocol_accounted_coverage": 0.0, "policy": policy_metadata,
             "request": {"mode": mode, "scorer": scorer, "expander": expander, "k": retrieval_depth},
             "query_ids": [record.qid for record in records], "outcomes": [],
             "last_event_sequence": 0, "inflight_outcome": None, "last_outcome": None,
             "journal": None if journal_path is None else {"file": journal_path.name}}
    if allow_adaptive_dense:
        if mode != "adaptive":
            raise EvaluationError("allow_adaptive_dense 仅用于 adaptive 评测")
        state["request"]["allow_adaptive_dense"] = True
    if request_tag is not None:
        state["request"]["run_tag"] = request_tag
    state["query_universe_sha256"] = hashlib.sha256(json.dumps(
        state["query_ids"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    def save_state(event=None, outcome=None):
        state["http_success_coverage"] = state["http_200"] / len(records)
        state["protocol_accounted_coverage"] = (state["http_200"] + state["expected_unsupported"]) / len(records)
        if event is not None:
            state["last_event_sequence"] += 1
            if journal_path is not None:
                _append_request_event(journal_path, {"event": event,
                    "sequence": state["last_event_sequence"], "attempt": state["attempted_requests"],
                    "outcome": outcome})
            state["inflight_outcome"] = outcome if event == "BEGIN" else None
            if event == "OUTCOME":
                state["last_outcome"] = outcome
        if outcome_path is not None:
            if state["status"] == "COMPLETE":
                _atomic_json(outcome_path, state)
            else:
                snapshot = {key: value for key, value in state.items()
                            if key not in {"query_ids", "outcomes", "policy"}}
                snapshot["policy"] = None if policy_metadata is None else {
                    key: policy_metadata[key] for key in ("sha256", "queries_sha256",
                    "preflight_parser_sha256", "parser_identity_scope")}
                _atomic_json(outcome_path, snapshot)

    # Publish INCOMPLETE before touching the run. Abrupt interruptions cannot
    # leave a successful-looking sidecar beside a partial TREC file.
    if outcome_path is not None:
        outcome_path.open("x", encoding="utf-8").close()
        journal_path.open("x", encoding="utf-8").close()
        _append_request_event(journal_path, {"event": "START", "sequence": 0,
            "query_ids": state["query_ids"], "query_universe_sha256": state["query_universe_sha256"],
            "policy": policy_metadata, "request": state["request"]})
    save_state()
    if output_path is not None:
        output_path.open("x", encoding="utf-8").close()
    run: Run = {}
    latencies: dict[str, float] = {}
    routing: dict[str, QueryRouting] = {}
    for record in records:
        state["attempted_requests"] += 1
        outcome = {"qid": record.qid, "status": "INCOMPLETE", "http_status": None,
                   "latency_ms": None, "routing": None}
        state["outcomes"].append(outcome)
        save_state("BEGIN", outcome)
        expected_rejection = False
        try:
            try:
                rows, latency, observation = _request_search(
                    endpoint, record.query, k=retrieval_depth, mode=mode,
                    scorer=scorer, expander=expander, timeout=timeout, request_tag=request_tag,
                    allow_adaptive_dense=allow_adaptive_dense,
                )
            except HTTPResponseError as error:
                outcome.update(http_status=error.status, latency_ms=error.latency_ms,
                               body=error.body)
                if (record.qid not in expected_ids or error.status != 400
                        or error.body != {"error": DENSE_CONSTRAINT_ERROR}):
                    raise
                expected_rejection = True
            if not expected_rejection:
                outcome.update(http_status=200, latency_ms=latency, routing=asdict(observation))
                if record.qid in expected_ids:
                    raise EvaluationError(f"policy 预声明查询 {record.qid!r} 意外返回 HTTP200")
                if fail_on_fallback and observation.fallback:
                    raise EvaluationError(f"查询 {record.qid!r} 发生向量降级，拒绝该评测 run")
        except Exception as error:
            state["unexpected_errors"] += 1
            outcome.update(status="UNEXPECTED_ERROR", error_type=type(error).__name__)
            transport = getattr(error, "transport", None) if isinstance(error, EvaluationError) else None
            if transport is not None:
                outcome["latency_ms"] = transport["elapsed_ms"]
                outcome["transport_error"] = dict(transport)
            save_state("OUTCOME", outcome)
            raise
        if expected_rejection:
            run[record.qid] = []
            state["expected_unsupported"] += 1
            outcome["status"] = "EXPECTED_UNSUPPORTED"
            save_state("OUTCOME", outcome)
            continue
        run[record.qid] = rows
        latencies[record.qid] = latency
        routing[record.qid] = observation
        if run_output is not None:
            _write_trec_run(run_output, record.qid, rows, tag)
        state["http_200"] += 1
        outcome["status"] = "SUCCESS"
        save_state("OUTCOME", outcome)
    # A qrels file often contains more than one split.  HTTP evaluation must
    # score exactly the queries sent in this invocation, while ``evaluate_run``
    # intentionally uses the union for standalone run scoring.
    selected_qrels = {record.qid: qrels.get(record.qid, {}) for record in records}
    summary = evaluate_run(selected_qrels, run, k=k, latencies=latencies, map_k=map_k,
                           routing=routing)
    state["status"] = "COMPLETE"
    state["routing_observed_queries"] = len(routing)
    state["success_latency_scope"] = "validated_HTTP200_only"
    if journal_path is not None:
        state["journal"]["sha256"] = _sha256(journal_path)
    save_state()
    return replace(summary, requests=state)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 HTTP /search 评测 TREC qrels")
    parser.add_argument("--qrels", required=True, type=Path, help="TREC qrels 文件")
    parser.add_argument("--queries", type=Path, help="qid<TAB>query 或 JSONL；HTTP 模式必需")
    parser.add_argument("--endpoint", help="搜索服务 URL（如 http://127.0.0.1:8000/search）")
    parser.add_argument("--run", type=Path, help="已有 TREC run；提供后不请求 HTTP")
    parser.add_argument("--output-run", type=Path, help="HTTP 模式下写出的 TREC run")
    parser.add_argument("--output-json", type=Path, help="将汇总 JSON 写入文件")
    parser.add_argument("--output-requests", type=Path, help="持久 request outcomes；默认 OUTPUT_RUN.requests.json")
    parser.add_argument("--unsupported-policy", type=Path, help="显式 SHA 绑定的 expected dense-constraints HTTP400 policy")
    parser.add_argument("--include-per-query", action="store_true", help="在 JSON 输出中包含逐查询指标")
    parser.add_argument("--k", type=int, default=10, help="nDCG/Recall/MRR cutoff，默认 10")
    parser.add_argument("--map-k", type=int, default=100, help="MAP cutoff，默认 100")
    parser.add_argument("--mode", choices=("lexical", "dense", "hybrid", "adaptive"), default="lexical")
    parser.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--expander", choices=("none", "rm3", "slm"), default="none")
    parser.add_argument("--timeout", type=float, default=10.0, help="单次 HTTP 超时秒数")
    parser.add_argument("--tag", default="http-search", help="TREC run tag")
    parser.add_argument("--request-tag", help="HTTP run_tag used to correlate request and server logs")
    parser.add_argument("--fail-on-fallback", action="store_true",
                        help="HTTP 评测遇到 adaptive 向量降级时返回非零状态")
    parser.add_argument("--allow-adaptive-dense", action="store_true",
                        help="允许实验 Router 选择 adaptive 的纯向量动作")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        output_paths = [path for path in (args.output_json, args.output_run,
                        args.output_requests if args.output_requests is not None else
                        None if args.output_run is None else Path(str(args.output_run) + ".requests.json"))
                        if path is not None]
        outcome_path = (args.output_requests if args.output_requests is not None else
                        None if args.output_run is None else Path(str(args.output_run) + ".requests.json"))
        if outcome_path is not None:
            output_paths.append(Path(str(outcome_path) + ".events.jsonl"))
        reserved_paths = output_paths + [path.with_name(path.name + ".tmp") for path in output_paths]
        if len({path.resolve() for path in reserved_paths}) != len(reserved_paths):
            raise EvaluationError("output 路径不能相同")
        if any(path.exists() or path.with_name(path.name + ".tmp").exists() for path in output_paths):
            raise EvaluationError("output 已存在，拒绝覆盖旧评测证据")
        qrels = load_qrels(args.qrels)
        if args.run is not None:
            if (args.endpoint or args.queries or args.output_run or args.fail_on_fallback or args.allow_adaptive_dense
                    or args.unsupported_policy or args.output_requests or args.request_tag is not None):
                raise EvaluationError("--run 不能与 HTTP 选项同时使用")
            summary = evaluate_run(qrels, load_run(args.run), k=args.k, map_k=args.map_k)
        else:
            if not args.endpoint or not args.queries:
                raise EvaluationError("HTTP 模式需要同时提供 --endpoint 和 --queries")
            summary = evaluate_http(
                args.endpoint, load_queries(args.queries), qrels, k=args.k, mode=args.mode,
                scorer=args.scorer, expander=args.expander, timeout=args.timeout,
                run_output=args.output_run, tag=args.tag, map_k=args.map_k,
                fail_on_fallback=args.fail_on_fallback, unsupported_policy=args.unsupported_policy,
                queries_source=args.queries, request_output=args.output_requests,
                request_tag=args.request_tag, allow_adaptive_dense=args.allow_adaptive_dense,
            )
        output = json.dumps(summary.to_dict(args.include_per_query), ensure_ascii=False, indent=2)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("x", encoding="utf-8") as handle:
                handle.write(output + "\n")
        print(output)
    except EvaluationError as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
