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
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EvaluationError(ValueError):
    """Input, transport, or response errors raised by the evaluator."""


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
            if mode=='adaptive' and allow_adaptive_dense and routing.executed_mode=='dense':
                expected_mode='dense'
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


def _request_search(endpoint: str, query: str, *, k: int, mode: str, scorer: str, expander: str, timeout: float) -> tuple[list[tuple[str, float]], float, QueryRouting]:
    payload = json.dumps({
        "query": query,
        "k": k,
        "mode": mode,
        "scorer": scorer,
        "expander": expander,
    }, ensure_ascii=False).encode("utf-8")
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
        body = error.read().decode("utf-8", errors="replace")
        raise EvaluationError(f"HTTP {error.code}: {body[:500]}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise EvaluationError(f"请求搜索服务失败: {error}") from error
    latency_ms = (time.perf_counter() - started) * 1000.0
    if status < 200 or status >= 300:
        raise EvaluationError(f"搜索服务返回 HTTP {status}")
    try:
        value = json.loads(body)
    except json.JSONDecodeError as error:
        raise EvaluationError("搜索服务返回的不是 JSON") from error
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise EvaluationError("搜索服务响应缺少 results 数组")
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
    routing = QueryRouting.from_response(value.get("routing"), mode)
    return results, latency_ms, routing


def _write_trec_run(path: str | Path, qid: str, rows: Sequence[tuple[str, float]], tag: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # This helper is used through the batch writer below; kept separate to
    # make its TREC formatting easy to test.
    with path.open("a", encoding="utf-8") as handle:
        for rank, (docid, score) in enumerate(rows, 1):
            handle.write(f"{qid} Q0 {docid} {rank} {score:.8f} {tag}\n")


def evaluate_http(endpoint: str, queries: Iterable[QueryRecord], qrels: Mapping[str, Mapping[str, float]], *, k: int = 10, mode: str = "lexical", scorer: str = "bm25", expander: str = "none", timeout: float = 10.0, run_output: str | Path | None = None, tag: str = "http-search", map_k: int = 100, fail_on_fallback: bool = False) -> EvaluationSummary:
    """Query a public HTTP endpoint and evaluate its responses.

    Retrieve ``max(k, map_k)`` results, then apply each metric cutoff separately.
    When ``run_output`` is supplied, a TREC run is written while requests are
    made. The output is replaced at the beginning of each evaluation. Routing
    outcomes are always reported; ``fail_on_fallback`` rejects degraded runs.
    """

    if type(k) is not int or k <= 0 or type(map_k) is not int or map_k <= 0:
        raise EvaluationError("评测 cutoff 必须是正整数")
    retrieval_depth = max(k, map_k)
    if retrieval_depth > 200:
        raise EvaluationError("HTTP 召回深度 max(k, map_k) 不能超过 API 上限 200")
    if timeout <= 0 or not math.isfinite(timeout):
        raise EvaluationError("timeout 必须是正数")
    records = list(queries)
    if not records:
        raise EvaluationError("queries 为空")
    if not tag or any(ch.isspace() for ch in tag):
        raise EvaluationError("run tag 必须是非空单词")
    if run_output is not None:
        output_path = Path(run_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    run: Run = {}
    latencies: dict[str, float] = {}
    routing: dict[str, QueryRouting] = {}
    for record in records:
        rows, latency, observation = _request_search(
            endpoint, record.query, k=retrieval_depth, mode=mode,
            scorer=scorer, expander=expander, timeout=timeout,
        )
        if fail_on_fallback and observation.fallback:
            raise EvaluationError(f"查询 {record.qid!r} 发生向量降级，拒绝该评测 run")
        run[record.qid] = rows
        latencies[record.qid] = latency
        routing[record.qid] = observation
        if run_output is not None:
            _write_trec_run(run_output, record.qid, rows, tag)
    # A qrels file often contains more than one split.  HTTP evaluation must
    # score exactly the queries sent in this invocation, while ``evaluate_run``
    # intentionally uses the union for standalone run scoring.
    selected_qrels = {record.qid: qrels.get(record.qid, {}) for record in records}
    return evaluate_run(selected_qrels, run, k=k, latencies=latencies, map_k=map_k,
                        routing=routing)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过 HTTP /search 评测 TREC qrels")
    parser.add_argument("--qrels", required=True, type=Path, help="TREC qrels 文件")
    parser.add_argument("--queries", type=Path, help="qid<TAB>query 或 JSONL；HTTP 模式必需")
    parser.add_argument("--endpoint", help="搜索服务 URL（如 http://127.0.0.1:8000/search）")
    parser.add_argument("--run", type=Path, help="已有 TREC run；提供后不请求 HTTP")
    parser.add_argument("--output-run", type=Path, help="HTTP 模式下写出的 TREC run")
    parser.add_argument("--output-json", type=Path, help="将汇总 JSON 写入文件")
    parser.add_argument("--include-per-query", action="store_true", help="在 JSON 输出中包含逐查询指标")
    parser.add_argument("--k", type=int, default=10, help="nDCG/Recall/MRR cutoff，默认 10")
    parser.add_argument("--map-k", type=int, default=100, help="MAP cutoff，默认 100")
    parser.add_argument("--mode", choices=("lexical", "dense", "hybrid", "adaptive"), default="lexical")
    parser.add_argument("--scorer", choices=("bm25", "tfidf"), default="bm25")
    parser.add_argument("--expander", choices=("none", "rm3", "slm"), default="none")
    parser.add_argument("--timeout", type=float, default=10.0, help="单次 HTTP 超时秒数")
    parser.add_argument("--tag", default="http-search", help="TREC run tag")
    parser.add_argument("--fail-on-fallback", action="store_true",
                        help="HTTP 评测遇到 adaptive 向量降级时返回非零状态")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        qrels = load_qrels(args.qrels)
        if args.run is not None:
            if args.endpoint or args.queries or args.output_run or args.fail_on_fallback:
                raise EvaluationError("--run 不能与 HTTP 选项同时使用")
            summary = evaluate_run(qrels, load_run(args.run), k=args.k, map_k=args.map_k)
        else:
            if not args.endpoint or not args.queries:
                raise EvaluationError("HTTP 模式需要同时提供 --endpoint 和 --queries")
            summary = evaluate_http(args.endpoint, load_queries(args.queries), qrels, k=args.k, mode=args.mode, scorer=args.scorer, expander=args.expander, timeout=args.timeout, run_output=args.output_run, tag=args.tag, map_k=args.map_k, fail_on_fallback=args.fail_on_fallback)
        output = json.dumps(summary.to_dict(args.include_per_query), ensure_ascii=False, indent=2)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(output + "\n", encoding="utf-8")
        print(output)
    except EvaluationError as error:
        print(f"evaluation error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
