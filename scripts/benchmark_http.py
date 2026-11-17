"""Measure public HTTP search latency and preserve query-level evidence.

Examples:
    python scripts/benchmark_http.py --endpoint http://127.0.0.1:18082/search \
      --queries official-queries.jsonl --limit 32 --modes lexical dense \
      --label official-baseline --out runs/official-baseline.json

The output contains request/client timings, server stage timings and ranked IDs.
It is a latency benchmark, not a relevance evaluation.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import http.client
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from urllib.parse import urlsplit


def read_queries(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            if line.lstrip().startswith("{"):
                row = json.loads(line)
                qid, query = row.get("qid"), row.get("query")
            else:
                fields = line.rstrip("\n").split("\t", 1)
                if len(fields) != 2:
                    raise ValueError(f"invalid query row {line_number}")
                qid, query = fields
            if not isinstance(qid, str) or not qid or not isinstance(query, str) or not query.strip():
                raise ValueError(f"invalid query identity at row {line_number}")
            rows.append({"qid": qid, "query": query})
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("no queries selected")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def one_request(endpoint: str, timeout: float, row: dict, mode: str, k: int,
                run_tag: str) -> dict:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("endpoint must be an HTTP(S) search URL")
    body = json.dumps({"query": row["query"], "mode": mode, "k": k,
                       "run_tag": run_tag}, ensure_ascii=False).encode("utf-8")
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port, timeout=timeout)
    path = parsed.path or "/search"
    if parsed.query:
        path += "?" + parsed.query
    started = time.perf_counter()
    status = None
    try:
        connection.request("POST", path, body=body,
                           headers={"Content-Type": "application/json", "Accept": "application/json"})
        response = connection.getresponse()
        status = response.status
        raw = response.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {"error": "non-JSON response"}
    except (OSError, http.client.HTTPException) as error:
        elapsed_ms = (time.perf_counter() - started) * 1000
        payload = {"error": type(error).__name__}
    finally:
        connection.close()
    results = payload.get("results") if isinstance(payload, dict) else None
    return {
        "qid": row["qid"], "query": row["query"], "mode": mode,
        "http_status": status, "ok": status == 200 and isinstance(results, list),
        "client_ms": elapsed_ms,
        "server_ms": payload.get("took_ms") if isinstance(payload, dict) else None,
        "timings_ms": payload.get("timings_ms") if isinstance(payload, dict) else None,
        "error": payload.get("error") if status != 200 and isinstance(payload, dict) else None,
        "results": [{"doc_id": item.get("doc_id"), "score": item.get("score")}
                    for item in results] if isinstance(results, list) else [],
        "run_tag": run_tag,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="full /search URL")
    parser.add_argument("--queries", type=Path, help="JSONL qid/query or TSV qid<TAB>query")
    parser.add_argument("--query", action="append", default=[], help="ad-hoc query; repeatable")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=["lexical"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.limit < 1 or args.k < 1 or args.k > 200 or args.warmup < 0 or args.concurrency < 1:
        parser.error("limit/concurrency must be positive, k must be 1..200, warmup must be nonnegative")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("timeout must be positive and finite")
    if not args.modes or any(mode not in ("lexical", "dense", "hybrid", "adaptive") for mode in args.modes):
        parser.error("invalid modes")
    if bool(args.queries) == bool(args.query):
        parser.error("provide exactly one of --queries and --query")
    rows = (read_queries(args.queries, args.limit) if args.queries
            else [{"qid": f"adhoc-{i + 1:03d}", "query": query}
                  for i, query in enumerate(args.query)])
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = None
    started_wall = time.perf_counter()
    warmups = []
    for index in range(min(args.warmup, len(rows))):
        for mode in args.modes:
            row = rows[index]
            warmups.append(one_request(
                args.endpoint, args.timeout, row, mode, args.k,
                f"{args.label}:warmup:{mode}:{index}:{row['qid']}"))
    jobs = [(row, mode) for mode in args.modes for row in rows]
    def run(job):
        row, mode = jobs[job]
        return one_request(args.endpoint, args.timeout, row, mode, args.k,
                           f"{args.label}:measured:{mode}:{job}:{row['qid']}")
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        measurements = list(executor.map(run, range(len(jobs))))
    elapsed_s = time.perf_counter() - started_wall
    by_mode = {}
    for mode in args.modes:
        selected = [row for row in measurements if row["mode"] == mode and row["ok"]]
        stage_names = ("lexical", "router", "encode", "vector", "fusion")
        by_mode[mode] = {
            "client_ms": summarize([row["client_ms"] for row in selected]),
            "server_ms": summarize([float(row["server_ms"]) for row in selected
                                    if isinstance(row.get("server_ms"), (int, float))]),
            "stages_ms": {name: summarize([float(row["timings_ms"].get(name, 0.0))
                                           for row in selected if isinstance(row.get("timings_ms"), dict)])
                          for name in stage_names},
            "successful": len(selected),
            "failed": sum(row["mode"] == mode and not row["ok"] for row in measurements),
        }
    receipt = {
        "schema_version": 1,
        "label": args.label,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "HTTP latency only; no relevance claim",
        "git_head": git_head,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "logical_cpus": os.cpu_count()},
        "config": {"endpoint": args.endpoint, "query_count": len(rows), "modes": args.modes,
                   "k": args.k, "warmup_per_mode": min(args.warmup, len(rows)),
                   "concurrency": args.concurrency, "timeout_s": args.timeout},
        "wall_seconds_including_warmup": elapsed_s,
        "summary": by_mode,
        "warmups": warmups,
        "measurements": measurements,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"label": args.label, "out": str(args.out), "summary": by_mode},
                     ensure_ascii=False, indent=2))
    return 1 if any(not row["ok"] for row in measurements) else 0


if __name__ == "__main__":
    raise SystemExit(main())
