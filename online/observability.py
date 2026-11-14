"""Privacy-preserving structured search logs.

Search requests are logged as JSON lines to stderr by default. The record only
contains engineering metadata: never the query text, request body, headers,
full exceptions or credentials. Serialization and total HTTP elapsed are
separate fields so a slow client/socket is not mistaken for slow retrieval.
"""

from datetime import datetime, timezone
import json
import math
import sys
import threading

RUN_TAG_LIMIT = 128
TIMING_STAGES = ("lexical", "router", "encode", "vector", "fusion")


def bounded_run_tag(value) -> str:
    """Run tags are caller-controlled text; keep them one bounded JSON string."""
    if not isinstance(value, str):
        return ""
    return value[:RUN_TAG_LIMIT]


def short_corpus_identity(value) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:12]


def _milliseconds(value):
    if value is None or type(value) is bool or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return round(float(value), 3)


def _timings(value):
    if not isinstance(value, dict):
        return None
    stages = TIMING_STAGES + (("rerank",) if "rerank" in value else ())
    return {name: _milliseconds(value.get(name)) for name in stages}


def _utc_now() -> str:
    return (datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"))


class SearchLogger:
    """Thread-safe JSONL search event sink (stderr unless a sink is injected)."""

    def __init__(self, sink=None):
        self._sink = sys.stderr if sink is None else sink
        self._lock = threading.Lock()

    def emit(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                          sort_keys=True, allow_nan=False) + "\n"
        with self._lock:
            try:
                self._sink.write(line)
                flush = getattr(self._sink, "flush", None)
                if callable(flush):
                    flush()
            except Exception:
                # Observability must never take down an otherwise healthy request.
                pass

    def search(self, *, status: int, run_tag="", requested_mode=None,
               executed_mode=None, reason=None, vector_attempted=None,
               vector_used=None, fallback=None, policy_version=None,
               dataset=None, corpus_sha256=None, took_ms=None, timings_ms=None,
               serialization_ms=None, http_elapsed_ms=None, rerank=None) -> None:
        self.emit({
            "event": "search",
            "time": _utc_now(),
            "status": status,
            "run_tag": bounded_run_tag(run_tag),
            "requested_mode": requested_mode,
            "executed_mode": executed_mode,
            "reason": reason,
            "vector_attempted": vector_attempted,
            "vector_used": vector_used,
            "fallback": fallback,
            "policy_version": policy_version,
            "dataset": dataset,
            "corpus_identity": short_corpus_identity(corpus_sha256),
            "took_ms": took_ms,
            "timings_ms": _timings(timings_ms),
            "serialization_ms": _milliseconds(serialization_ms),
            "http_elapsed_ms": _milliseconds(http_elapsed_ms),
            **({"rerank": rerank} if rerank is not None else {}),
        })
