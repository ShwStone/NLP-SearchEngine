"""HTTP 服务（在线组）。只用标准库，零依赖。

    python -m online.server --index fixtures/tiny_index --port 8000
    python -m online.server --config /path/to/serving.json

    curl -s localhost:8000/health
    curl -s localhost:8000/index/info
    curl -s -X POST localhost:8000/search \
        -H 'Content-Type: application/json' \
        -d '{"query": "心肌梗死 症状", "k": 5}'

版本化 profile 与零散 CLI flag 互斥：一个字段只有一处来源。请求体在读取前
按 ``max_body_bytes`` 拒绝，socket 读写受 ``request_timeout_s`` 约束；检索事件
以 JSONL 写 stderr（不记录 query/请求头/完整异常/凭据）。
"""

import argparse
import json
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from contracts.api import ApiError, SEARCH_INSTANCE_HEADER, SearchRequest
from contracts.retrieval import RetrievalConfig, RetrievalUnavailable, VectorManifest
from offline.disk_index import DiskIndex

from .engine import SearchEngine
from .embeddings import SentenceTransformerEncoder
from .expanders import ExpanderNotReady
from .observability import SearchLogger
from .parser import QuerySyntaxError
from .reranking import HybridReranker, RerankUnavailable
from .router import load_router
from .serving_config import (
    DEFAULT_CANDIDATE_K, DEFAULT_HOST, DEFAULT_MAX_BODY_BYTES, DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT_S, DEFAULT_ROUTER_FACTORY, DEFAULT_RRF_C,
    DEFAULT_VECTOR_TIMEOUT_S, SCHEMA_VERSION, ServingConfigError,
    load_serving_config, validate_vector_url,
)
from .vector_index import QdrantVectorIndex


STATIC_ROOT = Path(__file__).resolve().parent.parent / "web"

_LEGACY_FLAGS = {
    "index": "--index",
    "host": "--host",
    "port": "--port",
    "candidate_k": "--candidate-k",
    "rrf_c": "--rrf-c",
    "vector_url": "--vector-url",
    "vector_manifest": "--vector-manifest",
    "hnsw_ef": "--hnsw-ef",
    "bge_model_path": "--bge-model-path",
    "encoder_device": "--encoder-device",
    "router_factory": "--router-factory",
    "adaptive_policy": "--adaptive-policy",
    "vector_timeout_s": "--vector-timeout-s",
    "max_body_bytes": "--max-body-bytes",
    "request_timeout_s": "--request-timeout-s",
}


class _BodyError(Exception):
    def __init__(self, status: int, message: str, reason: str):
        super().__init__(message)
        self.status = status
        self.message = message
        self.reason = reason


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _identity_reader(engine: SearchEngine):
    """Lazily read the lexical identity; never fail a request on log metadata."""
    cache = {}

    def identity():
        if "value" not in cache:
            try:
                manifest = engine.reader.manifest()
                dataset = manifest.get("dataset") if isinstance(manifest, dict) else None
                corpus = None
                try:
                    index_identity = engine.info().get("index_identity") or {}
                    corpus = index_identity.get("corpus_sha256")
                except Exception:
                    corpus = None
                cache["value"] = (dataset, corpus)
            except Exception:
                cache["value"] = (None, None)
        return cache["value"]

    return identity


def make_handler(engine: SearchEngine, static_root: Path | None = None, *,
                 serving: dict | None = None,
                 max_body_bytes: int | None = None,
                 request_timeout_s: float | None = None,
                 search_log: SearchLogger | None = None):
    """Create an API handler and the small browser UI's static-file routes.

    ``static_root`` stays the second positional argument for compatibility.
    ``serving`` is the mutable effective-profile mapping echoed by
    ``/index/info``; the CLI updates host/port after binding port 0. New
    keyword arguments are keyword-only: request bounds and the log sink.
    """
    root = (static_root or STATIC_ROOT).resolve()
    serving_state = serving if isinstance(serving, dict) else dict(serving or {})
    serving_state.setdefault("schema_version", SCHEMA_VERSION)
    serving_state.setdefault("profile", None)
    serving_state.setdefault("host", DEFAULT_HOST)
    serving_state.setdefault("port", DEFAULT_PORT)
    serving_state.setdefault("max_body_bytes", DEFAULT_MAX_BODY_BYTES)
    serving_state.setdefault("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S)
    serving_state.setdefault("router_factory", DEFAULT_ROUTER_FACTORY)
    if max_body_bytes is not None:
        serving_state["max_body_bytes"] = max_body_bytes
    if request_timeout_s is not None:
        serving_state["request_timeout_s"] = request_timeout_s
    body_limit = serving_state["max_body_bytes"]
    read_timeout = serving_state["request_timeout_s"]
    logger = search_log if search_log is not None else SearchLogger()
    identity = _identity_reader(engine)
    instance_id = uuid4().hex
    policy = getattr(engine, "policy", None)
    policy_version = (policy.get("policy_version") if isinstance(policy, dict) else None)

    class Handler(BaseHTTPRequestHandler):
        timeout = read_timeout

        def _send(self, code: int, payload: dict) -> float:
            started = time.perf_counter()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            serialization_ms = (time.perf_counter() - started) * 1000
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header(SEARCH_INSTANCE_HEADER, instance_id)
            self.end_headers()
            self.wfile.write(body)
            return serialization_ms

        def _send_bytes(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(SEARCH_INSTANCE_HEADER, instance_id)
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, path: str) -> bool:
            if path == "/":
                relative = Path("index.html")
            elif path.startswith("/static/"):
                relative = Path(unquote(path[len("/static/"):]))
            else:
                return False

            # Resolve both normal ``..`` segments and symlinks before checking
            # containment.  A rejected path is reported as a normal 404.
            try:
                candidate = (root / relative).resolve()
                candidate.relative_to(root)
                if not candidate.is_file():
                    raise FileNotFoundError
                body = candidate.read_bytes()
            except (FileNotFoundError, OSError, ValueError):
                self._send(404, {"error": "unknown path"})
                return True

            content_type, _ = mimetypes.guess_type(candidate.name)
            if content_type is None:
                content_type = "application/octet-stream"
            if content_type.startswith("text/") or content_type in {
                "application/javascript", "application/json",
            }:
                content_type += "; charset=utf-8"
            self._send_bytes(200, body, content_type)
            return True

        def log_message(self, fmt, *args):  # 安静一点
            pass

        def _serving_info(self) -> dict:
            info = dict(serving_state)
            if info.get("profile") is None:
                info["profile"] = identity()[0]
            return info

        def _log_search(self, *, status: int, run_tag="", requested_mode=None,
                        executed_mode=None, reason=None, vector_attempted=None,
                        vector_used=None, fallback=None, took_ms=None,
                        timings_ms=None, serialization_ms=None,
                        http_elapsed_ms=None, rerank=None):
            dataset, corpus = identity()
            logger.search(
                status=status, run_tag=run_tag, requested_mode=requested_mode,
                executed_mode=executed_mode, reason=reason,
                vector_attempted=vector_attempted, vector_used=vector_used,
                fallback=fallback, policy_version=policy_version,
                dataset=dataset, corpus_sha256=corpus, took_ms=took_ms,
                timings_ms=timings_ms, serialization_ms=serialization_ms,
                http_elapsed_ms=http_elapsed_ms,
                rerank=rerank,
            )

        def do_GET(self):
            path = urlsplit(self.path).path
            if self._serve_static(path):
                return
            reader = engine.reader
            if path == "/health":
                m = reader.manifest()
                self._send(200, {"status": "ok", "index": {
                    "dataset": m["dataset"],
                    "n_docs": m["n_docs"],
                    "schema_version": m["schema_version"],
                    "analyzer_hash": m["analyzer"]["config_hash"],
                }})
            elif path == "/index/info":
                retrieval = engine.info()
                retrieval["serving"] = self._serving_info()
                self._send(200, {**reader.manifest(), "retrieval": retrieval})
            else:
                self._send(404, {"error": "unknown path"})

        def _read_json_body(self):
            if self.headers.get("Transfer-Encoding") is not None:
                raise _BodyError(400, "unsupported Transfer-Encoding",
                                 "invalid_request_framing")
            lengths = self.headers.get_all("Content-Length")
            if lengths is None or len(lengths) != 1:
                raise _BodyError(400, "invalid Content-Length",
                                 "invalid_request_framing")
            raw_length = lengths[0].strip()
            # ``str.isdigit`` admits non-ASCII digits that ``int`` rejects, and
            # ``int`` refuses very long digit strings; bound the conversion by
            # the decimal width of the limit itself.
            if not raw_length or not raw_length.isascii() or not raw_length.isdigit():
                raise _BodyError(400, "invalid Content-Length",
                                 "invalid_request_framing")
            digits = raw_length.lstrip("0") or "0"
            if len(digits) > len(str(body_limit)):
                raise _BodyError(413, "request body too large", "payload_too_large")
            length = int(digits)
            if length > body_limit:
                raise _BodyError(413, "request body too large", "payload_too_large")
            try:
                data = self.rfile.read(length)
            except TimeoutError as error:
                raise _BodyError(408, "request body read timed out",
                                 "request_timeout") from error
            except OSError as error:
                raise _BodyError(400, "request body read failed",
                                 "invalid_request") from error
            if len(data) != length:
                raise _BodyError(400, "incomplete request body", "invalid_request")
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                # Invalid UTF-8 is a client error, not a dropped connection or
                # a thread traceback; the raw bytes are never echoed.
                raise _BodyError(400, "request body is not valid JSON",
                                 "invalid_request") from error
            try:
                return json.loads(text) if text else {}
            except (ValueError, RecursionError) as error:
                # JSONDecodeError is a ValueError; integer digit limits and
                # excessive nesting must use the same sanitized client error.
                raise _BodyError(400, "request body is not valid JSON",
                                 "invalid_request") from error

        def do_POST(self):
            started = time.perf_counter()
            if urlsplit(self.path).path != "/search":
                self._send(404, {"error": "unknown path"})
                return
            try:
                payload = self._read_json_body()
            except _BodyError as error:
                # The payload may still be in flight; do not parse it as a new request.
                self.close_connection = True
                serialization_ms = self._send(error.status, {"error": error.message})
                self._log_search(
                    status=error.status, reason=error.reason,
                    serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            try:
                req = SearchRequest.from_dict(payload)
            except (ApiError, ValueError) as error:
                serialization_ms = self._send(400, {"error": str(error)})
                self._log_search(
                    status=400, reason="invalid_request",
                    serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            try:
                resp = engine.search(req)
            except (QuerySyntaxError, ApiError) as error:
                serialization_ms = self._send(400, {"error": str(error)})
                self._log_search(
                    status=400, run_tag=req.run_tag, requested_mode=req.mode,
                    reason="invalid_request", serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            except ExpanderNotReady as error:
                serialization_ms = self._send(501, {"error": str(error)})
                self._log_search(
                    status=501, run_tag=req.run_tag, requested_mode=req.mode,
                    reason="expander_unavailable", serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            except RetrievalUnavailable as error:
                attempted = req.mode in ("dense", "hybrid")
                serialization_ms = self._send(
                    503, {"error": str(error), "code": "vector_unavailable"})
                self._log_search(
                    status=503, run_tag=req.run_tag, requested_mode=req.mode,
                    reason="vector_unavailable", vector_attempted=attempted,
                    vector_used=False, fallback=False,
                    serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            except RerankUnavailable:
                serialization_ms = self._send(
                    503, {"error": "cross-encoder unavailable", "code": "reranker_unavailable"})
                self._log_search(
                    status=503, run_tag=req.run_tag, requested_mode=req.mode,
                    reason="reranker_unavailable", vector_attempted=True, vector_used=True,
                    fallback=False, serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            except Exception:
                # Plugin/engine failures must not masquerade as empty results, and
                # the response/log must not expose the raw exception.
                serialization_ms = self._send(500, {"error": "internal server error"})
                self._log_search(
                    status=500, run_tag=req.run_tag, requested_mode=req.mode,
                    reason="internal_error", serialization_ms=serialization_ms,
                    http_elapsed_ms=(time.perf_counter() - started) * 1000,
                )
                return
            serialization_ms = self._send(200, resp.to_dict())
            routing = resp.routing
            self._log_search(
                status=200, run_tag=req.run_tag,
                requested_mode=routing.requested_mode,
                executed_mode=routing.executed_mode, reason=routing.reason,
                vector_attempted=routing.vector_attempted,
                vector_used=routing.vector_used, fallback=routing.fallback,
                took_ms=resp.took_ms, timings_ms=resp.timings_ms,
                serialization_ms=serialization_ms,
                http_elapsed_ms=(time.perf_counter() - started) * 1000,
                rerank=resp.rerank,
            )

    return Handler


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="搜索引擎 HTTP 服务")
    ap.add_argument("--config", help="ServingConfig v1 JSON profile")
    ap.add_argument("--index", help="词法索引目录")
    ap.add_argument("--host", help="监听地址；默认 127.0.0.1")
    ap.add_argument("--port", type=int, help="监听端口；默认 8000")
    ap.add_argument("--candidate-k", type=int, help="每路候选池；默认 200")
    ap.add_argument("--rrf-c", type=float, help="RRF 常数；默认 60.0")
    ap.add_argument("--adaptive-policy", choices=("rules-v1", "recall-v1", "hybrid-v1"),
                    help="BM25-first adaptive routing policy for the built-in rule router")
    ap.add_argument("--vector-url", help="Qdrant URL; requires --vector-manifest")
    ap.add_argument("--vector-manifest", help="published vector_manifest.json")
    ap.add_argument("--hnsw-ef", type=int,
                    help="positive Qdrant HNSW search beam; requires vector arguments")
    ap.add_argument("--vector-timeout-s", type=float,
                    help="Qdrant client timeout in seconds; requires vector arguments")
    ap.add_argument("--bge-model-path",
                    help="verified local pinned BGE directory; requires vector arguments")
    ap.add_argument("--encoder-device", help="device for --bge-model-path (default: cpu)")
    ap.add_argument("--router-factory", help="policy selector: rule or trusted module:factory")
    ap.add_argument("--max-body-bytes", type=int, help="max request body bytes; default 65536")
    ap.add_argument("--request-timeout-s", type=float,
                    help="socket read timeout in seconds; default 10.0")
    return ap


def _legacy_settings(ap: argparse.ArgumentParser, args) -> dict:
    if args.index is None:
        ap.error("必须提供 --config 或 --index")
    if not isinstance(args.index, str) or not args.index.strip():
        ap.error("--index 必须是非空路径")
    host = DEFAULT_HOST if args.host is None else args.host
    if not isinstance(host, str) or not host.strip() or host != host.strip():
        ap.error("--host 必须是非空且无首尾空白的字符串")
    port = DEFAULT_PORT if args.port is None else args.port
    if type(port) is not int or not 0 <= port <= 65535:
        ap.error("--port 必须在 0..65535 之间")
    candidate_k = DEFAULT_CANDIDATE_K if args.candidate_k is None else args.candidate_k
    if type(candidate_k) is not int or candidate_k < 1:
        ap.error("--candidate-k 必须是正整数")
    rrf_c = DEFAULT_RRF_C if args.rrf_c is None else args.rrf_c
    if type(rrf_c) is bool or not isinstance(rrf_c, (int, float)) \
            or rrf_c <= 0 or rrf_c == float("inf") or rrf_c != rrf_c:
        ap.error("--rrf-c 必须是有限正数")
    if bool(args.vector_url) != bool(args.vector_manifest):
        ap.error("--vector-url 与 --vector-manifest 必须同时提供")
    if args.hnsw_ef is not None:
        if not args.vector_url or not args.vector_manifest:
            ap.error("--hnsw-ef requires --vector-url and --vector-manifest")
        if type(args.hnsw_ef) is not int or args.hnsw_ef <= 0:
            ap.error("--hnsw-ef must be a positive integer")
    vector_timeout = (DEFAULT_VECTOR_TIMEOUT_S if args.vector_timeout_s is None
                      else args.vector_timeout_s)
    if args.vector_timeout_s is not None:
        if not args.vector_url or not args.vector_manifest:
            ap.error("--vector-timeout-s requires --vector-url and --vector-manifest")
        if (type(vector_timeout) is bool or not isinstance(vector_timeout, (int, float))
                or not (vector_timeout > 0) or vector_timeout == float("inf")
                or vector_timeout != vector_timeout):
            ap.error("--vector-timeout-s must be a positive finite number")
    if args.bge_model_path is not None and not args.vector_manifest:
        ap.error("--bge-model-path requires --vector-url and --vector-manifest")
    if args.encoder_device is not None and args.bge_model_path is None:
        ap.error("--encoder-device requires --bge-model-path")
    if args.bge_model_path == "" or args.encoder_device == "":
        ap.error("BGE model path and encoder device must not be empty")
    if args.vector_url is not None:
        try:
            validate_vector_url(args.vector_url)
        except ServingConfigError as error:
            ap.error(str(error))
    max_body_bytes = (DEFAULT_MAX_BODY_BYTES if args.max_body_bytes is None
                      else args.max_body_bytes)
    if type(max_body_bytes) is not int or max_body_bytes < 1:
        ap.error("--max-body-bytes 必须是正整数")
    request_timeout_s = (DEFAULT_REQUEST_TIMEOUT_S if args.request_timeout_s is None
                         else args.request_timeout_s)
    if (type(request_timeout_s) is bool or not isinstance(request_timeout_s, (int, float))
            or not (request_timeout_s > 0) or request_timeout_s == float("inf")
            or request_timeout_s != request_timeout_s):
        ap.error("--request-timeout-s 必须是有限正数")
    router_factory = (DEFAULT_ROUTER_FACTORY if args.router_factory is None
                      else args.router_factory)
    if not isinstance(router_factory, str) or not router_factory.strip():
        ap.error("--router-factory 必须是非空字符串")
    return {
        "host": host,
        "port": port,
        "candidate_k": candidate_k,
        "rrf_c": rrf_c,
        "vector_timeout_s": vector_timeout,
        "forward_vector_timeout": args.vector_timeout_s is not None,
        "max_body_bytes": max_body_bytes,
        "request_timeout_s": request_timeout_s,
        "router_factory": router_factory,
    }


def _load_reader(ap: argparse.ArgumentParser, path) -> DiskIndex:
    try:
        return DiskIndex(str(path))
    except Exception as error:
        ap.error(f"词法索引加载失败: {error}")


def _load_vector_backend(ap: argparse.ArgumentParser, *, manifest_path, url,
                         hnsw_ef, timeout_s, forward_timeout, model_path, device):
    try:
        manifest = VectorManifest.from_dict(json.loads(
            Path(manifest_path).read_text(encoding="utf-8")))
    except Exception as error:
        ap.error(f"向量 manifest 加载失败: {error}")
    options = {}
    if hnsw_ef is not None:
        options["hnsw_ef"] = hnsw_ef
    if forward_timeout:
        options["timeout_s"] = timeout_s
    spec = manifest.embedding
    try:
        if model_path is not None:
            from adapters.embedding import BGEEncoder
            encoder = BGEEncoder(spec=spec, model_path=str(model_path),
                                 device=device or "cpu")
        else:
            encoder = SentenceTransformerEncoder(
                spec.model_id, spec.revision, spec.dimension, spec.max_tokens,
                spec.query_prefix, spec.document_prefix,
            )
        vectors = QdrantVectorIndex(manifest, url, **options)
    except (OSError, ValueError, RetrievalUnavailable) as error:
        ap.error(f"向量后端初始化失败: {error}")
    return encoder, vectors


def _emit_startup(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), file=sys.stderr,
          flush=True)


def main(argv=None):
    ap = _parser()
    args = ap.parse_args(argv)

    config = None
    if args.config is not None:
        explicit = [flag for attribute, flag in _LEGACY_FLAGS.items()
                    if getattr(args, attribute) is not None]
        if explicit:
            ap.error("--config 不能与旧版参数同时使用: " + ", ".join(explicit))
        try:
            config = load_serving_config(args.config)
        except ServingConfigError as error:
            ap.error(f"配置加载失败: {error}")

    if config is not None:
        host = config.serve.host
        port = config.serve.port
        candidate_k = config.retrieval.candidate_k
        rrf_c = config.retrieval.rrf_c
        max_body_bytes = config.serve.max_body_bytes
        request_timeout_s = config.serve.request_timeout_s
        router_factory = config.policy_factory
        index_path = config.index
        vector_timeout_s = (config.vector.timeout_s if config.vector is not None
                            else DEFAULT_VECTOR_TIMEOUT_S)
        forward_vector_timeout = True
        expected_dataset = config.dataset
    else:
        settings = _legacy_settings(ap, args)
        host = settings["host"]
        port = settings["port"]
        candidate_k = settings["candidate_k"]
        rrf_c = settings["rrf_c"]
        max_body_bytes = settings["max_body_bytes"]
        request_timeout_s = settings["request_timeout_s"]
        router_factory = settings["router_factory"]
        index_path = Path(args.index)
        vector_timeout_s = settings["vector_timeout_s"]
        forward_vector_timeout = settings["forward_vector_timeout"]
        expected_dataset = None

    if args.adaptive_policy is not None and router_factory != DEFAULT_ROUTER_FACTORY:
        ap.error("--adaptive-policy requires the built-in rule router")
    retrieval_config = RetrievalConfig(
        candidate_k=candidate_k, rrf_c=rrf_c,
        adaptive_policy=args.adaptive_policy or "rules-v1",
    )
    encoder = vectors = None
    vector_record = None
    embedding_record = None
    model_path = config.embedding.model_path if config and config.embedding else None
    device = config.embedding.device if config and config.embedding else None

    if config is not None:
        # Dataset identity is checked before any model/backend construction or bind.
        reader = _load_reader(ap, index_path)
        manifest = reader.manifest()
        actual_dataset = manifest.get("dataset") if isinstance(manifest, dict) else None
        if actual_dataset != expected_dataset:
            ap.error(f"dataset 不匹配：配置为 {expected_dataset!r}，"
                     f"词法索引为 {actual_dataset!r}")
        if config.vector is not None:
            encoder, vectors = _load_vector_backend(
                ap, manifest_path=config.vector.manifest, url=config.vector.url,
                hnsw_ef=config.vector.hnsw_ef, timeout_s=vector_timeout_s,
                forward_timeout=forward_vector_timeout, model_path=model_path,
                device=device,
            )
            vector_record = {
                "url": config.vector.url,
                "manifest": str(config.vector.manifest),
                "hnsw_ef": config.vector.hnsw_ef,
                "timeout_s": vector_timeout_s,
            }
            embedding_record = {
                "model_path": str(model_path) if model_path is not None else None,
                "device": device,
            }
    else:
        # Preserve the legacy order: backend/model failures happen before the
        # lexical index is loaded, so an unusable vector profile fails fast.
        if args.vector_manifest:
            encoder, vectors = _load_vector_backend(
                ap, manifest_path=Path(args.vector_manifest), url=args.vector_url,
                hnsw_ef=args.hnsw_ef, timeout_s=vector_timeout_s,
                forward_timeout=forward_vector_timeout,
                model_path=args.bge_model_path, device=args.encoder_device,
            )
            vector_record = {
                "url": args.vector_url,
                "manifest": str(Path(args.vector_manifest).resolve()),
                "hnsw_ef": args.hnsw_ef,
                "timeout_s": vector_timeout_s if forward_vector_timeout else None,
            }
            embedding_record = {
                "model_path": (str(Path(args.bge_model_path).resolve())
                               if args.bge_model_path is not None else None),
                "device": args.encoder_device,
            }
        reader = _load_reader(ap, index_path)

    try:
        router = load_router(router_factory, retrieval_config)
    except Exception as error:
        ap.error(f"路由策略加载失败: {error}")
    try:
        reranker = None
        if config is not None and config.rerank is not None:
            from adapters.reranker import FrozenCrossEncoder
            from .ce_gate import load_gate
            settings = config.rerank
            gate = (load_gate(settings.gate_path, settings.gate_sha256)
                    if settings.gate_path is not None else None)
            reranker = HybridReranker(FrozenCrossEncoder(
                settings.model_path, device=settings.device, batch_size=settings.batch_size),
                settings.depth, gate=gate, capture_features=settings.capture_features)
        engine = SearchEngine(reader, encoder, vectors, retrieval_config, router,
                              **({"reranker": reranker} if reranker is not None else {}))
    except Exception as error:
        ap.error(f"检索引擎初始化失败: {error}")

    manifest = reader.manifest()
    actual_dataset = manifest.get("dataset") if isinstance(manifest, dict) else None
    info = engine.info()
    serving_state = {
        "schema_version": SCHEMA_VERSION,
        "profile": actual_dataset,
        "host": host,
        "port": port,
        "max_body_bytes": max_body_bytes,
        "request_timeout_s": request_timeout_s,
        "router_factory": router_factory,
    }
    handler = make_handler(engine, serving=serving_state,
                           search_log=SearchLogger())
    if not _is_loopback(host):
        print(f"警告：监听非回环地址 {host}；本服务没有认证/TLS/限流，"
              "请在前置代理限制访问。", file=sys.stderr, flush=True)
    try:
        server = ThreadingHTTPServer((host, port), handler)
    except OSError as error:
        ap.error(f"无法监听 {host}:{port}: {error}")
    actual_host, actual_port = server.server_address[0], server.server_address[1]
    serving_state["host"], serving_state["port"] = actual_host, actual_port

    _emit_startup({
        "event": "startup",
        "dataset": actual_dataset,
        "n_docs": manifest.get("n_docs") if isinstance(manifest, dict) else None,
        "host": actual_host,
        "port": actual_port,
        "max_body_bytes": max_body_bytes,
        "request_timeout_s": request_timeout_s,
        "router_factory": router_factory,
        "policy_version": router.policy_version,
        "index": str(index_path),
        "vector": vector_record,
        "embedding": embedding_record if vectors is not None else None,
        # Same identity echo as /index/info: index/vector identity, search
        # parameters and the policy the engine actually loaded.
        "index_identity": info.get("index_identity"),
        "vector_identity": info.get("vector_identity"),
        "search_config": info.get("search_config"),
        "policy": info.get("policy"),
        **({"query_encoder_runtime": info["query_encoder_runtime"]}
           if "query_encoder_runtime" in info else {}),
        **({"reranker": info["reranker"]} if "reranker" in info else {}),
    })
    print(f"索引已加载（{engine.reader.n_docs} 篇），监听 {actual_host}:{actual_port}",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
