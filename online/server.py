"""HTTP 服务（在线组）。只用标准库，零依赖。

    python -m online.server --index fixtures/tiny_index --port 8000

    curl -s localhost:8000/health
    curl -s localhost:8000/index/info
    curl -s -X POST localhost:8000/search \
        -H 'Content-Type: application/json' \
        -d '{"query": "心肌梗死 症状", "k": 5}'
"""

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from contracts.api import ApiError, SearchRequest
from contracts.retrieval import RetrievalConfig, RetrievalUnavailable, VectorManifest
from offline.disk_index import DiskIndex

from .engine import SearchEngine
from .embeddings import SentenceTransformerEncoder
from .expanders import ExpanderNotReady
from .parser import QuerySyntaxError
from .vector_index import QdrantVectorIndex


STATIC_ROOT = Path(__file__).resolve().parent.parent / "web"


def make_handler(engine: SearchEngine, static_root: Path | None = None):
    """Create an API handler and the small browser UI's static-file routes.

    ``static_root`` is injectable for tests, while the CLI uses the repository's
    ``web`` directory.  Static paths are resolved before opening files and must
    remain below the root, so URL traversal and symlink escapes are rejected.
    """
    root = (static_root or STATIC_ROOT).resolve()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, code: int, body: bytes, content_type: str):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
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
                self._send(200, {**reader.manifest(), "retrieval": engine.info()})
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self):
            if urlsplit(self.path).path != "/search":
                self._send(404, {"error": "unknown path"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                req = SearchRequest.from_dict(payload)
            except (json.JSONDecodeError, ApiError, ValueError) as e:
                self._send(400, {"error": str(e)})
                return
            try:
                resp = engine.search(req)
            except (QuerySyntaxError, ApiError) as e:
                self._send(400, {"error": str(e)})
                return
            except ExpanderNotReady as e:
                self._send(501, {"error": str(e)})
                return
            except RetrievalUnavailable as e:
                self._send(503, {"error": str(e), "code": "vector_unavailable"})
                return
            self._send(200, resp.to_dict())

    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="搜索引擎 HTTP 服务")
    ap.add_argument("--index", required=True)
    ap.add_argument("--host", default="127.0.0.1", help="监听地址；0.0.0.0 允许外部访问")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--candidate-k", type=int, default=200)
    ap.add_argument("--rrf-c", type=float, default=60.0)
    ap.add_argument("--adaptive-policy", choices=("rules-v1", "recall-v1", "hybrid-v1"),
                    default="rules-v1", help="BM25-first adaptive routing policy")
    ap.add_argument("--vector-url", help="Qdrant URL; requires --vector-manifest")
    ap.add_argument("--vector-manifest", help="published vector_manifest.json")
    args = ap.parse_args()

    reader = DiskIndex(args.index)
    encoder = vectors = None
    if bool(args.vector_url) != bool(args.vector_manifest):
        ap.error("--vector-url 与 --vector-manifest 必须同时提供")
    if args.vector_manifest:
        try:
            manifest = VectorManifest.from_dict(json.loads(
                Path(args.vector_manifest).read_text(encoding="utf-8")))
            spec = manifest.embedding
            encoder = SentenceTransformerEncoder(
                spec.model_id, spec.revision, spec.dimension, spec.max_tokens,
                spec.query_prefix, spec.document_prefix,
            )
            vectors = QdrantVectorIndex(manifest, args.vector_url)
        except (OSError, ValueError, RetrievalUnavailable) as error:
            ap.error(f"向量后端初始化失败: {error}")
    engine = SearchEngine(reader, encoder, vectors, RetrievalConfig(
        candidate_k=args.candidate_k, rrf_c=args.rrf_c, adaptive_policy=args.adaptive_policy))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"索引已加载（{engine.reader.n_docs} 篇），监听 {server.server_address[0]}:{server.server_port}",
          flush=True)
    server.serve_forever()
