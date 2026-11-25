"""Loopback reverse proxy for the multi-dataset browser demo.

The public server terminates TLS and rate limits requests. This gateway runs on
the retrieval host and only exposes explicitly configured dataset upstreams.

    python -m online.gateway --config deploy/config/datasets.example.json
"""

import argparse
from dataclasses import dataclass
import http.client
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from contracts.api import SEARCH_INSTANCE_HEADER


DEFAULT_BODY_LIMIT = 64 * 1024
DEFAULT_TIMEOUT_S = 45.0


class GatewayConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    id: str
    label: str
    description: str
    upstream: str
    modes: tuple[str, ...]
    examples: tuple[str, ...]
    n_docs: int

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "modes": list(self.modes),
            "examples": list(self.examples),
            "n_docs": self.n_docs,
        }


@dataclass(frozen=True)
class GatewayConfig:
    host: str
    port: int
    default_dataset: str
    timeout_s: float
    max_body_bytes: int
    datasets: dict[str, Dataset]


def _string(value, name: str, *, nonempty=True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise GatewayConfigError(f"{name} must be a nonempty string")
    return value


def _upstream(value: str) -> str:
    parsed = urlsplit(_string(value, "dataset.upstream"))
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
        raise GatewayConfigError("dataset.upstream must be a bare HTTP(S) origin without credentials")
    if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise GatewayConfigError("dataset.upstream must use a loopback host")
    try:
        parsed.port
    except ValueError as error:
        raise GatewayConfigError("dataset.upstream has an invalid port") from error
    return value.rstrip("/")


def load_config(path: str | Path) -> GatewayConfig:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GatewayConfigError(f"cannot read gateway config: {error}") from error
    if not isinstance(raw, dict):
        raise GatewayConfigError("gateway config must be a JSON object")
    allowed = {"schema_version", "host", "port", "default_dataset", "timeout_s",
               "max_body_bytes", "datasets"}
    if set(raw) - allowed or raw.get("schema_version") != 1:
        raise GatewayConfigError("unsupported gateway config schema or fields")
    host = _string(raw.get("host", "127.0.0.1"), "host")
    port = raw.get("port", 18083)
    if type(port) is not int or not 0 <= port <= 65535:
        raise GatewayConfigError("port must be in 0..65535")
    timeout_s = raw.get("timeout_s", DEFAULT_TIMEOUT_S)
    if (type(timeout_s) is bool or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise GatewayConfigError("timeout_s must be a positive finite number")
    body_limit = raw.get("max_body_bytes", DEFAULT_BODY_LIMIT)
    if type(body_limit) is not int or body_limit < 1:
        raise GatewayConfigError("max_body_bytes must be a positive integer")
    rows = raw.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise GatewayConfigError("datasets must be a nonempty list")
    datasets = {}
    for row in rows:
        if not isinstance(row, dict):
            raise GatewayConfigError("each dataset must be an object")
        fields = {"id", "label", "description", "upstream", "modes", "examples", "n_docs"}
        if set(row) != fields:
            raise GatewayConfigError("dataset fields must be exactly " + ", ".join(sorted(fields)))
        dataset_id = _string(row["id"], "dataset.id")
        if not dataset_id.replace("-", "").replace("_", "").isalnum() or dataset_id in datasets:
            raise GatewayConfigError("dataset.id must be unique and contain only letters, digits, - or _")
        modes = row["modes"]
        if (not isinstance(modes, list) or not modes
                or any(mode not in ("lexical", "dense", "hybrid", "adaptive") for mode in modes)
                or len(set(modes)) != len(modes)):
            raise GatewayConfigError("dataset.modes contains an invalid or duplicate mode")
        examples = row["examples"]
        if (not isinstance(examples, list)
                or any(not isinstance(example, str) or not example.strip() for example in examples)):
            raise GatewayConfigError("dataset.examples must contain nonempty strings")
        n_docs = row["n_docs"]
        if type(n_docs) is not int or n_docs <= 0:
            raise GatewayConfigError("dataset.n_docs must be a positive integer")
        datasets[dataset_id] = Dataset(
            id=dataset_id,
            label=_string(row["label"], "dataset.label"),
            description=_string(row["description"], "dataset.description", nonempty=False),
            upstream=_upstream(row["upstream"]),
            modes=tuple(modes),
            examples=tuple(examples),
            n_docs=n_docs,
        )
    default = _string(raw.get("default_dataset"), "default_dataset")
    if default not in datasets:
        raise GatewayConfigError("default_dataset must name a configured dataset")
    return GatewayConfig(host, port, default, float(timeout_s), body_limit, datasets)


def _request(upstream: str, method: str, path: str, body: bytes | None,
             content_type: str | None, timeout: float):
    parsed = urlsplit(upstream)
    connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port, timeout=timeout)
    headers = {"Accept": "application/json", "Connection": "close"}
    if body is not None:
        headers["Content-Type"] = content_type or "application/json"
        headers["Content-Length"] = str(len(body))
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        payload = response.read()
        return response.status, payload, response.headers.get("Content-Type"), response.headers.get(SEARCH_INSTANCE_HEADER)
    finally:
        connection.close()


def make_handler(config: GatewayConfig, *, state: dict | None = None):
    serving_state = state if state is not None else {"host": config.host, "port": config.port}
    instance_error = b'{"error":"dataset upstream unavailable","code":"dataset_unavailable"}'

    class Handler(BaseHTTPRequestHandler):
        timeout = 15.0

        def log_message(self, fmt, *args):
            pass

        def _json(self, status: int, payload: dict, *, dataset: str | None = None):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._bytes(status, body, "application/json; charset=utf-8", dataset=dataset)

        def _bytes(self, status: int, body: bytes, content_type: str, *,
                   dataset: str | None = None, instance: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", content_type or "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if dataset:
                self.send_header("X-Search-Dataset", dataset)
            if instance:
                self.send_header(SEARCH_INSTANCE_HEADER, instance)
            self.end_headers()
            self.wfile.write(body)

        def _dataset(self, dataset_id: str) -> Dataset | None:
            return config.datasets.get(dataset_id)

        def _proxy(self, dataset: Dataset, method: str, path: str, body=None,
                   content_type=None, timeout=None):
            try:
                status, payload, response_type, instance = _request(
                    dataset.upstream, method, path, body, content_type,
                    config.timeout_s if timeout is None else timeout,
                )
            except (OSError, http.client.HTTPException):
                self._bytes(503, instance_error, "application/json; charset=utf-8", dataset=dataset.id)
                return
            self._bytes(status, payload, response_type or "application/json; charset=utf-8",
                        dataset=dataset.id, instance=instance)

        def do_GET(self):
            split = urlsplit(self.path)
            path = split.path
            suffix = f"?{split.query}" if split.query else ""
            if path == "/datasets":
                # Catalog loading stays fast even while the 8M process starts.
                # The browser obtains live status independently for each dataset.
                self._json(200, {"default_dataset": config.default_dataset,
                                 "datasets": [row.public() for row in config.datasets.values()]})
                return
            if path == "/gateway/health":
                self._json(200, {"status": "ok", "default_dataset": config.default_dataset,
                                 "datasets": list(config.datasets)})
                return
            if path == "/health":
                self._proxy(config.datasets[config.default_dataset], "GET", "/health")
                return
            if path == "/index/info":
                self._proxy(config.datasets[config.default_dataset], "GET", "/index/info")
                return
            if path.startswith("/doc/") and len(path) > len("/doc/") and "/" not in path[len("/doc/"):]:
                self._proxy(config.datasets[config.default_dataset], "GET", f"{path}{suffix}")
                return
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "datasets" and parts[2] == "info":
                dataset = self._dataset(parts[1])
                if dataset:
                    self._proxy(dataset, "GET", "/index/info")
                    return
            if len(parts) == 4 and parts[0] == "datasets" and parts[2] == "doc" and parts[3]:
                dataset = self._dataset(parts[1])
                if dataset:
                    self._proxy(dataset, "GET", f"/doc/{parts[3]}{suffix}")
                    return
            self._json(404, {"error": "unknown path"})

        def do_POST(self):
            started_path = urlsplit(self.path).path
            dataset_id = config.default_dataset
            if started_path != "/search":
                parts = started_path.strip("/").split("/")
                if len(parts) != 3 or parts[0] != "datasets" or parts[2] != "search":
                    self._json(404, {"error": "unknown path"})
                    return
                dataset_id = parts[1]
            dataset = self._dataset(dataset_id)
            if dataset is None:
                self._json(404, {"error": "unknown dataset"})
                return
            lengths = self.headers.get_all("Content-Length")
            if self.headers.get("Transfer-Encoding") is not None or lengths is None or len(lengths) != 1:
                self.close_connection = True
                self._json(400, {"error": "invalid request framing"})
                return
            raw_length = lengths[0].strip()
            if not raw_length or not raw_length.isascii() or not raw_length.isdigit():
                self.close_connection = True
                self._json(400, {"error": "invalid Content-Length"})
                return
            if len(raw_length.lstrip("0") or "0") > len(str(config.max_body_bytes)):
                self.close_connection = True
                self._json(413, {"error": "request body too large"})
                return
            length = int(raw_length)
            if length > config.max_body_bytes:
                self.close_connection = True
                self._json(413, {"error": "request body too large"})
                return
            try:
                body = self.rfile.read(length)
            except (TimeoutError, OSError):
                self.close_connection = True
                self._json(400, {"error": "incomplete request body"})
                return
            if len(body) != length:
                self.close_connection = True
                self._json(400, {"error": "incomplete request body"})
                return
            self._proxy(dataset, "POST", "/search", body,
                        self.headers.get("Content-Type"))

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except GatewayConfigError as error:
        parser.error(str(error))
    state = {"host": config.host, "port": config.port}
    try:
        server = ThreadingHTTPServer((config.host, config.port), make_handler(config, state=state))
    except OSError as error:
        parser.error(f"cannot listen on {config.host}:{config.port}: {error}")
    state["host"], state["port"] = server.server_address[:2]
    print(json.dumps({"event": "gateway_startup", "host": state["host"], "port": state["port"],
                      "default_dataset": config.default_dataset,
                      "datasets": list(config.datasets)}, ensure_ascii=False), file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
