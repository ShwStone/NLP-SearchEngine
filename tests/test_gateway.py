import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from online.gateway import Dataset, GatewayConfig, GatewayConfigError, load_config, make_handler


class UpstreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Search-Instance", "upstream-instance")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "index": {"dataset": "upstream"}})
        elif self.path == "/index/info":
            self._json({"dataset": "upstream", "n_docs": 20})
        elif self.path.startswith("/doc/"):
            self._json({"doc_id": self.path[len("/doc/"):], "text": "full document", "path": self.path})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.server.received.append(payload)
        self._json({"results": [], "echo": payload})


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        self.upstream.received = []
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.upstream_thread.start()
        origin = f"http://127.0.0.1:{self.upstream.server_port}"
        self.datasets = {
            "news20k": Dataset("news20k", "新闻", "自爬", origin,
                               ("lexical", "dense"), ("核电",), 20_000),
            "official8m": Dataset("official8m", "全集", "官方", origin,
                                  ("lexical", "dense", "hybrid", "adaptive"), ("节假日",), 8_096_668),
        }
        self.config = GatewayConfig("127.0.0.1", 0, "news20k", 2.0, 1024, self.datasets)
        self.gateway = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.config))
        self.gateway_thread = threading.Thread(target=self.gateway.serve_forever, daemon=True)
        self.gateway_thread.start()

    def tearDown(self):
        self.gateway.shutdown()
        self.gateway.server_close()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.gateway_thread.join()
        self.upstream_thread.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.gateway.server_port, timeout=3)
        payload = json.dumps(body).encode() if body is not None else None
        connection.request(method, path, body=payload,
                           headers={"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        raw = response.read()
        result = response.status, json.loads(raw), dict(response.headers)
        connection.close()
        return result

    def test_catalog_is_public_and_does_not_expose_upstreams(self):
        status, payload, _ = self.request("GET", "/datasets")
        self.assertEqual(status, 200)
        self.assertEqual(payload["default_dataset"], "news20k")
        self.assertEqual([row["id"] for row in payload["datasets"]], ["news20k", "official8m"])
        self.assertNotIn("upstream", json.dumps(payload))
        self.assertEqual(payload["datasets"][0]["examples"], ["核电"])

    def test_dataset_and_legacy_routes_preserve_body_and_identity(self):
        for path in ("/health", "/index/info", "/datasets/official8m/info"):
            status, payload, headers = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertIn("dataset", payload.get("index", payload))
            self.assertEqual(headers["X-Search-Instance"], "upstream-instance")
        status, payload, headers = self.request(
            "POST", "/datasets/official8m/search", {"query": "节假日", "mode": "lexical", "k": 3})
        self.assertEqual(status, 200)
        self.assertEqual(payload["echo"]["query"], "节假日")
        self.assertEqual(headers["X-Search-Dataset"], "official8m")
        status, payload, _ = self.request("POST", "/search", {"query": "核电"})
        self.assertEqual(status, 200)
        self.assertEqual(self.upstream.received[-1], {"query": "核电"})

    def test_document_routes_forward_doc_id_and_query_string(self):
        status, payload, headers = self.request(
            "GET", "/datasets/official8m/doc/abc123?q=%E8%8A%82%E5%81%87%E6%97%A5")
        self.assertEqual(status, 200)
        self.assertEqual(payload["path"], "/doc/abc123?q=%E8%8A%82%E5%81%87%E6%97%A5")
        self.assertEqual(headers["X-Search-Dataset"], "official8m")
        status, payload, headers = self.request("GET", "/doc/crawl-000001?q=test")
        self.assertEqual(status, 200)
        self.assertEqual(payload["path"], "/doc/crawl-000001?q=test")
        self.assertEqual(headers["X-Search-Dataset"], "news20k")
        self.assertEqual(self.request("GET", "/datasets/missing/doc/abc123")[0], 404)
        self.assertEqual(self.request("GET", "/doc/")[0], 404)

    def test_server_doc_endpoint_returns_full_text_and_highlights(self):
        from online.engine import SearchEngine
        from online.memory_index import MemoryIndex
        from online.server import make_handler as make_server_handler

        corpus = [
            {"doc_id": "d1", "title": "测试标题", "text": "心肌梗死发作前常有胸闷胸痛症状，需要及时就医。"},
            {"doc_id": "d2", "title": "", "text": "iPhone掉水里怎么办？第一步立刻关机。"},
        ]
        engine = SearchEngine(MemoryIndex.from_corpus(corpus))
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_server_handler(engine))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            conn.request("GET", "/doc/d1?q=%E5%BF%83%E8%82%8C%E6%A2%97%E6%AD%BB")
            resp = conn.getresponse()
            body = json.loads(resp.read())
            self.assertEqual(resp.status, 200)
            self.assertEqual(body["doc_id"], "d1")
            self.assertEqual(body["title"], "测试标题")
            self.assertEqual(body["char_count"], len(corpus[0]["text"]))
            self.assertTrue(body["highlights"])
            self.assertEqual(body["highlights"][0]["start"], 0)
            self.assertEqual(body["highlights"][0]["end"], 4)
            conn.request("GET", "/doc/nonexistent-id")
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 404)
            conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_unknown_routes_and_oversized_bodies_are_rejected_locally(self):
        self.assertEqual(self.request("GET", "/datasets/missing/info")[0], 404)
        self.assertEqual(self.request("POST", "/datasets/missing/search", {"query": "x"})[0], 404)
        self.assertEqual(self.request("GET", "/datasets/news20k/search")[0], 404)
        status, _, _ = self.request("POST", "/search", {"query": "x" * 2000})
        self.assertEqual(status, 413)
        self.assertEqual(len(self.upstream.received), 0)

    def test_unavailable_upstream_returns_sanitized_503(self):
        self.upstream.shutdown()
        self.upstream.server_close()
        status, payload, headers = self.request("GET", "/datasets/news20k/info")
        self.assertEqual(status, 503)
        self.assertEqual(payload["code"], "dataset_unavailable")
        self.assertNotIn("127.0.0.1", json.dumps(payload))
        self.assertEqual(headers["X-Search-Dataset"], "news20k")

    def test_config_validation_and_relative_example(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gateway.json"
            path.write_text(json.dumps({
                "schema_version": 1, "default_dataset": "mini", "port": 0,
                "datasets": [{"id": "mini", "label": "100k", "description": "",
                              "upstream": "http://127.0.0.1:18081",
                              "modes": ["lexical"], "examples": ["QQ"],
                              "n_docs": 100000}],
            }), encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(loaded.datasets["mini"].n_docs, 100000)
            raw = json.loads(path.read_text())
            raw["datasets"][0]["upstream"] = "http://user:secret@example.com"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(GatewayConfigError):
                load_config(path)
            raw["datasets"][0]["upstream"] = "http://127.0.0.1:not-a-port"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(GatewayConfigError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
