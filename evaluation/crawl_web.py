"""Bounded, reproducible web-crawl and annotation workflow.

This module deliberately stops before relevance judgments.  Crawled pages are
an input corpus; queries and qrels must be supplied or reviewed by a person.
The default crawler is same-domain, HTTP(S)-only, robots-aware, size-bounded,
and deterministic (FIFO traversal with normalized URLs).  It never writes
credentials, cookies, or response headers to the output corpus.

Examples::

    python -m evaluation.crawl_web crawl --seed https://example.org/ \
      --out-corpus data/crawl/corpus.jsonl \
      --annotation-template data/crawl/annotations.template.jsonl

    # After manually filling query/relevance fields:
    python -m evaluation.crawl_web finalize --corpus data/crawl/corpus.jsonl \
      --annotations data/crawl/annotations.jsonl \
      --queries-out data/crawl/queries.jsonl \
      --qrels-out data/crawl/qrels.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


class CrawlError(ValueError):
    """Invalid crawl input or an annotation that cannot be evaluated safely."""


@dataclass(frozen=True)
class CrawlConfig:
    max_documents: int = 100
    max_depth: int = 1
    max_bytes: int = 1_000_000
    timeout: float = 10.0
    delay_seconds: float = 0.5
    same_domain: bool = True
    user_agent: str = "NLP-SearchEngine-course-crawler/1.0"
    allow_robots_failure: bool = False

    def __post_init__(self) -> None:
        if type(self.max_documents) is not int or self.max_documents <= 0:
            raise CrawlError("max_documents 必须是正整数")
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise CrawlError("max_depth 必须是非负整数")
        if type(self.max_bytes) is not int or self.max_bytes <= 0:
            raise CrawlError("max_bytes 必须是正整数")
        if self.timeout <= 0 or self.delay_seconds < 0:
            raise CrawlError("timeout 必须为正数，delay_seconds 必须为非负数")
        if not self.user_agent or any(ch in self.user_agent for ch in "\r\n"):
            raise CrawlError("user_agent 无效")


@dataclass(frozen=True)
class CrawledPage:
    doc_id: str
    url: str
    title: str
    text: str
    links: tuple[str, ...]


def normalize_url(value: str) -> str:
    """Return a safe canonical URL, rejecting credentials and non-web URLs."""

    if not isinstance(value, str) or not value.strip():
        raise CrawlError("URL 不能为空")
    parts = urlsplit(value.strip())
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        raise CrawlError("只允许 HTTP(S) URL")
    if parts.username is not None or parts.password is not None:
        raise CrawlError("URL 不允许携带凭据")
    host = parts.hostname
    if not host:
        raise CrawlError("URL 缺少主机名")
    # urlsplit().port also validates malformed ports; do not preserve fragments.
    try:
        port = parts.port
    except ValueError as error:
        raise CrawlError("URL 端口无效") from error
    netloc = host.lower()
    if ":" in host and not host.startswith("["):
        netloc = f"[{netloc}]"
    if port is not None and not ((parts.scheme.lower() == "http" and port == 80)
                                 or (parts.scheme.lower() == "https" and port == 443)):
        netloc += f":{port}"
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript", "template", "svg"}:
            self._skip_depth += 1
        if tag == "a" and self._skip_depth == 0:
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "template", "svg"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        clean = re.sub(r"\s+", " ", data).strip()
        if not clean:
            return
        if self._in_title:
            self.title_parts.append(clean)
        self.text_parts.append(clean)

    def parse(self, base_url: str) -> tuple[str, str, tuple[str, ...]]:
        links: list[str] = []
        for raw in self.links:
            try:
                candidate = normalize_url(urljoin(base_url, raw))
            except CrawlError:
                continue
            if candidate not in links:
                links.append(candidate)
        return (" ".join(self.title_parts).strip(),
                " ".join(self.text_parts).strip(), tuple(links))


def _read_url(url: str, config: CrawlConfig) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": config.user_agent, "Accept": "text/html"})
    try:
        with urlopen(request, timeout=config.timeout) as response:
            final_url = normalize_url(response.geturl())
            if urlsplit(final_url).netloc.lower() != urlsplit(url).netloc.lower():
                raise CrawlError("拒绝跨主机重定向")
            content_type = response.headers.get_content_type()
            chunks: list[bytes] = []
            total = 0
            while total <= config.max_bytes:
                chunk = response.read(min(64 * 1024, config.max_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
            if total > config.max_bytes:
                raise CrawlError(f"响应超过 max_bytes: {url}")
            charset = response.headers.get_content_charset() or "utf-8"
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise CrawlError(f"抓取失败 {url}: {error}") from error
    try:
        return content_type, b"".join(chunks).decode(charset, errors="replace")
    except (LookupError, UnicodeError) as error:
        raise CrawlError(f"无法解码响应 {url}") from error


def _robots_checker(config: CrawlConfig) -> Callable[[str], bool]:
    cache: dict[str, RobotFileParser | None] = {}

    def allowed(url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in cache:
            parser = RobotFileParser(f"{origin}/robots.txt")
            try:
                parser.read()
            except (HTTPError, URLError, TimeoutError, OSError):
                cache[origin] = None
            else:
                cache[origin] = parser
        parser = cache[origin]
        if parser is None:
            return config.allow_robots_failure
        return parser.can_fetch(config.user_agent, url)

    return allowed


def crawl(seed_urls: Iterable[str], config: CrawlConfig | None = None,
          *, fetch: Callable[[str, CrawlConfig], tuple[str, str]] | None = None,
          robots_allowed: Callable[[str], bool] | None = None) -> list[CrawledPage]:
    """Crawl a bounded corpus; injected fetch/robots callbacks make tests offline."""

    config = config or CrawlConfig()
    seeds = [normalize_url(url) for url in seed_urls]
    if not seeds:
        raise CrawlError("至少需要一个 seed URL")
    allowed_hosts = {urlsplit(url).netloc.lower() for url in seeds}
    fetch = fetch or _read_url
    robots_allowed = robots_allowed or _robots_checker(config)
    queue: deque[tuple[str, int]] = deque((url, 0) for url in seeds)
    queued = set(url for url, _ in queue)
    pages: list[CrawledPage] = []
    while queue and len(pages) < config.max_documents:
        url, depth = queue.popleft()
        if not robots_allowed(url):
            continue
        try:
            content_type, body = fetch(url, config)
        except CrawlError:
            continue
        if content_type and content_type.lower() not in ("text/html", "application/xhtml+xml"):
            continue
        parser = _PageParser()
        try:
            parser.feed(body)
            parser.close()
        except (ValueError, AssertionError):
            continue
        title, text, links = parser.parse(url)
        if not text:
            continue
        doc_id = f"crawl-{len(pages) + 1:06d}"
        pages.append(CrawledPage(doc_id, url, title or url, text, links))
        if depth >= config.max_depth:
            continue
        for link in links:
            if config.same_domain and urlsplit(link).netloc.lower() not in allowed_hosts:
                continue
            if link in queued:
                continue
            queued.add(link)
            queue.append((link, depth + 1))
        if config.delay_seconds:
            time.sleep(config.delay_seconds)
    return pages


def write_corpus(path: str | Path, pages: Sequence[CrawledPage]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for page in pages:
            handle.write(json.dumps({"doc_id": page.doc_id, "title": page.title,
                                     "text": page.text, "url": page.url},
                                    ensure_ascii=False, separators=(",", ":")) + "\n")


def write_annotation_template(path: str | Path, pages: Sequence[CrawledPage]) -> None:
    """Write an explicit human-labeling template; this is not an evaluation qrels."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, page in enumerate(pages, 1):
            row = {"qid": f"crawl-q{index:06d}", "query": "",
                   "doc_id": page.doc_id, "relevance": None,
                   "source_url": page.url,
                   "label_status": "needs_human_annotation"}
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _load_corpus_ids(path: str | Path) -> set[str]:
    ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise CrawlError(f"corpus 第 {line_no} 行不是 JSON") from error
            doc_id = row.get("doc_id") if isinstance(row, dict) else None
            if not isinstance(doc_id, str) or not doc_id or doc_id in ids:
                raise CrawlError(f"corpus 第 {line_no} 行 doc_id 无效或重复")
            ids.add(doc_id)
    if not ids:
        raise CrawlError("corpus 为空")
    return ids


def finalize_annotations(annotation_path: str | Path, corpus_path: str | Path,
                         queries_out: str | Path, qrels_out: str | Path) -> int:
    """Validate human judgments and write evaluator-compatible JSONL/TSV."""

    corpus_ids = _load_corpus_ids(corpus_path)
    records: dict[str, dict] = {}
    with Path(annotation_path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise CrawlError(f"annotations 第 {line_no} 行不是 JSON") from error
            if not isinstance(row, dict):
                raise CrawlError(f"annotations 第 {line_no} 行必须是对象")
            qid, query, doc_id, relevance = (row.get("qid"), row.get("query"),
                                              row.get("doc_id"), row.get("relevance"))
            if not isinstance(qid, str) or not qid or not isinstance(query, str) or not query.strip():
                raise CrawlError(f"annotations 第 {line_no} 行需要非空 qid/query")
            if doc_id not in corpus_ids or not isinstance(doc_id, str):
                raise CrawlError(f"annotations 第 {line_no} 行引用未知 doc_id")
            if type(relevance) is not int or not 0 <= relevance <= 2:
                raise CrawlError(f"annotations 第 {line_no} 行 relevance 必须是 0、1 或 2")
            existing = records.setdefault(qid, {"query": query.strip(), "judgments": {}})
            if existing["query"] != query.strip() or doc_id in existing["judgments"]:
                raise CrawlError(f"annotations 第 {line_no} 行 qid/query 或 doc_id 重复")
            existing["judgments"][doc_id] = relevance
    if not records:
        raise CrawlError("annotations 为空")
    if any(not any(value > 0 for value in row["judgments"].values())
           for row in records.values()):
        raise CrawlError("每个 qid 至少需要一个正相关人工标注")
    query_path, qrels_path = Path(queries_out), Path(qrels_out)
    query_path.parent.mkdir(parents=True, exist_ok=True)
    qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with query_path.open("w", encoding="utf-8", newline="\n") as queries, \
            qrels_path.open("w", encoding="utf-8", newline="\n") as qrels:
        for qid in sorted(records):
            row = records[qid]
            queries.write(json.dumps({"qid": qid, "query": row["query"]},
                                     ensure_ascii=False, separators=(",", ":")) + "\n")
            for doc_id in sorted(row["judgments"]):
                qrels.write(f"{qid} 0 {doc_id} {row['judgments'][doc_id]}\n")
    return len(records)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建有边界的网络爬取测试集")
    sub = parser.add_subparsers(dest="command", required=True)
    crawl_parser = sub.add_parser("crawl", help="抓取 HTML 并输出未标注 corpus")
    crawl_parser.add_argument("--seed", action="append", required=True)
    crawl_parser.add_argument("--out-corpus", required=True, type=Path)
    crawl_parser.add_argument("--annotation-template", type=Path)
    crawl_parser.add_argument("--max-documents", type=int, default=100)
    crawl_parser.add_argument("--max-depth", type=int, default=1)
    crawl_parser.add_argument("--max-bytes", type=int, default=1_000_000)
    crawl_parser.add_argument("--timeout", type=float, default=10.0)
    crawl_parser.add_argument("--delay-seconds", type=float, default=0.5)
    crawl_parser.add_argument("--allow-robots-failure", action="store_true")
    finalize = sub.add_parser("finalize", help="校验人工标注并写 queries/qrels")
    finalize.add_argument("--corpus", required=True, type=Path)
    finalize.add_argument("--annotations", required=True, type=Path)
    finalize.add_argument("--queries-out", required=True, type=Path)
    finalize.add_argument("--qrels-out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "crawl":
            config = CrawlConfig(max_documents=args.max_documents, max_depth=args.max_depth,
                                 max_bytes=args.max_bytes, timeout=args.timeout,
                                 delay_seconds=args.delay_seconds,
                                 allow_robots_failure=args.allow_robots_failure)
            pages = crawl(args.seed, config)
            write_corpus(args.out_corpus, pages)
            if args.annotation_template:
                write_annotation_template(args.annotation_template, pages)
            print(json.dumps({"documents": len(pages), "corpus": str(args.out_corpus)},
                             ensure_ascii=False))
        else:
            count = finalize_annotations(args.annotations, args.corpus,
                                         args.queries_out, args.qrels_out)
            print(json.dumps({"queries": count, "queries_out": str(args.queries_out),
                              "qrels_out": str(args.qrels_out)}, ensure_ascii=False))
    except (CrawlError, OSError) as error:
        print(f"crawl error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
