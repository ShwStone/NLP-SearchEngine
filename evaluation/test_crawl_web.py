import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from http.client import IncompleteRead

from evaluation.crawl_web import (
    CrawlConfig,
    CrawlError,
    _read_url,
    crawl,
    finalize_annotations,
    normalize_url,
    write_annotation_template,
    write_corpus,
)


class CrawlWebTests(unittest.TestCase):
    def test_url_normalization_rejects_credentials_and_fragments(self):
        self.assertEqual(normalize_url("HTTPS://Example.org/a#part"), "https://example.org/a")
        self.assertEqual(normalize_url("https://例子.org/政策/文件?主题=人工智能"),
                         "https://xn--fsqu00a.org/%E6%94%BF%E7%AD%96/%E6%96%87%E4%BB%B6?%E4%B8%BB%E9%A2%98=%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD")
        with self.assertRaises(CrawlError):
            normalize_url("https://user:password@example.org/")
        with self.assertRaises(CrawlError):
            normalize_url("file:///tmp/secret")

    def test_bounded_same_domain_bfs_and_html_extraction(self):
        pages = {
            "https://example.org/": ("text/html", """
                <html><title>首页</title><body>第一篇 <a href='/next#x'>下一页</a>
                <a href='https://other.example/x'>外站</a><script>secret()</script></body></html>
            """),
            "https://example.org/next": ("text/html", "<title>下一页</title>第二篇"),
        }

        def fetch(url, _config):
            return pages[url]

        result = crawl(["https://example.org/"], CrawlConfig(max_documents=2,
                       max_depth=1, delay_seconds=0), fetch=fetch,
                       robots_allowed=lambda _url: True)
        self.assertEqual([page.doc_id for page in result], ["crawl-000001", "crawl-000002"])
        self.assertEqual(result[0].title, "首页")
        self.assertIn("第一篇", result[0].text)
        self.assertNotIn("secret", result[0].text)
        self.assertEqual(result[1].url, "https://example.org/next")

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(CrawlError):
            CrawlConfig(max_documents=0)
        with self.assertRaises(CrawlError):
            crawl([], CrawlConfig(delay_seconds=0))

    def test_incomplete_http_response_is_converted_to_crawl_error(self):
        class BrokenResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return "https://example.org/"

            class Headers:
                def get_content_type(self):
                    return "text/html"

                def get_content_charset(self):
                    return "utf-8"

            headers = Headers()

            def read(self, _size):
                raise IncompleteRead(b"partial", 1)

        with patch("evaluation.crawl_web.urlopen", return_value=BrokenResponse()):
            with self.assertRaises(CrawlError):
                _read_url("https://example.org/", CrawlConfig(delay_seconds=0))

    def test_annotation_template_and_finalization_require_human_positive_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pages = crawl(["https://example.org/"], CrawlConfig(max_depth=0, delay_seconds=0),
                          fetch=lambda _url, _config: ("text/html", "<title>题目</title>正文"),
                          robots_allowed=lambda _url: True)
            corpus = root / "corpus.jsonl"
            template = root / "annotations.template.jsonl"
            write_corpus(corpus, pages)
            write_annotation_template(template, pages)
            row = json.loads(template.read_text(encoding="utf-8"))
            row["query"] = "题目"
            row["relevance"] = 2
            annotations = root / "annotations.jsonl"
            annotations.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            queries = root / "queries.jsonl"
            qrels = root / "qrels.tsv"
            self.assertEqual(finalize_annotations(annotations, corpus, queries, qrels), 1)
            self.assertIn('"qid":"crawl-q000001"', queries.read_text(encoding="utf-8"))
            self.assertEqual(qrels.read_text(encoding="utf-8"), "crawl-q000001 0 crawl-000001 2\n")

            row["relevance"] = None
            annotations.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaises(CrawlError):
                finalize_annotations(annotations, corpus, queries, qrels)


if __name__ == "__main__":
    unittest.main()
