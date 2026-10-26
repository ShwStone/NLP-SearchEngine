import json
import tempfile
import unittest
from pathlib import Path

from evaluation.crawl_web import (
    CrawlConfig,
    CrawlError,
    crawl,
    finalize_annotations,
    normalize_url,
    write_annotation_template,
    write_corpus,
)


class CrawlWebTests(unittest.TestCase):
    def test_url_normalization_rejects_credentials_and_fragments(self):
        self.assertEqual(normalize_url("HTTPS://Example.org/a#part"), "https://example.org/a")
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
