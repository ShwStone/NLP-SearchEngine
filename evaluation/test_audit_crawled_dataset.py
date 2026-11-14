from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation.audit_crawled_dataset import exact_body_overlap, main, overlap, raw_extraction_stats, residual_duplicates
from evaluation.clean_news import TextSignature, extract_article


class CrawledDatasetAuditTest(unittest.TestCase):
    def test_raw_extraction_cli_reports_partial_results_with_failure_status(self):
        raw = {"accepted_documents": 3, "counts": {"byte_hash_and_extraction_pass": 2, "missing_raw_file": 1}}
        result = {"corpus": {"raw_extraction": raw}}
        with (patch.object(sys, "argv", ["audit_crawled_dataset", "unused", "--replay-raw-extraction"]),
              patch("evaluation.audit_crawled_dataset.audit", return_value=result) as run_audit,
              patch("builtins.print") as output):
            self.assertEqual(main(), 2)
        self.assertTrue(run_audit.call_args.kwargs["check_raw_extraction"])
        self.assertEqual(json.loads(output.call_args.args[0]), result)

    def test_raw_extraction_requires_capture_bytes_and_frozen_article(self):
        url = "https://www.news.cn/example/c.html"
        text = "公开新闻正文测试材料" * 20
        raw = f'<html><title>测试报道-新华网</title><div id="detailContent">{text}</div></html>'.encode()
        raw_hash = hashlib.sha256(raw).hexdigest()
        article = extract_article(raw.decode(), url)
        self.assertTrue(article.accepted)
        documents = {"crawl-000001": {"title": article.title, "text": article.text}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "raw" / raw_hash[:2] / (raw_hash + ".html.gz")
            path.parent.mkdir(parents=True)
            path.write_bytes(gzip.compress(raw))
            sidecar = root / "article_sidecar.jsonl"
            sidecar.write_text(json.dumps({"outcome": "success", "representative_seq": 1, "source_url": url,
                                           "clean_metadata": asdict(article.metadata), "captures": [{"source_url": url,
                                           "http_bytes_sha256": raw_hash, "charset": "utf-8",
                                           "cache_path": path.relative_to(root).as_posix()}]}) + "\n", encoding="utf-8")

            passed = raw_extraction_stats(root, sidecar, documents, 1_000_000)
            self.assertEqual(passed["counts"], {"byte_hash_and_extraction_pass": 1})
            self.assertEqual(passed["selectors"], {"#detailContent": 1})

            path.write_bytes(gzip.compress(raw + b"changed"))
            self.assertEqual(raw_extraction_stats(root, sidecar, documents, 1_000_000)["counts"],
                             {"raw_hash_or_size_difference": 1})

            path.write_bytes(b"invalid gzip")
            self.assertEqual(raw_extraction_stats(root, sidecar, documents, 1_000_000)["counts"],
                             {"raw_read_failure": 1})

    def test_final_dedup_matches_pairs_across_bit_and_length_boundaries(self):
        fixtures = {
            "original": TextSignature("first", 0, 100),
            "exact": TextSignature("first", 0, 100),
            "near": TextSignature("second", (1 << 32) | (1 << 16) | 1, 90),
            "four_bits": TextSignature("third", (1 << 48) | (2 << 32) | (2 << 16) | 2, 100),
            "too_short": TextSignature("fourth", 0, 80),
        }
        with patch("evaluation.audit_crawled_dataset.signature", side_effect=fixtures.__getitem__):
            result = residual_duplicates(fixtures)
        self.assertEqual(result, {"documents": 5, "normalized_exact_pairs": 1,
                                  "nonexact_near_pairs": 2, "band_candidate_pairs": 6})

    def test_query_overlap_distinguishes_normalized_exact_and_near(self):
        self.assertEqual(
            overlap(["ＡＢＣＤＥＦＧＨＩ！"], ["abcdefghi", "abcdefghij", "unrelated"]),
            {"compared_pairs": 3, "normalized_exact": 1, "nonexact_bigram_jaccard_ge_0_8": 1},
        )

    def test_exact_body_overlap_checks_full_count_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            data = b'{"text":"same"}\n{"text":"same"}\n{"text":"same!"}\n'
            path.write_bytes(data)
            documents = {"1": {"text": "same"}, "2": {"text": "Same!"}}
            freeze = {"documents": 3, "corpus_sha256": hashlib.sha256(data).hexdigest()}

            result = exact_body_overlap(path, documents, freeze)
            self.assertEqual(result["official_documents"], 3)
            self.assertEqual(result["official_matching_documents"], 2)
            self.assertEqual(result["news_documents_with_exact_match"], 1)
            self.assertEqual(result["official_corpus_sha256"], freeze["corpus_sha256"])

            with self.assertRaisesRegex(ValueError, "document count or SHA-256"):
                exact_body_overlap(path, documents, {**freeze, "documents": 2})
            with self.assertRaisesRegex(ValueError, "document count or SHA-256"):
                exact_body_overlap(path, documents, {**freeze, "corpus_sha256": "0" * 64})
