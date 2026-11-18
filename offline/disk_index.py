"""DiskIndex：读离线组产出的索引目录，实现 IndexReader 协议。

加载时的两道闸门：
1. manifest 里 analyzer.config_hash 必须等于当前 contracts.analyzer 的配置哈希，
   否则拒绝启动（并行期 90% 的"查不到东西"都是分词器不一致）；
2. 各文件 sha256 与 manifest 一致（可选关闭，CI 冒烟时省时间）。

v1 兼容路径全量读进内存；compact-v1 目录自动选择按需读取的 PackedIndex，
在线调用方仍只依赖 DiskIndex 入口和 IndexReader 协议。
"""

import json
import math
from pathlib import Path
from typing import Iterable, Optional

from contracts import analyzer
from contracts.index_format import (
    DOCS, OFFSETS, POSTINGS, STATS, load_manifest, verify_files,
)
from contracts.reader import Document


class DiskIndex:
    def __new__(cls, index_dir, verify_checksums: bool = True, *, cache_bytes=None):
        root = Path(index_dir)
        if cls is DiskIndex and (root / "compact.json").exists():
            if (root / "manifest.json").exists():
                raise ValueError("ambiguous index directory: both v1 and compact manifests")
            from .packed_index import PackedIndex
            # Compact is a production artifact: its full checksum gate cannot
            # be disabled using the legacy v1 compatibility option.
            options = {} if cache_bytes is None else {"cache_bytes": cache_bytes}
            return PackedIndex(root, **options)
        return super().__new__(cls)

    def __init__(self, index_dir, verify_checksums: bool = True, *, cache_bytes=None):
        index_dir = Path(index_dir)
        self._manifest = load_manifest(index_dir)
        if type(self._manifest.n_docs) is not int or self._manifest.n_docs <= 0:
            raise ValueError("manifest.n_docs 必须为正整数")
        if self._manifest.analyzer_hash != analyzer.config_hash():
            raise ValueError(
                "分词器配置不一致：索引由 "
                f"{self._manifest.analyzer_name}@{self._manifest.analyzer_version} 构建，"
                f"当前 analyzer 为 {analyzer.NAME}@{analyzer.VERSION}。"
                "请确认两侧 contracts/ 一致后重建索引。"
            )
        if verify_checksums:
            verify_files(index_dir, self._manifest)

        # Checksums establish that files belong to one build, but do not make
        # malformed JSONL safe to consume.  Validate IDs and row cardinality
        # here so a partial/hand-edited index fails at startup instead of
        # surfacing as a confusing KeyError during a later query.
        raw_docs: dict = {}
        with open(index_dir / DOCS, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    d = json.loads(line)
                    docid = d["docid"]
                    doc_id, title, text = d["doc_id"], d["title"], d["text"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"docs.jsonl 第 {line_no} 行格式错误") from error
                if (type(docid) is not int or not 0 <= docid < self._manifest.n_docs
                        or docid in raw_docs):
                    raise ValueError(f"docs.jsonl 第 {line_no} 行 docid 无效或重复")
                if (not isinstance(doc_id, str) or not doc_id
                        or not isinstance(title, str) or not isinstance(text, str)):
                    raise ValueError(f"docs.jsonl 第 {line_no} 行文档字段无效")
                raw_docs[docid] = d
        expected_ids = set(range(self._manifest.n_docs))
        if set(raw_docs) != expected_ids:
            raise ValueError("docs.jsonl 必须完整覆盖 0..n_docs-1")

        offsets: dict = {}
        with open(index_dir / OFFSETS, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    d = json.loads(line)
                    docid, spans = d["docid"], d["spans"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"offsets.jsonl 第 {line_no} 行格式错误") from error
                if (type(docid) is not int or docid not in expected_ids
                        or docid in offsets or not isinstance(spans, list)):
                    raise ValueError(f"offsets.jsonl 第 {line_no} 行无效或重复")
                checked = []
                previous_token = -1
                for span in spans:
                    if (not isinstance(span, list) or len(span) != 4
                            or any(type(value) is not int for value in span)):
                        raise ValueError(f"offsets.jsonl 第 {line_no} 行 span 格式错误")
                    token_i, token_end, char_start, char_end = span
                    if (token_end != token_i + 1 or token_i != len(checked)
                            or not 0 <= char_start < char_end <= len(raw_docs[docid]["text"])):
                        raise ValueError(f"offsets.jsonl 第 {line_no} 行 span 越界")
                    previous_token = token_i
                    checked.append((char_start, char_end))
                offsets[docid] = checked
        if set(offsets) != expected_ids:
            raise ValueError("offsets.jsonl 必须完整覆盖 0..n_docs-1")
        self._docs: dict = {
            docid: Document(
                docid=docid, doc_id=d["doc_id"], title=d["title"],
                text=d["text"], offsets=offsets.get(docid, []),
            )
            for docid, d in raw_docs.items()
        }
        self._postings: dict = {}
        with open(index_dir / POSTINGS, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    d = json.loads(line)
                    term, df, docs = d["term"], d["df"], d["docs"]
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(f"postings.jsonl 第 {line_no} 行格式错误") from error
                if (not isinstance(term, str) or not term or type(df) is not int
                        or df < 1 or not isinstance(docs, list) or term in self._postings):
                    raise ValueError(f"postings.jsonl 第 {line_no} 行字段无效或重复")
                parsed = []
                previous_docid = -1
                for row in docs:
                    if (not isinstance(row, list) or len(row) != 3
                            or any(type(value) is not int for value in row[:2])
                            or not isinstance(row[2], list)):
                        raise ValueError(f"postings.jsonl 第 {line_no} 行 posting 格式错误")
                    docid, tf, positions = row
                    if (not 0 <= docid < self._manifest.n_docs or docid <= previous_docid
                            or tf != len(positions) or tf < 1
                            or any(type(pos) is not int for pos in positions)
                            or positions != sorted(set(positions))
                            or any(pos < 0 for pos in positions)):
                        raise ValueError(f"postings.jsonl 第 {line_no} 行 posting 值无效")
                    previous_docid = docid
                    parsed.append((docid, tf, positions))
                if df != len(parsed):
                    raise ValueError(f"postings.jsonl 第 {line_no} 行 df 不匹配")
                self._postings[term] = parsed
        with open(index_dir / STATS, encoding="utf-8") as f:
            try:
                stats = json.load(f)
                lengths, avgdl = stats["doc_lengths"], stats["avgdl"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError("stats.json 格式错误") from error
        if (not isinstance(lengths, dict) or not isinstance(avgdl, (int, float))
                or isinstance(avgdl, bool) or not math.isfinite(avgdl)):
            raise ValueError("stats.json 字段无效")
        try:
            self._doc_lengths = {int(k): v for k, v in lengths.items()}
        except (TypeError, ValueError) as error:
            raise ValueError("stats.json doc_lengths 格式错误") from error
        if (set(self._doc_lengths) != expected_ids
                or any(type(v) is not int or v < 0 for v in self._doc_lengths.values())
                or not math.isclose(avgdl, self._manifest.avgdl, rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("stats.json 与 manifest 的文档统计不一致")
        for term, postings in self._postings.items():
            for docid, _, positions in postings:
                if any(position >= self._doc_lengths[docid] for position in positions):
                    raise ValueError(f"postings.jsonl 词项 {term!r} 的位置超出文档长度")
        self.n_docs = self._manifest.n_docs
        self.avgdl = self._manifest.avgdl

    def postings(self, term: str) -> Optional[list]:
        return self._postings.get(term)

    def df(self, term: str) -> int:
        pl = self._postings.get(term)
        return len(pl) if pl else 0

    def doc_length(self, docid: int) -> int:
        return self._doc_lengths[docid]

    def doc(self, docid: int) -> Document:
        return self._docs[docid]

    def vocab(self) -> Iterable[str]:
        return self._postings.keys()

    def manifest(self) -> dict:
        return self._manifest.to_dict()
