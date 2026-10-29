"""接口 A：索引产物的磁盘格式（离线 → 在线）。

索引是一个目录，包含 5 个文件：

    manifest.json    版本、数据集、文档数、平均分词长度、分词器配置哈希、各文件 sha256
    docs.jsonl       {"docid": int, "doc_id": str, "title": str, "text": str}
    postings.jsonl   {"term": str, "df": int, "docs": [[docid, tf, [pos...]], ...]}  按 term 排序
    offsets.jsonl    {"docid": int, "spans": [[tok_i, tok_i+1, char_start, char_end], ...]}
    stats.json       {"doc_lengths": {"docid": int, ...}, "avgdl": float}

离线组只写、在线组只读。任何一方改格式，先改这里并 bump SCHEMA_VERSION。
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

MANIFEST = "manifest.json"
DOCS = "docs.jsonl"
POSTINGS = "postings.jsonl"
OFFSETS = "offsets.jsonl"
STATS = "stats.json"

REQUIRED_FILES = [DOCS, POSTINGS, OFFSETS, STATS]


@dataclass(frozen=True)
class BM25Spec:
    k1: float = 1.2
    b: float = 0.75
    idf: str = "log1p-lucene"

    def __post_init__(self):
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("BM25 k1 必须为有限正数")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("BM25 b 必须在 [0, 1]")
        if self.idf != "log1p-lucene":
            raise ValueError("当前仅支持 log1p-lucene IDF")

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "idf": self.idf}


@dataclass(frozen=True)
class Manifest:
    dataset: str
    n_docs: int
    avgdl: float
    analyzer_name: str
    analyzer_version: str
    analyzer_hash: str
    files: dict = field(default_factory=dict)  # 文件名 -> sha256
    schema_version: int = SCHEMA_VERSION
    bm25: BM25Spec = field(default_factory=BM25Spec)  # v1 可选扩展，旧产物用默认参数

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "analyzer": {
                "name": self.analyzer_name,
                "version": self.analyzer_version,
                "config_hash": self.analyzer_hash,
            },
            "files": self.files,
            "bm25": self.bm25.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "Manifest":
        try:
            m = Manifest(
                schema_version=d["schema_version"],
                dataset=d["dataset"],
                n_docs=d["n_docs"],
                avgdl=d["avgdl"],
                analyzer_name=d["analyzer"]["name"],
                analyzer_version=d["analyzer"]["version"],
                analyzer_hash=d["analyzer"]["config_hash"],
                files=d.get("files", {}),
                bm25=BM25Spec(**d.get("bm25", {})),
            )
        except KeyError as e:
            raise ValueError(f"manifest 缺字段: {e}") from e
        if m.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version={m.schema_version}，本代码只支持 {SCHEMA_VERSION}；"
                "请先确认两侧 contracts/ 版本一致"
            )
        return m


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(index_dir: Path) -> Manifest:
    with open(Path(index_dir) / MANIFEST, encoding="utf-8") as f:
        return Manifest.from_dict(json.load(f))


def verify_files(index_dir: Path, manifest: Manifest, *, require_hashes: bool = False) -> None:
    """校验各文件 sha256；生产交付可要求每个文件都有摘要。

    默认保持旧 v1 产物兼容。正式发布校验必须传入 ``require_hashes=True``。
    """
    for name in REQUIRED_FILES:
        p = Path(index_dir) / name
        if not p.exists():
            raise ValueError(f"索引缺文件: {p}")
        expected = manifest.files.get(name)
        if require_hashes and (not isinstance(expected, str) or len(expected) != 64
                                or any(ch not in "0123456789abcdef" for ch in expected)):
            raise ValueError(f"{name} 缺少有效 SHA256；生产交付不得使用兼容旧产物")
        if expected is not None and sha256_of(p) != expected:
            raise ValueError(f"{name} 校验和不匹配：索引文件与 manifest 不是同一次构建")
