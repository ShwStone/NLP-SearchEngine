"""混合检索接口 A2：离线向量产物、在线编码器/向量库、路由与融合配置。

不依赖任何模型或数据库 SDK。真实 Qdrant/HNSW 适配器和内存测试替身
都必须遵守这里的 ID、编码、归一化及排序约定。详见 contracts/RETRIEVAL.md。
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol, Sequence

from .reader import IndexReader

VECTOR_SCHEMA_VERSION = 1
VECTOR_MANIFEST = "vector_manifest.json"
VECTOR_BUILD_METADATA_KEY = "nlp_search_engine"


class RetrievalUnavailable(RuntimeError):
    """模型或向量后端未就绪/运行失败；强制语义模式转 HTTP 503。"""


@dataclass(frozen=True)
class EmbeddingSpec:
    model_id: str
    revision: str              # 必须固定版本，禁止仅填 latest
    dimension: int
    max_tokens: int = 512      # 由具体编码器按自身 tokenizer 右侧截断
    normalize: bool = True
    distance: str = "cosine"
    document_template: str = "title\ntext"
    query_prefix: str = ""
    document_prefix: str = ""

    def __post_init__(self):
        if not self.model_id or not self.revision or self.revision == "latest":
            raise ValueError("编码器必须指定 model_id 与固定 revision")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("dimension 必须是正整数")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens 必须是正整数")
        if self.normalize is not True or self.distance != "cosine":
            raise ValueError("v1 只支持 L2 归一化向量与 cosine")
        if self.document_template != "title\ntext":
            raise ValueError("v1 文档编码输入固定为 title + 换行 + text")


@dataclass(frozen=True)
class VectorManifest:
    corpus_sha256: str         # 按内部 docid 排序的 ID、标题、正文摘要
    n_docs: int
    embedding: EmbeddingSpec
    collection: str
    backend: str = "qdrant-hnsw"
    index_params: dict = field(default_factory=dict)
    schema_version: int = VECTOR_SCHEMA_VERSION
    snapshot_files: dict = field(default_factory=dict)  # 相对文件名 -> SHA256；memory-exact 可为空

    def __post_init__(self):
        if self.schema_version != VECTOR_SCHEMA_VERSION:
            raise ValueError("不支持的向量契约版本")
        if type(self.n_docs) is not int or self.n_docs <= 0:
            raise ValueError("向量索引必须包含文档")
        if (len(self.corpus_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.corpus_sha256)):
            raise ValueError("corpus_sha256 必须是 SHA256 十六进制摘要")
        if not self.collection or self.backend not in ("qdrant-hnsw", "memory-exact"):
            raise ValueError("必须指定 collection 与受支持的 backend")
        for name, digest in self.snapshot_files.items():
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError("快照只允许产物目录内的文件名")
            if (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
                raise ValueError("快照必须提供 SHA256 校验和")

    def to_dict(self) -> dict:
        return asdict(self)

    def build_metadata(self) -> dict:
        """Metadata persisted with the collection before snapshot creation.

        Snapshot hashes are excluded: embedding them in the snapshot itself
        would create a circular checksum dependency. All build identity fields
        (including the full embedding spec) remain independently verifiable.
        """
        value = self.to_dict()
        value.pop("snapshot_files")
        return value

    @classmethod
    def from_dict(cls, value: dict) -> "VectorManifest":
        value = dict(value)
        value["embedding"] = EmbeddingSpec(**value["embedding"])
        return cls(**value)


@dataclass(frozen=True)
class VectorRecord:
    docid: int                # Qdrant point ID = 倒排索引内部 docid
    doc_id: str                # 外部评测 ID，作为 payload 保存
    vector: Sequence[float]


@dataclass(frozen=True)
class VectorHit:
    docid: int
    doc_id: str
    score: float               # cosine，越大越相关；不是距离


class TextEncoder(Protocol):
    spec: EmbeddingSpec

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        """应用 document_prefix、tokenizer 截断、L2 归一化；保持输入顺序。"""
        ...

    def encode_query(self, query: str) -> list[float]:
        """应用 query_prefix、相同 tokenizer/截断规则与 L2 归一化。"""
        ...


class VectorWriter(Protocol):
    def replace(self, manifest: VectorManifest, records: Iterable[VectorRecord]) -> VectorManifest:
        """在新 collection 构建全部记录，成功后发布快照；失败不得覆盖旧库。

        必须消费完整 records 并检查数量、唯一 ID、向量维度与归一化。
        返回补齐 snapshot_files 的 manifest，不得改变其他字段。快照写到调用方
        约定的产物目录；memory-exact 可无快照。不得提前暴露未完成的 collection。
        """
        ...


class VectorReader(Protocol):
    def manifest(self) -> VectorManifest:
        ...

    def search(self, query_vector: Sequence[float], limit: int) -> list[VectorHit]:
        """最多 limit 条唯一命中，score 降序、同分内部 docid 升序。

        返回的 docid/doc_id 必须与同版倒排索引一致；失败抛 RetrievalUnavailable。
        """
        ...


@dataclass(frozen=True)
class RetrievalConfig:
    candidate_k: int = 200     # 每路候选池；offset + k 不得超过它
    rrf_c: float = 60.0
    min_vocab_coverage: float = 0.7
    min_relative_gap: float = 0.1
    policy_version: str = "rules-v1"
    adaptive_policy: str = "rules-v1"

    def __post_init__(self):
        if self.adaptive_policy not in ("rules-v1", "recall-v1", "hybrid-v1"):
            raise ValueError("unknown adaptive_policy")
        if type(self.candidate_k) is not int or self.candidate_k < 1:
            raise ValueError("candidate_k 必须是正整数")
        if not math.isfinite(self.rrf_c) or self.rrf_c <= 0:
            raise ValueError("rrf_c 必须为有限正数")
        for value in (self.min_vocab_coverage, self.min_relative_gap):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("路由阈值必须在 [0, 1]")


def corpus_fingerprint(reader: IndexReader) -> str:
    """独立于磁盘格式的语料快照标识；离线和在线用同一算法。"""
    digest = hashlib.sha256()
    for docid in range(reader.n_docs):
        doc = reader.doc(docid)
        row = [docid, doc.doc_id, doc.title, doc.text]
        digest.update((json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                       + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_vector(vector: Sequence[float], spec: EmbeddingSpec) -> None:
    if len(vector) != spec.dimension:
        raise ValueError("向量维度与 EmbeddingSpec 不一致")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in vector):
        raise ValueError("向量必须包含有限数值")
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError("向量必须已 L2 归一化且不能为零向量")


def check_alignment(reader: IndexReader, vectors: VectorReader, encoder: TextEncoder) -> None:
    manifest = vectors.manifest()
    if manifest.backend == "qdrant-hnsw" and not manifest.snapshot_files:
        raise ValueError("生产向量库缺快照校验信息")
    if manifest.embedding != encoder.spec:
        raise ValueError("在线编码器与离线 embedding 配置不一致")
    if (manifest.n_docs != reader.n_docs
            or manifest.corpus_sha256 != corpus_fingerprint(reader)):
        raise ValueError("向量库与倒排索引不是同一语料/ID 快照")
