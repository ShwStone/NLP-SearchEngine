"""离线统一构建入口（依赖注入骨架，不绑定模型/数据库 SDK）。"""

from pathlib import Path

from contracts.index_format import BM25Spec
from contracts.retrieval import TextEncoder, VectorWriter
from .build import build
from .disk_documents import DiskDocuments
from .vector_build import build_vectors


def build_hybrid(corpus_path: Path, out_dir: Path, dataset: str,
                 encoder: TextEncoder, writer: VectorWriter, collection: str,
                 batch_size: int = 32, backend: str = "qdrant-hnsw",
                 index_params: dict | None = None, bm25: BM25Spec | None = None) -> dict:
    """一个新的产物目录 = bm25/ + vector_manifest.json + 后端向量库快照。

    BM25 构建指倒排表、df/tf、长度统计与位置，不是提前计算查询得分。
    以 vector_manifest.json 成功发布作为双索引构建完成标志。
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("双索引构建必须使用新的空输出目录")
    lexical = build(corpus_path, out_dir / "bm25", dataset, bm25)
    with DiskDocuments(out_dir / "bm25") as documents:
        vectors = build_vectors(documents, encoder, writer, out_dir,
                                collection, batch_size, backend, index_params)
    return {"lexical": lexical.to_dict(), "vector": vectors.to_dict()}
