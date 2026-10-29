"""离线统一构建入口（依赖注入骨架，不绑定模型/数据库 SDK）。"""

import shutil
import tempfile
import time
from pathlib import Path

from contracts.index_format import BM25Spec
from contracts.retrieval import TextEncoder, VectorWriter
from .build import build
from .disk_index import DiskIndex
from .provenance import (PROVENANCE_FILE, REPORT_FILE, make_provenance,
                         make_report, write_provenance, write_report)
from .vector_build import build_vectors


def build_hybrid(corpus_path: Path, out_dir: Path, dataset: str,
                 encoder: TextEncoder, writer: VectorWriter, collection: str,
                 batch_size: int = 32, backend: str = "qdrant-hnsw",
                 index_params: dict | None = None, bm25: BM25Spec | None = None) -> dict:
    """一个新的产物目录 = bm25/ + vector_manifest.json + 后端向量库快照。

    BM25 构建指倒排表、df/tf、长度统计与位置，不是提前计算查询得分。
    所有文件先写入同文件系统的 staging root；以根目录原子改名作为双索引构建完成标志。
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("双索引构建必须使用新的空输出目录")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    phases = {}
    try:
        phase_start = time.perf_counter()
        lexical = build(corpus_path, stage / "bm25", dataset, bm25)
        phases["bm25"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        reader = DiskIndex(stage / "bm25")
        vectors = build_vectors(reader, encoder, writer, stage,
                                collection, batch_size, backend, index_params)
        phases["vector"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        provenance = make_provenance(corpus_path, dataset, reader,
                                      vector_manifest=vectors)
        write_provenance(stage / PROVENANCE_FILE, provenance)
        phases["provenance"] = time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        report = make_report(stage, dataset=dataset, n_docs=reader.n_docs,
                             phase_seconds=phases)
        write_report(stage / REPORT_FILE, report)
        phases["report"] = time.perf_counter() - phase_start
        if out_dir.exists():
            out_dir.rmdir()
        stage.replace(out_dir)
        return {"lexical": lexical.to_dict(), "vector": vectors.to_dict(),
                "provenance": provenance, "report": report}
    except Exception:
        # A failed build never touches an existing published directory; all
        # partial files are confined to and removed from the staging root.
        shutil.rmtree(stage, ignore_errors=True)
        raise
