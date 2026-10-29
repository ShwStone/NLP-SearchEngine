"""索引构建（离线组）。

用法：
    python -m offline.build --corpus fixtures/tiny_corpus.jsonl \
        --out fixtures/tiny_index --dataset tiny

产物符合 contracts/index_format.py 的接口 A 约定。
验收：连跑两次 manifest 中各文件 sha256 一致（确定性构建）。
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

from contracts import analyzer
from contracts.conformance import build_expected
from contracts.index_format import (
    DOCS, MANIFEST, OFFSETS, POSTINGS, STATS,
    BM25Spec, Manifest, sha256_of,
)


def build(corpus_path: Path, out_dir: Path, dataset: str,
          bm25: BM25Spec | None = None) -> Manifest:
    """Build A1 in a sibling staging directory, then publish it atomically.

    This v1 builder still materializes the conformance oracle in memory; the
    RFC's large-corpus streaming path is a separate implementation and must
    not be inferred from this legacy entry point.
    """
    corpus = []
    with open(corpus_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"语料第 {line_number} 行不是合法 JSON") from error
            if (not isinstance(row, dict) or not isinstance(row.get("doc_id"), str)
                    or not row["doc_id"] or not isinstance(row.get("title"), str)
                    or not isinstance(row.get("text"), str)):
                raise ValueError(f"语料第 {line_number} 行需要非空 doc_id、字符串 title/text")
            corpus.append(row)
    ids = [row["doc_id"] for row in corpus]
    if not corpus:
        raise ValueError("语料不能为空")
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id 必须唯一，不能构建无法对齐的双索引")
    exp = build_expected(corpus)

    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("索引输出目录必须不存在或为空；不会覆盖既有产物")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=out_dir.parent))
    try:
        with open(stage / DOCS, "w", encoding="utf-8") as f:
            for docid in range(exp.n_docs):
                doc_id, title, text, _ = exp.docs[docid]
                f.write(json.dumps(
                    {"docid": docid, "doc_id": doc_id, "title": title, "text": text},
                    ensure_ascii=False) + "\n")
        with open(stage / POSTINGS, "w", encoding="utf-8") as f:
            for term in sorted(exp.postings):
                pl = exp.postings[term]
                f.write(json.dumps({"term": term, "df": len(pl), "docs": pl},
                                   ensure_ascii=False) + "\n")
        with open(stage / OFFSETS, "w", encoding="utf-8") as f:
            for docid in range(exp.n_docs):
                spans = exp.docs[docid][3]
                f.write(json.dumps({"docid": docid, "spans": spans},
                                   ensure_ascii=False) + "\n")
        with open(stage / STATS, "w", encoding="utf-8") as f:
            json.dump({"doc_lengths": {str(k): v for k, v in exp.doc_lengths.items()},
                       "avgdl": exp.avgdl}, f)

        files = {name: sha256_of(stage / name)
                 for name in (DOCS, POSTINGS, OFFSETS, STATS)}
        manifest = Manifest(
            dataset=dataset,
            n_docs=exp.n_docs,
            avgdl=exp.avgdl,
            analyzer_name=analyzer.NAME,
            analyzer_version=analyzer.VERSION,
            analyzer_hash=analyzer.config_hash(),
            files=files,
            bm25=bm25 or BM25Spec(),
        )
        with open(stage / MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)
        if out_dir.exists():
            out_dir.rmdir()
        stage.replace(out_dir)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="构建倒排索引")
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    args = ap.parse_args()

    t0 = time.perf_counter()
    m = build(args.corpus, args.out, args.dataset, BM25Spec(k1=args.k1, b=args.b))
    dt = time.perf_counter() - t0
    print(f"完成：{m.n_docs} 篇文档，avgdl={m.avgdl:.1f}，耗时 {dt:.2f}s -> {args.out}")
