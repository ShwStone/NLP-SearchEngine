"""SQLite-backed streaming builder for the v1 lexical index format.

The legacy builder keeps a full conformance oracle in Python memory.  This
builder keeps document identities, lengths, and postings in a temporary
SQLite database instead, then emits the unchanged JSONL A1 files in order.
It is intended for staged large-corpus builds; resource usage still depends
on the corpus, SQLite temporary storage, JSONL output, and the tokenizer.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from contracts import analyzer
from contracts.index_format import (
    DOCS, MANIFEST, OFFSETS, POSTINGS, STATS, BM25Spec, Manifest, sha256_of,
)


def _write_postings(database: sqlite3.Connection, path: Path) -> None:
    """Stream one canonical postings JSON object at a time."""

    with path.open("w", encoding="utf-8", newline="\n") as output:
        terms = database.execute(
            "SELECT term, COUNT(*) FROM postings GROUP BY term ORDER BY term"
        )
        for term, df in terms:
            output.write(
                '{"term":' + json.dumps(term, ensure_ascii=False)
                + ',"df":' + str(int(df)) + ',"docs":['
            )
            first = True
            for docid, tf, positions in database.execute(
                "SELECT docid, tf, positions FROM postings WHERE term=? ORDER BY docid",
                (term,),
            ):
                if not first:
                    output.write(",")
                first = False
                output.write(json.dumps(
                    [int(docid), int(tf), json.loads(positions)],
                    ensure_ascii=False, separators=(",", ":"),
                ))
            output.write("]}\n")


def build(corpus_path: Path, out_dir: Path, dataset: str,
          bm25: BM25Spec | None = None) -> Manifest:
    """Build a v1 lexical index without materializing the corpus in Python."""

    corpus_path, out_dir = Path(corpus_path), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("索引输出目录必须不存在或为空；不会覆盖既有产物")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.stream-", dir=out_dir.parent))
    database = sqlite3.connect(stage / "build.sqlite")
    try:
        database.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE ids(doc_id TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE lengths(docid INTEGER PRIMARY KEY, length INTEGER NOT NULL);
            CREATE TABLE postings(
                term TEXT NOT NULL,
                docid INTEGER NOT NULL,
                tf INTEGER NOT NULL,
                positions TEXT NOT NULL,
                PRIMARY KEY(term, docid)
            ) WITHOUT ROWID;
            """
        )
        n_docs = 0
        total_length = 0
        with corpus_path.open(encoding="utf-8") as source, \
                (stage / DOCS).open("w", encoding="utf-8", newline="\n") as docs, \
                (stage / OFFSETS).open("w", encoding="utf-8", newline="\n") as offsets:
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
                try:
                    database.execute("INSERT INTO ids(doc_id) VALUES (?)", (row["doc_id"],))
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"doc_id 必须唯一：{row['doc_id']}") from error
                tokens = analyzer.analyze(row["text"])
                if not tokens:
                    raise ValueError(f"语料第 {line_number} 行没有可分析 token")
                per_term: dict[str, list[int]] = {}
                spans = []
                for position, token in enumerate(tokens):
                    spans.append([position, position + 1, token.start, token.end])
                    per_term.setdefault(token.term, []).append(position)
                docs.write(json.dumps(
                    {"docid": n_docs, "doc_id": row["doc_id"],
                     "title": row["title"], "text": row["text"]},
                    ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
                offsets.write(json.dumps(
                    {"docid": n_docs, "spans": spans},
                    ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
                database.execute("INSERT INTO lengths(docid, length) VALUES (?, ?)",
                                 (n_docs, len(tokens)))
                database.executemany(
                    "INSERT INTO postings(term, docid, tf, positions) VALUES (?, ?, ?, ?)",
                    [(term, n_docs, len(positions), json.dumps(positions, separators=(",", ":")))
                     for term, positions in per_term.items()],
                )
                total_length += len(tokens)
                n_docs += 1
                if n_docs % 10000 == 0:
                    database.commit()
        if n_docs == 0:
            raise ValueError("语料不能为空")
        database.commit()
        avgdl = total_length / n_docs
        with (stage / STATS).open("w", encoding="utf-8", newline="\n") as stats:
            stats.write('{"doc_lengths":{')
            first = True
            for docid, length in database.execute("SELECT docid, length FROM lengths ORDER BY docid"):
                if not first:
                    stats.write(",")
                first = False
                stats.write(json.dumps(str(docid)) + ":" + str(int(length)))
            stats.write('},"avgdl":' + json.dumps(avgdl) + '}\n')
        _write_postings(database, stage / POSTINGS)
        database.close()
        database = None
        (stage / "build.sqlite").unlink()
        files = {name: sha256_of(stage / name)
                 for name in (DOCS, POSTINGS, OFFSETS, STATS)}
        manifest = Manifest(
            dataset=dataset, n_docs=n_docs, avgdl=avgdl,
            analyzer_name=analyzer.NAME, analyzer_version=analyzer.VERSION,
            analyzer_hash=analyzer.config_hash(), files=files,
            bm25=bm25 or BM25Spec(),
        )
        (stage / MANIFEST).write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if out_dir.exists():
            out_dir.rmdir()
        stage.replace(out_dir)
        return manifest
    except Exception:
        if database is not None:
            database.close()
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 SQLite 支持的流式倒排索引")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()
    started = time.perf_counter()
    manifest = build(args.corpus, args.out, args.dataset,
                      BM25Spec(k1=args.k1, b=args.b))
    print(f"完成：{manifest.n_docs} 篇文档，avgdl={manifest.avgdl:.1f}，"
          f"耗时 {time.perf_counter() - started:.2f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
