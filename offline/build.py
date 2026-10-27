"""索引构建（离线组）。

用法：
    python -m offline.build --corpus fixtures/tiny_corpus.jsonl \
        --out fixtures/tiny_index --dataset tiny

产物符合 contracts/index_format.py 的接口 A 约定。
验收：连跑两次 manifest 中各文件 sha256 一致（确定性构建）。
"""

import argparse
import json
import time
import sqlite3
import tempfile
import sys
from pathlib import Path

from contracts import analyzer
from contracts.index_format import (
    DOCS, MANIFEST, OFFSETS, POSTINGS, STATS,
    BM25Spec, Manifest, sha256_of,
)


def build(corpus_path: Path, out_dir: Path, dataset: str,
          bm25: BM25Spec | None = None) -> Manifest:
    """Bounded-memory builder; SQLite is scratch storage, published format stays v1.

    The production algorithm is independent from the conformance oracle. Build in
    a fresh staging directory and publish only after every artifact is complete.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("词法构建必须使用新的空输出目录")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{out_dir.name}.build-', dir=out_dir.parent) as tmp:
        stage = Path(tmp)
        database = stage / 'postings.scratch.sqlite'
        connection = sqlite3.connect(database)
        lengths, ids, pending = {}, set(), []
        try:
            # Scratch is rebuildable. Durability applies to the final checksummed files.
            connection.execute('PRAGMA journal_mode=OFF')
            connection.execute('PRAGMA synchronous=OFF')
            connection.execute('PRAGMA temp_store=FILE')
            connection.execute('PRAGMA cache_size=-65536')
            connection.execute('CREATE TABLE posting(term TEXT,docid INTEGER,tf INTEGER,positions TEXT,PRIMARY KEY(term,docid)) WITHOUT ROWID')
            with open(corpus_path, encoding='utf-8') as source, (stage/DOCS).open('w',encoding='utf-8') as docs, (stage/OFFSETS).open('w',encoding='utf-8') as offsets:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row,dict) or not isinstance(row.get('doc_id'),str) or not row['doc_id']:
                        raise ValueError(f'语料第 {line_number} 行 doc_id 必须为非空字符串')
                    if row['doc_id'] in ids:
                        raise ValueError(f'语料第 {line_number} 行 doc_id 重复')
                    if not isinstance(row.get('text'),str) or not isinstance(row.get('title',''),str):
                        raise ValueError(f'语料第 {line_number} 行 title/text 必须为字符串')
                    tokens = analyzer.analyze(row['text'])
                    if not tokens:
                        raise ValueError(f'语料第 {line_number} 行正文没有有效词项')
                    docid = len(lengths)
                    ids.add(row['doc_id']);lengths[docid]=len(tokens)
                    docs.write(json.dumps(dict(docid=docid,doc_id=row['doc_id'],title=row.get('title',''),text=row['text']),ensure_ascii=False)+'\n')
                    offsets.write(json.dumps(dict(docid=docid,spans=[[i,i+1,t.start,t.end] for i,t in enumerate(tokens)]),ensure_ascii=False,separators=(',',':'))+'\n')
                    per_term = {}
                    for i, token in enumerate(tokens):
                        per_term.setdefault(token.term,[]).append(i)
                    pending.extend((term,docid,len(positions),json.dumps(positions,separators=(',',':'))) for term,positions in per_term.items())
                    if len(pending)>=10000:
                        connection.executemany('INSERT INTO posting VALUES(?,?,?,?)',pending)
                        connection.commit();pending.clear()
                    if (docid+1)%5000==0:
                        print(f'lexical scratch: {docid+1} documents',file=sys.stderr,flush=True)
            if not lengths:
                raise ValueError('语料不能为空')
            if pending:
                connection.executemany('INSERT INTO posting VALUES(?,?,?,?)',pending)
                connection.commit();pending.clear()
            with (stage/POSTINGS).open('w+b') as output:
                previous=None;count=0;slot=0
                def finish_term():
                    output.write(b']}\n')
                    end=output.tell();output.seek(slot)
                    encoded=f'{count:12d}'.encode()
                    if len(encoded)!=12:raise ValueError('df exceeds streaming format capacity')
                    output.write(encoded);output.seek(end)
                for term,docid,tf,positions in connection.execute('SELECT term,docid,tf,positions FROM posting ORDER BY term,docid'):
                    if term!=previous:
                        if previous is not None:finish_term()
                        output.write(b'{"term":'+json.dumps(term,ensure_ascii=False).encode()+b',"df":')
                        slot=output.tell();output.write(b'           0'+b',"docs":[')
                        previous=term;count=0
                    if count:output.write(b',')
                    output.write(f'[{docid},{tf},{positions}]'.encode())
                    count+=1
                if previous is not None:finish_term()
        finally:
            connection.close()
        database.unlink()
        avgdl=sum(lengths.values())/len(lengths)
        with (stage/STATS).open('w',encoding='utf-8') as target:
            json.dump(dict(doc_lengths={str(k):v for k,v in lengths.items()},avgdl=avgdl),target)
        files={name:sha256_of(stage/name) for name in (DOCS,POSTINGS,OFFSETS,STATS)}
        manifest=Manifest(dataset=dataset,n_docs=len(lengths),avgdl=avgdl,
                          analyzer_name=analyzer.NAME,analyzer_version=analyzer.VERSION,
                          analyzer_hash=analyzer.config_hash(),files=files,bm25=bm25 or BM25Spec())
        (stage/MANIFEST).write_text(json.dumps(manifest.to_dict(),ensure_ascii=False,indent=2),encoding='utf-8')
        if out_dir.exists():out_dir.rmdir()  # Empty reserved directory only; never remove a published build.
        stage.replace(out_dir)
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
