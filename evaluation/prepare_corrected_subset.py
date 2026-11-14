"""Bind historical lexical capture to the index and corrected dense capture.

Lexical scores are preserved; only global internal IDs and hashed provenance
are added. This does not claim a new lexical retrieval run.
"""
import argparse
import json
from pathlib import Path

from contracts.index_format import sha256_of
from evaluation.prepare_bm25_router import jsonl_rows
from evaluation.merge_full_captures import merge
from offline.disk_index import DiskIndex


def prepare(index, historical, dense_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    reader = DiskIndex(index)
    ids = {reader.doc(i).doc_id: i for i in range(reader.n_docs)}
    if len(ids) != reader.n_docs:
        raise ValueError('duplicate corpus external IDs')
    historical = Path(historical)
    previous = json.loads((historical.parent/'protocol.json').read_text())
    dp = json.loads((Path(dense_dir)/'protocol.json').read_text())
    if (previous['queries_sha256'] != dp['queries_sha256'] or
            previous['vector_manifest_sha256'] != dp['vector_manifest_sha256'] or
            previous['k'] != dp['k'] or reader.n_docs != dp['documents']):
        raise ValueError('historical input identity mismatch')
    source_rows = list(jsonl_rows(historical))
    if len(source_rows) != dp['queries']:
        raise ValueError('historical query count mismatch')
    rows = []
    for row in source_rows:
        hits = []
        for hit in row['lexical']:
            if hit['doc_id'] not in ids:
                raise ValueError('unknown lexical document')
            hits.append(dict(hit, docid=ids[hit['doc_id']]))
        rows.append({'query_id':row['query_id'], 'text':row['text'],
                     'explicit_constraints':row['explicit_constraints'], 'lexical':hits})
    output.mkdir(parents=True)
    lexical_dir = output/'lexical'
    lexical_dir.mkdir()
    with (lexical_dir/'capture.jsonl').open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    protocol = {'queries':len(rows),'documents':reader.n_docs,'k':dp['k'],
                'queries_sha256':dp['queries_sha256'],'qrels_read':False,
                'capture_sha256':sha256_of(lexical_dir/'capture.jsonl'),
                'source_historical_capture_sha256':sha256_of(historical),
                'source_historical_protocol_sha256':sha256_of(historical.parent/'protocol.json'),
                'index_manifest_sha256':sha256_of(Path(index)/'manifest.json'),
                'scope':'historical lexical results with index-verified external/internal IDs'}
    (lexical_dir/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return merge(lexical_dir, dense_dir, output/'merged')


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('index','historical','dense-dir','output'):
        p.add_argument('--'+name,required=True)
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
