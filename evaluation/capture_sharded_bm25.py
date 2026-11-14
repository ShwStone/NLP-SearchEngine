"""Offline shard-by-shard full BM25 capture; timings are not HTTP latencies."""
import argparse
import gc
import json
from pathlib import Path
import time

from contracts import analyzer
from contracts.index_format import sha256_of
from offline.disk_index import DiskIndex
from online.parser import parse
from online.scorers import boolean_docs
from evaluation.prepare_bm25_router import jsonl_rows
from evaluation.sharded_bm25 import shard_topk, merge_topk


def capture(index_dir, queries, statistics, output, k=200):
    if not 1 <= k <= 200:
        raise ValueError('k must be in 1..200')
    stats = json.loads(Path(statistics).read_text())
    if stats['queries_sha256'] != sha256_of(queries):
        raise ValueError('query bytes differ from frozen statistics')
    if stats['analyzer_hash'] != analyzer.config_hash():
        raise ValueError('analyzer mismatch')
    rows = list(jsonl_rows(queries))
    qids = [row.get('query_id', row.get('qid')) for row in rows]
    if any(not isinstance(q, str) or not q for q in qids) or len(set(qids)) != len(qids):
        raise ValueError('missing or duplicate query IDs')
    parsed = [parse(row.get('query', row.get('text', ''))) for row in rows]
    terms = [[t.term for raw in p['terms'] for t in analyzer.analyze(raw)] for p in parsed]
    rankings = [[] for _ in rows]
    identities = [{} for _ in rows]
    offset = 0
    started = time.monotonic()
    for shard in stats['shards']:
        if shard['offset'] != offset or Path(shard['shard']).name != shard['shard']:
            raise ValueError('invalid shard order or path')
        directory = Path(index_dir) / shard['shard']
        if sha256_of(directory / 'manifest.json') != shard['manifest_sha256']:
            raise ValueError('shard manifest changed')
        reader = DiskIndex(directory, verify_checksums=True)
        if reader.n_docs != shard['documents']:
            raise ValueError('shard size mismatch')
        for i, p in enumerate(parsed):
            candidates = (boolean_docs(reader, p['ast'], analyzer.analyze)
                          if p.get('explicit_constraints', False) else None)
            hits = shard_topk(reader, terms[i], n_docs=stats['n_docs'], avgdl=stats['avgdl'],
                              dfs=stats['dfs'], offset=offset, k=k, candidate_docs=candidates)
            new_ids = {docid: reader.doc(docid - offset).doc_id for docid, _ in hits}
            rankings[i] = merge_topk([rankings[i], hits], k)
            identities[i].update(new_ids)
            identities[i] = {d: identities[i][d] for d, _ in rankings[i]}
        offset += reader.n_docs
        del reader
        gc.collect()
        print('scored shard', shard['shard'], 'documents', offset, flush=True)
    if offset != stats['n_docs']:
        raise ValueError('incomplete global document coverage')
    protocol = {'queries': len(rows), 'documents': offset, 'k': k,
                'statistics_sha256': sha256_of(statistics), 'queries_sha256': sha256_of(queries),
                'batch_wall_seconds': time.monotonic() - started,
                'scope': 'offline batch candidate generation; not HTTP latency', 'qrels_read': False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'capture.jsonl').open('w', encoding='utf-8', newline='\n') as handle:
        for i, row in enumerate(rows):
            handle.write(json.dumps({'query_id': qids[i], 'text': row.get('query', row.get('text')),
                                    'explicit_constraints': parsed[i].get('explicit_constraints', False),
                                    'lexical': [{'doc_id': identities[i][d], 'docid': d, 'score': s}
                                                for d, s in rankings[i]]}, ensure_ascii=False) + '\n')
    protocol['capture_sha256'] = sha256_of(output / 'capture.jsonl')
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return protocol


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('index-dir', 'queries', 'statistics', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--k', type=int, default=200)
    capture(**vars(p.parse_args()))
