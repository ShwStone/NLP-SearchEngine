"""Capture full-corpus dense candidates without reading relevance labels.

Explicit HNSW parameters are passed to the backend, not attached as unused
adapter attributes. Encoding/query timings are components, not HTTP latency.
"""
import argparse
import json
import math
from pathlib import Path
import time

from contracts.index_format import sha256_of
from contracts.retrieval import VectorManifest
from online.parser import parse
from online.vector_index import QdrantVectorIndex
from evaluation.prepare_bm25_router import jsonl_rows


def capture(queries, vector_manifest, model_path, url, output, k=200, hnsw_ef=128):
    from adapters.embedding import BGEEncoder
    from qdrant_client import models
    if type(k) is not int or not 1 <= k <= 200 or type(hnsw_ef) is not int or hnsw_ef < 1:
        raise ValueError('invalid candidate budget or HNSW ef')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = list(jsonl_rows(queries))
    ids = [r.get('query_id', r.get('qid')) for r in rows]
    if any(not isinstance(q, str) or not q for q in ids) or len(set(ids)) != len(ids):
        raise ValueError('invalid/duplicate query IDs')
    texts = [r.get('query', r.get('text')) for r in rows]
    if any(not isinstance(q, str) or not q for q in texts):
        raise ValueError('invalid query text')
    manifest = VectorManifest.from_dict(json.loads(Path(vector_manifest).read_text()))
    vector = QdrantVectorIndex(manifest, url)
    encoder = BGEEncoder(manifest.embedding, str(model_path), 'cuda', batch_size=32)
    protocol = {'queries': len(rows), 'queries_sha256': sha256_of(queries),
                'vector_manifest_sha256': sha256_of(vector_manifest),
                'model_inventory_sha256': sha256_of(Path(model_path) / 'model_inventory.json'),
                'documents': manifest.n_docs, 'k': k,
                'search_params': {'hnsw_ef': hnsw_ef, 'exact': False},
                'qrels_read': False, 'explicit_dense_skipped': 0,
                'latency_scope': 'query encoding and Qdrant components; not end-to-end HTTP'}
    output.mkdir(parents=True)
    (output / 'protocol.pending.json').write_text(json.dumps(protocol, indent=2) + '\n')
    with (output / 'capture.pending.jsonl').open('x', encoding='utf-8') as handle:
        for i, (qid, text) in enumerate(zip(ids, texts)):
            constrained = parse(text).get('explicit_constraints', False)
            result = {'query_id': qid, 'text': text, 'explicit_constraints': constrained, 'dense': []}
            if constrained:
                protocol['explicit_dense_skipped'] += 1
            else:
                start = time.perf_counter()
                query_vector = encoder.encode_query(text)
                encoded = time.perf_counter()
                points = vector._client.query_points(
                    collection_name=manifest.collection, query=query_vector, limit=k,
                    search_params=models.SearchParams(hnsw_ef=hnsw_ef, exact=False),
                    with_payload=['doc_id']).points
                queried = time.perf_counter()
                seen, seen_points = set(), set()
                for point in sorted(points, key=lambda p: (-p.score, p.id)):
                    doc_id = (point.payload or {}).get('doc_id')
                    if (type(point.id) is not int or not 0 <= point.id < manifest.n_docs or
                            not isinstance(doc_id, str) or not doc_id or doc_id in seen or
                            point.id in seen_points or not math.isfinite(point.score) or
                            not -1.0001 <= point.score <= 1.0001):
                        raise ValueError('invalid dense hit identity or score')
                    seen.add(doc_id)
                    seen_points.add(point.id)
                    result['dense'].append({'doc_id': doc_id, 'docid': point.id, 'score': point.score})
                if len(points) != k:
                    raise ValueError('incomplete candidate count')
                result['component_ms'] = {'encoding': (encoded-start)*1000,
                                          'qdrant': (queried-encoded)*1000}
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            if (i+1) % 100 == 0:
                print('dense captured', i+1, '/', len(rows), flush=True)
    if sha256_of(queries) != protocol['queries_sha256']:
        raise ValueError('queries changed during capture')
    QdrantVectorIndex(manifest, url)
    (output / 'capture.pending.jsonl').rename(output / 'capture.jsonl')
    protocol['capture_sha256'] = sha256_of(output / 'capture.jsonl')
    (output / 'protocol.pending.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (output / 'protocol.pending.json').rename(output / 'protocol.json')
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('queries', 'vector-manifest', 'model-path', 'url', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--k', type=int, default=200)
    parser.add_argument('--hnsw-ef', type=int, default=128)
    print(json.dumps(capture(**vars(parser.parse_args())), indent=2))
