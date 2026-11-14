"""Validate and stream completed encoding chunks in global document-ID order.

This is an ingestion primitive, not a published vector index. Downstream callers
must still check collection identity/counts and snapshots before publication.
"""
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import numpy as np

from contracts.index_format import sha256_of
from contracts.retrieval import VectorRecord
from evaluation.prepare_bm25_router import jsonl_rows


def iter_shard_records(source, directory, spec, global_offset=0):
    directory = Path(directory)
    summary = json.loads((directory / 'encoding-summary.json').read_text(encoding='utf-8'))
    if summary['source_sha256'] != sha256_of(source) or summary['embedding'] != asdict(spec):
        raise ValueError('encoding source or model identity mismatch')
    source_rows = iter(jsonl_rows(source))
    consumed = 0
    for i, record in enumerate(summary['chunks']):
        stem = f'chunk-{i:05d}'
        meta = json.loads((directory / (stem + '.json')).read_text(encoding='utf-8'))
        if meta != record or record['offset'] != consumed:
            raise ValueError('chunk metadata or order mismatch')
        for key in ('source_sha256', 'embedding', 'chunk_size', 'dtype'):
            if record[key] != summary[key]:
                raise ValueError('chunk identity differs from summary')
        if (type(record['documents']) is not int or
                not 1 <= record['documents'] <= summary['chunk_size']):
            raise ValueError('invalid chunk cardinality')
        vectors_path = directory / (stem + '.npy')
        ids_path = directory / (stem + '.ids.json')
        if (sha256_of(vectors_path) != record['vectors_sha256'] or
                sha256_of(ids_path) != record['ids_sha256']):
            raise ValueError('chunk hash mismatch')
        ids = json.loads(ids_path.read_text(encoding='utf-8'))
        rows = list(islice(source_rows, record['documents']))
        if len(rows) != record['documents'] or ids != [r['doc_id'] for r in rows]:
            raise ValueError('source/encoded ID order mismatch')
        vectors = np.load(vectors_path, mmap_mode='r', allow_pickle=False)
        if vectors.dtype != np.float32 or vectors.shape != (len(ids), spec.dimension):
            raise ValueError('invalid vectors shape or dtype')
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('nonfinite or unnormalized vectors')
        for j, doc_id in enumerate(ids):
            yield VectorRecord(global_offset + consumed + j, doc_id, vectors[j].tolist())
        consumed += len(ids)
    if next(source_rows, None) is not None or consumed != summary['documents']:
        raise ValueError('incomplete encoding coverage')
