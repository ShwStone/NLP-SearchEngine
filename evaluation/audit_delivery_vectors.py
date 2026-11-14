"""Reencode a fixed, label-independent sample across every corpus shard.

This is a sampled numeric check, never proof that every vector was reencoded.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from contracts.index_format import sha256_of
from contracts.retrieval import VectorManifest
from online.vector_index import QdrantVectorIndex


def audit(corpus_dir, statistics, vector_manifest, model_path, url, output):
    from adapters.embedding import BGEEncoder
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    stats = json.loads(Path(statistics).read_text())
    manifest = VectorManifest.from_dict(json.loads(Path(vector_manifest).read_text()))
    if stats['n_docs'] != manifest.n_docs:
        raise ValueError('corpus document counts disagree')
    vector = QdrantVectorIndex(manifest, url)
    encoder = BGEEncoder(manifest.embedding, str(model_path), 'cuda', batch_size=32)
    selected, offset = [], 0
    for shard in stats['shards']:
        if shard['offset'] != offset or Path(shard['shard']).name != shard['shard']:
            raise ValueError('invalid shard offsets/path')
        source = Path(corpus_dir) / shard['shard']
        if sha256_of(source) != shard['source_sha256']:
            raise ValueError('corpus source changed')
        # First point per shard, fixed before reading queries or labels.
        with source.open(encoding='utf-8') as handle:
            row = json.loads(next(handle))
        selected.append((offset, row))
        offset += shard['documents']
    if offset != manifest.n_docs:
        raise ValueError('incomplete shard coverage')
    expected = np.asarray(encoder.encode_documents(
        [row['title'] + '\n' + row['text'] for _, row in selected]), dtype=np.float32)
    points = vector._client.retrieve(manifest.collection,
                                    ids=[i for i, _ in selected],
                                    with_payload=True, with_vectors=True)
    by_id = {p.id: p for p in points}
    if set(by_id) != {i for i, _ in selected} or len(points) != len(selected):
        raise ValueError('sample readback incomplete/duplicated')
    actual = []
    for i, row in selected:
        point = by_id[i]
        if (point.payload or {}).get('doc_id') != row['doc_id']:
            raise ValueError('sample point identity mismatch')
        actual.append(point.vector)
    actual = np.asarray(actual, dtype=np.float32)
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or
            not np.allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-4)):
        raise ValueError('invalid sample vector values')
    cosine = np.sum(expected * actual, axis=1)
    error = float(np.max(np.abs(expected-actual)))
    if error > 1e-4 or float(cosine.min()) < .99999:
        raise ValueError(f'reencoded vector mismatch: max_abs={error}, min_cosine={cosine.min()}')
    result = {'status': 'verified', 'scope': 'sampled numeric reencoding, first point of every shard',
              'sample_size': len(selected), 'shards': len(stats['shards']),
              'all_vectors_verified': False, 'embedding': asdict(manifest.embedding),
              'max_absolute_error': error, 'minimum_cosine': float(cosine.min()),
              'absolute_error_tolerance': 1e-4, 'minimum_cosine_threshold': .99999,
              'statistics_sha256': sha256_of(statistics),
              'vector_manifest_sha256': sha256_of(vector_manifest),
              'model_inventory_sha256': sha256_of(Path(model_path) / 'model_inventory.json'),
              'query_or_qrels_read': False}
    with output.open('x') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('corpus-dir', 'statistics', 'vector-manifest', 'model-path', 'url', 'output'):
        parser.add_argument('--' + name, required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
