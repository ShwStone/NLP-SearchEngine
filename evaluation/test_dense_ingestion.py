"""Exercise stable point IDs and restart behavior with Qdrant's local engine."""
import json
from dataclasses import replace

import pytest

from contracts.retrieval import EmbeddingSpec, VectorManifest
from evaluation.encode_full_dense_chunks import encode_shard
from evaluation.ingest_dense_shard import ingest_shard


def test_replayed_shard_preserves_count_ids_and_vectors(tmp_path):
    qdrant = pytest.importorskip('qdrant_client')
    spec = EmbeddingSpec('fixture-model', 'fixture-revision', dimension=2)
    class Encoder:
        def __init__(self):
            self.spec = spec
        def encode_documents(self, texts):
            return [[1., 0.], [0., 1.]][:len(texts)]
    source = tmp_path / 'corpus'
    source.write_text(''.join(json.dumps(dict(doc_id=str(i), title='', text='doc')) + '\n'
                              for i in range(2)))
    encoded = tmp_path / 'encoded'
    encode_shard(source, encoded, Encoder(), chunk_size=2)
    manifest = VectorManifest('a' * 64, 12, spec, 'isolated-ingestion-fixture',
                              index_params={'m': 16, 'ef_construct': 100})
    client = qdrant.QdrantClient(':memory:')
    try:
        for _ in range(2):
            result = ingest_shard(client, manifest, source, encoded, 10, 2, batch_size=1)
            assert not result['collection_published']
            assert client.count(manifest.collection, exact=True).count == 2
        points = client.retrieve(manifest.collection, [10, 11], with_vectors=True)
        points = sorted(points, key=lambda p: p.id)
        assert [p.payload['doc_id'] for p in points] == ['0', '1']
        assert points[0].vector == pytest.approx([1., 0.])
        assert points[1].vector == pytest.approx([0., 1.])
        with pytest.raises(ValueError, match='identity mismatch'):
            ingest_shard(client, replace(manifest, corpus_sha256='b' * 64),
                         source, encoded, 10, 2)
        assert client.count(manifest.collection, exact=True).count == 2
    finally:
        client.close()
