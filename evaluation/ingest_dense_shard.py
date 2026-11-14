"""Idempotent bounded-memory ingestion into an explicitly bound build collection.

This primitive never publishes a completed VectorManifest or snapshot. Retrying
a partial shard revalidates and upserts the entire shard with stable global IDs.
"""
from itertools import islice

from contracts.retrieval import VECTOR_BUILD_METADATA_KEY
from evaluation.stream_dense_chunks import iter_shard_records


def ensure_collection(client, manifest):
    from qdrant_client import models
    if not client.collection_exists(manifest.collection):
        client.create_collection(
            collection_name=manifest.collection,
            vectors_config=models.VectorParams(size=manifest.embedding.dimension,
                                               distance=models.Distance.COSINE, on_disk=True),
            hnsw_config=models.HnswConfigDiff(**manifest.index_params),
            on_disk_payload=True,
            metadata={VECTOR_BUILD_METADATA_KEY: manifest.build_metadata()},
        )
    info = client.get_collection(manifest.collection)
    metadata = info.config.metadata
    if not metadata or metadata.get(VECTOR_BUILD_METADATA_KEY) != manifest.build_metadata():
        raise ValueError('refusing unrelated collection: build identity mismatch')
    vectors = info.config.params.vectors
    if vectors.size != manifest.embedding.dimension or vectors.distance != models.Distance.COSINE:
        raise ValueError('collection vector configuration mismatch')


def ingest_shard(client, manifest, source, directory, offset, expected_documents, batch_size=128):
    from qdrant_client import models
    if (type(offset) is not int or offset < 0 or type(expected_documents) is not int
            or expected_documents < 1 or offset + expected_documents > manifest.n_docs
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('invalid shard range or batch size')
    ensure_collection(client, manifest)
    records = iter(iter_shard_records(source, directory, manifest.embedding, offset))
    consumed = 0
    while batch := list(islice(records, batch_size)):
        if consumed + len(batch) > expected_documents:
            raise ValueError('shard exceeds declared range')
        if [r.docid for r in batch] != list(range(offset + consumed, offset + consumed + len(batch))):
            raise ValueError('noncontiguous point IDs')
        client.upsert(collection_name=manifest.collection,
                      points=[models.PointStruct(id=r.docid, vector=list(r.vector),
                                                  payload={'doc_id': r.doc_id}) for r in batch],
                      wait=True)
        consumed += len(batch)
    if consumed != expected_documents:
        raise ValueError('incomplete shard; collection remains unpublished')
    # Count the declared range, not the entire partially built collection.
    count = client.count(collection_name=manifest.collection, exact=True,
                         count_filter=models.Filter(must=[models.HasIdCondition(
                             has_id=list(range(offset, offset + expected_documents)))] )).count
    if count != expected_documents:
        raise ValueError('exact shard point count mismatch')
    return {'offset': offset, 'documents': consumed, 'exact_count_verified': True,
            'collection_published': False}
