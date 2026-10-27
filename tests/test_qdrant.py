"""Qdrant adapter regressions using a small SDK double, without optional deps."""

import copy
import io
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from contracts.retrieval import (
    VECTOR_BUILD_METADATA_KEY, EmbeddingSpec, RetrievalUnavailable, VectorManifest,
)
from offline.vector_build import QdrantSnapshotWriter, build_vectors
from online.memory_index import MemoryIndex
from online.vector_index import QdrantVectorIndex


class _NotFound(Exception):
    status_code = 404


class _Encoder:
    spec = EmbeddingSpec("fixture-model", "fixture-v1", dimension=2)

    def encode_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _Client:
    def __init__(self):
        self.collections = {}
        self.records = {}
        self.count_calls = []
        self.snapshots = []

    def get_collection(self, collection_name):
        if collection_name not in self.collections:
            raise _NotFound(collection_name)
        return self.collections[collection_name]

    def create_collection(self, collection_name, vectors_config, metadata, **kwargs):
        assert collection_name not in self.collections
        self.collections[collection_name] = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=vectors_config), metadata=copy.deepcopy(metadata)),
            points_count=999,  # This approximate value must never govern startup/publication.
        )
        self.records[collection_name] = {}

    def upsert(self, collection_name, points, wait):
        assert wait is True
        self.records[collection_name].update({point.id: point for point in points})

    def delete_collection(self,collection_name):
        del self.collections[collection_name]
        del self.records[collection_name]

    def count(self, collection_name, exact):
        self.count_calls.append((collection_name, exact))
        return SimpleNamespace(count=len(self.records[collection_name]))

    def create_snapshot(self, collection_name):
        self.snapshots.append(collection_name)
        return SimpleNamespace(name="build.snapshot")

    def query_points(self, collection_name, query, limit, with_payload):
        assert with_payload is True
        return SimpleNamespace(points=[
            SimpleNamespace(id=point.id, score=0.5, payload=point.payload)
            for point in list(self.records[collection_name].values())[:limit]
        ])


@contextmanager
def _sdk(client):
    models = SimpleNamespace(
        VectorParams=SimpleNamespace, PointStruct=SimpleNamespace, HnswConfigDiff=SimpleNamespace,
        Distance=SimpleNamespace(COSINE="Cosine"),
    )
    module = SimpleNamespace(QdrantClient=lambda **_: client, models=models)
    with patch.dict(sys.modules, {"qdrant_client": module}):
        yield


def _manifest():
    return VectorManifest(
        "a" * 64, 2, _Encoder.spec, "fixture-v1",
        snapshot_files={"build.snapshot": "b" * 64},
    )


def _seed(manifest):
    client = _Client()
    client.create_collection(
        manifest.collection,
        SimpleNamespace(size=manifest.embedding.dimension, distance="Cosine"),
        {VECTOR_BUILD_METADATA_KEY: manifest.build_metadata()},
    )
    client.upsert(manifest.collection, [
        SimpleNamespace(id=i, payload={"doc_id": f"d{i}"}) for i in range(manifest.n_docs)
    ], wait=True)
    return client


def _reader():
    return MemoryIndex.from_corpus([
        {"doc_id": "d0", "title": "", "text": "red apple"},
        {"doc_id": "d1", "title": "", "text": "blue apple"},
    ])


def _raises(kind, fn):
    try:
        fn()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def test_qdrant_writer_persists_metadata_before_snapshot_and_reader_checks_it():
    client = _Client()
    with tempfile.TemporaryDirectory() as directory, _sdk(client):
        root = Path(directory)
        with patch("urllib.request.urlopen", return_value=io.BytesIO(b"snapshot bytes")):
            manifest = build_vectors(
                _reader(), _Encoder(), QdrantSnapshotWriter("http://fixture", root),
                root, "new-build",
            )
        metadata = client.get_collection(manifest.collection).config.metadata[VECTOR_BUILD_METADATA_KEY]
        assert metadata == manifest.build_metadata()
        assert "snapshot_files" not in metadata
        assert metadata["embedding"] == manifest.to_dict()["embedding"]
        assert (root / "vector_manifest.json").is_file()
        index = QdrantVectorIndex(manifest, "http://fixture")
        assert [hit.doc_id for hit in index.search([1.0, 0.0], 2)] == ["d0", "d1"]
        assert client.count_calls == [("new-build", True), ("new-build", True)]
        assert client.snapshots == ["new-build"]
        # Snapshot hashes are transport artifacts, not part of the collection identity.
        assert replace(manifest, snapshot_files={"other.snapshot": "c" * 64}).build_metadata() == metadata


def test_qdrant_reader_rejects_stale_build_identity_even_at_same_dimension():
    manifest = _manifest()
    changes = {
        "corpus_sha256": "c" * 64, "n_docs": 1, "collection": "other-build",
        "backend": "memory-exact", "schema_version": 99, "index_params": {"m": 32},
    }
    embedding_changes = {
        "model_id": "old-model", "revision": "old-revision", "dimension": 3,
        "max_tokens": 64, "normalize": False, "distance": "dot",
        "document_template": "text", "query_prefix": "query: ", "document_prefix": "passage: ",
    }
    for key, value in changes.items():
        client = _seed(manifest)
        client.collections[manifest.collection].config.metadata[VECTOR_BUILD_METADATA_KEY][key] = value
        with _sdk(client):
            _raises(ValueError, lambda: QdrantVectorIndex(manifest, "http://fixture"))
    for key, value in embedding_changes.items():
        client = _seed(manifest)
        metadata = client.collections[manifest.collection].config.metadata[VECTOR_BUILD_METADATA_KEY]
        metadata["embedding"][key] = value
        with _sdk(client):
            _raises(ValueError, lambda: QdrantVectorIndex(manifest, "http://fixture"))


def test_qdrant_reader_requires_metadata_and_single_cosine_vectors():
    manifest = _manifest()
    for metadata in (None, {}, {"unrelated": "metadata"}):
        client = _seed(manifest)
        client.collections[manifest.collection].config.metadata = metadata
        with _sdk(client):
            _raises(ValueError, lambda: QdrantVectorIndex(manifest, "http://fixture"))
    for vectors in (None, {"named": SimpleNamespace(size=2, distance="Cosine")},
                    SimpleNamespace(size=3, distance="Cosine"),
                    SimpleNamespace(size=2, distance="Euclid")):
        client = _seed(manifest)
        client.collections[manifest.collection].config.params.vectors = vectors
        with _sdk(client):
            _raises(ValueError, lambda: QdrantVectorIndex(manifest, "http://fixture"))


def test_qdrant_reader_uses_exact_count_and_wraps_transport_failures():
    manifest = _manifest()
    client = _seed(manifest)
    with _sdk(client):
        for count in (0, 1, 3, None, True):
            with patch.object(client, "count", return_value=SimpleNamespace(count=count)):
                _raises(ValueError, lambda: QdrantVectorIndex(manifest, "http://fixture"))
        for method in ("get_collection", "count"):
            with patch.object(client, method, side_effect=TimeoutError("offline")):
                _raises(RetrievalUnavailable, lambda: QdrantVectorIndex(manifest, "http://fixture"))
        index = QdrantVectorIndex(manifest, "http://fixture")
        with patch.object(client, "query_points", side_effect=TimeoutError("offline")):
            _raises(RetrievalUnavailable, lambda: index.search([1.0, 0.0], 2))


def test_qdrant_writer_does_not_publish_unverified_metadata_or_point_count():
    for broken_field in ("metadata", "count"):
        client = _Client()
        original_get = client.get_collection

        def get_collection(name):
            info = original_get(name)
            if broken_field == "metadata":
                info.config.metadata = None
            return info

        with tempfile.TemporaryDirectory() as directory, _sdk(client):
            root = Path(directory)
            with patch.object(client, "get_collection", side_effect=get_collection), \
                    patch.object(client, "count", return_value=SimpleNamespace(count=1)):
                _raises(ValueError, lambda: build_vectors(
                    _reader(), _Encoder(), QdrantSnapshotWriter("http://fixture", root),
                    root, "new-build",
                ))
            assert not (root / "vector_manifest.json").exists()
            assert client.snapshots == []
            assert client.collections == {}


def test_qdrant_writer_never_overwrites_an_existing_collection():
    manifest = _manifest()
    client = _seed(manifest)
    original = copy.deepcopy(client.records)
    with tempfile.TemporaryDirectory() as directory, _sdk(client):
        root = Path(directory)
        _raises(ValueError, lambda: build_vectors(
            _reader(), _Encoder(), QdrantSnapshotWriter("http://fixture", root),
            root, manifest.collection,
        ))
        assert not (root / "vector_manifest.json").exists()
    assert client.records == original
    assert client.snapshots == []


def test_qdrant_writer_does_not_treat_transport_failure_as_a_new_collection():
    client = _Client()
    with tempfile.TemporaryDirectory() as directory, _sdk(client):
        root = Path(directory)
        with patch.object(client, "get_collection", side_effect=TimeoutError("offline")):
            _raises(RuntimeError, lambda: build_vectors(
                _reader(), _Encoder(), QdrantSnapshotWriter("http://fixture", root),
                root, "new-build",
            ))
        assert not (root / "vector_manifest.json").exists()
        assert client.collections == {}


ALL_TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
