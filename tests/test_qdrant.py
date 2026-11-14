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

from contracts.index_format import sha256_of
from contracts.retrieval import (
    VECTOR_BUILD_METADATA_KEY, EmbeddingSpec, RetrievalUnavailable, VectorManifest, VectorRecord,
)
from offline.build_hybrid import build_hybrid
from offline.provenance import PROVENANCE_FILE, REPORT_FILE
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

    def create_collection(self, collection_name, vectors_config, metadata):
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

    def count(self, collection_name, exact):
        self.count_calls.append((collection_name, exact))
        return SimpleNamespace(count=len(self.records[collection_name]))

    def create_snapshot(self, collection_name):
        self.snapshots.append(collection_name)
        return SimpleNamespace(name="build.snapshot")

    def delete_collection(self, collection_name):
        del self.collections[collection_name]
        del self.records[collection_name]

    def query_points(self, collection_name, query, limit, with_payload, **kwargs):
        assert with_payload is True
        self.last_search_params = kwargs.get('search_params')
        return SimpleNamespace(points=[
            SimpleNamespace(id=point.id, score=0.5, payload=point.payload)
            for point in list(self.records[collection_name].values())[:limit]
        ])


@contextmanager
def _sdk(client):
    models = SimpleNamespace(
        VectorParams=SimpleNamespace, PointStruct=SimpleNamespace,
        Distance=SimpleNamespace(COSINE="Cosine"),
        SearchParams=SimpleNamespace,
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


def test_qdrant_search_parameters_reach_sdk_and_reject_invalid_values():
    manifest = _manifest()
    client = _seed(manifest)
    with _sdk(client):
        index = QdrantVectorIndex(manifest, 'http://fixture', ef=128)
        index.search([1.0, 0.0], 2)
        assert vars(client.last_search_params) == {'hnsw_ef': 128, 'exact': False}
        index.search_params = {'hnsw_ef': 256, 'exact': True}
        index.search([1.0, 0.0], 2)
        assert vars(client.last_search_params) == {'hnsw_ef': 256, 'exact': True}
        for ef in (0, -1, True, 1.5, '128'):
            _raises(ValueError, lambda: QdrantVectorIndex(manifest, 'http://fixture', ef=ef))
        _raises(ValueError, lambda: QdrantVectorIndex(manifest, 'http://fixture', exact=1))
        index.search_params = {'hnsw_ef': 0, 'exact': False}
        _raises(RetrievalUnavailable, lambda: index.search([1.0, 0.0], 2))


def test_qdrant_legacy_search_receives_frozen_parameters():
    manifest = _manifest()
    client = _seed(manifest)
    received = {}
    def search(**kwargs):
        received.update(kwargs)
        return []
    legacy = SimpleNamespace(get_collection=client.get_collection, count=client.count, search=search)
    with _sdk(legacy):
        index = QdrantVectorIndex(manifest, 'http://fixture', ef=128, exact=True)
        assert index.search([1.0, 0.0], 2) == []
        assert vars(received['search_params']) == {'hnsw_ef': 128, 'exact': True}


def test_qdrant_actual_hnsw_settings_must_match_declared_manifest():
    manifest = replace(_manifest(), index_params={'m': 16, 'ef_construct': 100})
    client = _seed(manifest)
    config = client.collections[manifest.collection].config
    config.hnsw_config = SimpleNamespace(m=16, ef_construct=100)
    with _sdk(client):
        QdrantVectorIndex(manifest, 'http://fixture')
        config.hnsw_config.m = 32
        _raises(ValueError, lambda: QdrantVectorIndex(manifest, 'http://fixture'))


def test_qdrant_hybrid_bundle_stages_snapshot_and_can_retry_failed_publication():
    client = _Client()
    corpus = Path(__file__).resolve().parents[1] / "fixtures" / "tiny_corpus.jsonl"
    with tempfile.TemporaryDirectory() as directory, _sdk(client):
        root = Path(directory)
        bundle = root / "bundle"
        writer = QdrantSnapshotWriter("http://fixture", bundle)

        def download(*args, **kwargs):
            assert not bundle.exists()
            return io.BytesIO(b"snapshot bytes")

        def build_bundle():
            return build_hybrid(corpus, bundle, "fixture", _Encoder(), writer, "new-build")

        with patch("urllib.request.urlopen", side_effect=download):
            with patch("offline.build_hybrid.write_provenance", side_effect=OSError("test failure")):
                _raises(OSError, build_bundle)
            assert not bundle.exists()
            assert list(root.iterdir()) == []
            assert client.collections == {}
            result = build_bundle()

        assert writer.snapshot_dir == bundle
        assert list(root.iterdir()) == [bundle]
        assert {path.name for path in bundle.iterdir()} == {
            "bm25", "build.snapshot", "vector_manifest.json", PROVENANCE_FILE, REPORT_FILE,
        }
        assert sha256_of(bundle / "build.snapshot") == result["vector"]["snapshot_files"]["build.snapshot"]


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


def test_qdrant_writer_streams_records_and_keeps_unrelated_collection_on_failure():
    original = _manifest()
    new_build = replace(original, collection="new-build", snapshot_files={})
    for failure in (None, "generator", "upsert"):
        client = _seed(original)
        original_points = copy.deepcopy(client.records[original.collection])
        upsert = client.upsert

        def records():
            yield VectorRecord(0, "d0", [1.0, 0.0])
            assert set(client.records[new_build.collection]) == {0}
            if failure == "generator":
                raise RuntimeError("encoding interrupted")
            yield VectorRecord(1, "d1", [1.0, 0.0])

        def write_batch(collection_name, points, wait):
            if failure == "upsert" and points[0].id == 1:
                raise RuntimeError("second batch failed")
            upsert(collection_name, points, wait)

        with tempfile.TemporaryDirectory() as directory, _sdk(client):
            root = Path(directory)
            writer = QdrantSnapshotWriter("http://fixture", root, batch_size=1)
            with patch.object(client, "upsert", side_effect=write_batch), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(b"snapshot")
            ):
                if failure is None:
                    assert writer.replace(new_build, records()).snapshot_files
                    assert set(client.records[new_build.collection]) == {0, 1}
                else:
                    _raises(RuntimeError, lambda: writer.replace(new_build, records()))
                    assert new_build.collection not in client.collections
                    assert not list(root.iterdir())
            assert client.records[original.collection] == original_points


ALL_TESTS = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
