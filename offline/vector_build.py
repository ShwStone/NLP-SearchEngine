"""离线向量数据库构建骨架：批量编码 -> 写入向量库 -> 发布 manifest。

真实模型和 Qdrant 快照上传/发布由适配器实现。本模块不 import online。
"""

import json
from dataclasses import replace
from pathlib import Path

from contracts.index_format import sha256_of
from contracts.reader import DocumentSource
from contracts.retrieval import (
    VECTOR_BUILD_METADATA_KEY, VECTOR_MANIFEST, TextEncoder, VectorManifest,
    VectorRecord, VectorWriter,
    corpus_fingerprint, validate_vector,
)


def build_vectors(reader: DocumentSource, encoder: TextEncoder, writer: VectorWriter,
                  out_dir: Path, collection: str, batch_size: int = 32,
                  backend: str = "qdrant-hnsw", index_params: dict | None = None
                  ) -> VectorManifest:
    """仅在 writer 成功消费全部记录后发布元数据；目标目录须为新的构建目录。

    一文档一向量。title + '\\n' + text，前缀/截断由编码器按 spec 执行。
    不允许覆盖已发布的向量 manifest；真实后端也须使用新的 collection。
    """
    out_dir = Path(out_dir)
    target = out_dir / VECTOR_MANIFEST
    if target.exists():
        raise ValueError("向量产物已存在；请使用新的输出目录与 collection")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    ids = [reader.doc(i).doc_id for i in range(reader.n_docs)]
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id 不唯一")
    manifest = VectorManifest(
        corpus_sha256=corpus_fingerprint(reader), n_docs=reader.n_docs,
        embedding=encoder.spec, collection=collection, backend=backend,
        index_params=dict(index_params or {}),
    )
    consumed = 0

    def records():
        nonlocal consumed
        for start in range(0, reader.n_docs, batch_size):
            docs = [reader.doc(i) for i in range(start, min(start + batch_size, reader.n_docs))]
            vectors = encoder.encode_documents([f"{d.title}\n{d.text}" for d in docs])
            if len(vectors) != len(docs):
                raise ValueError("编码器输出数量与输入文档数不符")
            for doc, vector in zip(docs, vectors):
                validate_vector(vector, encoder.spec)
                consumed += 1
                yield VectorRecord(doc.docid, doc.doc_id, tuple(vector))

    out_dir.mkdir(parents=True, exist_ok=True)
    published = writer.replace(manifest, records())
    if consumed != reader.n_docs:
        raise ValueError("writer 未消费全部文档，不能发布 manifest")
    if not isinstance(published, VectorManifest) or replace(published, snapshot_files={}) != manifest:
        raise ValueError("writer 必须返回仅补齐 snapshot_files 的 VectorManifest")
    if backend == "qdrant-hnsw" and not published.snapshot_files:
        raise ValueError("Qdrant 构建必须交付带校验和的快照")
    for name, digest in published.snapshot_files.items():
        snapshot = out_dir / name
        if not snapshot.is_file() or sha256_of(snapshot) != digest:
            raise ValueError("快照不存在或 SHA256 校验失败")
    manifest = published
    pending = target.with_suffix(".json.tmp")
    pending.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
                       encoding="utf-8")
    pending.replace(target)
    return manifest


class QdrantSnapshotWriter:
    """Build a Qdrant collection and publish a verified local snapshot.

    Qdrant and ``qdrant-client`` must support collection metadata (1.16+).
    The SDK is imported lazily so the stdlib lexical workflow keeps working
    without the optional stack. Each build uses a fresh, versioned collection.
    """

    def __init__(self, url: str, snapshot_dir: Path, batch_size: int = 128,
                 timeout: float = 30.0):
        self.url = url
        self.snapshot_dir = Path(snapshot_dir)
        self.batch_size = batch_size
        self.timeout = timeout

    def replace(self, manifest: VectorManifest, records) -> VectorManifest:
        self._owned_collection=None
        self._cleanup_client=None
        try:
            return self._replace(manifest,records)
        except BaseException:
            if self._owned_collection is not None:
                try:self._cleanup_client.delete_collection(collection_name=self._owned_collection)
                except Exception:pass
            raise

    def _replace(self, manifest: VectorManifest, records) -> VectorManifest:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正数")
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as error:
            raise RuntimeError("QdrantSnapshotWriter 需要安装 qdrant-client") from error
        client = QdrantClient(url=self.url, timeout=self.timeout)
        try:
            client.get_collection(manifest.collection)
        except Exception as error:
            # Only a confirmed 404 means the versioned collection is new.
            # Network, TLS, and authentication failures must stop the build
            # instead of being misread as an empty destination.
            if getattr(error, "status_code", None) != 404:
                raise RuntimeError(f"无法检查 Qdrant collection: {error}") from error
        else:
            raise ValueError(f"Qdrant collection 已存在: {manifest.collection}")
        extra={}
        if manifest.index_params:
            extra['hnsw_config']=models.HnswConfigDiff(**manifest.index_params)
        client.create_collection(
            collection_name=manifest.collection,
            vectors_config=models.VectorParams(
                size=manifest.embedding.dimension, distance=models.Distance.COSINE,
            ),
            metadata={VECTOR_BUILD_METADATA_KEY: manifest.build_metadata()},
            **extra,
        )
        self._owned_collection=manifest.collection
        self._cleanup_client=client
        ids,external_ids,batch=set(),set(),[]
        try:
            for record in records:
                validate_vector(record.vector,manifest.embedding)
                if type(record.docid) is not int or not 0<=record.docid<manifest.n_docs or record.docid in ids:
                    raise ValueError('invalid or duplicate point id')
                if not isinstance(record.doc_id,str) or not record.doc_id or record.doc_id in external_ids:
                    raise ValueError('invalid or duplicate external id')
                ids.add(record.docid);external_ids.add(record.doc_id)
                batch.append(models.PointStruct(id=record.docid,vector=list(record.vector),payload={'doc_id':record.doc_id}))
                if len(batch)>=self.batch_size:
                    client.upsert(collection_name=manifest.collection,points=batch,wait=True);batch=[]
            if batch:client.upsert(collection_name=manifest.collection,points=batch,wait=True)
            if len(ids)!=manifest.n_docs:raise ValueError('incomplete vector records')
        except Exception:
            # Only this newly-created unpublished collection is owned here.
            try:client.delete_collection(collection_name=manifest.collection)
            except Exception:pass
            raise
        info = client.get_collection(manifest.collection)
        metadata = getattr(getattr(info, "config", None), "metadata", None)
        if (not isinstance(metadata, dict)
                or metadata.get(VECTOR_BUILD_METADATA_KEY) != manifest.build_metadata()):
            raise ValueError("Qdrant 未保存构建元数据；需要 Qdrant / qdrant-client 1.16+")
        # CollectionInfo.points_count is approximate; startup/publication needs
        # the exact count, even while Qdrant is optimizing segments.
        count = client.count(collection_name=manifest.collection, exact=True).count
        if type(count) is not int or count != manifest.n_docs:
            raise ValueError(f"Qdrant 点数不符: {count} != {manifest.n_docs}")
        try:
            snapshot = client.create_snapshot(collection_name=manifest.collection)
            snapshot_name = getattr(snapshot, "name", None) or getattr(snapshot, "result", None)
            if not isinstance(snapshot_name, str):
                raise ValueError("Qdrant 未返回快照文件名")
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            target = self.snapshot_dir / Path(snapshot_name).name
            from urllib.request import urlopen
            import urllib.parse
            endpoint = (self.url.rstrip("/") + "/collections/" +
                        urllib.parse.quote(manifest.collection, safe="") + "/snapshots/" +
                        urllib.parse.quote(Path(snapshot_name).name, safe=""))
            with urlopen(endpoint, timeout=self.timeout) as response, target.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        except Exception as error:
            raise RuntimeError(f"Qdrant 快照发布失败: {error}") from error
        return replace(manifest, snapshot_files={target.name: sha256_of(target)})
