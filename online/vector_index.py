"""在线向量库读取接口的内存替身与可选 Qdrant 只读适配器。"""

import math
from typing import Sequence

from contracts.retrieval import (
    VECTOR_BUILD_METADATA_KEY, RetrievalUnavailable, VectorHit, VectorManifest,
    VectorRecord, validate_vector,
)


class MemoryVectorIndex:
    """精确余弦全扫，仅用于小样本契约测试，不是 HNSW 性能替代品。"""

    def __init__(self, manifest: VectorManifest, records: list[VectorRecord]):
        if manifest.backend != "memory-exact":
            raise ValueError("内存替身必须显式标为 memory-exact")
        if len(records) != manifest.n_docs:
            raise ValueError("向量记录数与 manifest 不符")
        if sorted(r.docid for r in records) != list(range(manifest.n_docs)):
            raise ValueError("内部 docid 必须唯一、连续且与倒排索引对齐")
        if len({r.doc_id for r in records}) != len(records):
            raise ValueError("外部 doc_id 必须唯一")
        for record in records:
            validate_vector(record.vector, manifest.embedding)
        self._manifest = manifest
        self._records = tuple(VectorRecord(r.docid, r.doc_id, tuple(r.vector)) for r in records)

    def manifest(self) -> VectorManifest:
        return self._manifest

    @property
    def search_config(self) -> dict:
        return {"hnsw_ef": None, "exact": True, "timeout_s": None}

    def search(self, query_vector: Sequence[float], limit: int) -> list[VectorHit]:
        validate_vector(query_vector, self._manifest.embedding)
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        hits = [VectorHit(r.docid, r.doc_id,
                          max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(query_vector, r.vector)))))
                for r in self._records]
        return sorted(hits, key=lambda h: (-h.score, h.docid))[:limit]


class QdrantVectorIndex:
    """Read-only Qdrant adapter with manifest/collection compatibility checks."""

    def __init__(self, manifest: VectorManifest, url: str, *, hnsw_ef: int | None = None,
                 timeout_s: float = 30.0, ef: int | None = None, exact: bool = False):
        if ef is not None:
            if hnsw_ef is not None and hnsw_ef != ef:
                raise ValueError("ef and hnsw_ef must agree")
            hnsw_ef = ef
        if type(exact) is not bool:
            raise ValueError("exact must be boolean")
        if hnsw_ef is not None and (type(hnsw_ef) is not int or hnsw_ef <= 0):
            raise ValueError("hnsw_ef must be a positive integer")
        if (type(timeout_s) is bool or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("timeout_s must be a positive finite number of seconds")
        if manifest.backend != "qdrant-hnsw":
            raise ValueError("Qdrant 适配器需要 qdrant-hnsw manifest")
        try:
            from qdrant_client import QdrantClient
        except ImportError as error:
            raise RetrievalUnavailable("QdrantVectorIndex 需要安装 qdrant-client") from error
        self._manifest = manifest
        self.search_params = {"hnsw_ef": hnsw_ef, "exact": exact}
        self._timeout_s = float(timeout_s)
        self._client = QdrantClient(url=url, timeout=self._timeout_s)
        self._collection = manifest.collection
        try:
            info = self._client.get_collection(self._collection)
        except Exception as error:
            raise RetrievalUnavailable(f"无法读取 Qdrant collection: {error}") from error
        config = getattr(info, "config", None)
        params = getattr(config, "params", None)
        vectors = getattr(params, "vectors", None)
        if getattr(vectors, "size", None) != manifest.embedding.dimension:
            raise ValueError("Qdrant collection 维度与 manifest 不一致或不是单向量配置")
        distance = getattr(vectors, "distance", None)
        if getattr(distance, "value", distance) != "Cosine":
            raise ValueError("Qdrant collection 必须使用 Cosine 距离")
        metadata = getattr(config, "metadata", None)
        if not isinstance(metadata, dict) or VECTOR_BUILD_METADATA_KEY not in metadata:
            raise ValueError("Qdrant collection 缺少构建元数据；请使用 1.16+ 服务和客户端重建")
        if metadata[VECTOR_BUILD_METADATA_KEY] != manifest.build_metadata():
            raise ValueError("Qdrant collection 构建元数据与 manifest 不一致，请加载同版本产物")
        actual_hnsw = getattr(config, "hnsw_config", None)
        if any(type(getattr(actual_hnsw, name, None)) is not type(expected)
               or getattr(actual_hnsw, name, None) != expected
               for name, expected in manifest.index_params.items()):
            raise ValueError("Qdrant actual HNSW build parameters mismatch")
        try:
            count = self._client.count(collection_name=self._collection, exact=True).count
        except Exception as error:
            raise RetrievalUnavailable(f"无法核对 Qdrant collection 点数: {error}") from error
        if type(count) is not int or count != manifest.n_docs:
            raise ValueError(f"Qdrant collection 点数与 manifest 不一致: {count} != {manifest.n_docs}")

    def manifest(self) -> VectorManifest:
        return self._manifest

    @property
    def search_config(self) -> dict:
        return {**self.search_params, "timeout_s": self._timeout_s}

    def search(self, query_vector: Sequence[float], limit: int) -> list[VectorHit]:
        validate_vector(query_vector, self._manifest.embedding)
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit 必须是正整数")
        try:
            params = self.search_params
            if (not isinstance(params, dict) or set(params) != {"hnsw_ef", "exact"}
                    or (params["hnsw_ef"] is not None
                        and (type(params["hnsw_ef"]) is not int or params["hnsw_ef"] <= 0))
                    or type(params["exact"]) is not bool):
                raise ValueError("invalid Qdrant search parameters")
            search_options = {}
            if params["hnsw_ef"] is not None or params["exact"]:
                from qdrant_client import models

                search_options["search_params"] = models.SearchParams(**params)
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self._collection, query=list(query_vector), limit=limit,
                    with_payload=True, **search_options,
                )
                points = getattr(response, "points", response)
            else:
                points = self._client.search(
                    collection_name=self._collection, query_vector=list(query_vector),
                    limit=limit, with_payload=True, **search_options,
                )
        except Exception as error:
            raise RetrievalUnavailable(f"Qdrant 查询失败: {error}") from error
        hits = []
        try:
            for point in points:
                docid = int(point.id)
                score = float(point.score)
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if type(point.id) is not int or not isinstance(doc_id, str):
                    raise ValueError("Qdrant payload 缺少有效 docid")
                if not math.isfinite(score) or not -1.0001 <= score <= 1.0001:
                    raise ValueError("Qdrant 返回非法 cosine 分数")
                hits.append(VectorHit(docid, doc_id, score))
        except (TypeError, ValueError, AttributeError) as error:
            raise RetrievalUnavailable(f"Qdrant 返回违反向量契约: {error}") from error
        return sorted(hits, key=lambda hit: (-hit.score, hit.docid))[:limit]
