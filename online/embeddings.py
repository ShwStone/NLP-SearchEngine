"""Optional Sentence-Transformers encoder adapter.

The default repository remains zero-dependency. Importing this module does not
load a model; a clear ``RetrievalUnavailable`` is raised only when the optional
dependency or the requested model revision is unavailable.
"""

from __future__ import annotations

from contracts.retrieval import EmbeddingSpec, RetrievalUnavailable, validate_vector


class SentenceTransformerEncoder:
    """Encode documents and queries with one pinned Sentence-Transformers model."""

    def __init__(self, model_id: str, revision: str, dimension: int,
                 max_tokens: int = 512, query_prefix: str = "",
                 document_prefix: str = "", batch_size: int = 32):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        self.spec = EmbeddingSpec(
            model_id=model_id, revision=revision, dimension=dimension,
            max_tokens=max_tokens, query_prefix=query_prefix,
            document_prefix=document_prefix,
        )
        self.batch_size = batch_size
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RetrievalUnavailable(
                "SentenceTransformerEncoder 需要安装 sentence-transformers"
            ) from error
        try:
            self._model = SentenceTransformer(model_id, revision=revision)
            self._model.max_seq_length = max_tokens
        except Exception as error:
            raise RetrievalUnavailable(f"无法加载 embedding 模型: {error}") from error

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            values = self._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=False,
                show_progress_bar=False, batch_size=self.batch_size,
            )
            vectors = [[float(component) for component in vector] for vector in values]
        except Exception as error:
            raise RetrievalUnavailable(f"embedding 编码失败: {error}") from error
        for vector in vectors:
            validate_vector(vector, self.spec)
        return vectors

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.spec.document_prefix + text for text in texts])

    def encode_query(self, query: str) -> list[float]:
        return self._encode([self.spec.query_prefix + query])[0]
