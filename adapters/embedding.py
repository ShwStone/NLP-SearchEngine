"""Frozen BGE CLS encoder, shared by document and query entry points."""

import threading

from contracts.retrieval import EmbeddingSpec, RetrievalUnavailable

BGE_MODEL = "BAAI/bge-small-zh-v1.5"
BGE_REVISION = "7999e1d3359715c523056ef9478215996d62a620"


def bge_spec():
    return EmbeddingSpec(BGE_MODEL, BGE_REVISION, dimension=512,
                         query_prefix="为这个句子生成表示以用于检索相关文章：")


class BGEEncoder:
    def __init__(self, spec=None, model_path=None, device="cpu", batch_size=32):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec or bge_spec()
        if self.spec.model_id != BGE_MODEL or self.spec.revision != BGE_REVISION:
            raise ValueError("BGEEncoder only supports the pinned bge-small-zh-v1.5 revision")
        if self.spec.dimension != 512 or self.spec.max_tokens > 512:
            raise ValueError("BGE dimension/max_tokens mismatch")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.device = device
        self._lock = threading.Lock()
        source = model_path or self.spec.model_id
        kwargs = {"local_files_only": True} if model_path else {"revision": self.spec.revision}
        # A local copy must carry the downloaded revision's SHA256 inventory.
        if model_path:
            from .model_files import verify_model
            verify_model(model_path, self.spec)
        self.tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        self.tokenizer.truncation_side = "right"
        self.model = AutoModel.from_pretrained(source, use_safetensors=True, **kwargs).to(device).eval()
        self._torch = torch

    def _encode(self, texts):
        try:
            vectors = []
            with self._lock, self._torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    inputs = self.tokenizer(texts[start:start + self.batch_size], padding=True,
                                            truncation=True, max_length=self.spec.max_tokens,
                                            return_tensors="pt").to(self.device)
                    # BGE v1.5 uses CLS pooling, not mean pooling.
                    cls = self.model(**inputs).last_hidden_state[:, 0]
                    normalized = self._torch.nn.functional.normalize(cls.float(), p=2, dim=1)
                    vectors.extend(normalized.cpu().tolist())
            return vectors
        except Exception as error:
            raise RetrievalUnavailable(f"BGE encoding failed: {error}") from error

    def encode_documents(self, texts):
        return self._encode([self.spec.document_prefix + text for text in texts])

    def encode_query(self, query):
        return self._encode([self.spec.query_prefix + query])[0]
