"""Local, revision-pinned BGE cross-encoder for online reranking."""

from collections.abc import Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import threading

from contracts.index_format import sha256_of
from contracts.retrieval import RetrievalUnavailable

MODEL_ID = "BAAI/bge-reranker-base"
REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"
MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 20
MAX_BATCH_SIZE = 32
MAX_DEPTH = 200

# Hugging Face's tree for REVISION publishes LFS SHA256s and Git blob IDs.
WEIGHTS_SHA256 = "ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd"
_LFS_SHA256 = {
    "model.safetensors": WEIGHTS_SHA256,
    "tokenizer.json": "9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
}
_GIT_BLOBS = {
    "config.json": "ff248a5f7d76e27b10778b3c593c9fba1df4c448",
    "tokenizer_config.json": "059214673d9d6d2ee319411e2ffec8c024b816d5",
    "special_tokens_map.json": "68171d1ff68b731a33d119708476692c094a466b",
}
_REQUIRED_FILES = {*_GIT_BLOBS, "model.safetensors", "tokenizer.json"}
_INVENTORY = "model_inventory.json"
_UNUSED_SNAPSHOT_ENTRIES = {".gitattributes", "README.md", "onnx", "pytorch_model.bin"}


def _verify_model(path):
    inventory_bytes = (path / _INVENTORY).read_bytes()
    inventory = json.loads(inventory_bytes)
    if (inventory["model_id"], inventory["revision"]) != (MODEL_ID, REVISION):
        raise ValueError("local cross-encoder identity does not match the frozen revision")
    inventory_files = inventory["files"]
    allowed_model_files = _GIT_BLOBS.keys() | _LFS_SHA256.keys()
    entries = {entry.name for entry in path.iterdir()}
    if entries - allowed_model_files - _UNUSED_SNAPSHOT_ENTRIES - {_INVENTORY}:
        raise ValueError("cross-encoder directory contains an unexpected entry")
    if not isinstance(inventory_files, dict) or not _REQUIRED_FILES <= inventory_files.keys():
        raise ValueError("cross-encoder inventory is missing a required model file")
    if inventory_files.keys() - allowed_model_files - _UNUSED_SNAPSHOT_ENTRIES:
        raise ValueError("cross-encoder inventory contains an unexpected entry")
    for name in sorted((entries | inventory_files.keys()) & allowed_model_files):
        file = path / name
        if name in _GIT_BLOBS:
            data = file.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
            if blob != _GIT_BLOBS[name]:
                raise ValueError(f"cross-encoder revision checksum mismatch: {name}")
            digest = hashlib.sha256(data).hexdigest()
        else:
            digest = sha256_of(file)
            if digest != _LFS_SHA256[name]:
                raise ValueError(f"cross-encoder revision checksum mismatch: {name}")
        if inventory_files.get(name) != digest:
            raise ValueError(f"cross-encoder inventory checksum mismatch: {name}")
    return hashlib.sha256(inventory_bytes).hexdigest()


class FrozenCrossEncoder:
    def __init__(self, path, *, device="cpu", batch_size=DEFAULT_BATCH_SIZE):
        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"cross-encoder batch size must be an integer from 1 to {MAX_BATCH_SIZE}")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
            raise ValueError("cross-encoder device must be cpu, cuda or cuda:<index>")
        root = Path(path).resolve()
        try:
            inventory_sha256 = _verify_model(root)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RetrievalUnavailable(f"cross-encoder model verification failed: {error}") from error
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            local_options = {"local_files_only": True, "trust_remote_code": False}
            self._tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True, **local_options)
            if not self._tokenizer.is_fast:
                raise ValueError("cross-encoder requires the verified fast tokenizer")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                root, use_safetensors=True, **local_options
            ).float().to(device).eval()
        except Exception as error:
            raise RetrievalUnavailable(f"cross-encoder model initialization failed: {error}") from error
        self._torch = torch
        self._device = device
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._identity = {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "weights_sha256": WEIGHTS_SHA256,
            "inventory_sha256": inventory_sha256,
            "device": device,
            "dtype": "float32",
            "batch_size": batch_size,
            "batch_strategy": "input-order",
            "max_length": MAX_LENGTH,
        }

    @property
    def identity(self):
        return dict(self._identity)

    def score(self, query, texts):
        if (not isinstance(query, str) or not isinstance(texts, Sequence)
                or isinstance(texts, (str, bytes))):
            raise ValueError("cross-encoder expects a query and a sequence of document texts")
        if len(texts) > MAX_DEPTH:
            raise ValueError(f"cross-encoder supports at most {MAX_DEPTH} documents per query")
        if any(not isinstance(document, str) for document in texts):
            raise ValueError("cross-encoder document texts must be strings")
        values = []
        try:
            with self._lock, self._torch.inference_mode():
                for start in range(0, len(texts), self._batch_size):
                    batch = list(texts[start:start + self._batch_size])
                    inputs = self._tokenizer(
                        [query] * len(batch), batch, padding=True, truncation=True,
                        max_length=MAX_LENGTH, return_tensors="pt"
                    ).to(self._device)
                    logits = self._model(**inputs).logits
                    if tuple(logits.shape) != (len(batch), 1):
                        raise ValueError("cross-encoder returned an unexpected logit shape")
                    scores = logits.reshape(-1).float().cpu().tolist()
                    if len(scores) != len(batch) or any(not math.isfinite(score) for score in scores):
                        raise ValueError("cross-encoder returned invalid logits")
                    values.extend(scores)
        except Exception as error:
            raise RetrievalUnavailable(f"cross-encoder scoring failed: {error}") from error
        return values
