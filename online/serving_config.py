"""Strict ServingConfig v1 loader for the online HTTP service.

The configuration is a versioned service profile, not a bag of overrides:
unknown fields, wrong-but-coercible types, non-finite numbers and blank paths
are rejected. Paths are interpreted literally relative to the configuration
file's directory (no ``~``/``$VAR`` interpolation), so a copied profile cannot
silently point at a different artifact set.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from contracts.retrieval import RetrievalConfig

SCHEMA_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_BODY_BYTES = 65536
DEFAULT_REQUEST_TIMEOUT_S = 10.0
DEFAULT_VECTOR_TIMEOUT_S = 30.0
DEFAULT_CANDIDATE_K = 200
DEFAULT_RRF_C = 60.0
DEFAULT_ROUTER_FACTORY = "rule"
DEFAULT_DEVICE = "cpu"

_TOP_LEVEL_KEYS = frozenset({
    "schema_version", "dataset", "index", "vector", "embedding",
    "retrieval", "serve", "policy", "rerank",
})
_VECTOR_KEYS = frozenset({"url", "manifest", "hnsw_ef", "timeout_s"})
_EMBEDDING_KEYS = frozenset({"model_path", "device"})
_RETRIEVAL_KEYS = frozenset({"candidate_k", "rrf_c"})
_SERVE_KEYS = frozenset({"host", "port", "max_body_bytes", "request_timeout_s"})
_POLICY_KEYS = frozenset({"factory"})
_RERANK_KEYS = frozenset({"model_path", "depth", "device", "batch_size",
                          "capture_features", "gate"})
_GATE_KEYS = frozenset({"model_path", "sha256"})


class ServingConfigError(ValueError):
    """The serving profile violates the frozen v1 schema."""


@dataclass(frozen=True)
class VectorConfig:
    url: str
    manifest: Path
    hnsw_ef: int | None = None
    timeout_s: float = DEFAULT_VECTOR_TIMEOUT_S


@dataclass(frozen=True)
class EmbeddingConfig:
    model_path: Path | None = None
    device: str | None = None


@dataclass(frozen=True)
class RerankConfig:
    model_path: Path
    depth: int
    device: str = DEFAULT_DEVICE
    batch_size: int = 20
    capture_features: bool = False
    gate_path: Path | None = None
    gate_sha256: str | None = None


@dataclass(frozen=True)
class RetrievalSettings:
    candidate_k: int = DEFAULT_CANDIDATE_K
    rrf_c: float = DEFAULT_RRF_C


@dataclass(frozen=True)
class ServeSettings:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S


@dataclass(frozen=True)
class ServingConfig:
    schema_version: int
    dataset: str
    index: Path
    vector: VectorConfig | None
    embedding: EmbeddingConfig | None
    retrieval: RetrievalSettings
    serve: ServeSettings
    policy_factory: str
    rerank: RerankConfig | None = None

    def retrieval_config(self) -> RetrievalConfig:
        return RetrievalConfig(candidate_k=self.retrieval.candidate_k,
                               rrf_c=self.retrieval.rrf_c)

    def serving_info(self) -> dict:
        """The frozen ``retrieval.serving`` block for ``/index/info``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "profile": self.dataset,
            "host": self.serve.host,
            "port": self.serve.port,
            "max_body_bytes": self.serve.max_body_bytes,
            "request_timeout_s": self.serve.request_timeout_s,
            "router_factory": self.policy_factory,
        }


def _reject_json_constant(value: str):
    raise ServingConfigError(f"配置包含非有限数值: {value}")


def _load_json(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        reason = error.strerror or type(error).__name__
        raise ServingConfigError(f"无法读取配置文件 {path}: {reason}") from error
    try:
        value = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ServingConfigError(
            f"配置文件不是合法 JSON（第 {error.lineno} 行第 {error.colno} 列）"
        ) from error
    if not isinstance(value, dict):
        raise ServingConfigError("配置顶层必须是 JSON 对象")
    return value


def _object(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ServingConfigError(f"{label} 必须是 JSON 对象")
    return value


def _reject_unknown(value: dict, allowed: frozenset, label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ServingConfigError(f"{label} 包含未知字段: {unknown}")


def _require(block: dict, names: tuple, label: str) -> None:
    missing = [name for name in names if name not in block]
    if missing:
        raise ServingConfigError(f"{label} 缺少必填字段: {missing}")


def _text(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ServingConfigError(f"{label} 必须是非空字符串")
    if value != value.strip():
        raise ServingConfigError(f"{label} 不能包含首尾空白")
    if any(ord(char) < 32 for char in value):
        raise ServingConfigError(f"{label} 不能包含控制字符")
    return value


def _integer(value, label: str, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ServingConfigError(f"{label} 必须是整数（不接受布尔值或字符串）")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ServingConfigError(f"{label} 必须 >= {minimum}")
        raise ServingConfigError(f"{label} 必须在 {minimum}..{maximum} 之间")
    return value


def _positive_number(value, label: str) -> float:
    if type(value) is bool or not isinstance(value, (int, float)):
        raise ServingConfigError(f"{label} 必须是有限正数")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ServingConfigError(f"{label} 必须是有限正数")
    return number


def _resolve_path(base: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ServingConfigError(f"{label} 必须是非空路径字符串")
    if "\x00" in value:
        raise ServingConfigError(f"{label} 不能包含 NUL 字符")
    path = Path(value)
    # Absolute paths are kept absolute; everything else is config-relative.
    # ``Path`` never expands ``~`` or ``$VAR``, which is exactly the contract.
    return (path if path.is_absolute() else base / path).resolve()


def validate_vector_url(value, label: str = "vector.url") -> str:
    """Accept only credential-free ``http(s)`` URLs; never echo the value."""
    text = _text(value, label)
    if any(char.isspace() for char in text):
        raise ServingConfigError(f"{label} 不能包含空白字符")
    try:
        parts = urlsplit(text)
    except ValueError as error:
        raise ServingConfigError(f"{label} 不是合法的 http(s) 服务地址") from error
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ServingConfigError(f"{label} 必须是 http(s) 服务地址")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ServingConfigError(f"{label} 不能包含 userinfo；请勿把凭据写入配置")
    if parts.query or parts.fragment or "?" in text or "#" in text:
        raise ServingConfigError(f"{label} 不能包含 query 或 fragment")
    return text


def load_serving_config(path) -> ServingConfig:
    """Load and strictly validate one ServingConfig v1 JSON profile."""
    config_path = Path(path).resolve()
    value = _load_json(config_path)
    base = config_path.parent
    _reject_unknown(value, _TOP_LEVEL_KEYS, "配置")
    _require(value, ("schema_version", "dataset", "index"), "配置")

    schema_version = _integer(value["schema_version"], "schema_version",
                              SCHEMA_VERSION, SCHEMA_VERSION)
    dataset = _text(value["dataset"], "dataset")
    index = _resolve_path(base, value["index"], "index")

    vector = None
    if "vector" in value:
        block = _object(value["vector"], "vector")
        _reject_unknown(block, _VECTOR_KEYS, "vector")
        _require(block, ("url", "manifest"), "vector")
        vector = VectorConfig(
            url=validate_vector_url(block["url"]),
            manifest=_resolve_path(base, block["manifest"], "vector.manifest"),
            hnsw_ef=(_integer(block["hnsw_ef"], "vector.hnsw_ef", 1)
                     if "hnsw_ef" in block else None),
            timeout_s=(_positive_number(block["timeout_s"], "vector.timeout_s")
                       if "timeout_s" in block else DEFAULT_VECTOR_TIMEOUT_S),
        )

    embedding = None
    if "embedding" in value:
        if vector is None:
            raise ServingConfigError("embedding 不能脱离 vector 单独出现")
        block = _object(value["embedding"], "embedding")
        _reject_unknown(block, _EMBEDDING_KEYS, "embedding")
        model_path = (_resolve_path(base, block["model_path"], "embedding.model_path")
                      if "model_path" in block else None)
        device = None
        if "device" in block:
            if model_path is None:
                raise ServingConfigError("embedding.device 只在提供 embedding.model_path 时有意义")
            device = _text(block["device"], "embedding.device")
        if model_path is not None and device is None:
            device = DEFAULT_DEVICE
        embedding = EmbeddingConfig(model_path=model_path, device=device)

    retrieval = RetrievalSettings()
    if "retrieval" in value:
        block = _object(value["retrieval"], "retrieval")
        _reject_unknown(block, _RETRIEVAL_KEYS, "retrieval")
        retrieval = RetrievalSettings(
            candidate_k=(_integer(block["candidate_k"], "retrieval.candidate_k", 1)
                         if "candidate_k" in block else DEFAULT_CANDIDATE_K),
            rrf_c=(_positive_number(block["rrf_c"], "retrieval.rrf_c")
                   if "rrf_c" in block else DEFAULT_RRF_C),
        )
    # Reuse the frozen retrieval contract for its range checks as well.
    RetrievalConfig(candidate_k=retrieval.candidate_k, rrf_c=retrieval.rrf_c)

    serve = ServeSettings()
    if "serve" in value:
        block = _object(value["serve"], "serve")
        _reject_unknown(block, _SERVE_KEYS, "serve")
        serve = ServeSettings(
            host=(_text(block["host"], "serve.host") if "host" in block else DEFAULT_HOST),
            port=(_integer(block["port"], "serve.port", 0, 65535)
                  if "port" in block else DEFAULT_PORT),
            max_body_bytes=(_integer(block["max_body_bytes"], "serve.max_body_bytes", 1)
                            if "max_body_bytes" in block else DEFAULT_MAX_BODY_BYTES),
            request_timeout_s=(_positive_number(block["request_timeout_s"],
                                                "serve.request_timeout_s")
                               if "request_timeout_s" in block
                               else DEFAULT_REQUEST_TIMEOUT_S),
        )

    policy_factory = DEFAULT_ROUTER_FACTORY
    if "policy" in value:
        block = _object(value["policy"], "policy")
        _reject_unknown(block, _POLICY_KEYS, "policy")
        if "factory" in block:
            policy_factory = _text(block["factory"], "policy.factory")

    rerank = None
    if "rerank" in value:
        block = _object(value["rerank"], "rerank")
        _reject_unknown(block, _RERANK_KEYS, "rerank")
        _require(block, ("model_path", "depth"), "rerank")
        if vector is None:
            raise ServingConfigError("rerank 需要 vector 配置")
        capture_features = block.get("capture_features", False)
        if type(capture_features) is not bool:
            raise ServingConfigError("rerank.capture_features 必须是布尔值")
        depth = _integer(block["depth"], "rerank.depth", 1, min(40, retrieval.candidate_k))
        gate_path = gate_sha256 = None
        if "gate" in block:
            gate = _object(block["gate"], "rerank.gate")
            _reject_unknown(gate, _GATE_KEYS, "rerank.gate")
            _require(gate, ("model_path", "sha256"), "rerank.gate")
            if depth != 40:
                raise ServingConfigError("rerank.gate 需要 depth=40")
            gate_path = _resolve_path(base, gate["model_path"], "rerank.gate.model_path")
            gate_sha256 = _text(gate["sha256"], "rerank.gate.sha256").lower()
            if len(gate_sha256) != 64 or any(char not in "0123456789abcdef" for char in gate_sha256):
                raise ServingConfigError("rerank.gate.sha256 必须是 64 位十六进制 SHA256")
        rerank = RerankConfig(
            model_path=_resolve_path(base, block["model_path"], "rerank.model_path"),
            depth=depth,
            device=_text(block.get("device", DEFAULT_DEVICE), "rerank.device"),
            batch_size=_integer(block.get("batch_size", 20), "rerank.batch_size", 1, 32),
            capture_features=capture_features,
            gate_path=gate_path, gate_sha256=gate_sha256,
        )

    return ServingConfig(
        schema_version=schema_version,
        dataset=dataset,
        index=index,
        vector=vector,
        embedding=embedding,
        retrieval=retrieval,
        serve=serve,
        policy_factory=policy_factory,
        rerank=rerank,
    )
