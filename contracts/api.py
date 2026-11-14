"""接口 B：在线服务对外的 HTTP API（在线 → UI / 评测器）。

    GET  /health      -> {"status": "ok", "index": {"dataset", "n_docs", "schema_version", "analyzer_hash"}}
    GET  /index/info  -> manifest + retrieval 能力/配置
    POST /search      -> SearchRequest 进，SearchResponse 出

三条硬规则（contracts/RETRIEVAL.md）：
1. highlights 的 start/end 是 doc.text 上的字符下标，UI 只做合并与 HTML 转义；
2. 评测器只走 HTTP，不许 import 内部模块；输出 TREC run 文件；
3. 同分按 docid 升序，保证结果可复现。
"""

from dataclasses import asdict, dataclass, field

API_VERSION = "2.0"
SEARCH_INSTANCE_HEADER = "X-Search-Instance"
MODES = ("lexical", "dense", "hybrid", "adaptive")
SCORERS = ("bm25", "tfidf")
EXPANDERS = ("none", "rm3", "slm")


class ApiError(ValueError):
    """请求参数不合法。server 层转成 HTTP 400。"""


@dataclass(frozen=True)
class SearchRequest:
    query: str
    k: int = 10
    offset: int = 0
    scorer: str = "bm25"
    expander: str = "none"
    run_tag: str = ""

    mode: str = "lexical"     # 默认保留无向量依赖的原调用方式

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ApiError("query 必须是非空字符串")
        if type(self.k) is not int or not 1 <= self.k <= 200:
            raise ApiError("k 必须是 1..200 的整数")
        if type(self.offset) is not int or self.offset < 0:
            raise ApiError("offset 必须是非负整数")
        if self.scorer not in SCORERS:
            raise ApiError(f"scorer 只能是 {SCORERS}")
        if self.expander not in EXPANDERS:
            raise ApiError(f"expander 只能是 {EXPANDERS}")
        if self.mode not in MODES:
            raise ApiError(f"mode 只能是 {MODES}")
        if not isinstance(self.run_tag, str):
            raise ApiError("run_tag 必须是字符串")
        if self.mode != "lexical" and (self.scorer != "bm25" or self.expander != "none"):
            raise ApiError("非 lexical 模式仅支持 scorer=bm25、expander=none")

    @staticmethod
    def from_dict(d: dict) -> "SearchRequest":
        if not isinstance(d, dict):
            raise ApiError("请求必须是 JSON 对象")
        unknown = set(d) - set(SearchRequest.__dataclass_fields__)
        if unknown or "query" not in d:
            raise ApiError(f"请求缺 query 或包含未知字段: {sorted(unknown)}")
        return SearchRequest(**d)


@dataclass(frozen=True)
class Highlight:
    start: int   # doc.text 字符下标（含）
    end: int     # doc.text 字符下标（不含）
    kind: str    # "original" | "expanded"

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "kind": self.kind}


@dataclass(frozen=True)
class ResultItem:
    doc_id: str
    rank: int
    score: float
    title: str
    snippet: str
    highlights: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "rank": self.rank,
            "score": round(self.score, 6),
            "title": self.title,
            "snippet": self.snippet,
            "highlights": [h.to_dict() for h in self.highlights],
        }


@dataclass(frozen=True)
class RoutingInfo:
    requested_mode: str
    executed_mode: str
    reason: str
    vector_used: bool
    vector_attempted: bool
    fallback: bool
    score_kind: str
    candidate_k: int
    policy_version: str


@dataclass(frozen=True)
class SearchResponse:
    took_ms: int
    parsed: dict          # 查询解析回显，如 {"type": "terms", "terms": [...]}
    expanded: list        # [{"term": str, "w": float, "src": "rm3"|"slm"}]
    results: list         # [ResultItem]
    routing: RoutingInfo
    timings_ms: dict = field(default_factory=dict)
    rerank: dict | None = None

    def to_dict(self) -> dict:
        return {
            "api_version": API_VERSION,
            "routing": asdict(self.routing),
            "timings_ms": self.timings_ms,
            "took_ms": self.took_ms,
            "parsed": self.parsed,
            "expanded": self.expanded,
            "results": [r.to_dict() for r in self.results],
            **({"rerank": self.rerank} if self.rerank is not None else {}),
        }


# 给 AI 生成的 UI / 评测器做契约测试用
REQUEST_JSON_SCHEMA = {
    "type": "object",
    "required": ["query"],
    "additionalProperties": False,
    "allOf": [{
        "if": {"required": ["mode"], "properties": {"mode": {"enum": ["dense", "hybrid", "adaptive"]}}},
        "then": {"properties": {"scorer": {"const": "bm25"}, "expander": {"const": "none"}}},
    }],
    "properties": {
        "query": {"type": "string", "minLength": 1, "pattern": "\\S"},
        "k": {"type": "integer", "minimum": 1, "maximum": 200, "default": 10},
        "offset": {"type": "integer", "minimum": 0, "default": 0},
        "scorer": {"enum": list(SCORERS), "default": "bm25"},
        "expander": {"enum": list(EXPANDERS), "default": "none"},
        "run_tag": {"type": "string", "default": ""},
        "mode": {"enum": list(MODES), "default": "lexical"},
    },
}

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "required": ["api_version", "took_ms", "parsed", "expanded", "results", "routing", "timings_ms"],
    "properties": {
        "api_version": {"const": API_VERSION},
        "routing": {
            "type": "object",
            "required": ["requested_mode", "executed_mode", "reason", "vector_used", "vector_attempted", "fallback", "score_kind", "candidate_k", "policy_version"],
            "properties": {
                "requested_mode": {"enum": list(MODES)},
                "executed_mode": {"enum": ["lexical", "dense", "hybrid"]},
                "reason": {"type": "string"},
                "vector_used": {"type": "boolean"},
                "vector_attempted": {"type": "boolean"},
                "fallback": {"type": "boolean"},
                "score_kind": {"enum": ["bm25", "tfidf", "cosine", "rrf"]},
                "candidate_k": {"type": "integer", "minimum": 1},
                "policy_version": {"type": "string"},
            },
        },
        "timings_ms": {
            "type": "object",
            "required": ["lexical", "router", "encode", "vector", "fusion"],
            "properties": {name: {"type": "number", "minimum": 0}
                           for name in ("lexical", "router", "encode", "vector", "fusion", "rerank")},
            "additionalProperties": False,
        },
        "rerank": {
            "type": "object",
            "required": ["depth", "scored"],
            "additionalProperties": False,
            "properties": {
                "depth": {"type": "integer", "minimum": 1, "maximum": 40},
                "scored": {"type": "integer", "minimum": 0, "maximum": 40},
                "features": {"type": "array", "minItems": 4, "maxItems": 4,
                             "items": {"type": "number"}},
                "gate_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            },
        },
        "took_ms": {"type": "integer", "minimum": 0},
        "parsed": {"type": "object"},
        "expanded": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["term", "w", "src"],
                "properties": {
                    "term": {"type": "string"},
                    "w": {"type": "number"},
                    "src": {"enum": ["rm3", "slm"]},
                },
            },
        },
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["doc_id", "rank", "score", "title", "snippet", "highlights"],
                "properties": {
                    "doc_id": {"type": "string"},
                    "rank": {"type": "integer"},
                    "score": {"type": "number"},
                    "title": {"type": "string"},
                    "snippet": {"type": "string"},
                    "highlights": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["start", "end", "kind"],
                            "properties": {
                                "start": {"type": "integer", "minimum": 0},
                                "end": {"type": "integer", "minimum": 0},
                                "kind": {"enum": ["original", "expanded"]},
                            },
                        },
                    },
                },
            },
        },
    },
}
