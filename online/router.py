"""首轮检索后的轻量规则路由，不读取 qrels，也不调用语言模型。

阈值仅为可执行骨架默认值，不代表经过验证的质量/成本最优配置。
"""

from dataclasses import dataclass, field
from typing import Protocol

from contracts.retrieval import RetrievalConfig


@dataclass(frozen=True)
class QueryFeatures:
    explicit_constraints: bool
    n_terms: int
    vocab_coverage: float
    lexical_hits: int
    relative_gap: float | None
    statistics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RouteDecision:
    use_vector: bool
    reason: str
    semantic_mode: str = 'hybrid'


class Router(Protocol):
    """Local prototype seam; coordinate its final contract with RFC-0003."""
    policy_version: str
    requires_statistics: bool

    def decide(self, features: QueryFeatures) -> RouteDecision: ...


class RuleRouter:
    requires_statistics = False

    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.policy_version = config.policy_version if config.adaptive_policy=='rules-v1' else config.adaptive_policy

    def decide(self, features: QueryFeatures) -> RouteDecision:
        if features.explicit_constraints:
            return RouteDecision(False, "explicit_constraints")
        if self.config.adaptive_policy == "hybrid-v1":
            return RouteDecision(True, "quality_first_hybrid")
        if features.n_terms == 0 or features.lexical_hits == 0:
            return RouteDecision(True, "no_lexical_hits")
        if features.vocab_coverage < self.config.min_vocab_coverage:
            return RouteDecision(True, "low_vocab_coverage")
        if self.config.adaptive_policy == "recall-v1" and features.lexical_hits < 2:
            return RouteDecision(True, "insufficient_lexical_evidence")
        if (features.relative_gap is not None
                and features.relative_gap < self.config.min_relative_gap):
            return RouteDecision(True, "small_relative_gap")
        return RouteDecision(False, "lexical_sufficient")
