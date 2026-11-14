"""首轮检索后的轻量规则路由，不读取 qrels，也不调用语言模型。

阈值仅为可执行骨架默认值，不代表经过验证的质量/成本最优配置。

这里同时定义外部策略的注入缝：`Router` 只要求固定的 `policy_version` 和非空
`reason` 的 `decide(features)`。统计钩子由 SearchEngine 在 adaptive 决策窗口内调用。
"""

import importlib
from dataclasses import dataclass, field
from typing import Protocol

from contracts.retrieval import RetrievalConfig

from .router_features import FEATURE_VERSION, FEATURE_VERSION_V2


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
    semantic_mode: str = "hybrid"


class Router(Protocol):
    """策略缝：policy_version 与 decide 必填。

    可选属性：
    - `name`：info() 中展示的策略名；缺省用类名。
    - `requires_statistics = True`：此时须实现
      `statistics(reader, terms_tuple, ranked_tuple) -> dict` 或
      `statistics_for_query(query, reader, terms_tuple, ranked_tuple) -> dict`。
    """

    policy_version: str

    def decide(self, features: QueryFeatures) -> RouteDecision:
        ...


class RuleRouter:
    name = "rule"
    requires_statistics = False

    def __init__(self, config: RetrievalConfig):
        self.config = config

    @property
    def policy_version(self) -> str:
        if self.config.adaptive_policy == "rules-v1":
            return self.config.policy_version
        return self.config.adaptive_policy

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


def router_identity(router) -> dict:
    """校验策略契约并返回可公开的策略身份；不信任注入对象的字段类型。"""
    version = getattr(router, "policy_version", None)
    if not isinstance(version, str) or not version:
        raise ValueError("路由策略必须提供非空字符串 policy_version")
    if not callable(getattr(router, "decide", None)):
        raise ValueError("路由策略必须实现 decide(features)")
    requires_statistics = getattr(router, "requires_statistics", False)
    if type(requires_statistics) is not bool:
        raise ValueError("路由策略 requires_statistics 必须是布尔值")
    if (requires_statistics
            and not callable(getattr(router, "statistics", None))
            and not callable(getattr(router, "statistics_for_query", None))
            and getattr(router, "feature_version", None) not in (FEATURE_VERSION, FEATURE_VERSION_V2)):
        raise ValueError(
            "requires_statistics=True 时必须实现统计钩子或声明支持的特征版本")
    name = getattr(router, "name", None)
    if not isinstance(name, str) or not name:
        name = type(router).__name__
    return {"name": name, "policy_version": version,
            "requires_statistics": requires_statistics}


def load_router(selector: str, config: RetrievalConfig) -> Router:
    """按配置选择策略：``rule`` 内置；其他值为受信任本地 ``module:factory``。"""
    if not isinstance(selector, str) or not selector:
        raise ValueError("policy selector 必须是非空字符串")
    if selector == "rule":
        router = RuleRouter(config)
    else:
        module_name, separator, factory_name = selector.partition(":")
        if not separator or not module_name or not factory_name or ":" in factory_name:
            raise ValueError("自定义路由策略必须使用 module:factory 形式")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise ValueError(f"无法导入路由策略模块 {module_name}: {error}") from error
        factory = getattr(module, factory_name, None)
        if not callable(factory):
            raise ValueError(f"路由策略模块 {module_name} 缺少可调用工厂 {factory_name}")
        router = factory(config)
    router_identity(router)
    return router
