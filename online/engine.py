"""在线混合检索骨架：parse -> BM25 -> router -> 可选向量召回 -> RRF。

真实模型/数据库通过 contracts.retrieval 注入；词法模式不需要向量依赖。
"""

import math
import time
from dataclasses import asdict
from heapq import nsmallest

from contracts import analyzer
from contracts.api import ApiError, ResultItem, RoutingInfo, SearchRequest, SearchResponse
from contracts.reader import IndexReader
from contracts.retrieval import (
    RetrievalConfig, RetrievalUnavailable, TextEncoder, VectorReader,
    check_alignment, valid_query_encoder_runtime, validate_vector,
)

from . import parser, scorers
from .ce_gate import features as ce_gate_features
from .expanders import EXPANDERS
from .fusion import rrf
from .highlight import build_highlights
from .router import QueryFeatures, RouteDecision, Router, RuleRouter, router_identity
from .router_features import FEATURE_VERSION_V2, extract_statistics, extract_statistics_v2

SNIPPET_LEN = 80


class SearchEngine:
    def __init__(self, reader: IndexReader, encoder: TextEncoder | None = None,
                 vectors: VectorReader | None = None, config: RetrievalConfig | None = None,
                 router: Router | None = None, bm25_accelerator=None, *, reranker=None):
        self.reader = reader
        self.encoder = encoder
        self.vectors = vectors
        self.config = config or RetrievalConfig()
        if bm25_accelerator is not None and bm25_accelerator.reader is not reader:
            raise ValueError('BM25 accelerator must be bound to the same immutable reader')
        self.bm25_accelerator = bm25_accelerator
        if (encoder is None) != (vectors is None):
            raise ValueError("encoder 与 vectors 必须同时提供或同时省略")
        if vectors is not None:
            check_alignment(reader, vectors, encoder)
        runtime = getattr(encoder, "query_encoder_runtime", None)
        self._query_encoder_runtime = dict(runtime) if valid_query_encoder_runtime(runtime) else None
        self.router = RuleRouter(self.config) if router is None else router
        self.policy = router_identity(self.router)
        check_binding = getattr(self.router, "check_binding", None)
        check_retrieval_binding = getattr(self.router, "check_retrieval_binding", None)
        if callable(check_binding):
            check_binding(reader, self.config)
        if not callable(check_retrieval_binding):
            check_retrieval_binding = getattr(self.router, "check_vector_binding", None)
        if callable(check_retrieval_binding):
            check_retrieval_binding(reader, self.config, vectors)
        if reranker is not None and (vectors is None or reranker.depth > self.config.candidate_k):
            raise ValueError("混合重排需要向量后端且深度不得超过 candidate_k")
        self.reranker = reranker
        if reranker is not None:
            reranker.check_binding(self.info())

    def info(self) -> dict:
        """能力与配置回显，不把尚未实现的后端宣称为可用。

        ``vector_ready`` 只表示启动时已配置并通过对齐检查，不是后端实时健康。
        ``index_identity.corpus_sha256`` 复用启动对齐时已验证的向量指纹；纯词法
        服务不做额外全文扫描，回 null。
        """
        manifest = self.reader.manifest()
        vector = self.vectors.manifest() if self.vectors is not None else None
        if vector is not None:
            vector_identity = vector.to_dict()
            search_config = getattr(self.vectors, "search_config", None)
            vector_identity["search"] = (dict(search_config)
                                         if isinstance(search_config, dict) else None)
        else:
            vector_identity = None
        vector_search_params = getattr(self.vectors, "search_params", None)
        if (isinstance(vector_search_params, dict)
                and type(vector_search_params.get("exact")) is bool):
            ef = vector_search_params.get("hnsw_ef")
            vector_search_params = ({"hnsw_ef": ef, "exact": vector_search_params["exact"]}
                                    if ef is None or type(ef) is int else None)
        else:
            vector_search_params = None
        # ``config.policy_version`` mirrors the policy that actually took effect
        # (an injected router may declare its own); ``self.config`` itself stays
        # the untouched frozen retrieval configuration.
        config_echo = {**asdict(self.config),
                       "policy_version": self.policy["policy_version"]}
        return {
            "vector_ready": self.vectors is not None,
            "config": config_echo,
            "vector_manifest": vector.to_dict() if vector is not None else None,
            "index_identity": {
                "dataset": manifest["dataset"],
                "n_docs": manifest["n_docs"],
                "schema_version": manifest["schema_version"],
                "analyzer_hash": manifest["analyzer"]["config_hash"],
                "corpus_sha256": vector.corpus_sha256 if vector is not None else None,
                "bm25": manifest.get("bm25"),
            },
            "vector_identity": vector_identity,
            "vector_search_params": vector_search_params,
            **({"query_encoder_runtime": dict(self._query_encoder_runtime)}
               if self._query_encoder_runtime is not None else {}),
            "search_config": {"candidate_k": self.config.candidate_k,
                              "rrf_c": self.config.rrf_c},
            "policy": dict(self.policy),
            "router": {
                "policy_version": self.policy["policy_version"],
                "requires_statistics": self.policy["requires_statistics"],
            },
            "bm25_accelerator": ("compiled-float64-topk"
                                 if self.bm25_accelerator is not None else None),
            **({"reranker": self.reranker.identity} if self.reranker is not None else {}),
        }

    def _decide(self, features: QueryFeatures) -> RouteDecision:
        decision = self.router.decide(features)
        if (not isinstance(decision, RouteDecision)
                or type(decision.use_vector) is not bool
                or not isinstance(decision.reason, str) or not decision.reason
                or decision.semantic_mode not in ("dense", "hybrid")):
            raise ValueError("路由策略必须返回合法的 use_vector、reason 和 semantic_mode")
        return decision

    def _features(self, query: str, terms: list, ranked: list) -> QueryFeatures:
        unique_terms = set(terms)
        coverage = (sum(self.reader.df(t) > 0 for t in unique_terms) / len(unique_terms)
                    if unique_terms else 0.0)
        gap = ((ranked[0][1] - ranked[1][1]) / max(abs(ranked[0][1]), 1e-12)
               if len(ranked) >= 2 else None)
        statistics = {}
        if self.policy["requires_statistics"]:
            query_statistics = getattr(self.router, "statistics_for_query", None)
            args = (self.reader, tuple(terms), tuple(ranked))
            statistics_hook = getattr(self.router, "statistics", None)
            if callable(query_statistics):
                statistics = query_statistics(query, *args)
            elif callable(statistics_hook):
                statistics = statistics_hook(*args)
            else:
                # RFC-0001 policies use the engine's versioned feature extractors.
                extractor = (extract_statistics_v2
                             if self.router.feature_version == FEATURE_VERSION_V2
                             else extract_statistics)
                statistics = extractor(query, *args)
            if not isinstance(statistics, dict):
                raise ValueError("路由策略 statistics 必须返回 dict")
        return QueryFeatures(False, len(unique_terms), coverage, len(ranked), gap,
                             statistics)

    def _vector_search(self, query: str, timings: dict) -> list[tuple[int, float]]:
        if self.encoder is None or self.vectors is None:
            raise RetrievalUnavailable("未配置编码器与向量库")
        start = time.perf_counter()
        try:
            vector = self.encoder.encode_query(query)
            validate_vector(vector, self.encoder.spec)
        except ValueError as error:
            raise RetrievalUnavailable(f"查询编码器违反向量契约: {error}") from error
        finally:
            timings["encode"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        try:
            hits = self.vectors.search(vector, self.config.candidate_k)
            if len(hits) > self.config.candidate_k or len({h.docid for h in hits}) != len(hits):
                raise ValueError("向量结果数量超限或 docid 重复")
            external_id = getattr(self.reader, "external_doc_id", None)
            for hit in hits:
                if type(hit.docid) is not int or not 0 <= hit.docid < self.reader.n_docs:
                    raise ValueError("向量结果包含未知 docid")
                doc_id = (external_id(hit.docid) if callable(external_id)
                          else self.reader.doc(hit.docid).doc_id)
                if doc_id != hit.doc_id:
                    raise ValueError("向量结果外部 doc_id 与倒排索引不一致")
                if not math.isfinite(hit.score) or not -1.0001 <= hit.score <= 1.0001:
                    raise ValueError("向量结果不是合法 cosine 分数")
            return sorted(((h.docid, h.score) for h in hits), key=lambda row: (-row[1], row[0]))
        except ValueError as error:
            raise RetrievalUnavailable(f"向量后端违反契约: {error}") from error
        finally:
            timings["vector"] = (time.perf_counter() - start) * 1000

    def search(self, req: SearchRequest) -> SearchResponse:
        t0 = time.perf_counter()
        timings = {name: 0.0 for name in ("lexical", "router", "encode", "vector", "fusion")}
        if req.mode != "lexical" and req.offset + req.k > self.config.candidate_k:
            raise ApiError("非 lexical 模式 offset + k 不得超过服务 candidate_k")
        parsed = parser.parse(req.query)  # 语法错误必须报 400，不能静默放松约束
        explicit = parsed.get("explicit_constraints", False)
        if explicit and req.mode == "dense":
            raise ApiError("dense 模式无法保证显式词法约束，请使用 lexical、hybrid 或 adaptive")
        terms = [token.term for raw in parsed["terms"] for token in analyzer.analyze(raw)]
        candidate_docs = None
        if explicit:
            candidate_docs = scorers.boolean_docs(self.reader, parsed["ast"], analyzer.analyze)
        if explicit and req.expander != "none":
            raise ApiError("显式布尔/短语约束不支持 query expansion")
        expanded = EXPANDERS[req.expander](self.reader, req.query, terms)
        expanded_terms = [item["term"] for item in expanded]
        scoring_terms = terms + expanded_terms
        term_weights = {term: 1.0 for term in terms}
        term_weights.update({item["term"]: float(item["w"]) for item in expanded})

        ranked = []
        if req.mode != "dense":
            start = time.perf_counter()
            if (self.bm25_accelerator is not None and req.mode != "lexical"
                    and req.scorer == "bm25" and not explicit):
                ranked = self.bm25_accelerator.topk(scoring_terms, self.config.candidate_k,
                                                    term_weights=term_weights)
            else:
                scores = scorers.SCORERS[req.scorer](self.reader, scoring_terms,
                                                 candidate_docs=candidate_docs,
                                                 term_weights=term_weights)
                if candidate_docs:
                    # A NOT-only query, or the negative branch of an OR, has no
                    # positive BM25 contribution. Keep those hard-filtered docs
                    # with deterministic zero scores instead of dropping them.
                    scores = {docid: scores.get(docid, 0.0) for docid in candidate_docs}
                limit = req.offset + req.k if req.mode == "lexical" else self.config.candidate_k
                ranked = nsmallest(limit, scores.items(), key=lambda row: (-row[1], row[0]))
            timings["lexical"] = (time.perf_counter() - start) * 1000
        if req.mode != "lexical":
            ranked = ranked[:self.config.candidate_k]

        rerank_features = None
        rerank_feature_ms = 0.0
        if (self.reranker is not None and self.reranker.records_features
                and req.mode == "hybrid" and not explicit):
            start = time.perf_counter()
            rerank_features = ce_gate_features(self.reader, terms, ranked)
            rerank_feature_ms = (time.perf_counter() - start) * 1000

        reason = f"forced_{req.mode}"
        use_vector = req.mode in ("dense", "hybrid")
        semantic_mode = "dense" if req.mode == "dense" else "hybrid"
        if req.mode == "adaptive":
            start = time.perf_counter()
            if explicit:
                # 显式约束由引擎直接短路，任何注入策略都不能把它送进向量召回。
                use_vector, reason = False, "explicit_constraints"
            else:
                decision = self._decide(self._features(req.query, terms, ranked))
                use_vector, reason = decision.use_vector, decision.reason
                semantic_mode = decision.semantic_mode
            timings["router"] = (time.perf_counter() - start) * 1000

        executed = "lexical"
        score_kind = req.scorer
        fallback = False
        vector_used = False
        if use_vector:
            try:
                dense_ranked = self._vector_search(req.query, timings)
                if candidate_docs is not None:
                    dense_ranked = [(docid, score) for docid, score in dense_ranked
                                    if docid in candidate_docs]
            except RetrievalUnavailable:
                if req.mode != "adaptive":
                    raise
                # 明确回显退化；固定 hybrid/dense 不能假装成功，以免污染评测。
                fallback, reason = True, "vector_unavailable"
            else:
                vector_used = True
                if semantic_mode == "dense":
                    ranked, executed, score_kind = dense_ranked, "dense", "cosine"
                else:
                    start = time.perf_counter()
                    ranked = rrf([[d for d, _ in ranked], [d for d, _ in dense_ranked]], self.config.rrf_c)
                    ranked = ranked[:self.config.candidate_k]
                    timings["fusion"] = (time.perf_counter() - start) * 1000
                    executed, score_kind = "hybrid", "rrf"

        results = []
        document_text = getattr(self.reader, "document_text", None)
        read_document = document_text if callable(document_text) else self.reader.doc
        rerank = None
        if self.reranker is not None and req.mode == "hybrid" and not explicit and ranked:
            start = time.perf_counter()
            ranked, rerank = self.reranker.rerank(req.query, ranked, read_document,
                                                features=rerank_features)
            timings["rerank"] = rerank_feature_ms + (time.perf_counter() - start) * 1000
        for rank, (docid, score) in enumerate(ranked[req.offset:req.offset + req.k], req.offset + 1):
            # Rendering re-analyzes the original text; it does not consume the
            # stored token-offset table required by the conformance doc() API.
            doc = read_document(docid)
            results.append(ResultItem(
                doc_id=doc.doc_id, rank=rank, score=score, title=doc.title,
                snippet=doc.text[:SNIPPET_LEN], highlights=build_highlights(doc, set(terms)),
            ))
        routing = RoutingInfo(req.mode, executed, reason, vector_used, use_vector, fallback,
                              score_kind, self.config.candidate_k,
                              self.policy["policy_version"])
        return SearchResponse(int((time.perf_counter() - t0) * 1000), parsed, expanded,
                              results, routing, {k: round(v, 3) for k, v in timings.items()},
                              rerank=rerank)
