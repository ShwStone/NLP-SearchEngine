"""在线混合检索骨架：parse -> BM25 -> router -> 可选向量召回 -> RRF。

真实模型/数据库通过 contracts.retrieval 注入；词法模式不需要向量依赖。
"""

import math
import time
from dataclasses import asdict

from contracts import analyzer
from contracts.api import ApiError, ResultItem, RoutingInfo, SearchRequest, SearchResponse
from contracts.reader import IndexReader
from contracts.retrieval import (
    RetrievalConfig, RetrievalUnavailable, TextEncoder, VectorReader,
    check_alignment, validate_vector,
)

from . import parser, scorers
from .expanders import EXPANDERS
from .fusion import rrf
from .highlight import build_highlights
from .router import QueryFeatures, RuleRouter

SNIPPET_LEN = 80


class SearchEngine:
    def __init__(self, reader: IndexReader, encoder: TextEncoder | None = None,
                 vectors: VectorReader | None = None, config: RetrievalConfig | None = None):
        self.reader = reader
        self.encoder = encoder
        self.vectors = vectors
        self.config = config or RetrievalConfig()
        self.router = RuleRouter(self.config)
        if (encoder is None) != (vectors is None):
            raise ValueError("encoder 与 vectors 必须同时提供或同时省略")
        if vectors is not None:
            check_alignment(reader, vectors, encoder)

    def info(self) -> dict:
        """能力与配置回显，不把尚未实现的后端宣称为可用。"""
        return {
            "vector_ready": self.vectors is not None,
            "config": asdict(self.config),
            "vector_manifest": self.vectors.manifest().to_dict() if self.vectors else None,
        }

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
            for hit in hits:
                if type(hit.docid) is not int or not 0 <= hit.docid < self.reader.n_docs:
                    raise ValueError("向量结果包含未知 docid")
                if self.reader.doc(hit.docid).doc_id != hit.doc_id:
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
            scores = scorers.SCORERS[req.scorer](self.reader, scoring_terms,
                                                 candidate_docs=candidate_docs,
                                                 term_weights=term_weights)
            if candidate_docs:
                # A NOT-only query, or the negative branch of an OR, has no
                # positive BM25 contribution. Keep those hard-filtered docs
                # with deterministic zero scores instead of dropping them.
                scores = {docid: scores.get(docid, 0.0) for docid in candidate_docs}
            ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
            timings["lexical"] = (time.perf_counter() - start) * 1000
        if req.mode != "lexical":
            ranked = ranked[:self.config.candidate_k]

        reason = f"forced_{req.mode}"
        use_vector = req.mode in ("dense", "hybrid")
        if req.mode == "adaptive":
            start = time.perf_counter()
            unique_terms = set(terms)
            coverage = sum(self.reader.df(t) > 0 for t in unique_terms) / len(unique_terms) if unique_terms else 0.0
            gap = ((ranked[0][1] - ranked[1][1]) / max(abs(ranked[0][1]), 1e-12)
                   if len(ranked) >= 2 else None)
            decision = self.router.decide(QueryFeatures(explicit, len(unique_terms), coverage, len(ranked), gap))
            use_vector, reason = decision.use_vector, decision.reason
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
                if req.mode == "dense":
                    ranked, executed, score_kind = dense_ranked, "dense", "cosine"
                else:
                    start = time.perf_counter()
                    ranked = rrf([[d for d, _ in ranked], [d for d, _ in dense_ranked]], self.config.rrf_c)
                    ranked = ranked[:self.config.candidate_k]
                    timings["fusion"] = (time.perf_counter() - start) * 1000
                    executed, score_kind = "hybrid", "rrf"

        results = []
        for rank, (docid, score) in enumerate(ranked[req.offset:req.offset + req.k], req.offset + 1):
            doc = self.reader.doc(docid)
            results.append(ResultItem(
                doc_id=doc.doc_id, rank=rank, score=score, title=doc.title,
                snippet=doc.text[:SNIPPET_LEN], highlights=build_highlights(doc, set(terms)),
            ))
        routing = RoutingInfo(req.mode, executed, reason, vector_used, use_vector, fallback,
                              score_kind, self.config.candidate_k,
                              self.config.policy_version if self.config.adaptive_policy == "rules-v1"
                              else self.config.adaptive_policy)
        return SearchResponse(int((time.perf_counter() - t0) * 1000), parsed, expanded,
                              results, routing, {k: round(v, 3) for k, v in timings.items()})
