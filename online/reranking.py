"""Optional cross-encoder control on a fixed prefix of hybrid candidates."""

import math

from contracts.ce import GATE_DEPTH, gate_binding


class RerankUnavailable(RuntimeError):
    """The selected cross-encoder did not produce a usable ranking."""


class HybridReranker:
    def __init__(self, scorer, depth: int, *, gate=None, capture_features=False):
        if type(depth) is not int or not 1 <= depth <= GATE_DEPTH:
            raise ValueError("hybrid rerank depth must be between 1 and 40")
        if gate is not None and depth != GATE_DEPTH:
            raise ValueError("the learned rerank gate requires depth 40")
        self.scorer = scorer
        self.depth = depth
        self.gate = gate
        self.records_features = capture_features or gate is not None

    @property
    def identity(self) -> dict:
        identity = {"depth": self.depth, "candidate_source": "hybrid_rrf",
                    "model": dict(self.scorer.identity)}
        if self.records_features:
            identity["capture_features"] = True
        if self.gate is not None:
            identity["gate_sha256"] = self.gate.sha256
        return identity

    def check_binding(self, retrieval):
        if self.gate is not None:
            binding = gate_binding(retrieval, self.scorer.identity)
            self.gate.check_binding(binding["retrieval"], binding["model"])

    def rerank(self, query, ranked, read_document, *, features=None):
        metadata = {"depth": self.depth, "scored": 0}
        if self.records_features:
            metadata["features"] = features
        if self.gate is not None:
            metadata["gate_sha256"] = self.gate.sha256
            if not self.gate.use_ce(features):
                return ranked, metadata
        prefix = ranked[:self.depth]
        documents = [read_document(docid) for docid, _ in prefix]
        texts = [doc.title + "\n" + doc.text for doc in documents]
        try:
            scores = self.scorer.score(query, texts)
            if (len(scores) != len(prefix)
                    or any(type(score) not in (int, float) or not math.isfinite(score)
                           for score in scores)):
                raise ValueError("invalid cross-encoder scores")
        except (OSError, TypeError, RuntimeError, ValueError, OverflowError) as error:
            raise RerankUnavailable("cross-encoder scoring failed") from error
        ordered = sorted(zip(scores, documents, prefix),
                         key=lambda row: (-row[0], row[1].doc_id))
        # Preserve first-stage RRF scores. CE scores rank this prefix only.
        metadata["scored"] = len(prefix)
        return [item for _, _, item in ordered] + ranked[len(prefix):], metadata
