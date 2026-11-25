"""Query expansion implementations.

Expansion only changes lexical ranking and never relaxes explicit AND/NOT or
phrase constraints. RM3 is deterministic pseudo-relevance feedback; SLM
remains an optional external-model integration and returns 501.
"""

from collections import Counter, defaultdict

from contracts import analyzer
from contracts.reader import IndexReader


class ExpanderNotReady(NotImplementedError):
    pass


def expand_none(reader: IndexReader, query: str, terms: list) -> list:
    """不扩展。"""
    return []


def expand_rm3(reader: IndexReader, query: str, terms: list) -> list:
    """Return deterministic RM3-style feedback terms from the first 8 hits."""
    if not terms:
        return []
    from .scorers import topk
    ranked = topk("bm25", reader, terms, 8)
    if not ranked:
        return []
    original = set(terms)
    estimates: defaultdict[str, float] = defaultdict(float)
    for rank, (docid, _) in enumerate(ranked, 1):
        tokens = analyzer.analyze(reader.doc(docid).text)
        counts = Counter(token.term for token in tokens)
        if not counts:
            continue
        rank_weight = 1.0 / rank
        denominator = float(len(tokens))
        for term, count in counts.items():
            if term not in original:
                estimates[term] += rank_weight * count / denominator
    if not estimates:
        return []
    top = sorted(estimates.items(), key=lambda row: (-row[1], row[0]))[:5]
    maximum = top[0][1]
    return [{"term": term, "w": round(min(0.75, value / maximum * 0.75), 6), "src": "rm3"}
            for term, value in top if value > 0]


def expand_slm(reader: IndexReader, query: str, terms: list) -> list:
    """SLM expansion requires a pinned model and an offline cache."""
    raise ExpanderNotReady("SLM 需要外部固定模型与离线缓存，当前未启用")


EXPANDERS = {"none": expand_none, "rm3": expand_rm3, "slm": expand_slm}
