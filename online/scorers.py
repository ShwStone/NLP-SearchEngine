"""打分器注册点（在线组）。

新增打分器 = 在这里多注册一个函数，不改其他代码。
骨架版实现了 BM25 与 TF-IDF（暴力可对照版），参数对齐
MTEB 官方基线常用值 k1=1.2, b=0.75；E2 复现官方实现时再调 k1/b。
"""

import math
from collections import defaultdict
from heapq import nsmallest

from contracts.index_format import BM25Spec
from contracts.reader import IndexReader


NUMPY_MIN_POSTINGS = 64
DENSE_MIN_POSTINGS = 2 * 1024 * 1024
DENSE_POSTINGS_PER_DOC = 4


def bm25(reader: IndexReader, terms: list, k1: float | None = None, b: float | None = None,
         match_all: bool = False, candidate_docs: set[int] | None = None,
         term_weights: dict[str, float] | None = None) -> dict:
    """返回 {docid: score}；自然语言查询默认词项并集，硬约束由过滤层处理。"""
    spec = BM25Spec(**reader.manifest().get("bm25", {}))
    k1 = spec.k1 if k1 is None else k1
    b = spec.b if b is None else b
    BM25Spec(k1=k1, b=b)  # 同时校验显式覆盖参数
    # In the default OR case every scoring posting belongs to the union;
    # avoid constructing that union (and reading each posting list twice).
    cand = (set(candidate_docs) if candidate_docs is not None else
            _candidate_docs(reader, terms, True) if match_all else None)
    scores = defaultdict(float)
    n = reader.n_docs
    term_weights = term_weights or {}
    for term in terms:
        df = reader.df(term)
        if df == 0:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for docid, tf in _term_frequencies(reader, term):
            if cand is not None and docid not in cand:
                continue
            dl = reader.doc_length(docid)
            scores[docid] += term_weights.get(term, 1.0) * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / reader.avgdl))
    return dict(scores)


def tfidf(reader: IndexReader, terms: list, match_all: bool = False,
          candidate_docs: set[int] | None = None,
          term_weights: dict[str, float] | None = None) -> dict:
    """兼容的 TF-IDF 基线：tf 取 1+log，idf 取 log(N/df)。"""
    cand = (set(candidate_docs) if candidate_docs is not None else
            _candidate_docs(reader, terms, True) if match_all else None)
    scores = defaultdict(float)
    n = reader.n_docs
    term_weights = term_weights or {}
    for term in terms:
        df = reader.df(term)
        if df == 0:
            continue
        idf = math.log(n / df)
        for docid, tf in _term_frequencies(reader, term):
            if cand is not None and docid not in cand:
                continue
            scores[docid] += term_weights.get(term, 1.0) * (1 + math.log(tf)) * idf
    return dict(scores)


def _term_frequencies(reader: IndexReader, term: str):
    """Use ordered (docid, tf) pairs without decoding optional positions."""
    frequencies = getattr(reader, "term_frequencies", None)
    if callable(frequencies):
        return frequencies(term)
    postings = reader.postings(term)
    if postings is None:
        return None
    return ((docid, tf) for docid, tf, _ in postings)


def _bm25_weights(lengths, frequencies, term_weight, idf, k1, b, avgdl, np):
    """Elementwise operations in the scalar implementation's original order."""
    tf = frequencies.astype(np.float64)
    numerator = (term_weight * idf) * tf
    numerator = numerator * (k1 + 1)
    denominator = (b * lengths.astype(np.float64)) / avgdl
    denominator = (1 - b) + denominator
    denominator = k1 * denominator
    return numerator / (tf + denominator)


def _rank_docids(docids, scores, limit, np):
    """Exact (-score, docid) top-k without sorting the whole candidate union."""
    if limit >= len(scores):
        order = np.lexsort((docids, -scores))
    else:
        boundary = -np.partition(-scores, limit - 1)[limit - 1]
        above = np.flatnonzero(scores > boundary)
        ties = np.flatnonzero(scores == boundary)
        # Both single blocks and np.unique output are docid-sorted, so the
        # first required ties are exactly the smallest external-ranking IDs.
        chosen = np.concatenate((above, ties[:limit - len(above)]))
        order = chosen[np.lexsort((docids[chosen], -scores[chosen]))]
    return list(zip(docids[order].tolist(), scores[order].tolist()))


def _numpy_bm25_topk(reader, terms: list, limit: int,
                     term_weights: dict[str, float] | None = None):
    """Exact sparse-union BM25, or None for the dependency-free fallback.

    Fixed numpy overhead loses on tiny posting lists, so the scalar path keeps
    queries below 64 postings per scored term. Memory depends on matched
    postings rather than the total document count.
    """
    frequency_arrays = getattr(reader, "frequency_arrays", None)
    document_lengths = getattr(reader, "document_lengths", None)
    if not callable(frequency_arrays) or not callable(document_lengths) or limit < 1:
        return None
    if reader.avgdl == 0:
        return None
    weights = term_weights or {}
    blocks = []
    total_postings = 0
    for term in terms:
        weight = float(weights.get(term, 1.0))
        if not math.isfinite(weight):
            return None
        df = reader.df(term)
        if df == 0:
            continue
        arrays = frequency_arrays(term)
        if arrays is None:
            return None
        blocks.append((term, df, arrays, weight))
        total_postings += df
    if not blocks or total_postings < NUMPY_MIN_POSTINGS * len(blocks):
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    spec = BM25Spec(**reader.manifest().get("bm25", {}))
    k1, b = spec.k1, spec.b
    lengths = np.frombuffer(document_lengths(), dtype=np.uint32)
    n = reader.n_docs
    docid_parts, score_parts = [], []
    for term, df, arrays, weight in blocks:
        docids = np.frombuffer(arrays[0], dtype=np.uint32)
        frequencies = np.frombuffer(arrays[1], dtype=np.uint32)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        docid_parts.append(docids)
        score_parts.append(_bm25_weights(lengths[docids], frequencies, weight, idf,
                                         k1, b, reader.avgdl, np))
    if len(docid_parts) == 1:
        docids, scores = docid_parts[0], score_parts[0]
    else:
        concatenated = np.concatenate(docid_parts)
        values = np.concatenate(score_parts)
        if total_postings >= max(DENSE_MIN_POSTINGS, n // DENSE_POSTINGS_PER_DOC):
            # Large unions make np.unique's sort dominate. One dense bincount
            # preserves term-major accumulation order without per-term arrays.
            dense_scores = np.bincount(concatenated, weights=values, minlength=n)
            matched = np.zeros(n, dtype=np.bool_)
            matched[concatenated] = True
            docids = np.flatnonzero(matched)
            scores = dense_scores[docids]
        else:
            docids, inverse = np.unique(concatenated, return_inverse=True)
            scores = np.bincount(inverse, weights=values, minlength=len(docids))
    return _rank_docids(docids, scores, limit, np)


def topk(scorer: str, reader: IndexReader, terms: list, limit: int, *,
         match_all: bool = False, candidate_docs: set[int] | None = None,
         term_weights: dict[str, float] | None = None, selector=nsmallest):
    """Return bounded deterministic ranking, using an optional exact fast path."""
    if scorer == "bm25" and not match_all and candidate_docs is None:
        ranked = _numpy_bm25_topk(reader, terms, limit, term_weights)
        if ranked is not None:
            return ranked
    scores = SCORERS[scorer](reader, terms, match_all=match_all,
                             candidate_docs=candidate_docs, term_weights=term_weights)
    return selector(limit, scores.items(), key=lambda row: (-row[1], row[0]))


def _candidate_docs(reader: IndexReader, terms: list, match_all: bool = False) -> set:
    """普通查询取词项并集；仅显式要求 match_all 时取交集。"""
    sets = []
    for t in terms:
        pl = _term_frequencies(reader, t)
        if pl is None:
            if match_all:
                return set()
            continue
        sets.append({docid for docid, _ in pl})
    if not sets:
        return set()
    return set.intersection(*sets) if match_all else set.union(*sets)


def phrase_docs(reader: IndexReader, terms: list[str]) -> set[int]:
    """Return documents containing ``terms`` at consecutive token positions."""
    if not terms:
        return set()
    postings = [reader.postings(term) for term in terms]
    if any(not posting for posting in postings):
        return set()
    positions = [{docid: set(pos) for docid, _, pos in posting} for posting in postings]
    common = set(positions[0])
    for mapping in positions[1:]:
        common.intersection_update(mapping)
    matched = set()
    for docid in common:
        starts = positions[0][docid]
        if any(all(start + offset in positions[offset][docid]
                   for offset in range(1, len(positions))) for start in starts):
            matched.add(docid)
    return matched


def boolean_docs(reader: IndexReader, node: dict, analyze) -> set[int]:
    """Evaluate the parser AST as a hard lexical filter."""
    op = node["op"]
    if op == "term":
        terms = [token.term for token in analyze(node["value"])]
        return _candidate_docs(reader, terms, match_all=False)
    if op == "phrase":
        terms = [token.term for token in analyze(node["value"])]
        return phrase_docs(reader, terms)
    if op == "not":
        return set(range(reader.n_docs)) - boolean_docs(reader, node["child"], analyze)
    children = [boolean_docs(reader, child, analyze) for child in node["children"]]
    if not children:
        return set()
    if op == "and":
        result = children[0]
        for child in children[1:]:
            result = result & child
        return result
    result = set()
    for child in children:
        result |= child
    return result


SCORERS = {"bm25": bm25, "tfidf": tfidf}
