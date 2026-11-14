"""Exact BM25 shard scoring using corpus-wide statistics.

Local Top-k is safe to merge only after replacing N, df and avgdl with global
statistics. Local document IDs are shifted by cumulative shard document counts.
No corpus-wide reader is materialized in memory.
"""
from heapq import nsmallest
from online.scorers import bm25


class GlobalStatisticsView:
    def __init__(self, local, n_docs, avgdl, dfs):
        self.local = local
        self.n_docs = n_docs
        self.avgdl = avgdl
        self.dfs = dfs

    def df(self, term):
        # Missing statistics must fail rather than silently use local df.
        return self.dfs[term]

    def postings(self, term):
        return self.local.postings(term) or []

    def doc_length(self, docid):
        return self.local.doc_length(docid)

    def manifest(self):
        return self.local.manifest()


def shard_topk(reader, terms, *, n_docs, avgdl, dfs, offset, k, candidate_docs=None):
    if k < 1 or offset < 0:
        raise ValueError('invalid k or document offset')
    view = GlobalStatisticsView(reader, n_docs, avgdl, dfs)
    scores = bm25(view, terms, candidate_docs=candidate_docs)
    if candidate_docs is not None:
        scores = {docid: scores.get(docid, 0.0) for docid in candidate_docs}
    return nsmallest(k, ((docid + offset, score) for docid, score in scores.items()),
                     key=lambda hit: (-hit[1], hit[0]))


def merge_topk(shard_results, k):
    if k < 1:
        raise ValueError('k must be positive')
    return nsmallest(k, (hit for rows in shard_results for hit in rows),
                     key=lambda hit: (-hit[1], hit[0]))
