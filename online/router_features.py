"""Versioned BM25-first statistics; no embeddings, qrels or dense candidates."""

import math
import statistics

FEATURE_VERSION = 'bm25-statistics-v1'
FEATURE_NAMES = (
    'query_chars', 'n_terms', 'ascii_fraction', 'digit_fraction',
    'vocab_coverage', 'mean_idf', 'max_idf', 'std_idf', 'sum_df_fraction',
    'candidate_hits', 'top1', 'mean10', 'cv10', 'relative_gap', 'relative_gap10',
    'entropy10', 'top1_share10',
)


def extract_statistics(query, reader, terms, ranked):
    unique = sorted(set(terms))
    dfs = [reader.df(term) for term in unique]
    idfs = [math.log1p((reader.n_docs-df+.5)/(df+.5)) for df in dfs if df]
    scores = [score for _, score in ranked[:10]]
    if any(not math.isfinite(s) or s < 0 for s in scores):
        raise ValueError('BM25 statistics require finite nonnegative scores')
    top = scores[0] if scores else 0.
    total = sum(scores)
    mean = statistics.mean(scores) if scores else 0.
    probabilities = [s/total for s in scores if s > 0] if total else []
    entropy = -sum(p*math.log(p) for p in probabilities)
    values = [len(query), len(unique), sum(ord(c)<128 for c in query)/max(len(query), 1),
              sum(c.isdigit() for c in query)/max(len(query), 1),
              sum(df>0 for df in dfs)/max(len(dfs), 1),
              statistics.mean(idfs) if idfs else 0., max(idfs, default=0.),
              statistics.pstdev(idfs) if idfs else 0., sum(dfs)/reader.n_docs,
              len(ranked), top, mean, statistics.pstdev(scores)/max(mean, 1e-12) if scores else 0.,
              (top-scores[1])/max(top, 1e-12) if len(scores)>1 else 0.,
              (top-scores[-1])/max(top, 1e-12) if len(scores)>1 else 0.,
              entropy/math.log(len(scores)) if len(scores)>1 else 0., top/total if total else 0.]
    result = dict(zip(FEATURE_NAMES, values))
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('nonfinite BM25 statistics')
    return result
