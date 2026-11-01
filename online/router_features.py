"""Versioned BM25-first statistics; no embeddings, qrels or dense candidates."""

import math
import statistics
from bisect import bisect_left

FEATURE_VERSION = 'bm25-statistics-v1'
FEATURE_VERSION_V2 = 'bm25-statistics-v2'
FEATURE_NAMES = (
    'query_chars', 'n_terms', 'ascii_fraction', 'digit_fraction',
    'vocab_coverage', 'mean_idf', 'max_idf', 'std_idf', 'sum_df_fraction',
    'candidate_hits', 'top1', 'mean10', 'cv10', 'relative_gap', 'relative_gap10',
    'entropy10', 'top1_share10',
)
FEATURE_NAMES_V2 = FEATURE_NAMES + (
    'top1_term_coverage', 'top10_term_coverage', 'top10_all_terms',
    'top10_mean_term_coverage', 'top10_rare_term_coverage', 'candidate_log',
)

FAST_FEATURE_NAMES = ('query_chars','n_terms','ascii_fraction','digit_fraction',
    'candidate_hits','top1','mean10','cv10','relative_gap','relative_gap10',
    'entropy10','top1_share10','candidate_log')
BASE_FEATURE_NAMES = FEATURE_NAMES


def extract_statistics_fast(query, terms, ranked):
    """Cheap query statistics derived only from the already returned BM25 list."""
    import math
    scores=[float(score) for _,score in ranked[:10]]
    top=scores[0] if scores else 0.;total=sum(scores);mean=(sum(scores)/len(scores) if scores else 0.)
    probabilities=[s/total for s in scores if s>0] if total else []
    entropy=-sum(p*math.log(p) for p in probabilities)
    unique=sorted(set(terms));values=[len(query),len(unique),sum(ord(c)<128 for c in query)/max(len(query),1),
        sum(c.isdigit() for c in query)/max(len(query),1),len(ranked),top,mean,
        (sum((s-mean)**2 for s in scores)/len(scores))**.5/max(mean,1e-12) if scores else 0.,
        (top-scores[1])/max(top,1e-12) if len(scores)>1 else 0.,
        (top-scores[-1])/max(top,1e-12) if len(scores)>1 else 0.,
        entropy/math.log(len(scores)) if len(scores)>1 else 0.,top/total if total else 0.,math.log1p(len(ranked))]
    result=dict(zip(FAST_FEATURE_NAMES,values))
    if not all(math.isfinite(v) for v in result.values()):raise ValueError('nonfinite fast BM25 statistics')
    return result


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


def extract_statistics_v2(query, reader, terms, ranked):
    """Add BM25 top-document term coverage without reading dense/qrels."""
    result = extract_statistics(query, reader, terms, ranked)
    unique = sorted(set(terms))
    top_docs = [docid for docid, _ in ranked[:10]]
    top_set = set(top_docs)
    posting_docs = {}
    for term in unique:
        posting = reader.postings(term)
        matched = set()
        if posting:
            for docid in top_docs:
                i = bisect_left(posting, docid, key=lambda row: row[0])
                if i < len(posting) and posting[i][0] == docid:
                    matched.add(docid)
        posting_docs[term] = matched
    if unique:
        counts = [sum(docid in posting_docs[term] for term in unique) / len(unique)
                  for docid in top_docs]
        top1 = counts[0] if counts else 0.0
        top10 = sum(bool(posting_docs[term] & top_set) for term in unique) / len(unique)
        all_terms = float(all(count >= 1.0 for count in counts)) if counts else 0.0
        rare = [term for term in unique if reader.df(term) <= max(1, reader.n_docs // 100)]
        rare_coverage = (sum(bool(posting_docs[term] & top_set) for term in rare) / len(rare)
                         if rare else 0.0)
    else:
        counts = []
        top1 = top10 = all_terms = rare_coverage = 0.0
    result.update(top1_term_coverage=top1, top10_term_coverage=top10,
                  top10_all_terms=all_terms,
                  top10_mean_term_coverage=sum(counts) / len(counts) if counts else 0.0,
                  top10_rare_term_coverage=rare_coverage,
                  candidate_log=math.log1p(len(ranked)))
    if set(result) != set(FEATURE_NAMES_V2) or not all(math.isfinite(v) for v in result.values()):
        raise ValueError('invalid BM25 v2 statistics')
    return result
